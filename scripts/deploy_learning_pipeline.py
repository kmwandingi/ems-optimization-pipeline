#!/usr/bin/env python
"""
Azure ML Deployment Script for Complete EMS Optimization Pipeline
Deploys FULL optimization pipeline: building + prices → optimized schedules + learning.
Uses agent optimizers with strict compliance.
"""

import os
import sys
import json
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import pickle
import tempfile

# Add notebooks directory to path for agent imports
sys.path.append(str(Path.cwd() / "notebooks"))
sys.path.append(str(Path.cwd() / "notebooks" / "utils"))
sys.path.append(str(Path.cwd() / "utils"))

# Import agent classes
try:
    from agents.ProbabilityModelAgent import ProbabilityModelAgent
    from agents.BatteryAgent import BatteryAgent
    from agents.EVAgent import EVAgent
    from agents.PVAgent import PVAgent
    from agents.GridAgent import GridAgent
    from agents.FlexibleDeviceAgent import FlexibleDevice
    from agents.GlobalOptimizer import GlobalOptimizer
    from agents.GlobalConnectionLayer import GlobalConnectionLayer
    print("✓ Successfully imported agent classes")
except ImportError as e:
    print(f"CRITICAL ERROR: Failed to import agent classes: {e}")
    sys.exit(1)

from device_specs import device_specs
from mlflow_tracker import EMS_OptimizationTracker

# Import centralized configuration
try:
    from config_loader import get_config
    config = get_config()
    CONFIG_AVAILABLE = True
    print("✓ Configuration system loaded")
except ImportError as e:
    print(f"⚠ Configuration loader not available: {e}")
    CONFIG_AVAILABLE = False

# Azure ML imports
try:
    import azureml.core
    from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig
    from azureml.core.compute import ComputeTarget
    from azureml.core.runconfig import DockerConfiguration
    AZUREML_AVAILABLE = True
    print("✓ Azure ML SDK available")
except ImportError:
    print("Installing Azure ML SDK...")
    os.system("pip install azureml-sdk[notebooks] azureml-mlflow")
    try:
        import azureml.core
        from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig
        from azureml.core.compute import ComputeTarget
        from azureml.core.runconfig import DockerConfiguration
        AZUREML_AVAILABLE = True
        print("✓ Azure ML SDK installed and available")
    except ImportError as e:
        print(f"CRITICAL ERROR: Cannot install Azure ML SDK: {e}")
        sys.exit(1)

class EMSOptimizationModel(mlflow.pyfunc.PythonModel):
    """
    Complete EMS Optimization Model that:
    1. Takes building_id + devices + prices → Returns optimized schedules
    2. Learns from actual user behavior → Updates PMFs
    """
    
    def load_context(self, context):
        """Initialize the optimization system with all agents."""
        print("🔧 Loading EMS optimization system...")
        
        # Initialize DuckDB connection
        self.db_path = self._find_database()
        print(f"✓ Found database: {self.db_path}")
        
        # Load device specifications
        self.device_specs = device_specs
        print(f"✓ Loaded device specs for {len(self.device_specs)} device types")
        
        # Initialize probability model agent
        self.prob_agent = ProbabilityModelAgent()
        print("✓ Initialized ProbabilityModelAgent")
        
        # Load existing probability distributions if available
        self._load_existing_probabilities()
        
        print("✅ EMS optimization system ready!")

    def _find_database(self):
        """Find the EMS database file."""
        possible_paths = [
            Path.cwd() / "ems_data.duckdb",
            Path.cwd() / "notebooks" / "ems_data.duckdb",
            Path(__file__).parent.parent / "ems_data.duckdb",
            Path(__file__).parent.parent / "notebooks" / "ems_data.duckdb"
        ]
        
        for path in possible_paths:
            if path.exists():
                return str(path)
        
        raise FileNotFoundError("EMS database not found")

    def _load_existing_probabilities(self):
        """Load existing probability distributions from DuckDB."""
        try:
            import duckdb
            con = duckdb.connect(self.db_path, read_only=True)
            probs_df = con.execute("SELECT * FROM device_hourly_probabilities").df()
            con.close()
            
            # Convert to agent format
            self.prob_agent = ProbabilityModelAgent(prob_dist_df=probs_df)
            print(f"✓ Loaded existing probabilities for {len(probs_df)} device-hour combinations")
        except Exception as e:
            print(f"⚠ No existing probabilities found: {e}")
            self.prob_agent = ProbabilityModelAgent()

    def predict(self, context, model_input):
        """
        Main prediction method that handles two modes:
        1. OPTIMIZE: Get optimized schedules for a building
        2. LEARN: Update probabilities based on actual user behavior
        """
        
        if isinstance(model_input, pd.DataFrame):
            model_input = model_input.to_dict('records')[0]
        
        mode = model_input.get('mode', 'optimize')
        
        if mode == 'optimize':
            return self._optimize_schedules(model_input)
        elif mode == 'learn':
            return self._learn_from_behavior(model_input)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'optimize' or 'learn'")

    def _optimize_schedules(self, input_data):
        """
        Optimize device schedules for a building.
        
        Input format:
        {
            'mode': 'optimize',
            'building_id': 'DE_KN_residential1',
            'target_date': '2015-05-23',
            'price_profile': [0.25, 0.23, 0.21, ...],  # 24 hourly prices
            'battery_enabled': True,
            'ev_enabled': False,
            'grid_params': {'import_price': 0.25, 'export_price': 0.05}
        }
        
        Returns optimized schedules for all devices.
        """
        import duckdb
        print(f"🎯 Starting optimization for {input_data.get('building_id')}")
        
        building_id = input_data['building_id']
        target_date = input_data['target_date']
        price_profile = input_data.get('price_profile', [0.25] * 24)
        
        # Get building data from DuckDB
        con = duckdb.connect(self.db_path, read_only=True)
        view_name = f"{building_id}_processed_data"
        
        # Get data for target date
        target_data = con.execute(f"""
            SELECT * FROM {view_name}
            WHERE DATE(utc_timestamp) = '{target_date}'
            ORDER BY utc_timestamp
        """).df()
        
        if len(target_data) == 0:
            con.close()
            raise ValueError(f"No data found for {building_id} on {target_date}")
        
        print(f"✓ Retrieved {len(target_data)} hours of data for {target_date}")
        
        # Initialize all agents with current probabilities and configuration
        agents = self._initialize_optimization_agents(building_id, input_data)
        
        # Run real optimization using GlobalOptimizer
        optimization_results = self._run_optimization(
            building_id=building_id,
            target_date=target_date,
            weather_data=target_data,
            price_profile=price_profile,
            agents=agents
        )
        
        con.close()
        
        print("✅ Optimization completed successfully")
        
        return {
            'status': 'success',
            'building_id': building_id,
            'target_date': target_date,
            'optimized_schedules': optimization_results['schedules'],
            'total_cost': optimization_results['total_cost'],
            'savings_vs_baseline': optimization_results.get('savings', 0),
            'timestamp': datetime.now().isoformat()
        }

    def _learn_from_behavior(self, input_data):
        """
        Learn from actual user behavior to update probability distributions.
        
        Input format:
        {
            'mode': 'learn',
            'building_id': 'DE_KN_residential1',
            'actual_usage': {
                'DE_KN_residential1_heat_pump': [0, 0, 1, 1, 0, ...],  # 24 hourly values
                'DE_KN_residential1_dishwasher': [0, 0, 0, 1, 0, ...]
            },
            'date': '2015-05-23'
        }
        
        Updates PMFs based on actual device usage patterns.
        """
        print(f"🧠 Learning from user behavior for {input_data.get('building_id')}")
        
        building_id = input_data['building_id']
        actual_usage = input_data['actual_usage']
        date = input_data['date']
        
        # Update probability distributions using agent method
        updated_devices = []
        
        for device_id, hourly_usage in actual_usage.items():
            # Convert hourly usage to learning format
            learning_data = []
            for hour, usage_value in enumerate(hourly_usage):
                if usage_value > 0:  # Device was active
                    learning_data.append({
                        'hour': hour,
                        'device_id': device_id,
                        'usage': float(usage_value),
                        'date': date
                    })
            
            if learning_data:
                # Update probabilities for this device using the correct method
                # For simplicity, just track that we got observations for this device
                # In production, this would use the full update_user_probability_model workflow
                for obs in learning_data:
                    # Simplified learning update - in production would use full method
                    if not hasattr(self.prob_agent, 'latest_distributions'):
                        self.prob_agent.latest_distributions = {}
                    
                    if device_id not in self.prob_agent.latest_distributions:
                        self.prob_agent.latest_distributions[device_id] = {h: 1.0/24 for h in range(24)}
                    
                    # Increase probability for observed hour (simplified Bayesian update)
                    observed_hour = obs['hour']
                    current_prob = self.prob_agent.latest_distributions[device_id].get(observed_hour, 1.0/24)
                    self.prob_agent.latest_distributions[device_id][observed_hour] = min(current_prob * 1.1, 0.5)
                    
                    # Normalize probabilities
                    total_prob = sum(self.prob_agent.latest_distributions[device_id].values())
                    for h in range(24):
                        self.prob_agent.latest_distributions[device_id][h] /= total_prob
                
                updated_devices.append(device_id)
        
        # Save updated probabilities back to DuckDB
        self._save_updated_probabilities(building_id)
        
        print(f"✅ Updated probabilities for {len(updated_devices)} devices")
        
        # Get the updated PMFs to return
        updated_pmfs = {}
        for device_id in updated_devices:
            if hasattr(self.prob_agent, 'latest_distributions') and device_id in self.prob_agent.latest_distributions:
                updated_pmfs[device_id] = self.prob_agent.latest_distributions[device_id]
        
        return {
            'status': 'success',
            'building_id': building_id,
            'updated_devices': updated_devices,
            'updated_pmfs': updated_pmfs,
            'learning_date': date,
            'timestamp': datetime.now().isoformat()
        }

    def _initialize_optimization_agents(self, building_id, input_data):
        """Initialize all optimization agents with current state."""
        agents = {}
        
        # Battery agent with configuration
        if input_data.get('battery_enabled', True):
            if 'battery_params' in input_data:
                battery_params = input_data['battery_params']
            elif CONFIG_AVAILABLE:
                battery_params = config.get_battery_config('large')
            else:
                # Fallback hardcoded parameters
                battery_params = {
                    "max_charge_rate": 5.0,
                    "max_discharge_rate": 5.0,
                    "initial_soc": 8.0,
                    "soc_min": 2.0,
                    "soc_max": 15.0,
                    "capacity": 15.0,
                    "degradation_rate": 0.001,
                    "efficiency_charge": 0.95,
                    "efficiency_discharge": 0.95
                }
            agents['battery'] = BatteryAgent(**battery_params)
        
        # EV agent with configuration
        if input_data.get('ev_enabled', False):
            if 'ev_params' in input_data:
                ev_params = input_data['ev_params']
            elif CONFIG_AVAILABLE:
                ev_params = config.get_ev_config('default')
            else:
                # Fallback hardcoded parameters
                ev_params = {
                    "capacity": 50.0,
                    "max_charge_rate": 11.0,
                    "max_discharge_rate": 11.0,
                    "initial_soc": 40.0,
                    "soc_min": 10.0,
                    "soc_max": 50.0,
                    "efficiency_charge": 0.9,
                    "efficiency_discharge": 0.9,
                    "must_be_full_by_hour": 7
                }
            agents['ev'] = EVAgent(**ev_params)
        
        # Grid agent with configuration
        if 'grid_params' in input_data:
            grid_params = input_data['grid_params']
        elif CONFIG_AVAILABLE:
            grid_params = config.get_grid_config('default')
        else:
            # Fallback hardcoded parameters
            grid_params = {
                "import_price": 0.25,
                "export_price": 0.05,
                "max_import": 15.0,
                "max_export": 15.0
            }
        agents['grid'] = GridAgent(**grid_params)
        
        # PV agent (if building has solar)
        agents['pv'] = PVAgent()
        
        # Device agents will be created in the real optimization method
        # when we have the full weather data context required by FlexibleDevice
        agents['device_specs'] = self.device_specs
        
        return agents

    def _get_device_probabilities(self, device_id):
        """Get current probability profile for a device."""
        if hasattr(self.prob_agent, 'latest_distributions') and device_id in self.prob_agent.latest_distributions:
            return self.prob_agent.latest_distributions[device_id]
        else:
            # Return uniform distribution as fallback
            return {hour: 1.0/24 for hour in range(24)}

    def _run_optimization(self, building_id, target_date, weather_data, price_profile, agents):
        """
        Run optimization using GlobalOptimizer.optimize_phases_centralized().
        Uses agent optimizers with strict compliance.
        """
        from agents.GlobalConnectionLayer import GlobalConnectionLayer
        
        # Initialize devices properly for GlobalOptimizer
        devices = []
        
        # Create FlexibleDeviceAgent instances for known flexible devices only
        flexible_devices = ['dishwasher', 'washing_machine', 'heat_pump', 'freezer']
        
        for device_type in flexible_devices:
            if device_type in self.device_specs:
                specs = self.device_specs[device_type]
                device_id = f"{building_id}_{device_type}"
                
                # Get device probabilities
                device_probs = self._get_device_probabilities(device_id)
                
                # Prepare device-specific data
                device_data = weather_data.copy()
                device_column = f"{building_id}_{device_type}"
                
                if device_column in device_data.columns:
                    # Use ONLY real device data - NO SYNTHETIC DATA ALLOWED
                    device_data[device_type] = device_data[device_column]
                else:
                    # STRICT COMPLIANCE: Skip devices not in real data - no fallbacks allowed
                    print(f"⚠ Device {device_id} not found in building data - skipping (no synthetic data)")
                    continue
                
                # Create proper FlexibleDeviceAgent
                device_agent = FlexibleDevice(
                    data=device_data,
                    device_name=device_type,
                    category=specs.get('category', 'Partially Flexible'),
                    power_rating=specs.get('power_rating', 1.0),
                    global_layer=None,  # Will be set by GlobalOptimizer
                    max_shift_hours=specs.get('max_shift_hours', 6),
                    is_flexible=specs.get('is_flexible', True),
                    battery_agent=agents.get('battery'),
                    pv_agent=agents.get('pv'),
                    spec=specs
                )
                
                # Set learned probabilities
                device_agent.hour_probability = device_probs
                devices.append(device_agent)
        
        # Create GlobalConnectionLayer with configuration
        if CONFIG_AVAILABLE:
            building_config = config.get_building_config('residential')
            base_load = building_config.get('max_building_load', 50.0)
            load_buffer = building_config.get('load_buffer', 1.2)
            max_building_load = base_load * load_buffer if agents.get('ev') else base_load
        else:
            max_building_load = 65.0 if agents.get('ev') else 50.0
        global_layer = GlobalConnectionLayer(max_building_load, 24)
        
        # Update device references to global layer
        for device in devices:
            device.global_layer = global_layer
        
        # Create WeatherAgent EXACTLY like direct pipeline
        weather_agent = None
        try:
            from agents.WeatherAgent import WeatherAgent
            weather_agent = WeatherAgent(weather_data)
        except Exception as e:
            print(f"⚠ WeatherAgent initialization failed: {e}")
        
        # Create GlobalOptimizer instance with configuration
        if CONFIG_AVAILABLE:
            optimization_config = config.get_optimization_config()
            max_iterations = optimization_config.get('global_optimizer', {}).get('max_iterations', 1)
            online_iterations = optimization_config.get('global_optimizer', {}).get('online_iterations', 1)
        else:
            max_iterations = 1
            online_iterations = 1
            
        optimizer = GlobalOptimizer(
            devices=devices,
            global_layer=global_layer,
            pv_agent=agents.get('pv'),
            weather_agent=weather_agent,
            battery_agent=agents.get('battery'),
            ev_agent=agents.get('ev'),
            grid_agent=agents['grid'],
            max_iterations=max_iterations,
            online_iterations=online_iterations
        )
        
        # MANDATORY: Use GlobalOptimizer.optimize_phases_centralized method
        success = optimizer.optimize_phases_centralized(
            devices=devices,
            global_layer=global_layer,
            pv_agent=agents.get('pv'),
            battery_agent=agents.get('battery'),
            ev_agent=agents.get('ev'),
            grid_agent=agents['grid'],
            weather_agent=weather_agent
        )
        
        if not success:
            # ERROR: Agent method failed - no fallbacks allowed
            raise RuntimeError("CRITICAL: GlobalOptimizer.optimize_phases_centralized() returned False - optimization failed")
        
        # Extract optimized schedules and calculate total cost from REAL optimization results
        total_cost = 0.0
        optimized_schedules = {}
        
        # Get battery schedule if available - EXACTLY like direct pipeline
        if agents.get('battery') and hasattr(agents['battery'], 'hourly_charge') and hasattr(agents['battery'], 'hourly_discharge'):
            battery_cost = sum(agents['battery'].hourly_charge[h] * price_profile[h] for h in range(24))
            battery_savings = sum(agents['battery'].hourly_discharge[h] * price_profile[h] for h in range(24))
            total_cost += battery_cost - battery_savings
            optimized_schedules['battery_charge'] = agents['battery'].hourly_charge[:24]
            optimized_schedules['battery_discharge'] = agents['battery'].hourly_discharge[:24]
        
        # Get device schedules from FlexibleDeviceAgent optimization results
        for device in devices:
            if hasattr(device, 'phases_optimized_schedule'):
                schedule = device.phases_optimized_schedule[:24]
                device.optimized_schedule = schedule
                optimized_schedules[f"{building_id}_{device.device_name}"] = schedule
                
                # Calculate cost for this device
                for hour, power in enumerate(schedule):
                    total_cost += power * price_profile[hour]
            elif hasattr(device, 'optimized_schedule'):
                schedule = device.optimized_schedule[:24]
                optimized_schedules[f"{building_id}_{device.device_name}"] = schedule
                
                # Calculate cost for this device
                for hour, power in enumerate(schedule):
                    total_cost += power * price_profile[hour]
        
        # Calculate baseline cost EXACTLY like direct pipeline
        import numpy as np
        total_energy = sum(sum(schedule) for schedule in optimized_schedules.values() 
                          if isinstance(schedule, list))
        if total_energy > 0:
            expensive_hours_avg = np.mean(np.sort(price_profile)[-6:])  # Top 6 expensive hours
            baseline_cost = total_energy * expensive_hours_avg
            savings = max(0, baseline_cost - total_cost)
        else:
            savings = 0
        
        return {
            'schedules': optimized_schedules,
            'total_cost': total_cost,
            'savings': savings
        }

    def _save_updated_probabilities(self, building_id):
        """Save updated probability distributions back to DuckDB."""
        try:
            import duckdb
            con = duckdb.connect(self.db_path)
            
            # Convert current probabilities to DataFrame format
            prob_records = []
            for device_id, hour_probs in self.prob_agent.latest_distributions.items():
                for hour, prob in hour_probs.items():
                    prob_records.append({
                        'device_id': device_id,
                        'hour': int(hour),
                        'probability': float(prob),
                        'updated_at': datetime.now().isoformat()
                    })
            
            if prob_records:
                prob_df = pd.DataFrame(prob_records)
                
                # Update the probabilities table
                con.execute("DROP TABLE IF EXISTS device_hourly_probabilities")
                con.execute("""
                    CREATE TABLE device_hourly_probabilities AS 
                    SELECT * FROM prob_df
                """)
                con.close()
                print(f"✓ Saved {len(prob_records)} probability updates to DuckDB")
        except Exception as e:
            print(f"⚠ Failed to save probabilities: {e}")
        

def setup_azure_workspace():
    """Initialize Azure ML workspace from config.json"""
    try:
        # Load workspace configuration
        with open('config.json', 'r') as f:
            config = json.load(f)
        
        print(f"Connecting to Azure ML workspace: {config['workspace_name']}")
        
        # Connect to workspace
        ws = Workspace(
            subscription_id=config['subscription_id'],
            resource_group=config['resource_group'],
            workspace_name=config['workspace_name']
        )
        
        print(f"✓ Connected to workspace: {ws.name}")
        return ws
        
    except Exception as e:
        print(f"CRITICAL ERROR: Cannot connect to Azure ML workspace: {e}")
        print("Ensure you're authenticated with Azure CLI: az login")
        sys.exit(1)

def setup_mlflow_tracking(ws):
    """Configure MLflow to track to Azure ML workspace"""
    # Set MLflow tracking URI to Azure ML workspace
    mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
    print(f"✓ MLflow tracking configured for Azure ML")
    
    # Set experiment
    experiment_name = "EMS_Learning_Pipeline_Production"
    mlflow.set_experiment(experiment_name)
    print(f"✓ Using experiment: {experiment_name}")
    
    return experiment_name

def create_environment(ws):
    """Create Azure ML environment with all dependencies"""
    env_name = "ems-learning-env"
    
    try:
        # Try to get existing environment
        env = Environment.get(workspace=ws, name=env_name)
        print(f"✓ Using existing environment: {env_name}")
    except:
        # Create new environment
        print(f"Creating new environment: {env_name}")
        env = Environment(name=env_name)
        
        # Define dependencies
        env.python.conda_dependencies.add_pip_package("mlflow>=2.0")
        env.python.conda_dependencies.add_pip_package("azureml-mlflow")
        env.python.conda_dependencies.add_pip_package("pandas>=1.5.0")
        env.python.conda_dependencies.add_pip_package("numpy>=1.20.0") 
        env.python.conda_dependencies.add_pip_package("matplotlib>=3.5.0")
        env.python.conda_dependencies.add_pip_package("seaborn>=0.11.0")
        env.python.conda_dependencies.add_pip_package("duckdb>=0.8.0")
        env.python.conda_dependencies.add_pip_package("pulp>=2.7.0")
        env.python.conda_dependencies.add_conda_package("python=3.9")
        
        # Register environment
        env.register(workspace=ws)
        print(f"✓ Registered environment: {env_name}")
    
    return env

def setup_duckdb_connection(building_id):
    """
    Setup DuckDB connection for Azure ML deployment.
    ENFORCES "USE AGENT OPTIMIZERS" - all data stays in DuckDB.
    """
    import duckdb
    
    print(f"📊 Setting up DuckDB connection for {building_id}...")
    
    # Connect to DuckDB database  
    db_path = Path.cwd() / "ems_data.duckdb"
    if not db_path.exists():
        db_path = Path.cwd() / "notebooks" / "ems_data.duckdb"
    
    if not db_path.exists():
        raise FileNotFoundError(f"EMS database not found at {db_path}")
    
    con = duckdb.connect(str(db_path), read_only=True)
    view_name = f"{building_id}_processed_data"
    
    # Validate data exists and get metadata
    row_count = con.execute(f"SELECT COUNT(*) as count FROM {view_name}").df()['count'][0]
    col_count = len(con.execute(f"DESCRIBE {view_name}").df())
    date_range = con.execute(f"SELECT MIN(DATE(utc_timestamp)) as min_date, MAX(DATE(utc_timestamp)) as max_date FROM {view_name}").df()
    
    print(f"✓ Connected to DuckDB: {row_count} rows, {col_count} columns")
    print(f"✓ Date range: {date_range['min_date'][0]} to {date_range['max_date'][0]}")
    print("✓ All data remains in DuckDB - no unnecessary DataFrame loading")
    
    return con, view_name

def run_learning_pipeline_with_mlflow(building_id="DE_KN_residential1", n_days=3):
    """
    Run the learning pipeline with MLflow tracking and model registration.
    Uses agent optimizers with strict compliance.
    """
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"ems_learning_production_{datetime.now().strftime('%Y%m%d_%H%M')}"):
        
        # Log parameters
        mlflow.log_params({
            "building_id": building_id,
            "n_days": n_days,
            "pipeline": "learning_production",
            "deployment_mode": "azure_ml",
            "timestamp": datetime.now().isoformat()
        })
        
        print(f"🎓 Starting learning pipeline for {building_id} with {n_days} days")
        
        # Setup DuckDB connection (using existing infrastructure)
        con, view_name = setup_duckdb_connection(building_id)
        
        # Get available days
        full_days_df = con.execute(f"""
            SELECT DATE(utc_timestamp) as day, COUNT(*) as hour_count 
            FROM {view_name} 
            GROUP BY DATE(utc_timestamp) 
            HAVING COUNT(*) = 24 
            ORDER BY DATE(utc_timestamp)
        """).df()
        
        full_days = pd.to_datetime(full_days_df['day']).dt.date.tolist()
        selected_days = full_days[:n_days]
        training_days = selected_days[:max(1, n_days//2)]
        
        print(f"✓ Using {len(training_days)} training days: {training_days}")
        
        # Initialize probability agent (USING AGENT ONLY)
        # Get existing priors from DuckDB
        try:
            priors_df = con.execute("SELECT * FROM device_hourly_probabilities").df()
            prob_agent = ProbabilityModelAgent(prob_dist_df=priors_df)
            print(f"✓ Initialized ProbabilityModelAgent with existing priors")
        except Exception as e:
            print(f"⚠ No existing priors found: {e}")
            # Create empty probability agent for training
            prob_agent = ProbabilityModelAgent()
            prob_agent.latest_distributions = {}
        
        # Get training data from DuckDB
        training_data = con.execute(f"""
            SELECT *, EXTRACT(hour FROM utc_timestamp) as hour, DATE(utc_timestamp) as day
            FROM {view_name} 
            WHERE DATE(utc_timestamp) IN ({','.join([f"'{day}'" for day in training_days])})
            ORDER BY utc_timestamp
        """).df()
        
        print(f"✓ Loaded {len(training_data)} training records")
        
        # Run probability training (USING AGENT METHOD ONLY)
        print("🎓 Training probability model using ProbabilityModelAgent.train()...")
        
        # Convert training days to string format for train method
        training_days_str = [day.strftime('%Y-%m-%d') for day in training_days]
        
        # System parameters for training from configuration
        if CONFIG_AVAILABLE:
            BATTERY_PARAMS = config.get_battery_config('large')
            GRID_PARAMS = config.get_grid_config('default')
            max_building_load = config.get('building.residential.max_building_load', 65.0)
        else:
            # Fallback hardcoded parameters
            BATTERY_PARAMS = {
                "max_charge_rate": 5.0,
                "max_discharge_rate": 5.0,
                "initial_soc": 8.0,
                "soc_min": 2.0,
                "soc_max": 15.0,
                "capacity": 15.0,
                "degradation_rate": 0.001,
                "efficiency_charge": 0.95,
                "efficiency_discharge": 0.95
            }
            
            GRID_PARAMS = {
                "import_price": 0.25,
                "export_price": 0.05,
                "max_import": 15.0,
                "max_export": 15.0
            }
            max_building_load = 65.0
        
        updated_specs, device_probabilities = prob_agent.train(
            building_id=building_id,
            days_list=training_days_str,
            device_specs=device_specs,
            weather_df=training_data,
            forecast_df=training_data,
            parquet_dir="not-used-with-DuckDB",
            max_building_load=max_building_load,
            battery_params=BATTERY_PARAMS,
            flexible_params={},
            grid_params=GRID_PARAMS
        )
        
        if not device_probabilities:
            raise RuntimeError("CRITICAL: ProbabilityModelAgent.train() failed - no fallbacks allowed")
        
        # device_probabilities is already returned directly from train method
        convergence_metrics = {}
        
        print(f"✓ Training completed for {len(device_probabilities)} devices")
        
        # Log training metrics
        mlflow.log_metrics({
            "devices_trained": len(device_probabilities),
            "training_days": len(training_days),
            "total_training_records": len(training_data)
        })
        
        # Log convergence metrics if available
        for device_id, metrics in convergence_metrics.items():
            if isinstance(metrics, dict):
                for metric_name, value in metrics.items():
                    mlflow.log_metric(f"{device_id}_{metric_name}", value)
        
        # Create training metadata
        training_metadata = {
            "building_id": building_id,
            "training_days": [str(d) for d in training_days],
            "devices_trained": list(device_probabilities.keys()),
            "training_timestamp": datetime.now().isoformat(),
            "convergence_metrics": convergence_metrics
        }
        
        # Create and register MLflow model
        print("📦 Registering model with MLflow...")
        
        model = EMSOptimizationModel()
        
        # Log model with artifacts
        model_name = "ems_optimizer"
        
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=model,
            registered_model_name=model_name,
            pip_requirements=[
                "pandas>=1.5.0",
                "numpy>=1.20.0",
                "duckdb>=0.8.0",
                "PuLP>=2.7.0",
                "mlflow>=2.0"
            ]
        )
        
        print(f"✓ Model registered: {model_name}")
        
        # Log training data sample as artifact
        sample_data = training_data.head(100)
        sample_file = "training_data_sample.csv"
        sample_data.to_csv(sample_file, index=False)
        mlflow.log_artifact(sample_file)
        os.remove(sample_file)
        
        # Log device probabilities as artifact
        prob_file = "device_probabilities.json"
        with open(prob_file, 'w') as f:
            # Convert to JSON-serializable format - handle nested structures
            json_probs = {}
            for device_id, hour_probs in device_probabilities.items():
                if isinstance(hour_probs, dict):
                    json_probs[device_id] = {}
                    for hour, prob in hour_probs.items():
                        if isinstance(prob, (int, float)):
                            json_probs[device_id][str(hour)] = float(prob)
                        else:
                            json_probs[device_id][str(hour)] = str(prob)
                else:
                    json_probs[device_id] = str(hour_probs)
            json.dump(json_probs, f, indent=2)
        mlflow.log_artifact(prob_file)
        os.remove(prob_file)
        
        # Log metadata
        metadata_file = "training_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(training_metadata, f, indent=2)
        mlflow.log_artifact(metadata_file)
        os.remove(metadata_file)
        
        print("✓ All artifacts logged to MLflow")
        
        # Close DuckDB connection
        con.close()
        
        return {
            'model_name': model_name,
            'device_probabilities': device_probabilities,
            'training_metadata': training_metadata,
            'run_id': mlflow.active_run().info.run_id
        }

def deploy_to_azure_ml():
    """Deploy the learning pipeline to Azure ML through MLflow"""
    print("🚀 Starting Azure ML deployment...")
    
    # Setup Azure ML workspace
    ws = setup_azure_workspace()
    
    # Configure MLflow tracking  
    experiment_name = setup_mlflow_tracking(ws)
    
    # Create environment
    env = create_environment(ws)
    
    # Run learning pipeline with MLflow
    print("🎓 Running learning pipeline...")
    results = run_learning_pipeline_with_mlflow()
    
    print(f"\n✅ DEPLOYMENT COMPLETE!")
    print(f"📊 Experiment: {experiment_name}")
    print(f"🏷️  Model: {results['model_name']}")
    print(f"🆔 Run ID: {results['run_id']}")
    print(f"🏢 Workspace: {ws.name}")
    
    return results

def test_deployed_model(model_name, building_id="DE_KN_residential1"):
    """Test the deployed model endpoint"""
    print(f"\n🧪 Testing deployed model: {model_name}")
    
    try:
        # Load the registered model
        client = mlflow.tracking.MlflowClient()
        latest_version = client.get_latest_versions(model_name, stages=["None"])[0]
        
        print(f"✓ Found model version: {latest_version.version}")
        
        # Load model for testing
        model_uri = f"models:/{model_name}/{latest_version.version}"
        loaded_model = mlflow.pyfunc.load_model(model_uri)
        
        print("✓ Model loaded successfully")
        
        # Test 1: Optimization mode
        print("\n1️⃣ Testing OPTIMIZATION mode...")
        optimization_input = {
            'mode': 'optimize',
            'building_id': building_id,
            'target_date': '2015-05-23',
            'price_profile': [0.25, 0.23, 0.21, 0.19, 0.18, 0.17, 0.20, 0.25, 
                             0.30, 0.28, 0.26, 0.24, 0.22, 0.20, 0.22, 0.25,
                             0.28, 0.32, 0.35, 0.33, 0.30, 0.28, 0.26, 0.24],
            'battery_enabled': True,
            'ev_enabled': False,
            'grid_params': {'import_price': 0.25, 'export_price': 0.05}
        }
        
        optimization_result = loaded_model.predict(optimization_input)
        print(f"✅ Optimization successful!")
        print(f"   Building: {optimization_result['building_id']}")
        print(f"   Date: {optimization_result['target_date']}")
        print(f"   Total cost: €{optimization_result['total_cost']:.2f}")
        print(f"   Savings: €{optimization_result['savings_vs_baseline']:.2f}")
        print(f"   Schedules: {len(optimization_result['optimized_schedules'])} devices")
        
        # Show sample device schedules
        print(f"\n📅 Sample device schedules (first 3 devices):")
        sample_count = 0
        for device_id, schedule in optimization_result['optimized_schedules'].items():
            if sample_count >= 3:
                break
            print(f"   {device_id}:")
            # Show active hours only to keep output clean
            active_hours = [f"h{i}:{schedule[i]:.1f}kW" for i in range(24) if schedule[i] > 0]
            if active_hours:
                print(f"     Active: {', '.join(active_hours)}")
            else:
                print(f"     Inactive all day")
            sample_count += 1
        
        # Test 2: Learning mode
        print("\n2️⃣ Testing LEARNING mode...")
        learning_input = {
            'mode': 'learn',
            'building_id': building_id,
            'actual_usage': {
                f'{building_id}_heat_pump': [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0,
                                           0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                f'{building_id}_dishwasher': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
            },
            'date': '2015-05-23'
        }
        
        learning_result = loaded_model.predict(learning_input)
        print(f"✅ Learning successful!")
        print(f"   Building: {learning_result['building_id']}")
        print(f"   Updated devices: {len(learning_result['updated_devices'])}")
        print(f"   Devices: {learning_result['updated_devices']}")
        
        # Show the actual updated PMFs
        if 'updated_pmfs' in learning_result:
            print(f"\n📊 Updated PMFs:")
            for device_id, pmf in learning_result['updated_pmfs'].items():
                print(f"   {device_id}:")
                # Show top 5 hours with highest probabilities
                sorted_hours = sorted(pmf.items(), key=lambda x: x[1], reverse=True)[:5]
                for hour, prob in sorted_hours:
                    print(f"     Hour {hour}: {prob:.4f}")
                print(f"     (showing top 5 of 24 hours)")
        
        print("\n✅ Complete optimization model testing successful!")
        return True
        
    except Exception as e:
        print(f"❌ Model testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main deployment function"""
    print("🏭 EMS Complete Optimization Pipeline - Azure ML Deployment")
    print("=" * 60)
    
    # Deploy to Azure ML
    results = deploy_to_azure_ml()
    
    # Test deployed model
    test_success = test_deployed_model(results['model_name'])
    
    if test_success:
        print("\n🎉 DEPLOYMENT AND TESTING SUCCESSFUL!")
        print("The EMS learning pipeline is now deployed to Azure ML")
        print("and ready for production use through MLflow.")
    else:
        print("\n❌ DEPLOYMENT FAILED TESTING")
        sys.exit(1)

if __name__ == "__main__":
    main()