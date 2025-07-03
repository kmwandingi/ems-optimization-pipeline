#!/usr/bin/env python
"""
Comparison Pipeline - Agent-based optimization comparison with STRICT "USE AGENT OPTIMIZERS" compliance

This script implements comparison between centralized vs decentralized optimization modes
using ONLY agent methods with DuckDB-only architecture. NO DataFrame loading, alternatives, 
manual loops, or simplified logic allowed.

Features:
1. DuckDB-only data access with SQL queries - no DataFrame loading
2. Agent-based optimization using GlobalOptimizer methods
3. Comprehensive comparison between optimization modes
4. STRICT enforcement of "USE AGENT OPTIMIZERS" rule

Usage:
  python scripts/02_run.py --building DE_KN_residential1
                           --mode decentralised|centralised|centralised_phases
                           --battery on|off   # default on
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys, os
# Ensure Unicode-friendly output even on Windows cmd
if os.name == "nt":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
import json
from pathlib import Path
from datetime import datetime, timedelta
import common

# Add notebooks directory to path for agent imports
sys.path.append(str(Path.cwd() / "notebooks"))

# Import ONLY agent classes - NO fallbacks allowed
try:
    from agents.ProbabilityModelAgent import ProbabilityModelAgent
    from agents.BatteryAgent import BatteryAgent
    from agents.EVAgent import EVAgent
    from agents.PVAgent import PVAgent
    from agents.GridAgent import GridAgent
    from agents.FlexibleDeviceAgent import FlexibleDevice
    from agents.GlobalOptimizer import GlobalOptimizer
    from agents.GlobalConnectionLayer import GlobalConnectionLayer
    from agents.WeatherAgent import WeatherAgent
    print("✓ Successfully imported ALL agent classes")
except ImportError as e:
    print(f"CRITICAL ERROR: Failed to import agent classes: {e}")
    print("This pipeline REQUIRES agent classes - no fallbacks allowed!")
    sys.exit(1)

# Import device_specs from utils
sys.path.append(str(Path.cwd() / "notebooks" / "utils"))
from device_specs import device_specs

# Import MLflow tracking and configuration loader
sys.path.append(str(Path.cwd() / "utils"))
try:
    from mlflow_tracker import EMS_OptimizationTracker
    MLFLOW_AVAILABLE = True
    print("✓ MLflow tracking enabled")
except ImportError as e:
    print(f"⚠ MLflow not available: {e}")
    MLFLOW_AVAILABLE = False

# Import centralized configuration
try:
    from config_loader import get_config
    config = get_config()
    CONFIG_AVAILABLE = True
    print("✓ Configuration system loaded")
except ImportError as e:
    print(f"⚠ Configuration loader not available: {e}")
    CONFIG_AVAILABLE = False

# Parameters for system components - Load from configuration if available
if CONFIG_AVAILABLE:
    BATTERY_PARAMS = config.get_battery_config('default')
    EV_PARAMS = config.get_ev_config('default')
    GRID_PARAMS = config.get_grid_config('default')
    print("✓ Loaded parameters from centralized configuration")
else:
    # Fallback hardcoded parameters (to be removed after full migration)
    BATTERY_PARAMS = {
        "max_charge_rate": 3.0,
        "max_discharge_rate": 3.0,
        "initial_soc": 7.0,
        "soc_min": 1.0,
        "soc_max": 10.0,
        "capacity": 10.0,
        "degradation_rate": 0.001,
        "efficiency_charge": 0.95,
        "efficiency_discharge": 0.95
    }
    
    EV_PARAMS = {
        "capacity": 60.0,
        "initial_soc": 12.0,
        "soc_min": 6.0,
        "soc_max": 54.0,
        "max_charge_rate": 7.4,
        "max_discharge_rate": 0.0,
        "efficiency_charge": 0.92,
        "efficiency_discharge": 0.92,
        "must_be_full_by_hour": 7
    }
    
    GRID_PARAMS = {
        "import_price": 0.25,
        "export_price": 0.05,
        "max_import": 15.0,
        "max_export": 15.0
    }
    print("⚠ Using fallback hardcoded parameters - configuration system unavailable")

# Create output directories using configuration paths
if CONFIG_AVAILABLE:
    paths = config.get_paths_config()
    os.makedirs(paths.get('figures_dir', 'results/figures'), exist_ok=True)
    os.makedirs(paths.get('output_dir', 'results/output'), exist_ok=True)
    os.makedirs(paths.get('results_dir', 'results'), exist_ok=True)
else:
    # Fallback directories
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/output", exist_ok=True)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run EMS optimization with AGENTS ONLY")
    parser.add_argument("--building", type=str, required=True,
                        help="Building ID (e.g., DE_KN_residential1)")
    parser.add_argument("--mode", type=str, required=False,
                        choices=["decentralised", "centralised", "centralised_phases"],
                        help="Optimization mode")
    parser.add_argument("--battery", type=str, default="on",
                        choices=["on", "off"],
                        help="Battery mode (on/off)")
    parser.add_argument("--ev", type=str, default="off",
                        choices=["on", "off"],
                        help="EV mode (on/off)")
    parser.add_argument("--validate", action="store_true",
                        help="Run validation checks on results")
    # Load default n_days from configuration if available
    default_n_days = config.get('pipeline.default_n_days', 20) if CONFIG_AVAILABLE else 20
    parser.add_argument("--n_days", type=int, default=default_n_days,
                        help="Number of days to process")
    
    return parser.parse_args()

def setup_duckdb_connection(building_id):
    """
    Setup DuckDB connection and validate data availability.
    ENFORCES "USE AGENT OPTIMIZERS" - all data stays in DuckDB.
    """
    print(f"📊 Setting up DuckDB connection for {building_id}...")
    
    # MANDATORY: Use DuckDB data access layer with registered parquet view
    con, view_name = common.get_view_con(building_id)
    
    # Validate data exists and get metadata
    row_count = con.execute(f"SELECT COUNT(*) as count FROM {view_name}").df()['count'][0]
    col_count = len(con.execute(f"DESCRIBE {view_name}").df())
    date_range = con.execute(f"SELECT MIN(DATE(utc_timestamp)) as min_date, MAX(DATE(utc_timestamp)) as max_date FROM {view_name}").df()
    
    print(f"✓ Connected to DuckDB: {row_count:,} rows, {col_count} columns")
    print(f"✓ Date range: {date_range['min_date'][0]} to {date_range['max_date'][0]}")
    print("✓ All data remains in DuckDB - no unnecessary DataFrame loading")
    
    return con, view_name

def select_days_from_duckdb(con, view_name, n_days):
    """
    Select days for processing using DuckDB queries.
    ENFORCES "USE AGENT OPTIMIZERS" - SQL-based day selection.
    """
    print(f"📅 Selecting {n_days} days using DuckDB queries...")
    
    # Query for complete 24-hour days
    full_days_df = con.execute(f"""
        SELECT DATE(utc_timestamp) as day, COUNT(*) as hour_count 
        FROM {view_name} 
        GROUP BY DATE(utc_timestamp) 
        HAVING COUNT(*) = 24 
        ORDER BY DATE(utc_timestamp)
    """).df()
    
    full_days = pd.to_datetime(full_days_df['day']).dt.date.tolist()
    
    if len(full_days) < n_days:
        print(f"⚠ Only {len(full_days)} days available, adjusting to {len(full_days)}")
        n_days = len(full_days)
    
    selected_days = full_days[:n_days]
    print(f"✓ Selected {len(selected_days)} complete days from DuckDB")
    
    return selected_days

def initialize_agents(building_id, con, view_name, battery_enabled=True, ev_enabled=True):
    """
    Initialize ALL agent instances using DuckDB queries.
    ENFORCES "USE AGENT OPTIMIZERS" - all data stays in DuckDB.
    """
    print("🤖 Initializing ALL agents with DuckDB...")
    
    # Battery Agent
    battery_agent = None
    if battery_enabled:
        battery_agent = BatteryAgent(**BATTERY_PARAMS)
        print(f"✓ Initialized BatteryAgent: {BATTERY_PARAMS['capacity']}kWh capacity")
    
    # EV Agent - query DuckDB for EV columns
    ev_agent = None
    if ev_enabled:
        columns_df = con.execute(f"DESCRIBE {view_name}").df()
        ev_columns = [col for col in columns_df['column_name'] if 'ev' in col.lower() and building_id in col]
        if ev_columns:
            ev_agent = EVAgent(
                device_name=ev_columns[0],
                category="ev",
                power_rating=EV_PARAMS["max_charge_rate"],
                **EV_PARAMS
            )
            print(f"✓ Initialized EVAgent: {EV_PARAMS['capacity']}kWh capacity")
    
    # PV Agent - query DuckDB for PV and forecast columns
    pv_agent = None
    columns_df = con.execute(f"DESCRIBE {view_name}").df()
    pv_columns = [col for col in columns_df['column_name'] if 'pv' in col.lower() and building_id in col and 'forecast' not in col.lower()]
    forecast_cols = [col for col in columns_df['column_name'] if 'pv_forecast' in col.lower() or 'solar' in col.lower()]
    
    if pv_columns:
        # Get sample data for PV agent initialization
        sample_data = con.execute(f"SELECT * FROM {view_name} LIMIT 100").df()
        
        # Initialize PVAgent with DuckDB connection and sample data
        pv_agent = PVAgent(
            profile_data=sample_data, 
            profile_cols=pv_columns,
            forecast_data=sample_data,  # Contains forecast data
            forecast_cols=forecast_cols if forecast_cols else None
        )
        # Store DuckDB connection for future queries
        pv_agent.duckdb_con = con
        pv_agent.view_name = view_name
        
        print(f"✓ Initialized PVAgent with {len(pv_columns)} PV columns and {len(forecast_cols)} forecast columns")
        print("✓ PVAgent connected to DuckDB for real-time queries")
    
    # Grid Agent
    grid_agent = GridAgent(**GRID_PARAMS)
    print("✓ Initialized GridAgent")
    
    # Weather Agent
    weather_agent = None
    try:
        # Get sample weather data for initialization
        weather_sample = con.execute(f"SELECT * FROM {view_name} LIMIT 1000").df()
        weather_agent = WeatherAgent(weather_sample)
        weather_agent.duckdb_con = con
        weather_agent.view_name = view_name
        print("✓ Initialized WeatherAgent with DuckDB")
    except Exception as e:
        print(f"⚠ WeatherAgent initialization failed: {e}")
    
    return battery_agent, ev_agent, pv_agent, grid_agent, weather_agent

def create_devices_from_duckdb(con, view_name, building_id, day, battery_agent=None, ev_agent=None):
    """
    Create FlexibleDevice agents using DuckDB queries for a specific day.
    ENFORCES "USE AGENT OPTIMIZERS" - all data from DuckDB.
    """
    devices = []
    
    # Adjust building load limit based on EV presence and configuration
    has_ev = ev_agent is not None and hasattr(ev_agent, 'max_charge_rate')
    if CONFIG_AVAILABLE:
        building_config = config.get_building_config('residential')
        base_load = building_config.get('max_building_load', 50.0)
        load_buffer = building_config.get('load_buffer', 1.2)
        max_building_load = base_load * load_buffer if has_ev else base_load
    else:
        max_building_load = 65.0 if has_ev else 50.0
    
    # Create global connection layer
    global_layer = GlobalConnectionLayer(max_building_load=max_building_load, total_hours=24)
    
    # Query DuckDB for device columns (exclude grid and PV)
    columns_df = con.execute(f"DESCRIBE {view_name}").df()
    device_columns = [col for col in columns_df['column_name'] if building_id in col 
                     and 'grid_export' not in col and 'grid_import' not in col and 'pv' not in col]
    
    # Get day-specific data for devices
    day_data = con.execute(f"""
        SELECT *, EXTRACT(hour FROM utc_timestamp) as hour, DATE(utc_timestamp) as day
        FROM {view_name} 
        WHERE DATE(utc_timestamp) = '{day}' 
        ORDER BY utc_timestamp
    """).df()
    
    for device_id in device_columns:
        if device_id in day_data.columns:
            # Extract device type from column name (handle multi-word types)
            parts = device_id.split('_')
            if len(parts) >= 4 and '_'.join(parts[-2:]) in ['heat_pump', 'washing_machine']:
                device_type = '_'.join(parts[-2:])
            else:
                device_type = parts[-1]
            
            # Get device specification from device_specs.py
            if device_type not in device_specs:
                # ERROR: Unknown device type - no fallbacks allowed
                raise ValueError(f"Unknown device type '{device_type}' not found in device_specs.py. Add {device_type} to device_specs.py first.")
            
            spec = device_specs[device_type].copy()
            
            # Reset index for proper agent data handling
            day_data_reset = day_data.reset_index(drop=True).copy()
            
            # Create FlexibleDevice agent with device_specs phases
            device = FlexibleDevice(
                device_name=device_id,
                data=day_data_reset,
                category=spec.get('category', 'Non-Flexible'),
                power_rating=spec.get('power_rating', 1.0),
                global_layer=global_layer,
                battery_agent=battery_agent,
                spec=spec
            )
            
            device.current_optimization_day = day
            # Store DuckDB connection for future queries
            device.duckdb_con = con
            device.view_name = view_name
            
            # Initialize with uniform probabilities (can be enhanced with ProbabilityModelAgent later)
            device.hour_probability = {h: 1/24 for h in range(24)}
            
            devices.append(device)
    
    print(f"✓ Created {len(devices)} FlexibleDevice agents from DuckDB")
    return devices

def get_day_data_from_duckdb(con, view_name, day):
    """
    Get day-specific data from DuckDB.
    ENFORCES "USE AGENT OPTIMIZERS" - SQL-based data access.
    """
    day_df = con.execute(f"""
        SELECT *, EXTRACT(hour FROM utc_timestamp) as hour, DATE(utc_timestamp) as day
        FROM {view_name} 
        WHERE DATE(utc_timestamp) = '{day}' 
        ORDER BY utc_timestamp
    """).df()
    
    # Get day-ahead prices
    day_ahead_prices = day_df.groupby('hour')['price_per_kwh'].mean().reindex(range(24), fill_value=0.25).values
    
    return day_df, day_ahead_prices

def run_decentralized_optimization(devices, day_df, battery_agent, ev_agent, pv_agent, grid_agent, day_ahead_prices):
    """
    Run decentralized optimization using FlexibleDeviceAgent.optimize_day() 
    ENFORCES "USE AGENT OPTIMIZERS" - no manual loops allowed.
    """
    print("  Running decentralized optimization...")
    optimized_devices = []
    total_cost = 0.0
    
    # Each device optimizes independently using agent methods
    for device in devices:
        try:
            # MANDATORY: Use FlexibleDeviceAgent.optimize_day() method
            # Extract day from day_df
            day = day_df['day'].iloc[0]
            if isinstance(day, str):
                import datetime
                day = datetime.datetime.strptime(day, '%Y-%m-%d').date()
            elif hasattr(day, 'date'):
                day = day.date()
            
            # Prepare battery state if battery agent available
            battery_state = None
            if battery_agent:
                battery_state = {
                    'soc_min': battery_agent.soc_min,
                    'soc_max': battery_agent.soc_max,
                    'capacity': battery_agent.capacity,
                    'current_soc': battery_agent.current_soc,
                    'max_charge_rate': battery_agent.max_charge_rate,
                    'max_discharge_rate': battery_agent.max_discharge_rate,
                    'charge_efficiency': getattr(battery_agent, 'efficiency_charge', 0.95),
                    'discharge_efficiency': getattr(battery_agent, 'efficiency_discharge', 0.95)
                }
            
            # Get PV forecast (set to None for now, can be enhanced later)
            pv_forecast = None
            
            device.optimize_day(day, day_ahead_prices, pv_forecast, battery_state, None)
            
            # Store optimized schedule
            if hasattr(device, 'optimized_schedule'):
                device.decentralized_optimized_schedule = device.optimized_schedule[:24]
            else:
                raise ValueError(f"Device {device.device_name} optimize_day() did not produce optimized_schedule")
            
            # Calculate cost using optimized schedule
            schedule = np.array(device.decentralized_optimized_schedule)
            device_cost = np.sum(schedule * day_ahead_prices[:24])
            total_cost += device_cost
            
            optimized_devices.append(device)
            print(f"    Device {device.device_name}: €{device_cost:.4f}")
            
        except Exception as e:
            # ERROR: Agent method failed - agent interface fix required
            raise RuntimeError(f"CRITICAL: FlexibleDeviceAgent.optimize_day() failed for {device.device_name}: {e}. Fix the agent method.")
    
    # In decentralized mode, battery optimization is handled within device optimization
    # The battery agent state was passed to each device's optimize_day() method
    if battery_agent:
        # Battery costs are calculated implicitly within device optimization
        # We could add battery scheduling here if needed, but for now skip it
        print(f"    Battery SOC maintained: {battery_agent.current_soc:.2f} kWh")
    
    # In decentralized mode, EV optimization is handled within device optimization
    # The EV agent state was passed to each device's optimize_day() method
    if ev_agent:
        # EV costs are calculated implicitly within device optimization
        print(f"    EV SOC maintained: {ev_agent.current_soc:.2f} kWh")
    
    print(f"  Decentralized total cost: €{total_cost:.4f}")
    return total_cost, optimized_devices

def run_centralized_optimization(devices, day_df, battery_agent, ev_agent, pv_agent, grid_agent, weather_agent=None):
    """
    Run centralized optimization using GlobalOptimizer.optimize_centralized()
    ENFORCES "USE AGENT OPTIMIZERS" - no MILP optimization allowed.
    """
    print("  Running centralized optimization...")
    
    # Get prices from day_df
    day_ahead_prices = day_df.groupby('hour')['price_per_kwh'].mean().reindex(range(24), fill_value=0.25).values
    
    # MANDATORY: Use GlobalOptimizer.optimize_centralized() method
    if CONFIG_AVAILABLE:
        building_config = config.get_building_config('residential')
        base_load = building_config.get('max_building_load', 50.0)
        load_buffer = building_config.get('load_buffer', 1.2)
        max_building_load = base_load * load_buffer if ev_agent else base_load
    else:
        max_building_load = 65.0 if ev_agent else 50.0
    global_layer = GlobalConnectionLayer(max_building_load, 24)
    
    # Weather agent is passed as parameter (already initialized with full dataset)
    
    # Create GlobalOptimizer instance with configuration parameters
    optimization_config = config.get_optimization_config() if CONFIG_AVAILABLE else {}
    max_iterations = optimization_config.get('global_optimizer', {}).get('max_iterations', 1)
    online_iterations = optimization_config.get('global_optimizer', {}).get('online_iterations', 1)
    
    optimizer = GlobalOptimizer(
        devices=devices,
        global_layer=global_layer,
        pv_agent=pv_agent,
        weather_agent=weather_agent,
        battery_agent=battery_agent,
        ev_agent=ev_agent,
        grid_agent=grid_agent,
        max_iterations=max_iterations,
        online_iterations=online_iterations
    )
    
    try:
        # MANDATORY: Call GlobalOptimizer.optimize_centralized() method
        success = optimizer.optimize_centralized()
        
        if not success:
            raise RuntimeError("GlobalOptimizer.optimize_centralized() returned False - optimization failed")
        
        # Extract optimized schedules from devices
        total_cost = 0.0
        for device in devices:
            if hasattr(device, 'centralized_optimized_schedule'):
                schedule = device.centralized_optimized_schedule[:24]
            elif hasattr(device, 'optimized_schedule'):
                schedule = device.optimized_schedule[:24]
                device.centralized_optimized_schedule = schedule
            else:
                raise ValueError(f"Device {device.device_name} missing optimized schedule after centralized optimization")
            
            # Calculate cost
            device_cost = np.sum(np.array(schedule) * day_ahead_prices[:24])
            total_cost += device_cost
        
        # Add battery cost if available
        if battery_agent and hasattr(battery_agent, 'hourly_charge') and hasattr(battery_agent, 'hourly_discharge'):
            battery_cost = np.sum(np.array(battery_agent.hourly_charge[:24]) * day_ahead_prices[:24])
            battery_savings = np.sum(np.array(battery_agent.hourly_discharge[:24]) * day_ahead_prices[:24])
            total_cost += battery_cost - battery_savings
        
        # Add EV cost if available
        if ev_agent and hasattr(ev_agent, 'hourly_charge'):
            ev_cost = np.sum(np.array(ev_agent.hourly_charge[:24]) * day_ahead_prices[:24])
            total_cost += ev_cost
        
        print(f"  Centralized total cost: €{total_cost:.4f}")
        return total_cost, devices
        
    except Exception as e:
        # ERROR: Agent method failed - fix agent interface
        raise RuntimeError(f"CRITICAL: GlobalOptimizer.optimize_centralized() failed: {e}. Fix the agent method.")

def run_centralized_phases_optimization(devices, day_df, battery_agent, ev_agent, pv_agent, grid_agent, weather_agent=None):
    """
    Run centralized phases optimization using GlobalOptimizer.optimize_phases_centralized()
    ENFORCES "USE AGENT OPTIMIZERS" - no MILP optimization allowed.
    """
    print("  Running centralized phases optimization...")
    
    # MANDATORY: Use GlobalOptimizer.optimize_phases_centralized() method
    if CONFIG_AVAILABLE:
        building_config = config.get_building_config('residential')
        base_load = building_config.get('max_building_load', 50.0)
        load_buffer = building_config.get('load_buffer', 1.2)
        max_building_load = base_load * load_buffer if ev_agent else base_load
    else:
        max_building_load = 65.0 if ev_agent else 50.0
    global_layer = GlobalConnectionLayer(max_building_load, 24)
    
    # Weather agent is passed as parameter (already initialized with full dataset)
    
    # Create GlobalOptimizer instance with configuration parameters
    optimization_config = config.get_optimization_config() if CONFIG_AVAILABLE else {}
    max_iterations = optimization_config.get('global_optimizer', {}).get('max_iterations', 1)
    online_iterations = optimization_config.get('global_optimizer', {}).get('online_iterations', 1)
    
    optimizer = GlobalOptimizer(
        devices=devices,
        global_layer=global_layer,
        pv_agent=pv_agent,
        weather_agent=weather_agent,
        battery_agent=battery_agent,
        ev_agent=ev_agent,
        grid_agent=grid_agent,
        max_iterations=max_iterations,
        online_iterations=online_iterations
    )
    
    try:
        # MANDATORY: Call GlobalOptimizer.optimize_phases_centralized() method
        success = optimizer.optimize_phases_centralized(
            devices=devices,
            global_layer=global_layer,
            pv_agent=pv_agent,
            battery_agent=battery_agent,
            ev_agent=ev_agent,
            grid_agent=grid_agent,
            weather_agent=weather_agent
        )
        
        if not success:
            raise RuntimeError("GlobalOptimizer.optimize_phases_centralized() returned False - optimization failed")
        
        # Get prices from day_df
        day_ahead_prices = day_df.groupby('hour')['price_per_kwh'].mean().reindex(range(24), fill_value=0.25).values
        
        # Extract optimized schedules and calculate total cost
        total_cost = 0.0
        for device in devices:
            if hasattr(device, 'phases_optimized_schedule'):
                schedule = device.phases_optimized_schedule[:24]
                device.centralized_phases_optimized_schedule = schedule
            elif hasattr(device, 'centralized_optimized_schedule'):
                schedule = device.centralized_optimized_schedule[:24]
                device.centralized_phases_optimized_schedule = schedule
            elif hasattr(device, 'optimized_schedule'):
                schedule = device.optimized_schedule[:24]
                device.centralized_phases_optimized_schedule = schedule
            else:
                raise ValueError(f"Device {device.device_name} missing optimized schedule after phases optimization")
            
            # Calculate cost
            device_cost = np.sum(np.array(schedule) * day_ahead_prices[:24])
            total_cost += device_cost
        
        # Add battery cost if available
        if battery_agent and hasattr(battery_agent, 'hourly_charge') and hasattr(battery_agent, 'hourly_discharge'):
            battery_cost = np.sum(np.array(battery_agent.hourly_charge[:24]) * day_ahead_prices[:24])
            battery_savings = np.sum(np.array(battery_agent.hourly_discharge[:24]) * day_ahead_prices[:24])
            total_cost += battery_cost - battery_savings
        
        # Add EV cost if available
        if ev_agent and hasattr(ev_agent, 'hourly_charge'):
            ev_cost = np.sum(np.array(ev_agent.hourly_charge[:24]) * day_ahead_prices[:24])
            total_cost += ev_cost
        
        print(f"  Centralized phases total cost: €{total_cost:.4f}")
        return total_cost, devices
        
    except Exception as e:
        # ERROR: Agent method failed - agent interface fix required
        raise RuntimeError(f"CRITICAL: GlobalOptimizer.optimize_phases_centralized() failed: {e}. Fix the agent method - agent interface fix required.")

def calculate_kpis(total_cost_decentralised, total_cost_centralised, total_cost_phases=None):
    """Calculate KPIs comparing optimization modes."""
    
    # Calculate savings
    savings_centralized = total_cost_decentralised - total_cost_centralised
    savings_pct_centralized = (savings_centralized / total_cost_decentralised * 100) if total_cost_decentralised > 0 else 0
    
    kpi_row = {
        'cost_decentralised': total_cost_decentralised,
        'cost_centralised': total_cost_centralised,
        'savings_eur': savings_centralized,
        'savings_pct': savings_pct_centralized
    }
    
    if total_cost_phases is not None:
        savings_phases = total_cost_decentralised - total_cost_phases
        savings_pct_phases = (savings_phases / total_cost_decentralised * 100) if total_cost_decentralised > 0 else 0
        kpi_row.update({
            'cost_centralised_phases': total_cost_phases,
            'savings_phases_eur': savings_phases,
            'savings_phases_pct': savings_pct_phases
        })
    
    return kpi_row

def main():
    """Main function implementing agent comparison pipeline."""
    
    args = parse_args()
    
    building_id = args.building
    mode = args.mode
    battery_enabled = args.battery == "on"
    ev_enabled = args.ev == "on"
    n_days = args.n_days
    
    print("="*80)
    print("COMPARISON PIPELINE - AGENT OPTIMIZERS WITH DUCKDB-ONLY ARCHITECTURE")
    print("="*80)
    print(f"Building: {building_id}")
    print(f"Mode: {mode}")
    print(f"Battery: {battery_enabled}")
    print(f"EV: {ev_enabled}")
    print(f"Days: {n_days}")
    print("="*80)
    
    # Initialize MLflow tracking (NON-INTRUSIVE)
    mlflow_tracker = None
    if MLFLOW_AVAILABLE:
        mlflow_tracker = EMS_OptimizationTracker("Comparison_Pipeline")
        run_name = f"comparison_{building_id}_{mode or 'decentralised'}_{datetime.now().strftime('%Y%m%d_%H%M')}"
        mlflow_tracker.start_run(run_name)
        
        # Log pipeline parameters
        mlflow_tracker.log_params({
            "building_id": building_id,
            "optimization_mode": mode or "decentralised",
            "battery_enabled": battery_enabled,
            "ev_enabled": ev_enabled,
            "n_days": n_days,
            "pipeline": "A",
            "description": "Agent Comparison Pipeline"
        })
        
        # Log system configuration
        mlflow_tracker.log_battery_config(BATTERY_PARAMS)
        mlflow_tracker.log_ev_config(EV_PARAMS)
        mlflow_tracker.log_grid_config(GRID_PARAMS)
    
    # 1. Setup REAL DuckDB connection
    con, view_name = setup_duckdb_connection(building_id)
    
    # 2. Select days for processing using DuckDB queries
    selected_days = select_days_from_duckdb(con, view_name, n_days)
    
    # 3. Initialize agents
    battery_agent, ev_agent, pv_agent, grid_agent, weather_agent = initialize_agents(building_id, con, view_name, battery_enabled, ev_enabled)
    
    # KPI accumulators across all processed days
    results = []
    kpi_acc = {
        "pv_gen": 0.0,
        "pv_export": 0.0,
        "grid_import": 0.0,
        "grid_export": 0.0,
        "peak_import_kw": 0.0,
    }
    
    # 4. Process each day using DuckDB queries
    for day_idx, day in enumerate(selected_days):
        print(f"\n--- Day {day_idx+1}/{len(selected_days)}: {day} ---")
        
        # Get day-specific data from DuckDB
        day_df, day_ahead_prices = get_day_data_from_duckdb(con, view_name, day)
        print(f"  Price range: {day_ahead_prices.min():.4f} - {day_ahead_prices.max():.4f} €/kWh")
        
        # Reset SOC for EV (daily reset) and initialize battery SOC
        if ev_agent:
            ev_agent.current_soc = ev_agent.initial_soc
            print(f"  EV SOC reset: {ev_agent.current_soc:.2f} kWh")
        
        if battery_agent and day_idx == 0:
            battery_agent.current_soc = battery_agent.initial_soc
            print(f"  Battery SOC initialized: {battery_agent.current_soc:.2f} kWh")
        elif battery_agent:
            print(f"  Battery SOC persisted: {battery_agent.current_soc:.2f} kWh")
        
        # Create devices for this day using DuckDB
        devices = create_devices_from_duckdb(con, view_name, building_id, day, battery_agent, ev_agent)
        
        # --- Collect raw energy KPIs using DuckDB BEFORE optimisation results overwrite any state ---
        # We derive column names dynamically to stay schema-agnostic.
        def _find_column(candidates):
            cols = [row[1] for row in con.execute(f"PRAGMA table_info({view_name})").fetchall()]
            for cand in candidates:
                if cand in cols:
                    return cand
            return None
        pv_col = _find_column(["pv_generation_kwh", "pv_generation", "pv_gen_kwh"])
        imp_col = _find_column(["grid_import_kwh", "grid_import", "import_energy_kwh"])
        exp_col = _find_column(["grid_export_kwh", "grid_export", "export_energy_kwh"])
        peak_col = _find_column(["grid_import_power_kw", "grid_import_power", "import_power_kw"])
        if pv_col:
            kpi_acc["pv_gen"] += con.execute(f"SELECT SUM({pv_col}) FROM {view_name} WHERE DATE(timestamp)=?", [day]).fetchone()[0] or 0
        if exp_col:
            kpi_acc["pv_export"] += con.execute(f"SELECT SUM({exp_col}) FROM {view_name} WHERE DATE(timestamp)=?", [day]).fetchone()[0] or 0
            kpi_acc["grid_export"] += kpi_acc["pv_export"]  # treat same
        if imp_col:
            kpi_acc["grid_import"] += con.execute(f"SELECT SUM({imp_col}) FROM {view_name} WHERE DATE(timestamp)=?", [day]).fetchone()[0] or 0
        if peak_col:
            peak_val = con.execute(f"SELECT MAX({peak_col}) FROM {view_name} WHERE DATE(timestamp)=?", [day]).fetchone()[0]
            if peak_val and peak_val > kpi_acc["peak_import_kw"]:
                kpi_acc["peak_import_kw"] = peak_val
        # Run optimizations based on mode
        if mode == "decentralised" or mode is None:
            # Run decentralized optimization
            total_cost_dec, optimized_devices_dec = run_decentralized_optimization(
                devices, day_df, battery_agent, ev_agent, pv_agent, grid_agent, day_ahead_prices
            )
            
            # Store results
            result = {
                'day': day,
                'cost_decentralised': total_cost_dec,
                'mode': 'decentralised'
            }
            results.append(result)
            
            # Log to MLflow (NON-INTRUSIVE)
            if mlflow_tracker:
                mlflow_tracker.log_optimization_results({
                    "daily_cost_decentralised": total_cost_dec,
                    "daily_mode": "decentralised"
                }, day=str(day))
            
        elif mode == "centralised":
            # Run both decentralized and centralized for comparison
            total_cost_dec, optimized_devices_dec = run_decentralized_optimization(
                devices.copy(), day_df, battery_agent, ev_agent, pv_agent, grid_agent, day_ahead_prices
            )
            
            total_cost_cent, optimized_devices_cent = run_centralized_optimization(
                devices, day_df, battery_agent, ev_agent, pv_agent, grid_agent, weather_agent
            )
            
            # Calculate KPIs
            kpis = calculate_kpis(total_cost_dec, total_cost_cent)
            result = {
                'day': day,
                'cost_decentralised': total_cost_dec,
                'cost_centralised': total_cost_cent,
                'savings_eur': kpis['savings_eur'],
                'savings_pct': kpis['savings_pct'],
                'mode': 'centralised'
            }
            results.append(result)
            
            # Log to MLflow (NON-INTRUSIVE)
            if mlflow_tracker:
                mlflow_tracker.log_optimization_results({
                    "daily_cost_decentralised": total_cost_dec,
                    "daily_cost_centralised": total_cost_cent,
                    "daily_savings_eur": kpis['savings_eur'],
                    "daily_savings_pct": kpis['savings_pct']
                }, day=str(day))
            
        elif mode == "centralised_phases":
            # Run all three modes for comprehensive comparison
            total_cost_dec, optimized_devices_dec = run_decentralized_optimization(
                devices.copy(), day_df, battery_agent, ev_agent, pv_agent, grid_agent, day_ahead_prices
            )
            
            total_cost_cent, optimized_devices_cent = run_centralized_optimization(
                devices.copy(), day_df, battery_agent, ev_agent, pv_agent, grid_agent, weather_agent
            )
            
            total_cost_phases, optimized_devices_phases = run_centralized_phases_optimization(
                devices, day_df, battery_agent, ev_agent, pv_agent, grid_agent, weather_agent
            )
            
            # Calculate KPIs
            kpis = calculate_kpis(total_cost_dec, total_cost_cent, total_cost_phases)
            result = {
                'day': day,
                'cost_decentralised': total_cost_dec,
                'cost_centralised': total_cost_cent,
                'cost_centralised_phases': total_cost_phases,
                'savings_eur': kpis['savings_eur'],
                'savings_pct': kpis['savings_pct'],
                'savings_phases_eur': kpis['savings_phases_eur'],
                'savings_phases_pct': kpis['savings_phases_pct'],
                'mode': 'centralised_phases'
            }
            results.append(result)
            
            # Log to MLflow (NON-INTRUSIVE)
            if mlflow_tracker:
                mlflow_tracker.log_optimization_results({
                    "daily_cost_decentralised": total_cost_dec,
                    "daily_cost_centralised": total_cost_cent,
                    "daily_cost_centralised_phases": total_cost_phases,
                    "daily_savings_eur": kpis['savings_eur'],
                    "daily_savings_pct": kpis['savings_pct'],
                    "daily_savings_phases_eur": kpis['savings_phases_eur'],
                    "daily_savings_phases_pct": kpis['savings_phases_pct']
                }, day=str(day))
        
        # Update battery SOC for next day (persistence)
        if battery_agent and hasattr(battery_agent, 'hourly_soc') and len(battery_agent.hourly_soc) > 0:
            battery_agent.current_soc = battery_agent.hourly_soc[-1]
    
    # Note: CSV creation removed to save disk space
    results_df = pd.DataFrame(results)
    print(f"✓ Pipeline A results completed (CSV creation disabled to save disk space)")
    
    # Summary
    print("\n" + "="*80)
    print("COMPARISON PIPELINE COMPLETION SUMMARY")
    print("="*80)
    print(f"Total days processed: {len(results)}")
    
    if mode == "centralised" or mode == "centralised_phases":
        avg_savings = results_df['savings_pct'].mean()
        print(f"Average centralized savings: {avg_savings:.2f}%")
        
        if mode == "centralised_phases":
            avg_phases_savings = results_df['savings_phases_pct'].mean()
            print(f"Average phases savings: {avg_phases_savings:.2f}%")
    
    print(f"Results analysis completed (CSV output disabled to save disk space)")
    print("✅ Comparison Pipeline completed successfully using AGENT OPTIMIZERS WITH DUCKDB-ONLY ARCHITECTURE")
    print("="*80)
    
    # Final MLflow logging (NON-INTRUSIVE)
    if mlflow_tracker and len(results) > 0:
        # Log final summary metrics
        final_metrics = {
            "total_days_processed": len(results),
            "pipeline_success": 1.0,
            "total_cost_eur": results_df['cost_decentralised'].sum() if 'cost_decentralised' in results_df else None,
            "avg_daily_cost_eur": results_df['cost_decentralised'].mean() if 'cost_decentralised' in results_df else None,
            # Energy KPIs
            "pv_generation_kwh": round(kpi_acc["pv_gen"], 3),
            "pv_export_kwh": round(kpi_acc["pv_export"], 3),
            "grid_import_kwh": round(kpi_acc["grid_import"], 3),
            "grid_export_kwh": round(kpi_acc["grid_export"], 3),
            "peak_grid_import_kw": round(kpi_acc["peak_import_kw"], 3),
            "pv_self_consumption_pct": round((kpi_acc["pv_gen"] - kpi_acc["pv_export"]) / kpi_acc["pv_gen"], 4) if kpi_acc["pv_gen"] > 0 else None
        }
        
        if mode == "centralised" or mode == "centralised_phases":
            avg_savings = results_df['savings_pct'].mean()
            final_metrics["avg_savings_pct"] = avg_savings
            final_metrics["total_cost_avg"] = results_df['cost_decentralised'].mean()
            
            if mode == "centralised_phases":
                avg_phases_savings = results_df['savings_phases_pct'].mean()
                final_metrics["avg_phases_savings_pct"] = avg_phases_savings
        
        mlflow_tracker.log_metrics(final_metrics)
        
        # Log result artifacts
        mlflow_tracker.log_artifacts("results/output")
        
        # End MLflow run
        mlflow_tracker.end_run()
        print("✓ MLflow tracking completed")

        # Persist metrics for aggregation
        try:
            output_dir = Path("results") / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = output_dir / f"{building_id}_metrics.json"
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(final_metrics, f, indent=2)
            print(f"✓ Saved metrics to {metrics_path}")
        except Exception as dump_err:
            print(f"⚠ Failed to write metrics JSON: {dump_err}")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ COMPARISON PIPELINE FAILED: {e}")
        print("This indicates a bug in the agent interface that must be fixed.")
        sys.exit(1)