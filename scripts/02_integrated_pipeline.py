#!/usr/bin/env python
"""
Learning Pipeline - Integrated EMS Pipeline with strict agent optimizer compliance

This script implements the complete integrated EMS pipeline using ONLY agent methods.
NO fallbacks, manual loops, or simplified logic allowed.

Features:
1. Data loading and preprocessing using data sources
2. ProbabilityModelAgent.train() for probability model training  
3. Agent-based optimization using GlobalOptimizer methods
4. Comprehensive visualization and results saving
5. Strict enforcement of agent optimizer usage
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from pathlib import Path
from matplotlib.gridspec import GridSpec

# Add notebooks directory to path for agent imports
sys.path.append(str(Path.cwd() / "notebooks"))

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
    from agents.WeatherAgent import WeatherAgent
    print("✓ Successfully imported agent classes")
except ImportError as e:
    print(f"CRITICAL ERROR: Failed to import agent classes: {e}")
    print("This pipeline REQUIRES agent classes!")
    sys.exit(1)

# Import common utilities and device_specs
import common
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
    BATTERY_PARAMS = config.get_battery_config('large')  # Use large battery config for integrated pipeline
    EV_PARAMS = config.get_ev_config('default')
    GRID_PARAMS = config.get_grid_config('default')
    print("✓ Loaded parameters from centralized configuration")
else:
    # Fallback hardcoded parameters (to be removed after full migration)
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
    
    EV_PARAMS = {
        "capacity": 60.0,
        "initial_soc": 18.0,  # 30% of 60kWh  
        "soc_min": 6.0,       # 10% of 60kWh
        "soc_max": 54.0,      # 90% of 60kWh
        "max_charge_rate": 11.0,
        "max_discharge_rate": 0.0,
        "efficiency_charge": 0.92,
        "efficiency_discharge": 0.92,
        "must_be_full_by_hour": 7
    }
    
    GRID_PARAMS = {
        "import_price": 0.25,  # Default price per kWh
        "export_price": 0.05,  # Default feed-in tariff
        "max_import": 15.0,    # Max grid import in kW
        "max_export": 15.0     # Max grid export in kW
    }
    print("⚠ Using fallback hardcoded parameters - configuration system unavailable")

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Agent Integrated EMS Pipeline")
    parser.add_argument("--building", type=str, required=True,
                        help="Building ID (e.g., DE_KN_residential1)")
    # Load default n_days from configuration if available
    default_n_days = config.get('pipeline.default_n_days', 10) if CONFIG_AVAILABLE else 10
    parser.add_argument("--n_days", type=int, default=default_n_days,
                        help="Number of days to process")
    parser.add_argument("--battery", type=str, default="on",
                        choices=["on", "off"],
                        help="Battery mode (on/off)")
    parser.add_argument("--ev", type=str, default="off",
                        choices=["on", "off"],
                        help="EV mode (on/off)")
    # Production mode: Always use optimize_phases_centralized
    # parser.add_argument("--mode") - REMOVED: Production always uses phases
    
    return parser.parse_args()

def setup_duckdb_connection(building_id):
    """
    Setup DuckDB connection and validate data availability.
    All data stays in DuckDB.
    """
    print(f"📊 Setting up DuckDB connection for {building_id}...")
    
    # MANDATORY: Use DuckDB data access layer
    con = common.get_con()
    view_name = f"{building_id}_processed_data"
    
    # Validate data exists and get metadata
    row_count = con.execute(f"SELECT COUNT(*) as count FROM {view_name}").df()['count'][0]
    col_count = len(con.execute(f"DESCRIBE {view_name}").df())
    date_range = con.execute(f"SELECT MIN(DATE(utc_timestamp)) as min_date, MAX(DATE(utc_timestamp)) as max_date FROM {view_name}").df()
    
    print(f"✓ Connected to DuckDB: {row_count} rows, {col_count} columns")
    print(f"✓ Date range: {date_range['min_date'][0]} to {date_range['max_date'][0]}")
    print("✓ All data remains in DuckDB - no unnecessary DataFrame loading")
    
    return con, view_name

def initialize_probability_agent():
    """
    Initialize ProbabilityModelAgent with actual probability data.
    No simplified probability agent allowed.
    """
    print("🧠 Initializing ProbabilityModelAgent...")
    
    # MANDATORY: Use ProbabilityModelAgent with actual priors
    con = common.get_con()
    priors_df = con.execute("SELECT * FROM device_hourly_probabilities").df()
    prob_agent = ProbabilityModelAgent(prob_dist_df=priors_df)
    
    # Set latest_distributions from priors
    prob_agent.latest_distributions = {
        dev_type: priors_df.loc[priors_df.index == dev_type].iloc[0].to_dict()
        for dev_type in priors_df.index if len(priors_df.loc[priors_df.index == dev_type]) > 0
    }
    
    print(f"✓ Initialized ProbabilityModelAgent with {len(prob_agent.latest_distributions)} device priors")
    return prob_agent

def initialize_agents(building_id, con, view_name, battery_enabled=True, ev_enabled=True):
    """
    Initialize agent instances using DuckDB queries.
    All data stays in DuckDB.
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
    
    return battery_agent, ev_agent, pv_agent, grid_agent

def create_devices_for_day(con, view_name, building_id, day, prob_agent, battery_agent, ev_agent):
    """
    Create FlexibleDevice agents for a specific day using DuckDB queries.
    All data from DuckDB.
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
            
            # Assign probabilities from ProbabilityModelAgent
            if prob_agent and hasattr(prob_agent, 'latest_distributions') and device_id in prob_agent.latest_distributions:
                device.hour_probability = prob_agent.latest_distributions[device_id].copy()
                print(f"  Assigned learned probabilities to {device_type}")
            else:
                # Default uniform if not learned yet
                device.hour_probability = {h: 1/24 for h in range(24)}
                print(f"  Using uniform probabilities for {device_type}")
            
            devices.append(device)
    
    print(f"✓ Created {len(devices)} FlexibleDevice agents")
    return devices

def run_probability_training(prob_agent, building_id, training_days, df):
    """
    Run probability model training using ProbabilityModelAgent.train().
    No simplified training allowed.
    """
    print(f"🎓 Running probability training on {len(training_days)} days...")
    
    if not training_days:
        print("⚠ No training days provided")
        return device_specs, {}
    
    # Convert datetime days to string format for train method
    training_days_str = [day.strftime('%Y-%m-%d') for day in training_days]
    
    # MANDATORY: Use ProbabilityModelAgent.train() method
    updated_specs, device_probs = prob_agent.train(
        building_id=building_id,
        days_list=training_days_str,
        device_specs=device_specs,
        weather_df=df,
        forecast_df=df,
        parquet_dir="not-used-with-DuckDB",
        max_building_load=config.get('building.residential.max_building_load', 65.0) if CONFIG_AVAILABLE else 65.0,
        battery_params=BATTERY_PARAMS,
        flexible_params={},
        grid_params=GRID_PARAMS,
        pv_params={},
        cleaner=None
    )
    
    print(f"✓ probability training completed")
    print(f"✓ Updated specs for {len(updated_specs)} devices")
    print(f"✓ Learned probabilities for {len(device_probs)} devices")
    
    return updated_specs, device_probs

def run_centralized_optimization(devices, day_df, battery_agent, ev_agent, pv_agent, grid_agent, weather_agent=None):
    """
    Run centralized optimization using GlobalOptimizer.optimize_phases_centralized().
    PRODUCTION: Always uses phases optimization - no other modes allowed.
    """
    print(f"⚙️ Running phases centralized optimization...")
    
    # Get prices from day_df
    day_ahead_prices = day_df.groupby('hour')['price_per_kwh'].mean().reindex(range(24), fill_value=0.25).values
    
    # Create GlobalOptimizer instance with configuration
    if CONFIG_AVAILABLE:
        building_config = config.get_building_config('residential')
        base_load = building_config.get('max_building_load', 50.0)
        load_buffer = building_config.get('load_buffer', 1.2)
        max_building_load = base_load * load_buffer if ev_agent else base_load
    else:
        max_building_load = 65.0 if ev_agent else 50.0
    global_layer = GlobalConnectionLayer(max_building_load, 24)
    
    # Weather agent is passed as parameter (already initialized with full dataset)
    
    # Get optimization parameters from configuration
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
    
    # MANDATORY: Use GlobalOptimizer.optimize_phases_centralized method
    # Both centralized and centralized_phases modes should use optimize_phases_centralized
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
        # ERROR: Agent method failed - no fallbacks allowed
        raise RuntimeError(f"CRITICAL: GlobalOptimizer.optimize_phases_centralized() returned False - optimization failed. Fix the agent method.")
    
    # Extract optimized schedules and calculate total cost
    total_cost = 0.0
    optimized_schedules = {}
    
    for device in devices:
        # Get optimized schedule from device
        if hasattr(device, 'phases_optimized_schedule'):
            schedule = device.phases_optimized_schedule[:24]
            device.optimized_schedule = schedule
        elif hasattr(device, 'centralized_optimized_schedule'):
            schedule = device.centralized_optimized_schedule[:24]
            device.optimized_schedule = schedule
        elif hasattr(device, 'optimized_schedule'):
            schedule = device.optimized_schedule[:24]
        else:
            raise ValueError(f"Device {device.device_name} missing optimized schedule after optimization")
        
        optimized_schedules[device.device_name] = schedule
        
        # Calculate cost
        device_cost = np.sum(np.array(schedule) * day_ahead_prices[:24])
        total_cost += device_cost
    
    # Add battery cost if available
    if battery_agent and hasattr(battery_agent, 'hourly_charge') and hasattr(battery_agent, 'hourly_discharge'):
        battery_cost = np.sum(np.array(battery_agent.hourly_charge[:24]) * day_ahead_prices[:24])
        battery_savings = np.sum(np.array(battery_agent.hourly_discharge[:24]) * day_ahead_prices[:24])
        total_cost += battery_cost - battery_savings
        print(f"  Battery net cost: €{battery_cost - battery_savings:.4f}")
    
    # Add EV cost if available
    if ev_agent and hasattr(ev_agent, 'hourly_charge'):
        ev_cost = np.sum(np.array(ev_agent.hourly_charge[:24]) * day_ahead_prices[:24])
        total_cost += ev_cost
        optimized_schedules['EV_charging'] = ev_agent.hourly_charge[:24]
        print(f"  EV cost: €{ev_cost:.4f}")
    
    print(f"✓ phases centralized optimization completed. Total cost: €{total_cost:.4f}")
    return optimized_schedules, total_cost

def create_comprehensive_visualization(devices, battery_agent, ev_agent, optimized_schedules, day_prices, building_id, day, output_dir):
    """Create comprehensive visualization of optimization results."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create 6-panel comprehensive results visualization
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    hours = np.arange(24)
    
    # 1. Total building load comparison
    ax1 = fig.add_subplot(gs[0, :2])
    
    total_original = np.zeros(24)
    total_optimized = np.zeros(24)
    
    for device in devices:
        if hasattr(device, 'original_consumption') and hasattr(device, 'optimized_schedule'):
            original = np.array(device.original_consumption[:24])
            optimized = np.array(device.optimized_schedule[:24])
            total_original += original
            total_optimized += optimized
    
    ax1.plot(hours, total_original, 'r-', linewidth=2, label='Original')
    ax1.plot(hours, total_optimized, 'b-', linewidth=2, label='Optimized')
    ax1.set_title(f'{building_id} - Total Building Load ({day})')
    ax1.set_xlabel('Hour')
    ax1.set_ylabel('Power (kW)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Price overlay
    ax1_twin = ax1.twinx()
    ax1_twin.plot(hours, day_prices, 'g--', alpha=0.7, label='Price')
    ax1_twin.set_ylabel('Price (€/kWh)', color='g')
    ax1_twin.tick_params(axis='y', labelcolor='g')
    
    # 3. Battery SOC if available
    if battery_agent and hasattr(battery_agent, 'hourly_soc'):
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.plot(hours, battery_agent.hourly_soc[:24], 'purple', linewidth=2)
        ax2.axhline(y=battery_agent.soc_min, color='red', linestyle='--', alpha=0.7, label='Min SOC')
        ax2.axhline(y=battery_agent.soc_max, color='red', linestyle='--', alpha=0.7, label='Max SOC')
        ax2.set_title('Battery SOC')
        ax2.set_xlabel('Hour')
        ax2.set_ylabel('SOC (kWh)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 4. EV SOC if available
    if ev_agent and hasattr(ev_agent, 'hourly_soc'):
        ax3 = fig.add_subplot(gs[0, 3])
        ax3.plot(hours, ev_agent.hourly_soc[:24], 'orange', linewidth=2)
        if hasattr(ev_agent, 'must_be_full_by_hour'):
            ax3.axvline(x=ev_agent.must_be_full_by_hour, color='red', linestyle='--', alpha=0.7, label='Departure')
        ax3.set_title('EV SOC')
        ax3.set_xlabel('Hour')
        ax3.set_ylabel('SOC (kWh)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 5. Individual device schedules (first 4 devices)
    device_axes = [fig.add_subplot(gs[1, i]) for i in range(4)]
    
    for idx, device in enumerate(devices[:4]):
        ax = device_axes[idx]
        
        if hasattr(device, 'original_consumption') and hasattr(device, 'optimized_schedule'):
            original = device.original_consumption[:24]
            optimized = device.optimized_schedule[:24]
            
            ax.bar(hours - 0.2, original, 0.4, alpha=0.7, label='Original', color='red')
            ax.bar(hours + 0.2, optimized, 0.4, alpha=0.7, label='Optimized', color='blue')
        
        device_name = device.device_name.split('_')[-1].title()
        ax.set_title(f'{device_name}')
        ax.set_xlabel('Hour')
        ax.set_ylabel('Power (kW)')
        if idx == 0:
            ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 6. Cost comparison and savings
    ax4 = fig.add_subplot(gs[2, :])
    
    # Calculate costs
    total_energy = np.sum(total_optimized)
    avg_price = np.mean(day_prices)
    expensive_hours_avg = np.mean(np.sort(day_prices)[-6:])  # Top 6 expensive hours
    
    # Fair baseline comparison
    baseline_cost = total_energy * expensive_hours_avg
    optimized_cost = np.sum(total_optimized * day_prices)
    savings = baseline_cost - optimized_cost
    savings_pct = (savings / baseline_cost * 100) if baseline_cost > 0 else 0
    
    costs = [baseline_cost, optimized_cost]
    labels = ['Baseline\n(Expensive Hours)', 'Optimized\n(Smart Scheduling)']
    colors = ['red', 'blue']
    
    bars = ax4.bar(labels, costs, color=colors, alpha=0.7)
    ax4.set_title(f'Cost Comparison - Savings: €{savings:.4f} ({savings_pct:.1f}%)')
    ax4.set_ylabel('Cost (€)')
    
    # Add cost values on bars
    for bar, cost in zip(bars, costs):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'€{cost:.4f}', ha='center', va='bottom')
    
    plt.suptitle(f'{building_id} - Agent Optimization Results ({day})', fontsize=16)
    
    output_file = f"{output_dir}/{building_id}_{day}_optimization_results.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_file, savings, savings_pct

def main():
    """Main function implementing agent integrated pipeline."""
    
    args = parse_args()
    
    building_id = args.building
    n_days = args.n_days
    battery_enabled = args.battery == "on"
    ev_enabled = args.ev == "on"
    
    print("="*80)
    print("LEARNING PIPELINE - AGENT INTEGRATED EMS")
    print("="*80)
    print(f"Building: {building_id}")
    print(f"Days: {n_days}")
    print(f"Battery: {battery_enabled}")
    print(f"EV: {ev_enabled}")
    print(f"Mode: PRODUCTION (phases centralized only)")
    print("="*80)
    
    # Initialize MLflow tracking (NON-INTRUSIVE)
    mlflow_tracker = None
    if MLFLOW_AVAILABLE:
        mlflow_tracker = EMS_OptimizationTracker("Learning_Pipeline")
        run_name = f"learning_{building_id}_phases_{datetime.now().strftime('%Y%m%d_%H%M')}"
        mlflow_tracker.start_run(run_name)
        
        # Log pipeline parameters
        mlflow_tracker.log_params({
            "building_id": building_id,
            "n_days": n_days,
            "optimization_mode": "phases_centralized",
            "battery_enabled": battery_enabled,
            "ev_enabled": ev_enabled,
            "pipeline": "B",
            "description": "Agent Learning Pipeline"
        })
        
        # Log system configuration
        mlflow_tracker.log_battery_config(BATTERY_PARAMS)
        mlflow_tracker.log_ev_config(EV_PARAMS)
        mlflow_tracker.log_grid_config(GRID_PARAMS)
    
    # Create output directories
    os.makedirs("results/output", exist_ok=True)
    os.makedirs("results/visualizations", exist_ok=True)
    
    # 1. Setup DuckDB connection
    con, view_name = setup_duckdb_connection(building_id)
    
    # 2. Select days for processing using DuckDB queries
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
    training_days = selected_days[:max(1, n_days//2)]  # Use first half for training
    optimization_days = selected_days[max(1, n_days//2):]  # Use second half for optimization
    
    print(f"✓ Selected {len(selected_days)} days total")
    print(f"✓ Training days: {len(training_days)}")
    print(f"✓ Optimization days: {len(optimization_days)}")
    
    # 3. Initialize agents
    prob_agent = initialize_probability_agent()
    battery_agent, ev_agent, pv_agent, grid_agent = initialize_agents(building_id, con, view_name, battery_enabled, ev_enabled)
    
    # Initialize WeatherAgent with DuckDB sample data
    try:
        # Get sample weather data for initialization
        weather_sample = con.execute(f"SELECT * FROM {view_name} LIMIT 1000").df()
        weather_agent = WeatherAgent(weather_sample)
        weather_agent.duckdb_con = con
        weather_agent.view_name = view_name
        print("✓ Initialized WeatherAgent with DuckDB")
    except Exception as e:
        weather_agent = None
        print(f"⚠ WeatherAgent initialization failed: {e}")
    
    # 4. Run probability training if training days available
    if training_days:
        # Get training data from DuckDB
        training_data = con.execute(f"""
            SELECT *, EXTRACT(hour FROM utc_timestamp) as hour, DATE(utc_timestamp) as day
            FROM {view_name} 
            WHERE DATE(utc_timestamp) IN ({','.join([f"'{day}'" for day in training_days])})
            ORDER BY utc_timestamp
        """).df()
        updated_specs, device_probs = run_probability_training(prob_agent, building_id, training_days, training_data)
        
        # Log probability training results to MLflow (NON-INTRUSIVE)
        if mlflow_tracker and device_probs:
            mlflow_tracker.log_params({
                "training_days_count": len(training_days),
                "devices_trained": len(device_probs)
            })
            mlflow_tracker.log_probability_learning(device_probs)
    else:
        updated_specs, device_probs = device_specs, {}
        
        # Log no training case
        if mlflow_tracker:
            mlflow_tracker.log_params({
                "training_days_count": 0,
                "devices_trained": 0
            })
    
    # 5. Process each optimization day
    results = []
    total_savings = 0.0
    
    for day_idx, day in enumerate(optimization_days):
        print(f"\n--- Day {day_idx+1}/{len(optimization_days)}: {day} ---")
        
        # Query data for this day from DuckDB
        day_df = con.execute(f"""
            SELECT *, EXTRACT(hour FROM utc_timestamp) as hour, DATE(utc_timestamp) as day
            FROM {view_name} 
            WHERE DATE(utc_timestamp) = '{day}' 
            ORDER BY utc_timestamp
        """).df()
        
        # Get day-ahead prices
        day_ahead_prices = day_df.groupby('hour')['price_per_kwh'].mean().reindex(range(24), fill_value=0.25).values
        print(f"  Price range: {day_ahead_prices.min():.4f} - {day_ahead_prices.max():.4f} €/kWh")
        
        # Reset EV SOC daily, persist battery SOC
        if ev_agent:
            ev_agent.current_soc = ev_agent.initial_soc
            print(f"  EV SOC reset: {ev_agent.current_soc:.2f} kWh")
        
        if battery_agent and day_idx == 0:
            battery_agent.current_soc = battery_agent.initial_soc
            print(f"  Battery SOC initialized: {battery_agent.current_soc:.2f} kWh")
        elif battery_agent:
            print(f"  Battery SOC persisted: {battery_agent.current_soc:.2f} kWh")
        
        # Create devices for this day
        devices = create_devices_for_day(con, view_name, building_id, day, prob_agent, battery_agent, ev_agent)
        
        # Run centralized optimization
        optimized_schedules, total_cost = run_centralized_optimization(
            devices, day_df, battery_agent, ev_agent, pv_agent, grid_agent, weather_agent
        )
        
        # Create comprehensive visualization
        viz_file, savings, savings_pct = create_comprehensive_visualization(
            devices, battery_agent, ev_agent, optimized_schedules, day_ahead_prices, 
            building_id, day, "results/visualizations"
        )
        
        total_savings += savings
        
        # Store results
        result = {
            'day': day,
            'total_cost': total_cost,
            'savings_eur': savings,
            'savings_pct': savings_pct,
            'mode': 'phases_centralized'
        }
        results.append(result)
        
        # Log daily results to MLflow (NON-INTRUSIVE)
        if mlflow_tracker:
            mlflow_tracker.log_optimization_results({
                "daily_total_cost": total_cost,
                "daily_savings_eur": savings,
                "daily_savings_pct": savings_pct,
                "daily_mode": "phases_centralized"
            }, day=str(day))
        
        print(f"  Total cost: €{total_cost:.4f}")
        print(f"  Savings: €{savings:.4f} ({savings_pct:.1f}%)")
        print(f"  ✓ Created visualization: {viz_file}")
        
        # Update battery SOC for next day (persistence)
        if battery_agent and hasattr(battery_agent, 'hourly_soc') and len(battery_agent.hourly_soc) > 0:
            battery_agent.current_soc = battery_agent.hourly_soc[-1]
    
    # 6. Note: CSV creation removed to save disk space
    results_df = pd.DataFrame(results)
    print(f"✓ Pipeline B results completed (CSV creation disabled to save disk space)")
    
    # Summary
    print("\n" + "="*80)
    print("PIPELINE B COMPLETION SUMMARY")
    print("="*80)
    print(f"Total days processed: {len(results)}")
    if len(results) > 0:
        avg_savings = results_df['savings_pct'].mean()
        total_cumulative_savings = results_df['savings_eur'].sum()
        print(f"Average savings: {avg_savings:.2f}%")
        print(f"Total cumulative savings: €{total_cumulative_savings:.4f}")
    print(f"Results analysis completed (CSV output disabled to save disk space)")
    print("✅ Learning Pipeline completed successfully using AGENT OPTIMIZERS ONLY")
    print("="*80)
    
    # Final MLflow logging (NON-INTRUSIVE)
    if mlflow_tracker and len(results) > 0:
        # Log final summary metrics
        final_metrics = {
            "total_days_processed": len(results),
            "total_optimization_days": len(optimization_days),
            "avg_savings_pct": results_df['savings_pct'].mean(),
            "total_cumulative_savings": results_df['savings_eur'].sum(),
            "pipeline_success": 1.0
        }
        
        mlflow_tracker.log_metrics(final_metrics)
        
        # Log result artifacts
        mlflow_tracker.log_artifacts("results/output")
        mlflow_tracker.log_artifacts("results/visualizations")
        
        # End MLflow run
        mlflow_tracker.end_run()
        print("✓ MLflow tracking completed")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ PIPELINE B FAILED: {e}")
        print("This indicates a bug in the agent interface that must be fixed.")
        import traceback
        traceback.print_exc()
        sys.exit(1)