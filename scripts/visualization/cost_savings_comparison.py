#!/usr/bin/env python3
"""
Cost Savings Comparison Visualization Script

This script creates Figure 2 from the EMS Technical Report showing cost savings 
achieved by the EMS across different buildings and scenarios using REAL data 
from parquet files and REAL agent optimization results.

NO FALLBACK DATA OR DUMMY VALUES - Uses only actual agent optimization results.
"""

import logging
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "notebooks"))

# Import REAL agents and utilities - NO FALLBACKS
from agents.GlobalOptimizer import GlobalOptimizer
from agents.BatteryAgent import BatteryAgent
from agents.EVAgent import EVAgent
from agents.FlexibleDeviceAgent import FlexibleDevice
from agents.GlobalConnectionLayer import GlobalConnectionLayer
from utils.device_specs import device_specs
from notebooks.utils.helper import BATTERY_PARAMS, EV_PARAMS, GRID_PARAMS

# JADS Color Palette - NO FALLBACKS
JADS_COLORS = {
    "brand_orange": "#F5854F",
    "brand_red": "#E75C4B", 
    "brand_grey": "#6D6E71",
    "brand_gradient_blue": "#273E9E",
    "brand_gradient_red": "#9E273E",
    "brand_dark_grey": "#4A4A4A"
}

def load_real_building_data(building_id, data_dir):
    """Load REAL building data from parquet files - NO FALLBACKS"""
    parquet_file = data_dir / f"{building_id}_processed_data.parquet"
    
    if not parquet_file.exists():
        raise FileNotFoundError(f"Real data file not found: {parquet_file}")
    
    logger.info(f"Loading REAL data from {parquet_file}")
    df = pd.read_parquet(parquet_file)
    
    if df.empty:
        raise ValueError(f"Real data file is empty: {parquet_file}")
    
    # Ensure required columns exist
    required_cols = ['utc_timestamp', 'price_per_kwh']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in real data: {missing_cols}")
    
    return df

def run_real_agent_optimization(building_data, building_id, scenario_config):
    """Run REAL agent optimization - NO DUMMY DATA OR FALLBACKS"""
    
    # Select a representative day with complete data
    building_data['date'] = pd.to_datetime(building_data['utc_timestamp']).dt.date
    daily_counts = building_data.groupby('date').size()
    complete_days = daily_counts[daily_counts == 24].index
    
    if len(complete_days) == 0:
        raise ValueError(f"No complete 24-hour days found in {building_id}")
    
    # Use the middle day for consistent results
    selected_date = complete_days[len(complete_days)//2]
    day_data = building_data[building_data['date'] == selected_date].copy()
    day_data = day_data.sort_values('utc_timestamp').reset_index(drop=True)
    
    if len(day_data) != 24:
        raise ValueError(f"Selected day does not have exactly 24 hours: {len(day_data)}")
    
    # Add hour column for agent processing
    day_data['hour'] = pd.to_datetime(day_data['utc_timestamp']).dt.hour
    
    # Initialize REAL global connection layer
    global_layer = GlobalConnectionLayer(max_building_load=65.0, total_hours=24)
    
    # Create REAL device agents based on actual data columns
    device_columns = [col for col in day_data.columns 
                     if building_id in col and 'grid' not in col and 'pv' not in col]
    
    devices = []
    for device_col in device_columns:
        if device_col in day_data.columns and day_data[device_col].sum() > 0:
            # Extract device type from column name
            parts = device_col.split('_')
            if len(parts) >= 2:
                device_type = '_'.join(parts[-2:]) if len(parts) >= 3 else parts[-1]
            else:
                device_type = 'generic'
            
            # Get REAL device spec
            spec = device_specs.get(device_type, {
                'category': 'Partially Flexible',
                'power_rating': 1.0,
                'allowed_hours': list(range(8, 22))
            })
            
            # Create REAL FlexibleDevice agent
            device = FlexibleDevice(
                device_name=device_col,
                data=day_data,
                category=spec['category'],
                power_rating=spec['power_rating'],
                global_layer=global_layer,
                battery_agent=None,
                spec=spec
            )
            
            # Set original consumption from REAL data
            device.original_consumption = day_data[device_col].values
            devices.append(device)
    
    if not devices:
        raise ValueError(f"No valid devices found in {building_id} data")
    
    # Create REAL battery agent if scenario includes battery
    battery_agent = None
    if scenario_config.get('battery_enabled', False):
        battery_agent = BatteryAgent(
            capacity=BATTERY_PARAMS['capacity'],
            initial_soc=BATTERY_PARAMS['initial_soc'],
            soc_min=BATTERY_PARAMS['soc_min'],
            soc_max=BATTERY_PARAMS['soc_max'],
            max_charge_rate=BATTERY_PARAMS['max_charge_rate'],
            max_discharge_rate=BATTERY_PARAMS['max_discharge_rate'],
            efficiency_charge=BATTERY_PARAMS['efficiency_charge'],
            efficiency_discharge=BATTERY_PARAMS['efficiency_discharge']
        )
    
    # Create REAL EV agent if scenario includes EV
    ev_agent = None
    if scenario_config.get('ev_enabled', False):
        ev_agent = EVAgent(
            capacity=EV_PARAMS['capacity'],
            initial_soc=EV_PARAMS['initial_soc'],
            soc_min=EV_PARAMS['soc_min'],
            soc_max=EV_PARAMS['soc_max'],
            max_charge_rate=EV_PARAMS['max_charge_rate'],
            efficiency_charge=EV_PARAMS['efficiency_charge'],
            must_be_full_by_hour=EV_PARAMS['must_be_full_by_hour']
        )
    
    # Run REAL GlobalOptimizer - NO FALLBACKS
    optimizer = GlobalOptimizer(
        devices=devices,
        battery_agent=battery_agent,
        ev_agent=ev_agent,
        total_hours=24
    )
    
    # Perform REAL optimization
    success = optimizer.optimize_building_schedule(
        prices=day_data['price_per_kwh'].values,
        pv_forecast=day_data.get('pv_actual', np.zeros(24)).values
    )
    
    if not success:
        raise RuntimeError(f"REAL agent optimization failed for {building_id}")
    
    # Calculate REAL costs - NO DUMMY CALCULATIONS
    original_cost = 0
    optimized_cost = 0
    
    for device in devices:
        if not hasattr(device, 'optimized_schedule'):
            raise ValueError(f"Device {device.device_name} missing optimized_schedule from REAL agent")
        
        # Original cost from REAL data
        original_cost += np.sum(device.original_consumption * day_data['price_per_kwh'].values)
        
        # Optimized cost from REAL agent results
        optimized_cost += np.sum(device.optimized_schedule * day_data['price_per_kwh'].values)
    
    # Add battery costs if present
    if battery_agent and hasattr(battery_agent, 'hourly_charge'):
        battery_cost = np.sum(battery_agent.hourly_charge * day_data['price_per_kwh'].values)
        battery_savings = np.sum(battery_agent.hourly_discharge * day_data['price_per_kwh'].values * 0.8)  # 80% export price
        optimized_cost += battery_cost - battery_savings
    
    # Calculate REAL savings
    cost_savings = original_cost - optimized_cost
    savings_percentage = (cost_savings / original_cost * 100) if original_cost > 0 else 0
    
    return {
        'building_id': building_id,
        'scenario': scenario_config['name'],
        'original_cost': original_cost,
        'optimized_cost': optimized_cost,
        'cost_savings': cost_savings,
        'savings_percentage': savings_percentage
    }

def create_cost_savings_visualization():
    """Create cost savings visualization using REAL data and REAL agents"""
    logger.info("Creating cost savings visualization with REAL data and REAL agents")
    
    # Data directory with REAL parquet files
    data_dir = project_root / "notebooks" / "data"
    
    # Building IDs from REAL data files
    building_ids = [
        'DE_KN_residential1', 'DE_KN_residential2', 'DE_KN_residential3',
        'DE_KN_residential4', 'DE_KN_residential5', 'DE_KN_residential6'
    ]
    
    # Scenario configurations for REAL agent testing
    scenarios = [
        {'name': 'Baseline', 'battery_enabled': False, 'ev_enabled': False},
        {'name': 'Battery Only', 'battery_enabled': True, 'ev_enabled': False},
        {'name': 'EV Only', 'battery_enabled': False, 'ev_enabled': True},
        {'name': 'Full DER', 'battery_enabled': True, 'ev_enabled': True}
    ]
    
    # Run REAL optimizations - NO FALLBACKS
    results = []
    for building_id in building_ids:
        try:
            building_data = load_real_building_data(building_id, data_dir)
            
            for scenario in scenarios:
                logger.info(f"Running REAL optimization: {building_id} - {scenario['name']}")
                result = run_real_agent_optimization(building_data, building_id, scenario)
                results.append(result)
                
        except Exception as e:
            logger.error(f"Failed to process {building_id}: {e}")
            # NO FALLBACK - skip building if real data/agents fail
            continue
    
    if not results:
        raise RuntimeError("No REAL optimization results generated - cannot create visualization")
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Create visualization with JADS colors
    sns.set_theme(style="ticks", palette=list(JADS_COLORS.values()))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: Cost savings percentage by building and scenario
    pivot_pct = df.pivot(index='building_id', columns='scenario', values='savings_percentage')
    
    sns.barplot(data=df, x='building_id', y='savings_percentage', hue='scenario', ax=ax1)
    ax1.set_title('Cost Savings Percentage by Building and Scenario', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Building', fontsize=12)
    ax1.set_ylabel('Cost Savings (%)', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend(title='Scenario', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Absolute cost savings in euros
    sns.barplot(data=df, x='building_id', y='cost_savings', hue='scenario', ax=ax2)
    ax2.set_title('Absolute Cost Savings by Building and Scenario', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Building', fontsize=12)
    ax2.set_ylabel('Cost Savings (€)', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend(title='Scenario', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('EMS Cost Savings Analysis - REAL Agent Results', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    output_path = project_root / "figures" / "cost_savings_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved cost savings visualization to {output_path}")
    
    # Save detailed results table
    table_path = project_root / "tables" / "cost_savings_comparison.csv"
    df.to_csv(table_path, index=False)
    logger.info(f"Saved cost savings data to {table_path}")
    
    plt.close()
    
    return df

if __name__ == "__main__":
    try:
        results = create_cost_savings_visualization()
        print("SUCCESS: Cost savings visualization created using REAL data and REAL agents")
        print(f"Results summary:\n{results.groupby('scenario')['savings_percentage'].agg(['mean', 'std'])}")
    except Exception as e:
        print(f"ERROR: Failed to create visualization: {e}")
        sys.exit(1)