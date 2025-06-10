#!/usr/bin/env python3
"""
Cost Savings by Building - Single Graph Script

Creates Figure 2: Cost Savings Percentage by Building and Scenario
Uses REAL data from parquet files and REAL agent optimization results.
ONE GRAPH ONLY - Well structured and formatted.
"""

import logging
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "notebooks"))

# Import REAL agents - NO FALLBACKS
from agents.GlobalOptimizer import GlobalOptimizer
from agents.FlexibleDeviceAgent import FlexibleDevice
from agents.GlobalConnectionLayer import GlobalConnectionLayer
from utils.device_specs import device_specs

# JADS Colors
JADS_COLORS = {
    "brand_orange": "#F5854F",
    "brand_red": "#E75C4B", 
    "brand_grey": "#6D6E71",
    "brand_gradient_blue": "#273E9E"
}

def load_and_optimize_building(building_id, data_dir):
    """Load REAL building data and run REAL optimization"""
    
    parquet_file = data_dir / f"{building_id}_processed_data.parquet"
    if not parquet_file.exists():
        raise FileNotFoundError(f"Real data file not found: {parquet_file}")
    
    df = pd.read_parquet(parquet_file)
    if df.empty:
        raise ValueError(f"Empty data file: {parquet_file}")
    
    # Handle datetime index
    if df.index.name == 'utc_timestamp':
        df = df.reset_index()
    
    # Get complete day
    df['date'] = pd.to_datetime(df['utc_timestamp']).dt.date
    daily_counts = df.groupby('date').size()
    complete_days = daily_counts[daily_counts == 24].index
    
    if len(complete_days) == 0:
        raise ValueError(f"No complete days in {building_id}")
    
    # Use middle day
    selected_date = complete_days[len(complete_days)//2]
    day_data = df[df['date'] == selected_date].copy()
    day_data = day_data.sort_values('utc_timestamp').reset_index(drop=True)
    day_data['hour'] = pd.to_datetime(day_data['utc_timestamp']).dt.hour
    
    # Create REAL devices
    global_layer = GlobalConnectionLayer(max_building_load=65.0, total_hours=24)
    device_columns = [col for col in day_data.columns 
                     if building_id in col and 'grid' not in col and 'pv' not in col]
    
    devices = []
    for device_col in device_columns:
        if device_col in day_data.columns and day_data[device_col].sum() > 0:
            parts = device_col.split('_')
            device_type = '_'.join(parts[-2:]) if len(parts) >= 3 else parts[-1]
            
            spec = device_specs.get(device_type, {
                'category': 'Partially Flexible',
                'power_rating': 1.0
            })
            
            device = FlexibleDevice(
                device_name=device_col,
                data=day_data,
                category=spec['category'],
                power_rating=spec['power_rating'],
                global_layer=global_layer,
                battery_agent=None,
                spec=spec
            )
            
            device.original_consumption = day_data[device_col].values
            devices.append(device)
    
    if not devices:
        raise ValueError(f"No devices found in {building_id}")
    
    # Run REAL optimization
    optimizer = GlobalOptimizer(
        devices=devices, 
        global_layer=global_layer,
        battery_agent=None, 
        ev_agent=None
    )
    # Run optimization
    results = optimizer.optimize()
    
    if results is None:
        raise RuntimeError(f"Optimization failed for {building_id}")
    
    # Calculate REAL savings
    original_cost = 0
    optimized_cost = 0
    
    for device in devices:
        if not hasattr(device, 'optimized_schedule'):
            raise ValueError(f"Missing optimized_schedule for {device.device_name}")
        
        original_cost += np.sum(device.original_consumption * day_data['price_per_kwh'].values)
        optimized_cost += np.sum(device.optimized_schedule * day_data['price_per_kwh'].values)
    
    savings_percentage = ((original_cost - optimized_cost) / original_cost * 100) if original_cost > 0 else 0
    
    return savings_percentage

def create_cost_savings_graph():
    """Create single well-formatted cost savings graph"""
    
    data_dir = project_root / "notebooks" / "data"
    building_ids = [
        'DE_KN_residential1', 'DE_KN_residential2', 'DE_KN_residential3',
        'DE_KN_residential4', 'DE_KN_residential5', 'DE_KN_residential6'
    ]
    
    # Collect REAL results
    results = []
    for building_id in building_ids:
        try:
            savings_pct = load_and_optimize_building(building_id, data_dir)
            results.append({
                'Building': building_id.replace('DE_KN_', '').title(),
                'Cost Savings (%)': savings_pct
            })
            logger.info(f"Processed {building_id}: {savings_pct:.1f}% savings")
        except Exception as e:
            logger.error(f"Failed {building_id}: {e}")
            continue
    
    if not results:
        raise RuntimeError("No REAL results generated")
    
    df = pd.DataFrame(results)
    
    # Create single well-formatted graph
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Bar plot with JADS colors
    bars = sns.barplot(
        data=df, 
        x='Building', 
        y='Cost Savings (%)',
        palette=[JADS_COLORS["brand_orange"]] * len(df),
        ax=ax
    )
    
    # Format the graph
    ax.set_title('Cost Savings by Building\nEMS Optimization Results', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Building', fontsize=14, fontweight='bold')
    ax.set_ylabel('Cost Savings (%)', fontsize=14, fontweight='bold')
    
    # Add value labels on bars
    for i, bar in enumerate(bars.patches):
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                   xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom',
                   fontsize=12, fontweight='bold')
    
    # Style improvements
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(df['Cost Savings (%)']) * 1.15)
    
    # Add summary statistics
    mean_savings = df['Cost Savings (%)'].mean()
    ax.axhline(y=mean_savings, color=JADS_COLORS["brand_red"], 
              linestyle='--', alpha=0.7, linewidth=2)
    ax.text(0.02, 0.98, f'Average: {mean_savings:.1f}%', 
           transform=ax.transAxes, fontsize=12, fontweight='bold',
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save outputs
    output_path = project_root / "figures" / "cost_savings_by_building.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved figure to {output_path}")
    
    table_path = project_root / "tables" / "cost_savings_by_building.csv"
    df.to_csv(table_path, index=False)
    logger.info(f"Saved data to {table_path}")
    
    plt.close()
    return df

if __name__ == "__main__":
    try:
        results = create_cost_savings_graph()
        print("SUCCESS: Cost savings graph created using REAL data and REAL agents")
        print(f"Average savings: {results['Cost Savings (%)'].mean():.1f}%")
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)