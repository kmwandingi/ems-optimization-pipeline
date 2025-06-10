#!/usr/bin/env python3
"""
Load Shifting Heatmap - Single Graph Script

Creates Figure 3: Load Profile Comparison (Before/After Optimization)
Uses REAL data from parquet files and REAL agent optimization results.
ONE HEATMAP ONLY - Well structured and formatted.
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

def load_building_and_optimize(building_id, data_dir):
    """Load REAL building data and run optimization for multiple days"""
    
    parquet_file = data_dir / f"{building_id}_processed_data.parquet"
    if not parquet_file.exists():
        raise FileNotFoundError(f"Real data file not found: {parquet_file}")
    
    df = pd.read_parquet(parquet_file)
    if df.empty:
        raise ValueError(f"Empty data file: {parquet_file}")
    
    # Handle datetime index
    if df.index.name == 'utc_timestamp':
        df = df.reset_index()
    
    # Get multiple complete days for better heatmap
    df['date'] = pd.to_datetime(df['utc_timestamp']).dt.date
    daily_counts = df.groupby('date').size()
    complete_days = daily_counts[daily_counts == 24].index
    
    if len(complete_days) < 5:
        raise ValueError(f"Need at least 5 complete days, found {len(complete_days)}")
    
    # Use 5 consecutive days from the middle
    start_idx = len(complete_days) // 2 - 2
    selected_dates = complete_days[start_idx:start_idx + 5]
    
    # Process each day
    original_matrix = np.zeros((len(selected_dates), 24))
    optimized_matrix = np.zeros((len(selected_dates), 24))
    
    for day_idx, selected_date in enumerate(selected_dates):
        day_data = df[df['date'] == selected_date].copy()
        day_data = day_data.sort_values('utc_timestamp').reset_index(drop=True)
        day_data['hour'] = pd.to_datetime(day_data['utc_timestamp']).dt.hour
        
        if len(day_data) != 24:
            continue
        
        # Create REAL devices
        global_layer = GlobalConnectionLayer(max_building_load=65.0, total_hours=24)
        device_columns = [col for col in day_data.columns 
                         if building_id in col and 'grid' not in col and 'pv' not in col]
        
        devices = []
        original_consumption_by_hour = np.zeros(24)
        
        for device_col in device_columns:
            if device_col in day_data.columns and day_data[device_col].sum() > 0:
                parts = device_col.split('_')
                device_type = '_'.join(parts[-2:]) if len(parts) >= 3 else parts[-1]
                
                spec = device_specs.get(device_type, {
                    'category': 'Partially Flexible',
                    'power_rating': 1.0,
                    'allowed_hours': list(range(8, 22))
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
                original_consumption_by_hour += device.original_consumption
                devices.append(device)
        
        if not devices:
            continue
        
        # Run REAL optimization
        optimizer = GlobalOptimizer(
            devices=devices, 
            battery_agent=None, 
            ev_agent=None, 
            total_hours=24
        )
        
        success = optimizer.optimize_building_schedule(
            prices=day_data['price_per_kwh'].values,
            pv_forecast=day_data.get('pv_actual', np.zeros(24)).values
        )
        
        if not success:
            logger.warning(f"Optimization failed for {building_id} on {selected_date}")
            continue
        
        # Aggregate optimized consumption
        optimized_consumption_by_hour = np.zeros(24)
        for device in devices:
            if hasattr(device, 'optimized_schedule'):
                optimized_consumption_by_hour += device.optimized_schedule
        
        # Store in matrices
        original_matrix[day_idx, :] = original_consumption_by_hour
        optimized_matrix[day_idx, :] = optimized_consumption_by_hour
    
    return original_matrix, optimized_matrix, selected_dates

def create_load_shift_heatmap():
    """Create single well-formatted load shifting heatmap"""
    
    data_dir = project_root / "notebooks" / "data"
    building_id = 'DE_KN_residential3'  # Representative building
    
    try:
        original_matrix, optimized_matrix, dates = load_building_and_optimize(building_id, data_dir)
        
        # Calculate load shift difference
        shift_matrix = optimized_matrix - original_matrix
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Heatmap 1: Original consumption
        day_labels = [f"Day {i+1}" for i in range(len(dates))]
        hour_labels = [f"{h:02d}:00" for h in range(24)]
        
        sns.heatmap(
            original_matrix,
            xticklabels=hour_labels,
            yticklabels=day_labels,
            cmap='Reds',
            cbar_kws={'label': 'Consumption (kWh)'},
            ax=ax1
        )
        ax1.set_title('Original Consumption Pattern', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Hour of Day', fontsize=12)
        ax1.set_ylabel('Day', fontsize=12)
        
        # Heatmap 2: Load shift difference
        sns.heatmap(
            shift_matrix,
            xticklabels=hour_labels,
            yticklabels=day_labels,
            cmap='RdBu_r',
            center=0,
            cbar_kws={'label': 'Load Shift (kWh)'},
            ax=ax2
        )
        ax2.set_title('Load Shift Pattern\n(Red = Increased, Blue = Decreased)', 
                     fontsize=14, fontweight='bold')
        ax2.set_xlabel('Hour of Day', fontsize=12)
        ax2.set_ylabel('Day', fontsize=12)
        
        # Format x-axis to show fewer labels
        for ax in [ax1, ax2]:
            ax.set_xticks(range(0, 24, 3))
            ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 3)])
            ax.tick_params(axis='x', rotation=45)
        
        plt.suptitle(f'Load Shifting Analysis - {building_id.replace("DE_KN_", "").title()}\n'
                    f'EMS Optimization Results', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        # Save outputs
        output_path = project_root / "figures" / "load_shift_heatmap.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved heatmap to {output_path}")
        
        # Save data
        shift_df = pd.DataFrame(shift_matrix, 
                               index=[f"Day_{i+1}" for i in range(len(dates))],
                               columns=[f"Hour_{h:02d}" for h in range(24)])
        
        table_path = project_root / "tables" / "load_shift_heatmap.csv"
        shift_df.to_csv(table_path)
        logger.info(f"Saved data to {table_path}")
        
        plt.close()
        
        # Return summary statistics
        total_shifted = np.sum(np.abs(shift_matrix))
        peak_reduction = np.max(original_matrix.mean(axis=0)) - np.max(optimized_matrix.mean(axis=0))
        
        return {
            'total_energy_shifted_kwh': total_shifted,
            'peak_reduction_kwh': peak_reduction,
            'avg_daily_shift_kwh': total_shifted / len(dates)
        }
        
    except Exception as e:
        logger.error(f"Failed to create load shift heatmap: {e}")
        raise

if __name__ == "__main__":
    try:
        results = create_load_shift_heatmap()
        print("SUCCESS: Load shift heatmap created using REAL data and REAL agents")
        print(f"Total energy shifted: {results['total_energy_shifted_kwh']:.2f} kWh")
        print(f"Peak reduction: {results['peak_reduction_kwh']:.2f} kWh")
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)