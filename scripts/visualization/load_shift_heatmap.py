#!/usr/bin/env python3
"""
Load Shifting Heatmap - Single Graph Script
Creates load profile comparison (Before/After Optimization) heatmap.
Uses REAL data from parquet files with load shifting simulation.
ONE HEATMAP ONLY - Well structured and formatted.
"""

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# JADS Colors
JADS_COLORS = {
    "brand_orange": "#F5854F",
    "brand_red": "#E75C4B", 
    "brand_grey": "#6D6E71",
    "brand_gradient_blue": "#273E9E"
}

def load_building_and_simulate_optimization(building_id, data_dir):
    """Load REAL building data and simulate optimization for multiple days"""
    
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
    
    # Use 5 consecutive complete days from the middle
    start_idx = len(complete_days) // 3
    selected_days = complete_days[start_idx:start_idx+5]
    
    # Get device columns
    device_columns = [col for col in df.columns 
                     if building_id in col 
                     and not any(term in col.lower() for term in ['grid', 'pv'])
                     and df[col].sum() > 0]
    
    if not device_columns:
        raise ValueError(f"No device columns found for {building_id}")
    
    # Process selected days
    original_consumption = np.zeros((5, 24))
    optimized_consumption = np.zeros((5, 24))
    
    for day_idx, selected_date in enumerate(selected_days):
        day_data = df[df['date'] == selected_date].copy()
        day_data = day_data.sort_values('utc_timestamp').reset_index(drop=True)
        
        if len(day_data) != 24:
            continue
        
        # Calculate total consumption for this day
        day_consumption = np.zeros(24)
        day_optimized = np.zeros(24)
        
        prices = day_data['price_per_kwh'].values[:24]
        
        for device_col in device_columns:
            device_consumption = day_data[device_col].values[:24]
            day_consumption += device_consumption
            
            # Simulate load shifting optimization
            optimized_device = device_consumption.copy()
            
            # Determine flexibility based on device type
            device_type = device_col.split('_')[-1].lower()
            if 'pump' in device_type or 'washing' in device_type:
                flexibility = 0.4  # High flexibility
            elif 'dishwasher' in device_type:
                flexibility = 0.3  # Medium flexibility
            else:
                flexibility = 0.1  # Low flexibility
            
            # Only shift if there's meaningful price variation
            if np.std(prices) > 0.01:
                total_consumption = np.sum(device_consumption)
                shift_amount = total_consumption * flexibility
                
                # Find expensive and cheap hours
                price_order = np.argsort(prices)
                expensive_hours = [h for h in price_order[-6:] if optimized_device[h] > 0]
                cheap_hours = [h for h in price_order[:6] if device_consumption[h] > 0]
                
                # Shift consumption from expensive to cheap hours
                remaining_shift = shift_amount
                for hour in expensive_hours:
                    if remaining_shift <= 0:
                        break
                    reduction = min(optimized_device[hour] * 0.6, remaining_shift)
                    optimized_device[hour] -= reduction
                    remaining_shift -= reduction
                
                # Distribute shifted load to cheap hours
                if cheap_hours and remaining_shift < shift_amount:
                    shifted_amount = shift_amount - remaining_shift
                    add_per_hour = shifted_amount / len(cheap_hours)
                    for hour in cheap_hours:
                        optimized_device[hour] += add_per_hour
            
            day_optimized += optimized_device
        
        original_consumption[day_idx] = day_consumption
        optimized_consumption[day_idx] = day_optimized
    
    # Calculate load shift (difference)
    load_shift = optimized_consumption - original_consumption
    
    return original_consumption, optimized_consumption, load_shift, selected_days

def create_load_shift_heatmap():
    """Create load shift heatmap using REAL data"""
    
    try:
        # Setup paths
        script_dir = Path(__file__).parent
        data_dir = script_dir / '..' / '..' / 'notebooks' / 'data'
        figures_dir = script_dir / '..' / '..' / 'figures'
        figures_dir.mkdir(exist_ok=True)
        
        # Get all building files
        building_files = list(data_dir.glob("DE_KN_*_processed_data.parquet"))
        
        # Try buildings until we find one that works
        for building_file in building_files:
            building_id = building_file.stem.replace('_processed_data', '')
            
            try:
                original, optimized, load_shift, selected_days = load_building_and_simulate_optimization(
                    building_id, data_dir
                )
                logger.info(f"✓ Using data from {building_id}")
                break
            except Exception as e:
                logger.debug(f"Skipping {building_id}: {e}")
                continue
        else:
            raise ValueError("No suitable building data found")
        
        # Create the heatmap
        sns.set_theme(style="white")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Original consumption heatmap
        im1 = ax1.imshow(original, cmap='Oranges', aspect='auto', interpolation='nearest')
        ax1.set_title('Original Consumption\n(kW)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Hour of Day')
        ax1.set_ylabel('Day')
        ax1.set_xticks(range(0, 24, 4))
        ax1.set_xticklabels(range(0, 24, 4))
        ax1.set_yticks(range(5))
        ax1.set_yticklabels([f'Day {i+1}' for i in range(5)])
        
        # Add colorbar for original consumption
        cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
        cbar1.set_label('Power (kW)', rotation=270, labelpad=15)
        
        # Load shift heatmap (difference)
        max_shift = np.max(np.abs(load_shift))
        im2 = ax2.imshow(load_shift, cmap='RdBu_r', aspect='auto', interpolation='nearest',
                        vmin=-max_shift, vmax=max_shift)
        ax2.set_title('Load Shift Difference\n(Red=Increase, Blue=Decrease)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Hour of Day')
        ax2.set_ylabel('Day')
        ax2.set_xticks(range(0, 24, 4))
        ax2.set_xticklabels(range(0, 24, 4))
        ax2.set_yticks(range(5))
        ax2.set_yticklabels([f'Day {i+1}' for i in range(5)])
        
        # Add colorbar for load shift
        cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
        cbar2.set_label('Load Change (kW)', rotation=270, labelpad=15)
        
        # Overall title
        fig.suptitle(f'Load Profile Comparison - {building_id.replace("DE_KN_", "")}', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save the figure
        output_path = figures_dir / 'load_shift_heatmap.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Calculate summary statistics
        total_original = np.sum(original)
        total_optimized = np.sum(optimized)
        total_shifted = np.sum(np.abs(load_shift))
        
        logger.info(f"✓ Load shift heatmap saved to {output_path}")
        logger.info(f"✓ Used REAL data from {building_id}")
        logger.info(f"✓ Total load shifted: {total_shifted:.2f} kWh over 5 days")
        logger.info(f"✓ Original total: {total_original:.2f} kWh, Optimized: {total_optimized:.2f} kWh")
        
        return str(output_path)
        
    except Exception as e:
        logger.error(f"✗ Error creating load shift heatmap: {e}")
        raise

if __name__ == "__main__":
    try:
        logger.info("Creating load shift heatmap...")
        output_file = create_load_shift_heatmap()
        logger.info(f"Success! Heatmap saved to: {output_file}")
    except Exception as e:
        logger.error(f"Failed to create heatmap: {e}")
        exit(1)