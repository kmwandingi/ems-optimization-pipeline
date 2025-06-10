#!/usr/bin/env python3
"""
Cost Savings by Building - Simple Single Graph Script
Creates cost savings comparison using REAL data patterns.
Uses REAL data from parquet files with realistic savings calculations.
ONE GRAPH ONLY - Well structured and formatted.
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

def calculate_savings_simple(building_id, data_dir):
    """Calculate cost savings using REAL data with load shifting simulation"""
    
    parquet_file = data_dir / f"{building_id}_processed_data.parquet"
    if not parquet_file.exists():
        return None
    
    try:
        df = pd.read_parquet(parquet_file)
        
        # Handle datetime index
        if df.index.name == 'utc_timestamp':
            df = df.reset_index()
        
        # Get complete day
        df['date'] = pd.to_datetime(df['utc_timestamp']).dt.date
        daily_counts = df.groupby('date').size()
        complete_days = daily_counts[daily_counts == 24].index
        
        if len(complete_days) == 0:
            return None
        
        # Use middle day
        selected_day = complete_days[len(complete_days)//2]
        day_data = df[df['date'] == selected_day].copy()
        day_data = day_data.sort_values('utc_timestamp').reset_index(drop=True)
        
        # Get device columns
        device_columns = [col for col in day_data.columns 
                         if building_id in col 
                         and not any(term in col.lower() for term in ['grid', 'pv'])
                         and day_data[col].sum() > 0]
        
        if not device_columns:
            return None
        
        # Calculate original and optimized costs
        prices = day_data['price_per_kwh'].values[:24]
        
        original_cost = 0
        optimized_cost = 0
        
        for device_col in device_columns:
            consumption = day_data[device_col].values[:24]
            
            # Original cost
            device_original_cost = np.sum(consumption * prices)
            original_cost += device_original_cost
            
            # Simulate load shifting optimization
            # Shift consumption from expensive to cheap hours
            price_order = np.argsort(prices)
            optimized_consumption = consumption.copy()
            
            # Determine flexibility based on device type
            device_type = device_col.split('_')[-1].lower()
            if 'pump' in device_type or 'washing' in device_type:
                flexibility = 0.4  # High flexibility
            elif 'dishwasher' in device_type:
                flexibility = 0.3  # Medium flexibility
            else:
                flexibility = 0.1  # Low flexibility
            
            total_consumption = np.sum(consumption)
            shift_amount = total_consumption * flexibility
            
            # Remove from most expensive hours with consumption
            expensive_hours = [h for h in price_order[-8:] if optimized_consumption[h] > 0]
            remaining_shift = shift_amount
            
            for hour in expensive_hours:
                if remaining_shift <= 0:
                    break
                reduction = min(optimized_consumption[hour] * 0.5, remaining_shift)
                optimized_consumption[hour] -= reduction
                remaining_shift -= reduction
            
            # Add to cheapest hours with existing consumption (realistic constraint)
            cheap_hours = [h for h in price_order[:8] if consumption[h] > 0]
            if cheap_hours:
                shifted_amount = shift_amount - remaining_shift
                add_per_hour = shifted_amount / len(cheap_hours)
                for hour in cheap_hours:
                    optimized_consumption[hour] += add_per_hour
            
            device_optimized_cost = np.sum(optimized_consumption * prices)
            optimized_cost += device_optimized_cost
        
        # Calculate savings percentage
        if original_cost > 0:
            savings_pct = ((original_cost - optimized_cost) / original_cost) * 100
            euro_savings = original_cost - optimized_cost
        else:
            savings_pct = 0
            euro_savings = 0
        
        return {
            'building_id': building_id,
            'original_cost': original_cost,
            'optimized_cost': optimized_cost,
            'savings_pct': savings_pct,
            'euro_savings': euro_savings,
            'devices_count': len(device_columns)
        }
        
    except Exception as e:
        logger.warning(f"Error processing {building_id}: {e}")
        return None

def create_cost_savings_graph():
    """Create cost savings by building graph using REAL data"""
    
    try:
        # Setup paths
        script_dir = Path(__file__).parent
        data_dir = script_dir / '..' / '..' / 'notebooks' / 'data'
        figures_dir = script_dir / '..' / '..' / 'figures'
        figures_dir.mkdir(exist_ok=True)
        
        # Get all building files
        building_files = list(data_dir.glob("DE_KN_*_processed_data.parquet"))
        
        if not building_files:
            raise FileNotFoundError("No building data files found")
        
        results = []
        
        # Calculate savings for each building
        for building_file in building_files:
            building_id = building_file.stem.replace('_processed_data', '')
            result = calculate_savings_simple(building_id, data_dir)
            
            if result:
                results.append(result)
                logger.info(f"✓ {building_id}: {result['savings_pct']:.1f}% savings")
        
        if not results:
            raise ValueError("No valid building results generated")
        
        # Create DataFrame
        df_results = pd.DataFrame(results)
        
        # Create the graph
        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create bar chart
        bars = ax.bar(range(len(df_results)), df_results['savings_pct'], 
                     color=JADS_COLORS['brand_orange'], alpha=0.8, width=0.6)
        
        # Customize the plot
        ax.set_xlabel('Building', fontsize=12, fontweight='bold')
        ax.set_ylabel('Cost Savings (%)', fontsize=12, fontweight='bold')
        ax.set_title('Cost Savings by Building\n(Load Shifting Optimization)', 
                    fontsize=14, fontweight='bold')
        
        # Set x-axis labels
        building_labels = [bid.replace('DE_KN_', '') for bid in df_results['building_id']]
        ax.set_xticks(range(len(df_results)))
        ax.set_xticklabels(building_labels, rotation=45, ha='right')
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, df_results['savings_pct'])):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Add average line
        avg_savings = df_results['savings_pct'].mean()
        ax.axhline(y=avg_savings, color=JADS_COLORS['brand_red'], 
                  linestyle='--', alpha=0.7, linewidth=2)
        ax.text(len(df_results)-1, avg_savings + 0.5, f'Avg: {avg_savings:.1f}%', 
               ha='right', va='bottom', color=JADS_COLORS['brand_red'], fontweight='bold')
        
        # Set y-axis to start from 0
        ax.set_ylim(0, max(df_results['savings_pct']) * 1.2)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the figure
        output_path = figures_dir / 'cost_savings_by_building.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ Cost savings graph saved to {output_path}")
        logger.info(f"✓ Used REAL data from {len(results)} buildings")
        logger.info(f"✓ Average savings: {avg_savings:.1f}%")
        
        return str(output_path)
        
    except Exception as e:
        logger.error(f"✗ Error creating cost savings graph: {e}")
        raise

if __name__ == "__main__":
    try:
        logger.info("Creating cost savings by building graph...")
        output_file = create_cost_savings_graph()
        logger.info(f"Success! Graph saved to: {output_file}")
    except Exception as e:
        logger.error(f"Failed to create graph: {e}")
        exit(1)