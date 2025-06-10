#!/usr/bin/env python3
"""
PV vs Demand Overlap Analysis - Single Graph Script
Creates PV generation vs demand overlap analysis chart.
Uses REAL data from parquet files and REAL PVAgent data.
ONE GRAPH ONLY - Well structured and formatted.
"""
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import sys
import os

# JADS Colors
JADS_COLORS = {
    "brand_orange": "#F5854F",
    "brand_red": "#E75C4B", 
    "brand_grey": "#6D6E71",
    "brand_gradient_blue": "#273E9E"
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def find_pv_building(data_dir):
    """Find building with actual PV data"""
    data_path = Path(data_dir)
    pv_buildings = []
    
    for file_path in data_path.glob("DE_KN_*_processed_data.parquet"):
        try:
            df_sample = pd.read_parquet(file_path, columns=None)
            pv_columns = [col for col in df_sample.columns if 'pv' in col.lower() or 'solar' in col.lower()]
            
            if pv_columns:
                for pv_col in pv_columns:
                    if df_sample[pv_col].sum() > 0:
                        building_id = file_path.stem.replace('_processed_data', '')
                        pv_buildings.append((building_id, pv_col))
                        logger.info(f"Found PV data in {building_id}: {pv_col}")
                        break
        except Exception as e:
            logger.warning(f"Error checking {file_path}: {e}")
            continue
    
    if not pv_buildings:
        raise ValueError("No buildings with actual PV data found")
    
    return pv_buildings[0]

def analyze_pv_demand_overlap(building_id, pv_column, data_dir):
    """Analyze PV generation vs demand overlap using REAL data"""
    file_path = Path(data_dir) / f"{building_id}_processed_data.parquet"
    
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    logger.info(f"Loading REAL data from {file_path}")
    df = pd.read_parquet(file_path)
    
    if pv_column not in df.columns:
        raise ValueError(f"PV column {pv_column} not found in data")
    
    # Get actual device demand columns (exclude PV, weather, prices, metadata)
    exclude_terms = ['pv', 'solar', 'generation', 'temperature', 'radiation', 'price', 
                    'total_consumption', 'net_energy', 'cost', 'year', 'flexibility', 
                    'power_rating', 'grid_export', 'grid_import']
    
    demand_columns = [col for col in df.columns 
                     if col not in [pv_column] 
                     and not any(term in col.lower() for term in exclude_terms)
                     and df[col].dtype in ['float64', 'int64']
                     and df[col].sum() > 0]  # Only columns with actual consumption
    
    if not demand_columns:
        raise ValueError("No demand columns found in data")
    
    logger.info(f"Using demand columns: {demand_columns}")
    
    # Calculate total demand
    df['total_demand'] = df[demand_columns].sum(axis=1)
    
    # Remove negative demand values (if any)
    df = df[df['total_demand'] >= 0].copy()
    
    # Ensure datetime column
    if df.index.name == 'utc_timestamp':
        df = df.reset_index()
        df.rename(columns={'utc_timestamp': 'datetime'}, inplace=True)
    elif 'datetime' not in df.columns:
        df.reset_index(inplace=True)
    
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['hour'] = df['datetime'].dt.hour
    
    # Filter for days with good PV generation
    daily_pv = df.groupby(df['datetime'].dt.date)[pv_column].sum()
    good_pv_days = daily_pv[daily_pv > daily_pv.quantile(0.7)].index
    
    if len(good_pv_days) == 0:
        raise ValueError("No days with sufficient PV generation found")
    
    df_filtered = df[df['datetime'].dt.date.isin(good_pv_days)].copy()
    logger.info(f"Analyzing {len(good_pv_days)} days with good PV generation")
    
    # Calculate hourly averages
    hourly_stats = df_filtered.groupby('hour').agg({
        pv_column: ['mean', 'std'],
        'total_demand': ['mean', 'std']
    }).round(2)
    
    # Flatten column names
    hourly_stats.columns = ['_'.join(col).strip() for col in hourly_stats.columns]
    
    # Calculate overlap metrics
    hourly_stats['pv_mean'] = hourly_stats[f'{pv_column}_mean']
    hourly_stats['demand_mean'] = hourly_stats['total_demand_mean']
    
    # Self-consumption potential (how much PV can be used directly)
    hourly_stats['self_consumption'] = np.minimum(hourly_stats['pv_mean'], hourly_stats['demand_mean'])
    hourly_stats['excess_pv'] = np.maximum(0, hourly_stats['pv_mean'] - hourly_stats['demand_mean'])
    hourly_stats['unmet_demand'] = np.maximum(0, hourly_stats['demand_mean'] - hourly_stats['pv_mean'])
    
    # Calculate overlap percentage
    hourly_stats['overlap_pct'] = (hourly_stats['self_consumption'] / 
                                  np.maximum(hourly_stats['pv_mean'], 0.001)) * 100
    
    return hourly_stats, building_id

def create_pv_demand_overlap_graph():
    """Create single well-formatted PV vs demand overlap graph"""
    try:
        # Setup paths
        script_dir = Path(__file__).parent
        data_dir = script_dir / '..' / '..' / 'notebooks' / 'data'
        figures_dir = script_dir / '..' / '..' / 'figures'
        figures_dir.mkdir(exist_ok=True)
        
        # Find building with PV data
        building_id, pv_column = find_pv_building(data_dir)
        
        # Analyze PV vs demand overlap using REAL data
        hourly_stats, building_id = analyze_pv_demand_overlap(building_id, pv_column, data_dir)
        
        # Create the graph
        sns.set_theme(style="whitegrid")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        
        hours = hourly_stats.index
        
        # Top subplot: PV Generation vs Demand
        ax1.fill_between(hours, 0, hourly_stats['pv_mean'], 
                        alpha=0.7, color=JADS_COLORS['brand_orange'], 
                        label='PV Generation')
        ax1.fill_between(hours, 0, hourly_stats['demand_mean'], 
                        alpha=0.5, color=JADS_COLORS['brand_grey'], 
                        label='Electricity Demand')
        
        # Highlight overlap area
        overlap_area = np.minimum(hourly_stats['pv_mean'], hourly_stats['demand_mean'])
        ax1.fill_between(hours, 0, overlap_area, 
                        alpha=0.8, color=JADS_COLORS['brand_gradient_blue'], 
                        label='Direct Use Potential')
        
        ax1.set_ylabel('Power (kW)')
        ax1.set_title(f'PV Generation vs Demand Overlap - {building_id}', 
                     fontsize=12, fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Bottom subplot: Overlap Percentage
        ax2.bar(hours, hourly_stats['overlap_pct'], 
               color=JADS_COLORS['brand_red'], alpha=0.7, width=0.8)
        ax2.set_xlabel('Hour of Day')
        ax2.set_ylabel('Self-Consumption\nPotential (%)')
        ax2.set_title('PV Self-Consumption Potential by Hour', 
                     fontsize=11, fontweight='bold')
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3)
        
        # Add summary statistics as text
        total_pv = hourly_stats['pv_mean'].sum()
        total_demand = hourly_stats['demand_mean'].sum()
        total_overlap = hourly_stats['self_consumption'].sum()
        overall_overlap_pct = (total_overlap / total_pv) * 100 if total_pv > 0 else 0
        
        ax2.text(0.02, 0.98, f'Daily Average Self-Consumption: {overall_overlap_pct:.1f}%', 
                transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save the figure
        output_path = figures_dir / 'pv_demand_overlap.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ PV vs demand overlap graph saved to {output_path}")
        logger.info(f"✓ Used REAL data from {building_id}")
        logger.info(f"✓ Overall self-consumption potential: {overall_overlap_pct:.1f}%")
        
        return str(output_path)
        
    except Exception as e:
        logger.error(f"✗ Error creating PV vs demand overlap graph: {e}")
        raise

if __name__ == "__main__":
    try:
        logger.info("Creating PV vs demand overlap analysis graph...")
        output_file = create_pv_demand_overlap_graph()
        logger.info(f"Success! Graph saved to: {output_file}")
    except Exception as e:
        logger.error(f"Failed to create graph: {e}")
        sys.exit(1)