#!/usr/bin/env python3
"""
PV Self-Consumption and Battery Metrics - Single Graph Script
Creates Figure 5: PV Self-Consumption and Battery Metrics
Uses REAL data from parquet files with battery simulation.
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

def find_pv_buildings(data_dir):
    """Find buildings with PV data"""
    data_path = Path(data_dir)
    pv_buildings = []
    
    for file_path in data_path.glob("DE_KN_*_processed_data.parquet"):
        try:
            df = pd.read_parquet(file_path)
            pv_columns = [col for col in df.columns if 'pv' in col.lower() and df[col].sum() > 0]
            
            if pv_columns:
                building_id = file_path.stem.replace('_processed_data', '')
                pv_buildings.append((building_id, pv_columns[0]))
                logger.info(f"Found PV in {building_id}: {pv_columns[0]}")
        except Exception as e:
            logger.debug(f"Error checking {file_path}: {e}")
    
    return pv_buildings

def analyze_pv_self_consumption_simple(building_id, pv_column, data_dir):
    """Analyze PV self-consumption using REAL data with battery simulation"""
    
    file_path = Path(data_dir) / f"{building_id}_processed_data.parquet"
    df = pd.read_parquet(file_path)
    
    # Handle datetime index
    if df.index.name == 'utc_timestamp':
        df = df.reset_index()
    
    # Get complete days
    df['date'] = pd.to_datetime(df['utc_timestamp']).dt.date
    daily_counts = df.groupby('date').size()
    complete_days = daily_counts[daily_counts == 24].index
    
    if len(complete_days) < 3:
        raise ValueError(f"Need at least 3 complete days, found {len(complete_days)}")
    
    # Use middle days for analysis
    selected_days = complete_days[len(complete_days)//3:len(complete_days)//3+5]
    
    # Get device columns (demand)
    device_columns = [col for col in df.columns 
                     if building_id in col 
                     and not any(term in col.lower() for term in ['pv', 'grid'])
                     and df[col].sum() > 0]
    
    metrics_without_battery = []
    metrics_with_battery = []
    
    for selected_date in selected_days:
        day_data = df[df['date'] == selected_date].copy()
        if len(day_data) != 24:
            continue
            
        day_data = day_data.sort_values('utc_timestamp').reset_index(drop=True)
        
        # Get PV generation and total demand
        pv_generation = day_data[pv_column].values[:24]
        total_demand = day_data[device_columns].sum(axis=1).values[:24]
        
        # Skip days with no PV or demand
        if np.sum(pv_generation) == 0 or np.sum(total_demand) == 0:
            continue
        
        # === WITHOUT BATTERY ===
        # Direct self-consumption (min of PV and demand at each hour)
        direct_self_consumption = np.minimum(pv_generation, total_demand)
        excess_pv = np.maximum(0, pv_generation - total_demand)
        unmet_demand = np.maximum(0, total_demand - pv_generation)
        
        self_consumption_rate = np.sum(direct_self_consumption) / np.sum(pv_generation) * 100
        pv_export = np.sum(excess_pv)
        grid_import = np.sum(unmet_demand)
        
        metrics_without_battery.append({
            'self_consumption_rate': self_consumption_rate,
            'pv_export': pv_export,
            'grid_import': grid_import,
            'total_pv': np.sum(pv_generation),
            'total_demand': np.sum(total_demand)
        })
        
        # === WITH BATTERY SIMULATION ===
        battery_capacity = 10.0  # kWh
        battery_efficiency = 0.9
        battery_soc = 5.0  # Start at 50%
        battery_cycles = 0
        total_battery_throughput = 0
        
        improved_self_consumption = 0
        improved_export = 0
        improved_import = 0
        
        for hour in range(24):
            pv_hour = pv_generation[hour]
            demand_hour = total_demand[hour]
            
            # Direct use first
            direct_use = min(pv_hour, demand_hour)
            improved_self_consumption += direct_use
            
            remaining_pv = pv_hour - direct_use
            remaining_demand = demand_hour - direct_use
            
            # Use battery for remaining demand/generation
            if remaining_pv > 0:  # Excess PV - charge battery
                charge_amount = min(remaining_pv, battery_capacity - battery_soc, 3.0)  # 3kW max charge
                battery_soc += charge_amount * battery_efficiency
                total_battery_throughput += charge_amount
                improved_export += remaining_pv - charge_amount
                
            elif remaining_demand > 0:  # Unmet demand - discharge battery
                discharge_amount = min(remaining_demand, battery_soc, 3.0)  # 3kW max discharge
                battery_soc -= discharge_amount
                total_battery_throughput += discharge_amount
                improved_import += remaining_demand - discharge_amount
        
        # Calculate battery cycles (rough estimate)
        battery_cycles = total_battery_throughput / (battery_capacity * 2)
        battery_efficiency_actual = (total_battery_throughput * battery_efficiency) / max(total_battery_throughput, 0.001)
        
        improved_self_consumption_rate = improved_self_consumption / np.sum(pv_generation) * 100
        
        metrics_with_battery.append({
            'self_consumption_rate': improved_self_consumption_rate,
            'pv_export': improved_export,
            'grid_import': improved_import,
            'battery_cycles': battery_cycles,
            'battery_efficiency': battery_efficiency_actual,
            'total_pv': np.sum(pv_generation),
            'total_demand': np.sum(total_demand)
        })
    
    # Average across days
    if not metrics_without_battery or not metrics_with_battery:
        raise ValueError("No valid days for analysis")
    
    avg_without = pd.DataFrame(metrics_without_battery).mean().to_dict()
    avg_with = pd.DataFrame(metrics_with_battery).mean().to_dict()
    
    return avg_without, avg_with

def create_pv_self_consumption_graph():
    """Create PV self-consumption and battery metrics graph"""
    
    try:
        # Setup paths
        script_dir = Path(__file__).parent
        data_dir = script_dir / '..' / '..' / 'notebooks' / 'data'
        figures_dir = script_dir / '..' / '..' / 'figures'
        figures_dir.mkdir(exist_ok=True)
        
        # Find buildings with PV
        pv_buildings = find_pv_buildings(data_dir)
        
        if not pv_buildings:
            raise ValueError("No buildings with PV data found")
        
        all_results_without = []
        all_results_with = []
        
        # Analyze each PV building
        for building_id, pv_column in pv_buildings[:3]:  # Use first 3 PV buildings
            try:
                without_battery, with_battery = analyze_pv_self_consumption_simple(
                    building_id, pv_column, data_dir
                )
                all_results_without.append(without_battery)
                all_results_with.append(with_battery)
                logger.info(f"✓ Analyzed {building_id}")
            except Exception as e:
                logger.warning(f"Skipping {building_id}: {e}")
                continue
        
        if not all_results_without:
            raise ValueError("No valid PV analysis results")
        
        # Average across buildings
        avg_without = pd.DataFrame(all_results_without).mean()
        avg_with = pd.DataFrame(all_results_with).mean()
        
        # Create 2x2 subplot
        sns.set_theme(style="whitegrid")
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))
        
        # Plot 1: Self-Consumption Rate
        scenarios = ['Without Battery', 'With Battery']
        self_consumption_rates = [avg_without['self_consumption_rate'], avg_with['self_consumption_rate']]
        
        bars1 = ax1.bar(scenarios, self_consumption_rates, 
                       color=[JADS_COLORS['brand_grey'], JADS_COLORS['brand_orange']], alpha=0.8)
        ax1.set_ylabel('Self-Consumption Rate (%)')
        ax1.set_title('PV Self-Consumption Rate', fontweight='bold')
        ax1.set_ylim(0, 100)
        
        for bar, value in zip(bars1, self_consumption_rates):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: PV Export
        pv_exports = [avg_without['pv_export'], avg_with['pv_export']]
        bars2 = ax2.bar(scenarios, pv_exports,
                       color=[JADS_COLORS['brand_grey'], JADS_COLORS['brand_orange']], alpha=0.8)
        ax2.set_ylabel('PV Export (kWh/day)')
        ax2.set_title('Grid Export Reduction', fontweight='bold')
        
        for bar, value in zip(bars2, pv_exports):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 3: Battery Cycles (only for with-battery scenario)
        battery_cycles = avg_with['battery_cycles']
        ax3.bar(['Battery Cycles'], [battery_cycles], 
               color=JADS_COLORS['brand_gradient_blue'], alpha=0.8)
        ax3.set_ylabel('Cycles per Day')
        ax3.set_title('Battery Utilization', fontweight='bold')
        ax3.text(0, battery_cycles + 0.01, f'{battery_cycles:.2f}',
                ha='center', va='bottom', fontweight='bold')
        
        # Plot 4: Battery Efficiency
        battery_efficiency = avg_with['battery_efficiency'] * 100
        ax4.bar(['Battery Efficiency'], [battery_efficiency],
               color=JADS_COLORS['brand_gradient_blue'], alpha=0.8)
        ax4.set_ylabel('Efficiency (%)')
        ax4.set_title('Battery Round-Trip Efficiency', fontweight='bold')
        ax4.set_ylim(0, 100)
        ax4.text(0, battery_efficiency + 1, f'{battery_efficiency:.1f}%',
                ha='center', va='bottom', fontweight='bold')
        
        plt.suptitle('PV Self-Consumption and Battery Metrics\n(Average Across Buildings with PV)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save the figure
        output_path = figures_dir / 'pv_self_consumption_battery_metrics.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ PV self-consumption graph saved to {output_path}")
        logger.info(f"✓ Used REAL data from {len(all_results_without)} buildings with PV")
        logger.info(f"✓ Self-consumption improved: {avg_without['self_consumption_rate']:.1f}% → {avg_with['self_consumption_rate']:.1f}%")
        
        return str(output_path)
        
    except Exception as e:
        logger.error(f"✗ Error creating PV self-consumption graph: {e}")
        raise

if __name__ == "__main__":
    try:
        logger.info("Creating PV self-consumption and battery metrics graph...")
        output_file = create_pv_self_consumption_graph()
        logger.info(f"Success! Graph saved to: {output_file}")
    except Exception as e:
        logger.error(f"Failed to create graph: {e}")
        exit(1)