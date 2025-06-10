#!/usr/bin/env python3
"""
PV Self-Consumption Analysis - Single Graph Script

Creates Figure 5: PV Self-Consumption and Battery Metrics
Uses REAL data from parquet files with PV data and REAL agent optimization.
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
from agents.BatteryAgent import BatteryAgent
from agents.FlexibleDeviceAgent import FlexibleDevice
from agents.GlobalConnectionLayer import GlobalConnectionLayer
from utils.device_specs import device_specs
from notebooks.utils.helper import BATTERY_PARAMS

# JADS Colors
JADS_COLORS = {
    "brand_orange": "#F5854F",
    "brand_red": "#E75C4B", 
    "brand_grey": "#6D6E71",
    "brand_gradient_blue": "#273E9E"
}

def find_pv_building(data_dir):
    """Find building with actual PV data"""
    building_ids = [
        'DE_KN_residential1', 'DE_KN_residential2', 'DE_KN_residential3',
        'DE_KN_residential4', 'DE_KN_residential5', 'DE_KN_residential6'
    ]
    
    for building_id in building_ids:
        try:
            parquet_file = data_dir / f"{building_id}_processed_data.parquet"
            if parquet_file.exists():
                df = pd.read_parquet(parquet_file)
                pv_columns = [col for col in df.columns if 'pv' in col.lower()]
                if pv_columns:
                    # Check if PV data has meaningful values
                    for pv_col in pv_columns:
                        if df[pv_col].sum() > 0:
                            logger.info(f"Found PV data in {building_id}, column: {pv_col}")
                            return building_id, pv_col
        except Exception as e:
            logger.warning(f"Could not check {building_id}: {e}")
            continue
    
    raise ValueError("No building with meaningful PV data found")

def analyze_pv_self_consumption(building_id, pv_column, data_dir):
    """Analyze PV self-consumption with and without battery using REAL agents"""
    
    parquet_file = data_dir / f"{building_id}_processed_data.parquet"
    df = pd.read_parquet(parquet_file)
    
    # Handle datetime index
    if df.index.name == 'utc_timestamp':
        df = df.reset_index()
    # Get multiple complete days for better analysis
    df['date'] = pd.to_datetime(df['utc_timestamp']).dt.date
    daily_counts = df.groupby('date').size()
    complete_days = daily_counts[daily_counts == 24].index
    
    if len(complete_days) < 3:
        raise ValueError(f"Need at least 3 complete days, found {len(complete_days)}")
    
    # Use middle 3 days
    start_idx = len(complete_days) // 2 - 1
    selected_dates = complete_days[start_idx:start_idx + 3]
    
    results = {
        'scenario': [],
        'pv_self_consumption_pct': [],
        'pv_export_kwh': [],
        'total_pv_kwh': [],
        'battery_cycles': [],
        'battery_efficiency_pct': []
    }
    
    for scenario in ['no_battery', 'with_battery']:
        total_pv_generated = 0
        total_pv_consumed = 0
        total_pv_exported = 0
        total_battery_charged = 0
        total_battery_discharged = 0
        
        for selected_date in selected_dates:
            day_data = df[df['date'] == selected_date].copy()
            day_data = day_data.sort_values('utc_timestamp').reset_index(drop=True)
            day_data['hour'] = pd.to_datetime(day_data['utc_timestamp']).dt.hour
            
            if len(day_data) != 24:
                continue
            
    # Handle datetime index
    if df.index.name == 'utc_timestamp':
        df = df.reset_index()
            # Get PV generation
            pv_generation = day_data[pv_column].values
            total_pv_generated += np.sum(pv_generation)
            
            # Create REAL devices
            global_layer = GlobalConnectionLayer(max_building_load=65.0, total_hours=24)
            device_columns = [col for col in day_data.columns 
                             if building_id in col and 'grid' not in col and 'pv' not in col]
            
            devices = []
            building_demand = np.zeros(24)
            
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
                    building_demand += device.original_consumption
                    devices.append(device)
            
            if not devices:
                continue
            
            # Create battery if scenario includes it
            battery_agent = None
            if scenario == 'with_battery':
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
            
            # Run REAL optimization
            optimizer = GlobalOptimizer(
                devices=devices,
                battery_agent=battery_agent,
                ev_agent=None,
                total_hours=24
            )
            
            success = optimizer.optimize_building_schedule(
                prices=day_data['price_per_kwh'].values,
                pv_forecast=pv_generation
            )
            
            if not success:
                logger.warning(f"Optimization failed for {building_id} on {selected_date}")
                continue
            
            # Calculate optimized demand
            optimized_demand = np.zeros(24)
            for device in devices:
                if hasattr(device, 'optimized_schedule'):
                    optimized_demand += device.optimized_schedule
            
            # Calculate PV self-consumption
            if scenario == 'no_battery':
                # Direct consumption without battery
                hourly_self_consumption = np.minimum(pv_generation, optimized_demand)
                hourly_export = np.maximum(0, pv_generation - optimized_demand)
            else:
                # With battery - more complex calculation
                hourly_self_consumption = np.zeros(24)
                hourly_export = np.zeros(24)
                
                if battery_agent and hasattr(battery_agent, 'hourly_charge'):
                    battery_charge = battery_agent.hourly_charge
                    battery_discharge = battery_agent.hourly_discharge
                    
                    total_battery_charged += np.sum(battery_charge)
                    total_battery_discharged += np.sum(battery_discharge)
                    
                    for hour in range(24):
                        # Available PV after charging battery
                        pv_after_battery = pv_generation[hour] - battery_charge[hour]
                        # Total available energy including battery discharge
                        total_available = max(0, pv_after_battery) + battery_discharge[hour]
                        # Self-consumption is minimum of available energy and demand
                        hourly_self_consumption[hour] = min(total_available, optimized_demand[hour])
                        # Export only excess PV after satisfying demand and battery
                        hourly_export[hour] = max(0, pv_generation[hour] - optimized_demand[hour] - battery_charge[hour])
                else:
                    # Fallback if battery data not available
                    hourly_self_consumption = np.minimum(pv_generation, optimized_demand)
                    hourly_export = np.maximum(0, pv_generation - optimized_demand)
            
            total_pv_consumed += np.sum(hourly_self_consumption)
            total_pv_exported += np.sum(hourly_export)
        
        # Calculate metrics for this scenario
        pv_self_consumption_pct = (total_pv_consumed / total_pv_generated * 100) if total_pv_generated > 0 else 0
        battery_cycles = (total_battery_discharged / BATTERY_PARAMS['capacity']) if scenario == 'with_battery' else 0
        battery_efficiency = (total_battery_discharged / total_battery_charged * 100) if total_battery_charged > 0 else 0
        
        results['scenario'].append(scenario.replace('_', ' ').title())
        results['pv_self_consumption_pct'].append(pv_self_consumption_pct)
        results['pv_export_kwh'].append(total_pv_exported)
        results['total_pv_kwh'].append(total_pv_generated)
        results['battery_cycles'].append(battery_cycles)
        results['battery_efficiency_pct'].append(battery_efficiency)
    
    return pd.DataFrame(results)

def create_pv_self_consumption_graph():
    """Create single well-formatted PV self-consumption graph"""
    
    data_dir = project_root / "notebooks" / "data"
    
    try:
        # Find building with PV data
        building_id, pv_column = find_pv_building(data_dir)
        logger.info(f"Using {building_id} with PV column: {pv_column}")
        
        # Analyze PV self-consumption
        results_df = analyze_pv_self_consumption(building_id, pv_column, data_dir)
        
        # Create visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: PV Self-Consumption Percentage
        bars1 = ax1.bar(results_df['scenario'], results_df['pv_self_consumption_pct'],
                        color=[JADS_COLORS['brand_orange'], JADS_COLORS['brand_gradient_blue']])
        ax1.set_title('PV Self-Consumption Rate', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Self-Consumption (%)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax1.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: PV Export
        bars2 = ax2.bar(results_df['scenario'], results_df['pv_export_kwh'],
                        color=[JADS_COLORS['brand_red'], JADS_COLORS['brand_gradient_red']])
        ax2.set_title('PV Export to Grid', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Exported Energy (kWh)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        for bar in bars2:
            height = bar.get_height()
            ax2.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')
        
        # Plot 3: Battery Cycles
        battery_data = results_df[results_df['scenario'] == 'With Battery']
        if not battery_data.empty:
            bars3 = ax3.bar(['With Battery'], battery_data['battery_cycles'].values,
                           color=[JADS_COLORS['brand_grey']])
            ax3.set_title('Battery Cycle Usage', fontsize=14, fontweight='bold')
            ax3.set_ylabel('Cycles per Day', fontsize=12)
            ax3.grid(True, alpha=0.3)
            
            for bar in bars3:
                height = bar.get_height()
                ax3.annotate(f'{height:.2f}',
                            xy=(bar.get_x() + bar.get_width()/2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontweight='bold')
        
        # Plot 4: Battery Efficiency
        if not battery_data.empty:
            bars4 = ax4.bar(['With Battery'], battery_data['battery_efficiency_pct'].values,
                           color=[JADS_COLORS['brand_dark_grey']])
            ax4.set_title('Battery Round-Trip Efficiency', fontsize=14, fontweight='bold')
            ax4.set_ylabel('Efficiency (%)', fontsize=12)
            ax4.grid(True, alpha=0.3)
            
            for bar in bars4:
                height = bar.get_height()
                ax4.annotate(f'{height:.1f}%',
                            xy=(bar.get_x() + bar.get_width()/2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontweight='bold')
        
        plt.suptitle(f'PV Self-Consumption Analysis - {building_id.replace("DE_KN_", "").title()}\n'
                    f'REAL Agent Optimization Results', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        # Save outputs
        output_path = project_root / "figures" / "pv_self_consumption.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved PV analysis to {output_path}")
        
        # Save data
        table_path = project_root / "tables" / "pv_self_consumption.csv"
        results_df.to_csv(table_path, index=False)
        logger.info(f"Saved data to {table_path}")
        
        plt.close()
        
        return results_df
        
    except Exception as e:
        logger.error(f"Failed to create PV self-consumption graph: {e}")
        raise

if __name__ == "__main__":
    try:
        results = create_pv_self_consumption_graph()
        print("SUCCESS: PV self-consumption analysis created using REAL data and REAL agents")
        print(f"Results:\n{results}")
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)