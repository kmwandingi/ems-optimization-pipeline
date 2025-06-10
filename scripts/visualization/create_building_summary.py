#!/usr/bin/env python3
"""
Create building summary from the parquet data files
"""

import pandas as pd
import json
from pathlib import Path

def create_building_summary():
    """Create a comprehensive building summary from the data files"""
    data_dir = Path(__file__).parent.parent.parent / "notebooks" / "data"
    building_summary = {}
    
    # Get all parquet files
    parquet_files = list(data_dir.glob("*_processed_data.parquet"))
    
    for file_path in parquet_files:
        building_id = file_path.stem.replace("_processed_data", "")
        df = pd.read_parquet(file_path)
        
        # Initialize building info
        building_info = {
            'building_type': 'residential' if 'residential' in building_id else 'industrial',
            'flexible_devices': [],
            'pv_system': False,
            'battery_storage': False,
            'ev_charging': False,
            'peak_load': 0.0,
            'avg_load': 0.0,
            'load_factor': 0.0
        }
        
        # Extract devices from column names
        for col in df.columns:
            if building_id in col:
                device_type = col.replace(f"{building_id}_", "")
                
                # Check for PV system
                if device_type == 'pv':
                    building_info['pv_system'] = True
                
                # Check for flexible devices
                elif device_type in ['washing_machine', 'dishwasher', 'freezer', 'heat_pump', 
                                   'refrigerator', 'circulation_pump', 'tumble_dryer']:
                    # Check if device actually has data
                    if df[col].sum() > 0:
                        building_info['flexible_devices'].append(device_type)
                
                # Check for EV charging (might be in heat_pump or separate column)
                elif 'ev' in device_type.lower():
                    building_info['ev_charging'] = True
        
        # Calculate load statistics from total_consumption if available
        if 'total_consumption' in df.columns:
            consumption_data = df['total_consumption'].dropna()
            if len(consumption_data) > 0:
                building_info['peak_load'] = round(consumption_data.max(), 2)
                building_info['avg_load'] = round(consumption_data.mean(), 2)
                if building_info['peak_load'] > 0:
                    building_info['load_factor'] = round(building_info['avg_load'] / building_info['peak_load'], 3)
        
        building_summary[building_id] = building_info
    
    # Save the summary
    summary_path = data_dir / "building_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(building_summary, f, indent=2)
    
    print(f"Building summary created and saved to: {summary_path}")
    print(f"Buildings processed: {len(building_summary)}")
    
    for building_id, info in building_summary.items():
        print(f"\n{building_id}:")
        print(f"  Type: {info['building_type']}")
        print(f"  PV System: {info['pv_system']}")
        print(f"  Flexible Devices: {info['flexible_devices']}")
        print(f"  Peak Load: {info['peak_load']} kWh")
    
    return building_summary

if __name__ == "__main__":
    create_building_summary()