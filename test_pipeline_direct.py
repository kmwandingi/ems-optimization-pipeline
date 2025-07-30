#!/usr/bin/env python3
"""
Test the actual pipeline code directly with synthetic data
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add notebooks directory to path
notebooks_dir = Path(__file__).parent / 'notebooks'
sys.path.insert(0, str(notebooks_dir))

# Import everything needed for the pipeline
from agents.PVAgent import PVAgent
from agents.FlexibleDeviceAgent import FlexibleDevice
from agents.BatteryAgent import BatteryAgent
from agents.GlobalConnectionLayer import GlobalConnectionLayer
from agents.GridAgent import GridAgent
from agents.GlobalOptimizer import GlobalOptimizer
from agents.WeatherAgent import WeatherAgent
from utils.device_specs import device_specs
from utils.config import BATTERY_PARAMS, FLEXIBLE_PARAMS, GRID_PARAMS, PV_PARAMS

def enforce_price_variation_daily(df, *, min_unique=4, rel_range=0.10):
    """Copy of the enforce_price_variation_daily function"""
    df = df.copy()
    df['utc_timestamp'] = pd.to_datetime(df['utc_timestamp'])
    df['day'] = df['utc_timestamp'].dt.date

    patched = []
    for day, sub in df.groupby('day'):
        prices = sub['price_per_kwh']
        if prices.nunique() < min_unique or (prices.max() - prices.min()) / prices.mean() < rel_range:
            base = prices.mean() if prices.nunique() else 0.20
            hrs = sub['utc_timestamp'].dt.hour
            mult = np.where(
                (18 <= hrs) & (hrs < 22), 1.5,
                np.where((8 <= hrs) & (hrs < 16), 1.2, 0.8)
            )
            sub = sub.assign(price_per_kwh=base * mult)
            print(f"[enforce_price_variation_daily] Day {day}: flat tariff → applied TOU (base={base:.3f}).")
        patched.append(sub)

    return pd.concat(patched, axis=0).sort_index().drop(columns='day')

def filter_complete_days_for_all(devices, required_hours=24):
    """Copy of the filter function"""
    device_sets = []
    for dev in devices:
        counts = dev.data.groupby("day").size()
        device_sets.append(set(counts[counts == required_hours].index))

    global_days = set.intersection(*device_sets)
    if not global_days:
        print("No common complete days – result will be empty.")

    for dev in devices:
        dev.data = (dev.data[dev.data["day"].isin(global_days)]
                             .reset_index(drop=True))
        dev.original_consumption = dev.data[dev.device_name].values
        dev.optimized_consumption = dev.original_consumption.copy()
        n = len(dev.data)
        dev.battery_soc = np.zeros(n)
        dev.battery_charge = np.zeros(n)
        dev.battery_discharge = np.zeros(n)

def create_synthetic_data():
    """Create synthetic data with 2 days: one flat, one varied"""
    
    # Create 48 hours (2 days)
    timestamps = [datetime(2024, 1, 1) + timedelta(hours=h) for h in range(48)]
    
    # Day 1: Flat prices, Day 2: Varied prices
    prices = []
    for h in range(48):
        if h < 24:  # Day 1 - flat
            prices.append(0.20)
        else:  # Day 2 - varied
            hour_of_day = h % 24
            if 6 <= hour_of_day < 9 or 17 <= hour_of_day < 21:  # Peak
                prices.append(0.25)
            elif 10 <= hour_of_day < 16:  # Solar midday
                prices.append(0.15)
            else:  # Off-peak
                prices.append(0.20)
    
    # Create device consumption patterns
    data = pd.DataFrame({
        'utc_timestamp': timestamps,
        'price_per_kwh': prices,
        'hour': [t.hour for t in timestamps],
        'day': [t.date() for t in timestamps]
    })
    
    # Add device columns
    for device_key in device_specs.keys():
        consumption = np.zeros(48)
        
        if device_key == 'washing_machine':
            consumption[8:11] = [2.0, 2.5, 1.5]  # Day 1
            consumption[32:35] = [2.0, 2.5, 1.5]  # Day 2
        elif device_key == 'dishwasher':
            consumption[19:21] = [1.8, 1.2]  # Day 1
            consumption[43:45] = [1.8, 1.2]  # Day 2
        elif device_key == 'dryer':
            consumption[14:18] = [3.0, 3.2, 2.8, 2.5]  # Day 1
            consumption[38:42] = [3.0, 3.2, 2.8, 2.5]  # Day 2
        elif device_key == 'heat_pump':
            for h in range(48):
                hour_of_day = h % 24
                if 22 <= hour_of_day or hour_of_day < 6:
                    consumption[h] = 2.2
                elif 10 <= hour_of_day < 16:
                    consumption[h] = 1.0
                else:
                    consumption[h] = 1.5
        elif device_key == 'ev_charger':
            consumption[1:6] = [7.0, 7.0, 7.0, 7.0, 3.5]  # Day 1
            consumption[25:30] = [7.0, 7.0, 7.0, 7.0, 3.5]  # Day 2
        elif device_key == 'water_heater':
            consumption[5:7] = [4.0, 4.0]  # Day 1
            consumption[29:31] = [4.0, 4.0]  # Day 2
        
        data[device_key] = consumption
    
    # Add weather data
    data['temperature'] = np.random.uniform(15, 25, 48)
    data['radiation'] = np.random.uniform(0, 800, 48)
    
    return data

def run_optimization_test(data, enforce_variation=False, test_name=""):
    """Run the actual optimization pipeline"""
    
    print(f"\n{'='*60}")
    print(f"RUNNING TEST: {test_name}")
    print(f"Enforce price variation: {enforce_variation}")
    print(f"{'='*60}")
    
    # Apply price variation enforcement if requested
    if enforce_variation:
        data = enforce_price_variation_daily(data)
    
    # Create GlobalConnectionLayer
    global_layer = GlobalConnectionLayer(max_building_load=20.0, total_hours=24)
    
    # Create weather agent
    weather_df = data[['utc_timestamp', 'temperature', 'radiation']].copy()
    weather_agent = WeatherAgent(weather_df=weather_df)
    
    # Create battery agent
    battery_agent = BatteryAgent(**BATTERY_PARAMS)
    
    # Create devices
    devices = []
    max_shift_hours = FLEXIBLE_PARAMS.get("max_shift_hours", 16)
    
    for dev_key, specs in device_specs.items():
        if dev_key in data.columns:
            dev = FlexibleDevice(
                data=data,
                device_name=dev_key,
                category=specs['category'],
                power_rating=specs['power_rating'],
                global_layer=global_layer,
                max_shift_hours=max_shift_hours,
                is_flexible=(dev_key != "heat_pump"),
                battery_agent=battery_agent,
                spec=specs
            )
            devices.append(dev)
            global_layer.register_device(dev)
    
    print(f"Created {len(devices)} devices")
    
    # Create optimizer
    optimizer = GlobalOptimizer(
        devices=devices,
        global_layer=global_layer,
        weather_agent=weather_agent,
        battery_agent=battery_agent,
        max_iterations=1,
        online_iterations=3
    )
    
    # Filter and run optimization
    filter_complete_days_for_all(devices, required_hours=24)
    
    print("Running optimization...")
    optimizer.optimize_centralized()
    
    # Analyze results
    analyze_results(devices, data, test_name)
    
    return devices, optimizer

def analyze_results(devices, data, test_name):
    """Analyze optimization results"""
    
    print(f"\nRESULTS FOR: {test_name}")
    print("-" * 40)
    
    # Analyze by day
    for day_idx, day_date in enumerate(data['day'].unique()):
        day_mask = data['day'] == day_date
        day_data = data[day_mask]
        
        print(f"\nDAY {day_idx + 1}: {day_date}")
        print(f"Price variation (std): {day_data['price_per_kwh'].std():.6f}")
        print(f"Prices: {day_data['price_per_kwh'].tolist()}")
        
        # Calculate total loads
        total_original = 0
        total_optimized = 0
        
        for dev in devices:
            day_indices = dev.data[dev.data['day'] == day_date].index
            if len(day_indices) > 0:
                orig_consumption = dev.original_consumption[day_indices].sum()
                opt_consumption = dev.optimized_consumption[day_indices].sum()
                
                total_original += orig_consumption
                total_optimized += opt_consumption
                
                print(f"  {dev.device_name:15}: {orig_consumption:6.2f} -> {opt_consumption:6.2f} kWh")
        
        # Check battery activity
        if hasattr(devices[0], 'battery_charge') and devices[0].battery_charge is not None:
            day_indices = devices[0].data[devices[0].data['day'] == day_date].index
            day_charge = devices[0].battery_charge[day_indices].sum()
            day_discharge = devices[0].battery_discharge[day_indices].sum()
            
            print(f"  {'Battery Charge':15}: {day_charge:6.2f} kWh")
            print(f"  {'Battery Discharge':15}: {day_discharge:6.2f} kWh")
            print(f"  {'Net Battery':15}: {day_charge - day_discharge:+6.2f} kWh")
        
        print(f"  {'TOTAL BUILDING':15}: {total_original:6.2f} -> {total_optimized:6.2f} kWh")
        print(f"  {'LOAD REDUCTION':15}: {total_original - total_optimized:+6.2f} kWh")

def main():
    """Main test function"""
    
    print("Testing Actual Pipeline with Synthetic Data")
    print("="*50)
    
    # Create synthetic data
    data = create_synthetic_data()
    
    print(f"Created synthetic data: {len(data)} hours, {len(data['day'].unique())} days")
    print(f"Day 1 price std: {data[data['day'] == data['day'].iloc[0]]['price_per_kwh'].std():.6f}")
    print(f"Day 2 price std: {data[data['day'] == data['day'].iloc[24]]['price_per_kwh'].std():.6f}")
    
    # Test 1: Without price variation enforcement
    devices_no_enforce, _ = run_optimization_test(
        data.copy(), 
        enforce_variation=False, 
        test_name="NO PRICE VARIATION ENFORCEMENT"
    )
    
    # Test 2: With price variation enforcement
    devices_with_enforce, _ = run_optimization_test(
        data.copy(), 
        enforce_variation=True, 
        test_name="WITH PRICE VARIATION ENFORCEMENT"
    )
    
    # Compare results
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    print("This test uses your actual pipeline code with synthetic data")
    print("to trace the exact behavior difference between scenarios.")

if __name__ == "__main__":
    main()
