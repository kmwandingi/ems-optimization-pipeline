#!/usr/bin/env python3
"""
Minimal test script to trace battery optimization behavior with controlled data.
Tests both flat-price and varied-price scenarios to understand the actual code behavior.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
notebooks_dir = project_root / 'notebooks'
sys.path.append(str(notebooks_dir))

from agents.BatteryAgent import BatteryAgent
from agents.FlexibleDeviceAgent import FlexibleDevice
from agents.GlobalConnectionLayer import GlobalConnectionLayer
from agents.GlobalOptimizer import GlobalOptimizer
from utils.device_specs import device_specs
from utils.config import BATTERY_PARAMS

def create_test_data():
    """Create minimal test data with 2 days: one flat-price, one varied-price"""
    
    # Create 48 hours (2 days) of data
    timestamps = [datetime(2024, 1, 1) + timedelta(hours=h) for h in range(48)]
    
    # Day 1: Flat prices (0.20 €/kWh all hours)
    # Day 2: Varied prices (0.15 to 0.25 €/kWh)
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
    
    # Create device consumption patterns (6 devices)
    device_data = {}
    
    # Washing machine - runs 3 hours in morning
    washing_machine = np.zeros(48)
    washing_machine[8:11] = [2.0, 2.5, 1.5]  # Day 1
    washing_machine[32:35] = [2.0, 2.5, 1.5]  # Day 2
    
    # Dishwasher - runs 2 hours in evening
    dishwasher = np.zeros(48)
    dishwasher[19:21] = [1.8, 1.2]  # Day 1
    dishwasher[43:45] = [1.8, 1.2]  # Day 2
    
    # Dryer - runs 4 hours afternoon
    dryer = np.zeros(48)
    dryer[14:18] = [3.0, 3.2, 2.8, 2.5]  # Day 1
    dryer[38:42] = [3.0, 3.2, 2.8, 2.5]  # Day 2
    
    # Heat pump - continuous with variation
    heat_pump = np.full(48, 1.5)
    for h in range(48):
        hour_of_day = h % 24
        if 22 <= hour_of_day or hour_of_day < 6:  # Night - higher heating
            heat_pump[h] = 2.2
        elif 10 <= hour_of_day < 16:  # Midday - lower heating
            heat_pump[h] = 1.0
    
    # EV charger - charges at night
    ev_charger = np.zeros(48)
    ev_charger[1:6] = [7.0, 7.0, 7.0, 7.0, 3.5]  # Day 1 night
    ev_charger[25:30] = [7.0, 7.0, 7.0, 7.0, 3.5]  # Day 2 night
    
    # Water heater - heats in early morning
    water_heater = np.zeros(48)
    water_heater[5:7] = [4.0, 4.0]  # Day 1
    water_heater[29:31] = [4.0, 4.0]  # Day 2
    
    # Create DataFrame
    data = pd.DataFrame({
        'utc_timestamp': timestamps,
        'price_per_kwh': prices,
        'washing_machine': washing_machine,
        'dishwasher': dishwasher,
        'dryer': dryer,
        'heat_pump': heat_pump,
        'ev_charger': ev_charger,
        'water_heater': water_heater,
        'hour': [t.hour for t in timestamps],
        'day': [t.date() for t in timestamps]
    })
    
    return data

def run_test_optimization(data, use_battery=True, test_name=""):
    """Run optimization on test data and return results"""
    
    print(f"\n{'='*60}")
    print(f"RUNNING TEST: {test_name}")
    print(f"{'='*60}")
    
    # Create GlobalConnectionLayer
    global_layer = GlobalConnectionLayer(max_building_load=20.0, total_hours=24)
    
    # Create BatteryAgent if requested
    battery_agent = None
    if use_battery:
        battery_agent = BatteryAgent(**BATTERY_PARAMS)
    
    # Create devices
    devices = []
    device_columns = ['washing_machine', 'dishwasher', 'dryer', 'heat_pump', 'ev_charger', 'water_heater']
    
    for col in device_columns:
        if col in device_specs:
            spec = device_specs[col]
            dev = FlexibleDevice(
                data=data,
                device_name=col,
                category=spec['category'],
                power_rating=spec['power_rating'],
                global_layer=global_layer,
                max_shift_hours=16,
                is_flexible=(col != 'heat_pump'),  # Heat pump not flexible
                battery_agent=battery_agent,
                spec=spec
            )
            devices.append(dev)
            global_layer.register_device(dev)
    
    # Create optimizer
    optimizer = GlobalOptimizer(
        devices=devices,
        global_layer=global_layer,
        battery_agent=battery_agent,
        max_iterations=1,
        online_iterations=1
    )
    
    # Run optimization
    print(f"Running optimization with {len(devices)} devices...")
    optimizer.optimize_centralized()
    
    # Analyze results
    results = analyze_results(devices, data, test_name)
    
    return results, devices, optimizer

def analyze_results(devices, data, test_name):
    """Analyze and print optimization results"""
    
    results = {
        'test_name': test_name,
        'days': {},
        'total_original_load': 0,
        'total_optimized_load': 0,
        'battery_activity': {}
    }
    
    print(f"\nRESULTS ANALYSIS FOR: {test_name}")
    print("-" * 50)
    
    # Analyze by day
    for day_idx, day_date in enumerate(data['day'].unique()):
        day_mask = data['day'] == day_date
        day_data = data[day_mask]
        
        day_results = {
            'date': day_date,
            'prices': day_data['price_per_kwh'].tolist(),
            'price_variation': day_data['price_per_kwh'].std(),
            'devices': {},
            'total_original': 0,
            'total_optimized': 0
        }
        
        print(f"\nDAY {day_idx + 1}: {day_date}")
        print(f"Price variation (std): {day_results['price_variation']:.4f}")
        print(f"Prices: {day_results['prices'][:6]}...{day_results['prices'][-6:]}")
        
        # Analyze each device
        for dev in devices:
            day_indices = dev.data[dev.data['day'] == day_date].index
            
            if len(day_indices) > 0:
                orig_consumption = dev.original_consumption[day_indices]
                opt_consumption = dev.optimized_consumption[day_indices]
                
                orig_total = orig_consumption.sum()
                opt_total = opt_consumption.sum()
                
                day_results['devices'][dev.device_name] = {
                    'original': orig_total,
                    'optimized': opt_total,
                    'difference': orig_total - opt_total
                }
                
                day_results['total_original'] += orig_total
                day_results['total_optimized'] += opt_total
                
                print(f"  {dev.device_name:15}: {orig_total:6.2f} -> {opt_total:6.2f} kWh (diff: {orig_total-opt_total:+6.2f})")
        
        # Check battery activity
        if hasattr(devices[0], 'battery_charge') and devices[0].battery_charge is not None:
            day_charge = devices[0].battery_charge[day_indices].sum()
            day_discharge = devices[0].battery_discharge[day_indices].sum()
            
            day_results['battery_charge'] = day_charge
            day_results['battery_discharge'] = day_discharge
            
            print(f"  {'Battery Charge':15}: {day_charge:6.2f} kWh")
            print(f"  {'Battery Discharge':15}: {day_discharge:6.2f} kWh")
            print(f"  {'Net Battery':15}: {day_charge - day_discharge:+6.2f} kWh")
        
        print(f"  {'TOTAL BUILDING':15}: {day_results['total_original']:6.2f} -> {day_results['total_optimized']:6.2f} kWh")
        print(f"  {'LOAD REDUCTION':15}: {day_results['total_original'] - day_results['total_optimized']:+6.2f} kWh")
        
        results['days'][day_idx] = day_results
        results['total_original_load'] += day_results['total_original']
        results['total_optimized_load'] += day_results['total_optimized']
    
    print(f"\nOVERALL SUMMARY:")
    print(f"Total Original Load:  {results['total_original_load']:8.2f} kWh")
    print(f"Total Optimized Load: {results['total_optimized_load']:8.2f} kWh")
    print(f"Total Load Reduction: {results['total_original_load'] - results['total_optimized_load']:+8.2f} kWh")
    
    return results

def main():
    """Main test function"""
    
    print("Creating test data...")
    data = create_test_data()
    
    print(f"Test data created: {len(data)} hours, {len(data['day'].unique())} days")
    print(f"Day 1 price std: {data[data['day'] == data['day'].iloc[0]]['price_per_kwh'].std():.4f}")
    print(f"Day 2 price std: {data[data['day'] == data['day'].iloc[24]]['price_per_kwh'].std():.4f}")
    
    # Test 1: No battery
    results_no_battery, _, _ = run_test_optimization(
        data, use_battery=False, test_name="NO BATTERY"
    )
    
    # Test 2: With battery
    results_with_battery, devices_with_battery, optimizer_with_battery = run_test_optimization(
        data, use_battery=True, test_name="WITH BATTERY"
    )
    
    # Compare results
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    print(f"{'Scenario':<20} {'Day 1 (Flat)':<15} {'Day 2 (Varied)':<15} {'Total':<10}")
    print("-" * 65)
    
    no_batt_day1 = results_no_battery['days'][0]['total_optimized']
    no_batt_day2 = results_no_battery['days'][1]['total_optimized']
    no_batt_total = results_no_battery['total_optimized_load']
    
    with_batt_day1 = results_with_battery['days'][0]['total_optimized']
    with_batt_day2 = results_with_battery['days'][1]['total_optimized']
    with_batt_total = results_with_battery['total_optimized_load']
    
    print(f"{'No Battery':<20} {no_batt_day1:<15.2f} {no_batt_day2:<15.2f} {no_batt_total:<10.2f}")
    print(f"{'With Battery':<20} {with_batt_day1:<15.2f} {with_batt_day2:<15.2f} {with_batt_total:<10.2f}")
    print(f"{'Difference':<20} {with_batt_day1-no_batt_day1:<+15.2f} {with_batt_day2-no_batt_day2:<+15.2f} {with_batt_total-no_batt_total:<+10.2f}")
    
    return results_no_battery, results_with_battery

if __name__ == "__main__":
    results_no_battery, results_with_battery = main()
