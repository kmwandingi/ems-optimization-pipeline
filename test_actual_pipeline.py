#!/usr/bin/env python3
"""
Test the ACTUAL pipeline code with synthetic data to trace the real optimization behavior.
This uses the exact same run_building_optimization_direct function from the main script.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add notebooks directory to path
notebooks_dir = Path(__file__).parent / 'notebooks'
sys.path.insert(0, str(notebooks_dir))

# Import the actual pipeline functions
from utils.device_specs import device_specs
from utils.config import BATTERY_PARAMS, FLEXIBLE_PARAMS, GRID_PARAMS

def create_synthetic_building_data():
    """Create synthetic building data that mimics real parquet files"""
    
    print("Creating synthetic building data...")
    
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
    
    # Create realistic device consumption patterns
    data = pd.DataFrame({
        'utc_timestamp': timestamps,
        'price_per_kwh': prices,
        'hour': [t.hour for t in timestamps],
        'day': [t.date() for t in timestamps]
    })
    
    # Add device consumption columns based on device_specs
    for device_key in device_specs.keys():
        consumption = np.zeros(48)
        
        if device_key == 'washing_machine':
            # Runs 3 hours in morning on each day
            consumption[8:11] = [2.0, 2.5, 1.5]  # Day 1
            consumption[32:35] = [2.0, 2.5, 1.5]  # Day 2
            
        elif device_key == 'dishwasher':
            # Runs 2 hours in evening on each day
            consumption[19:21] = [1.8, 1.2]  # Day 1
            consumption[43:45] = [1.8, 1.2]  # Day 2
            
        elif device_key == 'dryer':
            # Runs 4 hours in afternoon on each day
            consumption[14:18] = [3.0, 3.2, 2.8, 2.5]  # Day 1
            consumption[38:42] = [3.0, 3.2, 2.8, 2.5]  # Day 2
            
        elif device_key == 'heat_pump':
            # Continuous operation with daily variation
            for h in range(48):
                hour_of_day = h % 24
                if 22 <= hour_of_day or hour_of_day < 6:  # Night - higher heating
                    consumption[h] = 2.2
                elif 10 <= hour_of_day < 16:  # Midday - lower heating
                    consumption[h] = 1.0
                else:
                    consumption[h] = 1.5
                    
        elif device_key == 'ev_charger':
            # Charges at night
            consumption[1:6] = [7.0, 7.0, 7.0, 7.0, 3.5]  # Day 1 night
            consumption[25:30] = [7.0, 7.0, 7.0, 7.0, 3.5]  # Day 2 night
            
        elif device_key == 'water_heater':
            # Heats in early morning
            consumption[5:7] = [4.0, 4.0]  # Day 1
            consumption[29:31] = [4.0, 4.0]  # Day 2
        
        # Add the device column to the dataframe
        data[device_key] = consumption
    
    # Add some weather data (required by pipeline)
    data['temperature'] = np.random.uniform(15, 25, 48)
    data['radiation'] = np.random.uniform(0, 800, 48)
    
    print(f"Created synthetic data: {len(data)} hours, {len(data['day'].unique())} days")
    print(f"Day 1 price std: {data[data['day'] == data['day'].iloc[0]]['price_per_kwh'].std():.6f}")
    print(f"Day 2 price std: {data[data['day'] == data['day'].iloc[24]]['price_per_kwh'].std():.6f}")
    
    # Show device activity summary
    device_cols = [col for col in data.columns if col in device_specs.keys()]
    print(f"\nDevice activity summary:")
    for col in device_cols:
        total_consumption = data[col].sum()
        print(f"  {col}: {total_consumption:.1f} kWh total")
    
    return data

def save_synthetic_data_as_parquet(data, building_id="SYNTHETIC_TEST"):
    """Save synthetic data as parquet file to mimic real data structure"""
    
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    parquet_path = data_dir / f"{building_id}_processed_data.parquet"
    data.to_parquet(parquet_path, index=False)
    
    print(f"Saved synthetic data to: {parquet_path}")
    return parquet_path

def run_pipeline_test_scenarios(building_id="SYNTHETIC_TEST"):
    """Run the actual pipeline with both scenarios and compare results"""
    
    # Import the actual pipeline function
    sys.path.insert(0, str(Path(__file__).parent))
    
    # Import the exact function from the main script
    exec(open("scripts/calculate_paper_metrics.py").read(), globals())
    
    print(f"\n{'='*80}")
    print("TESTING ACTUAL PIPELINE WITH SYNTHETIC DATA")
    print(f"{'='*80}")
    
    # Test Scenario 1: Without price variation enforcement (some flat days)
    print(f"\n--- SCENARIO 1: WITHOUT PRICE VARIATION ENFORCEMENT ---")
    
    # Temporarily disable price variation enforcement
    original_enforce_function = globals().get('enforce_price_variation_daily')
    
    def no_enforce_price_variation_daily(df, **kwargs):
        """Dummy function that doesn't enforce price variation"""
        return df
    
    globals()['enforce_price_variation_daily'] = no_enforce_price_variation_daily
    
    try:
        devices_scenario1, optimizer_scenario1, has_pv_scenario1 = run_building_optimization_direct(
            building_id=building_id,
            use_proxy_battery=True,
            device_specs=device_specs,
            days=2,
            parquet_dir="data",
            battery_params=BATTERY_PARAMS,
            flexible_params=FLEXIBLE_PARAMS,
            grid_params=GRID_PARAMS,
        )
        
        print(f"✓ Scenario 1 completed successfully")
        
    except Exception as e:
        print(f"✗ Scenario 1 failed: {e}")
        devices_scenario1 = None
    
    # Test Scenario 2: With price variation enforcement (all days varied)
    print(f"\n--- SCENARIO 2: WITH PRICE VARIATION ENFORCEMENT ---")
    
    # Restore original price variation enforcement
    if original_enforce_function:
        globals()['enforce_price_variation_daily'] = original_enforce_function
    
    try:
        devices_scenario2, optimizer_scenario2, has_pv_scenario2 = run_building_optimization_direct(
            building_id=building_id,
            use_proxy_battery=True,
            device_specs=device_specs,
            days=2,
            parquet_dir="data",
            battery_params=BATTERY_PARAMS,
            flexible_params=FLEXIBLE_PARAMS,
            grid_params=GRID_PARAMS,
        )
        
        print(f"✓ Scenario 2 completed successfully")
        
    except Exception as e:
        print(f"✗ Scenario 2 failed: {e}")
        devices_scenario2 = None
    
    # Compare results
    if devices_scenario1 and devices_scenario2:
        compare_scenarios(devices_scenario1, devices_scenario2)
    else:
        print("Cannot compare scenarios - one or both failed")

def compare_scenarios(devices_scenario1, devices_scenario2):
    """Compare the results from both scenarios"""
    
    print(f"\n{'='*80}")
    print("SCENARIO COMPARISON")
    print(f"{'='*80}")
    
    # Analyze each day separately
    for day_idx in range(2):
        day_name = "Day 1 (Flat)" if day_idx == 0 else "Day 2 (Varied)"
        
        print(f"\n--- {day_name} ---")
        
        # Get day data for scenario 1
        day_dates = devices_scenario1[0].data['day'].unique()
        if day_idx < len(day_dates):
            target_day = day_dates[day_idx]
            
            # Scenario 1 analysis
            day_mask_s1 = devices_scenario1[0].data['day'] == target_day
            day_data_s1 = devices_scenario1[0].data[day_mask_s1]
            
            # Scenario 2 analysis
            day_mask_s2 = devices_scenario2[0].data['day'] == target_day
            day_data_s2 = devices_scenario2[0].data[day_mask_s2]
            
            print(f"Prices S1: {day_data_s1['price_per_kwh'].tolist()[:6]}...{day_data_s1['price_per_kwh'].tolist()[-6:]}")
            print(f"Prices S2: {day_data_s2['price_per_kwh'].tolist()[:6]}...{day_data_s2['price_per_kwh'].tolist()[-6:]}")
            print(f"Price std S1: {day_data_s1['price_per_kwh'].std():.6f}")
            print(f"Price std S2: {day_data_s2['price_per_kwh'].std():.6f}")
            
            # Compare device loads
            total_orig_s1 = sum(dev.original_consumption[day_mask_s1].sum() for dev in devices_scenario1)
            total_opt_s1 = sum(dev.optimized_consumption[day_mask_s1].sum() for dev in devices_scenario1)
            
            total_orig_s2 = sum(dev.original_consumption[day_mask_s2].sum() for dev in devices_scenario2)
            total_opt_s2 = sum(dev.optimized_consumption[day_mask_s2].sum() for dev in devices_scenario2)
            
            print(f"Total Original Load - S1: {total_orig_s1:.2f} kWh, S2: {total_orig_s2:.2f} kWh")
            print(f"Total Optimized Load - S1: {total_opt_s1:.2f} kWh, S2: {total_opt_s2:.2f} kWh")
            print(f"Load Reduction - S1: {total_orig_s1 - total_opt_s1:.2f} kWh, S2: {total_orig_s2 - total_opt_s2:.2f} kWh")
            
            # Check battery activity
            if hasattr(devices_scenario1[0], 'battery_charge'):
                battery_charge_s1 = devices_scenario1[0].battery_charge[day_mask_s1].sum()
                battery_discharge_s1 = devices_scenario1[0].battery_discharge[day_mask_s1].sum()
                
                battery_charge_s2 = devices_scenario2[0].battery_charge[day_mask_s2].sum()
                battery_discharge_s2 = devices_scenario2[0].battery_discharge[day_mask_s2].sum()
                
                print(f"Battery Charge - S1: {battery_charge_s1:.2f} kWh, S2: {battery_charge_s2:.2f} kWh")
                print(f"Battery Discharge - S1: {battery_discharge_s1:.2f} kWh, S2: {battery_discharge_s2:.2f} kWh")
                print(f"Net Battery Flow - S1: {battery_charge_s1 - battery_discharge_s1:.2f} kWh, S2: {battery_charge_s2 - battery_discharge_s2:.2f} kWh")

def main():
    """Main test function"""
    
    print("Testing Actual Pipeline with Synthetic Data")
    print("="*50)
    
    # Create synthetic data
    synthetic_data = create_synthetic_building_data()
    
    # Save as parquet file
    parquet_path = save_synthetic_data_as_parquet(synthetic_data)
    
    # Run pipeline tests
    run_pipeline_test_scenarios()
    
    print(f"\n✓ Pipeline test completed!")

if __name__ == "__main__":
    main()
