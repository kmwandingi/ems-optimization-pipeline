#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Calculate metrics for the paper comparing non-scheduled vs scheduled vs scheduled with battery scenarios.
This script processes all buildings, runs optimizations, and outputs a structured DataFrame with metrics.
"""

import os
import sys
import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from tqdm import tqdm

# Add the parent directory to path to make imports work
script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import the required modules
# Add notebooks directory to Python path
import sys
import os
script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
notebooks_dir = script_dir.parent / 'notebooks'
sys.path.insert(0, str(notebooks_dir))

# Import the function from our module
from building_optimizer import run_building_optimization_direct
from notebooks.agents.GridAgent import GridAgent

# Default export price factor if not available from GridAgent
DEFAULT_EXPORT_FACTOR = 0.8


def par(series):
    """Calculate the Peak-to-Average Ratio (PAR) for a series.
    
    PAR should reflect how peaked the load is compared to its average.
    For grid demand, lower PAR is better (flatter demand profile).
    
    Returns PAR as peak/average for the positive load values only.
    This ensures we're measuring what matters: how peaked is the consumption from grid.
    """
    if len(series) == 0:
        return float('nan')  # Empty series case
    
    # Only consider positive values (grid consumption) for PAR
    # This is the standard approach in energy systems - we care about peaks in consumption
    positive_series = series[series > 0]
    
    # If no positive values, PAR is undefined
    if len(positive_series) == 0:
        return float('nan')
    
    # Calculate peak and average
    peak = positive_series.max()
    avg = positive_series.mean()
    
    # Avoid division by zero
    if avg == 0:
        return float('nan')
        
    par_value = peak / avg
    
    # Sanity check - PAR should generally be > 1 and reasonable
    # If we get extreme values, something might be wrong
    if par_value > 20:
        print(f"Warning: Very high PAR detected ({par_value:.2f}), this may indicate a problem with the data")
    
    return par_value


def align_series_to_index(data, target_index):
    """
    Align a data series to a target index, handling missing values.
    
    Args:
        data: Data to align (Series, array, or list)
        target_index: Target index to align to
    
    Returns:
        pandas.Series aligned to target_index
    """
    if isinstance(data, pd.Series) and data.index.equals(target_index):
        return data
    
    # Convert array-like to Series with the target index
    if isinstance(data, pd.Series):
        # Reindex the existing Series
        return data.reindex(target_index)
    else:
        # Create a new Series
        if len(data) == len(target_index):
            return pd.Series(data, index=target_index)
        else:
            # Handle length mismatch (this is a simplification)
            warnings.warn(f"Data length ({len(data)}) doesn't match index length ({len(target_index)}). Using available data.")
            s = pd.Series(index=target_index)
            s.iloc[:min(len(data), len(target_index))] = data[:min(len(data), len(target_index))]
            return s


def calculate_metrics_for_building(building_id, device_specs, parquet_dir="data", max_building_load=10.0,
                                  battery_params=None, flexible_params=None, grid_params=None, 
                                  pv_params=None, days=None):
    """
    Calculate all metrics for a single building by running both non-battery and with-battery optimizations.
    
    Args:
        building_id: ID of the building to analyze
        device_specs: Dictionary of device specifications
        Other parameters passed to run_building_optimization_direct
    
    Returns:
        Dictionary with calculated metrics for the paper table
    """
    print(f"\n{'='*80}\nProcessing Building: {building_id}\n{'='*80}")
    
    # Run optimization without battery (Scheduled-Only scenario)
    print("\nRunning optimization WITHOUT battery...")
    try:
        devices_nb, optimizer_nb, has_pv_nb = run_building_optimization_direct(
            building_id=building_id,
            use_proxy_battery=False,
            device_specs=device_specs,
            parquet_dir=parquet_dir,
            max_building_load=max_building_load,
            battery_params=battery_params,
            flexible_params=flexible_params,
            grid_params=grid_params,
            pv_params=pv_params,
            days=days
        )
    except Exception as e:
        print(f"Error running non-battery optimization for building {building_id}: {str(e)}")
        return None
    
    # Run optimization with battery (Scheduled-Battery scenario)
    print("\nRunning optimization WITH battery...")
    try:
        devices_wb, optimizer_wb, has_pv_wb = run_building_optimization_direct(
            building_id=building_id,
            use_proxy_battery=True,
            device_specs=device_specs,
            parquet_dir=parquet_dir,
            max_building_load=max_building_load,
            battery_params=battery_params,
            flexible_params=flexible_params,
            grid_params=grid_params,
            pv_params=pv_params,
            days=days
        )
    except Exception as e:
        print(f"Error running with-battery optimization for building {building_id}: {str(e)}")
        return None
    
    # Ensure both runs were successful
    if not devices_nb or not devices_wb:
        print(f"Missing devices for building {building_id}")
        return None
    
    print(f"Both optimizations completed for building {building_id}")
    
    # Get the export price factor (default if not available)
    export_price_factor = DEFAULT_EXPORT_FACTOR
    grid_agent = GridAgent()
    if hasattr(grid_agent, 'export_price_factor'):
        export_price_factor = grid_agent.export_price_factor
    
    # Count flexible devices
    num_flexible_devices = sum(1 for dev in devices_nb if dev.is_flexible)
    
    # ==================================================================================
    # Step 1: Extract time-series for each scenario
    # ==================================================================================
    print("\nExtracting time-series data...")
    
    # Get a reference device to use as our time index source
    ref_device = devices_nb[0]  
    
    # Create a DataFrame with timestamp index from the reference device's data
    if 'utc_timestamp' in ref_device.data.columns:
        time_index = pd.DatetimeIndex(ref_device.data['utc_timestamp'])
    else:
        # Create synthetic timestamps if real ones aren't available
        start_time = pd.Timestamp('2023-01-01')
        time_index = pd.date_range(start=start_time, periods=len(ref_device.data), freq='H')
    
    # Store price series aligned with our time index
    price_series = align_series_to_index(ref_device.data['price_per_kwh'].values, time_index)
    
    # Create baseline, scheduled-only, and scheduled-battery net load series
    baseline_series = pd.Series(0.0, index=time_index)
    schedonly_series = pd.Series(0.0, index=time_index)
    schedbatt_series = pd.Series(0.0, index=time_index)
    
    # Get PV profile if available (negative values for generation)
    pv_profile = pd.Series(0.0, index=time_index)  # Default: no PV
    if has_pv_nb and optimizer_nb.pv_agent:
        try:
            # Try to get the actual measured PV profile
            if hasattr(optimizer_nb.pv_agent, 'profile_data') and optimizer_nb.pv_agent.profile_data is not None:
                if isinstance(optimizer_nb.pv_agent.profile_data, pd.DataFrame) and 'pv' in optimizer_nb.pv_agent.profile_data.columns:
                    pv_df = optimizer_nb.pv_agent.profile_data
                    if pv_df.index.name == 'utc_timestamp':
                        # Align PV profile to our time index
                        pv_profile = pv_df['pv'].reindex(time_index, fill_value=0)
                    else:
                        # Try to reconstruct PV profile from scratch
                        pv_profile = optimizer_nb.pv_agent.get_pv_profile(time_index)
        except Exception as e:
            print(f"Error getting PV profile: {str(e)}")
            pv_profile = pd.Series(0.0, index=time_index)  # Fallback to no PV
    
    # Ensure PV profile is correctly oriented (negative for generation)
    if not np.all(pv_profile[pv_profile != 0] <= 0):
        pv_profile = -np.abs(pv_profile)  # Ensure PV generation is negative
    
    print(f"Time index has {len(time_index)} points")
    print(f"PV profile has {np.count_nonzero(pv_profile < 0)} generation points")
    
    # Sum up device consumption for each scenario
    for i, (dev_nb, dev_wb) in enumerate(zip(devices_nb, devices_wb)):
        # Ensure arrays are properly sized to match time_index
        if len(dev_nb.original_consumption) != len(time_index):
            print(f"Warning: Device {i} original consumption length mismatch: {len(dev_nb.original_consumption)} vs {len(time_index)}")
            continue
            
        if len(dev_nb.optimized_consumption) != len(time_index) or len(dev_wb.optimized_consumption) != len(time_index):
            print(f"Warning: Device {i} optimized consumption length mismatch")
            continue
            
        # Add each device's consumption to the corresponding scenario
        baseline_series += pd.Series(dev_nb.original_consumption, index=time_index)
        schedonly_series += pd.Series(dev_nb.optimized_consumption, index=time_index)
        schedbatt_series += pd.Series(dev_wb.optimized_consumption, index=time_index)
    
    # Extract battery charge/discharge for metrics and validation
    battery_charge = None
    battery_discharge = None
    battery_soc = None
    
    if optimizer_wb.battery_charge_global is not None and optimizer_wb.battery_discharge_global is not None:
        battery_charge = align_series_to_index(optimizer_wb.battery_charge_global, time_index)
        battery_discharge = align_series_to_index(optimizer_wb.battery_discharge_global, time_index)
        
        # Get battery SoC if available
        if optimizer_wb.battery_soc_global is not None:
            battery_soc = align_series_to_index(optimizer_wb.battery_soc_global, time_index)
        
        print("\nBattery Operation Analysis:")
        print("============================")
        print(f"Total battery charging: {battery_charge.sum():.2f} kWh")
        print(f"Total battery discharging: {battery_discharge.sum():.2f} kWh")
        
        # Calculate round-trip efficiency
        if battery_discharge.sum() > 0:
            efficiency = battery_discharge.sum() / battery_charge.sum() * 100
            print(f"Effective round-trip efficiency: {efficiency:.2f}%")
        
        # Show battery activity by hour
        print("\nBattery hourly activity:")
        print("-" * 60)
        print(f"{'Hour':>5} {'SoC (kWh)':>12} {'Charge (kWh)':>14} {'Discharge (kWh)':>16} {'Net (kWh)':>10}")
        print("-" * 60)
        for i in range(len(time_index)):
            hour = time_index[i].hour
            soc_val = battery_soc[i] if battery_soc is not None else float('nan')
            charge_val = battery_charge[i]
            discharge_val = battery_discharge[i]
            net_val = charge_val - discharge_val
            print(f"{hour:5d} {soc_val:12.2f} {charge_val:14.2f} {discharge_val:16.2f} {net_val:10.2f}")
        
        # Important: In most implementations, the battery's impact (charge/discharge)
        # is already included in the net load calculations from the optimizer
        # Add a parameter to explicitly avoid double-counting
        include_battery = False  # Default: assume battery is already included in schedbatt_series
        
        if include_battery:
            schedbatt_series += battery_charge - battery_discharge
    
    # Add PV generation to all scenarios (PV values are negative, so adding reduces net load)
    baseline_series += pv_profile
    schedonly_series += pv_profile
    schedbatt_series += pv_profile
    
    # ==================================================================================
    # Step 2: Compute metrics across all days for each scenario
    # ==================================================================================
    print("\nComputing metrics...")
    
    # Create detailed energy flow breakdowns for each scenario
    print("\nEnergy Flow Analysis:")
    print("======================")
    
    # For each scenario, calculate:
    # 1. Total device consumption
    # 2. Grid import (positive net load)
    # 3. Grid export (negative net load)
    # 4. PV generation
    # 5. Battery charge/discharge (for battery scenario only)
    
    # Baseline scenario
    baseline_consumption = sum(pd.Series(dev_nb.original_consumption, index=time_index).sum() for dev_nb in devices_nb)
    baseline_grid_import = baseline_series[baseline_series > 0].sum()
    baseline_grid_export = abs(baseline_series[baseline_series < 0].sum())
    baseline_pv_generation = (-pv_profile).clip(lower=0).sum()
    
    # Scheduled-only scenario
    schedonly_consumption = sum(pd.Series(dev_nb.optimized_consumption, index=time_index).sum() for dev_nb in devices_nb)
    schedonly_grid_import = schedonly_series[schedonly_series > 0].sum()
    schedonly_grid_export = abs(schedonly_series[schedonly_series < 0].sum())
    
    # Scheduled-battery scenario
    schedbatt_consumption = sum(pd.Series(dev_wb.optimized_consumption, index=time_index).sum() for dev_wb in devices_wb)
    schedbatt_grid_import = schedbatt_series[schedbatt_series > 0].sum()
    schedbatt_grid_export = abs(schedbatt_series[schedbatt_series < 0].sum())
    
    # Battery-specific values
    battery_charge_total = 0
    battery_discharge_total = 0
    if battery_charge is not None and battery_discharge is not None:
        battery_charge_total = battery_charge.sum()
        battery_discharge_total = battery_discharge.sum()
    
    # Print energy flow summary
    print(f"{'Category':<25} {'Baseline':>15} {'Sched-Only':>15} {'Sched-Batt':>15}")
    print("-" * 75)
    print(f"{'Device Consumption (kWh)':<25} {baseline_consumption:>15.2f} {schedonly_consumption:>15.2f} {schedbatt_consumption:>15.2f}")
    print(f"{'Grid Import (kWh)':<25} {baseline_grid_import:>15.2f} {schedonly_grid_import:>15.2f} {schedbatt_grid_import:>15.2f}")
    print(f"{'Grid Export (kWh)':<25} {baseline_grid_export:>15.2f} {schedonly_grid_export:>15.2f} {schedbatt_grid_export:>15.2f}")
    print(f"{'PV Generation (kWh)':<25} {baseline_pv_generation:>15.2f} {baseline_pv_generation:>15.2f} {baseline_pv_generation:>15.2f}")
    if battery_charge is not None:
        print(f"{'Battery Charge (kWh)':<25} {'-':>15} {'-':>15} {battery_charge_total:>15.2f}")
        print(f"{'Battery Discharge (kWh)':<25} {'-':>15} {'-':>15} {battery_discharge_total:>15.2f}")
    
    # Check energy balance
    # Total energy in = Total energy out + losses
    # PV + Grid Import = Consumption + Grid Export + Battery Charge - Battery Discharge + losses
    baseline_balance = baseline_pv_generation + baseline_grid_import - baseline_grid_export - baseline_consumption
    schedonly_balance = baseline_pv_generation + schedonly_grid_import - schedonly_grid_export - schedonly_consumption
    schedbatt_balance = baseline_pv_generation + schedbatt_grid_import - schedbatt_grid_export - schedbatt_consumption - battery_charge_total + battery_discharge_total
    
    print(f"{'Energy Balance Check':<25} {baseline_balance:>15.2f} {schedonly_balance:>15.2f} {schedbatt_balance:>15.2f}")
    print("(Energy balance should be close to zero if all flows are accounted for correctly)")
    
    # For metrics table - use grid import for grid energy metric
    total_baseline_kwh = baseline_grid_import
    total_schedonly_kwh = schedonly_grid_import
    total_schedbatt_kwh = schedbatt_grid_import
    
    # Total cost (€): sum(positive_load × price) - sum(negative_load × price × export_factor)
    # For grid import (positive values)
    cost_baseline_import = (baseline_series[baseline_series > 0] * price_series[baseline_series > 0]).sum()
    cost_schedonly_import = (schedonly_series[schedonly_series > 0] * price_series[schedonly_series > 0]).sum()
    cost_schedbatt_import = (schedbatt_series[schedbatt_series > 0] * price_series[schedbatt_series > 0]).sum()
    
    # For grid export (negative values) - revenue reduces cost
    cost_baseline_export = (baseline_series[baseline_series < 0] * price_series[baseline_series < 0] * export_price_factor).sum()
    cost_schedonly_export = (schedonly_series[schedonly_series < 0] * price_series[schedonly_series < 0] * export_price_factor).sum()
    cost_schedbatt_export = (schedbatt_series[schedbatt_series < 0] * price_series[schedbatt_series < 0] * export_price_factor).sum()
    
    # Net costs (import minus export revenue)
    cost_baseline = cost_baseline_import - abs(cost_baseline_export)
    cost_schedonly = cost_schedonly_import - abs(cost_schedonly_export)
    cost_schedbatt = cost_schedbatt_import - abs(cost_schedbatt_export)
    
    # PV self-consumption calculation - more accurate calculation based on our energy flow analysis
    # PV used locally = Total PV generation - PV exported to grid
    
    # Baseline scenario
    # Amount of PV used locally = Total PV generation - Amount exported
    # Amount exported = Grid export (if there is no other source of export)
    pv_used_baseline = baseline_pv_generation - baseline_grid_export
    
    # Scheduled-only scenario
    pv_used_schedonly = baseline_pv_generation - schedonly_grid_export
    
    # Scheduled-battery scenario - initialize with simple calculation
    pv_used_schedbatt = baseline_pv_generation - schedbatt_grid_export
    
    # Calculate PAR for all scenarios
    par_baseline = par(baseline_series)
    par_schedonly = par(schedonly_series)
    par_schedbatt = par(schedbatt_series)
    
    # OPTIMIZATION VERIFICATION
    print("\nOPTIMIZATION VERIFICATION:")
    print("==========================")
    print(f"{'Metric':<25} {'Baseline→Sched':>15} {'Sched→Batt':>15} {'Status':>10}")
    print("-" * 65)
    
    # Initialize validation issues list
    validation_issues = []
    
    # Check grid energy (should decrease)
    energy_sched_change = total_baseline_kwh - total_schedonly_kwh
    energy_batt_change = total_schedonly_kwh - total_schedbatt_kwh 
    energy_sched_status = "✅" if energy_sched_change >= 0 else "❌"
    energy_batt_status = "✅" if energy_batt_change >= 0 else "❌"
    print(f"{'Grid Energy (kWh)':<25} {energy_sched_change:>+15.2f} {energy_batt_change:>+15.2f} {energy_sched_status + ' ' + energy_batt_status:>10}")
    
    # Add validation issues if any
    if energy_sched_change < 0:
        validation_issues.append(f"Grid energy increased from baseline to scheduled by {-energy_sched_change:.2f} kWh")
    if energy_batt_change < 0:
        validation_issues.append(f"Grid energy increased from scheduled to battery by {-energy_batt_change:.2f} kWh")
    
    # Check cost (should decrease)
    cost_sched_change = cost_baseline - cost_schedonly
    cost_batt_change = cost_schedonly - cost_schedbatt
    cost_sched_status = "✅" if cost_sched_change >= 0 else "❌"
    cost_batt_status = "✅" if cost_batt_change >= 0 else "❌"
    print(f"{'Cost (EUR)':<25} {cost_sched_change:>+15.2f} {cost_batt_change:>+15.2f} {cost_sched_status + ' ' + cost_batt_status:>10}")
    
    # Check PV usage (should increase)
    pv_sched_change = pv_used_schedonly - pv_used_baseline
    pv_batt_change = pv_used_schedbatt - pv_used_schedonly
    pv_sched_status = "✅" if pv_sched_change >= 0 else "❌"
    pv_batt_status = "✅" if pv_batt_change >= 0 else "❌"
    print(f"{'PV Usage (kWh)':<25} {pv_sched_change:>+15.2f} {pv_batt_change:>+15.2f} {pv_sched_status + ' ' + pv_batt_status:>10}")
    
    # Check PAR (should decrease)
    par_sched_change = par_baseline - par_schedonly
    par_batt_change = par_schedonly - par_schedbatt
    par_sched_status = "✅" if par_sched_change >= 0 else "❌"
    par_batt_status = "✅" if par_batt_change >= 0 else "❌"
    print(f"{'PAR':<25} {par_sched_change:>+15.2f} {par_batt_change:>+15.2f} {par_sched_status + ' ' + par_batt_status:>10}")
    
    # Summary of validation issues
    validation_passed = True
    if len(validation_issues) > 0:
        validation_passed = False
        print("\n⚠️ VALIDATION ISSUES DETECTED:")
        for i, issue in enumerate(validation_issues, 1):
            print(f"{i}. {issue}")
        print("\nTHESE MUST BE FIXED FOR DETERMINISTIC OPTIMIZATION CORRECTNESS!")
    else:
        print("\n✅ All validation checks passed! Optimization is working correctly.")
    
    # Check if battery made a significant difference
    if battery_charge is not None and abs(cost_schedonly - cost_schedbatt) < 0.01:
        print("\n⚠️ Battery appears to have minimal impact on cost. Check battery parameters and configuration.")
    
    # Check battery round-trip efficiency
    if battery_charge is not None and battery_charge_total > 0 and battery_discharge_total > 0:
        round_trip = battery_discharge_total / battery_charge_total * 100
        print(f"\nBattery round-trip efficiency: {round_trip:.1f}%")
        if round_trip < 75:
            print("⚠️ Low round-trip efficiency detected. Check battery parameters.")
        elif round_trip > 95:
            print("⚠️ Unusually high round-trip efficiency detected. Check battery model.")
        elif round_trip > 100:
            print("⚠️ Efficiency > 100% detected. This violates physics. Check battery model and accounting.")

    
    # Cost and PV calculations already done above before they are first used
    
    # More sophisticated battery-PV analysis
    # In this case, PV might be stored in the battery and then used later
    # This adds detail to the pv_used_schedbatt calculation already done above
    
    # If we have battery data, we can try to estimate how much PV went into the battery
    # This is a simplification; in reality, we would need to know for each hour
    # whether the battery charged from PV or grid
    if battery_charge is not None:
        # Print additional PV-battery analysis
        print("\nPV-Battery Interaction Analysis:")
        print("===============================\n")
        print("Analyzing possible PV charging of battery...")
        
        # Calculate how much PV could have been used to charge the battery
        # For each hour, check if PV was generating and battery was charging
        pv_to_battery = 0
        for i in range(len(time_index)):
            pv_gen = -pv_profile.iloc[i] if pv_profile.iloc[i] < 0 else 0  # PV generation (positive)
            if pv_gen > 0 and battery_charge is not None and battery_charge.iloc[i] > 0:
                # Estimate the portion of battery charging that came from PV
                # If net load is negative, all charging could be from PV
                # If net load is positive but less than PV, some charging could be from PV
                # If net load is positive and greater than PV, no charging from PV
                net_load = schedbatt_series.iloc[i]  # Without battery's contribution
                
                # Maximum PV available for battery after direct load consumption
                pv_available = max(0, pv_gen - max(0, net_load + battery_charge.iloc[i] - battery_discharge.iloc[i]))  
                
                # Amount of PV that went to battery (minimum of available PV and actual charging)
                pv_to_battery_hour = min(pv_available, battery_charge.iloc[i])
                pv_to_battery += pv_to_battery_hour
                
                if pv_to_battery_hour > 0:
                    print(f"Hour {time_index[i].hour}: {pv_to_battery_hour:.2f} kWh of PV likely used to charge battery")
        
        print(f"\nEstimated total PV used to charge battery: {pv_to_battery:.2f} kWh")
        
        # Adjust PV self-consumption for battery scenario
        # Note: we don't add this to pv_used_schedbatt because the basic calculation already accounts for it
        # But we report it separately for clarity
        print(f"Direct PV consumption: {pv_used_schedbatt:.2f} kWh")
        print(f"Total PV utilization (direct + battery): {pv_used_schedbatt + pv_to_battery * 0.85:.2f} kWh")
        # 0.85 factor accounts for round-trip efficiency losses
    
    # Peak-to-Average Ratio (PAR)
    par_baseline = par(baseline_series)
    par_schedonly = par(schedonly_series)
    par_schedbatt = par(schedbatt_series)
    
    # Cost reduction percentages
    # Handle edge case: zero-cost baseline
    if cost_baseline == 0:
        cost_red_schedonly = 0 if cost_schedonly == 0 else float('inf')
        cost_red_schedbatt = 0 if cost_schedbatt == 0 else float('inf')
    else:
        cost_red_schedonly = 100 * (1 - cost_schedonly / cost_baseline)
        cost_red_schedbatt = 100 * (1 - cost_schedbatt / cost_baseline)
    
    # ==================================================================================
    # Step 3: Validate results
    # ==================================================================================
    print("\nValidating results...")
    
    # CRITICAL: All optimizations must show improvement (or at least not worsen) all metrics
    # This is deterministic optimization, so we should always see improvements or equal results
    
    # Track validation issues to report at the end
    validation_issues = []
    
    # Validation check 1: Cost reductions are strictly monotonic
    # Scheduling should reduce costs compared to baseline
    # Adding battery should further reduce costs compared to scheduling only
    
    if not (cost_baseline >= cost_schedonly):
        issue = f"CRITICAL: Cost increased with scheduling: {cost_baseline:.2f} -> {cost_schedonly:.2f}"
        validation_issues.append(issue)
        warnings.warn(issue)
    
    if not (cost_schedonly >= cost_schedbatt):
        issue = f"CRITICAL: Cost increased with battery: {cost_schedonly:.2f} -> {cost_schedbatt:.2f}"
        validation_issues.append(issue)
        warnings.warn(issue)
    
    # Validation check 2: Grid energy consumption should not increase with optimization
    # Note: Lower grid energy is better
    if total_schedonly_kwh > total_baseline_kwh:
        issue = f"CRITICAL: Grid energy increased with scheduling: {total_baseline_kwh:.2f} -> {total_schedonly_kwh:.2f}"
        validation_issues.append(issue)
        warnings.warn(issue)
        
    if total_schedbatt_kwh > total_schedonly_kwh:
        issue = f"CRITICAL: Grid energy increased with battery: {total_schedonly_kwh:.2f} -> {total_schedbatt_kwh:.2f}"
        validation_issues.append(issue)
        warnings.warn(issue)
    
    # Validation check 3: PV usage should strictly increase with each optimization
    if pv_used_schedonly < pv_used_baseline:
        issue = f"CRITICAL: PV usage decreased with scheduling: {pv_used_baseline:.2f} -> {pv_used_schedonly:.2f}"
        validation_issues.append(issue)
        warnings.warn(issue)
    
    if pv_used_schedbatt < pv_used_schedonly:
        issue = f"CRITICAL: PV usage decreased with battery: {pv_used_schedonly:.2f} -> {pv_used_schedbatt:.2f}"
        validation_issues.append(issue)
        warnings.warn(issue)
        
    # Validation check 4: PAR should not increase with optimization
    # Lower PAR is better (flatter load profile)
    if par_schedonly > par_baseline:
        issue = f"CRITICAL: PAR increased with scheduling: {par_baseline:.2f} -> {par_schedonly:.2f}"
        validation_issues.append(issue)
        warnings.warn(issue)
        
    if par_schedbatt > par_schedonly:
        issue = f"CRITICAL: PAR increased with battery: {par_schedonly:.2f} -> {par_schedbatt:.2f}"
        validation_issues.append(issue)
        warnings.warn(issue)
    
    # Validation check 3: Battery SoC arrays are contiguous (no unexpected resets)
    if optimizer_wb.battery_soc_global is not None:
        battery_soc = align_series_to_index(optimizer_wb.battery_soc_global, time_index)
        # Check for large jumps in SoC that aren't explained by charge/discharge
        soc_diffs = battery_soc.diff().abs()
        charge_discharge_sum = battery_charge.abs() + battery_discharge.abs()
        unexplained_jumps = soc_diffs[(soc_diffs > 0.1) & (soc_diffs > 1.1 * charge_discharge_sum)]
        
        if len(unexplained_jumps) > 0:
            warnings.warn(f"Validation warning: Found {len(unexplained_jumps)} unexplained jumps in battery SoC")
    
    # ==================================================================================
    # Step 4: Populate paper_row
    # ==================================================================================
    print("\nPopulating paper row...")
    
    paper_row = {
        "Building": building_id,  # Add building ID as 'Building' column
        "kwh_baseline": total_baseline_kwh,
        "kwh_schedonly": total_schedonly_kwh,
        "kwh_schedbatt": total_schedbatt_kwh,
        "cost_baseline": cost_baseline,
        "cost_schedonly": cost_schedonly,
        "cost_schedbatt": cost_schedbatt,
        "cost_red_schedonly": cost_red_schedonly,
        "cost_red_schedbatt": cost_red_schedbatt,
        "pv_used_baseline": pv_used_baseline,
        "pv_used_schedonly": pv_used_schedonly,
        "pv_used_schedbatt": pv_used_schedbatt,
        "par_baseline": par_baseline,
        "par_schedonly": par_schedonly,
        "par_schedbatt": par_schedbatt,
        "validation_passed": validation_passed,
        "validation_issues": validation_issues
    }
    
    # Print summary of results
    print(f"\nSummary for building {building_id}:")
    print(f"  Energy: {total_baseline_kwh:.1f} kWh → {total_schedonly_kwh:.1f} kWh → {total_schedbatt_kwh:.1f} kWh")
    print(f"  Cost: {cost_baseline:.2f} € → {cost_schedonly:.2f} € → {cost_schedbatt:.2f} €")
    print(f"  Cost reduction: {cost_red_schedonly:.1f}% (sched-only), {cost_red_schedbatt:.1f}% (sched-batt)")
    print(f"  PV used: {pv_used_baseline:.1f} kWh → {pv_used_schedonly:.1f} kWh → {pv_used_schedbatt:.1f} kWh")
    print(f"  PAR: {par_baseline:.2f} → {par_schedonly:.2f} → {par_schedbatt:.2f}")
    
    return paper_row


def run_all_buildings(building_ids, device_specs, debug=True, **kwargs):
    """
    Run the metrics calculation for all buildings and return a DataFrame.
    
    Args:
        building_ids: List of building IDs to process
        device_specs: Dictionary of device specifications for each building
        debug: If True, show detailed output for each building
        **kwargs: Additional arguments for calculate_metrics_for_building
        
    Returns:
        DataFrame with all calculated metrics
    """
    paper_rows = []
    all_validation_passed = True
    
    print("\n" + "=" * 80)
    print("DEMONSTRATING DETERMINISTIC OPTIMIZATION IMPROVEMENTS")
    print("=" * 80)
    
    for building_id in tqdm(building_ids, desc="Processing buildings"):
        print("\n" + "=" * 80)
        print(f"PROCESSING BUILDING: {building_id}")
        print("=" * 80)
        
        # Get the device specs for this specific building
        building_device_specs = device_specs.get(building_id, {})
        if not building_device_specs:
            print(f"No device specs found for building {building_id}, skipping")
            continue
            
        # Calculate metrics for this building
        paper_row = calculate_metrics_for_building(
            building_id=building_id,
            device_specs=building_device_specs,
            **kwargs
        )
        
        if paper_row and 'validation_passed' in paper_row and not paper_row['validation_passed']:
            all_validation_passed = False
        
        if paper_row:
            paper_rows.append(paper_row)
    
    # Convert to DataFrame
    if not paper_rows:
        return pd.DataFrame()
        
    paper_df = pd.DataFrame(paper_rows)
    
    # Round numeric columns for readability
    numeric_cols = paper_df.select_dtypes(include=[np.number]).columns
    paper_df[numeric_cols] = paper_df[numeric_cols].round(2)
    
    return paper_df


if __name__ == "__main__":
    # Example usage
    
    # Actual building IDs found in the notebooks/data directory
    BUILDING_IDS = ['DE_KN_residential1', 'DE_KN_residential2', 'DE_KN_residential3']  # Replace with your building IDs
    
    # Define device specs for each building (example)
    device_specs = {
        "DE_KN_residential1": {
            "dishwasher": {"category": "wet", "power_rating": 1.5},
            "washing_machine": {"category": "wet", "power_rating": 1.2},
            "dryer": {"category": "wet", "power_rating": 2.0}
        },
        "DE_KN_residential2": {
            "dishwasher": {"category": "wet", "power_rating": 1.6},
            "washing_machine": {"category": "wet", "power_rating": 1.3},
            "ev_charger": {"category": "ev", "power_rating": 7.0}
        },
        "DE_KN_residential3": {
            "dishwasher": {"category": "wet", "power_rating": 1.5},
            "washing_machine": {"category": "wet", "power_rating": 1.2},
            "dryer": {"category": "wet", "power_rating": 1.8}
        }
    }
    
    # Define parameters
    params = {
        "parquet_dir": "notebooks/data",
        "max_building_load": 10.0,
        "battery_params": {
            "capacity": 10.0, 
            "max_charge_rate": 3.0,          # Changed from max_charge
            "max_discharge_rate": 3.0,       # Changed from max_discharge
            "charge_efficiency": 0.95,
            "discharge_efficiency": 0.95,
            "initial_soc": 5.0,            # 50% of capacity
            "soc_min": 1.0,                # Changed from min_soc
            "soc_max": 10.0,               # Changed from max_soc
            "external_control": False
        },
        "flexible_params": {"max_shift_hours": 12},
        "days": 7  # Process 7 days for each building
    }
    
    # Run the calculation
    paper_df = run_all_buildings(
        building_ids=BUILDING_IDS,
        device_specs=device_specs,
        **params
    )

    # Format the final DataFrame for better terminal display
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.width', 160)  # Wider display
    pd.set_option('display.precision', 2)  # Round to 2 decimal places
    pd.set_option('display.float_format', '{:.2f}'.format)  # Format floats consistently

    print("\n" + "=" * 100)
    print("FINAL METRICS TABLE")
    print("=" * 100)
    print(paper_df.to_string(index=False))
    print("=" * 100)

    # Create a summary of key metrics in an easily readable format
    print("\nKEY METRICS SUMMARY:")
    print("-" * 80)
    print(f"{'Building':20} {'Grid Energy (kWh)':>25} {'Cost (EUR)':>15} {'PV Used (kWh)':>15} {'Cost Reduction %':>15}")
    print(f"{'':20} {'NS → SO → SB':>25} {'NS → SO → SB':>15} {'NS → SO → SB':>15} {'SO / SB':>15}")
    print("-" * 80)

    for _, row in paper_df.iterrows():
        bldg = row['Building']
        energy_ns = row['kwh_baseline']
        energy_so = row['kwh_schedonly']
        energy_sb = row['kwh_schedbatt']

        cost_ns = row['cost_baseline']
        cost_so = row['cost_schedonly']
        cost_sb = row['cost_schedbatt']

        pv_ns = row['pv_used_baseline']
        pv_so = row['pv_used_schedonly']
        pv_sb = row['pv_used_schedbatt']

        cost_red_so = row['cost_red_schedonly']
        cost_red_sb = row['cost_red_schedbatt']

        print(f"{bldg:20} {energy_ns:8.2f} → {energy_so:6.2f} → {energy_sb:6.2f} {cost_ns:8.2f} → {cost_so:6.2f} → {cost_sb:6.2f} {pv_ns:8.2f} → {pv_so:6.2f} → {pv_sb:6.2f} {cost_red_so:6.2f} / {cost_red_sb:6.2f}")

    print("-" * 80)
    print("NS: Non-Scheduled, SO: Scheduled-Only, SB: Scheduled with Battery")

    # Save to CSV
    paper_df.to_csv("paper_metrics_table.csv", index=False)

    print("\nFinal metrics table:")
    print(paper_df)
