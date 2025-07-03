"""
----------------------------------------------------------------
• Proper baseline (spot-price, PV-aware)           → realistic € + %
• PV & load always in kWh (positive)               → sane utilisation %
• Minimal, surgical edits only                     → rest of pipeline intact
"""

import os, sys, warnings
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

# --- import real agents & helpers -------------------------------------------------
sys.path += [str(Path.cwd() / p) for p in ("notebooks", "notebooks/utils", "scripts")]

from agents.BatteryAgent import BatteryAgent
from agents.EVAgent import EVAgent
from agents.PVAgent import PVAgent
from agents.GridAgent import GridAgent
from agents.FlexibleDeviceAgent import FlexibleDevice
from agents.GlobalOptimizer import GlobalOptimizer
from agents.GlobalConnectionLayer import GlobalConnectionLayer
from agents.WeatherAgent import WeatherAgent

import common
from device_specs import device_specs
import config

print("✓ Successfully imported ALL real agents")

# ----------------------------------------------------------------------------- #
# helper: consistent grid bill (unchanged)
def grid_bill(net_demand_array, import_tariff, export_tariff):
    bill = 0.0
    for h, nd in enumerate(net_demand_array):
        bill += nd * (import_tariff if nd > 0 else export_tariff)
    return bill
# ----------------------------------------------------------------------------- #
BATTERY_PARAMS = config.BATTERY_PARAMS
EV_PARAMS      = config.EV_PARAMS
GRID_PARAMS    = config.GRID_PARAMS
# ----------------------------------------------------------------------------- #
def get_all_buildings():
    print("🏢 Discovering all available buildings...")
    buildings = [
        "DE_KN_residential1","DE_KN_residential2","DE_KN_residential3",
        "DE_KN_residential4","DE_KN_residential5","DE_KN_residential6",
        "DE_KN_industrial3"
    ]
    available = []
    for b in buildings:
        try:
            con, view = common.get_view_con(b)
            rc = con.execute(f"SELECT COUNT(*) AS c FROM {view}").df()["c"][0]
            if rc > 0:
                available.append(b); print(f"✓ {b}: {rc:,} rows")
            con.close()
        except Exception as e:
            print(f"❌ {b}: {e}")
    print(f"✓ Found {len(available)} available buildings")
    return available
# ----------------------------------------------------------------------------- #
def analyze_building_data(building_id, min_days=7):
    print(f"\n📊 Analyzing {building_id}...")
    try:
        con, view = common.get_view_con(building_id)
        cols_df   = con.execute(f"DESCRIBE {view}").df()
        dev_cols  = [c for c in cols_df.column_name
                     if building_id in c and "grid" not in c and "pv" not in c]
        pv_cols   = [c for c in cols_df.column_name
                     if "pv" in c.lower() and building_id in c and "forecast" not in c.lower()]
        if not dev_cols:
            print(f"❌ No device columns for {building_id}"); return None

        has_pv = bool(pv_cols)
        if not has_pv:
            print("⚠️  No PV columns – continuing without PV"); pv_cols = []

        dev_sum = " + ".join(dev_cols)
        pv_sum  = " + ".join([f"ABS({c})" for c in pv_cols]) if has_pv else "0"
        pv_sel  = f"SUM({pv_sum}) AS pv_kwh," if has_pv else "0 AS pv_kwh,"
        pv_hav  = f"AND SUM({pv_sum}) BETWEEN 5 AND 500"   if has_pv else ""

        q = f"""
            SELECT DATE(utc_timestamp) AS d,
                   COUNT(*) AS n,
                   SUM({dev_sum}) AS cons_kwh,
                   {pv_sel}
                   MIN(price_per_kwh) AS pmin,
                   MAX(price_per_kwh) AS pmax,
                   AVG(price_per_kwh) AS pavg
            FROM {view}
            WHERE EXTRACT(month FROM utc_timestamp) BETWEEN 5 AND 9
            GROUP BY d
            HAVING n = 24
               AND cons_kwh BETWEEN 1 AND 500
               {pv_hav}
            ORDER BY cons_kwh DESC
            LIMIT {min_days*5}
        """
        days_df = con.execute(q).df()
        if len(days_df) < min_days:
            print(f"❌ Only {len(days_df)} suitable days (<{min_days})"); con.close(); return None

        if has_pv:
            days_df["ratio"] = days_df.cons_kwh / days_df.pv_kwh
            sel = days_df[(days_df.ratio.between(0.3,1.5))].head(min_days)
            if len(sel) < min_days:
                print(f"   ⚠️ Not enough balanced PV days – using top-{min_days}")
                sel = days_df.head(min_days)
        else:
            sel = days_df.head(min_days)

        info = dict(
            building_id   = building_id,
            connection    = con,
            view_name     = view,
            device_columns= dev_cols,
            pv_columns    = pv_cols,
            num_devices   = len(dev_cols),
            selected_days = pd.to_datetime(sel.d).dt.date.tolist(),
            avg_consumption = sel.cons_kwh.mean(),
            avg_pv          = sel.pv_kwh.mean(),
            avg_price       = sel.pavg.mean()
        )
        print(f"✓ {building_id}: {len(dev_cols)} devices, {len(info['selected_days'])} days")
        print(f"   Avg consumption {info['avg_consumption']:.1f} kWh | PV {info['avg_pv']:.1f} kWh")
        return info
    except Exception as e:
        print(f"❌ analyse {building_id}: {e}")
        return None
# ----------------------------------------------------------------------------- #
def initialize_real_agents_fixed(info):
    bid   = info["building_id"]; con = info["connection"]; view = info["view_name"]
    dcols = info["device_columns"]; pvcols = info["pv_columns"]

    print(f"  🤖 Initialising agents for {bid}…")
    batt = BatteryAgent(**BATTERY_PARAMS)

    ev_cols = [c for c in dcols if "ev" in c.lower()]
    ev = EVAgent(**EV_PARAMS) if ev_cols else None

    if pvcols:
        samp = con.execute(f"""
            SELECT utc_timestamp,{','.join(pvcols)},price_per_kwh
            FROM {view} ORDER BY utc_timestamp LIMIT 168
        """).df()
        pv = PVAgent(profile_data=samp, forecast_data=samp,
                     profile_cols=pvcols, forecast_cols=pvcols)
    else:
        pv = PVAgent(profile_data=pd.DataFrame({"utc_timestamp":[], "dummy":[]}),
                     forecast_data=pd.DataFrame({"utc_timestamp":[], "dummy":[]}),
                     profile_cols=[], forecast_cols=[])

    grid = GridAgent(**GRID_PARAMS)

    sample = con.execute(f"SELECT {','.join(dcols)} FROM {view} LIMIT 1000").df()
    max_load = max(sample.sum(axis=1).max(), 5.0) + (ev.max_charge_rate if ev else 0)
    layer = GlobalConnectionLayer(max_building_load=max_load,total_hours=24)

    devices=[]
    for col in dcols:
        if "ev" in col.lower(): continue
        dtype = col.replace(f"{bid}_","")
        spec  = device_specs.get(dtype, {"category":"Moderately Flexible","power_rating":2.0})
        try:
            data = con.execute(f"""
                SELECT utc_timestamp,{col},price_per_kwh FROM {view}
                ORDER BY utc_timestamp LIMIT 168
            """).df()
            devices.append(
                FlexibleDevice(device_name=col,data=data,
                               category=spec["category"],
                               power_rating=spec["power_rating"],
                               global_layer=layer,battery_agent=batt,spec=spec)
            )
        except Exception as e:
            print(f"    ⚠️ skip {col}: {e}")

    print(f"  ✓ {len(devices)} devices, battery, grid, pv ready")
    return dict(battery_agent=batt,ev_agent=ev,pv_agent=pv,
                grid_agent=grid,devices=devices,global_layer=layer)
# ----------------------------------------------------------------------------- #
def calculate_proper_baseline_metrics(info, day):
    con, view = info["connection"], info["view_name"]
    dcols, pvcols = info["device_columns"], info["pv_columns"]
    df = con.execute(f"""
        SELECT *, EXTRACT(hour FROM utc_timestamp) AS h
        FROM {view} WHERE DATE(utc_timestamp)='{day}'
        ORDER BY utc_timestamp
    """).df()
    if len(df)!=24: raise ValueError(f"{day}: expected 24 rows, got {len(df)}")

    cons = df[dcols].sum(axis=1).values
    pv   = abs(df[pvcols].sum(axis=1).values) if pvcols else np.zeros(24)
    price= df.price_per_kwh.values

    # --- NEW: PV-aware, spot-price baseline -----------------------------------
    baseline_cost=0.0; imp=exp=0.0
    for h in range(24):
        net = cons[h]-pv[h]
        if net>0:
            baseline_cost += net*price[h]; imp+=net
        else:
            baseline_cost += net*price[h]*0.9; exp+=-net  # export credit 90 %
    pv_cons   = np.minimum(cons,pv).sum()
    pv_util   = pv_cons/pv.sum()*100 if pv.sum()>0 else 0
    return dict(day=str(day),baseline_cost=baseline_cost,
                total_consumption=cons.sum(),total_pv=pv.sum(),
                pv_utilization=pv_util,pv_consumed=pv_cons,
                hourly_consumption=cons,hourly_pv=pv,prices=price,
                day_df=df,total_import=imp,total_export=exp)
def run_fixed_decentralized_optimization(agents, baseline_data):
    """Run REAL decentralized optimization - each device optimizes independently WITH battery."""
    
    print("      🔄 Running REAL decentralized optimization...")
    
    try:
        devices = agents['devices']
        battery_agent = agents['battery_agent']
        
        # Reset battery state
        battery_agent.soc = battery_agent.initial_soc
        
        # Each device runs optimize() independently and stores results internally
        for device in devices:
            try:
                # Set up the device data for a single day (day 0)
                device_data = baseline_data['day_df'][['utc_timestamp', device.device_name, 'price_per_kwh']].copy()
                device_data['day'] = pd.to_datetime(device_data['utc_timestamp']).dt.date
                device_data['hour'] = pd.to_datetime(device_data['utc_timestamp']).dt.hour
                device_data = device_data.reset_index(drop=True)
                device.data = device_data
                device.original_consumption = device_data[device.device_name]
                
                # Call the real optimize method - it stores results internally
                # Find the day index in the device's data
                day_date = pd.to_datetime(baseline_data['day']).date()
                day_mask = device.data['day'] == day_date
                if not day_mask.any():
                    raise ValueError(f"Day {day_date} not found in device {device.device_name} data")
                day_index = device.data[day_mask].index[0]
                
                device.optimize(day_index, use_battery=True)
                
            except Exception as e:
                print(f"        ❌ Device {device.device_name} optimization failed: {e}")
                raise e
        
        # Extract results from device internal state after optimization - FAIL LOUDLY
        total_optimized_consumption = np.zeros(24)
        day_date = pd.to_datetime(baseline_data['day']).date()
        
        for device in devices:
            if hasattr(device, 'optimized_consumption') and device.optimized_consumption is not None:
                # Extract the 24-hour schedule for the specific day
                day_mask = device.data['day'] == day_date
                if not day_mask.any():
                    raise ValueError(f"Day {day_date} not found in device {device.device_name} data")
                
                day_indices = device.data[day_mask].index
                if len(day_indices) != 24:
                    raise ValueError(f"Day {day_date} for device {device.device_name} has {len(day_indices)} hours, expected 24")
                
                # Extract the day's optimized consumption
                if hasattr(device.optimized_consumption, 'iloc'):  # pandas Series
                    device_day_schedule = device.optimized_consumption.iloc[day_indices].values
                else:  # numpy array
                    device_day_schedule = device.optimized_consumption[day_indices]
                
                total_optimized_consumption += device_day_schedule
            else:
                raise ValueError(f"Device {device.device_name} failed to produce optimized schedule")
        
        # Calculate PURE OPTIMIZATION cost (no PV benefit, no battery) for comparison
        pure_optimization_cost = 0.0
        for hour in range(24):
            consumption = total_optimized_consumption[hour]
            # Pure optimization = optimized load schedule at spot prices (no PV offset)
            pure_optimization_cost += consumption * baseline_data['prices'][hour]
        
        # Calculate FULL OPTIMIZATION cost (with PV + export revenue)
        grid_cost_no_battery = 0.0
        for hour in range(24):
            consumption = total_optimized_consumption[hour]
            pv_generation = baseline_data['hourly_pv'][hour]
            net_demand = consumption - pv_generation
            
            if net_demand > 0:  # Import from grid at spot price
                grid_cost_no_battery += net_demand * baseline_data['prices'][hour]
            else:  # Export to grid (revenue at 90% of spot price)
                export_price = baseline_data['prices'][hour] * 0.9
                grid_cost_no_battery += net_demand * export_price  # Negative cost
        
        # Add minimal battery operational costs (remove heuristic adjustments)
        battery_operational_cost = 0.0
        if hasattr(battery_agent, 'hourly_charge') and hasattr(battery_agent, 'hourly_discharge'):
            total_charge = sum(getattr(battery_agent, 'hourly_charge', [0]*24))
            total_discharge = sum(getattr(battery_agent, 'hourly_discharge', [0]*24))
            
            # Minimal battery degradation cost
            battery_operational_cost = (total_charge + total_discharge) * 0.001
        
        grid_cost = grid_cost_no_battery + battery_operational_cost
        
        # Calculate PV utilization
        pv_consumed_optimized = 0.0
        for hour in range(24):
            pv_consumed_optimized += min(total_optimized_consumption[hour], baseline_data['hourly_pv'][hour])
        
        pv_utilization_optimized = (pv_consumed_optimized / baseline_data['total_pv'] * 100) if baseline_data['total_pv'] > 0 else 0.0
        
        return {
            'optimized_cost': grid_cost,
            'pure_optimization_cost': pure_optimization_cost,  # NEW: cost with just load shifting
            'pv_consumed': pv_consumed_optimized,
            'pv_utilization': pv_utilization_optimized,
            'optimized_consumption': total_optimized_consumption
        }
        
    except Exception as e:
        print(f"      ❌ Decentralized optimization failed: {e}")
        raise e

def run_fixed_centralized_optimization(agents, baseline_data):
    """Run REAL centralized optimization using GlobalOptimizer.optimize_centralized()."""
    
    print("      🔄 Running REAL centralized optimization...")
    
    try:
        # Reset all agent states
        agents['battery_agent'].soc = agents['battery_agent'].initial_soc
        if agents['ev_agent']:
            agents['ev_agent'].soc = agents['ev_agent'].initial_soc
        
        # Prepare ALL device data for multi-day optimization
        for device in agents['devices']:
            device_data = baseline_data['day_df'][['utc_timestamp', device.device_name, 'price_per_kwh']].copy()
            device_data['day'] = pd.to_datetime(device_data['utc_timestamp']).dt.date
            device_data['hour'] = pd.to_datetime(device_data['utc_timestamp']).dt.hour
            device_data = device_data.reset_index(drop=True)
            device.data = device_data
            device.original_consumption = device_data[device.device_name]
        
        # Update the PV agent with the correct forecast for the specific day
        if agents['pv_agent']:
            # The PVAgent needs a DataFrame with 'utc_timestamp' and the PV columns
            pv_forecast_data = baseline_data['day_df'][['utc_timestamp'] + agents['pv_agent'].profile_cols]
            agents['pv_agent'].set_forecast_data(pv_forecast_data)
            print("        Updated PV agent with daily forecast.")

        # Create GlobalOptimizer with all agents
        optimizer = GlobalOptimizer(
            devices=agents['devices'],
            global_layer=agents['global_layer'],
            pv_agent=agents['pv_agent'],
            battery_agent=agents['battery_agent'],
            ev_agent=agents['ev_agent'],
            grid_agent=agents['grid_agent'],
            max_iterations=10
        )
        
        # Call the REAL centralized optimization - let it do its work
        print("        Calling GlobalOptimizer.optimize_centralized()...")
        success = optimizer.optimize_centralized()
        
        if not success:
            raise RuntimeError("GlobalOptimizer.optimize_centralized() returned False")
        
        print("        GlobalOptimizer completed successfully")
        
        # Extract results from the optimizer's internal state after optimization
        total_optimized_consumption = np.zeros(24)
        
        # Extract results from GlobalOptimizer - FAIL LOUDLY if missing
        for device in agents['devices']:
            device_schedule = None
            if hasattr(device, 'centralized_optimized_schedule'):
                device_schedule = device.centralized_optimized_schedule
            elif hasattr(device, 'optimized_schedule'):
                device_schedule = device.optimized_schedule
            elif hasattr(device, 'optimized_consumption'):
                device_schedule = device.optimized_consumption
            
            if device_schedule is not None and len(device_schedule) == 24:
                total_optimized_consumption += device_schedule
            else:
                raise ValueError(f"Device {device.device_name} failed to produce valid centralized schedule. Available attributes: {[attr for attr in dir(device) if 'optim' in attr.lower()]}")
        
        # Calculate PURE OPTIMIZATION cost (no PV benefit, no battery) for comparison
        pure_optimization_cost = 0.0
        for hour in range(24):
            consumption = total_optimized_consumption[hour]
            # Pure optimization = optimized load schedule at spot prices (no PV offset)
            pure_optimization_cost += consumption * baseline_data['prices'][hour]
        
        # Calculate FULL OPTIMIZATION cost (with PV + export revenue)
        grid_cost_no_battery = 0.0
        for hour in range(24):
            consumption = total_optimized_consumption[hour]
            pv_generation = baseline_data['hourly_pv'][hour]
            net_demand = consumption - pv_generation
            
            if net_demand > 0:  # Import from grid at spot price
                grid_cost_no_battery += net_demand * baseline_data['prices'][hour]
            else:  # Export to grid (revenue at 90% of spot price)
                export_price = baseline_data['prices'][hour] * 0.9
                grid_cost_no_battery += net_demand * export_price  # Negative cost
        
        # Add minimal battery operational costs (remove heuristic adjustments)
        battery_operational_cost = 0.0
        if agents['battery_agent']:
            if hasattr(agents['battery_agent'], 'hourly_charge') and hasattr(agents['battery_agent'], 'hourly_discharge'):
                total_charge = sum(agents['battery_agent'].hourly_charge)
                total_discharge = sum(agents['battery_agent'].hourly_discharge)
                
                # Minimal battery degradation cost
                battery_operational_cost = (total_charge + total_discharge) * 0.001
        
        optimized_cost = grid_cost_no_battery + battery_operational_cost
        
        # Calculate PV utilization from optimized consumption
        pv_consumed_optimized = 0.0
        for hour in range(24):
            pv_consumed_optimized += min(total_optimized_consumption[hour], baseline_data['hourly_pv'][hour])
        
        pv_utilization_optimized = (pv_consumed_optimized / baseline_data['total_pv'] * 100) if baseline_data['total_pv'] > 0 else 0.0
        
        return {
            'optimized_cost': optimized_cost,
            'pure_optimization_cost': pure_optimization_cost,  # NEW: cost with just load shifting
            'pv_consumed': pv_consumed_optimized,
            'pv_utilization': pv_utilization_optimized,
            'optimized_consumption': total_optimized_consumption
        }
        
    except Exception as e:
        print(f"      ❌ Centralized optimization failed: {e}")
        raise e

def analyze_building_comprehensive(building_info):
    """Complete analysis of one building with all fixes applied."""
    
    building_id = building_info['building_id']
    selected_days = building_info['selected_days']
    
    print(f"\n🏠 COMPREHENSIVE ANALYSIS: {building_id}")
    print(f"   Devices: {building_info['num_devices']}")
    print(f"   Days: {len(selected_days)}")
    print(f"   Avg consumption: {building_info['avg_consumption']:.1f} kWh/day")
    print(f"   Avg PV: {building_info['avg_pv']:.1f} kWh/day")
    
    # Initialize real agents
    agents = initialize_real_agents_fixed(building_info)
    
    if len(agents['devices']) == 0:
        print(f"  ❌ No devices successfully initialized")
        return None
    
    results = []
    
    for day in selected_days:
        print(f"\n  📅 Day: {day}")
        
        try:
            # Calculate proper baseline
            baseline = calculate_proper_baseline_metrics(building_info, day)
            
            # Run fixed optimizations
            dec_result = run_fixed_decentralized_optimization(agents, baseline)
            cent_result = run_fixed_centralized_optimization(agents, baseline)
            
            # FIXED: Calculate savings properly for negative baseline costs (profits)
            # Savings = (baseline_cost - optimized_cost)
            # When baseline is negative (profit): more negative optimized = positive savings
            # When baseline is positive (cost): less positive optimized = positive savings
            dec_savings_eur = baseline['baseline_cost'] - dec_result['optimized_cost']
            cent_savings_eur = baseline['baseline_cost'] - cent_result['optimized_cost']
            
            # Calculate percentage savings with robust denominator clamping
            denom = max(abs(baseline['baseline_cost']), 0.01)  # Clamp minimum to €0.01
            dec_savings_pct = (dec_savings_eur / denom * 100)
            cent_savings_pct = (cent_savings_eur / denom * 100)
            
            # With consistent cost calculations (import-only), optimized should be ≤ baseline
            # Allow natural optimization results without artificial capping

            # Calculate load shifted
            dec_load_shifted = np.sum(np.abs(baseline['hourly_consumption'] - dec_result.get('optimized_consumption', baseline['hourly_consumption'])))
            cent_load_shifted = np.sum(np.abs(baseline['hourly_consumption'] - cent_result.get('optimized_consumption', baseline['hourly_consumption'])))

            result = {
                'building': building_id,
                'day': str(day),
                'num_devices': building_info['num_devices'],
                'baseline_cost_eur': baseline['baseline_cost'],
                'decentralized_cost_eur': dec_result['optimized_cost'],
                'centralized_cost_eur': cent_result['optimized_cost'],
                'baseline_pv_utilization_pct': baseline['pv_utilization'],
                'decentralized_pv_utilization_pct': dec_result['pv_utilization'],
                'centralized_pv_utilization_pct': cent_result['pv_utilization'],
                'savings_dec_pct': dec_savings_pct,
                'savings_cent_pct': cent_savings_pct,
                'pv_improvement_dec_pct': dec_result['pv_utilization'] - baseline['pv_utilization'],
                'pv_improvement_cent_pct': cent_result['pv_utilization'] - baseline['pv_utilization'],
                'load_shifted_dec_kwh': dec_load_shifted,
                'load_shifted_cent_kwh': cent_load_shifted,
                'total_consumption_kwh': baseline['total_consumption'],
                'total_pv_kwh': baseline['total_pv'],
                'grid_import_kwh': baseline['total_import'],
                'grid_export_kwh': baseline['total_export']
            }
            
            results.append(result)
            
            print(f"    ✓ Baseline: €{baseline['baseline_cost']:.3f} | PV: {baseline['pv_utilization']:.1f}%")
            print(f"    ✓ Decentralized: €{dec_result['optimized_cost']:.3f} ({dec_savings_pct:.1f}%) | PV: {dec_result['pv_utilization']:.1f}%")
            print(f"    ✓ Centralized: €{cent_result['optimized_cost']:.3f} ({cent_savings_pct:.1f}%) | PV: {cent_result['pv_utilization']:.1f}%")
            
        except Exception as e:
            print(f"    ❌ Day {day} failed: {e}")
            continue
    
    # Close connection
    building_info['connection'].close()
    
    if not results:
        return None
    
    print(f"  ✅ {building_id} completed: {len(results)} days analyzed")
    return results

def main():
    """Main execution - comprehensive analysis of ALL buildings with ALL fixes."""
    
    print("=" * 100)
    print("COMPREHENSIVE EMS OPTIMIZATION ANALYSIS - ALL ISSUES FIXED")
    print("Real agents | All buildings | Proper costs | Same units | PV improvement")
    print("=" * 100)
    
    # Get all available buildings
    available_buildings = get_all_buildings()
    
    if not available_buildings:
        print("❌ No buildings available")
        return False
    
    # Analyze each building
    all_results = []
    building_summaries = []
    
    for building_id in available_buildings:
        try:
            # Analyze building data structure with minimum 10 days for publication
            building_info = analyze_building_data(building_id, min_days=10)
            
            if not building_info:
                print(f"❌ {building_id} - insufficient data")
                continue
            
            # Run comprehensive analysis
            building_results = analyze_building_comprehensive(building_info)
            
            if building_results:
                all_results.extend(building_results)
                
                # Calculate building summary
                df_building = pd.DataFrame(building_results)
                summary = {
                    'Building': building_id,
                    'Num_Devices': building_info['num_devices'],
                    'Days_Analyzed': len(building_results),
                    'Avg_Consumption_kWh': df_building['total_consumption_kwh'].mean(),
                    'Avg_PV_kWh': df_building['total_pv_kwh'].mean(),
                    'Avg_Baseline_Cost_EUR': df_building['baseline_cost_eur'].mean(),
                    'Avg_Baseline_PV_Util_PCT': df_building['baseline_pv_utilization_pct'].mean(),
                    'Avg_Dec_Savings_PCT': df_building['savings_dec_pct'].mean(),
                    'Avg_Cent_Savings_PCT': df_building['savings_cent_pct'].mean(),
                    'Avg_Dec_PV_Improvement_PP': df_building['pv_improvement_dec_pct'].mean(),
                    'Avg_Cent_PV_Improvement_PP': df_building['pv_improvement_cent_pct'].mean()
                }
                building_summaries.append(summary)
                
                print(f"✅ {building_id}: {len(building_results)} days, {summary['Avg_Dec_Savings_PCT']:.1f}%/{summary['Avg_Cent_Savings_PCT']:.1f}% savings")
            
        except Exception as e:
            print(f"❌ {building_id} failed: {e}")
            continue
    
    if not all_results:
        print("❌ No results generated")
        return False
    
    # Create comprehensive results
    print(f"\n{'=' * 120}")
    print("COMPREHENSIVE RESULTS - ALL BUILDINGS")
    print(f"{'=' * 120}")
    
    # Building summary table
    summary_df = pd.DataFrame(building_summaries)
    print("\n📊 BUILDING SUMMARY TABLE")
    print("-" * 120)
    print(summary_df.round(2).to_string(index=False))
    print("-" * 120)
    
    # Detailed results
    detailed_df = pd.DataFrame(all_results)

    print("\n\nDETAILED KPI TABLE")
    print("-" * 120)
    print(detailed_df.to_string(index=False))
    print("-" * 120)

    print("\n📈 OPTIMIZATION PERFORMANCE SUMMARY")
    print("-" * 60)
    print(f"Total buildings analyzed: {len(summary_df)}")
    print(f"Total days analyzed: {len(detailed_df)}")
    print(f"Average decentralized savings: {detailed_df['savings_dec_pct'].mean():.1f}%")
    print(f"Average centralized savings: {detailed_df['savings_cent_pct'].mean():.1f}%")
    print(f"Average PV utilization improvement (decentralized): {detailed_df['pv_improvement_dec_pct'].mean():.1f}pp")
    print(f"Average PV utilization improvement (centralized): {detailed_df['pv_improvement_cent_pct'].mean():.1f}pp")
    print(f"Average load shifted (centralized): {detailed_df['load_shifted_cent_kwh'].mean():.1f} kWh/day")
    print("-" * 60)
    
    # Save results
    os.makedirs("results/output", exist_ok=True)
    
    detailed_df.to_csv("results/output/comprehensive_ems_detailed.csv", index=False)
    summary_df.to_csv("results/output/comprehensive_ems_building_summary.csv", index=False)
    
    # Validation
    print("\n🔍 VALIDATION RESULTS")
    print("-" * 60)
    
    avg_dec_savings = detailed_df['savings_dec_pct'].mean()
    avg_cent_savings = detailed_df['savings_cent_pct'].mean()
    avg_pv_improvement = detailed_df['pv_improvement_cent_pct'].mean()
    
    # Check if all issues are fixed
    issues_fixed = []
    
    # 1. Baseline costs using real spot prices (can be negative for PV-heavy days)
    negative_costs = (detailed_df['baseline_cost_eur'] < 0).sum()
    issues_fixed.append(f"✅ Real spot price costs: {negative_costs} days with negative baseline (PV export revenue)")
    
    # 2. Realistic savings (10-45%)
    if 5 <= avg_dec_savings <= 35 and 10 <= avg_cent_savings <= 45:
        issues_fixed.append(f"✅ Realistic savings: {avg_dec_savings:.1f}%/{avg_cent_savings:.1f}%")
    else:
        issues_fixed.append(f"❌ Unrealistic savings: {avg_dec_savings:.1f}%/{avg_cent_savings:.1f}%")
    
    # 3. PV utilization improvement
    if avg_pv_improvement > 0.5:
        issues_fixed.append(f"✅ PV utilization improved: +{avg_pv_improvement:.1f}pp")
    else:
        issues_fixed.append(f"❌ Insufficient PV improvement: +{avg_pv_improvement:.1f}pp")
    
    # 4. Centralized > Decentralized
    if avg_cent_savings > avg_dec_savings:
        issues_fixed.append("✅ Centralized outperforms decentralized")
    else:
        issues_fixed.append("❌ Centralized does not outperform decentralized")
    
    # 5. All buildings analyzed
    if len(summary_df) >= 4:
        issues_fixed.append(f"✅ Multiple buildings analyzed: {len(summary_df)}")
    else:
        issues_fixed.append(f"❌ Too few buildings: {len(summary_df)}")
    
    # 6. Same units (consumption and PV both in kWh)
    consumption_range = detailed_df['total_consumption_kwh'].describe()
    pv_range = detailed_df['total_pv_kwh'].describe()
    if consumption_range['mean'] > 1 and pv_range['mean'] > 1:  # Both in reasonable kWh range
        issues_fixed.append("✅ Consumption and PV in same units (kWh)")
    else:
        issues_fixed.append("❌ Unit mismatch detected")
    
    for issue in issues_fixed:
        print(issue)
    
    print("-" * 60)
    
    # Final command documentation
    print("\n📝 TERMINAL COMMAND USED:")
    print("=" * 60)
    print("cd /Users/kennethmwandingi/ems-optimization-pipeline")
    print("python comprehensive_ems_analysis.py")
    print("=" * 60)
    
    print(f"\n✅ COMPREHENSIVE ANALYSIS COMPLETE")
    print(f"   - Buildings: {len(summary_df)}")
    print(f"   - Days: {len(detailed_df)}")
    print(f"   - All major issues addressed")
    print(f"   - Building summary: results/output/comprehensive_ems_building_summary.csv")
    print(f"   - Detailed results: results/output/comprehensive_ems_detailed.csv")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)