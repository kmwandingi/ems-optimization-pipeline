"""
KPI Calculation Module for EMS Optimization Pipeline

This module provides functions to calculate Key Performance Indicators (KPIs)
for different optimization strategies in the EMS pipeline.
"""

import sys
import os
import pandas as pd
import numpy as np
import duckdb
from typing import Dict, List, Optional, Union, Tuple

# Add the project root and notebooks directories to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
notebooks_dir = os.path.join(project_root, 'notebooks')
sys.path.insert(0, project_root)
sys.path.insert(0, notebooks_dir)  # The agents directory is inside notebooks

# Now we can import modules from the notebooks directory
from agents.BatteryAgent import BatteryAgent
from agents.EVAgent import EVAgent
from agents.PVAgent import PVAgent
from agents.GridAgent import GridAgent
from agents.FlexibleDeviceAgent import FlexibleDevice
from agents.GlobalConnectionLayer import GlobalConnectionLayer
from agents.GlobalOptimizer import GlobalOptimizer
import scripts.common as common  # DuckDB helper


def grid_cost(net_kwh, spot, grid):
    """
    Calculate grid cost based on net energy flow, spot prices, and grid parameters.
    
    Args:
        net_kwh: Net energy flow (positive = import, negative = export)
        spot: Spot prices
        grid: Grid agent with import/export price attributes
        
    Returns:
        float: Total cost
    """
    imp = net_kwh.clip(min=0)
    exp = (-net_kwh).clip(min=0)
    return float((imp * spot * grid.import_price - exp * grid.export_price).sum())


def calculate_kpis(building_id: str, 
                  n_days: int = 3,
                  use_battery: bool = True,
                  use_ev: bool = True,
                  battery_params: Optional[Dict] = None,
                  ev_params: Optional[Dict] = None,
                  grid_params: Optional[Dict] = None,
                  con: Optional[duckdb.DuckDBPyConnection] = None) -> pd.DataFrame:
    """
    Calculate KPIs for baseline, decentralized, and centralized optimization strategies.
    
    Args:
        building_id: ID of the building to analyze
        n_days: Number of days to include in analysis
        use_battery: Whether to include battery in the optimization
        use_ev: Whether to include EV in the optimization
        battery_params: Parameters for battery agent (if None, uses defaults)
        ev_params: Parameters for EV agent (if None, uses defaults)
        grid_params: Parameters for grid agent (if None, uses defaults)
        con: Optional DuckDB connection (if None, creates new connection)
        
    Returns:
        pd.DataFrame: DataFrame containing KPIs for each day and strategy
    """
    # Set default parameters if not provided
    if battery_params is None:
        battery_params = dict(
            max_charge_rate=3, max_discharge_rate=3, initial_soc=7,
            soc_min=1, soc_max=10, capacity=10,
            degradation_rate=1e-3, efficiency_charge=0.95,
            efficiency_discharge=0.95
        )
    
    if ev_params is None:
        ev_params = dict(
            capacity=60, initial_soc=12, soc_min=6, soc_max=54,
            max_charge_rate=7.4, max_discharge_rate=0,
            efficiency_charge=0.92, efficiency_discharge=0.92,
            must_be_full_by_hour=7
        )
    
    if grid_params is None:
        grid_params = dict(
            import_price=0.25, export_price=0.05,
            max_import=15, max_export=15
        )
    
    # Get DB connection and view if not provided
    if con is None:
        con, view_name = common.get_con(building_id=building_id)
    else:
        # If connection is provided but view_name is not, set it up
        view_name = f"{building_id}_processed_data"
    
    # Pull the raw data for the chosen days
    days = (con.execute(f"""
            SELECT DATE(utc_timestamp) AS d
            FROM   {view_name}
            GROUP  BY d HAVING COUNT(*) = 24
            ORDER  BY d
            LIMIT  {n_days}""")
            .fetchnumpy()['d'])
    
    df_all = (con.execute(f"""
              SELECT *, EXTRACT(hour FROM utc_timestamp) AS hour,
                       DATE(utc_timestamp) AS day
              FROM   {view_name}
              WHERE  DATE(utc_timestamp) IN ({','.join('?'*len(days))})
              ORDER  BY utc_timestamp""",
              list(days)).df())
    
    # any string → numeric, NaN → 0  (CRUCIAL for FlexibleDevice)
    for col in df_all.columns:
        if col not in ('utc_timestamp', 'day', 'hour'):
            df_all[col] = pd.to_numeric(df_all[col], errors='coerce').fillna(0.0)
    
    price_col = 'price_per_kwh'
    pv_cols = [c for c in df_all if 'pv' in c.lower() and 'forecast' not in c.lower()]
    load_cols = [c for c in df_all
                if c not in ('utc_timestamp', 'day', 'hour', price_col) + tuple(pv_cols)]
    
    # Instantiate "global" agents
    grid_agent = GridAgent(**grid_params)
    battery_agent = BatteryAgent(**battery_params) if use_battery else None
    ev_agent = EVAgent(**ev_params) if use_ev else None
    
    pv_agent = PVAgent(  # perfect forecast = actuals
        profile_data=df_all[['utc_timestamp', *pv_cols]],
        forecast_data=df_all[['utc_timestamp', *pv_cols]],
        profile_cols=pv_cols,
        forecast_cols=pv_cols)
    
    gcl = GlobalConnectionLayer(
        max_building_load=df_all[load_cols].sum(axis=1).max(),
        total_hours=len(df_all),
        export_price=grid_params['export_price'])
    
    # One FlexibleDevice per *load* column
    device_agents = []
    for col in load_cols:
        dev_df = df_all[['utc_timestamp', 'day', 'hour', price_col, col]].copy()
        dev_df[col] = pd.to_numeric(dev_df[col], errors='coerce').fillna(0.0)
        power_rating = float(dev_df[col].max())
        device_agents.append(
            FlexibleDevice(
                data=dev_df,
                device_name=col,
                category="Partially Flexible",
                power_rating=power_rating,
                global_layer=gcl,
                is_flexible=True,
                battery_agent=battery_agent,
                pv_agent=pv_agent
            )
        )
    
    # Simulation loop
    optimizer = GlobalOptimizer(
        devices=device_agents,
        global_layer=gcl,
        pv_agent=pv_agent,
        battery_agent=battery_agent,
        ev_agent=ev_agent,
        grid_agent=grid_agent
    )
    
    records = []
    for d in days:
        day_df = df_all[df_all.day == d].reset_index(drop=True)
        
        # -- baseline ----------------------------------------------------
        total_load = day_df[load_cols].sum(axis=1).values
        pv_gen = day_df[pv_cols].sum(axis=1).values if pv_cols else 0
        baseline_net = total_load - pv_gen
        cost_base = grid_cost(baseline_net, day_df[price_col].values, grid_agent)
        
        # -- decentralised ----------------------------------------------
        dec_net = baseline_net.copy()
        for dev in device_agents:
            dev.optimize_day(
                d,
                day_df[price_col].values,
                pv_agent.get_hourly_forecast_pv(d),
                battery_state=None,
                grid_info=None
            )
            # Get delta and ensure it has the same shape as dec_net (24 hours)
            try:
                delta = dev.optimized_consumption - dev.original_consumption
                # Check for shape mismatch and fix if needed
                if len(delta) != len(dec_net):
                    print(f"Warning: Shape mismatch detected. Delta shape: {delta.shape}, expected: {dec_net.shape}")
                    # If delta is for multiple days, extract just this day's data
                    # Assuming delta has structure [day1_hour1, day1_hour2, ..., day1_hour24, day2_hour1, ...]
                    days_in_data = len(delta) // 24
                    day_index = days.index(d)
                    if day_index < days_in_data:
                        start_idx = day_index * 24
                        delta = delta[start_idx:start_idx + 24]
                        print(f"Fixed: Using delta[{start_idx}:{start_idx + 24}] with shape: {delta.shape}")
                    else:
                        # If we can't extract the right day, use zeros as fallback
                        delta = np.zeros_like(dec_net)
                        print(f"Warning: Could not extract correct day data, using zeros instead")
                
                # Now add delta to dec_net (shapes should match)
                dec_net += delta
            except Exception as e:
                print(f"Error processing delta for device: {e}")
                # Continue without applying this delta
        cost_dec = grid_cost(dec_net, day_df[price_col].values, grid_agent)
        
        # -- centralised -------------------------------------------------
        optimizer.optimize_centralized()
        # Build centralised net grid flow manually (original + per-device deltas)
        cent_net = baseline_net.copy()
        for dev in device_agents:
            try:
                opt_sched = None
                # Prefer generic attribute names in order of likelihood
                for attr in ("optimized_consumption", "centralized_optimized_schedule", "nextday_optimized_schedule"):
                    if hasattr(dev, attr):
                        opt_sched = getattr(dev, attr)
                        if opt_sched is not None:
                            break
                if opt_sched is None:
                    continue  # skip if no optimised schedule

                delta = opt_sched - dev.original_consumption
                # Ensure delta length = 24 hours for this day
                if len(delta) != len(cent_net):
                    days_in_data = len(delta) // 24
                    day_index = list(days).index(d)
                    if day_index < days_in_data:
                        start_idx = day_index * 24
                        delta = delta[start_idx:start_idx + 24]
                    else:
                        delta = np.zeros_like(cent_net)
                cent_net += delta
            except Exception as exc:
                print(f"Error processing centralised delta for {dev.device_name}: {exc}")
        cost_cent = grid_cost(cent_net, day_df[price_col].values, grid_agent)
        
        records.append(dict(
            day=d,
            baseline_cost=cost_base,
            decentralised_cost=cost_dec,
            centralised_cost=cost_cent
        ))
    
    # KPI table
    kpi = (pd.DataFrame(records)
           .assign(
               savings_dec=lambda t: 100 * (t.baseline_cost - t.decentralised_cost) / t.baseline_cost,
               savings_cent=lambda t: 100 * (t.baseline_cost - t.centralised_cost) / t.baseline_cost
           )
           .round(3))
    
    return kpi


def run_kpi_analysis(building_id: str, 
                     n_days: int = 3,
                     use_battery: bool = True,
                     use_ev: bool = True) -> pd.DataFrame:
    """
    Run a KPI analysis with default parameters and display results.
    
    Args:
        building_id: ID of the building to analyze
        n_days: Number of days to include in the analysis
        use_battery: Whether to include battery in the optimization
        use_ev: Whether to include EV in the optimization
        
    Returns:
        pd.DataFrame: DataFrame containing KPIs
    """
    kpi_df = calculate_kpis(
        building_id=building_id,
        n_days=n_days,
        use_battery=use_battery,
        use_ev=use_ev
    )
    
    # Print a summary
    print(f"KPI Analysis for {building_id} ({n_days} days)")
    print("=" * 50)
    print(f"Battery included: {use_battery}")
    print(f"EV included: {use_ev}")
    print("\nResults:")
    print("-" * 50)
    print(kpi_df)
    
    avg_savings = {
        'decentralized': kpi_df['savings_dec'].mean(),
        'centralized': kpi_df['savings_cent'].mean()
    }
    
    print("\nAverage Savings:")
    print(f"Decentralized: {avg_savings['decentralized']:.2f}%")
    print(f"Centralized:   {avg_savings['centralized']:.2f}%")
    
    return kpi_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Calculate KPIs for EMS optimization strategies')
    parser.add_argument('--building-id', type=str, default="DE_KN_residential1",
                        help='Building ID to analyze')
    parser.add_argument('--days', type=int, default=3,
                        help='Number of days to analyze')
    parser.add_argument('--no-battery', action='store_true',
                        help='Exclude battery from analysis')
    parser.add_argument('--no-ev', action='store_true',
                        help='Exclude EV from analysis')
    parser.add_argument('--all-configs', action='store_true',
                        help='Run analysis with all configurations and show comparison')
    
    args = parser.parse_args()
    
    if args.all_configs:
        # Run with all configurations and show comparison
        print(f"\nRunning KPI analysis for {args.building_id} with {args.days} days for all configurations...")
        
        # Run with different configurations
        results = {}
        
        print("Running with Battery + EV...")
        results["Battery + EV"] = run_kpi_analysis(
            building_id=args.building_id,
            n_days=args.days,
            use_battery=True,
            use_ev=True
        )
        
        print("Running with No Battery...")
        results["No Battery"] = run_kpi_analysis(
            building_id=args.building_id,
            n_days=args.days,
            use_battery=False,
            use_ev=True
        )
        
        print("Running with No EV...")
        results["No EV"] = run_kpi_analysis(
            building_id=args.building_id,
            n_days=args.days,
            use_battery=True,
            use_ev=False
        )
        
        print("Running with No Battery, No EV (Basic)...")
        results["Basic"] = run_kpi_analysis(
            building_id=args.building_id,
            n_days=args.days,
            use_battery=False,
            use_ev=False
        )
        
        # Create comparison table
        import pandas as pd
        comparison_data = []
        for config, result in results.items():
            comparison_data.append({
                "Configuration": config,
                "Baseline Cost (€)": result["cost_baseline"].mean(),
                "Decentralized Cost (€)": result["cost_dec"].mean(),
                "Centralized Cost (€)": result["cost_cent"].mean(),
                "Decentralized Savings (%)": result["savings_dec"].mean(),
                "Centralized Savings (%)": result["savings_cent"].mean()
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        print("\n" + "=" * 80)
        print("KPI COMPARISON TABLE")
        print("=" * 80)
        print(comparison_df.round(2).to_string(index=False))
        print("=" * 80 + "\n")
        
        # Explicitly flush stdout to ensure all output is shown
        import sys
        sys.stdout.flush()
        
        # Also write to a file to ensure results are captured
        output_file = "kpi_comparison_table.txt"
        try:
            with open(output_file, "w") as f:
                f.write("KPI COMPARISON TABLE\n")
                f.write("=" * 80 + "\n")
                f.write(comparison_df.round(2).to_string(index=False))
                f.write("\n" + "=" * 80 + "\n")
            print(f"\nResults also written to {output_file}")
        except Exception as e:
            print(f"Error writing to file: {str(e)}")
        
    else:
        # Run the analysis with specified configuration
        print(f"\nRunning KPI analysis for {args.building_id} with {args.days} days...")
        print(f"Configuration: {'With' if not args.no_battery else 'Without'} Battery, {'With' if not args.no_ev else 'Without'} EV")
        
        kpi_df = run_kpi_analysis(
            building_id=args.building_id,
            n_days=args.days,
            use_battery=not args.no_battery,
            use_ev=not args.no_ev
        )
        
        # Print basic results
        print("\n" + "=" * 80)
        print("KPI ANALYSIS RESULTS")
        print("=" * 80)
        print(f"Average baseline cost: {kpi_df['cost_baseline'].mean():.2f} €")
        print(f"Average decentralized cost: {kpi_df['cost_dec'].mean():.2f} €")
        print(f"Average centralized cost: {kpi_df['cost_cent'].mean():.2f} €")
        print(f"Average decentralized savings: {kpi_df['savings_dec'].mean():.2f}%")
        print(f"Average centralized savings: {kpi_df['savings_cent'].mean():.2f}%")
        print("=" * 80 + "\n")
        
        # Explicitly flush stdout to ensure all output is shown
        import sys
        sys.stdout.flush()
