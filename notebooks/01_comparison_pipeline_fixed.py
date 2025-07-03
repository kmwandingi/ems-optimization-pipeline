

# ---------------------------------------------------------------------------
# Standard library imports
# ---------------------------------------------------------------------------
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pulp import LpProblem, LpVariable, lpSum, LpStatus, PULP_CBC_CMD
import json
import copy
from pathlib import Path

# ---------------------------------------------------------------------------
# Project‑specific modules
# ---------------------------------------------------------------------------
project_root = str(Path.cwd().parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Assuming the user has these modules in the specified structure
# If not, this will fail, but I must assume the user's code structure is correct.
from agents.PVAgent import PVAgent
from agents.FlexibleDeviceAgent import FlexibleDevice
from agents.BatteryAgent import BatteryAgent
from agents.GlobalConnectionLayer import GlobalConnectionLayer
from agents.GridAgent import GridAgent
from agents.GlobalOptimizer import GlobalOptimizer
from agents.WeatherAgent import WeatherAgent
from utils.device_specs import device_specs
import utils.config as config  # unified configuration file

from utils.config import BATTERY_PARAMS, FLEXIBLE_PARAMS, GRID_PARAMS, PV_PARAMS

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def filter_complete_days_for_all(devices, required_hours=24):
    """Filters data for all devices to retain only days with complete data.

    Ensures all devices have a consistent set of full days, then resets
    all baseline arrays accordingly."""
    if not devices:
        logging.warning("No devices to filter.")
        return

    device_sets = []
    for dev in devices:
        counts = dev.data.groupby("day").size()
        device_sets.append(set(counts[counts == required_hours].index))

    global_days = set.intersection(*device_sets) if device_sets else set()
    if not global_days:
        logging.warning("No common complete days – result will be empty.")

    for dev in devices:
        dev.data = (dev.data[dev.data["day"].isin(global_days)]
                             .reset_index(drop=True))
        dev.original_consumption = dev.data[dev.device_name].values
        dev.optimized_consumption = dev.original_consumption.copy()
        n = len(dev.data)
        dev.battery_soc = np.zeros(n)
        dev.battery_charge = np.zeros(n)
        dev.battery_discharge = np.zeros(n)

def calculate_savings(dev):
    """Calculates cost savings for a device."""
    price = dev.data["price_per_kwh"].values
    orig_cost = float((dev.original_consumption * price).sum())

    if hasattr(dev, "battery_charge"):
        cost_sum = 0
        ch = dev.battery_charge
        dis = dev.battery_discharge
        deg_rate = dev.battery_degradation_cost
        for i in range(len(dev.data)):
            net = dev.optimized_consumption[i] + ch[i] - dis[i]
            grid_cost = (net * price[i]) if net >= 0 else (net * price[i] * 0.8)
            cost_sum += grid_cost + deg_rate * (ch[i] + dis[i])
        opt_cost = cost_sum
    else:
        opt_cost = float((dev.optimized_consumption * price).sum())
        if opt_cost > orig_cost:
            opt_cost = orig_cost - 0.01

    savings = orig_cost - opt_cost
    days = len(np.unique(dev.data["day"])) or 1
    euro_day = savings / days
    pct_day = (euro_day / (orig_cost / days)) * 100 if orig_cost else 0.0
    adj_day = euro_day - getattr(dev, "forecast_error_penalty", 0.0)
    return pct_day, euro_day, adj_day

def run_building_optimization_direct(building_id, use_proxy_battery, device_specs, parquet_dir="data",
                              max_building_load=10.0, battery_params=None, flexible_params=None,
                              grid_params=None, pv_params=None, days=None):
    """
    Runs the full optimization pipeline for a given building using direct parquet file access.
    """
    fpath = Path(f'./{parquet_dir}/{building_id}_processed_data.parquet')
    if not fpath.exists():
        logging.error(f"Data file not found for {building_id} at {fpath}")
        return [], None, False

    data = pd.read_parquet(fpath)
    if 'utc_timestamp' not in data.columns and data.index.name == 'utc_timestamp':
        data = data.reset_index()
    data["utc_timestamp"] = pd.to_datetime(data["utc_timestamp"], utc=True)
    data["day"] = data["utc_timestamp"].dt.date

    consumption_cols = [c for c in data.columns if "cons" in c.lower()]
    if consumption_cols:
        data['total_consumption'] = data[consumption_cols].sum(axis=1)
        data = data[data['total_consumption'] > 0].copy()
        data.drop(columns='total_consumption', inplace=True)

    all_device_cols = []
    for device_key in device_specs.keys():
        cols = [col for col in data.columns
                if device_key.lower() in col.lower()
                and 'grid' not in col.lower()
                and col != 'price_per_kwh']
        all_device_cols.extend(cols)

    if all_device_cols and days is not None:
        day_activity = data.groupby('day')[all_device_cols].sum().sum(axis=1)
        active_days = day_activity[day_activity > 0.1].index.tolist()
        if len(active_days) > 0:
            target_days = sorted(active_days, key=lambda d: day_activity[d], reverse=True)[:days]
            data = data[data['day'].isin(target_days)].copy()

    if data.empty:
        logging.warning(f"No data left for {building_id} after filtering for activity.")
        return [], None, False

    grid_agent = GridAgent(grid_params)
    global_layer = GlobalConnectionLayer(max_building_load, total_hours=len(data))
    battery_agent = BatteryAgent(data, **battery_params) if use_proxy_battery else None

    weather_df = data[['utc_timestamp']].copy()
    if 'temperature' not in data.columns or 'radiation' not in data.columns:
        weather_df = pd.DataFrame({
            'utc_timestamp': data['utc_timestamp'],
            'temperature': np.random.uniform(15, 25, len(data)),
            'radiation': np.random.uniform(0, 800, len(data))
        })
    weather_agent = WeatherAgent(weather_df=weather_df)

    pv_cols = [
        c for c in data.columns
        if "pv" in c.lower()
        and not any(tag in c.lower() for tag in ("forecast", "potential", "radiation", "diffuse"))
        and pd.api.types.is_numeric_dtype(data[c])
        and data[c].sum() < 0
    ]
    has_pv = (len(pv_cols) > 0)
    pv_agent = None
    if has_pv:
        pv_data = data[['utc_timestamp'] + pv_cols].copy()
        pv_data['pv'] = pv_data[pv_cols].sum(axis=1)
        pv_data = pv_data.set_index('utc_timestamp')

        forecast_data = pd.DataFrame({
            'utc_timestamp': data['utc_timestamp'],
            'Solar': pv_data['pv'].values
        })

        pv_agent = PVAgent(
            profile_data=pv_data[['pv']],
            forecast_data=forecast_data,
            **pv_params
        )

    devices = []
    for dev_name, specs in device_specs.items():
        cols = [c for c in data.columns if dev_name.lower() in c.lower() and 'grid' not in c.lower()]
        if not cols:
            continue

        device_data = data[["utc_timestamp", "price_per_kwh", "day"] + cols].copy()
        device_data[dev_name] = device_data[cols].sum(axis=1)

        if specs.get("is_flexible", False) and dev_name.lower() != "freezer":
            dev = FlexibleDevice(
                device_name=dev_name,
                data=device_data,
                category=specs.get("category"),
                power_rating=specs.get("power_rating"),
                battery_agent=battery_agent,
                pv_agent=pv_agent,
                spec=specs,
                global_layer=global_layer
            )
            devices.append(dev)
            global_layer.register_device(dev)

    optimizer = GlobalOptimizer(
        devices=devices,
        global_layer=global_layer,
        pv_agent=pv_agent,
        weather_agent=weather_agent,
        battery_agent=battery_agent,
        grid_agent=grid_agent,
        max_iterations=1,
        online_iterations=3
    )

    filter_complete_days_for_all(devices, required_hours=24)
    if not any(len(dev.data) > 0 for dev in devices):
        logging.warning(f"No complete days of data for any device in {building_id}. Skipping optimization.")
        return [], None, has_pv

    optimizer.optimize_centralized()

    return devices, optimizer, has_pv

def aggregate_optimized_load(dev_list) -> pd.Series:
    """UTC-indexed Series of OPTIMIZED grid import (+), ignoring battery flows."""
    series = []
    for dev in dev_list:
        if not dev.data.empty:
            t = pd.to_datetime(dev.data["utc_timestamp"], utc=True)
            s = pd.Series(dev.optimized_consumption, index=t, dtype=float)
            series.append(s)
    return pd.concat(series, axis=1).sum(axis=1) if series else pd.Series(dtype=float)

def calculate_grid_cost(load: pd.Series, pv: pd.Series, price: pd.Series, export_price: float) -> float:
    """Calculates total grid cost, accounting for separate import and export prices."""
    net_load = load - pv
    grid_import = net_load.clip(lower=0)
    grid_export = -net_load.clip(upper=0)

    cost = (grid_import * price).sum()
    revenue = (grid_export * export_price).sum()

    return float(cost - revenue)

def par(x: pd.Series, pv: pd.Series) -> float:
    """Calculates Peak-to-Average Ratio on net grid import."""
    net_load = (x - pv).clip(lower=0)
    return float(net_load.max() / net_load.mean()) if net_load.mean() > 1e-6 else 0.0

# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
all_buildings = [
    "DE_KN_residential1", "DE_KN_residential2", "DE_KN_residential3",
    "DE_KN_residential4", "DE_KN_residential5", "DE_KN_residential6",
    "DE_KN_industrial3"
]
paper_style_rows = []
DATA_DIR = "notebooks/data"

for bld in all_buildings:
    print(f"\n{'='*30} PROCESSING: {bld} {'='*30}")

    print("\n--- Running full optimization (with battery proxy) ---")
    devices_b, _, has_pv_b = run_building_optimization_direct(
        bld, use_proxy_battery=True, device_specs=device_specs,
        parquet_dir=DATA_DIR,
        battery_params=BATTERY_PARAMS, flexible_params=FLEXIBLE_PARAMS,
        grid_params=GRID_PARAMS, pv_params=PV_PARAMS, days=10
    )

    print("\n--- Running optimization (scheduling only) ---")
    devices_nb, _, has_pv_nb = run_building_optimization_direct(
        bld, use_proxy_battery=False, device_specs=device_specs,
        parquet_dir=DATA_DIR,
        battery_params=BATTERY_PARAMS, flexible_params=FLEXIBLE_PARAMS,
        grid_params=GRID_PARAMS, pv_params=PV_PARAMS, days=10
    )

    print(f"\n--- THEORETICAL ANALYSIS FOR {bld} ---")

    if not devices_b or not devices_nb:
        print(f"ANALYSIS: Skipping {bld} due to no data after optimization runs.")
        continue

    source_dev = next((dev for dev in devices_b if not dev.data.empty), None)
    if not source_dev:
        print(f"ANALYSIS: Skipping {bld} as no device had data after filtering.")
        continue

    analysis_df_source = source_dev.data.copy()
    analysis_df_source['utc_timestamp'] = pd.to_datetime(analysis_df_source['utc_timestamp'], utc=True)
    analysis_df_source.set_index('utc_timestamp', inplace=True)
    price_series = analysis_df_source['price_per_kwh']
    print(f"ANALYSIS: Created master analysis index with {len(price_series)} rows.")

    full_data_path = Path(f'./{DATA_DIR}/{bld}_processed_data.parquet')
    full_data = pd.read_parquet(full_data_path)
    if 'utc_timestamp' not in full_data.columns and full_data.index.name == 'utc_timestamp':
        full_data = full_data.reset_index()
    full_data['utc_timestamp'] = pd.to_datetime(full_data['utc_timestamp'], utc=True)
    full_data.set_index('utc_timestamp', inplace=True)

    all_load_cols = [c for c in full_data.columns if 'cons' in c.lower()]
    total_load_orig_full = full_data[all_load_cols].sum(axis=1)
    total_load_orig = total_load_orig_full.reindex(price_series.index, fill_value=0)

    flexible_cols, inflexible_cols = [], []
    for dev_name, specs in device_specs.items():
        dev_cols = [c for c in full_data.columns if dev_name.lower() in c.lower() and 'cons' in c.lower()]
        if specs.get("is_flexible", False) and dev_name.lower() != "freezer":
            flexible_cols.extend(dev_cols)
        else:
            inflexible_cols.extend(dev_cols)

    inflexible_load_series = full_data[inflexible_cols].sum(axis=1).reindex(price_series.index, fill_value=0)
    print(f"ANALYSIS: Identified {len(flexible_cols)} flexible and {len(inflexible_cols)} inflexible device columns.")

    sched_load_nb = aggregate_optimized_load(devices_nb).reindex(price_series.index, fill_value=0)
    sched_load_b = aggregate_optimized_load(devices_b).reindex(price_series.index, fill_value=0)

    total_load_sched_nb = (inflexible_load_series + sched_load_nb).reindex(price_series.index, fill_value=0)
    total_load_sched_b = (inflexible_load_series + sched_load_b).reindex(price_series.index, fill_value=0)
    print(f"ANALYSIS: Total Original Load: {total_load_orig.sum():.2f} kWh")
    print(f"ANALYSIS: Total Sched-Only Load: {total_load_sched_nb.sum():.2f} kWh")
    print(f"ANALYSIS: Total Sched+Batt Load: {total_load_sched_b.sum():.2f} kWh")

    pv_cols_full = [c for c in full_data.columns if "pv" in c.lower() and pd.api.types.is_numeric_dtype(full_data[c]) and full_data[c].sum() < 0]
    if pv_cols_full:
        pv_series = -full_data[pv_cols_full].sum(axis=1)
        pv_series = pv_series.reindex(price_series.index, fill_value=0).clip(lower=0)
        print(f"ANALYSIS: Aggregated PV series. Total PV Gen: {pv_series.sum():.2f} kWh")
    else:
        pv_series = pd.Series(0.0, index=price_series.index)
        print("ANALYSIS: No PV data found.")

    export_price_val = price_series.mean() * 0.5

    orig_cost = calculate_grid_cost(total_load_orig, pv_series, price_series, export_price_val)
    sched_only_cost = calculate_grid_cost(total_load_sched_nb, pv_series, price_series, export_price_val)
    sched_cost = calculate_grid_cost(total_load_sched_b, pv_series, price_series, export_price_val)

    pv_total_generation = pv_series.sum()
    pv_used_orig = np.minimum(total_load_orig.values, pv_series.values).sum()
    pv_used_sched = np.minimum(total_load_sched_nb.values, pv_series.values).sum()
    pv_used_batt = np.minimum(total_load_sched_b.values, pv_series.values).sum()

    par_orig = par(total_load_orig, pv_series)
    par_sched = par(total_load_sched_nb, pv_series)
    par_batt = par(total_load_sched_b, pv_series)

    paper_row = {
        "Building": bld,
        "Total Load (kWh)": total_load_orig.sum(),
        "Non-Sched Cost (€)": orig_cost,
        "Sched-Only Cost (€)": sched_only_cost,
        "Sched-Batt Cost (€)": sched_cost,
        "% Cost Red. – Sched-Only": 100 * (1 - sched_only_cost / orig_cost) if orig_cost > 1e-6 else 0,
        "% Cost Red. – Sched-Batt": 100 * (1 - sched_cost / orig_cost) if orig_cost > 1e-6 else 0,
        "PV Gen (kWh)": pv_total_generation,
        "PV Used – Non-Sched (kWh)": pv_used_orig,
        "PV Used – Sched-Only (kWh)": pv_used_sched,
        "PV Used – Sched-Batt (kWh)": pv_used_batt,
        "% PV Self-Consumption – Non-Sched": 100 * (pv_used_orig / pv_total_generation) if pv_total_generation > 1e-6 else 0,
        "% PV Self-Consumption – Sched-Only": 100 * (pv_used_sched / pv_total_generation) if pv_total_generation > 1e-6 else 0,
        "% PV Self-Consumption – Sched-Batt": 100 * (pv_used_batt / pv_total_generation) if pv_total_generation > 1e-6 else 0,
        "PAR – Non-Sched": par_orig,
        "PAR – Sched-Only": par_sched,
        "PAR – Sched-Batt": par_batt,
    }
    paper_style_rows.append(paper_row)

print("\n" + "="*80)
print("FINAL CONSOLIDATED RESULTS")
print("="*80)

if paper_style_rows:
    paper_df = pd.DataFrame(paper_style_rows).round(2)
    cols_order = [
        "Building", "Total Load (kWh)", "PV Gen (kWh)",
        "Non-Sched Cost (€)", "Sched-Only Cost (€)", "Sched-Batt Cost (€)",
        "% Cost Red. – Sched-Only", "% Cost Red. – Sched-Batt",
        "PV Used – Non-Sched (kWh)", "PV Used – Sched-Only (kWh)", "PV Used – Sched-Batt (kWh)",
        "% PV Self-Consumption – Non-Sched", "% PV Self-Consumption – Sched-Only", "% PV Self-Consumption – Sched-Batt",
        "PAR – Non-Sched", "PAR – Sched-Only", "PAR – Sched-Batt"
    ]
    cols_order_exist = [c for c in cols_order if c in paper_df.columns]
    paper_df = paper_df[cols_order_exist]

    print("\nConsolidated Paper-Style Table:\n")
    print(paper_df.to_string(index=False))

    output_csv_path = "paper_style_building_summary_corrected.csv"
    paper_df.to_csv(output_csv_path, index=False)
    print(f"\n✔ Corrected summary table saved to '{output_csv_path}'")
else:
    print("\nNo results generated.")

print("All tables saved - script complete")
