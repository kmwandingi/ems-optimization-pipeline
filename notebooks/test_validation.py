# Standard library imports
import sys
import pandas as pd
from pathlib import Path

# Find the project root by looking for a known directory ('agents')
# and add it to the system path.
# This makes the script runnable from anywhere.
current_path = Path(__file__).resolve()
project_root = current_path.parent
while not (project_root / 'agents').is_dir() and project_root.parent != project_root:
    project_root = project_root.parent
sys.path.insert(0, str(project_root))

from utils.helper import run_building_optimization_direct
from utils.device_specs import device_specs
from utils.config import BATTERY_PARAMS, FLEXIBLE_PARAMS, GRID_PARAMS, PV_PARAMS

def generate_comparison_table():
    """
    Generates a comparison table for building energy optimization.
    """
    building_ids = [
        "DE_KN_industrial3",
        "DE_KN_residential1",
        "DE_KN_residential2",
        "DE_KN_residential3",
        "DE_KN_residential4",
        "DE_KN_residential5",
        "DE_KN_residential6",
    ]

    results = []

    for bld in building_ids:
        print(f"Processing building: {bld}")

        try:
            # Run optimization without battery
            devices_nb, global_layer_nb, pv_agent_nb, _, data_nb = run_building_optimization_direct(
                building_id=bld, use_proxy_battery=False, device_specs=device_specs,
                days=10, parquet_dir="data", battery_params=BATTERY_PARAMS,
                flexible_params=FLEXIBLE_PARAMS, grid_params=GRID_PARAMS, pv_params=PV_PARAMS
            )

            # Run optimization with battery
            devices_wb, global_layer_wb, pv_agent_wb, battery_agent_wb, data_wb = run_building_optimization_direct(
                building_id=bld, use_proxy_battery=True, device_specs=device_specs,
                days=10, parquet_dir="data", battery_params=BATTERY_PARAMS,
                flexible_params=FLEXIBLE_PARAMS, grid_params=GRID_PARAMS, pv_params=PV_PARAMS
            )

            # --- Aggregate Data ---
            # Original Load
            orig_series = data_nb['original_total_load']
            orig_cost = (orig_series * data_nb['price']).sum()

            # Scheduled-Only (No Battery)
            sched_series_nb = pd.Series(global_layer_nb.get_total_load(), index=global_layer_nb.master_analysis_index)
            sched_only_cost = (sched_series_nb * data_nb['price']).sum()

            # Scheduled + Battery
            sched_series_wb = pd.Series(global_layer_wb.get_total_load(), index=global_layer_wb.master_analysis_index)
            sched_cost = (sched_series_wb * data_wb['price']).sum()

            # --- Populate paper_row ---
            paper_row = {
                "Building": bld,
                "No. of Loads": len(devices_nb),
                "Non-Sched Load (kWh)": orig_series.sum(),
                "Sched-Only Load (kWh)": sched_series_nb.sum(),
                "Sched-Batt Load (kWh)": sched_series_wb.sum(),
                "Non-Sched Cost (€)": orig_cost,
                "Sched-Only Cost (€)": sched_only_cost,
                "Sched-Batt Cost (€)": sched_cost,
                "% Cost Red. – Sched-Only": 100 * (1 - sched_only_cost / orig_cost) if orig_cost > 0 else 0,
                "% Cost Red. – Sched-Batt": 100 * (1 - sched_cost / orig_cost) if orig_cost > 0 else 0,
            }
            results.append(paper_row)
        except Exception as e:
            print(f"Could not process building {bld}. Error: {e}")
            results.append({
                "Building": bld,
                "No. of Loads": "Error",
                "Non-Sched Load (kWh)": "Error",
                "Sched-Only Load (kWh)": "Error",
                "Sched-Batt Load (kWh)": "Error",
                "Non-Sched Cost (€)": "Error",
                "Sched-Only Cost (€)": "Error",
                "Sched-Batt Cost (€)": "Error",
                "% Cost Red. – Sched-Only": "Error",
                "% Cost Red. – Sched-Batt": "Error",
            })


    # --- Create and Display DataFrame ---
    summary_df = pd.DataFrame(results)
    print("\n--- Comparison Table ---")
    print(summary_df)
    summary_df.to_csv("paper_style_building_summary_corrected.csv", index=False)
    print("\nSuccessfully saved the table to paper_style_building_summary_corrected.csv")

if __name__ == "__main__":
    generate_comparison_table()
