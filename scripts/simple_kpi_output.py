"""
Very simple KPI table generator with robust error handling.
"""
import sys
import os
import pandas as pd
import traceback
from datetime import datetime

# Print starting message
print(f"Starting KPI analysis at {datetime.now()}")

try:
    # Add project root to Python path
    script_path = os.path.abspath(__file__)
    print(f"Script path: {script_path}")
    
    script_dir = os.path.dirname(script_path)
    print(f"Script directory: {script_dir}")
    
    project_root = os.path.dirname(script_dir)
    print(f"Project root: {project_root}")
    
    sys.path.insert(0, project_root)
    
    # Add notebooks directory to path for agent imports
    notebooks_dir = os.path.join(project_root, "notebooks")
    print(f"Adding notebooks directory: {notebooks_dir}")
    sys.path.insert(0, notebooks_dir)
    
    # Output file with absolute path
    output_file = os.path.join(project_root, "kpi_output.txt")
    print(f"Output will be written to: {output_file}")
    
    # Import the KPI calculation function
    print("Importing calculate_kpis module...")
    from scripts.calculate_kpis import run_kpi_analysis
    print("Successfully imported calculate_kpis module")
    
    # Use a single building for simplicity
    building_id = "DE_KN_residential1"
    n_days = 3  # Just 3 days for quick results
    
    print(f"Running KPI analysis for building {building_id} ({n_days} days)...")
    
    # Run with all components
    print("Running with all components (Battery + EV)...")
    full_results = run_kpi_analysis(
        building_id=building_id,
        n_days=n_days,
        use_battery=True,
        use_ev=True
    )
    print(f"Full results shape: {full_results.shape}")
    
    # Run with no battery
    print("Running with no battery...")
    no_battery_results = run_kpi_analysis(
        building_id=building_id,
        n_days=n_days,
        use_battery=False,
        use_ev=True
    )
    print(f"No battery results shape: {no_battery_results.shape}")
    
    # Run with no EV
    print("Running with no EV...")
    no_ev_results = run_kpi_analysis(
        building_id=building_id,
        n_days=n_days,
        use_battery=True,
        use_ev=False
    )
    print(f"No EV results shape: {no_ev_results.shape}")
    
    # Run with no components
    print("Running with no components (basic)...")
    basic_results = run_kpi_analysis(
        building_id=building_id,
        n_days=n_days,
        use_battery=False,
        use_ev=False
    )
    print(f"Basic results shape: {basic_results.shape}")
    
    # Create comparison data
    print("Creating comparison table...")
    comparison_data = [
        {
            "Configuration": "Battery + EV",
            "Baseline Cost (€)": full_results["cost_baseline"].mean(),
            "Decentralized Cost (€)": full_results["cost_dec"].mean(),
            "Centralized Cost (€)": full_results["cost_cent"].mean(),
            "Decentralized Savings (%)": full_results["savings_dec"].mean(),
            "Centralized Savings (%)": full_results["savings_cent"].mean()
        },
        {
            "Configuration": "No Battery",
            "Baseline Cost (€)": no_battery_results["cost_baseline"].mean(),
            "Decentralized Cost (€)": no_battery_results["cost_dec"].mean(),
            "Centralized Cost (€)": no_battery_results["cost_cent"].mean(),
            "Decentralized Savings (%)": no_battery_results["savings_dec"].mean(),
            "Centralized Savings (%)": no_battery_results["savings_cent"].mean()
        },
        {
            "Configuration": "No EV",
            "Baseline Cost (€)": no_ev_results["cost_baseline"].mean(),
            "Decentralized Cost (€)": no_ev_results["cost_dec"].mean(),
            "Centralized Cost (€)": no_ev_results["cost_cent"].mean(),
            "Decentralized Savings (%)": no_ev_results["savings_dec"].mean(),
            "Centralized Savings (%)": no_ev_results["savings_cent"].mean()
        },
        {
            "Configuration": "Basic (No Battery, No EV)",
            "Baseline Cost (€)": basic_results["cost_baseline"].mean(),
            "Decentralized Cost (€)": basic_results["cost_dec"].mean(),
            "Centralized Cost (€)": basic_results["cost_cent"].mean(),
            "Decentralized Savings (%)": basic_results["savings_dec"].mean(),
            "Centralized Savings (%)": basic_results["savings_cent"].mean()
        }
    ]
    
    # Create a DataFrame for the comparison table
    comparison_df = pd.DataFrame(comparison_data)
    
    # Format the table with rounded values
    formatted_table = comparison_df.round(2).to_string(index=False)
    
    print(f"Writing comparison table to {output_file}...")
    with open(output_file, "w") as f:
        f.write("\nKPI COMPARISON TABLE:\n")
        f.write("=" * 80 + "\n")
        f.write(formatted_table)
        f.write("\n" + "=" * 80 + "\n")
    
    print(f"Successfully wrote to {output_file}")
    print("\nTable contents:")
    print("=" * 80)
    print(formatted_table)
    print("=" * 80)
    
except Exception as e:
    print(f"ERROR: {str(e)}")
    traceback.print_exc()

print(f"Script completed at {datetime.now()}")
