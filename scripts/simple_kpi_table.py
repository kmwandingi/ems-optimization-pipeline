"""
Simple KPI table generator.
Just runs KPI analysis and outputs a comparison table - nothing else.
"""
import sys
import os
import pandas as pd

# Add project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

# Add notebooks directory to path for agent imports
notebooks_dir = os.path.join(project_root, "notebooks")
sys.path.insert(0, notebooks_dir)

# Import the KPI calculation function
from scripts.calculate_kpis import run_kpi_analysis

def main():
    # Use a single building for simplicity
    building_id = "DE_KN_residential1"
    n_days = 3  # Just 3 days for quick results
    
    print(f"Running KPI analysis for building {building_id} ({n_days} days)...")
    print("This may take a moment...")
    
    # Dictionary to store results
    results = {}
    
    # Configuration 1: With Battery and EV
    print("Running config: Battery + EV...")
    results["Battery + EV"] = run_kpi_analysis(
        building_id=building_id,
        n_days=n_days,
        use_battery=True,
        use_ev=True
    )
    
    # Configuration 2: No Battery, with EV
    print("Running config: No Battery...")
    results["No Battery"] = run_kpi_analysis(
        building_id=building_id,
        n_days=n_days,
        use_battery=False,
        use_ev=True
    )
    
    # Configuration 3: With Battery, No EV
    print("Running config: No EV...")
    results["No EV"] = run_kpi_analysis(
        building_id=building_id,
        n_days=n_days,
        use_battery=True,
        use_ev=False
    )
    
    # Configuration 4: No Battery, No EV (Basic)
    print("Running config: Basic (No Battery, No EV)...")
    results["Basic"] = run_kpi_analysis(
        building_id=building_id,
        n_days=n_days,
        use_battery=False,
        use_ev=False
    )
    
    # Create comparison data
    comparison_data = []
    for config_name, result_df in results.items():
        comparison_data.append({
            "Configuration": config_name,
            "Baseline Cost (€)": result_df["cost_baseline"].mean(),
            "Decentralized Cost (€)": result_df["cost_dec"].mean(),
            "Centralized Cost (€)": result_df["cost_cent"].mean(),
            "Decentralized Savings (%)": result_df["savings_dec"].mean(),
            "Centralized Savings (%)": result_df["savings_cent"].mean()
        })
    
    # Create a DataFrame for the comparison table
    comparison_df = pd.DataFrame(comparison_data)
    
    # Print the comparison table directly to the console
    print("\nKPI COMPARISON TABLE:")
    print("==========================================")
    print(comparison_df.round(2).to_string(index=False))
    print("==========================================")
    
    # Also write to stdout to ensure it's flushed
    import sys
    sys.stdout.write("\nKPI COMPARISON TABLE:\n")
    sys.stdout.write("==========================================\n")
    sys.stdout.write(comparison_df.round(2).to_string(index=False))
    sys.stdout.write("\n==========================================\n")
    sys.stdout.flush()

if __name__ == "__main__":
    main()
