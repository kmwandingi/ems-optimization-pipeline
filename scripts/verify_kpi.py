"""
Simple verification script for KPI calculation.
"""
import sys
import os

# Add project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

# Add notebooks directory to path
notebooks_dir = os.path.join(project_root, "notebooks")
sys.path.insert(0, notebooks_dir)

if __name__ == "__main__":
    print("Starting KPI verification script")
    print(f"Python path: {sys.path}")
    
    # Import the calculate_kpis module's main function
    from scripts.calculate_kpis import run_kpi_analysis
    
    # Run a minimal analysis to verify functionality
    print("\nRunning minimal KPI analysis...")
    result = run_kpi_analysis(
        building_id="DE_KN_residential1", 
        n_days=1,  # Just analyze one day
        use_battery=True,
        use_ev=False
    )
    
    if result is not None:
        print("\nKPI analysis succeeded!")
        print(f"Result has {len(result)} rows and {len(result.columns)} columns")
        print(f"Columns: {list(result.columns)}")
        
        # Print the first row as a sample
        if not result.empty:
            print("\nSample result (first row):")
            first_row = result.iloc[0]
            for col in result.columns:
                print(f"{col}: {first_row[col]}")
            
            # Calculate averages for key metrics
            print("\nAverage metrics:")
            print(f"Baseline cost: {result['cost_baseline'].mean():.2f} €")
            print(f"Decentralized cost: {result['cost_dec'].mean():.2f} €")
            print(f"Centralized cost: {result['cost_cent'].mean():.2f} €")
            print(f"Decentralized savings: {result['savings_dec'].mean():.2f}%")
            print(f"Centralized savings: {result['savings_cent'].mean():.2f}%")
    else:
        print("ERROR: KPI analysis returned None")
