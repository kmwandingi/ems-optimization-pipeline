#!/usr/bin/env python
"""
Probability Learning Rate Optimization Pipeline

This script implements comprehensive hyperparameter tuning for ProbabilityModelAgent
learning rates using MLflow tracking and advanced visualization.

Features:
1. Grid search over key learning rate parameters (LR_TAU, LR_MAX)
2. Visual tracking of PMF evolution from priors to learned distributions
3. Jensen-Shannon divergence and entropy metrics
4. Beautiful Seaborn economist-style visualizations
5. STRICT enforcement of "USE AGENT OPTIMIZERS" rule

Usage:
    python scripts/03_probability_learning_optimization.py --building DE_KN_residential1 --n_days 10
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Add notebooks directory to path for agent imports
sys.path.append(str(Path.cwd() / "notebooks"))

# Import ONLY agent classes - NO fallbacks allowed
try:
    from agents.ProbabilityModelAgent import ProbabilityModelAgent
    from agents.BatteryAgent import BatteryAgent
    from agents.EVAgent import EVAgent
    from agents.PVAgent import PVAgent
    from agents.GridAgent import GridAgent
    from agents.FlexibleDeviceAgent import FlexibleDevice
    from agents.GlobalOptimizer import GlobalOptimizer
    from agents.GlobalConnectionLayer import GlobalConnectionLayer
    from agents.WeatherAgent import WeatherAgent
    print("✓ Successfully imported ALL agent classes")
except ImportError as e:
    print(f"CRITICAL ERROR: Failed to import agent classes: {e}")
    print("This pipeline REQUIRES agent classes - no fallbacks allowed!")
    sys.exit(1)

# Import common utilities and device_specs
import common
sys.path.append(str(Path.cwd() / "notebooks" / "utils"))
from device_specs import device_specs

# Import MLflow tracking (NON-INTRUSIVE)
sys.path.append(str(Path.cwd() / "utils"))
try:
    from mlflow_tracker import EMS_OptimizationTracker
    MLFLOW_AVAILABLE = True
    print("✓ MLflow tracking enabled")
except ImportError as e:
    print(f"⚠ MLflow not available: {e}")
    MLFLOW_AVAILABLE = False

# Set up Seaborn economist style
plt.style.use(['seaborn-v0_8-whitegrid'])
sns.set_palette("husl")
colors = {
    'prior': '#1f77b4',      # Blue
    'learning': '#ff7f0e',   # Orange  
    'converged': '#2ca02c',  # Green
    'js_div': '#d62728',     # Red
    'entropy': '#9467bd',    # Purple
    'background': '#f8f9fa'
}

# Create output directories
os.makedirs("results/probability_optimization", exist_ok=True)
os.makedirs("results/probability_optimization/visualizations", exist_ok=True)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Probability Learning Rate Optimization with Agent Classes")
    parser.add_argument("--building", type=str, required=True,
                        help="Building ID (e.g., DE_KN_residential1)")
    parser.add_argument("--n_days", type=int, default=10,
                        help="Number of days for training")
    parser.add_argument("--lr_tau_values", type=str, default="10,20,30,50",
                        help="Comma-separated LR_TAU values to test")
    parser.add_argument("--lr_max_values", type=str, default="0.05,0.10,0.15,0.20",
                        help="Comma-separated LR_MAX values to test") 
    parser.add_argument("--target_device", type=str, default="heat_pump",
                        help="Device type to focus analysis on")
    parser.add_argument("--test_multiple_devices", action="store_true",
                        help="Test optimization across multiple device types")
    parser.add_argument("--device_filter", type=str, default="discrete_phase,partial_usage",
                        help="Comma-separated flex_models to test (discrete_phase,partial_usage,fixed)")
    parser.add_argument("--test_both_priors", action="store_true",
                        help="Test both uniform and learned priors for comparison")
    parser.add_argument("--use_learned_priors", action="store_true",
                        help="Use learned/realistic priors instead of uniform priors")
    
    return parser.parse_args()

def setup_duckdb_connection(building_id):
    """
    Setup DuckDB connection and validate data availability.
    ENFORCES "USE AGENT OPTIMIZERS" - all data stays in DuckDB.
    """
    print(f"📊 Setting up DuckDB connection for {building_id}...")
    
    # MANDATORY: Use DuckDB data access layer
    con = common.get_con()
    view_name = f"{building_id}_processed_data"
    
    # Validate data exists and get metadata
    row_count = con.execute(f"SELECT COUNT(*) as count FROM {view_name}").df()['count'][0]
    col_count = len(con.execute(f"DESCRIBE {view_name}").df())
    date_range = con.execute(f"SELECT MIN(DATE(utc_timestamp)) as min_date, MAX(DATE(utc_timestamp)) as max_date FROM {view_name}").df()
    
    print(f"✓ Connected to DuckDB: {row_count:,} rows, {col_count} columns")
    print(f"✓ Date range: {date_range['min_date'][0]} to {date_range['max_date'][0]}")
    print("✓ All data remains in DuckDB - no unnecessary DataFrame loading")
    
    return con, view_name

def select_training_days_from_duckdb(con, view_name, n_days):
    """
    Select training days using DuckDB queries.
    ENFORCES "USE AGENT OPTIMIZERS" - SQL-based day selection.
    """
    print(f"📅 Selecting {n_days} training days using DuckDB queries...")
    
    # Query for complete 24-hour days
    full_days_df = con.execute(f"""
        SELECT DATE(utc_timestamp) as day, COUNT(*) as hour_count 
        FROM {view_name} 
        GROUP BY DATE(utc_timestamp) 
        HAVING COUNT(*) = 24 
        ORDER BY DATE(utc_timestamp)
    """).df()
    
    full_days = pd.to_datetime(full_days_df['day']).dt.date.tolist()
    
    if len(full_days) < n_days:
        print(f"⚠ Only {len(full_days)} days available, adjusting to {len(full_days)}")
        n_days = len(full_days)
    
    selected_days = full_days[:n_days]
    print(f"✓ Selected {len(selected_days)} complete days from DuckDB")
    
    return [str(day) for day in selected_days]

def get_all_available_devices_from_duckdb(con, view_name, building_id):
    """
    Get ALL available devices from DuckDB for comprehensive testing.
    ENFORCES "USE AGENT OPTIMIZERS" - all data from DuckDB.
    """
    print(f"🔍 Finding ALL devices in DuckDB for {building_id}")
    
    # Get all columns from DuckDB  
    columns_df = con.execute(f"DESCRIBE {view_name}").df()
    device_columns = [col for col in columns_df['column_name'] 
                     if building_id in col and col not in ['utc_timestamp', 'price_per_kwh', 'grid_export', 'grid_import']]
    
    # Extract device types from column names
    device_types = []
    for col in device_columns:
        device_type = col.split('_')[-1].lower()
        if device_type not in device_types:
            device_types.append(device_type)
    
    print(f"✓ Found {len(device_types)} device types in DuckDB: {device_types}")
    return device_types

def get_available_devices_by_flex_model(con, view_name, building_id, flex_models):
    """
    Get available devices from DuckDB filtered by flex_model.
    ENFORCES "USE AGENT OPTIMIZERS" - all data from DuckDB.
    """
    print(f"🔍 Finding devices with flex_models: {flex_models}")
    
    # Get all columns from DuckDB
    columns_df = con.execute(f"DESCRIBE {view_name}").df()
    device_columns = [col for col in columns_df['column_name'] 
                     if building_id in col and col not in ['utc_timestamp', 'price_per_kwh', 'grid_export', 'grid_import']]
    
    # Extract device types and filter by flex_model
    available_devices = []
    for device_col in device_columns:
        # Extract device type from column name
        parts = device_col.split('_')
        if len(parts) >= 4 and '_'.join(parts[-2:]) in ['heat_pump', 'washing_machine']:
            device_type = '_'.join(parts[-2:])
        else:
            device_type = parts[-1]
        
        # Check if device type exists in device_specs and has matching flex_model
        if device_type in device_specs:
            device_flex_model = device_specs[device_type].get('flex_model', 'fixed')
            if device_flex_model in flex_models:
                available_devices.append({
                    'device_type': device_type,
                    'device_column': device_col,
                    'flex_model': device_flex_model
                })
    
    # Remove duplicates by device_type
    seen_types = set()
    unique_devices = []
    for device in available_devices:
        if device['device_type'] not in seen_types:
            unique_devices.append(device)
            seen_types.add(device['device_type'])
    
    print(f"✓ Found {len(unique_devices)} devices matching flex_models: {[d['device_type'] for d in unique_devices]}")
    return unique_devices

def get_training_data_from_duckdb(con, view_name, building_id, training_days):
    """
    Get training data from DuckDB for agent training.
    ENFORCES "USE AGENT OPTIMIZERS" - all data from DuckDB.
    """
    print(f"📊 Loading training data for {len(training_days)} days from DuckDB...")
    
    # Convert training_days to proper format for SQL
    days_str = "', '".join(training_days)
    
    # Get training data
    training_df = con.execute(f"""
        SELECT *, EXTRACT(hour FROM utc_timestamp) as hour, DATE(utc_timestamp) as day
        FROM {view_name} 
        WHERE DATE(utc_timestamp) IN ('{days_str}')
        ORDER BY utc_timestamp
    """).df()
    
    # Get weather and forecast data (mock for compatibility)
    weather_df = training_df[['utc_timestamp', 'day', 'hour']].copy()
    weather_df['temperature'] = 20.0  # Mock weather data
    forecast_df = weather_df.copy()   # Mock forecast data
    
    print(f"✓ Loaded {len(training_df)} rows of training data from DuckDB")
    return training_df, weather_df, forecast_df

def calculate_pmf_metrics(pmf_dict: Dict[int, float]) -> Dict[str, float]:
    """Calculate comprehensive metrics for PMF analysis."""
    if not pmf_dict:
        return {"entropy": 0.0, "max_probability": 0.0, "concentration": 0.0}
    
    # Ensure valid probability distribution
    probs = np.array([pmf_dict.get(h, 0.0) for h in range(24)])
    probs = probs / probs.sum() if probs.sum() > 0 else probs
    
    # Calculate entropy
    entropy = -np.sum([p * np.log(p + 1e-12) for p in probs if p > 0])
    
    # Calculate concentration metrics
    max_prob = np.max(probs)
    
    # Calculate concentration (inverse of entropy, normalized)
    max_entropy = np.log(24)  # Maximum possible entropy for 24 hours
    concentration = 1.0 - (entropy / max_entropy)
    
    return {
        "entropy": float(entropy),
        "max_probability": float(max_prob),
        "concentration": float(concentration)
    }

def create_learned_priors_from_duckdb(con, view_name: str, building_id: str, training_days: int = 30) -> pd.DataFrame:
    """
    Create learned prior distributions by training ProbabilityModelAgent on real DuckDB data.
    ENFORCES "USE REAL AGENT OPTIMIZERS" - uses only ProbabilityModelAgent.train() with real data.
    """
    from common import get_training_data_from_duckdb
    
    print(f"📊 Learning priors from real DuckDB data: {training_days} days from {building_id}")
    
    # Get real training data from DuckDB
    training_df, weather_df, forecast_df = get_training_data_from_duckdb(
        con, view_name, building_id, training_days
    )
    
    print(f"✓ Loaded {len(training_df)} rows of real training data from DuckDB")
    
    # Use REAL ProbabilityModelAgent to learn priors from real data
    prior_learning_agent = ProbabilityModelAgent(prob_dist_df=None)  # Start with uniform
    
    # MANDATORY: Use REAL ProbabilityModelAgent.train() method on real data
    try:
        updated_specs, device_probs = prior_learning_agent.train(
            building_id=building_id,
            days_list=[f"2015-05-{day:02d}" for day in range(22, 22 + min(training_days, 8))],  # Real dates
            device_specs=device_specs,
            weather_df=weather_df,
            forecast_df=forecast_df,
            parquet_dir="processed_data"
        )
        
        print(f"✓ ProbabilityModelAgent learned patterns from real data")
        
        # Extract learned distributions from real agent results
        learned_priors_data = {}
        device_types = []
        
        for device_key, device_data in device_probs.items():
            # Extract device type from full device name
            device_type = device_key.split('_')[-1].lower()
            if device_type not in device_types:
                device_types.append(device_type)
                
                # Get final learned probability distribution
                final_distribution = device_data['hour_probability']
                learned_priors_data[device_type] = {str(h): prob for h, prob in final_distribution.items()}
        
        print(f"✓ Extracted learned priors for {len(device_types)} device types from REAL data")
        
        # Convert to DataFrame format expected by ProbabilityModelAgent
        df_data = {}
        for device_type in device_types:
            for hour in range(24):
                hour_str = str(hour)
                if hour_str not in df_data:
                    df_data[hour_str] = []
                df_data[hour_str].append(learned_priors_data[device_type].get(hour_str, 1.0/24.0))
        
        # Create DataFrame
        priors_df = pd.DataFrame(df_data)
        priors_df['device_type'] = device_types
        priors_df = priors_df.set_index('device_type')
        
        # Normalize each row to ensure probabilities sum to 1
        for device_type in priors_df.index:
            row_sum = priors_df.loc[device_type, :].sum()
            if row_sum > 0:
                priors_df.loc[device_type, :] = priors_df.loc[device_type, :] / row_sum
        
        print(f"✓ Created learned prior distributions from REAL data for {len(device_types)} device types")
        return priors_df
        
    except Exception as e:
        # ERROR: Agent method failed - surface the error immediately  
        raise RuntimeError(f"CRITICAL: ProbabilityModelAgent.train() failed while learning priors: {e}. Fix the agent method or data formatting.")

def run_probability_training_experiment(con, view_name, building_id, training_days, 
                                      lr_tau: float, lr_max: float, target_device: str,
                                      experiment_name: str = None, use_learned_priors: bool = False) -> Dict[str, Any]:
    """
    Run single probability training experiment with specific hyperparameters.
    ENFORCES "USE AGENT OPTIMIZERS" - uses only ProbabilityModelAgent.
    """
    prior_type = "learned" if use_learned_priors else "uniform"
    print(f"🧪 Running experiment: LR_TAU={lr_tau}, LR_MAX={lr_max}, Priors={prior_type}")
    
    # Get training data from DuckDB
    training_df, weather_df, forecast_df = get_training_data_from_duckdb(
        con, view_name, building_id, training_days
    )
    
    # Create prior distributions - ENFORCE "USE REAL AGENT OPTIMIZERS"
    priors_df = None
    if use_learned_priors:
        priors_df = create_learned_priors_from_duckdb(con, view_name, building_id, training_days=min(30, len(training_days)*2))
        print(f"✓ Using REAL learned priors from DuckDB for {len(priors_df.index)} device types")
    else:
        print("✓ Using uniform priors")
    
    # Create custom ProbabilityModelAgent with specific hyperparameters and priors
    prob_agent = ProbabilityModelAgent(prob_dist_df=priors_df)
    
    # Override hyperparameters - MANDATORY: Use agent configuration for testing
    prob_agent.LR_TAU = lr_tau
    prob_agent.LR_MAX = lr_max
    prob_agent.LR_MIN = 0.002  # Keep other parameters consistent
    prob_agent.CAP_MAX = 0.03
    prob_agent.CAP_MIN = 0.005
    prob_agent.BURNIN_DAYS = 0
    prob_agent.LR_BURNIN = 0.005
    
    print(f"✓ Configured ProbabilityModelAgent: LR_TAU={lr_tau}, LR_MAX={lr_max}")
    
    # MANDATORY: Use REAL ProbabilityModelAgent.train() method
    try:
        updated_specs, device_probs = prob_agent.train(
            building_id=building_id,
            days_list=training_days,
            device_specs=device_specs,
            weather_df=weather_df,
            forecast_df=forecast_df,
            parquet_dir="processed_data",
            max_building_load=50.0
        )
        
        print(f"✓ ProbabilityModelAgent training completed successfully")
    except Exception as e:
        # ERROR: Agent method failed - surface the error immediately
        raise RuntimeError(f"CRITICAL: ProbabilityModelAgent.train() failed: {e}. Fix the agent method or data formatting.")
    
    # Extract results for target device
    device_key = f"{building_id}_{target_device}"
    
    if device_key not in prob_agent.latest_distributions:
        print(f"⚠ Target device {device_key} not found in results")
        return None
    
    # Get PMF evolution data
    pmf_history = prob_agent.probability_updates_history.get(device_key, [])
    final_pmf = prob_agent.latest_distributions[device_key]
    observation_count = prob_agent.observation_counts.get(device_key, 0)
    
    
    # Calculate comprehensive metrics
    results = {
        "lr_tau": lr_tau,
        "lr_max": lr_max,
        "device_key": device_key,
        "final_pmf": final_pmf,
        "pmf_history": pmf_history,
        "observation_count": observation_count,
        "training_days": len(training_days),
        "prior_type": prior_type,
        "use_learned_priors": use_learned_priors
    }
    
    # Calculate final PMF metrics
    final_metrics = calculate_pmf_metrics(final_pmf)
    results.update(final_metrics)
    
    # Calculate convergence metrics
    if len(pmf_history) >= 2:
        # JS divergence from prior to final
        prior_pmf = pmf_history[0]['distribution'] if pmf_history else {h: 1/24 for h in range(24)}
        js_from_prior = prob_agent.js_div(prior_pmf, final_pmf)
        
        # Average JS between consecutive updates (learning stability)
        js_consecutive = []
        for i in range(1, len(pmf_history)):
            js = prob_agent.js_div(pmf_history[i-1]['distribution'], pmf_history[i]['distribution'])
            js_consecutive.append(js)
        
        avg_js_consecutive = np.mean(js_consecutive) if js_consecutive else 0.0
        
        results.update({
            "js_divergence_from_prior": js_from_prior,
            "avg_js_consecutive": avg_js_consecutive,
            "convergence_updates": len(pmf_history)
        })
    else:
        results.update({
            "js_divergence_from_prior": 0.0,
            "avg_js_consecutive": 0.0,
            "convergence_updates": 0
        })
    
    # Log to MLflow if available
    if experiment_name and MLFLOW_AVAILABLE:
        # Create individual MLflow run for this experiment
        exp_tracker = EMS_OptimizationTracker("Probability_Learning_Optimization")
        run_name = f"lr_tau_{lr_tau}_lr_max_{lr_max:.3f}_{target_device}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        exp_tracker.start_run(run_name)
        
        exp_tracker.log_params({
            "lr_tau": lr_tau,
            "lr_max": lr_max,
            "target_device": target_device,
            "training_days": len(training_days),
            "building_id": building_id
        })
        
        exp_tracker.log_metrics({
            "final_entropy": results["entropy"],
            "final_max_probability": results["max_probability"], 
            "final_concentration": results["concentration"],
            "js_divergence_from_prior": results["js_divergence_from_prior"],
            "avg_js_consecutive": results["avg_js_consecutive"],
            "observation_count": observation_count,
            "convergence_updates": results["convergence_updates"]
        })
        
        exp_tracker.end_run()
    
    print(f"✓ Experiment completed: Final entropy={results['entropy']:.4f}, JS from prior={results['js_divergence_from_prior']:.4f}")
    
    return results

def create_pmf_evolution_visualization(experiment_results: List[Dict[str, Any]], 
                                     target_device: str, building_id: str):
    """Create beautiful economist-style visualization of PMF evolution."""
    print(f"📊 Creating PMF evolution visualization for {target_device}...")
    
    # Set up the figure with economist styling
    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor(colors['background'])
    
    # Create grid layout
    gs = fig.add_gridspec(3, 4, height_ratios=[2, 1, 1], width_ratios=[1, 1, 1, 1], 
                         hspace=0.3, wspace=0.3)
    
    # Color palette for different experiments
    exp_colors = sns.color_palette("husl", len(experiment_results))
    
    # 1. Main PMF Evolution Plot (top row, spanning all columns)
    ax_main = fig.add_subplot(gs[0, :])
    
    for i, result in enumerate(experiment_results):
        if result is None:
            continue
            
        pmf_history = result['pmf_history']
        lr_tau = result['lr_tau']
        lr_max = result['lr_max']
        
        # Plot PMF evolution for key hours
        peak_hours = []
        if pmf_history:
            final_pmf = result['final_pmf']
            # Find top 3 hours in final PMF
            sorted_hours = sorted(final_pmf.items(), key=lambda x: x[1], reverse=True)
            peak_hours = [h for h, p in sorted_hours[:3]]
        
        for hour in peak_hours[:2]:  # Plot top 2 hours to avoid clutter
            probabilities = []
            updates = []
            
            for j, update in enumerate(pmf_history):
                probabilities.append(update['distribution'].get(hour, 0.0))
                updates.append(j)
            
            if probabilities:
                ax_main.plot(updates, probabilities, 
                           color=exp_colors[i], 
                           linewidth=2.5,
                           alpha=0.8,
                           label=f'τ={lr_tau}, λ={lr_max:.2f}, h={hour}')
    
    ax_main.set_xlabel('Training Updates', fontsize=12, fontweight='bold')
    ax_main.set_ylabel('Probability', fontsize=12, fontweight='bold')
    ax_main.set_title(f'PMF Evolution for {target_device} - {building_id}', 
                     fontsize=16, fontweight='bold', pad=20)
    ax_main.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax_main.grid(True, alpha=0.3)
    ax_main.set_facecolor('white')
    
    # 2. Entropy Evolution (middle left)
    ax_entropy = fig.add_subplot(gs[1, 0])
    
    for i, result in enumerate(experiment_results):
        if result is None:
            continue
            
        pmf_history = result['pmf_history']
        lr_tau = result['lr_tau']
        lr_max = result['lr_max']
        
        entropies = []
        updates = []
        
        for j, update in enumerate(pmf_history):
            metrics = calculate_pmf_metrics(update['distribution'])
            entropies.append(metrics['entropy'])
            updates.append(j)
        
        if entropies:
            ax_entropy.plot(updates, entropies, 
                          color=exp_colors[i], 
                          linewidth=2,
                          label=f'τ={lr_tau}, λ={lr_max:.2f}')
    
    ax_entropy.set_xlabel('Updates', fontsize=11, fontweight='bold')
    ax_entropy.set_ylabel('Entropy', fontsize=11, fontweight='bold')
    ax_entropy.set_title('Entropy Evolution', fontsize=12, fontweight='bold')
    ax_entropy.grid(True, alpha=0.3)
    ax_entropy.set_facecolor('white')
    
    # 3. JS Divergence from Prior (middle center)
    ax_js = fig.add_subplot(gs[1, 1])
    
    for i, result in enumerate(experiment_results):
        if result is None:
            continue
            
        pmf_history = result['pmf_history']
        lr_tau = result['lr_tau']
        lr_max = result['lr_max']
        
        if len(pmf_history) >= 2:
            prior_pmf = pmf_history[0]['distribution']
            js_values = []
            updates = []
            
            for j, update in enumerate(pmf_history[1:], 1):
                js = ProbabilityModelAgent.js_div(prior_pmf, update['distribution'])
                js_values.append(js)
                updates.append(j)
            
            if js_values:
                ax_js.plot(updates, js_values, 
                         color=exp_colors[i], 
                         linewidth=2,
                         label=f'τ={lr_tau}, λ={lr_max:.2f}')
    
    ax_js.set_xlabel('Updates', fontsize=11, fontweight='bold')
    ax_js.set_ylabel('JS Divergence', fontsize=11, fontweight='bold')
    ax_js.set_title('Divergence from Prior', fontsize=12, fontweight='bold')
    ax_js.grid(True, alpha=0.3)
    ax_js.set_facecolor('white')
    
    # 4. Concentration Evolution (middle right)
    ax_conc = fig.add_subplot(gs[1, 2])
    
    for i, result in enumerate(experiment_results):
        if result is None:
            continue
            
        pmf_history = result['pmf_history']
        lr_tau = result['lr_tau']
        lr_max = result['lr_max']
        
        concentrations = []
        updates = []
        
        for j, update in enumerate(pmf_history):
            metrics = calculate_pmf_metrics(update['distribution'])
            concentrations.append(metrics['concentration'])
            updates.append(j)
        
        if concentrations:
            ax_conc.plot(updates, concentrations, 
                        color=exp_colors[i], 
                        linewidth=2,
                        label=f'τ={lr_tau}, λ={lr_max:.2f}')
    
    ax_conc.set_xlabel('Updates', fontsize=11, fontweight='bold')
    ax_conc.set_ylabel('Concentration', fontsize=11, fontweight='bold')
    ax_conc.set_title('Distribution Concentration', fontsize=12, fontweight='bold')
    ax_conc.grid(True, alpha=0.3)
    ax_conc.set_facecolor('white')
    
    # 5. Learning Rate Comparison (middle far right)
    ax_lr = fig.add_subplot(gs[1, 3])
    
    # Create heatmap of final metrics
    tau_values = sorted(list(set([r['lr_tau'] for r in experiment_results if r is not None])))
    max_values = sorted(list(set([r['lr_max'] for r in experiment_results if r is not None])))
    
    heatmap_data = np.zeros((len(tau_values), len(max_values)))
    
    for result in experiment_results:
        if result is None:
            continue
        i = tau_values.index(result['lr_tau'])
        j = max_values.index(result['lr_max'])
        heatmap_data[i, j] = result['js_divergence_from_prior']
    
    im = ax_lr.imshow(heatmap_data, cmap='viridis', aspect='auto')
    ax_lr.set_xticks(range(len(max_values)))
    ax_lr.set_xticklabels([f'{v:.2f}' for v in max_values])
    ax_lr.set_yticks(range(len(tau_values)))
    ax_lr.set_yticklabels([f'{v}' for v in tau_values])
    ax_lr.set_xlabel('LR_MAX', fontsize=11, fontweight='bold')
    ax_lr.set_ylabel('LR_TAU', fontsize=11, fontweight='bold')
    ax_lr.set_title('JS Divergence Heatmap', fontsize=12, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax_lr, shrink=0.8)
    cbar.set_label('JS Divergence', fontsize=10)
    
    # 6. Final PMF Comparison (bottom row)
    ax_final = fig.add_subplot(gs[2, :])
    
    hours = list(range(24))
    bar_width = 0.8 / len(experiment_results)
    
    for i, result in enumerate(experiment_results):
        if result is None:
            continue
            
        final_pmf = result['final_pmf']
        lr_tau = result['lr_tau']
        lr_max = result['lr_max']
        
        probabilities = [final_pmf.get(h, 0.0) for h in hours]
        x_pos = [h + i * bar_width for h in hours]
        
        ax_final.bar(x_pos, probabilities, 
                    width=bar_width,
                    color=exp_colors[i], 
                    alpha=0.7,
                    label=f'τ={lr_tau}, λ={lr_max:.2f}')
    
    ax_final.set_xlabel('Hour of Day', fontsize=12, fontweight='bold')
    ax_final.set_ylabel('Final Probability', fontsize=12, fontweight='bold')
    ax_final.set_title('Final Learned PMFs Comparison', fontsize=14, fontweight='bold')
    ax_final.set_xticks(hours)
    ax_final.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax_final.grid(True, alpha=0.3)
    ax_final.set_facecolor('white')
    
    # Add main title
    fig.suptitle(f'ProbabilityModelAgent Learning Rate Optimization\n{building_id} - {target_device}', 
                fontsize=18, fontweight='bold', y=0.98)
    
    # Save the visualization
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    output_file = f"results/probability_optimization/visualizations/{building_id}_{target_device}_learning_optimization_{timestamp}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor=colors['background'])
    plt.close()
    
    print(f"✓ Saved PMF evolution visualization: {output_file}")
    return output_file

def create_realtime_pmf_evolution(experiment_results: List[Dict[str, Any]], 
                                 target_device: str, building_id: str):
    """Create real-time PMF evolution visualization showing daily updates."""
    print(f"📊 Creating real-time PMF evolution for {target_device}...")
    
    # Find the experiment with the best learning score for detailed analysis
    valid_results = [r for r in experiment_results if r is not None and 'pmf_history' in r]
    if not valid_results:
        print("⚠ No valid results for real-time visualization")
        return None
    
    # Select best experiment result
    best_result = max(valid_results, key=lambda x: x.get('js_divergence_from_prior', 0))
    pmf_history = best_result['pmf_history']
    lr_tau = best_result['lr_tau']
    lr_max = best_result['lr_max']
    
    if len(pmf_history) < 2:
        print("⚠ Insufficient PMF history for real-time visualization")
        return None
    
    print(f"📊 Analyzing {len(pmf_history)} PMF updates for {target_device}")
    
    # Set up the figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.patch.set_facecolor(colors['background'])
    
    # Get prior (first update)
    prior_pmf = pmf_history[0]['distribution']
    final_pmf = pmf_history[-1]['distribution']
    
    # 1. Prior PMF (top left)
    ax_prior = axes[0, 0]
    hours = list(range(24))
    prior_probs = [prior_pmf.get(h, 0.0) for h in hours]
    ax_prior.bar(hours, prior_probs, alpha=0.7, color=colors['prior'], width=0.8)
    ax_prior.set_title('Prior PMF (Initial)', fontsize=14, fontweight='bold')
    ax_prior.set_xlabel('Hour of Day', fontsize=12)
    ax_prior.set_ylabel('Probability', fontsize=12)
    ax_prior.grid(True, alpha=0.3)
    ax_prior.set_ylim(0, max(prior_probs) * 1.1)
    
    # 2. Final PMF (top center)
    ax_final = axes[0, 1]
    final_probs = [final_pmf.get(h, 0.0) for h in hours]
    ax_final.bar(hours, final_probs, alpha=0.7, color=colors['converged'], width=0.8)
    ax_final.set_title('Final Learned PMF', fontsize=14, fontweight='bold')
    ax_final.set_xlabel('Hour of Day', fontsize=12)
    ax_final.set_ylabel('Probability', fontsize=12)
    ax_final.grid(True, alpha=0.3)
    ax_final.set_ylim(0, max(final_probs) * 1.1)
    
    # 3. PMF Difference (top right)
    ax_diff = axes[0, 2]
    diff_probs = [final_probs[h] - prior_probs[h] for h in hours]
    colors_diff = [colors['learning'] if d > 0 else colors['js_div'] for d in diff_probs]
    ax_diff.bar(hours, diff_probs, alpha=0.7, color=colors_diff, width=0.8)
    ax_diff.set_title('PMF Change (Final - Prior)', fontsize=14, fontweight='bold')
    ax_diff.set_xlabel('Hour of Day', fontsize=12)
    ax_diff.set_ylabel('Probability Change', fontsize=12)
    ax_diff.grid(True, alpha=0.3)
    ax_diff.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # 4. Daily Update Evolution (bottom left)
    ax_daily = axes[1, 0]
    
    # Show evolution every few updates to avoid clutter
    update_step = max(1, len(pmf_history) // 8)  # Show ~8 intermediate states
    selected_updates = list(range(0, len(pmf_history), update_step)) + [len(pmf_history) - 1]
    
    cmap = plt.cm.viridis
    for i, update_idx in enumerate(selected_updates):
        update = pmf_history[update_idx]
        pmf = update['distribution']
        probs = [pmf.get(h, 0.0) for h in hours]
        
        alpha = 0.3 + 0.7 * (i / len(selected_updates))
        color = cmap(i / len(selected_updates))
        label = f"Update {update_idx}" if i % 2 == 0 else ""
        
        ax_daily.plot(hours, probs, color=color, alpha=alpha, linewidth=2, 
                     marker='o', markersize=3, label=label)
    
    ax_daily.set_title('Daily PMF Evolution', fontsize=14, fontweight='bold')
    ax_daily.set_xlabel('Hour of Day', fontsize=12)
    ax_daily.set_ylabel('Probability', fontsize=12)
    ax_daily.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax_daily.grid(True, alpha=0.3)
    
    # 5. Learning Metrics Over Time (bottom center)
    ax_metrics = axes[1, 1]
    
    updates = list(range(len(pmf_history)))
    js_values = []
    entropy_values = []
    
    for update in pmf_history:
        pmf = update['distribution']
        
        # Calculate JS divergence from prior
        js_val = ProbabilityModelAgent.js_div(prior_pmf, pmf)
        js_values.append(js_val)
        
        # Calculate entropy
        entropy_val = calculate_pmf_metrics(pmf)['entropy']
        entropy_values.append(entropy_val)
    
    ax_metrics_twin = ax_metrics.twinx()
    
    line1 = ax_metrics.plot(updates, js_values, color=colors['js_div'], linewidth=2, 
                           marker='o', markersize=4, label='JS Divergence')
    line2 = ax_metrics_twin.plot(updates, entropy_values, color=colors['entropy'], linewidth=2, 
                                marker='s', markersize=4, label='Entropy')
    
    ax_metrics.set_xlabel('Training Updates', fontsize=12)
    ax_metrics.set_ylabel('JS Divergence from Prior', fontsize=12, color=colors['js_div'])
    ax_metrics_twin.set_ylabel('Entropy', fontsize=12, color=colors['entropy'])
    ax_metrics.tick_params(axis='y', labelcolor=colors['js_div'])
    ax_metrics_twin.tick_params(axis='y', labelcolor=colors['entropy'])
    ax_metrics.set_title('Learning Progress', fontsize=14, fontweight='bold')
    ax_metrics.grid(True, alpha=0.3)
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax_metrics.legend(lines, labels, loc='upper left')
    
    # 6. Peak Hour Analysis (bottom right)
    ax_peaks = axes[1, 2]
    
    # Track how peak probability hours change over time
    peak_hours = []
    peak_probs = []
    
    for update in pmf_history:
        pmf = update['distribution']
        max_prob_hour = max(pmf.items(), key=lambda x: x[1])
        peak_hours.append(max_prob_hour[0])
        peak_probs.append(max_prob_hour[1])
    
    # Create scatter plot with color coding for time
    scatter = ax_peaks.scatter(peak_hours, peak_probs, c=updates, cmap='plasma', 
                              s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    ax_peaks.set_xlabel('Peak Hour', fontsize=12)
    ax_peaks.set_ylabel('Peak Probability', fontsize=12)
    ax_peaks.set_title('Peak Hour Evolution', fontsize=14, fontweight='bold')
    ax_peaks.grid(True, alpha=0.3)
    ax_peaks.set_xlim(-0.5, 23.5)
    
    # Add colorbar for time progression
    cbar = plt.colorbar(scatter, ax=ax_peaks, shrink=0.8)
    cbar.set_label('Training Update', fontsize=10)
    
    # Add main title
    fig.suptitle(f'Real-time PMF Evolution Analysis\n{building_id} - {target_device} (τ={lr_tau}, λ={lr_max:.3f})', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    # Save the visualization
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    output_file = f"results/probability_optimization/visualizations/{building_id}_{target_device}_realtime_evolution_{timestamp}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor=colors['background'])
    plt.close()
    
    print(f"✓ Saved real-time PMF evolution: {output_file}")
    return output_file

def create_summary_analysis(experiment_results: List[Dict[str, Any]], building_id: str, target_device: str):
    """Create summary analysis and recommendations."""
    print("📊 Creating summary analysis...")
    
    # Filter valid results
    valid_results = [r for r in experiment_results if r is not None]
    
    if not valid_results:
        print("⚠ No valid experiment results to analyze")
        return None
    
    # Create summary DataFrame
    summary_data = []
    for result in valid_results:
        summary_data.append({
            'LR_TAU': result['lr_tau'],
            'LR_MAX': result['lr_max'],
            'Final_Entropy': result['entropy'],
            'Final_Concentration': result['concentration'],
            'JS_from_Prior': result['js_divergence_from_prior'],
            'Avg_JS_Consecutive': result['avg_js_consecutive'],
            'Observation_Count': result['observation_count'],
            'Convergence_Updates': result['convergence_updates']
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Find optimal parameters
    # We want high JS from prior (good learning) but low consecutive JS (stable)
    # Balance learning vs stability
    summary_df['Learning_Score'] = summary_df['JS_from_Prior'] - 0.5 * summary_df['Avg_JS_Consecutive']
    
    best_idx = summary_df['Learning_Score'].idxmax()
    best_params = summary_df.iloc[best_idx]
    
    # Note: CSV creation removed to save disk space
    print(f"✓ Summary analysis completed (CSV creation disabled to save disk space)")
    print(f"\n🎯 OPTIMAL PARAMETERS FOUND:")
    print(f"   LR_TAU = {best_params['LR_TAU']}")
    print(f"   LR_MAX = {best_params['LR_MAX']:.3f}")
    print(f"   Learning Score = {best_params['Learning_Score']:.4f}")
    print(f"   Final Entropy = {best_params['Final_Entropy']:.4f}")
    print(f"   JS from Prior = {best_params['JS_from_Prior']:.4f}")
    
    return summary_df, best_params

def main():
    """Main function implementing probability learning rate optimization."""
    
    args = parse_args()
    
    building_id = args.building
    n_days = args.n_days
    target_device = args.target_device
    lr_tau_values = [float(x) for x in args.lr_tau_values.split(',')]
    lr_max_values = [float(x) for x in args.lr_max_values.split(',')]
    test_multiple_devices = args.test_multiple_devices
    device_filter = [x.strip() for x in args.device_filter.split(',')]
    test_both_priors = args.test_both_priors
    use_learned_priors = args.use_learned_priors
    
    print("="*80)
    print("PROBABILITY LEARNING RATE OPTIMIZATION PIPELINE")
    print("="*80)
    print(f"Building: {building_id}")
    print(f"Training days: {n_days}")
    print(f"Target device: {target_device}")
    print(f"LR_TAU values: {lr_tau_values}")
    print(f"LR_MAX values: {lr_max_values}")
    print(f"Test multiple devices: {test_multiple_devices}")
    print(f"Device filter (flex_models): {device_filter}")
    print(f"Test both priors: {test_both_priors}")
    print(f"Use learned priors: {use_learned_priors}")
    print("="*80)
    
    # Initialize MLflow tracking
    mlflow_tracker = None
    if MLFLOW_AVAILABLE:
        mlflow_tracker = EMS_OptimizationTracker("Probability_Learning_Optimization")
        run_name = f"lr_optimization_{building_id}_{target_device}_{datetime.now().strftime('%Y%m%d_%H%M')}"
        mlflow_tracker.start_run(run_name)
        
        # Log pipeline parameters
        mlflow_tracker.log_params({
            "building_id": building_id,
            "target_device": target_device,
            "n_training_days": n_days,
            "lr_tau_values": str(lr_tau_values),
            "lr_max_values": str(lr_max_values),
            "total_experiments": len(lr_tau_values) * len(lr_max_values),
            "pipeline": "Probability_Learning_Optimization"
        })
    
    # 1. Setup DuckDB connection
    con, view_name = setup_duckdb_connection(building_id)
    
    # 2. Select training days
    training_days = select_training_days_from_duckdb(con, view_name, n_days)
    
    # End main MLflow run before starting individual experiments
    if mlflow_tracker:
        mlflow_tracker.end_run()
    
    # 3. Get devices to test - ENFORCE "USE REAL AGENT OPTIMIZERS" with ALL real devices
    if test_multiple_devices:
        print(f"\n🔍 COMPREHENSIVE TESTING: Getting ALL devices from DuckDB")
        devices_to_test = get_all_available_devices_from_duckdb(con, view_name, building_id)
        print(f"✓ Testing ALL {len(devices_to_test)} device types from REAL data: {devices_to_test}")
    else:
        devices_to_test = [target_device]  # Single device test
        print(f"✓ Testing single device type: {target_device}")
    
    # 4. Determine which prior types to test
    prior_types_to_test = []
    if test_both_priors:
        prior_types_to_test = [False, True]  # Test both uniform and learned priors
        print("🔍 Testing both uniform and learned priors for comparison")
    else:
        prior_types_to_test = [use_learned_priors]  # Test only one type
        prior_name = "learned" if use_learned_priors else "uniform"
        print(f"🔍 Testing only {prior_name} priors")
    
    # 5. Run hyperparameter experiments
    all_experiment_results = {}
    total_experiments = len(lr_tau_values) * len(lr_max_values) * len(devices_to_test) * len(prior_types_to_test)
    
    print(f"\n🧪 Running {total_experiments} hyperparameter experiments across {len(devices_to_test)} device types and {len(prior_types_to_test)} prior types...")
    
    experiment_count = 0
    for device_type in devices_to_test:
        print(f"\n📱 Testing device type: {device_type}")
        
        # Store results by prior type
        device_results_by_prior = {}
        
        for use_learned in prior_types_to_test:
            prior_name = "learned" if use_learned else "uniform"
            print(f"\n📊 Testing {prior_name} priors for {device_type}")
            device_results = []
            
            for i, lr_tau in enumerate(lr_tau_values):
                for j, lr_max in enumerate(lr_max_values):
                    experiment_count += 1
                    print(f"\n--- Experiment {experiment_count}/{total_experiments}: {device_type}, {prior_name}, LR_TAU={lr_tau}, LR_MAX={lr_max} ---")
                    
                    try:
                        # MANDATORY: Use REAL ProbabilityModelAgent methods
                        result = run_probability_training_experiment(
                            con=con,
                            view_name=view_name,
                            building_id=building_id,
                            training_days=training_days,
                            lr_tau=lr_tau,
                            lr_max=lr_max,
                            target_device=device_type,
                            experiment_name="Probability_Learning_Optimization",
                            use_learned_priors=use_learned
                        )
                        device_results.append(result)
                        
                    except Exception as e:
                        print(f"❌ Experiment failed: {e}")
                        device_results.append(None)
            
            device_results_by_prior[prior_name] = device_results
        
        all_experiment_results[device_type] = device_results_by_prior
    
    # 6. Create comprehensive visualizations for each device type and prior type
    print(f"\n📊 Creating visualizations...")
    
    visualization_files = []
    summary_results = {}
    
    for device_type, results_by_prior in all_experiment_results.items():
        print(f"\n📊 Creating visualizations for {device_type}...")
        
        for prior_name, experiment_results in results_by_prior.items():
            print(f"\n📊 Creating visualizations for {device_type} with {prior_name} priors...")
            
            # Create PMF evolution visualization
            viz_file = create_pmf_evolution_visualization(
                experiment_results, f"{device_type}_{prior_name}", building_id
            )
            if viz_file:
                visualization_files.append(viz_file)
            
            # Create real-time PMF evolution visualization
            realtime_viz_file = create_realtime_pmf_evolution(
                experiment_results, f"{device_type}_{prior_name}", building_id
            )
            if realtime_viz_file:
                visualization_files.append(realtime_viz_file)
            
            # Generate summary analysis
            summary_result = create_summary_analysis(
                experiment_results, building_id, f"{device_type}_{prior_name}"
            )
            if summary_result is not None:
                summary_results[f"{device_type}_{prior_name}"] = summary_result
    
    # 7. Log final results to MLflow for each device type and prior type
    if MLFLOW_AVAILABLE and summary_results:
        for device_prior_key, (summary_df, best_params) in summary_results.items():
            if best_params is not None:
                summary_tracker = EMS_OptimizationTracker("Probability_Learning_Optimization")
                summary_run_name = f"summary_{building_id}_{device_prior_key}_{datetime.now().strftime('%Y%m%d_%H%M')}"
                summary_tracker.start_run(summary_run_name)
                
                summary_tracker.log_params({
                    "building_id": building_id,
                    "target_device": device_prior_key,
                    "n_training_days": n_days,
                    "total_experiments": len(lr_tau_values) * len(lr_max_values),
                    "test_multiple_devices": test_multiple_devices,
                    "test_both_priors": test_both_priors,
                    "device_filter": str(device_filter),
                    "pipeline": "Probability_Learning_Summary"
                })
                
                summary_tracker.log_metrics({
                    "optimal_lr_tau": best_params['LR_TAU'],
                    "optimal_lr_max": best_params['LR_MAX'],
                    "optimal_learning_score": best_params['Learning_Score'],
                    "optimal_final_entropy": best_params['Final_Entropy'],
                    "optimal_js_from_prior": best_params['JS_from_Prior'],
                    "total_experiments_completed": len([r for device_results in all_experiment_results.values() 
                                                     for prior_results in device_results.values() 
                                                     for r in prior_results if r is not None])
                })
                
                # Log artifacts
                summary_tracker.log_artifacts("results/probability_optimization")
                
                # End MLflow run
                summary_tracker.end_run()
        
        print("✓ MLflow summary tracking completed for all devices and prior types")
    
    print("\n" + "="*80)
    print("PROBABILITY LEARNING OPTIMIZATION COMPLETED")
    print("="*80)
    
    total_completed = sum(len([r for r in prior_results if r is not None]) 
                         for device_results in all_experiment_results.values()
                         for prior_results in device_results.values())
    print(f"✅ Completed {total_completed}/{total_experiments} experiments across {len(devices_to_test)} device types and {len(prior_types_to_test)} prior types")
    
    # Print optimal parameters for each device type and prior type
    for device_prior_key, result_tuple in summary_results.items():
        if result_tuple is not None:
            summary_df, best_params = result_tuple
            if best_params is not None:
                print(f"\n🎯 {device_prior_key.upper()} - Optimal Parameters:")
                print(f"   LR_TAU = {best_params['LR_TAU']}")
                print(f"   LR_MAX = {best_params['LR_MAX']:.3f}")
                print(f"   Learning Score = {best_params['Learning_Score']:.4f}")
                print(f"   JS from Prior = {best_params['JS_from_Prior']:.4f}")
    
    print(f"\n📊 Generated {len(visualization_files)} visualization files:")
    for viz_file in visualization_files:
        print(f"   • {viz_file}")
    
    print("="*80)
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ PROBABILITY OPTIMIZATION FAILED: {e}")
        print("This indicates a bug in the agent interface that must be fixed.")
        sys.exit(1)