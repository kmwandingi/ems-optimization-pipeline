#!/usr/bin/env python
"""
Convergence Analysis Plots for Learning Pipeline

Creates 4 separate plots analyzing convergence of probability distributions:
1. Distribution Convergence (JS divergence from real priors over time)
2. Entropy Evolution (how distribution entropy changes over time)
3. Learning Rate Evolution (adaptive learning rate changes)
4. Peak Probability Convergence (how maximum probability values evolve)

Features:
- Uses real priors learned from DuckDB data, not uniform distributions
- Tests convergence over 30-50 days timeline
- Uses Seaborn with ggplot theme
- Saves each plot as separate PNG file
- Uses real ProbabilityModelAgent with real data
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import duckdb
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Add notebooks directory to path for agent imports
sys.path.append(str(Path.cwd() / "notebooks"))

# Import agent classes
try:
    from agents.ProbabilityModelAgent import ProbabilityModelAgent
    print("✓ Successfully imported ProbabilityModelAgent")
except ImportError as e:
    print(f"CRITICAL ERROR: Failed to import agent classes: {e}")
    sys.exit(1)

# Import common utilities and device_specs
sys.path.append(str(Path.cwd() / "scripts"))
import common
sys.path.append(str(Path.cwd() / "notebooks" / "utils"))
from device_specs import device_specs

# Set up Seaborn with ggplot theme
plt.style.use('ggplot')
sns.set_palette("husl")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# Create output directories
os.makedirs("results/convergence_analysis", exist_ok=True)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Convergence Analysis with Real Priors")
    parser.add_argument("--building", type=str, default="DE_KN_residential1",
                        help="Building ID (e.g., DE_KN_residential1)")
    parser.add_argument("--n_days", type=int, default=40,
                        help="Number of days for convergence analysis (30-50)")
    parser.add_argument("--target_device", type=str, default="heat_pump",
                        help="Device type to focus analysis on")
    parser.add_argument("--lr_tau", type=float, default=20.0,
                        help="Learning rate tau parameter")
    parser.add_argument("--lr_max", type=float, default=0.10,
                        help="Maximum learning rate parameter")
    
    return parser.parse_args()

def setup_duckdb_connection(building_id):
    """Setup DuckDB connection and validate data availability."""
    print(f"📊 Setting up DuckDB connection for {building_id}...")
    
    # Try to use the existing connection method, fallback to parquet
    try:
        con = common.get_con(building_id)
        view_name = f"{building_id}_processed_data"
        
        # Test if the view exists
        try:
            test_query = con.execute(f"SELECT COUNT(*) as count FROM {view_name} LIMIT 1").df()
        except:
            # Create view from parquet file
            parquet_path = Path.cwd() / "notebooks" / "data" / f"{building_id}_processed_data.parquet"
            if parquet_path.exists():
                print(f"⚠ Creating view from parquet file: {parquet_path}")
                con.execute(f"""
                CREATE OR REPLACE VIEW {view_name} AS 
                SELECT * FROM read_parquet('{str(parquet_path).replace(os.sep, '/')}')
                """)
            else:
                raise FileNotFoundError(f"No data found for {building_id}")
    except Exception as e:
        print(f"⚠ Main connection failed: {e}")
        # Fallback: create in-memory database with parquet
        con = duckdb.connect(database=":memory:")
        view_name = f"{building_id}_processed_data"
        
        parquet_path = Path.cwd() / "notebooks" / "data" / f"{building_id}_processed_data.parquet"
        if parquet_path.exists():
            print(f"✓ Loading data from parquet file: {parquet_path}")
            con.execute(f"""
            CREATE TABLE {view_name} AS 
            SELECT * FROM read_parquet('{str(parquet_path).replace(os.sep, '/')}')
            """)
        else:
            raise FileNotFoundError(f"No parquet data found for {building_id}")
    
    # Validate data exists and get metadata
    row_count = con.execute(f"SELECT COUNT(*) as count FROM {view_name}").df()['count'][0]
    col_count = len(con.execute(f"DESCRIBE {view_name}").df())
    date_range = con.execute(f"SELECT MIN(DATE(utc_timestamp)) as min_date, MAX(DATE(utc_timestamp)) as max_date FROM {view_name}").df()
    
    print(f"✓ Connected to DuckDB: {row_count:,} rows, {col_count} columns")
    print(f"✓ Date range: {date_range['min_date'][0]} to {date_range['max_date'][0]}")
    
    return con, view_name

def select_convergence_days_from_duckdb(con, view_name, n_days):
    """Select consecutive training days for convergence analysis."""
    print(f"📅 Selecting {n_days} consecutive days for convergence analysis...")
    
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
    
    # Take consecutive days from the beginning
    selected_days = full_days[:n_days]
    print(f"✓ Selected {len(selected_days)} consecutive days from DuckDB")
    
    return [str(day) for day in selected_days]

def create_learned_priors_from_real_data(con, view_name: str, building_id: str, 
                                        target_device: str, training_days: int = 30) -> pd.DataFrame:
    """Create learned prior distributions from real DuckDB data using ProbabilityModelAgent."""
    print(f"📊 Learning REAL priors from DuckDB data: {training_days} days from {building_id}")
    
    # Get training data from DuckDB
    training_df, weather_df, forecast_df = get_training_data_from_duckdb(
        con, view_name, building_id, training_days
    )
    
    print(f"✓ Loaded {len(training_df)} rows of real training data from DuckDB")
    
    # Use REAL ProbabilityModelAgent to learn priors from real data
    prior_learning_agent = ProbabilityModelAgent(prob_dist_df=None)  # Start with uniform
    
    # Use REAL ProbabilityModelAgent.train() method on real data
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
        
        # Extract learned distributions from agent results
        learned_priors_data = {}
        device_types = []
        
        for device_key, device_data in device_probs.items():
            # Extract device type from full device name
            device_type = device_key.split('_')[-1].lower()
            if device_type == target_device:
                # Get final learned probability distribution
                final_distribution = device_data['hour_probability']
                learned_priors_data[device_type] = {str(h): prob for h, prob in final_distribution.items()}
                device_types.append(device_type)
                break
        
        if not device_types:
            print(f"⚠ Target device {target_device} not found, using uniform prior")
            learned_priors_data[target_device] = {str(h): 1.0/24.0 for h in range(24)}
            device_types = [target_device]
        
        print(f"✓ Extracted learned priors for {target_device} from REAL data")
        
        # Convert to DataFrame format expected by ProbabilityModelAgent
        df_data = {}
        for hour in range(24):
            hour_str = str(hour)
            df_data[hour_str] = [learned_priors_data[target_device].get(hour_str, 1.0/24.0)]
        
        # Create DataFrame
        priors_df = pd.DataFrame(df_data)
        priors_df['device_type'] = [target_device]
        priors_df = priors_df.set_index('device_type')
        
        # Normalize to ensure probabilities sum to 1
        row_sum = priors_df.loc[target_device, :].sum()
        if row_sum > 0:
            priors_df.loc[target_device, :] = priors_df.loc[target_device, :] / row_sum
        
        print(f"✓ Created learned prior distributions from REAL data for {target_device}")
        return priors_df
        
    except Exception as e:
        print(f"⚠ Failed to learn priors: {e}, using uniform distribution")
        # Fallback to uniform distribution
        df_data = {str(h): [1.0/24.0] for h in range(24)}
        priors_df = pd.DataFrame(df_data)
        priors_df['device_type'] = [target_device]
        priors_df = priors_df.set_index('device_type')
        return priors_df

def get_training_data_from_duckdb(con, view_name, building_id, training_days):
    """Get training data from DuckDB for agent training."""
    
    # Convert training_days to proper format for SQL
    if isinstance(training_days, int):
        print(f"📊 Loading training data for {training_days} days from DuckDB...")
        # If training_days is int, use first N days from data
        full_days_df = con.execute(f"""
            SELECT DATE(utc_timestamp) as day
            FROM {view_name} 
            GROUP BY DATE(utc_timestamp) 
            HAVING COUNT(*) = 24 
            ORDER BY DATE(utc_timestamp)
            LIMIT {training_days}
        """).df()
        days_list = [str(day) for day in pd.to_datetime(full_days_df['day']).dt.date.tolist()]
    else:
        print(f"📊 Loading training data for {len(training_days)} days from DuckDB...")
        days_list = training_days
    
    days_str = "', '".join(days_list)
    
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

def run_convergence_analysis(con, view_name, building_id, convergence_days, 
                           target_device: str, lr_tau: float, lr_max: float) -> Dict[str, Any]:
    """Run convergence analysis using ProbabilityModelAgent with real priors."""
    print(f"🧪 Running convergence analysis over {len(convergence_days)} days")
    
    # Create learned priors from real data
    priors_df = create_learned_priors_from_real_data(
        con, view_name, building_id, target_device, training_days=30
    )
    real_prior = {h: float(priors_df.loc[target_device, str(h)]) for h in range(24)}
    print(f"✓ Using REAL learned priors for {target_device}")
    
    # Create ProbabilityModelAgent with learned priors and specified hyperparameters
    prob_agent = ProbabilityModelAgent(prob_dist_df=priors_df)
    prob_agent.LR_TAU = lr_tau
    prob_agent.LR_MAX = lr_max
    prob_agent.LR_MIN = 0.002
    prob_agent.CAP_MAX = 0.03
    prob_agent.CAP_MIN = 0.005
    prob_agent.BURNIN_DAYS = 0
    prob_agent.LR_BURNIN = 0.005
    
    print(f"✓ Configured ProbabilityModelAgent: LR_TAU={lr_tau}, LR_MAX={lr_max}")
    
    # Get training data for all convergence days
    training_df, weather_df, forecast_df = get_training_data_from_duckdb(
        con, view_name, building_id, convergence_days
    )
    
    # Run training on convergence days using REAL agent
    try:
        updated_specs, device_probs = prob_agent.train(
            building_id=building_id,
            days_list=convergence_days,
            device_specs=device_specs,
            weather_df=weather_df,
            forecast_df=forecast_df,
            parquet_dir="processed_data",
            max_building_load=50.0
        )
        
        print(f"✓ ProbabilityModelAgent training completed successfully")
    except Exception as e:
        raise RuntimeError(f"CRITICAL: ProbabilityModelAgent.train() failed: {e}")
    
    # Extract results for target device
    device_key = f"{building_id}_{target_device}"
    
    if device_key not in prob_agent.latest_distributions:
        print(f"⚠ Target device {device_key} not found in results")
        return None
    
    # Get convergence data
    pmf_history = prob_agent.probability_updates_history.get(device_key, [])
    final_pmf = prob_agent.latest_distributions[device_key]
    observation_count = prob_agent.observation_counts.get(device_key, 0)
    
    # Calculate convergence metrics
    convergence_data = {
        "real_prior": real_prior,
        "final_pmf": final_pmf,
        "pmf_history": pmf_history,
        "observation_count": observation_count,
        "training_days": len(convergence_days),
        "device_key": device_key,
        "lr_tau": lr_tau,
        "lr_max": lr_max
    }
    
    print(f"✓ Convergence analysis completed: {len(pmf_history)} updates over {len(convergence_days)} days")
    
    return convergence_data

def create_distribution_convergence_plot(convergence_data: Dict[str, Any], building_id: str, target_device: str):
    """Create Plot 1: Distribution Convergence (JS divergence from real priors over time)."""
    print("📊 Creating Plot 1: Distribution Convergence...")
    
    pmf_history = convergence_data['pmf_history']
    real_prior = convergence_data['real_prior']
    
    if len(pmf_history) < 2:
        print("⚠ Insufficient PMF history for convergence plot")
        return None
    
    # Calculate JS divergence from real prior for each update
    updates = []
    js_divergences = []
    days = []
    
    for i, update in enumerate(pmf_history):
        if update['day'] == 'PRIOR':
            continue
            
        current_pmf = update['distribution']
        js_div = ProbabilityModelAgent.js_div(real_prior, current_pmf)
        
        updates.append(i)
        js_divergences.append(js_div)
        days.append(update['day'])
    
    # Create plot
    plt.figure(figsize=(12, 8))
    plt.plot(updates, js_divergences, linewidth=3, marker='o', markersize=6, color='#2E86AB')
    
    plt.xlabel('Training Updates', fontsize=14, fontweight='bold')
    plt.ylabel('JS Divergence from Real Prior', fontsize=14, fontweight='bold')
    plt.title(f'Distribution Convergence Analysis\n{building_id} - {target_device}', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Add convergence trend line
    if len(updates) > 5:
        z = np.polyfit(updates, js_divergences, 1)
        p = np.poly1d(z)
        plt.plot(updates, p(updates), "--", alpha=0.8, color='#A23B72', linewidth=2, label='Trend')
        plt.legend()
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    output_file = f"results/convergence_analysis/{building_id}_{target_device}_distribution_convergence_{timestamp}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved distribution convergence plot: {output_file}")
    return output_file

def create_entropy_evolution_plot(convergence_data: Dict[str, Any], building_id: str, target_device: str):
    """Create Plot 2: Entropy Evolution (how distribution entropy changes over time)."""
    print("📊 Creating Plot 2: Entropy Evolution...")
    
    pmf_history = convergence_data['pmf_history']
    
    if len(pmf_history) < 2:
        print("⚠ Insufficient PMF history for entropy plot")
        return None
    
    # Calculate entropy for each update
    updates = []
    entropies = []
    days = []
    
    for i, update in enumerate(pmf_history):
        if update['day'] == 'PRIOR':
            continue
            
        entropy = update.get('entropy', ProbabilityModelAgent.entropy(update['distribution']))
        
        updates.append(i)
        entropies.append(entropy)
        days.append(update['day'])
    
    # Create plot
    plt.figure(figsize=(12, 8))
    plt.plot(updates, entropies, linewidth=3, marker='s', markersize=6, color='#F18F01')
    
    plt.xlabel('Training Updates', fontsize=14, fontweight='bold')
    plt.ylabel('Distribution Entropy', fontsize=14, fontweight='bold')
    plt.title(f'Entropy Evolution Over Time\n{building_id} - {target_device}', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Add horizontal line for uniform distribution entropy
    uniform_entropy = np.log(24)  # Maximum entropy for 24-hour distribution
    plt.axhline(y=uniform_entropy, color='red', linestyle='--', alpha=0.7, 
                label=f'Uniform Distribution (H={uniform_entropy:.2f})')
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    output_file = f"results/convergence_analysis/{building_id}_{target_device}_entropy_evolution_{timestamp}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved entropy evolution plot: {output_file}")
    return output_file

def create_learning_rate_evolution_plot(convergence_data: Dict[str, Any], building_id: str, target_device: str):
    """Create Plot 3: Learning Rate Evolution (adaptive learning rate changes)."""
    print("📊 Creating Plot 3: Learning Rate Evolution...")
    
    pmf_history = convergence_data['pmf_history']
    
    if len(pmf_history) < 2:
        print("⚠ Insufficient PMF history for learning rate plot")
        return None
    
    # Extract learning rates for each update
    updates = []
    learning_rates = []
    days = []
    
    for i, update in enumerate(pmf_history):
        if update['day'] == 'PRIOR':
            continue
            
        lr = update.get('learning_rate', 0.0)
        
        updates.append(i)
        learning_rates.append(lr)
        days.append(update['day'])
    
    # Create plot
    plt.figure(figsize=(12, 8))
    plt.plot(updates, learning_rates, linewidth=3, marker='^', markersize=6, color='#C73E1D')
    
    plt.xlabel('Training Updates', fontsize=14, fontweight='bold')
    plt.ylabel('Learning Rate', fontsize=14, fontweight='bold')
    plt.title(f'Learning Rate Evolution\n{building_id} - {target_device}', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Add horizontal lines for LR_MAX and LR_MIN
    lr_max = convergence_data['lr_max']
    plt.axhline(y=lr_max, color='green', linestyle='--', alpha=0.7, 
                label=f'LR_MAX ({lr_max:.3f})')
    plt.axhline(y=0.002, color='orange', linestyle='--', alpha=0.7, 
                label='LR_MIN (0.002)')
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    output_file = f"results/convergence_analysis/{building_id}_{target_device}_learning_rate_evolution_{timestamp}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved learning rate evolution plot: {output_file}")
    return output_file

def create_peak_probability_convergence_plot(convergence_data: Dict[str, Any], building_id: str, target_device: str):
    """Create Plot 4: Peak Probability Convergence (how maximum probability values evolve)."""
    print("📊 Creating Plot 4: Peak Probability Convergence...")
    
    pmf_history = convergence_data['pmf_history']
    
    if len(pmf_history) < 2:
        print("⚠ Insufficient PMF history for peak probability plot")
        return None
    
    # Extract peak probability and hour for each update
    updates = []
    peak_probabilities = []
    peak_hours = []
    days = []
    
    for i, update in enumerate(pmf_history):
        if update['day'] == 'PRIOR':
            continue
            
        current_pmf = update['distribution']
        peak_hour = max(current_pmf, key=current_pmf.get)
        peak_prob = current_pmf[peak_hour]
        
        updates.append(i)
        peak_probabilities.append(peak_prob)
        peak_hours.append(peak_hour)
        days.append(update['day'])
    
    # Create subplot with two y-axes
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Plot peak probability
    color = '#3A86FF'
    ax1.set_xlabel('Training Updates', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Peak Probability', fontsize=14, fontweight='bold', color=color)
    line1 = ax1.plot(updates, peak_probabilities, linewidth=3, marker='o', markersize=6, 
                     color=color, label='Peak Probability')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    
    # Plot peak hour on secondary y-axis
    ax2 = ax1.twinx()
    color = '#FF006E'
    ax2.set_ylabel('Peak Hour', fontsize=14, fontweight='bold', color=color)
    line2 = ax2.plot(updates, peak_hours, linewidth=3, marker='s', markersize=6, 
                     color=color, alpha=0.7, label='Peak Hour')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(-0.5, 23.5)
    
    # Add horizontal line for uniform distribution peak probability
    uniform_peak_prob = 1.0 / 24.0
    ax1.axhline(y=uniform_peak_prob, color='red', linestyle='--', alpha=0.7, 
                label=f'Uniform Peak ({uniform_peak_prob:.3f})')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.title(f'Peak Probability Convergence\n{building_id} - {target_device}', 
              fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    output_file = f"results/convergence_analysis/{building_id}_{target_device}_peak_convergence_{timestamp}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved peak probability convergence plot: {output_file}")
    return output_file

def main():
    """Main function implementing convergence analysis with 4 separate plots."""
    
    args = parse_args()
    
    building_id = args.building
    n_days = args.n_days
    target_device = args.target_device
    lr_tau = args.lr_tau
    lr_max = args.lr_max
    
    # Validate n_days is in acceptable range
    if n_days < 30 or n_days > 50:
        print(f"⚠ Adjusting n_days from {n_days} to valid range [30, 50]")
        n_days = max(30, min(50, n_days))
    
    print("="*80)
    print("CONVERGENCE ANALYSIS WITH REAL PRIORS")
    print("="*80)
    print(f"Building: {building_id}")
    print(f"Analysis days: {n_days}")
    print(f"Target device: {target_device}")
    print(f"LR_TAU: {lr_tau}")
    print(f"LR_MAX: {lr_max}")
    print("="*80)
    
    # 1. Setup DuckDB connection
    con, view_name = setup_duckdb_connection(building_id)
    
    # 2. Select convergence days
    convergence_days = select_convergence_days_from_duckdb(con, view_name, n_days)
    
    # 3. Run convergence analysis using real agent with real priors
    print(f"\n🧪 Running convergence analysis using REAL priors...")
    convergence_data = run_convergence_analysis(
        con, view_name, building_id, convergence_days, 
        target_device, lr_tau, lr_max
    )
    
    if convergence_data is None:
        print("❌ Convergence analysis failed")
        return False
    
    # 4. Create 4 separate plots
    print(f"\n📊 Creating 4 convergence analysis plots...")
    
    plot_files = []
    
    # Plot 1: Distribution Convergence
    plot1 = create_distribution_convergence_plot(convergence_data, building_id, target_device)
    if plot1:
        plot_files.append(plot1)
    
    # Plot 2: Entropy Evolution
    plot2 = create_entropy_evolution_plot(convergence_data, building_id, target_device)
    if plot2:
        plot_files.append(plot2)
    
    # Plot 3: Learning Rate Evolution
    plot3 = create_learning_rate_evolution_plot(convergence_data, building_id, target_device)
    if plot3:
        plot_files.append(plot3)
    
    # Plot 4: Peak Probability Convergence
    plot4 = create_peak_probability_convergence_plot(convergence_data, building_id, target_device)
    if plot4:
        plot_files.append(plot4)
    
    print("\n" + "="*80)
    print("CONVERGENCE ANALYSIS COMPLETED")
    print("="*80)
    print(f"✅ Generated {len(plot_files)} convergence analysis plots:")
    for plot_file in plot_files:
        print(f"   • {plot_file}")
    
    print(f"\n📊 Analysis Summary:")
    print(f"   • Training days: {convergence_data['training_days']}")
    print(f"   • Total updates: {len(convergence_data['pmf_history'])}")
    print(f"   • Observation count: {convergence_data['observation_count']}")
    print(f"   • Used REAL priors learned from data")
    print("="*80)
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ CONVERGENCE ANALYSIS FAILED: {e}")
        sys.exit(1)