#!/usr/bin/env python
"""
Simplified Convergence Analysis Plots for Learning Pipeline

Creates 4 separate plots analyzing convergence of probability distributions:
1. Distribution Convergence (JS divergence from real priors over time)
2. Entropy Evolution (how distribution entropy changes over time)
3. Learning Rate Evolution (adaptive learning rate changes)
4. Peak Probability Convergence (how maximum probability values evolve)

This simplified version uses a direct simulation approach.
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

# Set up Seaborn with ggplot theme
plt.style.use('ggplot')
sns.set_palette("husl")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# Create output directories
os.makedirs("results/convergence_analysis", exist_ok=True)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Simplified Convergence Analysis with Real Priors")
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

def load_real_data(building_id, target_device, n_days):
    """Load real usage data from parquet file."""
    print(f"📊 Loading real data for {building_id}...")
    
    parquet_path = Path.cwd() / "notebooks" / "data" / f"{building_id}_processed_data.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"No parquet data found for {building_id}")
    
    # Load parquet data
    df = pd.read_parquet(parquet_path)
    print(f"✓ Loaded {len(df)} rows from {parquet_path}")
    
    # Reset index to make utc_timestamp a column
    df = df.reset_index()
    
    # Add hour column
    df['hour'] = pd.to_datetime(df['utc_timestamp']).dt.hour
    df['date'] = pd.to_datetime(df['utc_timestamp']).dt.date
    
    # Find device column
    device_columns = [col for col in df.columns if target_device in col.lower() and building_id in col]
    if not device_columns:
        print(f"⚠ Device {target_device} not found, available columns: {[col for col in df.columns if building_id in col]}")
        # Use first available device column
        device_columns = [col for col in df.columns if building_id in col and 'timestamp' not in col and 'price' not in col]
        if device_columns:
            device_columns = [device_columns[0]]
    
    if not device_columns:
        raise ValueError(f"No device columns found for {building_id}")
    
    device_col = device_columns[0]
    print(f"✓ Using device column: {device_col}")
    
    # Get first n_days of complete days
    daily_counts = df.groupby('date').size()
    complete_days = daily_counts[daily_counts == 24].index.tolist()[:n_days]
    
    if len(complete_days) < n_days:
        print(f"⚠ Only {len(complete_days)} complete days available")
        n_days = len(complete_days)
    
    # Filter to selected days
    filtered_df = df[df['date'].isin(complete_days)]
    print(f"✓ Selected {n_days} days with {len(filtered_df)} total hours")
    
    return filtered_df, device_col, complete_days

def learn_real_prior(df, device_col, learning_days=30):
    """Learn real prior distribution from initial data."""
    print(f"📊 Learning real prior from first {learning_days} days...")
    
    # Use first learning_days to establish real prior
    dates = sorted(df['date'].unique())
    prior_dates = dates[:min(learning_days, len(dates))]
    prior_df = df[df['date'].isin(prior_dates)]
    
    # Calculate hourly usage probability from real data
    hourly_usage = prior_df.groupby('hour')[device_col].sum()
    total_usage = hourly_usage.sum()
    
    if total_usage > 0:
        real_prior = {h: float(hourly_usage.get(h, 0) / total_usage) for h in range(24)}
    else:
        # Fallback to uniform if no usage
        real_prior = {h: 1.0/24.0 for h in range(24)}
    
    # Normalize to ensure sum = 1
    total_prob = sum(real_prior.values())
    if total_prob > 0:
        real_prior = {h: p/total_prob for h, p in real_prior.items()}
    
    print(f"✓ Learned real prior from {len(prior_dates)} days")
    return real_prior

def simulate_learning_convergence(df, device_col, real_prior, lr_tau, lr_max, n_days):
    """Simulate learning convergence using real data."""
    print(f"🧪 Simulating learning convergence over {n_days} days...")
    
    # Initialize with uniform distribution
    current_pmf = {h: 1.0/24.0 for h in range(24)}
    observation_count = 0
    
    # Learning parameters
    lr_min = 0.002
    cap_max = 0.03
    cap_min = 0.005
    
    # Track convergence
    convergence_history = []
    
    # Helper functions
    def js_divergence(p, q):
        """Calculate Jensen-Shannon divergence."""
        p_arr = np.array([p.get(h, 0) for h in range(24)]) + 1e-12
        q_arr = np.array([q.get(h, 0) for h in range(24)]) + 1e-12
        p_arr, q_arr = p_arr/p_arr.sum(), q_arr/q_arr.sum()
        
        m = 0.5 * (p_arr + q_arr)
        kl_pm = np.sum(p_arr * np.log(p_arr / m))
        kl_qm = np.sum(q_arr * np.log(q_arr / m))
        return 0.5 * (kl_pm + kl_qm)
    
    def entropy(pmf):
        """Calculate entropy of PMF."""
        probs = np.array([pmf.get(h, 0) for h in range(24)]) + 1e-12
        probs = probs / probs.sum()
        return -np.sum(probs * np.log(probs))
    
    def adaptive_learning_rate(obs_count, lr_tau, lr_max, lr_min):
        """Calculate adaptive learning rate."""
        base_lr = 1.0 / (obs_count + lr_tau)
        return max(lr_min, min(lr_max, base_lr))
    
    # Process each day
    dates = sorted(df['date'].unique())[:n_days]
    
    for day_idx, date in enumerate(dates):
        day_df = df[df['date'] == date]
        
        # Find hours with device usage
        hourly_usage = day_df.groupby('hour')[device_col].sum()
        active_hours = hourly_usage[hourly_usage > 0].index.tolist()
        
        daily_updates = 0
        
        for hour in active_hours:
            observation_count += 1
            daily_updates += 1
            
            # Calculate learning rate
            lr = adaptive_learning_rate(observation_count, lr_tau, lr_max, lr_min)
            
            # Calculate update cap
            cap_day = max(cap_min, min(cap_max, cap_max * (lr / lr_max)))
            effective_cap = cap_day / daily_updates
            
            # Target distribution (one-hot for observed hour)
            target = {h: 1.0 if h == hour else 0.0 for h in range(24)}
            
            # Update PMF
            old_pmf = current_pmf.copy()
            for h in range(24):
                delta = lr * (target[h] - current_pmf[h])
                delta = np.clip(delta, -effective_cap, effective_cap)
                current_pmf[h] = max(0.0, current_pmf[h] + delta)
            
            # Normalize
            total = sum(current_pmf.values())
            if total > 0:
                current_pmf = {h: p/total for h, p in current_pmf.items()}
            
            # Calculate metrics
            js_from_prior = js_divergence(real_prior, current_pmf)
            js_from_prev = js_divergence(old_pmf, current_pmf)
            pmf_entropy = entropy(current_pmf)
            peak_hour = max(current_pmf, key=current_pmf.get)
            peak_prob = current_pmf[peak_hour]
            
            # Store convergence data
            convergence_history.append({
                'day': day_idx + 1,
                'update': len(convergence_history) + 1,
                'hour': hour,
                'learning_rate': lr,
                'js_divergence_from_prior': js_from_prior,
                'js_divergence_from_prev': js_from_prev,
                'entropy': pmf_entropy,
                'peak_hour': peak_hour,
                'peak_probability': peak_prob,
                'pmf': current_pmf.copy(),
                'observation_count': observation_count
            })
    
    print(f"✓ Completed simulation: {len(convergence_history)} updates over {len(dates)} days")
    
    return convergence_history, current_pmf

def create_distribution_convergence_plot(convergence_history, real_prior, building_id, target_device):
    """Create Plot 1: Distribution Convergence (JS divergence from real priors over time)."""
    print("📊 Creating Plot 1: Distribution Convergence...")
    
    updates = [h['update'] for h in convergence_history]
    js_divergences = [h['js_divergence_from_prior'] for h in convergence_history]
    
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

def create_entropy_evolution_plot(convergence_history, building_id, target_device):
    """Create Plot 2: Entropy Evolution (how distribution entropy changes over time)."""
    print("📊 Creating Plot 2: Entropy Evolution...")
    
    updates = [h['update'] for h in convergence_history]
    entropies = [h['entropy'] for h in convergence_history]
    
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

def create_learning_rate_evolution_plot(convergence_history, building_id, target_device, lr_max):
    """Create Plot 3: Learning Rate Evolution (adaptive learning rate changes)."""
    print("📊 Creating Plot 3: Learning Rate Evolution...")
    
    updates = [h['update'] for h in convergence_history]
    learning_rates = [h['learning_rate'] for h in convergence_history]
    
    plt.figure(figsize=(12, 8))
    plt.plot(updates, learning_rates, linewidth=3, marker='^', markersize=6, color='#C73E1D')
    
    plt.xlabel('Training Updates', fontsize=14, fontweight='bold')
    plt.ylabel('Learning Rate', fontsize=14, fontweight='bold')
    plt.title(f'Learning Rate Evolution\n{building_id} - {target_device}', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Add horizontal lines for LR_MAX and LR_MIN
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

def create_peak_probability_convergence_plot(convergence_history, building_id, target_device):
    """Create Plot 4: Peak Probability Convergence (how maximum probability values evolve)."""
    print("📊 Creating Plot 4: Peak Probability Convergence...")
    
    updates = [h['update'] for h in convergence_history]
    peak_probabilities = [h['peak_probability'] for h in convergence_history]
    peak_hours = [h['peak_hour'] for h in convergence_history]
    
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
    print("CONVERGENCE ANALYSIS WITH REAL PRIORS (Simplified)")
    print("="*80)
    print(f"Building: {building_id}")
    print(f"Analysis days: {n_days}")
    print(f"Target device: {target_device}")
    print(f"LR_TAU: {lr_tau}")
    print(f"LR_MAX: {lr_max}")
    print("="*80)
    
    # 1. Load real data
    df, device_col, selected_days = load_real_data(building_id, target_device, n_days)
    
    # 2. Learn real prior from initial data
    real_prior = learn_real_prior(df, device_col, learning_days=30)
    print(f"✓ Real prior learned: peak hours {sorted(real_prior, key=real_prior.get, reverse=True)[:3]}")
    
    # 3. Simulate learning convergence
    convergence_history, final_pmf = simulate_learning_convergence(
        df, device_col, real_prior, lr_tau, lr_max, n_days
    )
    
    # 4. Create 4 separate plots
    print(f"\n📊 Creating 4 convergence analysis plots...")
    
    plot_files = []
    
    # Plot 1: Distribution Convergence
    plot1 = create_distribution_convergence_plot(convergence_history, real_prior, building_id, target_device)
    if plot1:
        plot_files.append(plot1)
    
    # Plot 2: Entropy Evolution
    plot2 = create_entropy_evolution_plot(convergence_history, building_id, target_device)
    if plot2:
        plot_files.append(plot2)
    
    # Plot 3: Learning Rate Evolution
    plot3 = create_learning_rate_evolution_plot(convergence_history, building_id, target_device, lr_max)
    if plot3:
        plot_files.append(plot3)
    
    # Plot 4: Peak Probability Convergence
    plot4 = create_peak_probability_convergence_plot(convergence_history, building_id, target_device)
    if plot4:
        plot_files.append(plot4)
    
    print("\n" + "="*80)
    print("CONVERGENCE ANALYSIS COMPLETED")
    print("="*80)
    print(f"✅ Generated {len(plot_files)} convergence analysis plots:")
    for plot_file in plot_files:
        print(f"   • {plot_file}")
    
    print(f"\n📊 Analysis Summary:")
    print(f"   • Training days: {len(selected_days)}")
    print(f"   • Total updates: {len(convergence_history)}")
    print(f"   • Final observation count: {convergence_history[-1]['observation_count'] if convergence_history else 0}")
    print(f"   • Used REAL priors learned from data")
    print(f"   • Final JS divergence from prior: {convergence_history[-1]['js_divergence_from_prior']:.4f}" if convergence_history else "")
    print("="*80)
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ CONVERGENCE ANALYSIS FAILED: {e}")
        sys.exit(1)