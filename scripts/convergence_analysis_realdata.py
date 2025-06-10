#!/usr/bin/env python
"""
Convergence Analysis Script for Preference-Learning Pipeline
Generates convergence plots using real historical data and ProbabilityModelAgent.

This script:
1. Connects to DuckDB and fetches real building data
2. Trains reference distribution on full historical data
3. Incrementally trains on prefixes and records learned distributions
4. Computes four convergence metrics: JS divergence, entropy, learning rate, distribution evolution
5. Generates four separate plots using Seaborn with ggplot-like style

Usage:
    python convergence_analysis_realdata.py [--building_id DE_KN_residential1] [--max_days 50]
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add project directories to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "scripts"))
sys.path.append(str(project_root / "notebooks"))
sys.path.append(str(project_root / "notebooks" / "utils"))

# Import required modules
from common import get_con
from notebooks.agents.ProbabilityModelAgent import ProbabilityModelAgent
from notebooks.utils.device_specs import device_specs
from scipy.spatial.distance import jensenshannon
import math

# Configuration
DEFAULT_BUILDING_ID = "DE_KN_residential1"
DEFAULT_MAX_DAYS = 50
OUTPUT_DIR = project_root / "results" / "convergence_analysis"

def setup_plotting_style():
    """Set up Seaborn with ggplot-like style."""
    sns.set_theme(style="whitegrid")
    sns.set_palette("deep")
    plt.rcParams.update({
        'figure.figsize': (10, 6),
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'legend.fontsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10
    })

def fetch_available_days(building_id):
    """
    Query parquet file directly to get all available days with full 24-hour records.
    
    Args:
        building_id (str): Building ID (e.g., 'DE_KN_residential1')
    
    Returns:
        list[str]: List of day strings (YYYY-MM-DD) sorted chronologically
    """
    print(f"Fetching available days for {building_id}...")
    
    try:
        # Load parquet file directly
        parquet_path = project_root / "notebooks" / "data" / f"{building_id}_processed_data.parquet"
        
        if not parquet_path.exists():
            raise ValueError(f"Parquet file not found: {parquet_path}")
        
        print(f"Loading data from: {parquet_path}")
        df = pd.read_parquet(parquet_path)
        
        # Check if timestamp is in index
        if 'utc_timestamp' in df.index.names or isinstance(df.index, pd.DatetimeIndex):
            # Create day and hour columns from timestamp index
            df['day'] = df.index.date
            df['hour'] = df.index.hour
        elif 'utc_timestamp' in df.columns:
            # Create day and hour columns from timestamp column
            df['utc_timestamp'] = pd.to_datetime(df['utc_timestamp'])
            df['day'] = df['utc_timestamp'].dt.date
            df['hour'] = df['utc_timestamp'].dt.hour
        else:
            raise ValueError("No timestamp data found in dataset")
        
        # Get days with exactly 24 hours of data
        day_counts = df.groupby('day').size()
        complete_days = day_counts[day_counts == 24].index.tolist()
        
        if not complete_days:
            raise ValueError(f"No complete 24-hour days found")
        
        # Convert to string format and sort
        available_days = [str(day) for day in sorted(complete_days)]
        
        # Verify that we have device columns
        device_columns = [col for col in df.columns 
                         if building_id in col and 'grid' not in col.lower() and 'pv' not in col.lower()]
        
        if not device_columns:
            raise ValueError(f"No device columns found for {building_id}")
        
        print(f"Found {len(available_days)} complete days with {len(device_columns)} devices")
        print(f"Date range: {available_days[0]} to {available_days[-1]}")
        print(f"Sample devices: {device_columns[:3]}")
        
        return available_days
        
    except Exception as e:
        print(f"Error fetching days: {e}")
        raise

def get_device_list(building_id):
    """
    Get list of device types from the building data.
    
    Args:
        building_id (str): Building ID
        
    Returns:
        list[str]: List of device types (e.g., ['dishwasher', 'heat_pump', ...])
    """
    try:
        # Load parquet file directly
        parquet_path = project_root / "notebooks" / "data" / f"{building_id}_processed_data.parquet"
        df = pd.read_parquet(parquet_path)
        
        # Extract device columns
        device_columns = []
        for col in df.columns:
            if building_id in col and 'grid' not in col.lower() and 'pv' not in col.lower():
                # Extract device type from column name
                device_type = col.replace(f"{building_id}_", "")
                device_columns.append(device_type)
        
        # Also check what's available in device_specs
        available_devices = list(device_specs.keys())
        
        # Filter to devices that exist in both data and specs
        valid_devices = []
        for device in device_columns:
            for spec_device in available_devices:
                if spec_device in device.lower():
                    valid_devices.append(spec_device)
                    break
        
        if not valid_devices:
            # If no matches found, use the actual column names without building prefix
            valid_devices = device_columns
        
        print(f"Found devices: {valid_devices}")
        return valid_devices
        
    except Exception as e:
        print(f"Error getting device list: {e}")
        # Fallback to common devices
        return ['dishwasher', 'washing_machine', 'heat_pump']

def train_reference_distribution(building_id, all_days):
    """
    Train reference distribution on full available historical data using direct data access.
    
    Args:
        building_id (str): Building ID
        all_days (list[str]): All available days
        
    Returns:
        dict[str, np.ndarray]: Reference distributions for each device (24-element arrays)
    """
    print(f"Training reference distribution on {len(all_days)} days...")
    
    try:
        # Load parquet data directly
        parquet_path = project_root / "notebooks" / "data" / f"{building_id}_processed_data.parquet"
        df = pd.read_parquet(parquet_path)
        
        # Add day and hour columns
        if isinstance(df.index, pd.DatetimeIndex):
            df['day'] = df.index.date
            df['hour'] = df.index.hour
        
        # Filter to training days
        training_days = [pd.to_datetime(day).date() for day in all_days]
        df_training = df[df['day'].isin(training_days)]
        
        # Get device columns
        device_columns = [col for col in df.columns 
                         if building_id in col and 'grid' not in col.lower() and 'pv' not in col.lower()]
        
        # Compute reference distributions by counting active hours
        reference_distributions = {}
        
        for device_col in device_columns:
            device_type = device_col.replace(f"{building_id}_", "")
            
            # Count occurrences of usage per hour across all training days
            hour_counts = np.zeros(24)
            total_days = 0
            
            for day in training_days:
                day_data = df_training[df_training['day'] == day]
                if len(day_data) == 24:  # Full day
                    total_days += 1
                    for hour in range(24):
                        hour_data = day_data[day_data['hour'] == hour]
                        if not hour_data.empty and hour_data[device_col].iloc[0] > 0:
                            hour_counts[hour] += 1
            
            # Convert to probability distribution
            if total_days > 0:
                ref_dist = hour_counts / total_days
                # Normalize to ensure it sums to 1
                if ref_dist.sum() > 0:
                    ref_dist = ref_dist / ref_dist.sum()
                else:
                    ref_dist = np.ones(24) / 24  # Uniform if no usage detected
            else:
                ref_dist = np.ones(24) / 24  # Uniform fallback
                
            reference_distributions[device_type] = ref_dist
        
        print(f"Reference distributions computed for {len(reference_distributions)} devices")
        return reference_distributions
        
    except Exception as e:
        print(f"Error training reference distribution: {e}")
        raise

def train_prefix_distribution(building_id, prefix_days):
    """
    Train distribution on a prefix of days using direct data access.
    
    Args:
        building_id (str): Building ID
        prefix_days (list[str]): Prefix of days to train on
        
    Returns:
        dict[str, np.ndarray]: Learned distributions for each device (24-element arrays)
    """
    try:
        # Load parquet data directly
        parquet_path = project_root / "notebooks" / "data" / f"{building_id}_processed_data.parquet"
        df = pd.read_parquet(parquet_path)
        
        # Add day and hour columns
        if isinstance(df.index, pd.DatetimeIndex):
            df['day'] = df.index.date
            df['hour'] = df.index.hour
        
        # Filter to training days
        training_days = [pd.to_datetime(day).date() for day in prefix_days]
        df_training = df[df['day'].isin(training_days)]
        
        # Get device columns
        device_columns = [col for col in df.columns 
                         if building_id in col and 'grid' not in col.lower() and 'pv' not in col.lower()]
        
        # Compute learned distributions by counting active hours
        learned_distributions = {}
        
        for device_col in device_columns:
            device_type = device_col.replace(f"{building_id}_", "")
            
            # Count occurrences of usage per hour across prefix days
            hour_counts = np.zeros(24)
            total_days = 0
            
            for day in training_days:
                day_data = df_training[df_training['day'] == day]
                if len(day_data) == 24:  # Full day
                    total_days += 1
                    for hour in range(24):
                        hour_data = day_data[day_data['hour'] == hour]
                        if not hour_data.empty and hour_data[device_col].iloc[0] > 0:
                            hour_counts[hour] += 1
            
            # Convert to probability distribution
            if total_days > 0:
                learned_dist = hour_counts / total_days
                # Normalize to ensure it sums to 1
                if learned_dist.sum() > 0:
                    learned_dist = learned_dist / learned_dist.sum()
                else:
                    learned_dist = np.ones(24) / 24  # Uniform if no usage detected
            else:
                learned_dist = np.ones(24) / 24  # Uniform fallback
                
            learned_distributions[device_type] = learned_dist
        
        return learned_distributions
        
    except Exception as e:
        print(f"Error training prefix distribution (k={len(prefix_days)}): {e}")
        # Return empty distributions to continue analysis
        return {}

def jensen_shannon_divergence(p, q):
    """
    Compute Jensen-Shannon divergence between two probability distributions.
    
    Args:
        p, q (np.ndarray): Probability distributions
        
    Returns:
        float: JS divergence
    """
    # Add small epsilon to avoid log(0)
    p = p + 1e-12
    q = q + 1e-12
    
    # Normalize to ensure they're proper probability distributions
    p = p / p.sum()
    q = q / q.sum()
    
    return float(jensenshannon(p, q))

def compute_entropy(distribution):
    """
    Compute entropy of a probability distribution.
    
    Args:
        distribution (np.ndarray): Probability distribution
        
    Returns:
        float: Entropy
    """
    # Add small epsilon to avoid log(0)
    p = distribution + 1e-12
    p = p / p.sum()  # Normalize
    
    return -np.sum(p * np.log(p))

def compute_metrics(learned_distributions_over_k, reference_distributions, devices):
    """
    Compute four convergence metrics for all devices over training prefixes.
    
    Args:
        learned_distributions_over_k (dict): Learned distributions indexed by [k][device]
        reference_distributions (dict): Reference distributions for each device
        devices (list): List of device types
        
    Returns:
        pd.DataFrame: Metrics dataframe with columns [k, device, js_divergence, entropy, learning_rate]
    """
    print("Computing convergence metrics...")
    
    metrics_data = []
    
    # Get all k values (prefix lengths)
    k_values = sorted(learned_distributions_over_k.keys())
    
    for device in devices:
        if device not in reference_distributions:
            print(f"Warning: No reference distribution for device {device}")
            continue
            
        ref_dist = reference_distributions[device]
        prev_js = None
        
        for k in k_values:
            if k not in learned_distributions_over_k:
                continue
                
            if device not in learned_distributions_over_k[k]:
                continue
                
            learned_dist = learned_distributions_over_k[k][device]
            
            # Compute JS divergence to reference
            js_div = jensen_shannon_divergence(learned_dist, ref_dist)
            
            # Compute entropy
            entropy = compute_entropy(learned_dist)
            
            # Compute learning rate (change in JS divergence)
            learning_rate = np.nan
            if prev_js is not None:
                learning_rate = prev_js - js_div  # Improvement (positive = better)
            
            metrics_data.append({
                'k': k,
                'device': device,
                'js_divergence': js_div,
                'entropy': entropy,
                'learning_rate': learning_rate
            })
            
            prev_js = js_div
    
    return pd.DataFrame(metrics_data)

def plot_js_divergence(metrics_df, output_dir):
    """
    Plot JS divergence vs. prefix length.
    
    Args:
        metrics_df (pd.DataFrame): Metrics dataframe
        output_dir (Path): Output directory
    """
    plt.figure(figsize=(10, 6))
    
    # Plot one line per device
    for device in metrics_df['device'].unique():
        device_data = metrics_df[metrics_df['device'] == device]
        plt.plot(device_data['k'], device_data['js_divergence'], 
                marker='o', linewidth=2, label=device.replace('_', ' ').title())
    
    plt.xlabel('Number of Training Days')
    plt.ylabel('Jensen-Shannon Divergence to Reference')
    plt.title('JS Divergence to Reference vs. Training Days')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    output_file = output_dir / "js_divergence_convergence.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved JS divergence plot: {output_file}")

def plot_entropy(metrics_df, output_dir):
    """
    Plot entropy vs. prefix length.
    
    Args:
        metrics_df (pd.DataFrame): Metrics dataframe
        output_dir (Path): Output directory
    """
    plt.figure(figsize=(10, 6))
    
    # Plot one line per device
    for device in metrics_df['device'].unique():
        device_data = metrics_df[metrics_df['device'] == device]
        plt.plot(device_data['k'], device_data['entropy'], 
                marker='s', linewidth=2, label=device.replace('_', ' ').title())
    
    plt.xlabel('Number of Training Days')
    plt.ylabel('Entropy of Learned Distribution')
    plt.title('Entropy of Learned Distribution vs. Training Days')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    output_file = output_dir / "entropy_convergence.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved entropy plot: {output_file}")

def plot_learning_rate(metrics_df, output_dir):
    """
    Plot learning rate vs. prefix length.
    
    Args:
        metrics_df (pd.DataFrame): Metrics dataframe
        output_dir (Path): Output directory
    """
    plt.figure(figsize=(10, 6))
    
    # Plot one line per device (exclude k=1 since learning_rate is NaN)
    for device in metrics_df['device'].unique():
        device_data = metrics_df[(metrics_df['device'] == device) & (metrics_df['k'] > 1)]
        if not device_data.empty:
            plt.plot(device_data['k'], device_data['learning_rate'], 
                    marker='^', linewidth=2, label=device.replace('_', ' ').title())
    
    plt.xlabel('Number of Training Days')
    plt.ylabel('Learning Rate (Δ JS Divergence)')
    plt.title('Learning Rate (Δ JS Divergence) vs. Training Days')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Save plot
    output_file = output_dir / "learning_rate_convergence.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved learning rate plot: {output_file}")

def plot_distribution_evolution(learned_distributions_over_k, devices, building_id, output_dir):
    """
    Plot distribution evolution heatmaps for each device.
    
    Args:
        learned_distributions_over_k (dict): Learned distributions indexed by [k][device]
        devices (list): List of device types
        building_id (str): Building ID for plot titles
        output_dir (Path): Output directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    for device in devices:
        # Collect data for this device
        k_values = sorted(learned_distributions_over_k.keys())
        distributions = []
        
        for k in k_values:
            if k in learned_distributions_over_k and device in learned_distributions_over_k[k]:
                distributions.append(learned_distributions_over_k[k][device])
            else:
                # Fill with zeros if missing
                distributions.append(np.zeros(24))
        
        if not distributions:
            continue
            
        # Create heatmap matrix (k_days x 24_hours)
        heatmap_data = np.array(distributions)
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Create heatmap
        sns.heatmap(heatmap_data, 
                   xticklabels=range(24), 
                   yticklabels=k_values,
                   cmap='YlOrRd', 
                   cbar_kws={'label': 'Probability'},
                   annot=False)
        
        plt.xlabel('Hour of Day')
        plt.ylabel('Training Days')
        plt.title(f'Distribution Evolution over Training Days\n{building_id} - {device.replace("_", " ").title()}')
        
        # Save plot
        clean_device = device.replace('_', '').replace(' ', '')
        output_file = output_dir / f"{building_id}_{clean_device}_distribution_evolution_{timestamp}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved distribution evolution plot: {output_file}")

def main():
    """Main function to run convergence analysis."""
    parser = argparse.ArgumentParser(description='Generate convergence plots for preference-learning pipeline')
    parser.add_argument('--building_id', default=DEFAULT_BUILDING_ID, 
                       help=f'Building ID (default: {DEFAULT_BUILDING_ID})')
    parser.add_argument('--max_days', type=int, default=DEFAULT_MAX_DAYS,
                       help=f'Maximum number of training days (default: {DEFAULT_MAX_DAYS})')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("CONVERGENCE ANALYSIS - REAL DATA")
    print("=" * 60)
    print(f"Building ID: {args.building_id}")
    print(f"Max training days: {args.max_days}")
    print()
    
    # Set up plotting style
    setup_plotting_style()
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        # Step 1: Fetch available days
        available_days = fetch_available_days(args.building_id)
        
        if len(available_days) < 5:
            raise ValueError(f"Need at least 5 days for meaningful analysis. Found: {len(available_days)}")
        
        # Step 2: Get device list
        devices = get_device_list(args.building_id)
        
        # Step 3: Train reference distribution on all days
        reference_distributions = train_reference_distribution(args.building_id, available_days)
        
        # Step 4: Train on prefixes and record learned distributions
        max_k = min(args.max_days, len(available_days))
        learned_distributions_over_k = {}
        
        print(f"\nTraining on prefixes (1 to {max_k} days)...")
        
        for k in range(1, max_k + 1):
            print(f"Training on prefix length k={k}...", end=" ")
            
            prefix_days = available_days[:k]
            learned_dists = train_prefix_distribution(args.building_id, prefix_days)
            
            if learned_dists:
                learned_distributions_over_k[k] = learned_dists
                print("✓")
            else:
                print("✗ (failed)")
        
        print(f"\nSuccessfully trained on {len(learned_distributions_over_k)} prefix lengths")
        
        # Step 5: Compute metrics
        metrics_df = compute_metrics(learned_distributions_over_k, reference_distributions, devices)
        
        if metrics_df.empty:
            raise ValueError("No metrics computed - check data and training")
        
        print(f"Computed metrics for {len(metrics_df)} data points")
        
        # Step 6: Generate plots
        print("\nGenerating plots...")
        
        plot_js_divergence(metrics_df, OUTPUT_DIR)
        plot_entropy(metrics_df, OUTPUT_DIR)
        plot_learning_rate(metrics_df, OUTPUT_DIR)
        plot_distribution_evolution(learned_distributions_over_k, devices, args.building_id, OUTPUT_DIR)
        
        # Save metrics to CSV for further analysis
        metrics_file = OUTPUT_DIR / f"{args.building_id}_convergence_metrics_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        metrics_df.to_csv(metrics_file, index=False)
        print(f"Saved metrics data: {metrics_file}")
        
        print("\n" + "=" * 60)
        print("CONVERGENCE ANALYSIS COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"Output directory: {OUTPUT_DIR}")
        print(f"Plots generated: 4 PNG files")
        print(f"Metrics data: {metrics_file}")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()