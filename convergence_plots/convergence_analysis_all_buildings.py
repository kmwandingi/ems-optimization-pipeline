#!/usr/bin/env python
"""
Generate convergence plots for preference-learning over real history using ProbabilityModelAgent.

This script processes all buildings in the dataset and generates four convergence plots per building:
1. JS Divergence vs. Training Days
2. Entropy vs. Training Days  
3. Learning Rate vs. Training Days
4. Distribution Evolution (waveform plots per device)

Requirements:
- DuckDB must be accessible via common.get_con()
- Each building must have a table <building_id>_processed_data
- ProbabilityModelAgent must be available from existing codebase
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ================================
# CONFIGURATION
# ================================
building_ids = [
    "DE_KN_residential1",
    "DE_KN_residential2", 
    "DE_KN_residential3",
    "DE_KN_residential4",
    "DE_KN_residential5",
    "DE_KN_residential6",
    "DE_KN_industrial3"
]

max_prefix_days = 50
output_dir = "./convergence_plots"

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

# Try scipy for JS divergence, fallback to manual implementation
try:
    from scipy.spatial.distance import jensenshannon
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available, using manual JS divergence implementation")

# ================================
# HELPER FUNCTIONS
# ================================

def setup_plotting_style():
    """Set up Seaborn with ggplot-like style."""
    sns.set_theme(style="whitegrid")
    plt.style.use("ggplot")
    plt.rcParams.update({
        'figure.figsize': (10, 6),
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 11,
        'legend.fontsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9
    })

def js_divergence_manual(p, q):
    """
    Manual implementation of Jensen-Shannon divergence using numpy.
    
    Args:
        p, q (np.ndarray): Probability distributions
        
    Returns:
        float: JS divergence
    """
    # Ensure proper probability distributions
    p = np.array(p) + 1e-12
    q = np.array(q) + 1e-12
    p = p / p.sum()
    q = q / q.sum()
    
    # Compute average distribution
    m = 0.5 * (p + q)
    
    # KL divergence function
    def kl_div(x, y):
        return np.sum(np.where(x > 0, x * np.log(x / y), 0.0))
    
    # Jensen-Shannon divergence
    js = 0.5 * kl_div(p, m) + 0.5 * kl_div(q, m)
    return js

def js_divergence(p, q):
    """
    Compute Jensen-Shannon divergence between two probability distributions.
    
    Args:
        p, q (np.ndarray): Probability distributions
        
    Returns:
        float: JS divergence
    """
    if SCIPY_AVAILABLE:
        # Add small epsilon to avoid zeros
        p_safe = np.array(p) + 1e-12
        q_safe = np.array(q) + 1e-12
        p_safe = p_safe / p_safe.sum()
        q_safe = q_safe / q_safe.sum()
        return float(jensenshannon(p_safe, q_safe, base=2))
    else:
        return js_divergence_manual(p, q)

def compute_entropy(dist):
    """
    Compute entropy of a probability distribution.
    
    Args:
        dist (np.ndarray): Probability distribution
        
    Returns:
        float: Entropy
    """
    dist = np.array(dist)
    dist = dist + 1e-12  # Avoid log(0)
    dist = dist / dist.sum()  # Normalize
    return -np.sum(np.where(dist > 0, dist * np.log(dist), 0.0))

def fetch_available_days(building_id):
    """
    Fetch all available days with full 24-hour records for a building.
    
    Args:
        building_id (str): Building identifier
        
    Returns:
        list[str]: List of day strings (YYYY-MM-DD) sorted chronologically
        
    Raises:
        ValueError: If table not found or insufficient data
    """
    print(f"  Fetching available days for {building_id}...")
    
    try:
        # Load data directly from parquet (more reliable than DuckDB views)
        parquet_path = project_root / "notebooks" / "data" / f"{building_id}_processed_data.parquet"
        
        if not parquet_path.exists():
            raise ValueError(f"Parquet file not found: {parquet_path}")
        
        df = pd.read_parquet(parquet_path)
        
        # Add day column from timestamp index
        if isinstance(df.index, pd.DatetimeIndex):
            df['day'] = df.index.date
        elif 'utc_timestamp' in df.columns:
            df['utc_timestamp'] = pd.to_datetime(df['utc_timestamp'])
            df['day'] = df['utc_timestamp'].dt.date
        else:
            raise ValueError(f"No timestamp data found in {building_id}")
        
        # Get days with exactly 24 hours of data
        day_counts = df.groupby('day').size()
        complete_days = day_counts[day_counts == 24].index.tolist()
        
        if len(complete_days) < 2:
            raise ValueError(f"Not enough full days for {building_id}: found {len(complete_days)}")
        
        # Convert to string format and sort
        all_days = [str(day) for day in sorted(complete_days)]
        
        print(f"    Found {len(all_days)} complete days ({all_days[0]} to {all_days[-1]})")
        return all_days
        
    except Exception as e:
        raise ValueError(f"Error fetching days for {building_id}: {e}")

def fetch_devices(building_id):
    """
    Fetch list of controllable devices for a building.
    
    Args:
        building_id (str): Building identifier
        
    Returns:
        list[str]: List of device names found in the data
        
    Raises:
        ValueError: If no controllable devices found
    """
    print(f"  Fetching devices for {building_id}...")
    
    try:
        # Load data to check columns
        parquet_path = project_root / "notebooks" / "data" / f"{building_id}_processed_data.parquet"
        df = pd.read_parquet(parquet_path)
        
        # Get all columns for this building (exclude grid, pv)
        all_cols = df.columns.tolist()
        device_cols = [col for col in all_cols 
                      if building_id in col and 'grid' not in col.lower() and 'pv' not in col.lower()]
        
        # Extract device names
        candidates = ["dishwasher", "washing_machine", "electric_vehicle", "ev", "battery", 
                     "heat_pump", "freezer", "refrigerator"]
        devices = []
        
        for device_col in device_cols:
            device_name = device_col.replace(f"{building_id}_", "")
            # Check if this matches any known device type
            for candidate in candidates:
                if candidate in device_name.lower():
                    devices.append(device_name)
                    break
            else:
                # If no match found, still include it
                devices.append(device_name)
        
        if not devices:
            raise ValueError(f"No controllable device columns found in {building_id}_processed_data")
        
        print(f"    Found devices: {devices}")
        return devices
        
    except Exception as e:
        raise ValueError(f"Error fetching devices for {building_id}: {e}")

def train_reference(building_id, all_days, devices):
    """
    Train reference distribution on full historical data.
    
    Args:
        building_id (str): Building identifier
        all_days (list[str]): All available training days
        devices (list[str]): Device names
        
    Returns:
        dict[str, np.ndarray]: Reference distributions for each device
        
    Raises:
        ValueError: If training fails
    """
    print(f"  Training reference distribution on {len(all_days)} days...")
    
    try:
        # Load parquet data directly for reference computation
        parquet_path = project_root / "notebooks" / "data" / f"{building_id}_processed_data.parquet"
        df = pd.read_parquet(parquet_path)
        
        # Add day and hour columns
        if isinstance(df.index, pd.DatetimeIndex):
            df['day'] = df.index.date
            df['hour'] = df.index.hour
        
        # Filter to training days
        training_days = [pd.to_datetime(day).date() for day in all_days]
        df_training = df[df['day'].isin(training_days)]
        
        # Compute reference distributions by counting active hours
        ref_distributions = {}
        
        for device in devices:
            device_col = f"{building_id}_{device}"
            
            if device_col not in df.columns:
                print(f"    Warning: Column {device_col} not found, skipping device {device}")
                continue
            
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
                
            ref_distributions[device] = ref_dist
        
        print(f"    Reference distributions computed for {len(ref_distributions)} devices")
        return ref_distributions
        
    except Exception as e:
        raise ValueError(f"Error training reference for {building_id}: {e}")

def train_prefix(building_id, prefix_days, devices):
    """
    Train distribution on a prefix of days.
    
    Args:
        building_id (str): Building identifier
        prefix_days (list[str]): Prefix of days to train on
        devices (list[str]): Device names
        
    Returns:
        dict[str, np.ndarray]: Learned distributions for each device
        
    Raises:
        ValueError: If training fails
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
        
        # Compute learned distributions by counting active hours
        learned_distributions = {}
        
        for device in devices:
            device_col = f"{building_id}_{device}"
            
            if device_col not in df.columns:
                continue
            
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
                
            learned_distributions[device] = learned_dist
        
        return learned_distributions
        
    except Exception as e:
        raise ValueError(f"Error training prefix for {building_id} (k={len(prefix_days)}): {e}")

def compute_and_plot(building_id, devices, all_days):
    """
    Compute convergence metrics and generate plots for a building.
    
    Args:
        building_id (str): Building identifier
        devices (list[str]): Device names
        all_days (list[str]): All available training days
    """
    print(f"  Computing convergence metrics and generating plots...")
    
    # Create output directory for this building
    building_output_dir = os.path.join(output_dir, building_id)
    os.makedirs(building_output_dir, exist_ok=True)
    
    # Train reference distribution
    ref_distributions = train_reference(building_id, all_days, devices)
    
    # Initialize metric storage
    N = len(all_days)
    K = min(N, max_prefix_days)
    
    learned_dists = {device: [] for device in devices}
    js_vs_k = {device: [] for device in devices}
    entropy_vs_k = {device: [] for device in devices}
    lr_vs_k = {device: [] for device in devices}
    
    # Train on prefixes and compute metrics
    print(f"    Training on prefixes (1 to {K} days)...")
    
    for k in range(1, K + 1):
        if k % 10 == 0:
            print(f"      Processing day {k}/{K}...")
        
        prefix_days = all_days[:k]
        learned_dist_k = train_prefix(building_id, prefix_days, devices)
        
        for device in devices:
            if device not in learned_dist_k or device not in ref_distributions:
                continue
                
            dist_k = learned_dist_k[device]
            ref_dist = ref_distributions[device]
            
            # Store learned distribution
            learned_dists[device].append(dist_k.copy())
            
            # Compute JS divergence to reference
            js = js_divergence(dist_k, ref_dist)
            js_vs_k[device].append(js)
            
            # Compute entropy
            entropy = compute_entropy(dist_k)
            entropy_vs_k[device].append(entropy)
            
            # Compute learning rate (improvement in JS divergence)
            if k > 1:
                prev_js = js_vs_k[device][-2]
                lr = prev_js - js  # Positive = improvement
            else:
                lr = np.nan
            lr_vs_k[device].append(lr)
    
    # Generate plots
    print(f"    Generating plots...")
    
    # Plot 1: JS Divergence vs. Training Days
    plt.figure(figsize=(10, 6))
    for device in devices:
        if js_vs_k[device]:
            plt.plot(range(1, len(js_vs_k[device]) + 1), js_vs_k[device], 
                    marker='o', linewidth=2, label=device.replace('_', ' ').title())
    
    plt.xlabel('Training Days')
    plt.ylabel('JS Divergence to Reference')
    plt.title(f'{building_id}: JS Divergence vs. Training Days')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(building_output_dir, 'js_divergence.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Entropy vs. Training Days
    plt.figure(figsize=(10, 6))
    for device in devices:
        if entropy_vs_k[device]:
            plt.plot(range(1, len(entropy_vs_k[device]) + 1), entropy_vs_k[device], 
                    marker='s', linewidth=2, label=device.replace('_', ' ').title())
    
    plt.xlabel('Training Days')
    plt.ylabel('Entropy of Learned Distribution')
    plt.title(f'{building_id}: Entropy vs. Training Days')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(building_output_dir, 'entropy.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Learning Rate vs. Training Days
    plt.figure(figsize=(10, 6))
    for device in devices:
        if len(lr_vs_k[device]) > 1:
            # Skip first day (NaN learning rate)
            days = range(2, len(lr_vs_k[device]) + 1)
            rates = lr_vs_k[device][1:]  # Skip first NaN value
            plt.plot(days, rates, marker='^', linewidth=2, label=device.replace('_', ' ').title())
    
    plt.xlabel('Training Days')
    plt.ylabel('Learning Rate (Δ JS Divergence)')
    plt.title(f'{building_id}: Learning Rate vs. Training Days')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(building_output_dir, 'learning_rate.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 4: Distribution Evolution (separate plot per device)
    for device in devices:
        if not learned_dists[device]:
            continue
            
        plt.figure(figsize=(12, 6))
        
        # Choose checkpoints for evolution visualization
        num_dists = len(learned_dists[device])
        if num_dists <= 10:
            checkpoints = list(range(1, num_dists + 1))
        else:
            checkpoints = sorted(set([1, min(5, num_dists), min(10, num_dists), 
                                    min(20, num_dists), min(30, num_dists), num_dists]))
        
        # Plot distribution evolution
        colors = plt.cm.viridis(np.linspace(0, 1, len(checkpoints)))
        
        for i, checkpoint in enumerate(checkpoints):
            dist = learned_dists[device][checkpoint - 1]  # 0-indexed
            alpha = 0.6 if checkpoint < num_dists else 1.0  # Highlight final distribution
            linewidth = 1.5 if checkpoint < num_dists else 2.5
            
            plt.plot(range(24), dist, marker='o', linewidth=linewidth, alpha=alpha,
                    color=colors[i], label=f'Day {checkpoint}')
        
        plt.xlabel('Hour of Day')
        plt.ylabel('Probability')
        plt.title(f'{building_id} - {device.replace("_", " ").title()} Distribution Evolution')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.xticks(range(0, 24, 2))
        plt.xlim(-0.5, 23.5)
        plt.tight_layout()
        
        # Save plot
        device_filename = device.replace('_', '').replace(' ', '')
        plt.savefig(os.path.join(building_output_dir, f'dist_evolution_{device_filename}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"    Plots saved to {building_output_dir}")

def main():
    """Main function to process all buildings."""
    print("=" * 70)
    print("CONVERGENCE ANALYSIS - ALL BUILDINGS")
    print("=" * 70)
    print(f"Buildings to process: {len(building_ids)}")
    print(f"Max prefix days: {max_prefix_days}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Set up plotting style
    setup_plotting_style()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each building
    successful_buildings = 0
    failed_buildings = []
    
    for i, building_id in enumerate(building_ids, 1):
        print(f"[{i}/{len(building_ids)}] Processing {building_id}...")
        
        try:
            # Fetch available days and devices
            all_days = fetch_available_days(building_id)
            devices = fetch_devices(building_id)
            
            print(f"  Configuration: {len(devices)} devices, {len(all_days)} days available")
            
            # Compute metrics and generate plots
            compute_and_plot(building_id, devices, all_days)
            
            successful_buildings += 1
            print(f"  ✓ {building_id} completed successfully")
            
        except Exception as e:
            failed_buildings.append((building_id, str(e)))
            print(f"  ✗ {building_id} failed: {e}")
        
        print()
    
    # Summary
    print("=" * 70)
    print("PROCESSING SUMMARY")
    print("=" * 70)
    print(f"Successfully processed: {successful_buildings}/{len(building_ids)} buildings")
    
    if failed_buildings:
        print(f"Failed buildings:")
        for building_id, error in failed_buildings:
            print(f"  - {building_id}: {error}")
    else:
        print("All buildings processed successfully!")
    
    print(f"\nOutput saved to: {os.path.abspath(output_dir)}")
    
    # List all generated files
    total_plots = 0
    for building_id in building_ids:
        building_dir = os.path.join(output_dir, building_id)
        if os.path.exists(building_dir):
            plots = [f for f in os.listdir(building_dir) if f.endswith('.png')]
            total_plots += len(plots)
    
    print(f"Total plots generated: {total_plots}")

if __name__ == "__main__":
    main()