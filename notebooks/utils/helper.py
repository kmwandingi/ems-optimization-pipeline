#!/usr/bin/env python
"""
Helper functions for EMS optimization using agent optimizers.

This module provides utility functions that delegate to agent 
methods with proper DataFrame formatting.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta

# Add notebooks directory to path for agent imports
sys.path.append(str(Path(__file__).parent.parent))

# Import agent classes
from agents.ProbabilityModelAgent import ProbabilityModelAgent
from agents.BatteryAgent import BatteryAgent
from agents.EVAgent import EVAgent
from agents.PVAgent import PVAgent
from agents.GridAgent import GridAgent
from agents.FlexibleDeviceAgent import FlexibleDevice
from agents.GlobalOptimizer import GlobalOptimizer
from agents.GlobalConnectionLayer import GlobalConnectionLayer
from agents.WeatherAgent import WeatherAgent

# Import device_specs
from .device_specs import device_specs

# Default parameters for system components
BATTERY_PARAMS = {
    "max_charge_rate": 3.0,
    "max_discharge_rate": 3.0,
    "initial_soc": 7.0,
    "soc_min": 1.0,
    "soc_max": 10.0,
    "capacity": 10.0,
    "degradation_rate": 0.001,
    "efficiency_charge": 0.95,
    "efficiency_discharge": 0.95
}

EV_PARAMS = {
    "capacity": 60.0,
    "initial_soc": 12.0,
    "soc_min": 6.0,
    "soc_max": 54.0,
    "max_charge_rate": 7.4,
    "max_discharge_rate": 0.0,
    "efficiency_charge": 0.92,
    "efficiency_discharge": 0.92,
    "must_be_full_by_hour": 7
}

GRID_PARAMS = {
    "import_price": 0.25,
    "export_price": 0.05,
    "max_import": 15.0,
    "max_export": 15.0
}

def validate_dataframe_for_agents(df, expected_hours=24):
    """
    Validate that DataFrame has correct structure for agent consumption.
    Ensures proper data format.
    """
    if df is None or df.empty:
        raise ValueError("DataFrame is empty or None")
    
    # Check required columns
    required_columns = ['hour', 'day', 'price_per_kwh']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Check hour range
    unique_hours = sorted(df['hour'].unique())
    if len(unique_hours) != expected_hours:
        raise ValueError(f"DataFrame does not have exactly {expected_hours} hours. Found: {len(unique_hours)} hours")
    
    if expected_hours == 24 and (min(unique_hours) != 0 or max(unique_hours) != 23):
        raise ValueError(f"Hours should range from 0-23. Found: {min(unique_hours)}-{max(unique_hours)}")
    
    # Check for missing values in critical columns
    for col in required_columns:
        if df[col].isna().any():
            raise ValueError(f"Column '{col}' contains missing values")
    
    return True

def compute_device_savings(device):
    """
    Compute device savings using original vs optimized consumption.
    Uses agent-generated schedules only.
    """
    if not hasattr(device, 'original_consumption') or not hasattr(device, 'optimized_consumption'):
        # No fallback allowed - device must have consumption data from agent optimization
        raise ValueError(f"Device {device.device_name if hasattr(device, 'device_name') else 'Unknown'} "
                        f"missing required consumption attributes. Agent optimization must be run correctly.")
    else:
        # Calculate costs using device data
        if hasattr(device, 'data') and 'price_per_kwh' in device.data.columns:
            prices = device.data['price_per_kwh'].values[:24]
            original_cost = np.sum(np.array(device.original_consumption[:24]) * prices)
            optimized_cost = np.sum(np.array(device.optimized_consumption[:24]) * prices)
        else:
            # Use default prices if not available
            default_price = 0.25
            original_cost = np.sum(np.array(device.original_consumption[:24])) * default_price
            optimized_cost = np.sum(np.array(device.optimized_consumption[:24])) * default_price
    
    # Add battery costs if available
    if hasattr(device, 'battery_charge') and device.battery_charge is not None:
        if hasattr(device, 'data') and 'price_per_kwh' in device.data.columns:
            prices = device.data['price_per_kwh'].values[:24]
        else:
            prices = [0.25] * 24
        
        battery_cost = np.sum(np.array(device.battery_charge[:24]) * prices)
        battery_savings = np.sum(np.array(device.battery_discharge[:24]) * prices) if hasattr(device, 'battery_discharge') and device.battery_discharge else 0
        optimized_cost += battery_cost - battery_savings
    
    # Calculate savings
    euro_savings = original_cost - optimized_cost
    pct_savings = (euro_savings / original_cost * 100) if original_cost > 0 else 0
    adjusted_cost = optimized_cost
    
    return pct_savings, euro_savings, adjusted_cost

def plot_battery_schedule(battery_schedule, building_id, day_str):
    """
    Plot battery schedule using agent results.
    Plots agent-generated schedules only.
    """
    # Create output directory
    os.makedirs("results/plots", exist_ok=True)
    
    # Create plot
    plt.figure(figsize=(12, 6))
    hours = np.arange(24)
    
    # Plot schedule
    plt.bar(hours, battery_schedule, alpha=0.7, label='Battery Schedule')
    plt.xlabel('Hour')
    plt.ylabel('Power (kW)')
    plt.title(f'Battery Schedule - {building_id} - {day_str}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    output_file = f"results/plots/{building_id}_battery_schedule_{day_str}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_file

def plot_device_comparison(device, building_id, day_str):
    """
    Plot device original vs optimized consumption.
    Plots agent-generated schedules only.
    """
    # Create output directory
    os.makedirs("results/plots", exist_ok=True)
    
    # Create plot
    plt.figure(figsize=(12, 6))
    hours = np.arange(24)
    
    if hasattr(device, 'original_consumption') and hasattr(device, 'optimized_consumption'):
        original = device.original_consumption[:24]
        optimized = device.optimized_consumption[:24]
        
        plt.bar(hours - 0.2, original, 0.4, alpha=0.7, label='Original', color='red')
        plt.bar(hours + 0.2, optimized, 0.4, alpha=0.7, label='Optimized', color='blue')
    else:
        # Use device schedule if available
        if hasattr(device, 'optimized_schedule'):
            plt.bar(hours, device.optimized_schedule[:24], alpha=0.7, label='Optimized Schedule')
    
    plt.xlabel('Hour')
    plt.ylabel('Power (kW)')
    plt.title(f'Device Schedule - {building_id} - {day_str}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    device_name = device.device_name.split('_')[-1] if hasattr(device, 'device_name') else 'device'
    output_file = f"results/plots/{building_id}_{device_name}_comparison_{day_str}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_file

def load_building_day_devices(building_id, single_day, parquet_dir, device_specs):
    """
    Load building devices for a specific day using DuckDB queries.
    Loads data from DuckDB only.
    """
    # Import common here to avoid circular imports
    import common
    
    # Get DuckDB connection
    con = common.get_con()
    view_name = f"{building_id}_processed_data"
    
    # Query data for this day from DuckDB
    day_data = con.execute(f"""
        SELECT *, EXTRACT(hour FROM utc_timestamp) as hour, DATE(utc_timestamp) as day
        FROM {view_name} 
        WHERE DATE(utc_timestamp) = '{single_day}' 
        ORDER BY utc_timestamp
    """).df()
    
    if len(day_data) != 24:
        raise ValueError(f"Day {single_day} does not have exactly 24 hours. Found: {len(day_data)} hours")
    
    # Create devices
    devices = []
    
    # Create global connection layer
    max_building_load = 65.0  # Default building load limit
    global_layer = GlobalConnectionLayer(max_building_load=max_building_load, total_hours=24)
    
    # Query DuckDB for device columns (exclude grid and PV)
    columns_df = con.execute(f"DESCRIBE {view_name}").df()
    device_columns = [col for col in columns_df['column_name'] if building_id in col 
                     and 'grid_export' not in col and 'grid_import' not in col and 'pv' not in col]
    
    for device_id in device_columns:
        if device_id in day_data.columns:
            # Extract device type from column name
            parts = device_id.split('_')
            if len(parts) >= 4 and '_'.join(parts[-2:]) in ['heat_pump', 'washing_machine']:
                device_type = '_'.join(parts[-2:])
            else:
                device_type = parts[-1]
            
            # Get device specification
            if device_type in device_specs:
                spec = device_specs[device_type].copy()
            elif device_type in globals()['device_specs']:
                spec = globals()['device_specs'][device_type].copy()
            else:
                # Default spec if not found
                spec = {'category': 'Non-Flexible', 'power_rating': 1.0}
            
            # Reset index for proper agent data handling
            day_data_reset = day_data.reset_index(drop=True).copy()
            
            # Create FlexibleDevice agent
            device = FlexibleDevice(
                device_name=device_id,
                data=day_data_reset,
                category=spec.get('category', 'Non-Flexible'),
                power_rating=spec.get('power_rating', 1.0),
                global_layer=global_layer,
                battery_agent=None,  # No battery for probability training
                spec=spec
            )
            
            # Initialize with uniform probabilities and proper probability tracking
            device.hour_probability = {h: 1/24 for h in range(24)}
            device.observation_count = 0
            device.probability_updates = []  # Initialize empty updates list
            
            # Add initial prior entry to probability_updates to avoid IndexError
            device.probability_updates.append({
                'day': 'INITIAL_PRIOR',
                'actual_hour': 0,
                'distribution': device.hour_probability.copy(),
                'learning_rate': 0.0,
                'entropy': np.log(24),  # Maximum entropy for uniform distribution
                'day_type': 'weekday',
                'js_prior': 0.0,
                'js_prev': 0.0
            })
            
            devices.append(device)
    
    return devices

def validate_agent_results(devices, optimizer, battery_agent=None, ev_agent=None):
    """
    Validate that agent optimization results are consistent.
    Validates agent-generated results only.
    """
    validation_errors = []
    
    # Check that all devices have optimized schedules
    for device in devices:
        if not hasattr(device, 'optimized_schedule') and not hasattr(device, 'centralized_optimized_schedule') and not hasattr(device, 'phases_optimized_schedule'):
            validation_errors.append(f"Device {device.device_name} missing optimized schedule")
        
        # Check schedule length
        schedule = None
        if hasattr(device, 'phases_optimized_schedule'):
            schedule = device.phases_optimized_schedule
        elif hasattr(device, 'centralized_optimized_schedule'):
            schedule = device.centralized_optimized_schedule
        elif hasattr(device, 'optimized_schedule'):
            schedule = device.optimized_schedule
        
        if schedule and len(schedule) < 24:
            validation_errors.append(f"Device {device.device_name} schedule has {len(schedule)} hours instead of 24")
    
    # Check battery agent results if available
    if battery_agent:
        if not hasattr(battery_agent, 'hourly_soc'):
            validation_errors.append("BatteryAgent missing hourly_soc after optimization")
        elif len(battery_agent.hourly_soc) < 24:
            validation_errors.append(f"BatteryAgent hourly_soc has {len(battery_agent.hourly_soc)} hours instead of 24")
    
    # Check EV agent results if available
    if ev_agent:
        if not hasattr(ev_agent, 'hourly_soc'):
            validation_errors.append("EVAgent missing hourly_soc after optimization")
        elif len(ev_agent.hourly_soc) < 24:
            validation_errors.append(f"EVAgent hourly_soc has {len(ev_agent.hourly_soc)} hours instead of 24")
    
    if validation_errors:
        raise ValueError(f"Agent validation failed: {'; '.join(validation_errors)}")
    
    return True

###########################################################
#!/usr/bin/env python3
# ──────────────────────────────────────────────────────────────────────────
#  FULL, SELF-CONTAINED DEMO  —  Ready to copy-paste and run
#  Works with NumPy ≥ 2.0  (replaced ndarray.ptp → np.ptp)
# ──────────────────────────────────────────────────────────────────────────
import warnings, numpy as np, pandas as pd, matplotlib.pyplot as plt
from datetime        import date
from pathlib         import Path
from typing          import List
from matplotlib      import gridspec
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines   import Line2D

try:                                   # nicer defaults if seaborn is present
    import seaborn as sns
    sns.set_style("whitegrid")
except ImportError:
    pass
warnings.filterwarnings("ignore", category=FutureWarning)

# ──────────────────────────────────────────────────────────────────────────
#  DEVICE SPECS  (extend as desired)
# ──────────────────────────────────────────────────────────────────────────
device_specs = {
    "dishwasher": {
        "category": "Partially Flexible", "power_rating": 1.0,
        "allowed_hours": list(range(8, 22)),
        "phases": [
            {"duration": 1, "energy_kwh": 0.35, "peak_kw": 1.0},
            {"duration": 1, "energy_kwh": 0.25, "peak_kw": 1.0},
            {"duration": 1, "energy_kwh": 0.20, "peak_kw": 1.0},
        ],
        "flex_model": "discrete_phase",
    },
    "washing_machine": {
        "category": "Partially Flexible", "power_rating": 2.0,
        "allowed_hours": list(range(8, 22)),
        "phases": [
            {"duration": 1, "energy_kwh": 0.60, "peak_kw": 2.0},
            {"duration": 1, "energy_kwh": 0.80, "peak_kw": 2.0},
        ],
        "flex_model": "discrete_phase",
    },
    "heat_pump": {
        "category": "Highly Flexible", "power_rating": 3.0,
        "allowed_hours": list(range(6, 24)),
        "phases": [
            {"duration": 2, "energy_kwh": 2.8, "peak_kw": 3.0},
            {"duration": 2, "energy_kwh": 3.0, "peak_kw": 3.0},
            {"duration": 2, "energy_kwh": 2.8, "peak_kw": 3.0},
        ],
        "flex_model": "partial_usage",
    },
    "freezer": {
        "category": "Highly Flexible", "power_rating": 1.0,
        "allowed_hours": list(range(0, 24)),
        "phases": [{"duration": 1, "energy_kwh": 0.025, "peak_kw": 1.0}] * 24,
        "flex_model": "fixed",
    },
    "fridge": {
        "category": "Highly Flexible", "power_rating": 1.0,
        "allowed_hours": list(range(0, 24)),
        "phases": [{"duration": 1, "energy_kwh": 0.025, "peak_kw": 1.0}] * 24,
        "flex_model": "fixed",
    },
    "tumble_dryer": {
        "category": "Partially Flexible", "power_rating": 2.5,
        "allowed_hours": list(range(8, 22)),
        "phases": [
            {"duration": 1, "energy_kwh": 1.8, "peak_kw": 2.5},
            {"duration": 1, "energy_kwh": 1.8, "peak_kw": 2.5},
        ],
        "flex_model": "discrete_phase",
    },
}

# ──────────────────────────────────────────────────────────────────────────
#  COLOUR & TEXT HELPERS
# ──────────────────────────────────────────────────────────────────────────
_PALETTE = {
    "Original":              "#1f77b4",
    "Decentralized_NoBatt":  "#ff7f0e",
    "Decentralized_WithBatt":"#2ca02c",
    "SOC":                   "#1f77b4",
    "Charge":                "#2ca02c",
    "Discharge":             "#d62728",
}
def _get_color(k:str) -> str: return _PALETTE.get(k, "#333333")

# from data_processing.DataLoader import DataLoader
import logging
import os
import sys
import polars as pl
from pathlib import Path
from time import time

from agents.PVAgent import PVAgent
from agents.FlexibleDeviceAgent import FlexibleDevice
from agents.BatteryAgent import BatteryAgent
from agents.GlobalConnectionLayer import GlobalConnectionLayer
# from data_processing.BuildingDataPreProcess import BuildingDataPreProcess
# from data_processing.BuildingDataCleaner import BuildingDataCleaner
from agents.GridAgent import GridAgent
from agents.EVAgent import EVAgent
from utils.device_specs import device_specs
from agents.GlobalOptimizer import GlobalOptimizer
from agents.WeatherAgent import WeatherAgent

# Setup logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pulp import LpProblem, LpVariable, lpSum, LpStatus, PULP_CBC_CMD
from collections import defaultdict
from typing import List, Dict, Any
from datetime import datetime, timedelta, date as datetime_date
import logging
from typing import Dict, List
import logging
import pickle
import seaborn as sns
from typing import Dict, Any, List, Optional, Union
from sklearn.decomposition import PCA

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.colors as mcolors

# Add project root to path
project_root = str(Path.cwd().parent)
if project_root not in sys.path:
    sys.path.append(project_root)

###########################################################
# Helper functions 
###########################################################

def convert_dict_values_to_list(obj):
    """
    Recursively convert numpy types and arrays to Python native types for JSON serialization.
    Works with nested dictionaries and lists.
    
    Args:
        obj: The object to convert (dict, list, numpy array, etc.)
        
    Returns:
        A JSON-serializable version of the input object
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_dict_values_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_dict_values_to_list(i) for i in obj]
    elif isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    elif hasattr(obj, 'to_dict'):
        return convert_dict_values_to_list(obj.to_dict())
    else:
        return obj
def forecast_discount(forecast_value):
    # up to 40% discount if forecast_value ~1
    return 0.4 / (1 + np.exp(-10 * (forecast_value - 0.5)))


def get_season(date):
    if isinstance(date, (pd.Timestamp, datetime.datetime)):
        month = date.month
        day = date.day
    elif isinstance(date, datetime.date):
        month = date.month
        day = date.day
    else:
        raise ValueError("date must be datetime.date or datetime.datetime")
    if (month == 12 and day >= 21) or (month < 3) or (month == 3 and day < 20):
        return "Winter"
    elif (month == 3 and day >= 20) or (month < 6) or (month == 6 and day < 21):
        return "Spring"
    elif (month == 6 and day >= 21) or (month < 9) or (month == 9 and day < 23):
        return "Summer"
    else:
        return "Autumn"
    
def analyze_price_data(price_df):
    """
    Analyze electricity price data patterns and characteristics,
    focusing only on periods with valid price data.
    """
    import matplotlib.pyplot as plt
    import polars as pl
    import pandas as pd
    
    # Rename the price column for consistency
    price_df = price_df.rename({'DE_LU.2_price_day_ahead': 'price'})
    
    # Clean price data - remove nulls and get valid date range
    clean_price_df = price_df.filter(
        pl.col('price').is_not_null()
    ).with_columns([
        pl.col('utc_timestamp').dt.year().alias('year'),
        pl.col('utc_timestamp').dt.month().alias('month'),
        pl.col('utc_timestamp').dt.hour().alias('hour')
    ]).sort('utc_timestamp')
    
    # Print initial summary
    print("\nPrice Data Summary:")
    total_hours = price_df.height
    valid_hours = clean_price_df.height
    null_hours = total_hours - valid_hours
    
    print(f"Total Hours in Dataset: {total_hours}")
    print(f"Hours with Valid Prices: {valid_hours}")
    print(f"Hours with Null Prices: {null_hours}")
    print(f"\nValid Price Data Range:")
    print(f"Start: {clean_price_df['utc_timestamp'].min()}")
    print(f"End: {clean_price_df['utc_timestamp'].max()}")
    
    # Calculate price statistics
    price_stats = clean_price_df.select([
        pl.col('price').mean().alias('Mean'),
        pl.col('price').std().alias('Std Dev'),
        pl.col('price').min().alias('Min'),
        pl.col('price').max().alias('Max'),
        pl.col('price').median().alias('Median'),
        pl.col('price').quantile(0.25).alias('25th Percentile'),
        pl.col('price').quantile(0.75).alias('75th Percentile')
    ])
    
    print("\nPrice Statistics (€/MWh):")
    stats_dict = price_stats.row(0)
    for name, value in zip(price_stats.columns, stats_dict):
        print(f"{name}: {value:.2f}")
    
    # Calculate completeness by year
    yearly_stats = clean_price_df.group_by('year').agg([
        pl.len().alias('valid_hours'),
        (pl.len() / 8760 * 100).alias('completeness_percent')
    ]).sort('year')
    
    print("\nYearly Data Completeness:")
    print(yearly_stats)
    
    return clean_price_df

# OPTIMIZATION FUNCTIONS
# ----------------------------
from copy import deepcopy

logging.basicConfig(level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")

###############################################################################
# A) Utility Functions for Shading / Savings / Extracting Data
###############################################################################
def shade_price_hours(ax, price_by_hour):
    """(Not called in the main loop)"""
    low_thresh = price_by_hour.quantile(0.25)
    high_thresh = price_by_hour.quantile(0.75)
    for hr in range(24):
        p = price_by_hour.get(hr, 0.0)
        if p <= low_thresh:
            ax.axvspan(hr, hr+1, color='green', alpha=0.05)
        elif p >= high_thresh:
            ax.axvspan(hr, hr+1, color='red', alpha=0.05)

###############################
# 1) JADS PALETTE + COLOR MANAGER
###############################
def get_jads_color_palette():
    return {
        "brand_orange":       "#F5854F",  
        "brand_red":          "#E75C4B",
        "brand_grey":         "#6D6E71",
        "brand_gradient_blue":"#273E9E",
        "brand_gradient_red": "#9E273E",
        "brand_dark_grey":    "#4A4A4A",
        "black":              "#000000",
        "white":              "#FFFFFF"
    }

def create_jads_color_cycle():
    c = get_jads_color_palette()
    return [
        c["brand_orange"],
        c["brand_red"],
        c["brand_grey"],
        c["brand_gradient_blue"],
        c["brand_gradient_red"],
        c["brand_dark_grey"]
    ]

class ColorManager:
    """
    Dynamically assigns colors from a given cycle to unique entity names.
    Ensures each entity consistently gets the same color across all subplots.
    """
    def __init__(self, color_cycle=None):
        if color_cycle is None:
            color_cycle = create_jads_color_cycle()
        self.color_cycle = color_cycle
        self.mapping = {}
        self.index = 0

    def get_color(self, entity: str) -> str:
        if entity not in self.mapping:
            color = self.color_cycle[self.index % len(self.color_cycle)]
            self.mapping[entity] = color
            self.index += 1
        return self.mapping[entity]

# Create a global color manager for the script
g_color_mgr = ColorManager()

def plot_no_batt_vs_with_batt_dec_cent(dev_no_batt, dev_with_batt, building_id, has_pv):
    hours = np.arange(24)
    
    # Original consumption (same for both)
    orig_by_hour = pd.Series(dev_no_batt.original_consumption,
                             index=dev_no_batt.data['hour']).groupby(level=0).mean()
    
    # No battery: decentralized is in optimized_consumption, centralized in centralized_optimized_schedule
    decentralized_nb_by_hour = pd.Series(dev_no_batt.optimized_consumption,
                                     index=dev_no_batt.data['hour']).groupby(level=0).mean()
    centralized_nb_by_hour = None
    if hasattr(dev_no_batt, 'centralized_optimized_schedule'):
        centralized_nb_by_hour = pd.Series(dev_no_batt.centralized_optimized_schedule,
                                      index=dev_no_batt.data['hour']).groupby(level=0).mean()

    # With battery: same structure
    decentralized_wb_by_hour = None
    centralized_wb_by_hour = None
    if dev_with_batt is not None:
        decentralized_wb_by_hour = pd.Series(dev_with_batt.optimized_consumption,
                                       index=dev_with_batt.data['hour']).groupby(level=0).mean()
        if hasattr(dev_with_batt, 'centralized_optimized_schedule'):
            centralized_wb_by_hour = pd.Series(dev_with_batt.centralized_optimized_schedule,
                                          index=dev_with_batt.data['hour']).groupby(level=0).mean()

    # DEBUG: Print first few values of each array to verify they're different
    print(f"\nDEBUG INFO for {dev_no_batt.device_name}:")
    print("Original consumption (first 5 hours):", [f"{x:.6f}" for x in orig_by_hour.head(5).values])
    print("Decentralized (No Batt) (first 5 hours):", [f"{x:.6f}" for x in decentralized_nb_by_hour.head(5).values])
    if centralized_nb_by_hour is not None:
        print("Centralized (No Batt) (first 5 hours):", [f"{x:.6f}" for x in centralized_nb_by_hour.head(5).values])
    if decentralized_wb_by_hour is not None:
        print("Decentralized (With Batt) (first 5 hours):", [f"{x:.6f}" for x in decentralized_wb_by_hour.head(5).values])
    if centralized_wb_by_hour is not None:
        print("Centralized (With Batt) (first 5 hours):", [f"{x:.6f}" for x in centralized_wb_by_hour.head(5).values])

    # Compute savings metrics
    pct_dec_nb, euro_dec_nb, adjusted_dec_nb = compute_device_savings_dec_cent(dev_no_batt, use_optimized=True)
    pct_cent_nb, euro_cent_nb, adjusted_cent_nb = compute_device_savings_dec_cent(dev_no_batt, use_centralized=True) if centralized_nb_by_hour is not None else (0.0, 0.0, 0.0)
    
    if dev_with_batt is not None:
        pct_dec_wb, euro_dec_wb, adjusted_dec_wb = compute_device_savings_dec_cent(dev_with_batt, use_optimized=True)
        pct_cent_wb, euro_cent_wb, adjusted_cent_wb = compute_device_savings_dec_cent(dev_with_batt, use_centralized=True) if centralized_wb_by_hour is not None else (0.0, 0.0, 0.0)
    else:
        pct_dec_wb, euro_dec_wb, adjusted_dec_wb = (0.0, 0.0, 0.0)
        pct_cent_wb, euro_cent_wb, adjusted_cent_wb = (0.0, 0.0, 0.0)
    
    # DEBUG: Print the savings to verify they're calculated correctly
    print("\nSavings calculations:")
    print(f"Decentralized (No Batt): {pct_dec_nb:.2f}%, €{euro_dec_nb:.2f}, Adj: €{adjusted_dec_nb:.2f}")
    print(f"Centralized (No Batt): {pct_cent_nb:.2f}%, €{euro_cent_nb:.2f}, Adj: €{adjusted_cent_nb:.2f}")
    if dev_with_batt is not None:
        print(f"Decentralized (With Batt): {pct_dec_wb:.2f}%, €{euro_dec_wb:.2f}, Adj: €{adjusted_dec_wb:.2f}")
        print(f"Centralized (With Batt): {pct_cent_wb:.2f}%, €{euro_cent_wb:.2f}, Adj: €{adjusted_cent_wb:.2f}")

    price_hour = dev_no_batt.data.groupby('hour')['price_per_kwh'].mean()

    fig, ax = plt.subplots(figsize=(12,6))
    shade_price_hours(ax, price_hour)

    if has_pv and 'pv_actual' in dev_no_batt.data.columns:
        pv_by_hour = pd.Series(dev_no_batt.data['pv_actual'],
                               index=dev_no_batt.data['hour']).groupby(level=0).mean()
        threshold = pv_by_hour.quantile(0.75)
        for hr in range(24):
            if pv_by_hour.get(hr, 0.0) >= threshold:
                ax.axvspan(hr, hr+1, color='yellow', alpha=0.2,
                           label='High PV Production' if hr == 0 else None)

    # Plot original consumption
    ax.plot(hours, orig_by_hour.reindex(hours, fill_value=0.0),
            'k-', label="Original", linewidth=2)
    
    # Plot all available optimized schedules
    if decentralized_nb_by_hour is not None:
        ax.plot(hours, decentralized_nb_by_hour.reindex(hours, fill_value=0.0),
               color='green', linestyle='--', linewidth=2, label="Decentralized (No Batt)")
    
    if centralized_nb_by_hour is not None:
        ax.plot(hours, centralized_nb_by_hour.reindex(hours, fill_value=0.0),
               color='blue', linestyle='--', linewidth=2, label="Centralized (No Batt)")
    
    if decentralized_wb_by_hour is not None:
        ax.plot(hours, decentralized_wb_by_hour.reindex(hours, fill_value=0.0),
               color='orange', linestyle='-.', linewidth=2, label="Decentralized (With Batt)")
    
    if centralized_wb_by_hour is not None:
        ax.plot(hours, centralized_wb_by_hour.reindex(hours, fill_value=0.0),
               color='red', linestyle='-.', linewidth=2, label="Centralized (With Batt)")

    # Add annotations with better positioning
    annotations = []
    if decentralized_nb_by_hour is not None:
        annotations.append({
            "text": f"Dec-NB: {pct_dec_nb:.2f}%\n(€{euro_dec_nb:.2f})\nAdj: €{adjusted_dec_nb:.2f}",
            "color": "green",
            "xy": (4, decentralized_nb_by_hour.get(4, 0.01)),
            "xytext": (20, 30)
        })
    
    if centralized_nb_by_hour is not None:
        annotations.append({
            "text": f"Cent-NB: {pct_cent_nb:.2f}%\n(€{euro_cent_nb:.2f})\nAdj: €{adjusted_cent_nb:.2f}",
            "color": "blue",
            "xy": (11, centralized_nb_by_hour.get(11, 0.01)),
            "xytext": (-20, 40)
        })
    
    if decentralized_wb_by_hour is not None:
        annotations.append({
            "text": f"Dec-WB: {pct_dec_wb:.2f}%\n(€{euro_dec_wb:.2f})\nAdj: €{adjusted_dec_wb:.2f}",
            "color": "orange",
            "xy": (17, decentralized_wb_by_hour.get(17, 0.01)),
            "xytext": (30, -30)
        })
    
    if centralized_wb_by_hour is not None:
        annotations.append({
            "text": f"Cent-WB: {pct_cent_wb:.2f}%\n(€{euro_cent_wb:.2f})\nAdj: €{adjusted_cent_wb:.2f}",
            "color": "red",
            "xy": (21, centralized_wb_by_hour.get(21, 0.01)),
            "xytext": (-30, -40)
        })
    
    # Add annotations with minimal overlap
    # for i, anno in enumerate(annotations):
    #     ax.annotate(
    #         anno["text"],
    #         xy=anno["xy"],
    #         xytext=anno["xytext"],
    #         textcoords="offset points",
    #         ha='center', 
    #         va='center' if i % 2 == 0 else 'bottom',
    #         color=anno["color"],
    #         fontsize=9,
    #         bbox=dict(facecolor='white', alpha=0.7),
    #         arrowprops=dict(arrowstyle="->", color=anno["color"])
    #     )

    low_patch = Patch(color='green', alpha=0.1, label='Low Price Hours')
    high_patch = Patch(color='red', alpha=0.1, label='High Price Hours')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + [low_patch, high_patch],
             labels + ['Low Price Hours','High Price Hours'],
             loc='upper left')

    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Consumption (kWh)")
    ax.set_title(f"{dev_no_batt.device_name} — {building_id}\nOriginal vs Optimized Consumption")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_battery_usage_2subplots_dec_cent(device, building_id):
    """
    Plot battery SOC (top) and charge/discharge (bottom) for both optimization approaches.
    Fixed to handle missing legends properly.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Get timestamps
    timestamps = device.data['utc_timestamp']
    
    # Get the centralized battery data if available
    has_centralized = (hasattr(device, 'centralized_battery_soc') and 
                      device.centralized_battery_soc is not None)
    
    # Plot SOC for both approaches
    ax1.plot(timestamps, device.battery_soc, 
             label='Decentralized SOC', color='orange', linewidth=2)
    
    if has_centralized:
        ax1.plot(timestamps, device.centralized_battery_soc, 
                 label='Centralized SOC', color='red', linewidth=2, linestyle='--')
    
    # Add SOC min/max lines if we have access to them
    if hasattr(device, 'battery_agent') and device.battery_agent is not None:
        soc_min = device.battery_agent.soc_min
        soc_max = device.battery_agent.soc_max
        ax1.axhline(y=soc_min, color='grey', linestyle=':', label=f'Min SOC ({soc_min:.1f} kWh)')
        ax1.axhline(y=soc_max, color='grey', linestyle='-.', label=f'Max SOC ({soc_max:.1f} kWh)')
    
    ax1.set_ylabel('State of Charge (kWh)')
    ax1.set_title(f'Battery Usage for {device.device_name} — {building_id}')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    
    # Plot charge/discharge for both approaches
    ax2.plot(timestamps, device.battery_charge, 
             label='Decentralized Charge', color='green', linewidth=1.5)
    ax2.plot(timestamps, -device.battery_discharge, 
             label='Decentralized Discharge', color='blue', linewidth=1.5)
    
    if has_centralized:
        ax2.plot(timestamps, device.centralized_battery_charge, 
                 label='Centralized Charge', color='darkgreen', linestyle='--', linewidth=1.5)
        ax2.plot(timestamps, -device.centralized_battery_discharge, 
                 label='Centralized Discharge', color='darkblue', linestyle='--', linewidth=1.5)
    
    # Add power limits if we have access to them
    if hasattr(device, 'battery_agent') and device.battery_agent is not None:
        charge_limit = device.battery_agent.max_charge_rate
        discharge_limit = device.battery_agent.max_discharge_rate
        ax2.axhline(y=charge_limit, color='grey', linestyle=':', 
                   label=f'Max Charge Rate ({charge_limit:.1f} kW)')
        ax2.axhline(y=-discharge_limit, color='grey', linestyle='-.', 
                   label=f'Max Discharge Rate (-{discharge_limit:.1f} kW)')
    
    ax2.set_ylabel('Power (kW)')
    ax2.set_xlabel('Time')
    ax2.grid(True, alpha=0.3)
    
    # Add price information as background
    if 'price_per_kwh' in device.data.columns:
        price = device.data['price_per_kwh'].values
        price_norm = (price - np.min(price)) / (np.max(price) - np.min(price) + 1e-9)
        
        # Create a twin axis for the price
        ax3 = ax2.twinx()
        ax3.plot(timestamps, price, color='black', alpha=0.5, linewidth=1, label='Price (€/kWh)')
        ax3.set_ylabel('Price (€/kWh)')
        
        # FIX: Combine legends safely
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines3, labels3 = ax3.get_legend_handles_labels()
        
        # Create a single legend on ax2 with all items
        ax2.legend(lines1 + lines3, labels1 + labels3, loc='lower right')
        
        # FIX: Don't try to remove ax3's legend since it doesn't have one yet
        # ax3.get_legend().remove()  <- This was causing the error
    else:
        # If no price data, just show ax2's legend
        ax2.legend(loc='best')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\nBattery Usage Summary:")
    print("Decentralized Optimization:")
    print(f"  Initial SOC: {device.battery_soc[0]:.2f} kWh")
    print(f"  Final SOC: {device.battery_soc[-1]:.2f} kWh")
    print(f"  Max SOC: {np.max(device.battery_soc):.2f} kWh")
    print(f"  Min SOC: {np.min(device.battery_soc):.2f} kWh")
    print(f"  Total Energy Charged: {np.sum(device.battery_charge):.2f} kWh")
    print(f"  Total Energy Discharged: {np.sum(device.battery_discharge):.2f} kWh")
    
    if has_centralized:
        print("\nCentralized Optimization:")
        print(f"  Initial SOC: {device.centralized_battery_soc[0]:.2f} kWh")
        print(f"  Final SOC: {device.centralized_battery_soc[-1]:.2f} kWh")
        print(f"  Max SOC: {np.max(device.centralized_battery_soc):.2f} kWh")
        print(f"  Min SOC: {np.min(device.centralized_battery_soc):.2f} kWh")
        print(f"  Total Energy Charged: {np.sum(device.centralized_battery_charge):.2f} kWh")
        print(f"  Total Energy Discharged: {np.sum(device.centralized_battery_discharge):.2f} kWh")
        
        # Calculate difference metrics
        soc_diff = np.mean(np.abs(device.battery_soc - device.centralized_battery_soc))
        charge_diff = np.mean(np.abs(device.battery_charge - device.centralized_battery_charge))
        discharge_diff = np.mean(np.abs(device.battery_discharge - device.centralized_battery_discharge))
        
        print("\nDifference Between Approaches:")
        print(f"  Mean Absolute SOC Difference: {soc_diff:.2f} kWh")
        print(f"  Mean Absolute Charge Difference: {charge_diff:.2f} kW")
        print(f"  Mean Absolute Discharge Difference: {discharge_diff:.2f} kW")

# --------------------------------------------------------------------
# V) NEW: Plot Savings Uncertainty with Confidence Intervals
# --------------------------------------------------------------------
def plot_savings_uncertainty(dates, deterministic_savings, mc_savings_samples):
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.dates as mdates

    nominal = np.array(deterministic_savings)
    lower = []
    upper = []
    for samples in mc_savings_samples:
        samples = np.array(samples)
        lower.append(np.percentile(samples, 2.5))
        upper.append(np.percentile(samples, 97.5))
    lower = np.array(lower)
    upper = np.array(upper)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dates, nominal, marker='o', color='blue', label='Deterministic Savings')
    ax.fill_between(dates, lower, upper, color='blue', alpha=0.2, label='95% Confidence Interval (MC)')
    ax.set_ylabel("Daily Savings (€)")
    ax.set_title("Daily Savings: Deterministic vs. Forecast Uncertainty (Monte Carlo)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    all_samples = np.concatenate(mc_savings_samples)
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.boxplot(all_samples, patch_artist=True)
    ax2.set_ylabel("Daily Savings (€)")
    ax2.set_title("Distribution of Daily Savings (Monte Carlo Simulation)")
    plt.tight_layout()
    plt.show()

# --------------------------------------------------------------------
# D) Single-figure: Original vs SHIFT(no batt) vs SHIFT(with batt)
# --------------------------------------------------------------------
def plot_no_batt_vs_with_batt(dev_no_batt, dev_with_batt, building_id, has_pv):
    import pandas as pd
    from matplotlib.patches import Patch
    import matplotlib.pyplot as plt
    import numpy as np

    hours = np.arange(24)
    # Compute original and optimized consumption by hour
    orig_by_hour = pd.Series(dev_no_batt.original_consumption, index=dev_no_batt.data['hour']).groupby(level=0).mean()
    shift_nb_by_hour = pd.Series(dev_no_batt.optimized_consumption, index=dev_no_batt.data['hour']).groupby(level=0).mean()
    
    shift_batt_by_hour = None
    if dev_with_batt is not None:
        shift_batt_by_hour = pd.Series(dev_with_batt.optimized_consumption, index=dev_with_batt.data['hour']).groupby(level=0).mean()
    
    # Compute savings metrics
    pct_nb, euro_nb, adjusted_nb = compute_device_savings(dev_no_batt)
    if shift_batt_by_hour is not None:
        pct_batt, euro_batt, adjusted_batt = compute_device_savings(dev_with_batt)
    else:
        pct_batt, euro_batt, adjusted_batt = (0.0, 0.0, 0.0)
    
    price_hour = dev_no_batt.data.groupby('hour')['price_per_kwh'].mean()
    
    # Use JADS palette via global color manager
    orig_color    = g_color_mgr.get_color("Original")
    dec_nb_color  = g_color_mgr.get_color("Decentralized_NoBatt")
    dec_wb_color  = g_color_mgr.get_color("Decentralized_WithBatt")
    
    fig, ax = plt.subplots(figsize=(12,6))
    shade_price_hours(ax, price_hour)
    
    # Shade high PV production with very light yellow
    if has_pv and 'pv_actual' in dev_no_batt.data.columns:
        pv_by_hour = pd.Series(dev_no_batt.data['pv_actual'], index=dev_no_batt.data['hour']).groupby(level=0).mean()
        threshold = pv_by_hour.quantile(0.75)
        for hr in range(24):
            if pv_by_hour.get(hr, 0.0) >= threshold:
                ax.axvspan(hr, hr+1, color='yellow', alpha=0.05, label='High PV Production' if hr == 0 else None)
    
    # Plot the consumption curves
    ax.plot(hours, orig_by_hour.reindex(hours, fill_value=0.0), '-', color=orig_color, label="Original", linewidth=2)
    ax.plot(hours, shift_nb_by_hour.reindex(hours, fill_value=0.0), '--', color=dec_nb_color, label="Shifted (No Batt)", linewidth=2)
    if shift_batt_by_hour is not None:
        ax.plot(hours, shift_batt_by_hour.reindex(hours, fill_value=0.0), '-.', color=dec_wb_color, label="Shifted (With Batt)", linewidth=2)
    
    # Add annotations at mid-day
    mid_hour = 12
    y_nb = shift_nb_by_hour.get(mid_hour, 0.0)
    ax.annotate(f"{pct_nb:.2f}%\n(€{euro_nb:.2f})\nAdj: €{adjusted_nb:.2f}",
                xy=(mid_hour, y_nb), xytext=(0, 10), textcoords="offset points",
                ha='center', va='bottom', color=dec_nb_color, fontsize=10,
                bbox=dict(facecolor='white', alpha=0.7))
    if shift_batt_by_hour is not None:
        y_batt = shift_batt_by_hour.get(mid_hour, 0.0)
        ax.annotate(f"{pct_batt:.2f}%\n(€{euro_batt:.2f})\nAdj: €{adjusted_batt:.2f}",
                    xy=(mid_hour, y_batt), xytext=(0, -20), textcoords="offset points",
                    ha='center', va='top', color=dec_wb_color, fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.7))
    
    # Create legend patches for price shading (using light versions)
    low_patch = Patch(color=g_color_mgr.get_color("Decentralized_NoBatt"), alpha=0.03, label='Low Price Hours')
    high_patch = Patch(color=g_color_mgr.get_color("Decentralized_WithBatt"), alpha=0.03, label='High Price Hours')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + [low_patch, high_patch],
              labels + ['Low Price Hours','High Price Hours'],
              loc='upper left')
    
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Consumption (kWh)")
    ax.set_title(f"{dev_no_batt.device_name} — {building_id}\nOriginal vs Shifted (No Batt) vs Shifted (With Batt)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_no_batt_vs_with_batt_building(
    devices_no_batt, devices_with_batt, building_id, has_pv=False
):
    """
    Aggregates consumption across all devices in the building for each hour (0–23),
    then plots:
      - Original consumption (sum of all devices' original_consumption)
      - Shifted (No Batt) consumption
      - Shifted (With Batt) consumption

    Uses 'compute_building_savings()' to correctly account for battery synergy
    (export at 80% price, battery degradation cost, etc.) for the savings annotations.
    Also applies the JADS color palette, shades low/high price hours, and
    annotates approximate building-level synergy-based savings.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # Helper to sum a numeric array by hour-of-day
    def sum_by_hour(array, dev):
        s = pd.Series(array, index=dev.data['hour']).groupby(level=0).sum()
        return s.reindex(range(24), fill_value=0.0)

    # 1) Build aggregated consumption for Original, No Batt, With Batt (for plotting only)
    building_original = np.zeros(24)
    building_no_batt  = np.zeros(24)
    building_with_batt= np.zeros(24)

    # We'll pick the first device from either list to get a "reference" price_by_hour
    if len(devices_no_batt) > 0:
        dev_ref = devices_no_batt[0]
    elif len(devices_with_batt) > 0:
        dev_ref = devices_with_batt[0]
    else:
        print(f"No devices found for building {building_id}. Nothing to plot.")
        return

    # Accumulate sums for the no-battery scenario
    for dev in devices_no_batt:
        building_original += sum_by_hour(dev.original_consumption, dev).values
        building_no_batt  += sum_by_hour(dev.optimized_consumption, dev).values

    # Accumulate sums for the with-battery scenario
    if devices_with_batt:
        for dev in devices_with_batt:
            building_with_batt += sum_by_hour(dev.optimized_consumption, dev).values

    # 2) Retrieve a representative price_by_hour from dev_ref (for shading)
    hours = np.arange(24)
    price_hour = dev_ref.data.groupby('hour')['price_per_kwh'].mean().reindex(hours, fill_value=0.0)

    # 3) Use synergy-aware function to get building-level savings for each scenario
    pct_nb, euro_nb, _, _, total_orig_cost_nb, total_opt_cost_nb = compute_building_savings(devices_no_batt)
    pct_wb, euro_wb, _, _, total_orig_cost_wb, total_opt_cost_wb = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    if devices_with_batt:
        pct_wb, euro_wb, _, _, total_orig_cost_wb, total_opt_cost_wb = compute_building_savings(devices_with_batt)

    # 4) Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Shade low/high price hours with your existing function
    # (Make sure your shade_price_hours can handle lighter alpha if needed.)
    shade_price_hours(ax, price_hour)

    # JADS color manager (g_color_mgr) for consistent brand colors
    color_orig = g_color_mgr.get_color("Building_Original")
    color_nb   = g_color_mgr.get_color("Building_NoBatt")
    color_wb   = g_color_mgr.get_color("Building_WithBatt")

    # Plot lines
    ax.plot(hours, building_original,  color=color_orig, linewidth=2,
            label="Original (Sum of All Devices)")
    ax.plot(hours, building_no_batt,   color=color_nb,   linestyle='--', linewidth=2,
            label="Shifted (No Batt)")
    if np.any(building_with_batt > 0):
        ax.plot(hours, building_with_batt, color=color_wb, linestyle='-.', linewidth=2,
                label="Shifted (With Batt)")

    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Total Building Consumption (kWh)")
    ax.set_title(f"Building-Level: Original vs Shifted (No Batt) vs Shifted (With Batt)\n{building_id}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')

    # 5) Annotate synergy-based savings
    #   For example, place the "No Batt" annotation near hour=5, "With Batt" near hour=17
    #   Adjust offsets as needed for clarity.
    x_nb = 5
    ax.annotate(
        f"No Batt:\n{pct_nb:.1f}% (€{euro_nb:.2f})\nOrig: €{total_orig_cost_nb:.2f}, Opt: €{total_opt_cost_nb:.2f}",
        xy=(x_nb, building_no_batt[x_nb]),
        xytext=(0, 15), textcoords="offset points",
        ha='center', va='bottom', color=color_nb, fontsize=10,
        bbox=dict(facecolor='white', alpha=0.7),
        arrowprops=dict(arrowstyle="->", color=color_nb)
    )

    if np.any(building_with_batt > 0):
        x_wb = 17
        ax.annotate(
            f"With Batt:\n{pct_wb:.1f}% (€{euro_wb:.2f})\nOrig: €{total_orig_cost_wb:.2f}, Opt: €{total_opt_cost_wb:.2f}",
            xy=(x_wb, building_with_batt[x_wb]),
            xytext=(0, -25), textcoords="offset points",
            ha='center', va='top', color=color_wb, fontsize=10,
            bbox=dict(facecolor='white', alpha=0.7),
            arrowprops=dict(arrowstyle="->", color=color_wb)
        )

    plt.tight_layout()
    plt.show()
   

def plot_unified_battery_usage(devices_with_batt, building_id):
    """
    Creates a unified plot showing aggregated battery usage across all devices,
    grouped by hour of day, with correctly calculated SOC.
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Filter devices that have battery data
    valid_devices = [dev for dev in devices_with_batt if hasattr(dev, 'battery_soc') and dev.battery_soc is not None]
    
    if not valid_devices:
        print(f"No valid battery data found for devices in {building_id}")
        return
    
    # Setup hour range
    hours = np.arange(24)
    
    # Initialize arrays for combined hourly data
    combined_charge = np.zeros(24)
    combined_discharge = np.zeros(24)
    
    # Get SOC values from the first device (dishwasher) for reference
    first_device = valid_devices[0]  # This is the dishwasher based on the images
    first_device_soc = pd.Series(first_device.battery_soc, index=first_device.data['hour']).groupby(level=0).mean()
    
    # CRITICAL FIX: Use the actual starting SOC from hour 0 of the first device
    initial_soc = first_device_soc.get(0, 8.0)  # Default to 8.0 if hour 0 not found
    
    # 2) Aggregate charge/discharge data by hour across all devices
    for dev in valid_devices:
        # Group by hour and calculate mean charge/discharge
        hour_charge = pd.Series(dev.battery_charge, index=dev.data['hour']).groupby(level=0).mean()
        hour_discharge = pd.Series(dev.battery_discharge, index=dev.data['hour']).groupby(level=0).mean()
        
        # Add to combined values for each hour
        for hour in hours:
            if hour in hour_charge.index:
                combined_charge[hour] += hour_charge[hour]
            if hour in hour_discharge.index:
                combined_discharge[hour] += hour_discharge[hour]
    
    # 3) Calculate SOC by integrating charge/discharge
    # Get battery parameters from first device
    if hasattr(first_device, 'battery_agent') and first_device.battery_agent is not None:
        soc_min = first_device.battery_agent.soc_min
        soc_max = first_device.battery_agent.soc_max
        charge_efficiency = getattr(first_device.battery_agent, 'charge_efficiency', 0.95)
        discharge_efficiency = getattr(first_device.battery_agent, 'discharge_efficiency', 0.95)
    else:
        soc_min = 0.6
        soc_max = 12.0
        charge_efficiency = 0.95
        discharge_efficiency = 0.95
    
    # Calculate SOC trajectory
    soc_values = np.zeros(24)
    soc_values[0] = initial_soc  # Use the actual starting SOC from dishwasher
    
    for t in range(1, 24):
        soc_values[t] = soc_values[t-1] + (combined_charge[t-1] * charge_efficiency) - (combined_discharge[t-1] / discharge_efficiency)
        soc_values[t] = max(soc_min, min(soc_values[t], soc_max))
    
    # Remainder of the function stays the same...
    # 4) Get average price data by hour
    price_hour = pd.Series(0.0, index=hours)
    if 'price_per_kwh' in first_device.data.columns:
        price_hour = first_device.data.groupby('hour')['price_per_kwh'].mean().reindex(hours, fill_value=0.0)
    
    # 5) Create visualization
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Colors
    soc_color = 'orange'
    chg_color = 'darkred'
    dch_color = 'gray'
    
    # Plot SOC
    ax_top.plot(hours, soc_values, color=soc_color, linewidth=2, label='Battery SoC (Integrated)')
    ax_top.axhline(soc_min, color='grey', linestyle=':', label=f"Min SOC ({soc_min:.1f} kWh)")
    ax_top.axhline(soc_max, color='grey', linestyle='-.', label=f"Max SOC ({soc_max:.1f} kWh)")
    ax_top.set_ylabel("Battery SoC (kWh)")
    ax_top.set_title(f"Unified Battery Usage for all devices in {building_id}")
    ax_top.grid(True, alpha=0.3)
    ax_top.legend(loc='best')
    
    # Plot charge/discharge
    ax_bot.plot(hours, combined_charge, color=chg_color, linewidth=2, label="Combined Charge (kW)")
    ax_bot.plot(hours, -combined_discharge, color=dch_color, linestyle='--', linewidth=2, label="Combined Discharge (kW)")
    ax_bot.set_xlabel("Hour of Day")
    ax_bot.set_ylabel("Power (kW)")
    ax_bot.grid(True, alpha=0.3)
    
    # Twin axis for price
    ax_twin = ax_bot.twinx()
    ax_twin.plot(hours, price_hour, color='black', alpha=0.3, linewidth=1, label='Price (€/kWh)')
    ax_twin.set_ylabel("Price (€/kWh)")
    
    # Merge legends
    lines_bot, labels_bot = ax_bot.get_legend_handles_labels()
    lines_twin, labels_twin = ax_twin.get_legend_handles_labels()
    ax_bot.legend(lines_bot + lines_twin, labels_bot + labels_twin, loc='upper left')
    
    plt.tight_layout()
    plt.show()

def plot_battery_usage_2subplots(dev_with_batt, building_id):
    """
    Plots (top) the battery SoC by hour-of-day, 
    and (bottom) charge/discharge plus a twin axis for average price by hour. 
    Also annotates the device’s total cost savings with battery.
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # 1) Aggregate battery SoC, charge, discharge by hour-of-day
    hours = np.arange(24)
    soc_hour = (pd.Series(dev_with_batt.battery_soc, index=dev_with_batt.data['hour'])
                .groupby(level=0).mean())
    chg_hour = (pd.Series(dev_with_batt.battery_charge, index=dev_with_batt.data['hour'])
                .groupby(level=0).mean())
    dch_hour = (pd.Series(dev_with_batt.battery_discharge, index=dev_with_batt.data['hour'])
                .groupby(level=0).mean())
    
    # 2) Compute average price by hour
    if 'price_per_kwh' in dev_with_batt.data.columns:
        price_hour = (dev_with_batt.data.groupby('hour')['price_per_kwh']
                      .mean()
                      .reindex(hours, fill_value=0.0))
    else:
        price_hour = pd.Series([0.0]*24, index=hours)

    # 3) Prepare figure and color palette
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Retrieve colors from your JADS palette manager
    soc_color  = g_color_mgr.get_color("SOC")
    chg_color  = g_color_mgr.get_color("Charge")
    dch_color  = g_color_mgr.get_color("Discharge")

    # 4) Top subplot: Battery SoC
    ax_top.plot(hours, soc_hour.reindex(hours, fill_value=0.0),
                color=soc_color, linewidth=2, label='SoC')
    
    # Draw min/max lines if we have them
    if hasattr(dev_with_batt, 'battery_agent') and dev_with_batt.battery_agent is not None:
        soc_min = dev_with_batt.battery_agent.soc_min
        soc_max = dev_with_batt.battery_agent.soc_max
        ax_top.axhline(soc_min, color='grey', linestyle=':',
                       label=f"Min SOC ({soc_min:.1f} kWh)")
        ax_top.axhline(soc_max, color='grey', linestyle='-.',
                       label=f"Max SOC ({soc_max:.1f} kWh)")
    
    ax_top.set_ylabel("Battery SoC (kWh)", color=soc_color)
    ax_top.set_title(f"Battery SoC for {dev_with_batt.device_name} in {building_id}")
    ax_top.grid(True, alpha=0.3)
    ax_top.legend(loc='best')

    # 5) Bottom subplot: charge/discharge + price as twin axis
    ax_bot.plot(hours, chg_hour.reindex(hours, fill_value=0.0),
                color=chg_color, linewidth=2, label="Charge (kW)")
    ax_bot.plot(hours, -dch_hour.reindex(hours, fill_value=0.0),
                color=dch_color, linestyle='--', linewidth=2, label="Discharge (kW)")
    
    ax_bot.set_xlabel("Hour of Day")
    ax_bot.set_ylabel("Power (kW)")
    ax_bot.grid(True, alpha=0.3)
    
    # Twin axis for the average price by hour
    ax_twin = ax_bot.twinx()
    ax_twin.plot(hours, price_hour, color='black', alpha=0.3, linewidth=1, label='Price (€/kWh)')
    ax_twin.set_ylabel("Price (€/kWh)")

    # Merge legends from ax_bot + ax_twin
    lines_bot, labels_bot   = ax_bot.get_legend_handles_labels()
    lines_twin, labels_twin = ax_twin.get_legend_handles_labels()
    ax_bot.legend(lines_bot + lines_twin, labels_bot + labels_twin, loc='upper left')

    # 6) Annotate battery’s overall cost savings
    pct_batt, euro_batt, adjusted_batt = compute_device_savings(dev_with_batt)
    # ax_bot.annotate(
    #     f"Savings: {pct_batt:.2f}%\n(€{euro_batt:.2f})\nAdj: €{adjusted_batt:.2f}",
    #     xy=(0.98, 0.90), xycoords='axes fraction',
    #     ha='right', va='top',
    #     bbox=dict(facecolor='white', alpha=0.7),
    #     fontsize=9, color='black'
    # )

    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------------
# F) PLOTTING FUNCTIONS (ADDITIONAL DETAILED PLOTS)
# --------------------------------------------------------------------
def plot_additional_plots(devices, building_id):
    """
    (3) Original vs Initial Start
    (4) Original vs Initial vs Next Day
    (5.a) Heatmap of 'Initial Start' coverage
    (5.b) Heatmap of 'Next Day' coverage
    """
    # # (3) Original vs Initial Start
    # for dev in devices:
    #     if dev.weekday_optimized_schedule is not None:
    #         plt.figure(figsize=(12, 6))
    #         price_hour = dev.data.groupby('hour')['price_per_kwh'].mean()
    #         shade_price_hours(plt.gca(), price_hour)

    #         c_orig = "black"
    #         c_init = "purple"

    #         orig = pd.Series(dev.original_consumption, index=dev.data['hour']).groupby(level=0).mean()
    #         plt.plot(orig.index, orig.values, '-', color=c_orig, linewidth=2, label='Original')
    #         plt.step(range(24), dev.weekday_optimized_schedule,
    #                  where='post', color=c_init, linewidth=2, label='Initial Start')
    #         plt.xlabel("Hour of Day")
    #         plt.ylabel("Consumption (kWh)")
    #         plt.title(f"(3) Original vs Initial Start — {dev.device_name} — {building_id}")
    #         plt.legend()
    #         plt.grid(alpha=0.3)
    #         plt.show()

    # # (4) Original vs Initial Start vs Next Day
    # for dev in devices:
    #     if (dev.weekday_optimized_schedule is not None) or (dev.nextday_optimized_schedule is not None):
    #         plt.figure(figsize=(12, 6))
    #         price_hour = dev.data.groupby('hour')['price_per_kwh'].mean()
    #         shade_price_hours(plt.gca(), price_hour)

    #         c_orig = "black"
    #         c_init = "purple"
    #         c_nd   = "orange"

    #         orig = pd.Series(dev.original_consumption, index=dev.data['hour']).groupby(level=0).mean()
    #         plt.plot(orig.index, orig.values, '-', color=c_orig, linewidth=2, label='Original')
    #         if dev.weekday_optimized_schedule is not None:
    #             plt.step(range(24), dev.weekday_optimized_schedule,
    #                      where='post', color=c_init, linewidth=2, label='Initial Start')
    #         if dev.nextday_optimized_schedule is not None:
    #             plt.step(range(24), dev.nextday_optimized_schedule,
    #                      where='post', color=c_nd, linewidth=2, label='Next Day')
    #         plt.xlabel("Hour of Day")
    #         plt.ylabel("Consumption (kWh)")
    #         plt.title(f"(4) Original vs Initial Start vs Next Day — {dev.device_name} — {building_id}")
    #         plt.legend()
    #         plt.grid(alpha=0.3)
    #         plt.show()

    # (5.a) Heatmap of 'Initial Start' coverage
    init_matrix = []
    dev_names = []
    for dev in devices:
        schedule = dev.weekday_optimized_schedule if dev.weekday_optimized_schedule is not None else [0]*24
        init_matrix.append(schedule)
        dev_names.append(dev.device_name)
    plt.figure(figsize=(12, max(4, len(devices)/2)))
    sns.heatmap(init_matrix, cmap="YlOrRd", xticklabels=range(24), yticklabels=dev_names)
    plt.title(f"(5.a) Heatmap 'Initial Start' Coverage — {building_id}")
    plt.xlabel("Hour of Day")
    plt.ylabel("Devices")
    plt.show()

    # (5.b) Heatmap of 'Next Day' coverage
    nd_matrix = []
    nd_dev_names = []
    for dev in devices:
        schedule = dev.nextday_optimized_schedule if dev.nextday_optimized_schedule is not None else [0]*24
        nd_matrix.append(schedule)
        nd_dev_names.append(dev.device_name)
    plt.figure(figsize=(12, max(4, len(devices)/2)))
    sns.heatmap(nd_matrix, cmap="YlOrRd", xticklabels=range(24), yticklabels=nd_dev_names)
    plt.title(f"(5.b) Heatmap 'Next Day' Coverage — {building_id}")
    plt.xlabel("Hour of Day")
    plt.ylabel("Devices")
    plt.show()

# NEW (5.c): Heatmap of aggregated battery charging only (across all devices)

def plot_aggregated_battery_charging_mean(devices, building_id):
    """
    Plots a simplified battery charging heatmap with one row showing the average 
    charging per hour across all days and all devices.
    
    Args:
      devices (list): List of device objects with 'battery_charge' arrays.
      building_id (str): Identifier for the building.
    """
    # Collect all battery charge data by hour
    hourly_totals = np.zeros(24)
    hourly_counts = np.zeros(24)
    
    # Gather hourly charging data from all devices
    for dev in devices:
        if hasattr(dev, "battery_charge") and dev.battery_charge is not None:
            # For each device, sum up charges by hour
            for idx, charge in enumerate(dev.battery_charge):
                hour = dev.data['hour'].iloc[idx]
                hourly_totals[hour] += charge
                hourly_counts[hour] = 1
    
    # Calculate average per hour
    hourly_avg = np.divide(hourly_totals, hourly_counts, out=np.zeros_like(hourly_totals), where=hourly_counts>0)
    
    # Create a DataFrame with just one row
    battery_charge_df = pd.DataFrame([hourly_avg], index=[f"{building_id}_average"], columns=range(24))
    
    plt.figure(figsize=(12, 2))
    ax = sns.heatmap(
        battery_charge_df,
        cmap="Greens",
        annot=True,
        fmt=".2f",
        xticklabels=range(24),
        yticklabels=battery_charge_df.index
    )
    
    # Titles and labels
    plt.title(f"Battery Charging Schedule — {building_id}")
    plt.xlabel("Hour of Day")
    plt.tight_layout()
    plt.show()

def plot_battery_schedule(battery_charge, building_id, ds):
    """
    Plot a single‐row heatmap of the battery charge schedule for day `ds`.
    Saves figure to file instead of displaying it.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import os

    df = pd.DataFrame(
        [battery_charge],
        index=[f"{building_id}_battery"],
        columns=list(range(24))
    )

    fig = plt.figure(figsize=(12,2))
    ax = sns.heatmap(df,
                     cmap="Greens",
                     annot=True,
                     fmt=".2f",
                     cbar_kws={"label":"kWh charged"},
                     xticklabels=range(24),
                     yticklabels=df.index)
    ax.set_xlabel("Hour of Day")
    ax.set_title(f"Battery Charge Schedule — {building_id} [{ds}]")
    plt.tight_layout()
    
    # Create a plots directory if it doesn't exist
    os.makedirs("plots", exist_ok=True)
    
    # Save figure instead of displaying
    filename = f"plots/{building_id}_battery_schedule_{ds}.png"
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    print(f"Saved battery schedule plot to {filename}")
    plt.close(fig)

# in utils/helper.py, at the bottom (after your other plot functions)

def plot_battery_soc_history(battery_agent, dates, building_id, is_ev=False):
    """
    Plots end-of-day battery or EV State-of-Charge for a sequence of days.
    Saves figure to file instead of displaying it.

    Args:
      battery_agent: your single, persistent BatteryAgent or EVAgent instance
      dates (list of date or str): the same live_days you iterated over
      building_id (str): for your title
      is_ev (bool, optional): Whether this is an EV (True) or battery (False). Default is False.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import os

    # turn your dates into a proper index
    idx = pd.to_datetime(dates)

    # grab the history the agent accumulated
    soc = battery_agent.soc_history

    if len(idx) != len(soc):
        print(f"[warn] you passed {len(dates)} dates but have {len(soc)} soc points")
        # Handle the mismatch - take only the first len(idx) elements from soc if soc is longer
        if len(soc) > len(idx):
            soc = soc[:len(idx)]
        # Or if dates is longer, extend soc with NaN values
        elif len(idx) > len(soc):
            extension = [np.nan] * (len(idx) - len(soc))
            soc = np.concatenate([soc, extension])
    
    # build the plot
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(idx, soc, marker='o', linestyle='-')
    
    # Set title and labels based on whether this is battery or EV
    device_type = "EV" if is_ev else "Battery"
    ax.set_title(f"End-of-Day {device_type} SoC — {building_id}")
    ax.set_xlabel("Date")
    ax.set_ylabel("SoC (kWh)")
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Create a plots directory if it doesn't exist
    os.makedirs("plots", exist_ok=True)
    
    # Save figure instead of displaying
    filename = f"plots/{building_id}_{'ev' if is_ev else 'battery'}_soc_history.png"
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    print(f"Saved {'EV' if is_ev else 'Battery'} SOC history plot to {filename}")
    plt.close(fig)

def plot_multi_day_battery(scheduling_results, building_id):
    """
    Plots hourly battery SoC, charge, and discharge across multiple days.
    
    Args:
      scheduling_results: list of (date, devices) tuples from your live loop
      building_id: for titling
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # prepare
    dates = [ds for ds, _ in scheduling_results]
    n_days = len(dates)
    # each 'devices' is the list returned by run_building_optimization_single_day_direct
    # we assume battery arrays are on devices[0]
    soc_mat = []
    chg_mat = []
    dis_mat = []
    for _, devices in scheduling_results:
        dev = devices[0]
        # these should be length-24 lists or arrays
        soc = getattr(dev, "battery_soc_day", None)
        chg = getattr(dev, "battery_charge_day", None)
        dis = getattr(dev, "battery_discharge_day", None)
        if soc is None or chg is None or dis is None:
            raise ValueError(f"Device {dev.device_name} missing battery_day arrays")
        soc_mat.append(soc)
        chg_mat.append(chg)
        dis_mat.append(dis)
    soc_mat = np.array(soc_mat)        # shape (n_days, 24)
    chg_mat = np.array(chg_mat)
    dis_mat = np.array(dis_mat)

    # human‐readable row labels
    row_labels = [pd.to_datetime(d).strftime("%Y-%m-%d") for d in dates]
    hours = np.arange(24)

    # one figure, three stacked heatmaps
    fig, axes = plt.subplots(3, 1, figsize=(12, 4 + n_days*0.3), 
                             sharex=True, gridspec_kw={"height_ratios":[1,1,1]})
    
    # Top: State of Charge
    sns.heatmap(soc_mat, ax=axes[0],
                cmap="Oranges", cbar_kws={"label":"SoC (kWh)"},
                xticklabels=hours, yticklabels=row_labels)
    axes[0].set_ylabel("Date")
    axes[0].set_title(f"Battery State‐of‐Charge — {building_id}")

    # Middle: Charging
    sns.heatmap(chg_mat, ax=axes[1],
                cmap="Greens", cbar_kws={"label":"Charge (kW)"},
                xticklabels=hours, yticklabels=[])
    axes[1].set_ylabel("")

    # Bottom: Discharging (we’ll plot positive values, but label it “Discharge”)
    sns.heatmap(dis_mat, ax=axes[2],
                cmap="Reds", cbar_kws={"label":"Discharge (kW)"},
                xticklabels=hours, yticklabels=[])
    axes[2].set_ylabel("")

    # x‐axis label only on bottom
    axes[2].set_xlabel("Hour of Day")

    plt.tight_layout(h_pad=1.0)
    
    # Create a plots directory if it doesn't exist
    import os
    os.makedirs("plots", exist_ok=True)
    
    # Save figure instead of displaying
    filename = f"plots/{building_id}_multi_day_battery.png"
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    print(f"Saved multi-day battery plot to {filename}")
    plt.close(fig)

def plot_ev_schedule(ev_charge, building_id, ds):
    """
    Plot a single‐row heatmap of the EV charge schedule for day `ds`.
    Saves figure to file instead of displaying it.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import os
    import numpy as np

    # Debug print to check the ev_charge values
    print(f"DEBUG: EV charge values for {building_id} on {ds}: {ev_charge}")
    
    # Convert to array if not already
    if isinstance(ev_charge, list):
        ev_charge = np.array(ev_charge)
    
    # Create dataframe for heatmap
    df = pd.DataFrame(
        [ev_charge],
        index=[f"{building_id}_ev"],
        columns=list(range(24))
    )

    fig = plt.figure(figsize=(12,2))
    ax = sns.heatmap(df,
                     cmap="Blues",
                     annot=True,
                     fmt=".2f",
                     cbar_kws={"label":"kWh charged"},
                     xticklabels=range(24),
                     yticklabels=df.index)
    ax.set_xlabel("Hour of Day")
    ax.set_title(f"EV Charge Schedule — {building_id} [{ds}]")
    plt.tight_layout()
    
    # Create a plots directory if it doesn't exist
    os.makedirs("plots", exist_ok=True)
    
    # Save figure instead of displaying
    filename = f"plots/{building_id}_ev_schedule_{ds}.png"
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    print(f"Saved EV schedule plot to {filename}")
    plt.close(fig)


def plot_multi_day_battery_line(scheduling_results, building_id):
    """
    Like plot_unified_battery_usage, but across multiple live days end-to-end.
    
    Args:
      scheduling_results: list of (date, devices) tuples from your live loop
      building_id: string for titles/labels
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # Flatten SOC, charge, discharge, and price across days
    soc_all = []
    chg_all = []
    dis_all = []
    price_all = []
    timestamps = []
    
    for day, devices in scheduling_results:
        dev = devices[0]
        # day arrays (length 24)
        soc_day = np.asarray(dev.battery_soc_day)
        chg_day = np.asarray(dev.battery_charge_day)
        dis_day = np.asarray(dev.battery_discharge_day)
        # price by hour 0–23
        df = dev.data.copy()
        price_by_hour = df.groupby('hour')['price_per_kwh'] \
                         .mean().reindex(range(24), fill_value=np.nan).values
        
        soc_all.append(soc_day)
        chg_all.append(chg_day)
        dis_all.append(dis_day)
        price_all.append(price_by_hour)
        
        # build a continuous time index
        # hours since start: day_index*24 + np.arange(24)
        base = len(soc_all) - 1
        stamps = base*24 + np.arange(24)
        timestamps.append(stamps)
    
    # concatenate
    soc_all = np.concatenate(soc_all)
    chg_all = np.concatenate(chg_all)
    dis_all = np.concatenate(dis_all)
    price_all = np.concatenate(price_all)
    timestamps = np.concatenate(timestamps)
    
    # retrieve battery limits & efficiencies from first device
    first = scheduling_results[0][1][0]
    ba = getattr(first, 'battery_agent', None)
    if ba is not None:
        soc_min, soc_max = ba.soc_min, ba.soc_max
        # if your agent tracks efficiency:
        charge_eff = getattr(ba, 'charge_efficiency', 1.0)
        discharge_eff = getattr(ba, 'discharge_efficiency', 1.0)
    else:
        soc_min, soc_max = soc_all.min(), soc_all.max()
        charge_eff = discharge_eff = 1.0
    
    # plot
    fig, (ax_soc, ax_pow) = plt.subplots(2,1, figsize=(14,6), sharex=True,
                                         gridspec_kw={'height_ratios':[2,1]})
    
    # — SoC panel —
    ax_soc.plot(timestamps, soc_all, color='orange', lw=2, label='Battery SoC')
    ax_soc.axhline(soc_min, color='grey', ls=':', label=f"Min SoC ({soc_min:.1f}kWh)")
    ax_soc.axhline(soc_max, color='grey', ls='-.', label=f"Max SoC ({soc_max:.1f}kWh)")
    ax_soc.set_ylabel("State of Charge (kWh)")
    ax_soc.set_title(f"Battery SoC over {len(scheduling_results)} live days — {building_id}")
    ax_soc.grid(alpha=0.3)
    ax_soc.legend(loc='upper left')
    
    # — Power panel —
    ax_pow.plot(timestamps, chg_all * charge_eff, color='darkgreen', lw=2, label='Charge (kW)')
    ax_pow.plot(timestamps, -dis_all / discharge_eff, color='crimson', lw=2, ls='--', label='Discharge (kW)')
    ax_pow.set_ylabel("Power (kW)")
    ax_pow.grid(alpha=0.3)
    
    # twin axis for price
    ax_price = ax_pow.twinx()
    ax_price.plot(timestamps, price_all, color='black', alpha=0.4, lw=1, label='Price (€/kWh)')
    ax_price.set_ylabel("Price (€/kWh)", color='black')
    
    # combine legends
    h1, l1 = ax_pow.get_legend_handles_labels()
    h2, l2 = ax_price.get_legend_handles_labels()
    ax_pow.legend(h1+h2, l1+l2, loc='upper right')
    
    # x-axis ticks every 24h with date labels
    day_ticks = [i*24 for i in range(len(scheduling_results)+1)]
    day_labels = [pd.to_datetime(day).strftime("%m-%d") 
                  for day, _ in scheduling_results] + ['']
    ax_pow.set_xticks(day_ticks)
    ax_pow.set_xticklabels(day_labels)
    ax_pow.set_xlabel("Day (each tick = start of that day)")
    
    plt.tight_layout()
    plt.show()
