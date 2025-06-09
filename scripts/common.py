#!/usr/bin/env python
"""
Common utilities for the EMS system.
Contains reusable helpers from utils.helper and DuckDB connection function.
"""
import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

def get_con(building_id: str = None):
    """Get a connection to the DuckDB database with enhanced fallback."""
    this_file = Path(__file__).resolve()
    project_root = this_file.parent.parent
    db_path = project_root / "ems_data.duckdb"
    
    if not db_path.exists():
        raise FileNotFoundError(f"Database file not found: {db_path}")
    
    try:
        # First try: read-only mode (safer)
        con = duckdb.connect(str(db_path), read_only=True)
        
    except duckdb.IOException as lock_err:
        print("⚠️  DB locked – loading data into in-memory DuckDB:", lock_err)
        
        # Create in-memory database and load data
        con = duckdb.connect(database=":memory:")
        
        # Try to copy data from the locked file using read-only mode
        try:
            # Load the parquet files directly if available
            if building_id:
                parquet_candidates = [
                    project_root / "data" / f"{building_id}_processed_data.parquet",
                    project_root / "notebooks" / "data" / f"{building_id}_processed_data.parquet",
                ]
                
                for parquet_path in parquet_candidates:
                    if parquet_path.exists():
                        con.execute(f"""
                        CREATE TABLE {building_id}_processed_data AS 
                        SELECT * FROM read_parquet('{str(parquet_path).replace(os.sep, '/')}')
                        """)
                        print(f"✓ Loaded data from {parquet_path}")
                        break
                else:
                    raise FileNotFoundError("No parquet backup found")
                    
        except Exception as data_err:
            print(f"⚠️  Could not load data into memory: {data_err}")
            raise ConnectionError("Database locked and no data backup available")
    
    if building_id and not hasattr(con, '_view_registered'):
        try:
            # Check if view exists, create if needed
            result = con.execute(f"SELECT COUNT(*) FROM {building_id}_processed_data").fetchone()
            con._view_registered = True
        except:
            register_processed_view(con, building_id)
            con._view_registered = True
    
    return (con, f"{building_id}_processed_data") if building_id else con
def register_processed_view(con, building_id):
    """
    Register a DuckDB view pointing at the building's processed_data parquet.
    """
    view_name = f"{building_id}_processed_data"
    # locate project root relative to this script
    this_file = Path(__file__).resolve()
    project_root = this_file.parent.parent

    # look for our parquet under either <root>/data or <root>/notebooks/data
    candidates = [
        project_root / "data" / f"{view_name}.parquet",
        project_root / "notebooks" / "data" / f"{view_name}.parquet",
    ]
    for data_path in candidates:
        if data_path.exists():
            break
    else:
        # none found
        raise FileNotFoundError(
            "Tried:\n  " +
            "\n  ".join(str(p) for p in candidates) +
            "\nbut no parquet file was found."
        )
    
    # register parquet as a view in whichever DB we have (on-disk or in-memory)
    con.execute(
        f"CREATE OR REPLACE VIEW {view_name} AS "
        f"SELECT * FROM read_parquet('{str(data_path).replace(os.sep, '/')}')"
    )
    return view_name

def get_view_con(building_id: str):
    """
    Convenience: open DuckDB (on-disk or in-memory if locked) and
    register the processed_data parquet for `building_id` as a view.
    Returns: (con, view_name)
    """
    # 1) connect (will auto-fall-back to memory if locked)
    con = get_con()

    # 2) register the parquet as a view
    view_name = register_processed_view(con, building_id)

    return con, view_name


def preprocess(df):
    """Add time-based features to dataframe."""
    # Copy logic from utils.helper
    df = df.copy()

    # Add building ID based on column names if not present
    if 'building_id' not in df.columns:
        building_columns = [col for col in df.columns if col.startswith('DE_KN_')]
        if building_columns:
            building_id = building_columns[0].split('_')[0:3]
            df['building_id'] = '_'.join(building_id)

    # Process timestamp
    timestamp_col = None
    if 'utc_timestamp' in df.columns:
        timestamp_col = 'utc_timestamp'
    elif 'timestamp' in df.columns:
        timestamp_col = 'timestamp'

    if timestamp_col is not None:
        # Add day column
        if 'day' not in df.columns:
            df['day'] = pd.to_datetime(df[timestamp_col]).dt.date

        # Add hour column
        if 'hour' not in df.columns:
            df['hour'] = pd.to_datetime(df[timestamp_col]).dt.hour

        # Add weekday column
        if 'weekday' not in df.columns:
            df['weekday'] = pd.to_datetime(df[timestamp_col]).dt.weekday
    elif 'year' in df.columns:
        # Fallback to index-based calculation
        if 'day' not in df.columns:
            df['day'] = df.index.date if hasattr(df.index, 'date') else df.index
        if 'hour' not in df.columns:
            df['hour'] = df.index.hour if hasattr(df.index, 'hour') else df.index % 24
        if 'weekday' not in df.columns and 'day' in df.columns:
            df['weekday'] = pd.to_datetime(df['day']).dt.weekday

    # Rename columns if needed
    if 'price_per_kwh' in df.columns and 'price' not in df.columns:
        df['price'] = df['price_per_kwh']

    return df

def has_pv(df):
    """Check if dataframe has PV generation data."""
    pv_columns = [col for col in df.columns if 'pv' in col.lower()]
    return len(pv_columns) > 0

def get_pv_columns(df):
    """Get all PV-related columns in the dataframe."""
    return [col for col in df.columns if 'pv' in col.lower() and 'grid' not in col.lower()]

def extract_device_specs(df, building_id):
    """Extract device specs from dataframe columns based on the building ID."""
    # Default device categories and specifications
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
        "ev": {
            "category": "Highly Flexible", "power_rating": 7.4,
            "allowed_hours": list(range(0, 24)),  # EV can charge any time
            "flex_model": "partial_usage",  # EV charging is highly flexible
            "must_be_full_by_hour": 7,  # Must be charged by 7 AM
            "efficiency_charge": 0.92,
            "soc_min_pct": 0.1,
            "soc_max_pct": 0.9,
            "capacity_kwh": 60.0
        }
    }

    # Find device columns in the dataframe
    building_prefix = f"{building_id}_"
    device_columns = [col for col in df.columns if building_prefix in col
                     and col != f"{building_prefix}grid_import"
                     and col != f"{building_prefix}grid_export"
                     and "pv" not in col.lower()]

    # Extract device names and create specs
    specs = {}
    for col in device_columns:
        # Extract device type from column name
        device_type = col.replace(building_prefix, "")

        # Create a simplified device ID
        device_id = col

        # Determine base device type to get default specs
        base_type = None
        for known_type in device_specs:
            if known_type in device_type:
                base_type = known_type
                break

        # If we found a matching base type, use its specs
        if base_type:
            specs[device_id] = device_specs[base_type].copy()
        else:
            # Default to fixed model for unknown devices
            specs[device_id] = {
                "category": "Non-Flexible",
                "power_rating": 1.0,
                "allowed_hours": list(range(0, 24)),
                "flex_model": "fixed"
            }

    return specs

def close_figures():
    """Close all matplotlib figures."""
    plt.tight_layout()
    plt.close('all')

def plot_schedule_heatmap(devices, building_id, mode, day=None):
    """Create a heatmap of device schedules."""
    import numpy as np

    # Get JADS color palette if available
    try:
        # Define a JADS-like color palette for consistency
        JADS_COLORS = {
            "Original": "#4C72B0",           # Blue
            "Decentralized": "#55A868",      # Green
            "Decentralized_NoBatt": "#C44E52", # Red
            "Decentralized_WithBatt": "#8172B3", # Purple
            "Centralized": "#CCB974",        # Yellow
            "Centralized_Phases": "#64B5CD", # Light Blue
            "Building_Original": "#4C72B0",  # Blue
            "Building_NoBatt": "#C44E52",    # Red
            "Building_WithBatt": "#8172B3",  # Purple
            "Battery": "#0173B2",            # Dark Blue
            "EV": "#029E73",                 # Green
            "PV": "#D55E00",                 # Orange
            "Grid": "#CC78BC",               # Pink
            "Price": "#000000"               # Black
        }
        color_price = JADS_COLORS["Price"]
        cmap_devices = plt.cm.get_cmap('YlOrRd')  # Use YlOrRd for heatmap
        print(f"  Using JADS color palette")
    except Exception:
        # Fallback colors
        color_price = 'b'
        cmap_devices = plt.cm.get_cmap('YlOrRd')

    mat, names = [], []
    price_s = None
    prob_data = {}  # Store probability distributions if available

    # Extract schedules from devices
    for dev in devices:
        # Use appropriate schedule based on optimization mode
        if mode == "decentralised" and hasattr(dev, "optimized_consumption"):
            schedule = np.array(dev.optimized_consumption[:24])
        elif mode == "centralised_phases" and hasattr(dev, "nextday_optimized_schedule"):
            schedule = np.array(dev.nextday_optimized_schedule[:24])
        elif mode == "centralised" and hasattr(dev, "centralized_optimized_schedule"):
            schedule = np.array(dev.centralized_optimized_schedule[:24])
        else:
            # No fallback allowed - raise error if schedule not found for mode
            raise ValueError(f"Device {dev.device_name} does not have required schedule for mode '{mode}'. "
                           f"Expected attribute not found. Agent optimization must be run correctly.")

        mat.append(schedule)
        names.append(dev.device_name)

        # Get price data if available
        if price_s is None and hasattr(dev, "data") and "price_per_kwh" in dev.data.columns:
            price_s = dev.data.groupby("hour")["price_per_kwh"].mean().reindex(range(24))

        # Get probability distribution if available
        if hasattr(dev, "hour_probability") and dev.hour_probability:
            prob_data[dev.device_name] = dev.hour_probability

    # Convert to numpy array for processing
    mat = np.array(mat)

    # Use default price if none available
    if price_s is None:
        price_s = pd.Series([0.05]*24, index=range(24))

    # Row-wise normalization for better visualization
    norm = np.zeros_like(mat)
    for i in range(len(mat)):
        row_min, row_max = mat[i].min(), mat[i].max()
        if row_max > row_min:
            norm[i] = (mat[i] - row_min) / (row_max - row_min)

    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6 + len(devices)*0.3),
                                  gridspec_kw={'height_ratios': [1, 4], 'hspace': 0.05})

    # Plot price in top panel with proper color
    if color_price.startswith('#'):
        ax1.plot(range(24), price_s.values, 'o-', color=color_price, lw=2)
    else:
        ax1.plot(range(24), price_s.values, f'{color_price}-o', lw=2)
    ax1.set_ylabel("Price (€/kWh)")
    ax1.set_xlim(-0.5, 23.5)
    ax1.set_xticks([])
    ax1.grid(alpha=0.3)

    # Apply price shading for low/high price periods
    low_threshold = price_s.quantile(0.25)
    high_threshold = price_s.quantile(0.75)

    for hr in range(len(price_s)):
        if price_s[hr] <= low_threshold:
            ax1.axvspan(hr-0.5, hr+0.5, color='lightblue', alpha=0.2)
        elif price_s[hr] >= high_threshold:
            ax1.axvspan(hr-0.5, hr+0.5, color='lightcoral', alpha=0.2)

    # Plot device schedules as heatmap
    im = ax2.imshow(norm, aspect='auto', cmap=cmap_devices, vmin=0, vmax=1,
                   extent=(-0.5, 23.5, len(norm), 0))
    ax2.set_yticks(np.arange(len(norm))+0.5)
    ax2.set_yticklabels(names)
    ax2.set_xlabel("Hour")
    ax2.set_xticks(range(24))

    # Overlay probability distributions if available
    if prob_data:
        for i, dev_name in enumerate(names):
            if dev_name in prob_data:
                probs = prob_data[dev_name]
                for hour in range(24):
                    prob = probs.get(hour, 0.0)
                    if prob > 0.1:  # Only show significant probabilities
                        # Scale probability for visualization
                        prob_scaled = min(1.0, prob * 2)
                        rect = plt.Rectangle((hour-0.5, i-0.4), 1, 0.8,
                                           fill=True, alpha=0.3*prob_scaled,
                                           color='green', transform=ax2.transData)
                        ax2.add_patch(rect)

                        # Add text showing probability
                        if prob > 0.3:  # Only label high probabilities
                            ax2.text(hour, i+0.2, f"{prob:.2f}",
                                    ha='center', va='center', fontsize=7,
                                    color='darkgreen')

    # Add colorbar
    cbar = fig.colorbar(im, ax=[ax1, ax2], pad=0.01)
    cbar.set_label('Normalized Energy Consumption')

    # Add legend for probability if used
    if prob_data:
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.5, label='Usage Probability')
        ]
        ax2.legend(handles=legend_elements, loc='upper right')

    # Set title
    day_str = f" - {day}" if day else ""
    plt.suptitle(f"{building_id} - {mode.capitalize()} Optimization{day_str}", fontsize=14)

    # Save figure
    day_str = f"_{day}" if day else ""
    filename = f"results/figures/{building_id}{day_str}_{mode}_heat.png"
    plt.savefig(filename, bbox_inches='tight')
    close_figures()

    return filename

def plot_battery_soc(battery_agent, building_id, battery_mode):
    """Plot battery state of charge trajectory."""
    if not hasattr(battery_agent, 'hourly_soc') or len(battery_agent.hourly_soc) == 0:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    hours = range(len(battery_agent.hourly_soc))

    # Plot SOC
    ax.plot(hours, battery_agent.hourly_soc, 'b-', label='State of Charge (kWh)', linewidth=2)

    # Add charge/discharge if available
    if hasattr(battery_agent, 'hourly_charge') and len(battery_agent.hourly_charge) > 0:
        ax2 = ax.twinx()
        ax2.plot(hours, battery_agent.hourly_charge, 'g-', label='Charge (kW)', alpha=0.7)
        ax2.plot(hours, [-d for d in battery_agent.hourly_discharge], 'r-', label='Discharge (kW)', alpha=0.7)
        ax2.set_ylabel('Power (kW)')
        ax2.legend(loc='upper right')

    ax.set_xlabel('Hour')
    ax.set_ylabel('State of Charge (kWh)')
    ax.set_xlim(0, len(hours)-1)
    ax.set_xticks(range(0, len(hours), 4))
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')

    plt.title(f"{building_id} - Battery Operation - {battery_mode}")

    # Save figure
    filename = f"results/figures/{building_id}_{battery_mode}_soc.png"
    plt.savefig(filename, bbox_inches='tight')
    close_figures()

    return filename

def plot_cost_comparison(kpi_df, building_id, battery_mode):
    """Plot cost comparison between original and optimized schedules for different modes."""
    if kpi_df.empty:
        return None

    modes = kpi_df['mode'].unique()

    # Calculate average costs by mode
    avg_costs = {}
    for mode in modes:
        mode_data = kpi_df[kpi_df['mode'] == mode]
        avg_costs[mode] = {
            'original': mode_data['total_cost_original'].mean(),
            'optimized': mode_data['total_cost_optimized'].mean()
        }

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Prepare data for bar plot
    x = np.arange(len(modes))
    width = 0.35

    # Plot bars
    orig_bars = ax.bar(x - width/2, [avg_costs[m]['original'] for m in modes],
                      width, label='Original', color='steelblue')
    opt_bars = ax.bar(x + width/2, [avg_costs[m]['optimized'] for m in modes],
                     width, label='Optimized', color='indianred')

    # Add percentages on top of bars
    for i, mode in enumerate(modes):
        orig = avg_costs[mode]['original']
        opt = avg_costs[mode]['optimized']
        savings_pct = (orig - opt) / orig * 100 if orig > 0 else 0

        ax.text(i + width/2, opt + 0.1, f"{savings_pct:.1f}%",
               ha='center', va='bottom', fontsize=9, color='green')

    # Customize plot
    ax.set_ylabel('Average Cost (€)')
    ax.set_title(f"{building_id} - Cost Comparison - Battery {battery_mode}")
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in modes])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Save figure
    filename = f"results/figures/{building_id}_{battery_mode}_cost_bar.png"
    plt.savefig(filename, bbox_inches='tight')
    close_figures()

    return filename

def shade_price_hours(ax, price_series):
    """Shade low and high price hours in the plot."""
    # Define low and high price thresholds (lowest 25% and highest 25%)
    low_threshold = price_series.quantile(0.25)
    high_threshold = price_series.quantile(0.75)

    # Shade low price hours with light blue
    for hr in range(len(price_series)):
        if price_series[hr] <= low_threshold:
            ax.axvspan(hr, hr+1, color='lightblue', alpha=0.1)

    # Shade high price hours with light red
    for hr in range(len(price_series)):
        if price_series[hr] >= high_threshold:
            ax.axvspan(hr, hr+1, color='lightcoral', alpha=0.1)

def plot_decentralized_vs_centralized(devices_decentralized, devices_centralized, building_id, ev_agent=None):
    """
    Plot a comparison of decentralized vs centralized optimization schedules.

    Args:
        devices_decentralized: List of devices with decentralized optimization results
        devices_centralized: List of devices with centralized optimization results
        building_id: Building ID
        ev_agent: Optional EV agent for EV-specific visualization

    Returns:
        str: Path to the saved figure
    """
    import matplotlib.patches as mpatches

    # Get device load profiles for both optimizations
    total_dec = np.zeros(24)
    total_cent = np.zeros(24)

    # Sum up all device loads
    for dev in devices_decentralized:
        if hasattr(dev, 'optimized_consumption'):
            total_dec += np.array(dev.optimized_consumption[:24])

    for dev in devices_centralized:
        if hasattr(dev, 'centralized_optimized_schedule'):
            total_cent += np.array(dev.centralized_optimized_schedule[:24])

    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [1, 4]})

    # Get price data if available
    price_data = None
    for dev in devices_decentralized:
        if hasattr(dev, 'data') and 'price_per_kwh' in dev.data.columns:
            price_data = dev.data.groupby('hour')['price_per_kwh'].mean().reindex(range(24), fill_value=dev.data['price_per_kwh'].mean())
            break

    if price_data is None:
        price_data = pd.Series([0.25] * 24, index=range(24))

    # Plot price in top panel
    ax1.plot(range(24), price_data.values, 'b-o', lw=2)
    ax1.set_ylabel("Price (€/kWh)")
    ax1.set_xlim(-0.5, 23.5)
    ax1.set_xticks([])
    ax1.grid(alpha=0.3)

    # Shade high/low price hours
    shade_price_hours(ax2, price_data)

    # Plot device loads
    hours = range(24)
    ax2.plot(hours, total_dec, 'g-', label='Decentralized', linewidth=2)
    ax2.plot(hours, total_cent, 'r--', label='Centralized', linewidth=2)

    # Plot EV charging schedule if available
    if ev_agent and hasattr(ev_agent, 'hourly_charge'):
        # Decentralized EV charging
        if hasattr(ev_agent, 'hourly_charge') and hasattr(devices_decentralized[0], 'optimized_consumption'):
            ev_dec = np.array(ev_agent.hourly_charge[:24])
            ax2.bar(hours, ev_dec, color='green', alpha=0.3, label='EV Charging (Dec)')

        # Add EV state of charge as a secondary axis
        if hasattr(ev_agent, 'hourly_soc'):
            ax3 = ax2.twinx()
            ax3.plot(hours, ev_agent.hourly_soc[:24], 'b-', label='EV SOC', linewidth=1.5)
            ax3.set_ylabel('EV State of Charge (kWh)')

            # Create a custom legend
            handles, labels = ax2.get_legend_handles_labels()
            soc_line = mpatches.Patch(color='blue', label='EV SOC')
            handles.append(soc_line)
            ax2.legend(handles=handles, loc='upper left')
        else:
            ax2.legend(loc='upper left')
    else:
        ax2.legend(loc='upper left')

    # Customize plot
    ax2.set_xlabel("Hour")
    ax2.set_ylabel("Load (kWh)")
    ax2.set_xlim(-0.5, 23.5)
    ax2.set_xticks(range(0, 24, 1))
    ax2.grid(True, alpha=0.3)

    # Set title
    plt.suptitle(f"{building_id} - Decentralized vs. Centralized Optimization", fontsize=14)

    # Save figure
    filename = f"results/figures/{building_id}_dec_vs_cent_comparison.png"
    plt.savefig(filename, bbox_inches='tight')
    close_figures()

    return filename

def plot_centralized_phases_heatmap(devices, building_id, ev_agent=None):
    """
    Plot a heatmap of device schedules for centralized_phases optimization.

    Args:
        devices: List of devices with centralized_phases optimization results
        building_id: Building ID
        ev_agent: Optional EV agent for EV-specific visualization

    Returns:
        str: Path to the saved figure
    """
    # Extract schedules and device names
    mat = []
    names = []
    price_s = None

    # Extract schedules from devices
    for dev in devices:
        if hasattr(dev, "nextday_optimized_schedule"):
            schedule = np.array(dev.nextday_optimized_schedule[:24])
            mat.append(schedule)
            names.append(dev.device_name)

            # Get price data if available
            if price_s is None and hasattr(dev, "data") and "price_per_kwh" in dev.data.columns:
                price_s = dev.data.groupby("hour")["price_per_kwh"].mean().reindex(range(24))

    # Add EV schedule if available
    if ev_agent and hasattr(ev_agent, 'hourly_charge'):
        ev_schedule = np.array(ev_agent.hourly_charge[:24])
        mat.append(ev_schedule)
        names.append(ev_agent.device_name + " (charging)")

    # Convert to numpy array for processing
    mat = np.array(mat)

    # Use default price if none available
    if price_s is None:
        price_s = pd.Series([0.25]*24, index=range(24))

    # Row-wise normalization for better visualization
    norm = np.zeros_like(mat)
    for i in range(len(mat)):
        row_min, row_max = mat[i].min(), mat[i].max()
        if row_max > row_min:
            norm[i] = (mat[i] - row_min) / (row_max - row_min)

    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6 + len(devices)*0.3),
                                   gridspec_kw={'height_ratios': [1, 4], 'hspace': 0.05})

    # Plot price in top panel
    ax1.plot(range(24), price_s.values, 'b-o', lw=2)
    ax1.set_ylabel("Price (€/kWh)")
    ax1.set_xlim(-0.5, 23.5)
    ax1.set_xticks([])
    ax1.grid(alpha=0.3)

    # Add probability distributions if available
    probability_markers = []
    for i, dev in enumerate(devices):
        if hasattr(dev, 'hour_probability') and dev.hour_probability:
            prob_data = [dev.hour_probability.get(h, 0.0) for h in range(24)]

            # Normalize probabilities
            prob_max = max(prob_data)
            if prob_max > 0:
                scaled_probs = [p/prob_max for p in prob_data]

                # Add semi-transparent markers for probability
                for h, prob in enumerate(scaled_probs):
                    if prob > 0.2:  # Only mark significant probabilities
                        rect = plt.Rectangle((h, i), 1, 1, fill=True, alpha=0.3*prob,
                                            color='green', transform=ax2.transData)
                        probability_markers.append(rect)
                        ax2.add_patch(rect)

    # Plot device schedules as heatmap
    im = ax2.imshow(norm, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1,
                   extent=(-0.5, 23.5, len(norm), 0))
    ax2.set_yticks(np.arange(len(norm))+0.5)
    ax2.set_yticklabels(names)
    ax2.set_xlabel("Hour")
    ax2.set_xticks(range(24))

    # Add EV SOC line if available
    if ev_agent and hasattr(ev_agent, 'hourly_soc'):
        ax3 = ax2.twinx()

        # Normalize SOC for better visualization
        soc_min = min(ev_agent.hourly_soc[:24])
        soc_max = max(ev_agent.hourly_soc[:24])
        if soc_max > soc_min:
            normalized_soc = [(soc - soc_min) / (soc_max - soc_min) * len(norm) * 0.8 for soc in ev_agent.hourly_soc[:24]]
            ax3.plot(range(24), normalized_soc, 'b-', label='EV SOC', linewidth=2)
            ax3.set_ylabel('EV SOC (normalized)')
            ax3.set_ylim(0, len(norm))

            # Add SOC annotations
            for hour, soc in enumerate(ev_agent.hourly_soc[:24]):
                if hour % 3 == 0:  # Add labels every 3 hours to avoid clutter
                    ax3.text(hour, normalized_soc[hour] + 0.2, f"{soc:.1f} kWh",
                             ha='center', va='bottom', fontsize=8, color='blue')

    # Add colorbar
    cbar = fig.colorbar(im, ax=[ax1, ax2], pad=0.01)
    cbar.set_label('Normalized Energy Consumption')

    # Add legend for probability markers
    if probability_markers:
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.5, label='Usage Probability')
        ]
        ax2.legend(handles=legend_elements, loc='upper right')

    # Set title
    plt.suptitle(f"{building_id} - Centralized Phases Optimization Heatmap", fontsize=14)

    # Save figure
    filename = f"results/figures/{building_id}_centralized_phases_heatmap.png"
    plt.savefig(filename, bbox_inches='tight')
    close_figures()

    return filename

def assert_energy_balance(day_df, battery=None, ev=None, pv_gen=None, tol=0.1):
    """
    Ensure energy balance: grid_import + pv_gen + battery_discharge + ev_discharge ≥ 
                          device_load + battery_charge + ev_charge.
    
    Args:
        day_df (pd.DataFrame): DataFrame with hourly energy data for a day
        battery (BatteryAgent, optional): Battery agent with hourly charge/discharge data
        ev (EVAgent, optional): EV agent with hourly charge data
        pv_gen (np.array, optional): Hourly PV generation array if not in day_df
        tol (float, optional): Tolerance factor (as percentage of total load)
    
    Returns:
        tuple: (passed, error_msg, imbalance_hours)
    """
    import numpy as np
    
    # Extract device load columns (exclude grid, pv, ev, and non-numeric columns)
    building_id = day_df['building_id'].iloc[0] if 'building_id' in day_df.columns else None
    
    # Find all device columns, excluding EV if we're handling it separately
    device_cols = []
    ev_col = None
    
    for col in day_df.columns:
        # Check if this is a device column (belongs to this building, not grid or pv)
        if building_id and building_id in col and 'grid' not in col.lower() and 'pv' not in col.lower():
            if day_df[col].dtype in [np.float64, np.int64, float, int]:
                # Check if this is an EV column
                if 'ev' in col.lower():
                    ev_col = col
                else:
                    device_cols.append(col)
    
    # Get grid import if available
    grid_import = None
    for col in day_df.columns:
        if 'grid_import' in col.lower():
            grid_import = day_df[col].values
            break
    
    if grid_import is None:
        return False, "Grid import data not found", []
    
    # Get PV generation (convert negative values to positive since PV is generation)
    pv_columns = get_pv_columns(day_df)
    if pv_columns:
        pv_generation = -day_df[pv_columns].sum(axis=1).values  # Convert negative to positive
    elif pv_gen is not None:
        pv_generation = pv_gen
    else:
        pv_generation = np.zeros(len(day_df))
    
    # Aggregate all device loads (excluding EV if we have an EV agent)
    total_device_load = np.zeros(len(day_df))
    for col in device_cols:
        total_device_load += day_df[col].values
    
    # Add the EV load if it's in the data and we're not handling it separately with a proper EVAgent
    if ev_col and (ev is None or not hasattr(ev, 'device_name') or ev.device_name != ev_col):
        total_device_load += day_df[ev_col].values
    
    # Get battery charge/discharge
    battery_charge = np.zeros(len(day_df))
    battery_discharge = np.zeros(len(day_df))
    
    if battery and hasattr(battery, 'hourly_charge') and len(battery.hourly_charge) >= len(day_df):
        battery_charge = np.array(battery.hourly_charge[:len(day_df)])
        battery_discharge = np.array(battery.hourly_discharge[:len(day_df)])
    
    # Add EV charge/discharge if available
    ev_charge = np.zeros(len(day_df))
    ev_discharge = np.zeros(len(day_df))
    
    if ev and hasattr(ev, 'hourly_charge') and len(ev.hourly_charge) >= len(day_df):
        # Only count EV charge if it's the EV for this column
        if hasattr(ev, 'device_name') and ev.device_name == ev_col:
            ev_charge = np.array(ev.hourly_charge[:len(day_df)])
            # EV discharge might be zero in all cases since EVs typically don't discharge to grid
            if hasattr(ev, 'hourly_discharge') and len(ev.hourly_discharge) >= len(day_df):
                ev_discharge = np.array(ev.hourly_discharge[:len(day_df)])
    
    # Calculate energy balance
    # Supply = grid_import + pv_generation + battery_discharge + ev_discharge
    # Demand = total_device_load + battery_charge + ev_charge
    supply = grid_import + pv_generation + battery_discharge + ev_discharge
    demand = total_device_load + battery_charge + ev_charge
    
    # Check balance with tolerance
    imbalance = supply - demand
    tolerance = tol * max(1.0, total_device_load.sum())  # tolerance as percentage of total load, minimum 1.0
    
    # Identify hours with imbalance exceeding tolerance
    imbalance_hours = []
    for hour, value in enumerate(imbalance):
        if abs(value) > tolerance / len(imbalance):
            imbalance_hours.append((hour, value))
    
    if imbalance_hours:
        error_msg = f"Energy balance violated in {len(imbalance_hours)} hours: " + \
                   f"max imbalance={max([abs(v) for _, v in imbalance_hours]):.2f} kWh"
        return False, error_msg, imbalance_hours
    
    return True, "Energy balance satisfied", []

def assert_battery_limits(battery, tol=0.05):
    """
    Ensure battery SoC stays within limits: 0 ≤ SoC ≤ soc_max, |ΔSoC_hour| ≤ max_charge_rate.
    
    Args:
        battery (BatteryAgent): Battery agent with hourly SoC data
        tol (float, optional): Tolerance factor
    
    Returns:
        tuple: (passed, error_msg, violation_details)
    """
    # Skip check if no battery
    if not battery or not hasattr(battery, 'hourly_soc') or len(battery.hourly_soc) == 0:
        return True, "No battery to check", []
    
    violations = []
    
    # Check SoC bounds
    for hour, soc in enumerate(battery.hourly_soc):
        # Check if SoC is below min or above max (with small tolerance)
        if soc < (battery.soc_min - tol):
            violations.append((hour, f"SoC below min: {soc:.2f} < {battery.soc_min:.2f}"))
        elif soc > (battery.soc_max + tol):
            violations.append((hour, f"SoC above max: {soc:.2f} > {battery.soc_max:.2f}"))
    
    # Check hourly SoC changes against rate limits
    for hour in range(1, len(battery.hourly_soc)):
        delta_soc = battery.hourly_soc[hour] - battery.hourly_soc[hour-1]
        
        # Check charging (positive delta)
        if delta_soc > 0 and delta_soc > (battery.max_charge_rate + tol):
            violations.append((hour, f"Charging rate exceeded: {delta_soc:.2f} > {battery.max_charge_rate:.2f}"))
        
        # Check discharging (negative delta)
        elif delta_soc < 0 and abs(delta_soc) > (battery.max_discharge_rate + tol):
            violations.append((hour, f"Discharging rate exceeded: {abs(delta_soc):.2f} > {battery.max_discharge_rate:.2f}"))
    
    if violations:
        error_msg = f"Battery limits violated in {len(violations)} instances"
        return False, error_msg, violations
    
    return True, "Battery limits satisfied", []

def assert_savings(kpi_row, min_savings_pct=0.0):
    """
    Ensure centralized_phases delivers non-negative savings vs decentralized.
    
    Args:
        kpi_row (dict): Row of KPI data with optimization results
        min_savings_pct (float, optional): Minimum expected savings percentage
    
    Returns:
        tuple: (passed, error_msg, details)
    """
    if kpi_row['mode'] != 'centralised_phases':
        return True, "Not a centralised_phases row", {}
    
    original_cost = kpi_row['total_cost_original']
    optimized_cost = kpi_row['total_cost_optimized']
    savings = original_cost - optimized_cost
    savings_pct = kpi_row['savings_pct']
    
    # Ensure savings are non-negative
    if savings_pct < min_savings_pct:
        error_msg = f"Insufficient savings: {savings_pct:.2f}% (min required: {min_savings_pct:.2f}%)"
        details = {
            'original_cost': original_cost,
            'optimized_cost': optimized_cost,
            'savings': savings,
            'savings_pct': savings_pct,
            'required_pct': min_savings_pct
        }
        return False, error_msg, details
    
    return True, f"Savings of {savings_pct:.2f}% achieved (min required: {min_savings_pct:.2f}%)", {
        'savings_pct': savings_pct
    }

def generate_markdown_table(data, columns, title, caption=None):
    """
    Generate a Markdown table from data.
    
    Args:
        data (list of dict): Data to include in the table
        columns (list of tuples): List of (column_key, column_name, format_str) tuples
        title (str): Table title
        caption (str, optional): Table caption
    
    Returns:
        str: Markdown table
    """
    # Start with the title
    md = f"## {title}\n\n"
    
    # Add header row
    md += "| " + " | ".join([col[1] for col in columns]) + " |\n"
    
    # Add separator row
    md += "| " + " | ".join(["---" for _ in columns]) + " |\n"
    
    # Add data rows
    for row in data:
        row_values = []
        for key, _, fmt in columns:
            if key in row:
                if fmt:
                    row_values.append(fmt.format(row[key]))
                else:
                    row_values.append(str(row[key]))
            else:
                row_values.append("")
        md += "| " + " | ".join(row_values) + " |\n"
    
    # Add caption if provided
    if caption:
        md += f"\n*{caption}*\n\n"
    else:
        md += "\n"
    
    return md

def generate_result_health_report(kpi_df, building_id, mode, battery_mode, violations=None):
    """
    Generate a health report for the optimization results.
    
    Args:
        kpi_df (pd.DataFrame): KPI data for optimization runs
        building_id (str): Building ID
        mode (str): Optimization mode
        battery_mode (str): Battery mode (on/off)
        violations (dict, optional): Violations dict from assertion functions
    
    Returns:
        str: Markdown report
    """
    # Filter data for the specified mode
    mode_data = kpi_df[kpi_df['mode'] == mode]
    
    if mode_data.empty:
        return f"## Result Health Report\n\nNo data available for {building_id}, {mode}, battery {battery_mode}\n"
    
    report = f"## Result Health Report - {building_id} - {mode.capitalize()} - Battery {battery_mode}\n\n"
    
    # Basic statistics
    report += "### Basic Statistics\n\n"
    report += f"- **Number of days**: {len(mode_data)}\n"
    report += f"- **Average original cost**: {mode_data['total_cost_original'].mean():.2f} €\n"
    report += f"- **Average optimized cost**: {mode_data['total_cost_optimized'].mean():.2f} €\n"
    report += f"- **Average savings**: {mode_data['savings'].mean():.2f} € ({mode_data['savings_pct'].mean():.2f}%)\n"
    report += f"- **Median savings**: {mode_data['savings'].median():.2f} € ({mode_data['savings_pct'].median():.2f}%)\n"
    report += f"- **Max savings**: {mode_data['savings'].max():.2f} € ({mode_data['savings_pct'].max():.2f}%)\n"
    report += f"- **Min savings**: {mode_data['savings'].min():.2f} € ({mode_data['savings_pct'].min():.2f}%)\n\n"
    
    # Savings distribution
    report += "### Savings Distribution\n\n"
    savings_ranges = [
        (-float('inf'), 0),
        (0, 5),
        (5, 10),
        (10, 15),
        (15, 20),
        (20, float('inf'))
    ]
    
    distribution = []
    for low, high in savings_ranges:
        if high == float('inf'):
            count = len(mode_data[mode_data['savings_pct'] >= low])
            distribution.append(f"{low}%+: {count} days ({count/len(mode_data)*100:.1f}%)")
        elif low == -float('inf'):
            count = len(mode_data[mode_data['savings_pct'] < high])
            distribution.append(f"<{high}%: {count} days ({count/len(mode_data)*100:.1f}%)")
        else:
            count = len(mode_data[(mode_data['savings_pct'] >= low) & (mode_data['savings_pct'] < high)])
            distribution.append(f"{low}%-{high}%: {count} days ({count/len(mode_data)*100:.1f}%)")
    
    report += "- " + "\n- ".join(distribution) + "\n\n"
    
    # Violations if provided
    if violations:
        report += "### Validation Violations\n\n"
        
        if 'energy_balance' in violations:
            balance_violations = violations['energy_balance']
            report += f"- **Energy balance violations**: {len(balance_violations)} hours\n"
            for day, hours in balance_violations.items():
                report += f"  - Day {day}: {len(hours)} hours with imbalance\n"
        
        if 'battery_limits' in violations:
            battery_violations = violations['battery_limits']
            report += f"- **Battery limit violations**: {len(battery_violations)} instances\n"
            for day, violations_list in battery_violations.items():
                report += f"  - Day {day}: {len(violations_list)} violations\n"
        
        if 'savings' in violations:
            savings_violations = violations['savings']
            report += f"- **Savings violations**: {len(savings_violations)} days\n"
            for day, details in savings_violations.items():
                report += f"  - Day {day}: {details.get('savings_pct', 0):.2f}% (min required: {details.get('required_pct', 0):.2f}%)\n"
    
    return report

def assert_ev_window(ev_agent, schedule):
    """
    Ensure EV charging only occurs during allowed hours.
    
    Args:
        ev_agent: EV agent with allowed_hours attribute
        schedule: Dictionary or list with hourly charging schedule
    
    Returns:
        bool: True if all charging respects allowed hours
    """
    if not ev_agent or not hasattr(ev_agent, 'allowed_hours'):
        return True  # No restrictions to check
    
    allowed_hours = getattr(ev_agent, 'allowed_hours', list(range(24)))
    
    if isinstance(schedule, dict):
        charging_hours = schedule.keys()
    elif hasattr(ev_agent, 'hourly_charge'):
        charging_hours = [h for h, charge in enumerate(ev_agent.hourly_charge) if charge > 0]
    else:
        return True  # No schedule to check
    
    for hour in charging_hours:
        if hour not in allowed_hours:
            return False
    
    return True

def assert_ev_departure(ev_agent, departure_soc):
    """
    Ensure EV reaches target SOC by departure time.
    
    Args:
        ev_agent: EV agent with target SOC and must_be_full_by_hour
        departure_soc: SOC at departure time
    
    Returns:
        bool: True if departure SOC meets target
    """
    if not ev_agent or not hasattr(ev_agent, 'soc_max'):
        return True  # No target to check
    
    target_soc = ev_agent.soc_max * 0.95  # 95% of max capacity
    return departure_soc >= (target_soc - 0.01)  # Small tolerance

def to_markdown(df, tablefmt="pipe"):
    """
    Convert DataFrame to markdown table.
    
    Args:
        df: DataFrame to convert
        tablefmt: Table format (ignored, always uses pipe format)
    
    Returns:
        str: Markdown table string
    """
    try:
        # Try using pandas built-in method
        return df.to_markdown(index=False)
    except AttributeError:
        # Fallback to manual implementation
        if df.empty:
            return ""
        
        # Create header
        headers = list(df.columns)
        md = "| " + " | ".join(headers) + " |\n"
        md += "| " + " | ".join(["---" for _ in headers]) + " |\n"
        
        # Add rows
        for _, row in df.iterrows():
            values = [str(row[col]) for col in headers]
            md += "| " + " | ".join(values) + " |\n"
        
        return md

def write_csv_and_md(df, path_base):
    """
    Write DataFrame to both CSV and Markdown files.
    
    Args:
        df: DataFrame to write
        path_base: Base path without extension
    """
    # Note: CSV creation removed to save disk space
    # df.to_csv(f"{path_base}.csv", index=False)  # Disabled
    
    # Write Markdown
    with open(f"{path_base}.md", "w") as f:
        f.write(to_markdown(df))

def select_full_24h_days(df):
    """
    Select only days with complete 24h data and nonzero usage.
    
    Args:
        df: DataFrame with hourly data
    
    Returns:
        list: List of date objects for valid days
    """
    # Group by day and check for 24 hours
    daily_groups = df.groupby('day')
    valid_days = []
    
    for day, group in daily_groups:
        # Must have exactly 24 hours for consistent agent processing
        if len(group) == 24:
            # Check that hours are unique (no duplicates due to DST changes)
            unique_hours = group['hour'].nunique()
            if unique_hours == 24:
                # Check for nonzero usage in device columns
                device_cols = [col for col in group.columns 
                              if any(building in col for building in ['DE_KN_residential', 'DE_KN_industrial'])
                              and 'grid' not in col.lower() and 'pv' not in col.lower()]
                
                if device_cols:
                    total_usage = group[device_cols].sum().sum()
                    if total_usage > 0:
                        valid_days.append(day)
    
    return sorted(valid_days)

def plot_device_dec_vs_cont(dev_dec, dev_cont, building_id, has_pv, price_series):
    """
    Plot device-level comparison: decentralised vs centralised_continuous.
    
    Args:
        dev_dec: Device with decentralised optimization results
        dev_cont: Device with centralised_continuous optimization results
        building_id: Building ID
        has_pv: Whether building has PV
        price_series: Hourly price data
    """
    # Get JADS color palette
    try:
        COLOURS = {
            "Original": "#4C72B0",
            "Decentralised_WithBatt": "#55A868", 
            "Centralised": "#CCB974"
        }
    except:
        COLOURS = {
            "Original": "blue",
            "Decentralised_WithBatt": "green", 
            "Centralised": "orange"
        }
    
    fig, ax = plt.subplots(figsize=(12, 6))
    hours = range(24)
    
    # Plot three consumption curves
    original = getattr(dev_dec, 'original_consumption', np.zeros(24))[:24]
    decentralised = getattr(dev_dec, 'optimized_consumption', np.zeros(24))[:24]
    continuous = getattr(dev_cont, 'centralized_continuous_schedule', np.zeros(24))[:24]
    
    ax.plot(hours, original, 'o-', color=COLOURS["Original"], label='Original', linewidth=2)
    ax.plot(hours, decentralised, 's-', color=COLOURS["Decentralised_WithBatt"], label='Decentralised', linewidth=2)
    ax.plot(hours, continuous, '^-', color=COLOURS["Centralised"], label='Continuous', linewidth=2)
    
    # Add PV if available
    if has_pv and hasattr(dev_dec, 'data'):
        pv_cols = get_pv_columns(dev_dec.data)
        if pv_cols:
            pv_gen = -dev_dec.data.groupby('hour')[pv_cols].sum().sum(axis=1).reindex(range(24), fill_value=0)
            ax.fill_between(hours, 0, pv_gen, alpha=0.3, color='orange', label='PV Generation')
    
    # Shade price hours
    shade_price_hours(ax, price_series)
    
    # Calculate and annotate savings at hour 12
    if sum(original) > 0:
        dec_savings = (sum(original) - sum(decentralised)) / sum(original) * 100
        cont_savings = (sum(original) - sum(continuous)) / sum(original) * 100
        
        ax.text(12, max(max(original), max(decentralised), max(continuous)) * 0.8,
                f'Dec: {dec_savings:.1f}%\nCont: {cont_savings:.1f}%',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                ha='center', va='center')
    
    ax.set_xlabel('Hour')
    ax.set_ylabel('Energy (kWh)')
    ax.set_xlim(-0.5, 23.5)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    device_name = getattr(dev_dec, 'device_name', 'unknown')
    ax.set_title(f'{building_id} — {device_name.replace("_", " ").title()} — Original vs. Decentralised vs. Continuous')
    
    # Save figure
    filename = f"results/figures/{building_id}_{device_name}_dec_vs_cont.png"
    plt.savefig(filename, bbox_inches='tight')
    close_figures()
    
    return filename

def plot_building_dec_vs_cont(devices_dec, devices_cont, building_id, price_series):
    """
    Plot building-level comparison: decentralised vs centralised_continuous.
    
    Args:
        devices_dec: List of devices with decentralised results
        devices_cont: List of devices with centralised_continuous results
        building_id: Building ID
        price_series: Hourly price data
    """
    # Get JADS color palette
    try:
        COLOURS = {
            "Original": "#4C72B0",
            "Decentralised_WithBatt": "#55A868", 
            "Centralised": "#CCB974"
        }
    except:
        COLOURS = {
            "Original": "blue",
            "Decentralised_WithBatt": "green", 
            "Centralised": "orange"
        }
    
    fig, ax = plt.subplots(figsize=(12, 6))
    hours = range(24)
    
    # Sum all device loads
    building_original = np.zeros(24)
    building_decentralised = np.zeros(24)
    building_continuous = np.zeros(24)
    
    for dev in devices_dec:
        original = getattr(dev, 'original_consumption', np.zeros(24))[:24]
        decentralised = getattr(dev, 'optimized_consumption', np.zeros(24))[:24]
        building_original += original
        building_decentralised += decentralised
    
    for dev in devices_cont:
        continuous = getattr(dev, 'centralized_continuous_schedule', np.zeros(24))[:24]
        building_continuous += continuous
    
    # Plot building-level curves
    ax.plot(hours, building_original, 'o-', color=COLOURS["Original"], label='Original', linewidth=2)
    ax.plot(hours, building_decentralised, 's-', color=COLOURS["Decentralised_WithBatt"], label='Decentralised', linewidth=2)
    ax.plot(hours, building_continuous, '^-', color=COLOURS["Centralised"], label='Continuous', linewidth=2)
    
    # Shade price hours
    shade_price_hours(ax, price_series)
    
    # Calculate and annotate total savings
    if sum(building_original) > 0:
        dec_savings = (sum(building_original) - sum(building_decentralised)) / sum(building_original) * 100
        cont_savings = (sum(building_original) - sum(building_continuous)) / sum(building_original) * 100
        
        ax.text(12, max(max(building_original), max(building_decentralised), max(building_continuous)) * 0.8,
                f'Dec: {dec_savings:.1f}%\nCont: {cont_savings:.1f}%',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                ha='center', va='center')
    
    ax.set_xlabel('Hour')
    ax.set_ylabel('Total Building Load (kWh)')
    ax.set_xlim(-0.5, 23.5)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    ax.set_title(f'{building_id} — Building-level: Original vs. Decentralised vs. Continuous')
    
    # Save figure
    filename = f"results/figures/{building_id}_building_dec_vs_cont.png"
    plt.savefig(filename, bbox_inches='tight')
    close_figures()
    
    return filename

def make_cumulative_savings_plot(kpi_df, building_id):
    """
    Plot cumulative savings over time.
    
    Args:
        kpi_df: DataFrame with KPI data including savings_eur column
        building_id: Building ID
    """
    try:
        COLOURS = {"Centralised": "#CCB974"}
    except:
        COLOURS = {"Centralised": "orange"}
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort by day and calculate cumulative savings
    sorted_df = kpi_df.sort_values('day').reset_index(drop=True)
    cumulative_savings = sorted_df['savings_eur'].cumsum()
    
    # Plot cumulative savings
    days = range(1, len(cumulative_savings) + 1)
    ax.plot(days, cumulative_savings, 'o-', color=COLOURS["Centralised"], linewidth=2, markersize=6)
    
    # Scatter each day
    ax.scatter(days, cumulative_savings, color=COLOURS["Centralised"], s=50, alpha=0.7)
    
    # Annotate final total
    final_total = cumulative_savings.iloc[-1]
    ax.annotate(f'Total: {final_total:.2f} €', 
                xy=(len(days), final_total), 
                xytext=(len(days)*0.8, final_total*1.1),
                arrowprops=dict(arrowstyle='->', color='black'),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
    
    ax.set_xlabel('Day Index')
    ax.set_ylabel('Cumulative Savings (€)')
    ax.set_title(f'{building_id} — Cumulative Savings (Continuous vs. Baseline)')
    ax.grid(True, alpha=0.3)
    
    # Save figure
    filename = f"results/figures/{building_id}_cumulative_savings_cont_vs_dec.png"
    plt.savefig(filename, bbox_inches='tight')
    close_figures()
    
    return filename

# Data wrapper functions for agent compatibility

def prepare_agent_data_for_day(df, day, building_id):
    """
    Prepare data for agents - agents expect full dataset but filter internally by day.
    
    Args:
        df: Full DataFrame with all days
        day: Specific day (datetime.date object) to optimize
        building_id: Building ID
    
    Returns:
        tuple: (full_df, day_prices_array)
    """
    # Ensure day is available in data
    if day not in df['day'].values:
        raise ValueError(f"Day {day} not found in dataset")
    
    # Get price array for the day (agents expect this for cost calculations)
    day_mask = df['day'] == day
    day_df = df[day_mask].copy()
    
    # Ensure we have exactly 24 hours with unique hours
    if len(day_df) != 24 or day_df['hour'].nunique() != 24:
        raise ValueError(f"Day {day} does not have exactly 24 unique hours (has {len(day_df)} rows, {day_df['hour'].nunique()} unique hours)")
    
    # Sort by hour to ensure proper order
    day_df = day_df.sort_values('hour')
    
    # Get prices indexed by hour 0-23
    day_prices = day_df.set_index('hour')['price_per_kwh'].reindex(range(24), fill_value=df['price_per_kwh'].mean())
    
    return df, day_prices.values

def create_device_agents_for_optimization(df, building_id, day, prob_agent=None):
    """
    Create device agents with proper data formatting for optimization.
    
    Args:
        df: Full DataFrame (agents expect this)
        building_id: Building ID
        day: Day to optimize (datetime.date)
        prob_agent: ProbabilityModelAgent instance
    
    Returns:
        list: List of FlexibleDevice agents ready for optimization
    """
    import sys
    from pathlib import Path
    sys.path.append(str(Path.cwd() / "notebooks"))
    
    from agents.FlexibleDeviceAgent import FlexibleDevice
    
    # Get device specs
    device_specs = extract_device_specs(df, building_id)
    devices = []
    
    for device_id, spec in device_specs.items():
        if device_id in df.columns and 'grid_export' not in device_id:
            # Create device with FULL dataset as agents expect
            device = FlexibleDevice(
                device_name=device_id,
                data=df,  # Full dataset
                category=spec.get('category', 'Non-Flexible'),
                power_rating=spec.get('power_rating', 1.0),
                is_flexible=(spec.get('category') != 'Non-Flexible'),
                allowed_hours=spec.get('allowed_hours', list(range(24))),
                max_shift_hours=spec.get('max_shift_hours', 6),
                phases=spec.get('phases', []),
                efficiency=spec.get('efficiency', 1.0)
            )
            
            # Set device probabilities from prob_agent if available
            if prob_agent and hasattr(prob_agent, 'latest_distributions'):
                device_type = device_id.split('_')[-1]  # Extract device type
                if device_type in prob_agent.latest_distributions:
                    device.hour_probability = prob_agent.latest_distributions[device_type]
                else:
                    # Default uniform probability
                    device.hour_probability = {h: 1/24 for h in range(24)}
            
            devices.append(device)
    
    return devices

def create_battery_agent_for_optimization(building_id, battery_on=True):
    """
    Create battery agent with proper configuration.
    
    Args:
        building_id: Building ID
        battery_on: Whether battery is enabled
    
    Returns:
        BatteryAgent or None
    """
    if not battery_on:
        return None
    
    import sys
    from pathlib import Path
    sys.path.append(str(Path.cwd() / "notebooks"))
    
    from agents.BatteryAgent import BatteryAgent
    
    # Load battery parameters from config
    try:
        sys.path.append(str(Path.cwd() / "notebooks" / "utils"))
        from config import BATTERY_PARAMS
        battery_params = BATTERY_PARAMS
    except:
        # Fallback default parameters
        battery_params = {
            'soc_min': 0.1,
            'soc_max': 10.0,
            'max_charge_rate': 3.0,
            'max_discharge_rate': 3.0,
            'efficiency_charge': 0.95,
            'efficiency_discharge': 0.95,
            'degradation_cost': 0.0005,
            'current_soc': 5.0
        }
    
    return BatteryAgent(**battery_params)

def create_ev_agent_for_optimization(df, building_id, ev_on=True, prob_agent=None):
    """
    Create EV agent if EV column exists and EV is enabled.
    
    Args:
        df: Full DataFrame
        building_id: Building ID
        ev_on: Whether EV is enabled
        prob_agent: ProbabilityModelAgent instance
    
    Returns:
        EVAgent or None
    """
    if not ev_on:
        return None
    
    # Check if EV column exists
    ev_col = f"{building_id}_ev"
    if ev_col not in df.columns:
        return None
    
    import sys
    from pathlib import Path
    sys.path.append(str(Path.cwd() / "notebooks"))
    
    from agents.EVAgent import EVAgent
    
    # Load EV parameters from config
    try:
        sys.path.append(str(Path.cwd() / "notebooks" / "utils"))
        from config import EV_PARAMS
        ev_params = EV_PARAMS.copy()
    except:
        # Fallback default parameters
        ev_params = {
            'soc_min': 6.0,
            'soc_max': 54.0,
            'max_charge_rate': 7.4,
            'efficiency_charge': 0.92,
            'current_soc': 30.0,
            'must_be_full_by_hour': 7
        }
    
    # Infer must_be_full_by_hour from probability data if available
    if prob_agent and hasattr(prob_agent, 'latest_distributions') and 'ev' in prob_agent.latest_distributions:
        ev_probs = prob_agent.latest_distributions['ev']
        if ev_probs:
            # Find hour with highest probability as departure time
            max_prob_hour = max(ev_probs.keys(), key=lambda h: ev_probs[h])
            ev_params['must_be_full_by_hour'] = max_prob_hour
    
    # Create EV agent
    ev_agent = EVAgent(
        device_name=ev_col,
        data=df,
        **ev_params
    )
    
    return ev_agent

def get_day_prices_from_db(building_id, day):
    """
    Get price array for a specific day from DuckDB.
    
    Args:
        building_id: Building ID
        day: Day (datetime.date object)
    
    Returns:
        np.array: 24-hour price array
    """
    con = get_con()
    day_str = day.strftime('%Y-%m-%d')
    
    try:
        prices_df = con.execute(
            f"SELECT hour, price_per_kwh FROM {building_id}_processed_data WHERE day = '{day_str}'"
        ).df()
        
        if prices_df.empty:
            # Return default prices if no data
            return np.full(24, 0.25)
        
        # Create 24-hour array with proper indexing
        price_array = np.full(24, prices_df['price_per_kwh'].mean())
        for _, row in prices_df.iterrows():
            hour = int(row['hour'])
            if 0 <= hour < 24:
                price_array[hour] = row['price_per_kwh']
        
        return price_array
    except Exception as e:
        print(f"Warning: Failed to get prices for {building_id} on {day}: {e}")
        return np.full(24, 0.25)

def save_json_schedule(building_id, day, devices):
    """
    Save device schedules to JSON file.
    
    Args:
        building_id: Building ID
        day: Day (datetime.date object)
        devices: List of devices with optimization results
    """
    import json
    
    schedule_data = {
        'building_id': building_id,
        'day': day.strftime('%Y-%m-%d'),
        'devices': {}
    }
    
    for device in devices:
        device_name = getattr(device, 'device_name', 'unknown')
        
        # Try different schedule attributes
        schedule = None
        if hasattr(device, 'nextday_optimized_schedule'):
            schedule = device.nextday_optimized_schedule[:24]
        elif hasattr(device, 'centralized_optimized_schedule'):
            schedule = device.centralized_optimized_schedule[:24]
        elif hasattr(device, 'optimized_consumption'):
            schedule = device.optimized_consumption[:24]
        
        if schedule is not None:
            schedule_data['devices'][device_name] = [float(x) for x in schedule]
    
    # Save to file
    filename = f"results/output/{building_id}_{day.strftime('%Y-%m-%d')}_schedule.json"
    with open(filename, 'w') as f:
        json.dump(schedule_data, f, indent=2)
    
    return filename

# Note: centralised_continuous mode uses the existing run_centralized_optimization function
# from 02_run.py - no separate GlobalOptimiser class needed