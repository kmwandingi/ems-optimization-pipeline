# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║ V-27: PMF CONVERGENCE VISUALIZATION                                         ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
#
# Objective:
# My goal here is to create a clear, compelling visual that demonstrates the adaptive
# learning capability of the ProbabilityModelAgent. I will generate a line chart
# showing how the agent's belief about a device's start time—specifically, the
# probability of it starting at 19:00—evolves and converges as it processes more
# daily data. This will provide tangible proof of the "learning" process described
# in the report text.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

# --- Configuration & Setup ---
# I'm setting up the necessary paths and constants here. I'll add the project
# root to the system path to ensure my script can find the custom agent modules.
# This makes the script portable within the project structure.
try:
    # This will work when running from the project root.
    project_root = Path(__file__).resolve().parents[3]
except (NameError, IndexError):
    # This is a fallback for interactive environments like Jupyter.
    project_root = Path.cwd()

if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from agents.ProbabilityModelAgent import ProbabilityModelAgent
from utils.device_specs import device_specs

# I'm defining the key parameters for this simulation. I'll focus on a
# specific residential building and a common appliance, the washing machine,
# to make the visualization relatable. The target hour is 19:00, a typical
# evening usage time.
BUILDING_ID = 'DE_KN_residential1'
DEVICE_NAME = f'{BUILDING_ID}_washing_machine'
TARGET_HOUR = 19
SIMULATION_DAYS = 60
OUTPUT_DIR = project_root / 'notebooks' / 'report' / 'report_1' / 'assets'
DATA_PATH = project_root / 'processed_data' / f'{BUILDING_ID}_processed_data.parquet'

# I'll make sure the output directory exists before trying to save anything.
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║ SECTION 1: DATA LOADING AND PREPARATION                                   ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

def load_data(path: Path) -> pd.DataFrame:
    """
    Loads the processed parquet data for the specified building.
    My thinking is to centralize data loading into a single function to keep the
    main script clean and to handle potential file errors gracefully.
    """
    if not path.exists():
        raise FileNotFoundError(f"Critical error: Processed data not found at {path}.")
    df = pd.read_parquet(path)
    df['utc_timestamp'] = pd.to_datetime(df['utc_timestamp'])
    df['day'] = df['utc_timestamp'].dt.date
    return df

# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║ SECTION 2: LEARNING SIMULATION                                            ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

def simulate_learning_convergence(data: pd.DataFrame, device_name: str, target_hour: int, sim_days: int) -> list:
    """
    Simulates the day-by-day learning process and tracks probability convergence.
    My approach is to instantiate one agent and train it sequentially, mimicking
    how it would operate in the real world. This ensures the learning trajectory
    is authentic.
    """
    # I'm initializing the agent. I'm disabling DuckDB priors to ensure the
    # learning starts from a blank slate (uniform distribution) for this demo.
    prob_agent = ProbabilityModelAgent(building_id=BUILDING_ID, use_duckdb_priors=False)
    prob_agent.LR_MAX = 0.10
    prob_agent.LR_TAU = 20.0

    daily_probabilities = []
    unique_days = sorted(data['day'].unique())

    if len(unique_days) < sim_days:
        print(f'Warning: Not enough unique days in data ({len(unique_days)}), using all available.')
        sim_days = len(unique_days)

    # I'll loop through the specified number of days, feeding one day at a time.
    for i in range(sim_days):
        day = unique_days[i]
        day_str = str(day)
        
        # The train method updates the agent's internal probabilities.
        # I'm passing empty DataFrames for weather as they are not needed for this
        # specific device's probability model.
        _, device_probs = prob_agent.train(
            building_id=BUILDING_ID,
            days_list=[day_str],
            device_specs=device_specs,
            weather_df=pd.DataFrame(),
            forecast_df=pd.DataFrame()
        )
        
        # After training, I extract the new probability for the target hour.
        device_prob_data = device_probs.get(device_name, {})
        hour_prob_dict = device_prob_data.get('hour_probability', {})
        prob_at_target_hour = hour_prob_dict.get(target_hour, 1/24) # Default to uniform if not found
        daily_probabilities.append(prob_at_target_hour)
        
    return daily_probabilities

# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║ SECTION 3: PLOTTING                                                       ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

def plot_convergence(probabilities: list, device_name: str, target_hour: int, output_path: Path):
    """
    Generates and saves the final convergence plot.
    My focus here is on creating a publication-quality visual: clear, uncluttered,
    and using the specified brand colors for consistency.
    """
    days = range(1, len(probabilities) + 1)
    device_short_name = device_name.split('_')[-1].replace('_', ' ').title()
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # I'm using the specified blue color for the plot line.
    ax.plot(days, probabilities, marker='o', linestyle='-', color='#1f77b4', label=f'P(start at {target_hour}:00)')
    
    ax.set_title(f'Convergence of Learned Start-Hour Probability\nDevice: {device_short_name}', fontsize=16, pad=10)
    ax.set_xlabel('Days of Learning', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_ylim(0, max(probabilities) * 1.2 if probabilities else 0.5)
    ax.set_xlim(0, len(days) + 1)
    ax.legend(loc='upper right')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    fig.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"✅ Visual V-27 successfully generated and saved to: {output_path}")

# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║ SECTION 4: MAIN EXECUTION BLOCK                                           ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

if __name__ == '__main__':
    # This is the main execution block. I'm wrapping it in a try...except
    # block to catch any potential errors during the process and provide a
    # clear message.
    try:
        print(f"--- Generating Visual V-27: PMF Convergence for {DEVICE_NAME.split('_')[-1]} ---")
        print(f"1. Loading data from {DATA_PATH}...")
        data = load_data(DATA_PATH)
        
        print(f"2. Simulating learning process for {SIMULATION_DAYS} days...")
        probabilities = simulate_learning_convergence(data, DEVICE_NAME, TARGET_HOUR, SIMULATION_DAYS)
        
        print("3. Rendering and saving the plot...")
        output_file = OUTPUT_DIR / 'v27_pmf_convergence.png'
        plot_convergence(probabilities, DEVICE_NAME, TARGET_HOUR, output_file)
        
    except Exception as e:
        print(f"❌ An error occurred: {e}")
        # In a real-world scenario, I might add more specific error handling here.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(project_root))

from agents.ProbabilityModelAgent import ProbabilityModelAgent
from utils.device_specs import device_specs

# --- Configuration ---
BUILDING_ID = 'DE_KN_residential1' # Example building
DEVICE_NAME = 'DE_KN_residential1_washing_machine' # Example device
TARGET_HOUR = 19 # The specific hour to track
SIMULATION_DAYS = 60
OUTPUT_DIR = project_root / 'notebooks' / 'report' / 'report_1' / 'assets'

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Data Loading ---
@st.cache_data
def load_data(building_id):
    processed_data_path = project_root / 'processed_data' / f'{building_id}_processed_data.parquet'
    if not processed_data_path.exists():
        raise FileNotFoundError(f'Processed data not found at {processed_data_path}')
    df = pd.read_parquet(processed_data_path)
    df['utc_timestamp'] = pd.to_datetime(df['utc_timestamp'])
    df['day'] = df['utc_timestamp'].dt.date
    return df

# --- Simulation ---
def simulate_learning_convergence(data, device_name, target_hour, sim_days):
    prob_agent = ProbabilityModelAgent(building_id=BUILDING_ID, use_duckdb_priors=False)
    prob_agent.LR_MAX = 0.10
    prob_agent.LR_TAU = 20.0

    daily_probabilities = []
    unique_days = sorted(data['day'].unique())

    if len(unique_days) < sim_days:
        print(f'Warning: Not enough unique days in data ({len(unique_days)}), using all available.')
        sim_days = len(unique_days)

    for i in range(sim_days):
        day = unique_days[i]
        day_str = str(day)
        
        # Mock training on a single day
        prob_agent.train(
            building_id=BUILDING_ID,
            days_list=[day_str],
            device_specs=device_specs,
            weather_df=pd.DataFrame(), # Not needed for this simulation
            forecast_df=pd.DataFrame()
        )
        
        # Get the probability for the target hour
        device_probs = prob_agent.device_probabilities.get(device_name, {})
        hour_prob = device_probs.get('hour_probability', {})
        prob_at_target_hour = hour_prob.get(target_hour, 1/24) # Default to uniform if not found
        daily_probabilities.append(prob_at_target_hour)
        
    return daily_probabilities

# --- Plotting ---
def plot_convergence(probabilities, device_name, target_hour, output_path):
    days = range(1, len(probabilities) + 1)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(days, probabilities, marker='o', linestyle='-', color='#1f77b4', label=f'P(start at {target_hour}:00)')
    
    ax.set_title(f'Convergence of Learned Start-Hour Probability\nDevice: {device_name.split("_")[-1].replace("_", " ").title()}', fontsize=16)
    ax.set_xlabel('Days of Learning', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_ylim(0, max(probabilities) * 1.2 if probabilities else 0.5)
    ax.set_xlim(0, len(days) + 1)
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f'Saved convergence plot to {output_path}')

# --- Main Execution ---
if __name__ == '__main__':
    try:
        print('Loading data...')
        data = load_data(BUILDING_ID)
        
        print('Simulating learning process...')
        probabilities = simulate_learning_convergence(data, DEVICE_NAME, TARGET_HOUR, SIMULATION_DAYS)
        
        print('Plotting results...')
        output_file = OUTPUT_DIR / 'v27_pmf_convergence.png'
        plot_convergence(probabilities, DEVICE_NAME, TARGET_HOUR, output_file)
        
    except Exception as e:
        print(f'An error occurred: {e}')
