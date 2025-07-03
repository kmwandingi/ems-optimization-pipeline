import pandas as pd
from pathlib import Path
import sys
import numpy as np
import datetime

# Add necessary paths for imports
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root / 'scripts'))
sys.path.append(str(project_root / 'notebooks'))
sys.path.append(str(project_root / 'notebooks' / 'agents'))

# Import common functions
from common import get_con, select_full_24h_days, get_pv_columns

# Mock classes (defined inline for self-containment)
class MockFlexibleDevice:
    def __init__(self, data, device_name):
        self.device_name = device_name
        self.data = data
        if self.device_name in self.data.columns:
            self.original_consumption = self.data[self.device_name].values.copy()
        else:
            # Fallback if column not found, to prevent errors
            self.original_consumption = np.zeros(len(self.data))
        self.optimized_consumption = self.original_consumption.copy()

class MockBatteryAgent:
    def __init__(self, initial_soc=5.0, max_charge_rate=3.0, max_discharge_rate=3.0):
        self.current_soc = initial_soc
        self.max_charge_rate = max_charge_rate
        self.max_discharge_rate = max_discharge_rate
        self.hourly_charge = [0.0] * 24
        self.hourly_discharge = [0.0] * 24

    def get_battery_state(self):
        return {
            'current_soc': self.current_soc,
            'soc_min': 0.0,
            'soc_max': 10.0,
            'max_charge_rate': self.max_charge_rate,
            'max_discharge_rate': self.max_discharge_rate,
            'degradation_cost': 0.0,
            'hourly_plan': {}
        }

    def apply_battery_effect(self, consumption_profile):
        modified_profile = consumption_profile.copy()
        for h in range(24):
            if h < 4:
                charge_amount = min(self.max_charge_rate, 1.0)
                modified_profile[h] -= charge_amount
                self.hourly_charge[h] = charge_amount
            elif h >= 16 and h < 20:
                discharge_amount = min(self.max_discharge_rate, 1.0)
                modified_profile[h] += discharge_amount
                self.hourly_discharge[h] = discharge_amount
        return modified_profile

BUILDING_ID = 'DE_KN_residential1'
NUM_SIMULATION_DAYS = 10

def run_simulation():
    print("Python script started.") # Simple confirmation print
    try:
        con, view_name = get_con(BUILDING_ID)
        df = con.execute(f'SELECT * FROM {view_name}').fetchdf()

        df['utc_timestamp'] = pd.to_datetime(df['utc_timestamp'])
        df['day'] = df['utc_timestamp'].dt.date
        df['hour'] = df['utc_timestamp'].dt.hour

        valid_days = select_full_24h_days(df)

        if len(valid_days) < NUM_SIMULATION_DAYS:
            print(f'Error: Not enough valid days ({len(valid_days)}) for simulation. Need at least {NUM_SIMULATION_DAYS}.\n')
            return

        simulation_days = valid_days[:NUM_SIMULATION_DAYS]

        total_consumption_no_batt = 0.0
        total_pv_no_batt = 0.0
        total_consumption_with_batt = 0.0
        total_pv_with_batt = 0.0

        for i, day in enumerate(simulation_days):
            day_df = df[df['day'] == day].copy()

            device_cols = [c for c in df.columns if BUILDING_ID in c and 'grid' not in c.lower() and 'pv' not in c.lower() and c in day_df.columns]

            # --- Scenario 1: No Battery ---
            current_day_consumption_no_batt = 0.0
            for col in device_cols:
                mock_device = MockFlexibleDevice(day_df, col)
                current_day_consumption_no_batt += mock_device.original_consumption.sum()
            total_consumption_no_batt += current_day_consumption_no_batt

            pv_cols = get_pv_columns(day_df)
            if pv_cols:
                current_day_pv_no_batt = day_df[pv_cols].sum().sum()
                total_pv_no_batt += current_day_pv_no_batt

            # --- Scenario 2: With Battery ---
            current_day_consumption_with_batt = 0.0
            mock_battery = MockBatteryAgent()

            for col in device_cols:
                mock_device = MockFlexibleDevice(day_df, col)
                modified_consumption = mock_battery.apply_battery_effect(mock_device.original_consumption)
                current_day_consumption_with_batt += modified_consumption.sum()

            total_consumption_with_batt += current_day_consumption_with_batt

            if pv_cols:
                current_day_pv_with_batt = day_df[pv_cols].sum().sum()
                total_pv_with_batt += current_day_pv_with_batt

        results = {
            'Scenario': ['Without Battery', 'With Battery'],
            'Total Consumption (kWh)': [total_consumption_no_batt, total_consumption_with_batt],
            'Total PV (kWh)': [total_pv_no_batt, total_pv_with_batt]
        }
        results_df = pd.DataFrame(results)

        print("\nSimulation Results (10 Days):")
        print(results_df.to_markdown(index=False))

    except Exception as e:
        print(f"An error occurred during simulation: {e}")

    print("Script finished.") # Final confirmation print

if __name__ == '__main__':
    run_simulation()