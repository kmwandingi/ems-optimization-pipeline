import unittest
import pandas as pd
import numpy as np
import logging
import sys

# Add project root to path to allow direct script execution
sys.path.append('d:\\Kenneth - TU Eindhoven\\Jads\\Graduation Project 2024-2025\\ems_project\\ems-optimization-pipeline')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import agents and optimizer
from notebooks.agents.GlobalOptimizer import GlobalOptimizer
from notebooks.agents.PVAgent import PVAgent
from notebooks.agents.BatteryAgent import BatteryAgent
from notebooks.agents.FlexibleDeviceAgent import FlexibleDevice
from notebooks.agents.GridAgent import GridAgent

# Helper functions defined locally to make the test self-contained
def calculate_total_cost_from_series(net_load_series: pd.Series, price_series: pd.Series):
    """Calculates total grid cost. Assumes import and export prices are the same."""
    return (net_load_series * price_series).sum()

def calculate_self_consumption(consumption_series: pd.Series, pv_series: pd.Series):
    """Calculates total PV energy self-consumed (in kWh)."""
    # Self-consumption is the minimum of generation and consumption at each time step
    self_consumed_pv = np.minimum(consumption_series, pv_series)
    total_self_consumed = self_consumed_pv.sum()
    return total_self_consumed, 0 # Return 0 for rate as it's not used in assertions

class TestOptimizationValidation(unittest.TestCase):

    def setUp(self):
        """Set up a small, predictable test case."""
        logging.info("Setting up validation test...")

        # Time range for 24 hours
        self.time_index = pd.to_datetime(pd.date_range("2023-01-01", periods=24, freq="h"))

        # Create dummy device data
        self.device_data = pd.DataFrame(index=self.time_index)
        self.device_data['fixed_load'] = np.ones(24) * 2  # 2 kW constant load
        self.device_data['flexible_load_1'] = 0
        self.device_data.loc[self.time_index[10:14], 'flexible_load_1'] = 3 # 3kW for 4 hours

        # Create dummy PV data
        pv_generation = np.zeros(24)
        pv_generation[7:18] = 4 * np.sin(np.pi * (np.arange(11)) / 10) # Simple sine wave for PV
        self.pv_data = pd.DataFrame({'generation': pv_generation}, index=self.time_index)

        # Create dummy price data (high prices when flexible load is on)
        prices = np.ones(24) * 0.2 # base price
        prices[10:14] = 0.5 # peak price
        prices[2:8] = 0.1 # off-peak price
        self.price_data = pd.DataFrame({'price': prices}, index=self.time_index)

        # Device specifications
        # Device specifications - updated for FlexibleDevice class
        self.test_device_specs = {
            'fixed_load': {'is_flexible': False},
            'flexible_load_1': {
                'is_flexible': True,
                'flexibility_type': 'shiftable',
                'shifting_window_start': 0,
                'shifting_window_end': 23,
                'duration': 4,
                'daily_limit': 1,
                'power_rating': 3.0,
                'category': 'test_category'
            }
        }

        # The FlexibleDevice class expects a 'utc_timestamp' column
        self.device_data_with_ts = self.device_data.reset_index().rename(columns={'index': 'utc_timestamp'})

        # Battery parameters
        self.battery_params = {
            'capacity': 10,
            'soc_max': 10,
            'soc_min': 1,
            'max_charge_rate': 5,
            'max_discharge_rate': 5,
            'initial_soc': 5,
            'degradation_rate': 0.0001
        }

        logging.info("Test setup complete.")

    def run_optimization(self, use_battery=False):
        """Helper function to run the optimization pipeline."""
        pv_agent = PVAgent(profile_data=self.pv_data, profile_cols=['generation'])
        grid_agent = GridAgent(price_data=self.price_data)
        
        flexible_agents = []
        for name, spec in self.test_device_specs.items():
            if spec.get('is_flexible'):
                agent = FlexibleDevice(
                    data=self.device_data_with_ts,
                    device_name=name,
                    category=spec['category'],
                    power_rating=spec['power_rating'],
                    global_layer=None,  # Assuming None is acceptable for the test
                    spec=spec
                )
                flexible_agents.append(agent)

        battery_agent = BatteryAgent(**self.battery_params) if use_battery else None
        fixed_consumption = self.device_data[['fixed_load']].sum(axis=1)

        optimizer = GlobalOptimizer(
            pv_agent=pv_agent,
            battery_agent=battery_agent,
            grid_agent=grid_agent,
            flexible_agents=flexible_agents,
            fixed_consumption=fixed_consumption,
            time_steps=len(self.time_index)
        )

        prob, solved_schedule, solved_soc = optimizer.optimize_centralized()
        
        optimized_consumption = pd.Series(solved_schedule['total_flexible_load'], index=self.time_index)
        total_consumption = fixed_consumption + optimized_consumption
        net_load = total_consumption - pv_agent.get_pv_profile(self.time_index)

        if use_battery and battery_agent:
            battery_charge = pd.Series(battery_agent.charge_schedule, index=self.time_index)
            battery_discharge = pd.Series(battery_agent.discharge_schedule, index=self.time_index)
            net_load += battery_charge - battery_discharge

        return net_load, total_consumption, battery_agent

    def test_optimization_logic(self):
        """
        Validates that optimization reduces cost and that adding a battery helps.
        """
        logging.info("Starting optimization logic validation test...")

        # --- 1. Calculate Original Cost (No Optimization) ---
        original_total_consumption = self.device_data.sum(axis=1)
        original_net_load = original_total_consumption - self.pv_data['generation']
        
        cost_original = calculate_total_cost_from_series(
            net_load_series=original_net_load,
            price_series=self.price_data['price']
        )
        pv_sc_original, _ = calculate_self_consumption(
            consumption_series=original_total_consumption,
            pv_series=self.pv_data['generation']
        )
        logging.info(f"Original Cost: {cost_original:.4f}, PV Self-Consumption: {pv_sc_original:.4f}")

        # --- 2. Calculate Optimized Cost (No Battery) ---
        net_load_opt_no_batt, consumption_opt_no_batt, _ = self.run_optimization(use_battery=False)
        
        cost_opt_no_batt = calculate_total_cost_from_series(
            net_load_series=net_load_opt_no_batt,
            price_series=self.price_data['price']
        )
        pv_sc_opt_no_batt, _ = calculate_self_consumption(
            consumption_series=consumption_opt_no_batt,
            pv_series=self.pv_data['generation']
        )
        logging.info(f"Optimized Cost (No Battery): {cost_opt_no_batt:.4f}, PV Self-Consumption: {pv_sc_opt_no_batt:.4f}")

        # --- 3. Calculate Optimized Cost (With Battery) ---
        net_load_opt_with_batt, consumption_opt_with_batt, solved_battery_agent = self.run_optimization(use_battery=True)
        
        total_throughput = np.sum(solved_battery_agent.charge_schedule) + np.sum(solved_battery_agent.discharge_schedule)
        degradation_cost = self.battery_params['degradation_rate'] * total_throughput
        
        cost_opt_with_batt_grid = calculate_total_cost_from_series(
            net_load_series=net_load_opt_with_batt,
            price_series=self.price_data['price']
        )
        cost_opt_with_batt = cost_opt_with_batt_grid + degradation_cost

        pv_sc_opt_with_batt, _ = calculate_self_consumption(
            consumption_series=consumption_opt_with_batt,
            pv_series=self.pv_data['generation']
        )
        logging.info(f"Optimized Cost (With Battery): {cost_opt_with_batt:.4f} (Grid: {cost_opt_with_batt_grid:.4f}, Degradation: {degradation_cost:.4f}), PV Self-Consumption: {pv_sc_opt_with_batt:.4f}")

        # --- 4. Assertions ---
        logging.info("Performing assertions...")
        
        self.assertLessEqual(round(cost_opt_no_batt, 5), round(cost_original, 5),
                             "ERROR: Optimization without battery increased the cost!")

        self.assertLessEqual(round(cost_opt_with_batt, 5), round(cost_opt_no_batt, 5),
                             "ERROR: Optimization with battery increased the cost compared to no battery!")

        self.assertGreaterEqual(round(pv_sc_opt_no_batt, 5), round(pv_sc_original, 5),
                                "ERROR: PV self-consumption decreased after optimization!")
        
        self.assertGreaterEqual(round(pv_sc_opt_with_batt, 5), round(pv_sc_opt_no_batt, 5),
                                "ERROR: PV self-consumption with battery was worse than without!")

        logging.info("Validation test passed successfully!")

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
