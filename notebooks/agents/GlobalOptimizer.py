import numpy as np
import pandas as pd
import logging
import datetime
from datetime import timedelta
from typing import Dict, List, Tuple, Any, Optional
from pulp import (LpProblem, LpVariable, LpStatus,
                  LpMinimize, lpSum, PULP_CBC_CMD)

from agents.ProbabilityModelAgent import ProbabilityModelAgent
from agents.WeatherAgent import WeatherAgent
import time
from notebooks.utils.config import PV_WEIGHT
from agents.FlexibleDeviceAgent import FlexibleDevice, calculate_preference_penalty
from agents.GlobalConnectionLayer import GlobalConnectionLayer
from pulp import LpProblem, LpVariable, lpSum, LpMinimize, PULP_CBC_CMD, LpStatus, LpContinuous

###########################################################
# Helper functions 
###########################################################

def _add_storage_vars(prob: LpProblem,
                      tag: str,
                      hours: range,
                      cap_kwh: float,
                      soc_min: float,
                      soc_max: float,
                      init_soc: float,
                      chg_rate: float,
                      dis_rate: float = 0.0):
    """
    Register charge / discharge / soc variables for one storage device.

    Returns
    -------
    (charge, discharge, soc) – dicts keyed by hour.
    """
    charge    = LpVariable.dicts(f"{tag}Charge", hours, 0, chg_rate, LpContinuous)
    discharge = LpVariable.dicts(f"{tag}Disch",  hours, 0, dis_rate, LpContinuous)
    soc       = LpVariable.dicts(f"{tag}SOC",    hours, soc_min, soc_max, LpContinuous)

    for t in hours:
        if t == 0:
            prob += soc[t] == init_soc + charge[t] - discharge[t], f"SOC_{tag}_{t:02d}"
        else:
            prob += soc[t] == soc[t-1] + charge[t] - discharge[t], f"SOC_{tag}_{t:02d}"
    return charge, discharge, soc

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

from copy import deepcopy

def monte_carlo_simulation(device, day, effective_prices, battery_state, grid_info, num_simulations=100):
    """
    Runs a Monte Carlo simulation for a given device and day.
    For each simulation, it samples a forecast error realization (using a normal distribution with the computed standard deviation)
    and runs the MILP optimization using that simulated forecast.
    
    Returns:
        savings_samples: list of simulated savings outcomes (in €)
        violation_counts: list of constraint violation counts (if available; otherwise zeros)
    """
    # Get the nominal PV forecast (deterministic forecast)
    nominal_forecast = device.pv_agent.get_hourly_forecast_pv(day)
    # Get the error standard deviation (an array of 24 values)
    pv_error_std = device.pv_agent.compute_hourly_error_std(day)
    z_alpha = 1.645  # for chance constraint
    # Compute the chance-constrained forecast for reference (nominal minus error margin)
    chance_forecast = (nominal_forecast * len(device.pv_columns)) - (z_alpha * pv_error_std)
    
    savings_samples = []
    violation_counts = []
    
    for i in range(num_simulations):
        # Sample forecast errors for each hour
        simulated_errors = np.random.normal(loc=0.0, scale=pv_error_std)
        # Create a simulated forecast: nominal forecast plus sampled errors
        simulated_forecast = nominal_forecast + simulated_errors
        # Option 1: Run MILP using the deterministic (perfect) forecast
        device_det = deepcopy(device)
        device_det.savings = 0.0  # reset savings
        _ = device_det.optimize_day(day, effective_prices, nominal_forecast, battery_state, grid_info)
        savings_det = device_det.savings

        # Option 2: Run MILP using the simulated (stochastic) forecast
        device_stoch = deepcopy(device)
        device_stoch.savings = 0.0  # reset savings
        _ = device_stoch.optimize_day(day, effective_prices, simulated_forecast, battery_state, grid_info)
        savings_stoch = device_stoch.savings

        # Record the simulated savings; here we choose the stochastic outcome.
        savings_samples.append(savings_stoch)
        # (If your optimize_day() returns violation counts or you record them in the device, record them here.)
        violation_counts.append(0)  # Placeholder; replace with actual violation count if available.
    
    return savings_samples, violation_counts

    
class GlobalOptimizer:
    def __init__(self,
                 devices: List[FlexibleDevice],
                 global_layer: GlobalConnectionLayer,
                 pv_agent: Optional[Any] = None,
                 weather_agent: Optional[Any] = None,
                 battery_agent: Optional[Any] = None,
                 ev_agent: Optional[Any] = None,
                 grid_agent: Optional[Any] = None,
                 max_iterations: int = 1,
                 online_iterations: int = 3,
                 solver=None):
        if solver is None:
            solver = PULP_CBC_CMD(msg=0)
            
        self.devices = devices
        self.global_layer = global_layer
        self.pv_agent = pv_agent
        self.weather_agent = weather_agent
        self.battery_agent = battery_agent
        self.ev_agent = ev_agent
        self.grid_agent = grid_agent
        self.max_iterations = max_iterations
        self.online_iterations = online_iterations
        self.solver = solver
        self.iteration_history = []
        self.best_iteration = None
        self.best_savings = None
        
        # results holders – filled after optimisation
        if self.battery_agent:
            # Initialize with a default size, will be resized later if needed
            data_length = 24 # Default, will be updated in optimize method
            if self.devices and len(self.devices[0].data) > 0:
                data_length = len(self.devices[0].data)
            self.battery_charge_global = np.zeros(data_length)
            self.battery_discharge_global = np.zeros(data_length)
            self.battery_soc_global = np.full(data_length, self.battery_agent.current_soc if self.battery_agent else 0.0)
        else:
            self.battery_charge_global = np.array([])
            self.battery_discharge_global = np.array([])
            self.battery_soc_global = np.array([])
        
        self.ev_charge_global = None
        self.ev_discharge_global = None
        self.ev_soc_global = None
        
        # Store the solved schedules in the respective agents for later analysis
        if self.battery_agent:
            self.battery_agent.charge_schedule = self.battery_charge_global
            self.battery_agent.discharge_schedule = self.battery_discharge_global

    def optimize(self):
        """
        Offline multi-iteration pass (partial usage MILP) with battery state sharing.
        - Initializes global battery arrays if a battery is available.
        - For each device and for each day, passes the updated battery state (SOC, charge, discharge)
        to subsequent devices.
        - Includes debug print statements to trace the battery state.
        
        Returns:
            The iteration history list for compatibility with other methods
        """
        previous_limit_hits = float('inf')
        best_savings = -float('inf')
        
        # Get the maximum data length for battery arrays
        data_length = 24  # Default fallback
        if len(self.devices) > 0:
            data_length = len(self.devices[0].data)
        
        # CRITICAL FIX: Track hour-by-hour battery availability
        # Initialize global battery arrays
        if self.battery_agent is not None:
            battery_soc_global = np.full(data_length, self.battery_agent.current_soc)
            battery_charge_global = np.zeros(data_length)
            battery_discharge_global = np.zeros(data_length)
            
            # NEW: Create a dictionary to track hourly planned charging/discharging
            # This will be used to prevent double-booking the battery
            hourly_battery_plans = {}
            
            # Initialize for all days and hours
            all_days = set()
            for dev in self.devices:
                all_days.update(dev.data['day'].unique())
            
            for day in all_days:
                hourly_battery_plans[day] = {hour: {'charge': 0.0, 'discharge': 0.0} for hour in range(24)}
        else:
            battery_soc_global = battery_charge_global = battery_discharge_global = None
            hourly_battery_plans = None
        
        for iteration in range(self.max_iterations):
            iteration_results = {
                'limit_hits': 0, 
                'total_shifts': 0,
                'successful_shifts': 0, 
                'savings': 0.0
            }
            for dev in self.devices:
                dev.savings = 0.0
            
            self.broadcast_preferred_hours()
            grid_info = self.grid_agent.get_grid_info() if self.grid_agent else None
            full_prices = self.devices[0].data['price_per_kwh'].values  # assuming same price structure for all
            
            # Sort devices in descending order of average baseline consumption
            self.devices.sort(key=lambda d: np.mean(d.original_consumption), reverse=True)
            
            for dev in self.devices:
                # print(f"[GlobalOptimizer] Starting optimization for device {dev.device_name}")
                for day in dev.data['day'].unique():
                    day_idxs = dev.data[dev.data['day'] == day].index
                    if len(day_idxs) != 24:
                        # print(f"[GlobalOptimizer] Incomplete data for {dev.device_name} on {day}. Skipping day.")
                        continue

                    # CRITICAL FIX: Get basic battery state
                    battery_state = None
                    if self.battery_agent is not None:
                        battery_state = self.battery_agent.get_battery_state()
                        
                        # NEW: Add hour-specific battery availability information
                        # This prevents double-booking the battery across devices
                        battery_state['hourly_plan'] = hourly_battery_plans[day]
                    
                    pv_forecast = self.pv_agent.get_hourly_forecast_pv(day) if self.pv_agent is not None else None
                    shifts = dev.optimize_day(day, full_prices, pv_forecast, battery_state, grid_info)
                    iteration_results['total_shifts'] += len(shifts)
                    iteration_results['successful_shifts'] += sum(1 for s in shifts if s.get('success', False))

                    # store the daily penalty into a dictionary
                    if not hasattr(dev, 'forecast_error_penalty_daily'):
                        dev.forecast_error_penalty_daily = {}
                
                    # dev.forecast_error_penalty is whatever you computed inside optimize_day
                    dev.forecast_error_penalty_daily[day] = getattr(dev, 'forecast_error_penalty', 0.0)

                    # CRITICAL FIX: Update the hourly battery plans
                    if self.battery_agent is not None:
                        day_hours = self.data[self.data['day'] == day]['hour'].values if hasattr(self, 'data') else list(range(24))
                        
                        try:
                            # Update hourly battery plans with this device's decisions
                            for i, idx in enumerate(day_idxs):
                                hour = day_hours[i] if i < len(day_hours) else i % 24
                                
                                # Get this device's battery decisions for this hour
                                charge_val = dev.battery_charge[idx] if idx < len(dev.battery_charge) else 0.0
                                discharge_val = dev.battery_discharge[idx] if idx < len(dev.battery_discharge) else 0.0
                                
                                # Update the hourly plan
                                hourly_battery_plans[day][hour]['charge'] += charge_val
                                hourly_battery_plans[day][hour]['discharge'] += discharge_val
                                
                                # Also update the global arrays for consistency
                                if idx < len(battery_soc_global):
                                    battery_soc_global[idx] = dev.battery_soc[idx]
                                    battery_charge_global[idx] += charge_val
                                    battery_discharge_global[idx] += discharge_val
                            
                            # Get final SoC for this day if possible
                            if len(day_idxs) > 0 and day_idxs[-1] < len(battery_soc_global):
                                final_soc_day = dev.battery_soc[day_idxs[-1]]
                                # print(f"[GlobalOptimizer] Updated battery state for {dev.device_name} on {day}: Final SOC = {final_soc_day:.2f} kWh")
                        except Exception as e:
                            print(f"Error updating battery state: {e}")
                                
                    if self.battery_agent is not None:
                        self.battery_agent.update_daily_soc()
                        # Validate battery coordination
                        self.validate_battery_coordination(day, hourly_battery_plans)

                    iteration_results['savings'] += dev.savings
                    dev.iteration_consumption[iteration] = dev.optimized_consumption.copy()
                    for hour in range(24):
                        hidxs = dev.data.index[dev.data['hour'] == hour]
                        hour_load = dev.optimized_consumption[hidxs].mean() if len(hidxs) > 0 else 0.0
                        self.global_layer.update_load(hour, hour_load, add=True, device_name=dev.device_name)
                
                iteration_results['limit_hits'] = len(self.global_layer.limit_hits)
                eval_metrics = self.evaluate_performance()
                iteration_results.update(eval_metrics)
                self.iteration_history.append(iteration_results)
                
                if iteration_results['savings'] > best_savings:
                    best_savings = iteration_results['savings']
                    self.best_iteration = iteration
                    self.best_savings = iteration_results
                
                # print(f"[GlobalOptimizer] Iteration {iteration}: {iteration_results}")
                self.analyze_iteration(iteration_results, previous_limit_hits)
                
                previous_limit_hits = iteration_results['limit_hits']
                self.global_layer.limit_hits.clear()
                self.global_layer.hourly_load = np.zeros(self.global_layer.hourly_load.shape)
            
            for dev in self.devices:
                dev.offline_savings = dev.savings
            
            if self.battery_agent is not None:
                self.battery_soc_global = battery_soc_global.copy()
                self.battery_charge_global = battery_charge_global.copy()
                self.battery_discharge_global = battery_discharge_global.copy()
                # Ensure these array attributes are available for other methods
                self.battery_soc = self.battery_soc_global
                self.battery_charge = self.battery_charge_global
                self.battery_discharge = self.battery_discharge_global
                # print(f"[GlobalOptimizer] Final global SOC array: {self.battery_soc_global}")
            
            self.save_results("full_optimization.pkl")
            # print("[GlobalOptimizer] Offline results saved.")
            
            # Return the iteration history
            return self.iteration_history

    def optimize_centralized(self):
        """
        Centralized optimization that handles partial day data by filling
        missing hours with 0 consumption so that each device has a 24-element array.
        This fixes the problem where the old code skipped days if any device had <24 rows.
        Now with integrated battery modeling for optimal coordination.
        """
        from pulp import (LpProblem, LpVariable, LpMinimize, lpSum, PULP_CBC_CMD, LpStatus)
        import logging
        import numpy as np

        all_days = set()
        for dev in self.devices:
            all_days |= set(dev.data['day'].unique())

        print(f"Found {len(all_days)} unique days to optimize")

        if not all_days:
            print("ERROR: No days found at all for centralized optimization.")
            for dev in self.devices:
                dev.centralized_optimized_schedule = None
            return

        # 2) Prepare each device's schedule array
        for dev in self.devices:
            print(f"Device: {dev.device_name}, is_flexible: {dev.is_flexible}")
            # CRITICAL FIX: Initialize centralized_optimized_schedule to match the data length
            # This ensures the array can handle all indices from the device data
            if hasattr(dev, 'data') and len(dev.data) > 0:
                # Initialize with zeros and copy original data
                dev.centralized_optimized_schedule = np.zeros(len(dev.data))
                # Copy original consumption, handling size differences
                min_len = min(len(dev.original_consumption), len(dev.data))
                dev.centralized_optimized_schedule[:min_len] = dev.original_consumption[:min_len]
                
                # Fill remaining with device data if available
                if hasattr(dev, 'device_name') and dev.device_name in dev.data.columns:
                    dev.centralized_optimized_schedule = dev.data[dev.device_name].values.copy()
            else:
                # Fallback: use original consumption
                dev.centralized_optimized_schedule = np.copy(dev.original_consumption)

        n_hours = 24

        # 3) Solve a separate MILP for each day in all_days
        for day_val in sorted(all_days):
            print(f"\nProcessing day: {day_val}")
            prob = LpProblem(f"Centralized_Optimization_{day_val}", LpMinimize)
            
            x = {}            # shift variables
            cost_terms = []   # objective
            device_indices = {}
            
            # Get battery state if battery agent is available
            battery_state = None
            if self.battery_agent is not None:
                battery_state = self.battery_agent.get_battery_state()
                print(f"Battery available with capacity: {battery_state['soc_max']} kWh, " 
                     f"current SOC: {battery_state['current_soc']} kWh")
            
            # Get PV data from PV agent if available
            pv_data = None
            pv_uncertainty = np.ones(n_hours) * 0.1  # Default 10% uncertainty
            if self.pv_agent is not None:
                pv_data = self.pv_agent.get_hourly_forecast_pv(day_val)
                # Debug output for PV data
                if pv_data is not None and len(pv_data) > 0:
                    print(f"PV forecast available: {len(pv_data)} hours, max generation: {-np.min(pv_data):.3f} kWh")
                print(f"Using PV forecast for day {day_val}")
                
                # Calculate PV forecast uncertainty if available
                if hasattr(self.pv_agent, 'compute_hourly_error_std'):
                    try:
                        pv_uncertainty = self.pv_agent.compute_hourly_error_std(day_val)
                    except Exception as e:
                        print(f"Could not compute PV forecast uncertainty: {e}")
            
            # Build partial day data for each device
            for d_idx, dev in enumerate(self.devices):
                # Check which rows belong to this day
                day_mask = (dev.data['day'] == day_val)
                
                # We build consumption_24 of length 24, filling missing hour(s) with 0
                if not np.any(day_mask):
                    # This device has no rows at all for day_val => 0 usage
                    consumption_24 = np.zeros(n_hours)
                    prices_24 = np.zeros(n_hours)
                    print(f"  {dev.device_name}: No data for this day")
                else:
                    df_day = dev.data[day_mask].copy()
                    # We want exactly 24 entries. If some hours missing, fill them with 0
                    hour_consumption = {hr: 0.0 for hr in range(n_hours)}
                    hour_prices = {hr: 0.0 for hr in range(n_hours)}

                    # Gather actual data
                    for idx in df_day.index:
                        hr = df_day.loc[idx, 'hour']
                        # Use the hour as index into original_consumption, not the DataFrame index
                        # This assumes original_consumption is already set up as a 24-hour array
                        val = dev.original_consumption[hr]
                        pr = df_day.loc[idx, 'price_per_kwh']
                        hour_consumption[hr] = val
                        hour_prices[hr] = pr

                    consumption_24 = np.array([hour_consumption[h] for h in range(n_hours)])
                    prices_24 = np.array([hour_prices[h] for h in range(n_hours)])
                    
                    nonzero_hours = np.sum(consumption_24 > 0)
                    print(f"  {dev.device_name}: Found {nonzero_hours} hours with consumption > 0")

                # If the device is fully missing (no coverage), it's all zeros
                # Store references for building the MILP
                device_indices[dev] = (d_idx, consumption_24, prices_24)

            # Get prices for the day (use the first device with prices)
            day_prices = None
            for dev, (_, _, prices_24) in device_indices.items():
                if np.any(prices_24 > 0):
                    day_prices = prices_24
                    break
            
            if day_prices is None or np.all(day_prices == 0):
                print(f"WARNING: No price data available for day {day_val}. Skipping optimization.")
                continue

            # Export price is determined by a factor from the config, reflecting a realistic scenario
            export_price = [p * self.grid_agent.export_price_factor for p in day_prices]

            # --- Battery Variables ---
            # Create battery variables if battery_state is provided
            if battery_state is not None:
                print("Setting up battery variables and constraints...")
                charge = LpVariable.dicts("charge", range(n_hours), lowBound=0)
                discharge = LpVariable.dicts("discharge", range(n_hours), lowBound=0)
                soc = LpVariable.dicts("soc", range(n_hours),
                                     lowBound=battery_state['soc_min'],
                                     upBound=battery_state['soc_max'])
                
                # Binary variable y[t]: y[t]=1 means charging, y[t]=0 means discharging
                y = LpVariable.dicts("y", range(n_hours), cat="Binary")
                
                # Add all battery constraints using the centralized function
                if self.battery_agent is not None:
                    prob, battery_cost_terms = self.battery_agent.add_battery_constraints_to_milp(
                        prob=prob,
                        battery_state=battery_state,
                        n_periods=n_hours,
                        charge=charge,
                        discharge=discharge,
                        soc=soc,
                        prices=day_prices,
                        y=y,
                        cost_terms=cost_terms,
                        force_arbitrage=False,
                        problem_type="centralized",
                        name_prefix="Batt"
                    )
                    
                    # Update cost_terms with battery-related costs
                    cost_terms = battery_cost_terms
            
            # --- EV Variables ---
            # Create EV variables if EV agent is available
            ev_state = None
            ev_charge = None
            ev_discharge = None
            ev_soc = None
            
            if self.ev_agent is not None:
                print("Setting up EV variables and constraints...")
                ev_state = self.ev_agent.get_battery_state()
                ev_charge = LpVariable.dicts("EV_charge", range(n_hours), lowBound=0, 
                                          upBound=self.ev_agent.max_charge_rate)
                # EV discharge is optional (V2G capability)
                max_discharge = getattr(self.ev_agent, 'max_discharge_rate', 0.0)
                ev_discharge = LpVariable.dicts("EV_discharge", range(n_hours), lowBound=0, 
                                             upBound=max_discharge)
                ev_soc = LpVariable.dicts("EV_soc", range(n_hours),
                                       lowBound=ev_state['soc_min'],
                                       upBound=ev_state['soc_max'])
                
                # Binary variable for EV charging mode
                ev_y = LpVariable.dicts("EV_y", range(n_hours), cat="Binary")
                
                # Add EV constraints using the same function as battery
                prob, ev_cost_terms = self.ev_agent.add_battery_constraints_to_milp(
                    prob=prob,
                    battery_state=ev_state,
                    n_periods=n_hours,
                    charge=ev_charge,
                    discharge=ev_discharge,
                    soc=ev_soc,
                    prices=day_prices,
                    y=ev_y,
                    cost_terms=cost_terms,
                    force_arbitrage=False,  # Don't force arbitrage for EV
                    problem_type="centralized",
                    name_prefix="EV"
                )
                
                # Update cost_terms with EV-related costs
                cost_terms = ev_cost_terms
                
                print(f"EV constraints added. EV must be charged to {ev_state['soc_max']*0.98:.2f} kWh by hour {self.ev_agent.must_be_full_by_hour}")
            
            # Store these for later use when building the overall net load expressions
            pv_incentive_hours = []
            pv_incentive_values = []
            
            # Calculate PV incentive information for reporting
            pv_hours_available = 0
            pv_total_generation = 0.0
            if pv_data is not None:
                for h in range(len(pv_data)):
                    if pv_data[h] < -0.05:  # Threshold for significant PV generation
                        pv_hours_available += 1
                        pv_total_generation += -pv_data[h]  # Convert to positive value
                
                if pv_hours_available > 0:
                    print(f"PV available in {pv_hours_available} hours, total generation: {pv_total_generation:.2f} kWh")
            
            # 4) Now build the shift variables for all devices
            print("\nCreating shift variables...")
            for dev, (d_idx, consumption_24, prices_24) in device_indices.items():
                shift_vars_created = 0
                
                # Skip if the device is not flexible or has no data for this day
                if not dev.is_flexible or not np.any(consumption_24 > 0):
                    print(f"  {dev.device_name}: Skipping optimization (not flexible or no consumption)")
                    continue
                # Skip battery/storage devices - they're handled separately
                if hasattr(dev, 'get_battery_state'):
                    print(f"  {dev.device_name}: Skipping shift variables (handled as storage device)")
                    continue

                # print(f"  {dev.device_name}: Creating shift variables")
                max_shift = dev.max_shift_hours
                
                # IMPORTANT: Check if there's price variation in this day
                price_range = np.max(prices_24) - np.min(prices_24) if len(prices_24) > 0 else 0
                if price_range <= 0.0001:  # Effectively no price variation
                    print(f"    WARNING: No price variation for {dev.device_name} on {day_val}, skipping optimization")
                    continue
                
                # Create shifting variables x[d_idx, t, h]
                for t in range(n_hours):
                    # Only create shift variables for hours with consumption
                    if consumption_24[t] > 0:
                        for h in range(-max_shift, max_shift + 1):
                            target = t + h
                            if 0 <= target < n_hours:
                                # Create safe variable names with no negative signs
                                var_name = f"x_{d_idx}_{t}_{'p' if h>=0 else 'm'}{abs(h)}"
                                x[(d_idx, t, h)] = LpVariable(var_name, lowBound=0, upBound=1)
                                # Store the key for later reference
                                shift_key = (d_idx, t, h)
                                
                                # Initialize with regular price
                                eff_price = prices_24[target]
                                
                                # Apply a strong discount during PV generation hours
                                if pv_data is not None and target < len(pv_data):
                                    pv_generation = abs(pv_data[target])  # Convert to positive value
                                    
                                    # If we have significant PV generation, apply discount
                                    if pv_generation > 0.001:  # Minimum threshold
                                        pv_peak = max(0.1, -pv_data.min())  # Avoid divide-by-zero
                                        pv_ratio = pv_generation / pv_peak  # 0-1 ratio of current to peak
                                        
                                        # MODIFIED DISCOUNT CALCULATION: Make discount less dependent on price differential
                                        # and more directly tied to PV generation to improve PV utilization
                                        
                                        # Get the base price and the minimum price in the day
                                        base_price = prices_24[target]
                                        min_price = min(prices_24)
                                        
                                        # Calculate a more substantial discount based on absolute pricing advantage
                                        # plus a fixed incentive, rather than relative to export price
                                        discount = (base_price - min_price + 0.05) * pv_ratio * PV_WEIGHT
                                        
                                        # Apply discount to make PV hours more attractive
                                        eff_price = prices_24[target] - discount
                                # -----------------------------------------------------------------
                                
                                # CRITICAL FIX: Use effective price that accounts for PV availability
                                # This incentivizes shifting to PV generation hours
                                cost_terms.append(eff_price * consumption_24[t] * x[shift_key])
                                shift_vars_created += 1

                        # The sum of shifts for hour t must equal 1 (100% of consumption is allocated)
                        relevant_vars = [x[(d_idx, t, hh)]
                                        for hh in range(-max_shift, max_shift+1)
                                        if (d_idx, t, hh) in x]
                        if relevant_vars:
                            # Use safe variable naming for constraints
                            prob += lpSum(relevant_vars) == 1, f"Conservation_{d_idx}_{dev.device_name.replace('-','_')}_t{t}"
                
                if shift_vars_created > 0:
                    print(f"    Created {shift_vars_created} shift variables for {dev.device_name}")

            # 5) Enforce building load each hour with battery integration
            print("\nEnforcing building load constraints...")
            net_loads = {}
            for hour in range(n_hours):
                hour_load_list = []
                
                # Add loads from discrete-phase devices
                for dev, (d_idx, consumption_24, prices_24) in device_indices.items():
                    # Skip non-flexible devices
                    if not dev.is_flexible:
                        continue
                        
                    for t in range(n_hours):
                        if consumption_24[t] > 0:  # Only consider hours with consumption
                            for h in range(-max_shift, max_shift+1):
                                # Use safe tuple keys that won't cause issues with CBC
                                key = (d_idx, t, h)
                                if key in x and (t + h) == hour:
                                    hour_load_list.append(x[key] * consumption_24[t])
                
                # Add building load constraint integrating the battery and EV if available
                if hour_load_list or battery_state is not None or ev_state is not None:  # Only add constraint if there are components to consider
                    load_sum = lpSum(hour_load_list) if hour_load_list else 0
                    
                    # Add battery charge/discharge impact if available
                    if battery_state is not None:
                        load_sum += charge[hour] - discharge[hour]
                        
                    # Add EV charge/discharge impact if available
                    if ev_state is not None and ev_charge is not None:
                        load_sum += ev_charge[hour] - ev_discharge[hour]
                        
                    # Store net load 
                    net_loads[hour] = load_sum
                        
                    # Add the constraint with all components
                    prob += load_sum <= self.global_layer.max_building_load, f"BuildingLoad_hour_{hour}"
                    
            # No separate PV self-utilization incentive needed - 
            # we're now directly modifying the hourly prices in the cost function instead
            # This is more direct and avoids the complex feedback issues
            
            print(f"Added {len(prob.constraints)} constraints in total")

            # 6) Objective: sum of cost terms
            num_cost_terms = len(cost_terms)
            print(f"\nObjective function has {num_cost_terms} cost terms")
            
            if cost_terms:  # Only set objective if there are cost terms
                prob += lpSum(cost_terms), "TotalCost"
                
                # 7) Solve only if there are variables and constraints to optimize
                if len(prob.variables()) > 0 and len(prob.constraints) > 0:
                    print(f"Solving problem with {len(prob.variables())} variables and {len(prob.constraints)} constraints...")
                    
                    # Enable extensive debugging
                    import tempfile, os
                    from pathlib import Path
                    from pulp import COIN_CMD, PULP_CBC_CMD
                    
                    # Create a custom temp directory in the project instead of using system temp
                    # Find the project root directory
                    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
                    project_root = script_dir.parent.parent  # Go up two levels from agents/ folder
                    
                    # Create a temp dir inside the project
                    temp_dir = project_root / 'temp_optimizations'
                    os.makedirs(temp_dir, exist_ok=True)
                    
                    # Write LP file to custom directory
                    lp_path = os.path.join(temp_dir, 'debug_optimization.lp')
                    print(f"Writing LP file to {lp_path} for debugging")
                    try:
                        prob.writeLP(lp_path)
                    except Exception as write_error:
                        print(f"Warning: Failed to write LP file: {write_error}")
                        # Continue without the LP file if writing fails
                    
                    try:
                        # Try first with CBC CMD
                        solver = PULP_CBC_CMD(msg=False, options=['allowableGap', '0.01', 'ratioGap', '0.01'])
                        print("Attempting to solve with CBC CMD...")
                        prob.solve(solver)
                    except Exception as e:
                        print(f"CBC CMD failed: {e}")
                        try:
                            # Fall back to COIN CMD
                            print("Falling back to COIN CMD...")
                            solver = COIN_CMD(msg=True, keepFiles=True, path=None)
                            prob.solve(solver)
                        except Exception as e2:
                            print(f"COIN CMD also failed: {e2}")
                            # Last resort - try with default solver
                            print("Using PuLP default solver as last resort...")
                            prob.solve()
                    
                    status = LpStatus[prob.status]
                    print(f"Optimization status: {status}")
                    
                    # Get the solver's objective value directly
                    from pulp import value
                    solver_objective = value(prob.objective) if prob.status == 1 else None
                    print(f"Solver objective value: {solver_objective}")
                    
                    if status != "Optimal":
                        print(f"WARNING: day {day_val} not optimal (status={status}). Skipping fill.")
                        continue
                        
                    # Get PV data for this day if available
                    pv_data = None
                    if self.pv_agent is not None:
                        # For historical data, extract from profile_data if available
                        if hasattr(self.pv_agent, 'profile_data') and self.pv_agent.profile_data is not None:
                            # If profile_data has a 'day' column, filter by that
                            if 'day' in self.pv_agent.profile_data.columns:
                                day_mask = self.pv_agent.profile_data['day'] == day_val
                                if any(day_mask) and 'pv_summed' in self.pv_agent.profile_data.columns:
                                    pv_data = self.pv_agent.profile_data.loc[day_mask, 'pv_summed'].values
                            # Otherwise try to get forecast data for this day
                            elif hasattr(self.pv_agent, 'get_hourly_forecast_pv'):
                                try:
                                    # If day_val is a date-like object, use it directly
                                    pv_data = self.pv_agent.get_hourly_forecast_pv(day_val)
                                except:
                                    print(f"Could not get PV forecast for day {day_val}")
                                    pv_data = None
                    
                    # Evaluate costs for this specific day using the standardized helper
                    self._evaluate_costs_for_day(
                        day_idx=day_val,
                        price_vec=day_prices,
                        pv_vec=pv_data,
                        tag="continuous",
                        solver_obj=solver_objective
                    )
                    print(f"Day {day_val}: Costs evaluated and stored.")

                    # 8) Reconstruct schedules for each device
                    print("\nReconstructing optimized schedules...")
                    for dev, (d_idx, consumption_24, prices_24) in device_indices.items():
                        # Skip non-flexible devices or those with no consumption
                        if not dev.is_flexible or not np.any(consumption_24 > 0):
                            continue
                        
                        print(f"  {dev.device_name}: Processing optimal solution")    
                        # Build the optimized 24h array from the MILP solution
                        optimized_24 = np.zeros(n_hours)
                        shifts_applied = 0
                        
                        for t in range(n_hours):
                            if consumption_24[t] > 0:  # Only consider hours with consumption
                                for h in range(-max_shift, max_shift+1):
                                    shift_key = (d_idx, t, h)
                                    if shift_key in x and 0 <= t+h < n_hours:
                                        var_val = x[shift_key].varValue or 0.0
                                        if var_val > 0.001:  # Only count meaningful shifts
                                            shifts_applied += 1
                                            target_hour = t + h
                                            opt_value = var_val * consumption_24[t]
                                            optimized_24[target_hour] += opt_value
                                            print(f"    Shift from hour {t} to hour {target_hour}: {var_val:.4f} * {consumption_24[t]:.4f} = {opt_value:.4f}")
                        
                        print(f"    Applied {shifts_applied} shifts")
                        print(f"    Original: {consumption_24}")
                        print(f"    Optimized: {optimized_24}")
                        
                        # Check if there's a meaningful difference
                        diff = np.sum(np.abs(consumption_24 - optimized_24))
                        print(f"    Difference magnitude: {diff:.6f}")

                        # CRITICAL FIX: If there's any meaningful difference, update the schedule
                        if diff > 0.001:
                            # Insert the optimized schedule back into the device's data
                            day_mask = (dev.data['day'] == day_val)
                            if np.any(day_mask):
                                day_indexes = dev.data[day_mask].index
                                
                                # Ensure centralized_optimized_schedule is properly sized
                                if len(dev.centralized_optimized_schedule) < len(dev.data):
                                    # Resize the array to match data length
                                    new_schedule = np.zeros(len(dev.data))
                                    new_schedule[:len(dev.centralized_optimized_schedule)] = dev.centralized_optimized_schedule
                                    dev.centralized_optimized_schedule = new_schedule
                                
                                for row_idx in day_indexes:
                                    hr = dev.data.loc[row_idx, 'hour']
                                    old_val = dev.centralized_optimized_schedule[row_idx]
                                    new_val = optimized_24[hr]
                                    dev.centralized_optimized_schedule[row_idx] = new_val
                                    if abs(old_val - new_val) > 0.001:
                                        print(f"    Index {row_idx}, Hour {hr}: Changed from {old_val:.6f} to {new_val:.6f}")
                        else:
                            print(f"    No significant changes for {dev.device_name} on {day_val}")
                    
                    # Process battery results if available
                    if battery_state is not None:
                        print("\nProcessing battery optimization results...")
                        # Initialize battery arrays if they don't exist already
                        if not hasattr(self, 'battery_soc_global'):
                            data_length = len(self.devices[0].data) if self.devices else 24
                            self.battery_soc_global = np.full(data_length, battery_state['current_soc'])
                            self.battery_charge_global = np.zeros(data_length)
                            self.battery_discharge_global = np.zeros(data_length)
                        
                        # Get day indices for storing the battery results
                        day_indices = []
                        if self.devices:
                            dev = self.devices[0]  # Use the first device as reference
                            day_mask = (dev.data['day'] == day_val)
                            day_indices = dev.data[day_mask].index.tolist()
                        
                        # Extract battery variable values
                        for t in range(n_hours):
                            soc_val = soc[t].varValue if hasattr(soc[t], 'varValue') and soc[t].varValue is not None else battery_state['current_soc']
                            charge_val = charge[t].varValue if hasattr(charge[t], 'varValue') and charge[t].varValue is not None else 0.0
                            discharge_val = discharge[t].varValue if hasattr(discharge[t], 'varValue') and discharge[t].varValue is not None else 0.0
                            
                            print(f"  Hour {t}: SOC={soc_val:.2f}, Charge={charge_val:.2f}, Discharge={discharge_val:.2f}")
                            
                            # Update global arrays if we have the day indices
                            if t < len(day_indices):
                                idx = day_indices[t]
                                if idx < len(self.battery_soc_global):
                                    self.battery_soc_global[idx] = soc_val
                                    self.battery_charge_global[idx] = charge_val
                                    self.battery_discharge_global[idx] = discharge_val
                        
                        # Update battery agent state if available
                        if self.battery_agent is not None:
                            # Calculate battery throughput for the day
                            day_charge = sum(charge[t].varValue or 0 for t in range(n_hours))
                            day_discharge = sum(discharge[t].varValue or 0 for t in range(n_hours))
                            day_throughput = day_charge + day_discharge
                            
                            # Update battery cycle count
                            self.battery_agent.cycle_count += day_throughput / self.battery_agent.estimated_capacity
                            
                            # Update history arrays
                            self.battery_agent.charge_history.append(day_charge)
                            self.battery_agent.discharge_history.append(day_discharge)
                            
                            final_soc = soc[n_hours-1].varValue if hasattr(soc[n_hours-1], 'varValue') and soc[n_hours-1].varValue is not None else battery_state['current_soc']
                            # CRITICAL FIX: Ensure SOC stays within bounds to prevent infeasible subsequent days
                            final_soc = min(max(final_soc, self.battery_agent.soc_min), self.battery_agent.soc_max)
                            self.battery_agent.current_soc = final_soc
                            # Only append to soc_history if this is the first time we're seeing this day
                            if not hasattr(self.battery_agent, '_last_updated_day') or self.battery_agent._last_updated_day != day_val:
                                self.battery_agent.soc_history.append(final_soc)
                                self.battery_agent._last_updated_day = day_val
                            
                            # Populate hourly arrays for battery (ensure 24-hour arrays)
                            for t in range(24):
                                if t < n_hours:
                                    self.battery_agent.hourly_charge[t] = charge[t].varValue if hasattr(charge[t], 'varValue') and charge[t].varValue is not None else 0.0
                                    self.battery_agent.hourly_discharge[t] = discharge[t].varValue if hasattr(discharge[t], 'varValue') and discharge[t].varValue is not None else 0.0
                                    self.battery_agent.hourly_soc[t] = soc[t].varValue if hasattr(soc[t], 'varValue') and soc[t].varValue is not None else self.battery_agent.current_soc
                                else:
                                    # Pad remaining hours with zeros for charge/discharge, maintain last SOC
                                    self.battery_agent.hourly_charge[t] = 0.0
                                    self.battery_agent.hourly_discharge[t] = 0.0
                                    self.battery_agent.hourly_soc[t] = self.battery_agent.hourly_soc[t-1] if t > 0 else self.battery_agent.current_soc
                            
                            # CRITICAL FIX: Store the solved schedules in the agent for analysis
                            solved_battery_charge = np.array([charge[t].varValue if hasattr(charge[t], 'varValue') and charge[t].varValue is not None else 0.0 for t in range(n_hours)])
                            solved_battery_discharge = np.array([discharge[t].varValue if hasattr(discharge[t], 'varValue') and discharge[t].varValue is not None else 0.0 for t in range(n_hours)])
                            self.battery_agent.charge_schedule = solved_battery_charge
                            self.battery_agent.discharge_schedule = solved_battery_discharge

                            print(f"  Updated battery agent: SOC={final_soc:.2f}, day charge={day_charge:.2f}, day discharge={day_discharge:.2f}")
                    
                    # Process EV results if available
                    if ev_state is not None and self.ev_agent is not None:
                        print("\nProcessing EV optimization results...")
                        # Arrays are already initialized at the beginning of the method
                        
                        # Get day indices for storing the EV results
                        ev_day_indices = []
                        if self.devices:
                            dev = self.devices[0]  # Use the first device as reference
                            day_mask = (dev.data['day'] == day_val)
                            ev_day_indices = dev.data[day_mask].index.tolist()
                        
                        # Extract EV variable values
                        for t in range(n_hours):
                            ev_soc_val = ev_soc[t].varValue if hasattr(ev_soc[t], 'varValue') and ev_soc[t].varValue is not None else ev_state['current_soc']
                            ev_charge_val = ev_charge[t].varValue if hasattr(ev_charge[t], 'varValue') and ev_charge[t].varValue is not None else 0.0
                            ev_discharge_val = ev_discharge[t].varValue if hasattr(ev_discharge[t], 'varValue') and ev_discharge[t].varValue is not None else 0.0
                            
                            print(f"  Hour {t}: EV SOC={ev_soc_val:.2f}, Charge={ev_charge_val:.2f}, Discharge={ev_discharge_val:.2f}")
                            
                            # Update global arrays if we have the day indices
                            if t < len(ev_day_indices):
                                idx = ev_day_indices[t]
                                if idx < len(self.ev_soc_global):
                                    self.ev_soc_global[idx] = ev_soc_val
                                    self.ev_charge_global[idx] = ev_charge_val
                                    self.ev_discharge_global[idx] = ev_discharge_val
                        
                        # Update EV agent state if available
                        # Calculate EV throughput for the day
                        day_charge = sum(ev_charge[t].varValue or 0 for t in range(n_hours))
                        day_discharge = sum(ev_discharge[t].varValue or 0 for t in range(n_hours))
                        day_throughput = day_charge + day_discharge
                        
                        # Populate hourly arrays for EV (ensure 24-hour arrays)
                        for t in range(24):
                            if t < n_hours:
                                self.ev_agent.hourly_charge[t] = ev_charge[t].varValue if hasattr(ev_charge[t], 'varValue') and ev_charge[t].varValue is not None else 0.0
                                self.ev_agent.hourly_discharge[t] = ev_discharge[t].varValue if hasattr(ev_discharge[t], 'varValue') and ev_discharge[t].varValue is not None else 0.0
                                self.ev_agent.hourly_soc[t] = ev_soc[t].varValue if hasattr(ev_soc[t], 'varValue') and ev_soc[t].varValue is not None else self.ev_agent.current_soc
                            else:
                                # Pad remaining hours with zeros for charge/discharge, maintain last SOC
                                self.ev_agent.hourly_charge[t] = 0.0
                                self.ev_agent.hourly_discharge[t] = 0.0
                                self.ev_agent.hourly_soc[t] = self.ev_agent.hourly_soc[t-1] if t > 0 else self.ev_agent.current_soc
                        
                        # Update EV cycle count if attribute exists
                        if hasattr(self.ev_agent, 'cycle_count'):
                            self.ev_agent.cycle_count += day_throughput / (self.ev_agent.estimated_capacity or self.ev_agent.soc_max)
                        
                        # Update history arrays
                        if not hasattr(self.ev_agent, 'charge_history'):
                            self.ev_agent.charge_history = []
                        if not hasattr(self.ev_agent, 'discharge_history'):
                            self.ev_agent.discharge_history = []
                        if not hasattr(self.ev_agent, 'soc_history'):
                            self.ev_agent.soc_history = []
                            
                        self.ev_agent.charge_history.append(day_charge)
                        self.ev_agent.discharge_history.append(day_discharge)
                        
                        # Update current SOC
                        final_soc = ev_soc[n_hours-1].varValue if hasattr(ev_soc[n_hours-1], 'varValue') and ev_soc[n_hours-1].varValue is not None else ev_state['current_soc']
                        self.ev_agent.current_soc = final_soc
                        # Only append to soc_history if this is the first time we're seeing this day
                        if not hasattr(self.ev_agent, '_last_updated_day') or self.ev_agent._last_updated_day != day_val:
                            self.ev_agent.soc_history.append(final_soc)
                            self.ev_agent._last_updated_day = day_val
                        
                        # Push optimization results to EV agent for later access
                        self.ev_agent.hourly_charge = np.array([ev_charge[t].varValue or 0 for t in range(n_hours)])
                        self.ev_agent.hourly_discharge = np.array([ev_discharge[t].varValue or 0 for t in range(n_hours)])
                        self.ev_agent.hourly_soc = np.array([ev_soc[t].varValue or ev_state['current_soc'] for t in range(n_hours)])
                        
                        print(f"  Updated EV agent: SOC={final_soc:.2f}, day charge={day_charge:.2f}, day discharge={day_discharge:.2f}")
                else:
                    print(f"WARNING: No variables or constraints to optimize for day {day_val}.")

        # After optimization, make optimized_consumption point to centralized_optimized_schedule
        print("\nFinalizing optimization...")
        for dev in self.devices:
            if hasattr(dev, 'centralized_optimized_schedule'):
                # Check if there's a difference between original and optimized
                diff = np.sum(np.abs(dev.original_consumption - dev.centralized_optimized_schedule))
                print(f"{dev.device_name}: Difference between original and optimized: {diff:.6f}")
                
                # Update the device's optimized_consumption to match centralized_optimized_schedule
                dev.optimized_consumption = dev.centralized_optimized_schedule.copy()

        print("\nCentralized optimization complete with battery modeling!")
        
        # Calculate and store costs for each device
        print("\nCalculating and storing optimization costs...")
        for dev in self.devices:
            # Calculate original cost directly from optimization data
            price = dev.data['price_per_kwh'].values
            orig_cost = np.sum(dev.original_consumption * price)
            
            # Calculate optimized cost using centralized_optimized_schedule
            if hasattr(dev, 'centralized_optimized_schedule') and dev.centralized_optimized_schedule is not None:
                opt_cost = np.sum(dev.centralized_optimized_schedule * price)
            else:
                opt_cost = orig_cost  # No optimization occurred
                
            # Store these values directly on the device object
            dev.optimizer_original_cost = orig_cost
            dev.optimizer_optimized_cost = opt_cost
            dev.optimizer_savings = orig_cost - opt_cost
            
            # Include any battery-related costs for this device if applicable
            if hasattr(dev, 'battery_charge') and dev.battery_charge is not None and any(dev.battery_charge > 0):
                # Calculate battery degradation costs
                degrade_rate = getattr(dev.battery_agent, 'degradation_rate', 0.0) if hasattr(dev, 'battery_agent') else 0.0
                degradation_cost = np.sum(degrade_rate * (dev.battery_charge + dev.battery_discharge))
                dev.optimizer_battery_degradation_cost = degradation_cost
                print(f"{dev.device_name}: Original=${orig_cost:.2f}, Optimized=${opt_cost:.2f}, Savings=${(orig_cost - opt_cost):.2f}, Battery degradation=${degradation_cost:.2f}")
            else:
                print(f"{dev.device_name}: Original=${orig_cost:.2f}, Optimized=${opt_cost:.2f}, Savings=${(orig_cost - opt_cost):.2f}")
        
        # The cost evaluation is now done for each day inside the loop above
        return True
        
    def optimize_for_weekday(self):
        """
        Next-day aggregator pass (online next-day MILP using discrete phases).
        For each device, pick a single actual historical day (matching tomorrow’s weekday and season),
        override the prices with tomorrow’s actual day-ahead prices, and run the next-day MILP.
        """
        tomorrow = datetime.date.today() + timedelta(days=1)
        tweekday = tomorrow.strftime("%A")
        tseason = get_season(tomorrow)
        logging.info(f"Next-day aggregator for {tweekday} in {tseason}...")
        
        battery_state = (self.battery_agent.get_battery_state()
                         if self.battery_agent else None)
        grid_info = (self.grid_agent.get_grid_info()
                     if self.grid_agent else None)
        
        # Load offline results as warm start
        self.load_results("full_optimization.pkl")
        logging.info("Loaded offline for warm start...")
        
        # Clear device savings
        for dev in self.devices:
            dev.savings = 0.0
        
        # Perform multiple passes if desired
        for i in range(self.online_iterations):
            logging.info(f"Online iteration {i+1} / {self.online_iterations}")
            
            # --- In optimize_for_weekday(), replace the filtering block with:
            for dev in self.devices:
                # Try to filter by desired weekday and season
                cdata = dev.data[(dev.data["weekday"] == tweekday) & (dev.data["season"] == tseason)]
                if cdata.empty:
                    logging.warning(f"No data for {dev.device_name} on {tweekday}-{tseason}. Using most recent day as fallback.")
                    valid_days = sorted(dev.data["day"].unique())
                    picked_day = valid_days[-1]
                    cdata = dev.data[dev.data["day"] == picked_day].copy()
                else:
                    valid_days = sorted(cdata["day"].unique())
                    picked_day = valid_days[-1]
                print(f"Using actual day={picked_day} for device={dev.device_name} next-day optimization.")
                logging.info(f"Using actual day={picked_day} for device={dev.device_name} next-day optimization.")

                day_data = cdata[cdata["day"] == picked_day].copy()
                day_data.sort_values("hour", inplace=True)
                if len(day_data) != 24:
                    logging.warning(f"Incomplete or missing hours in {picked_day} for {dev.device_name}.")
                    continue

                # Extract actual prices (do not aggregate)
                prices = day_data[['hour', 'price_per_kwh']]
                
                # Aggregate the optimized consumption from all available rows
                # Fixed version that avoids IndexError by using day_data directly
                opt_df = pd.DataFrame({
                    'hour': day_data['hour'].values,
                    'optimized': day_data[dev.device_name].values  # Use the original consumption as a fallback
                })
                
                # If the device has an optimized_consumption that matches the data length,
                # use it instead of the original consumption
                if hasattr(dev, 'optimized_consumption') and len(dev.optimized_consumption) == len(dev.data):
                    try:
                        # Safe indexing that won't cause IndexError
                        opt_values = []
                        for idx in day_data.index:
                            if 0 <= idx < len(dev.optimized_consumption):
                                opt_values.append(dev.optimized_consumption[idx])
                            else:
                                # Fallback to original consumption if index is out of bounds
                                opt_values.append(day_data.loc[idx, dev.device_name])
                        
                        # Only update if we successfully got values for all hours
                        if len(opt_values) == len(day_data):
                            opt_df['optimized'] = opt_values
                    except Exception as e:
                        print(f"Warning: Could not use optimized_consumption for {dev.device_name}: {e}")
                agg_opt = opt_df.groupby('hour')['optimized'].mean().reset_index()
                
                # Merge aggregated optimized consumption with the actual prices
                agg_data = pd.merge(agg_opt, prices, on='hour', how='left')
                
                # Set the warm start schedule
                dev.weekday_optimized_schedule = agg_data['optimized'].values
                
                # Continue with the rest of the next-day optimization…
                if self.pv_agent is not None:
                    pv_forecast = self.pv_agent.get_hourly_forecast_pv(picked_day)
                else:
                    pv_forecast = [0.0] * 24
                
                if self.weather_agent and self.pv_agent is not None:
                    weather_fc_dict = self.weather_agent.get_all_hourly_forecasts(picked_day)
                else:
                    weather_fc_dict = {}
                
                dev.optimize_aggregated_day(
                    agg_data=agg_data,
                    pv_profile=None,
                    pv_forecast=pv_forecast,
                    weather_forecasts=weather_fc_dict, 
                    battery_state=battery_state,
                    grid_info=grid_info
                )
                
                dev.nextday_schedule_date = (picked_day.strftime("%Y-%m-%d") if isinstance(picked_day, datetime.date) else str(picked_day))

        
        for dev in self.devices:
            dev.nextday_savings = dev.savings
            dev.savings = 0.0
        
        self.save_results("weekday_optimization.pkl")
        logging.info("Weekday results saved using a single real day from cdata!")

    def optimize_phases_centralized(self, devices, global_layer,
                                    pv_agent=None, battery_agent=None,
                                    ev_agent=None, grid_agent=None, weather_agent=None):
        """
        Performs a centralized optimization of all device phases together.
        
        Args:
            devices: List of FlexibleDevice objects
            global_layer: The GlobalConnectionLayer for load constraints
            pv_agent: Optional PVAgent for PV integration
            battery_agent: Optional BatteryAgent for battery integration
            grid_agent: Optional GridAgent for grid constraints
            weather_agent: Optional WeatherAgent for weather integration
            
        Returns:
            success: Boolean indicating whether optimization was successful
        """
        import numpy as np
        from pulp import LpProblem, LpVariable, LpMinimize, lpSum, PULP_CBC_CMD, LpStatus
        from agents.FlexibleDeviceAgent import calculate_preference_penalty

        print("Starting centralized next-day optimization...")
        
        # 1) Gather building-wide data
        if not devices:
            print("No devices provided for optimization")
            return False
        
        reference_device = devices[0]
        day = reference_device.data['day'].unique()[0]
        hours = np.arange(24)
        prices = reference_device.data \
                    .groupby('hour')['price_per_kwh'] \
                    .mean().reindex(hours, fill_value=0).values
        pv_forecast = None
        if pv_agent is not None:
            pv_forecast = pv_agent.get_hourly_forecast_pv(day)
        weather_forecasts = weather_agent.get_all_hourly_forecasts(day) if weather_agent else None
        
        # Battery state setup
        battery_state = None
        if battery_agent is not None:
            battery_state = battery_agent.get_battery_state()
        
        # EV state setup
        ev_state = None
        if ev_agent is not None:
            # Safety check to prevent errors when grid_agent is passed instead of ev_agent
            if hasattr(ev_agent, 'get_battery_state'):
                ev_state = ev_agent.get_battery_state()
            else:
                print("Warning: ev_agent parameter does not have get_battery_state method. No EV optimization will be performed.")
                # Argument may have been passed in the wrong order
                ev_agent = None
        
        # 2) Initialize the optimization problem
        prob = LpProblem("DeviceScheduling", LpMinimize)
        all_cost_terms = []
        
        # 3) Build variables for discrete-phase devices
        device_phase_vars = {}
        # 3b) Build variables for partial-usage devices
        device_alloc_vars = {}
        
        for dev_idx, device in enumerate(devices):
            if not device.is_flexible:
                continue
                
            # Get the device's flexibility model (default to discrete_phase if not specified)
            flex_model = device.spec.get("flex_model", "discrete_phase")
            allowed_hours = getattr(device, 'allowed_hours', list(range(24)))
            
            if flex_model == "discrete_phase":
                # Handle discrete-phase devices (existing code)
                if not hasattr(device, 'phases') or not device.phases:
                    continue
                    
                phases = device.phases
                P = len(phases)
                print(f"Processing {device.device_name}: discrete-phase model with {P} phases, allowed hours: {allowed_hours}")
                
                # Decision vars X[(dev_idx,p,k)] = 1 if phase p starts at hour k
                X = {}
                for p in range(P):
                    for k in range(24):
                        name = f"X_{dev_idx}_{p}_{k}"
                        X[(dev_idx, p, k)] = LpVariable(name, cat="Binary")
                        if k not in allowed_hours:
                            prob += X[(dev_idx, p, k)] == 0, f"NotAllowed_{dev_idx}_{p}_{k}"
                device_phase_vars[dev_idx] = X

                # a) Each phase p must start exactly once
                for p in range(P):
                    prob += lpSum(X[(dev_idx, p, k)] for k in range(24)) == 1, \
                            f"PhaseStart_{dev_idx}_{p}"
                # b) Enforce that phase p+1 follows immediately after p
                for p in range(P - 1):
                    dur_p = phases[p]["duration"]
                    prob += (
                        lpSum(k * X[(dev_idx, p + 1, k)] for k in range(24))
                        == lpSum((k + dur_p) * X[(dev_idx, p, k)] for k in range(24))
                    ), f"Consec_{dev_idx}_{p}"

                # Build cost terms for discrete-phase devices
                for p in range(P):
                    dur = phases[p]["duration"]
                    e_kwh = phases[p]["energy_kwh"]
                    for k in range(24):
                        # base cost over its run
                        # Convert duration to integer for the range function
                        run_hours = range(k, min(int(k + dur), 24))
                        base_cost = sum(prices[r] for r in run_hours) * e_kwh
                        # PV discount
                        discount = 0.0
                        if pv_forecast is not None and k < len(pv_forecast):
                            pot = min(abs(pv_forecast[k]) / 32704.0, 1.0)
                            discount = 0.4 / (1 + np.exp(-10 * (pot - 0.5)))
                        # weather factor
                        factor = 1.0
                        if weather_forecasts is not None:
                            for cname, arr in weather_forecasts.items():
                                v = arr[k]
                                if 'temperature' in cname.lower() and v < 25:
                                    factor -= 0.01 * (25 - v)
                        adj = base_cost * (1 - discount) * factor
                        on_penalty = 0.01
                        pref_pen = calculate_preference_penalty(
                            device, k, device.PREFERENCE_PENALTY_WEIGHT
                        )
                        all_cost_terms.append((adj + on_penalty + pref_pen) * X[(dev_idx, p, k)])
            
            elif flex_model == "partial_usage":
                # Handle partial-usage devices with continuous allocation
                print(f"Processing {device.device_name}: partial-usage model, allowed hours: {allowed_hours}")
                
                # For partial-usage devices, each phase represents a fixed amount of energy that needs to be allocated
                # The total energy requirement is the sum of all phases' energy requirements
                if not device.phases:
                    print(f"Warning: {device.device_name} has no phases defined. Skipping.")
                    continue
                    
                phases = device.phases
                num_phases = len(phases)
                total_energy = sum(ph["energy_kwh"] for ph in phases)
                
                if total_energy <= 0:
                    print(f"Warning: {device.device_name} has invalid energy requirement. Skipping.")
                    continue
                
                # Simply follow the phases as they are defined
                phase_energies = [ph["energy_kwh"] for ph in phases]
                print(f"{device.device_name}: Following phases in order: {phase_energies}")
                
                # Simply follow the literature approach for partial-usage devices:
                # - Total energy requirement must be met (sum of phase energies)
                # - Energy can be distributed freely across allowed hours
                # - Hourly allocation cannot exceed the device's power rating
                
                # Create continuous variables for hourly energy allocation
                X = {}  # X[t] = energy allocated in hour t
                
                # Create allocation variables
                for t in range(24):
                    X[t] = LpVariable(f"alloc_{dev_idx}_{t}", lowBound=0)
                    # Enforce zero allocation outside allowed hours
                    if t not in allowed_hours:
                        prob += X[t] == 0, f"NotAllowed_{dev_idx}_{t}"
                
                # Calculate total required energy from all phases
                total_energy_requirement = sum(ph["energy_kwh"] for ph in phases)
                
                # CONSTRAINT 1: Total energy allocation must meet the requirement
                prob += lpSum(X[t] for t in range(24)) == total_energy_requirement, f"TotalEnergy_{dev_idx}"
                
                # CONSTRAINT 2: Hourly allocation cannot exceed power rating
                # (Convert power rating from kW to kWh for a 1-hour period)
                for t in allowed_hours:
                    prob += X[t] <= device.power_rating, f"PowerLimit_{dev_idx}_{t}"
                
                # Store the variables and energy requirement
                device_alloc_vars[dev_idx] = (X, total_energy_requirement)
                
                # Energy requirement constraint already added above
                
                # Add cost terms for partial-usage devices
                for t in range(24):
                    # Base cost is price * power allocation
                    base_cost = prices[t]
                    
                    # Apply same discounts as for discrete devices
                    discount = 0.0
                    if pv_forecast is not None and t < len(pv_forecast):
                        max_pv = max(abs(pv_forecast)) if pv_forecast is not None and len(pv_forecast) > 0 else 1.0
                        pot = min(abs(pv_forecast[t]) / max_pv, 1.0) if max_pv > 0 else 0.0
                        discount = 0.4 / (1 + np.exp(-10 * (pot - 0.5)))
                    
                    # Weather factor (same as discrete devices)
                    factor = 1.0
                    if weather_forecasts is not None:
                        for cname, arr in weather_forecasts.items():
                            v = arr[t]
                            if 'temperature' in cname.lower() and v < 25:
                                factor -= 0.01 * (25 - v)
                    
                    # Calculate adjusted cost
                    adj_cost = base_cost * (1 - discount) * factor
                    
                    # Add preference penalty
                    pref_pen = calculate_preference_penalty(
                        device, t, device.PREFERENCE_PENALTY_WEIGHT
                    )
                    
                    # Add to cost terms
                    all_cost_terms.append((adj_cost + pref_pen) * X[t])
        
        # 3) Battery variables & constraints
        charge, discharge, soc = None, None, None
        
        if battery_agent is not None:
            print(f"Battery available with capacity: {battery_agent.capacity} kWh, current SOC: {battery_agent.current_soc}")
            
            # Create battery variables using the helper function
            charge, discharge, soc = _add_storage_vars(
                prob, "Batt", hours,
                battery_agent.capacity,
                battery_agent.soc_min,
                battery_agent.soc_max,
                battery_agent.current_soc,
                battery_agent.max_charge_rate,
                battery_agent.max_discharge_rate
            )
            
            # Add to objective
            for t in hours:
                if t < len(prices):
                    all_cost_terms.append(prices[t] * charge[t])
                    all_cost_terms.append(-prices[t] * discharge[t])

        # EV variables
        ev_charge, ev_discharge, ev_soc = None, None, None
        
        if ev_agent is not None:
            print(f"EV available with capacity: {ev_agent.capacity} kWh, current SOC: {ev_agent.current_soc}")
            
            # Create EV variables using the helper function
            ev_charge, ev_discharge, ev_soc = _add_storage_vars(
                prob, "EV", hours,
                ev_agent.capacity,
                getattr(ev_agent, "soc_min", 0.0),
                ev_agent.soc_max,
                ev_agent.current_soc,
                ev_agent.max_charge_rate,
                getattr(ev_agent, "max_discharge_rate", 0.0)
            )
            
            # EV must have minimum charge by departure hour (realistic constraint)
            must_full = ev_agent.spec.get("must_be_full_by_hour", 7)
            # Instead of forcing full charge, ensure minimum viable charge (40% + buffer for daily usage)
            min_departure_soc = max(ev_agent.soc_min + 10.0, ev_agent.soc_max * 0.95)  # 40% or 24kWh minimum
            # EV must be fully charged BEFORE departure  
            if must_full > 0:
                prob += ev_soc[must_full - 1] >= min_departure_soc, "EV_fully_charged_before_departure"
            else:
                prob += ev_soc[0] >= min_departure_soc, "EV_fully_charged_before_departure"

            # No charging allowed at or after departure time
            for t in range(must_full, len(hours)):
                prob += ev_charge[t] == 0, f"EV_no_charge_after_departure_{t}"
            
            # Add to objective
            for t in hours:
                if t < len(prices):
                    all_cost_terms.append(prices[t] * ev_charge[t])
                    all_cost_terms.append(-prices[t] * ev_discharge[t])
        
        # 4) Building load constraints - combined for all device types
        print("Adding building-wide load constraints...")
        for t in hours:
            loads = []
            
            # Add loads from discrete-phase devices
            for dev_idx, device in enumerate(devices):
                if not device.is_flexible or dev_idx not in device_phase_vars:
                    continue
                    
                X = device_phase_vars[dev_idx]
                for p in range(len(device.phases)):
                    durp = device.phases[p]["duration"]
                    e_kwh = device.phases[p]["energy_kwh"]
                    # if phase p starts at k and runs through t, add its fraction
                    for k in range(24):
                        if k <= t < k + durp:
                            loads.append((e_kwh / durp) * X[(dev_idx, p, k)])
            
            # Add loads from partial-usage devices
            for dev_idx, (X, _) in device_alloc_vars.items():
                # For partial-usage devices, the variable directly represents power
                loads.append(X[t])
            
            # Add storage devices load (battery and/or EV if applicable)
            storage_terms = []
            if battery_agent is not None:
                storage_terms.append(charge[t] - discharge[t])
            if ev_agent is not None:
                storage_terms.append(ev_charge[t] - ev_discharge[t])
                
            if storage_terms:  # If we have any storage devices
                prob += lpSum(loads) + lpSum(storage_terms) <= global_layer.max_building_load, f"BuildingLoad_{t}"
            else:
                prob += lpSum(loads) <= global_layer.max_building_load, f"BuildingLoad_{t}"
        
        # 5) Objective - already built throughout the previous steps
        prob += lpSum(all_cost_terms), "TotalCost"
        
        # 6) Solve
        print("Solving centralized MILP...")
        solver = PULP_CBC_CMD(msg=False, timeLimit=300, presolve='on', cuts='on')
        prob.solve(solver)
        
        # Get the solver's objective value directly
        from pulp import value
        solver_objective = value(prob.objective) if prob.status == 1 else None
        print(f"Solver objective value: {solver_objective}")
        
        if LpStatus[prob.status] != 'Optimal':
            print(f"MILP not optimal ({LpStatus[prob.status]})")
            return False
        print("MILP solution found!")
        
        # 7) Extract schedules
        for dev_idx, device in enumerate(devices):
            # Skip non-flexible devices
            if not device.is_flexible:
                device.nextday_optimized_schedule = [0]*24
                continue
                
            # Initialize schedule
            sched = [0.0]*24
            
            # Get flex model
            flex_model = device.spec.get("flex_model", "discrete_phase")
            
            if flex_model == "discrete_phase" and dev_idx in device_phase_vars:
                # Extract schedules for discrete-phase devices
                X = device_phase_vars[dev_idx]
                for p in range(len(device.phases)):
                    durp = device.phases[p]["duration"]
                    e_kwh = device.phases[p]["energy_kwh"]
                    for k in range(24):
                        if X[(dev_idx, p, k)].varValue > 0.5:
                            # Convert duration to integer for the range function
                            for h in range(int(durp)):
                                if k+h < 24:
                                    sched[k+h] += e_kwh/durp
            
            elif flex_model == "partial_usage" and dev_idx in device_alloc_vars:
                # Extract schedules for partial-usage devices with phase ordering
                X, _ = device_alloc_vars[dev_idx]
                
                # If we're using the new phase-ordered model, extract phase assignments directly
                if hasattr(device, 'phases') and any(f"phase_{dev_idx}_" in v.name for v in prob.variables()):
                    # Find all Y variables for this device (phase assignments)
                    phase_vars = {}
                    for v in prob.variables():
                        if f"phase_{dev_idx}_" in v.name and hasattr(v, 'varValue'):
                            # Extract phase and hour from variable name
                            parts = v.name.split('_')
                            if len(parts) >= 4:
                                p = int(parts[2])
                                t = int(parts[3])
                                if v.varValue > 0.5:  # Binary variable is active
                                    phase_vars[(p, t)] = v.varValue
                    
                    # Assign exact phase energy values to the schedule for discrete-phase devices
                    for (p, t), val in phase_vars.items():
                        if p < len(device.phases):
                            sched[t] = device.phases[p]['energy_kwh']
                            print(f"  Phase {p} (energy={device.phases[p]['energy_kwh']} kWh) assigned to hour {t}")
                else:
                    # For partial-usage devices, use the direct energy allocation values
                    for t in range(24):
                        sched[t] = X[t].varValue
                        if sched[t] > 0.01:  # Only log non-zero allocations
                            print(f"  Hour {t}: {sched[t]:.2f} kWh allocated")
            
            device.nextday_optimized_schedule = sched
            cnt = sum(1 for x in sched if x > 0)
            print(f"Updated {device.device_name} schedule ({flex_model}): active in {cnt} hours")
        
        # 8) Update battery day profile
        if battery_agent is not None and 'charge' in locals():
            soc_vals = [soc[t].varValue for t in hours]
            ch_vals = [charge[t].varValue for t in hours]
            dis_vals= [discharge[t].varValue for t in hours]
            
            # Populate hourly arrays for battery (ensure 24-hour arrays)
            for t in range(24):
                if t < len(hours) and t < len(ch_vals):
                    battery_agent.hourly_charge[t] = ch_vals[t] if ch_vals[t] is not None else 0.0
                    battery_agent.hourly_discharge[t] = dis_vals[t] if dis_vals[t] is not None else 0.0
                    battery_agent.hourly_soc[t] = soc_vals[t] if soc_vals[t] is not None else battery_agent.current_soc
                else:
                    # Pad remaining hours with zeros for charge/discharge, maintain last SOC
                    battery_agent.hourly_charge[t] = 0.0
                    battery_agent.hourly_discharge[t] = 0.0
                    battery_agent.hourly_soc[t] = battery_agent.hourly_soc[t-1] if t > 0 else battery_agent.current_soc
            
            # Update battery history
            final_soc = soc_vals[-1]
            # CRITICAL FIX: Ensure SOC stays within bounds to prevent infeasible subsequent days
            final_soc = min(max(final_soc, battery_agent.soc_min), battery_agent.soc_max)
            battery_agent.soc_history.append(final_soc)
            battery_agent.charge_history.append(sum(ch_vals))
            battery_agent.discharge_history.append(sum(dis_vals))
            battery_agent.current_soc = final_soc
            for d in devices:
                d.battery_soc_day = soc_vals
                d.battery_charge_day = ch_vals
                d.battery_discharge_day = dis_vals
        
        # 9) Update EV day profile (CRITICAL FIX: This was missing!)
        if ev_agent is not None and 'ev_charge' in locals() and ev_charge is not None:
            ev_soc_vals = [ev_soc[t].varValue for t in hours]
            ev_ch_vals = [ev_charge[t].varValue for t in hours] 
            ev_dis_vals = [ev_discharge[t].varValue for t in hours]
            
            # Populate hourly arrays for EV (ensure 24-hour arrays)
            for t in range(24):
                if t < len(hours) and t < len(ev_ch_vals):
                    ev_agent.hourly_charge[t] = ev_ch_vals[t] if ev_ch_vals[t] is not None else 0.0
                    ev_agent.hourly_discharge[t] = ev_dis_vals[t] if ev_dis_vals[t] is not None else 0.0
                    ev_agent.hourly_soc[t] = ev_soc_vals[t] if ev_soc_vals[t] is not None else ev_agent.current_soc
                else:
                    # Pad remaining hours with zeros for charge/discharge, maintain last SOC
                    ev_agent.hourly_charge[t] = 0.0
                    ev_agent.hourly_discharge[t] = 0.0
                    ev_agent.hourly_soc[t] = ev_agent.hourly_soc[t-1] if t > 0 else ev_agent.current_soc
            
            # Update EV history
            final_soc = ev_soc_vals[-1]
            # Ensure SOC stays within bounds
            final_soc = min(max(final_soc, ev_agent.soc_min), ev_agent.soc_max)
            ev_agent.soc_history.append(final_soc)
            ev_agent.charge_history.append(sum(ev_ch_vals))
            if hasattr(ev_agent, 'discharge_history'):
                ev_agent.discharge_history.append(sum(ev_dis_vals))
            ev_agent.current_soc = final_soc
            
            # Add EV data to devices for visualization
            for d in devices:
                d.ev_soc_day = ev_soc_vals
                d.ev_charge_day = ev_ch_vals
                d.ev_discharge_day = ev_dis_vals
            
            print(f"  Updated EV agent: SOC={final_soc:.2f}, day charge={sum(ev_ch_vals):.2f}, day discharge={sum(ev_dis_vals):.2f}")

        # Transfer the nextday_optimized_schedule to phases_optimized_schedule for each device
        # so that the _evaluate_costs_for_day function can access it
        for device in devices:
            if hasattr(device, 'nextday_optimized_schedule'):
                device.phases_optimized_schedule = device.nextday_optimized_schedule
        
        # Get the day being optimized
        day = reference_device.data['day'].unique()[0] if len(reference_device.data['day'].unique()) > 0 else 0
        
        # Call the standardized cost evaluation helper
        self._evaluate_costs_for_day(day, prices, pv_forecast, "phases", solver_objective)
        print(f"Costs for day {day} evaluated and stored for standardized comparison with solver objective value: {solver_objective}")
        
        return True

    def optimize_for_weekday_centralized(self):
        """
        Enhanced version of optimize_for_weekday that uses a centralized approach.
        This replaces the hierarchical approach with a centralized MILP that optimizes
        all device phases together.
        """
        tomorrow = datetime.date.today() + timedelta(days=1)
        tweekday = tomorrow.strftime("%A")
        print(f"Next-day aggregator for {tweekday} (Centralized Method)")
        
        # Load offline results as warm start (same as original)
        self.load_results("full_optimization.pkl")
        print("Loaded offline results for warm start")
        
        # Clear device savings
        for dev in self.devices:
            dev.savings = 0.0
        
        # Use the centralized optimization function
        success = self.optimize_phases_centralized(
            self.devices, 
            self.global_layer,
            self.pv_agent, 
            self.battery_agent,
            self.ev_agent,
            self.grid_agent,
            self.weather_agent
        )
        
        if success:
            print("Centralized next-day optimization completed successfully")
            
            # Calculate savings for each device
            for dev in self.devices:
                # Skip if device has no nextday_optimized_schedule
                if not hasattr(dev, 'nextday_optimized_schedule') or dev.nextday_optimized_schedule is None:
                    continue
                
                # Extract prices for each hour
                prices = dev.data.groupby('hour')['price_per_kwh'].mean().reindex(range(24), fill_value=0).values
                
                # Calculate baseline cost (using original consumption)
                baseline_cost = 0.0
                for hour in range(24):
                    # Get original consumption for this hour
                    hour_mask = dev.data['hour'] == hour
                    if hour_mask.any():
                        orig_hour = dev.original_consumption[hour_mask].mean()
                    else:
                        orig_hour = 0.0
                    
                    baseline_cost += orig_hour * prices[hour]
                
                # Calculate optimized cost (using nextday_optimized_schedule)
                optimized_cost = 0.0
                for hour in range(24):
                    if hour < len(dev.nextday_optimized_schedule):
                        optimized_cost += dev.nextday_optimized_schedule[hour] * prices[hour]
                
                # Calculate battery impact if applicable
                battery_cost = 0.0
                if hasattr(dev, 'battery_charge_day') and hasattr(dev, 'battery_discharge_day'):
                    for hour in range(24):
                        # Cost to charge, minus value from discharge
                        battery_cost += dev.battery_charge_day[hour] * prices[hour] - dev.battery_discharge_day[hour] * prices[hour]
                
                # Total cost including battery
                total_cost = optimized_cost + battery_cost
                
                # Calculate and store savings
                dev.nextday_savings = baseline_cost - total_cost
                dev.savings = dev.nextday_savings  # Update current savings
                
                print(f"{dev.device_name}: baseline=${baseline_cost:.2f}, optimized=${total_cost:.2f}, savings=${dev.nextday_savings:.2f}")
        else:
            print("Centralized next-day optimization failed")
        
        # Save results (same as original)
        self.save_results("weekday_optimization_centralized.pkl")
        
        return success

    def broadcast_preferred_hours(self):
        for dev in self.devices:
            tot_atts = sum(sr['total'] for sr in dev.success_rates.values())
            tot_succ = sum(sr['success'] for sr in dev.success_rates.values())
            if tot_atts > 0 and tot_succ / tot_atts > 0.8:
                dev.update_preferred_hours()
                for hr in dev.preferred_hours[:3]:
                    for other in self.devices:
                        if other != dev:
                            other.conflict_count[hr] += 1
    
    def analyze_iteration(self, iteration_results, previous_limit_hits):
        hits = iteration_results['limit_hits']
        if hits > previous_limit_hits:
            for dev in self.devices:
                dev.price_sensitivity *= 1.1
        else:
            for dev in self.devices:
                dev.price_sensitivity *= 0.9

    # ──────────────────────────────────────────────────────────────────────
    def _evaluate_costs_for_day(
            self,
            day_idx:      int,
            price_vec:    np.ndarray,
            pv_vec:       np.ndarray | None,
            tag:          str,                   # "continuous" | "phases"
            solver_obj:   float = None
        ) -> None:
        """
        Compute and store original / optimised € for *this* day **and**
        **for the optimiser that just ran** (identified by `tag`).

        Result fields created
        ---------------------
        • per device   →  device.costs[ tag ]           = {'orig': €, 'opt': €, 'sav': €}
        • whole house  →  self.costs_by_day[ tag ][day] = {'orig': €, 'opt': €, 'sav': €, 'solver': €}
        • convenience  →  self.total_savings_{tag}
        """
        if pv_vec is None:
            pv_vec = np.zeros_like(price_vec)

        # ------------------------------------------------------------------  
        # 1. choose the attribute that holds the new schedule for *this* tag  
        # ------------------------------------------------------------------
        attr_map = {
            "continuous": "centralized_optimized_schedule",
            "phases"    : "phases_optimized_schedule",
            "decentralized": "optimized_consumption",
        }
        sched_attr = attr_map[tag]

        # ------------------------------------------------------------------  
        # 2. per-device baseline & optimised cost                             
        # ------------------------------------------------------------------
        for dev in self.devices:
            # Extract data for this specific day only
            if hasattr(dev, 'data') and 'day' in dev.data.columns:
                # Find the data for this specific day
                day_mask = dev.data['day'] == day_idx
                day_data = dev.data[day_mask]
                
                if len(day_data) == len(price_vec):
                    # Use the device consumption for this day only
                    orig_consumption_day = day_data[dev.device_name].values
                else:
                    # Fallback to slicing original_consumption array
                    orig_consumption_day = dev.original_consumption[:len(price_vec)]
                    if len(orig_consumption_day) < len(price_vec):
                        orig_consumption_day = np.pad(orig_consumption_day, 
                                                    (0, len(price_vec) - len(orig_consumption_day)), 
                                                    'constant')
            else:
                # Ensure original_consumption is the right length
                if len(dev.original_consumption) < len(price_vec):
                    print(f"Warning: {dev.device_name} original_consumption too short. Padding with zeros.")
                    dev.original_consumption = np.pad(dev.original_consumption, 
                                                    (0, len(price_vec) - len(dev.original_consumption)), 
                                                    'constant')
                orig_consumption_day = dev.original_consumption[:len(price_vec)]
                
            orig = np.sum(orig_consumption_day * price_vec)

            new_sched = getattr(dev, sched_attr, None)
            if new_sched is None:
                raise AttributeError(
                    f"[evaluate_costs] {sched_attr} missing on {dev.device_name}"
                )
                
            # Ensure new_sched is the right length - only use first 24 hours for daily evaluation
            if len(new_sched) > len(price_vec):
                # Truncate to match price_vec length (should be 24 hours for daily evaluation)
                new_sched_day = new_sched[:len(price_vec)]
            elif len(new_sched) < len(price_vec):
                print(f"Warning: {dev.device_name} {sched_attr} too short. Padding with zeros.")
                new_sched_day = np.pad(new_sched, 
                                      (0, len(price_vec) - len(new_sched)), 
                                      'constant')
            else:
                new_sched_day = new_sched
                
            opt = np.sum(new_sched_day * price_vec)

            # attach dict so you can keep many solvers side-by-side
            dct = dev.__dict__.setdefault("costs", {})
            dct[tag] = {"orig": orig, "opt": opt, "sav": orig - opt}

        # ------------------------------------------------------------------  
        # 3. grid import / export incl. batteries & PV                       
        # ------------------------------------------------------------------
        grid_opt = np.zeros_like(price_vec)
        grid_orig= np.zeros_like(price_vec)

        # batteries
        batt_chg = self.battery_agent.hourly_charge if self.battery_agent else np.zeros_like(price_vec)
        batt_dis = self.battery_agent.hourly_discharge if self.battery_agent else np.zeros_like(price_vec)
        ev_chg = self.ev_agent.hourly_charge if self.ev_agent else np.zeros_like(price_vec)
        ev_dis = self.ev_agent.hourly_discharge if self.ev_agent else np.zeros_like(price_vec)
        
        # Ensure all battery arrays have the right length
        def pad_array(arr, target_len):
            if len(arr) < target_len:
                return np.pad(arr, (0, target_len - len(arr)), 'constant')
            return arr
            
        batt_chg = pad_array(batt_chg, len(price_vec))
        batt_dis = pad_array(batt_dis, len(price_vec))
        ev_chg = pad_array(ev_chg, len(price_vec))
        ev_dis = pad_array(ev_dis, len(price_vec))

        for h in range(len(price_vec)):
            for dev in self.devices:
                # Make sure we don't go out of bounds
                if h < len(dev.original_consumption):
                    grid_orig[h] += dev.original_consumption[h]
                
                new_sched = getattr(dev, sched_attr)
                if h < len(new_sched):
                    grid_opt[h] += new_sched[h]

            # Add battery impact
            if h < len(batt_chg):
                grid_opt[h] += batt_chg[h] - batt_dis[h]
            
            # Add EV impact
            if h < len(ev_chg):
                grid_opt[h] += ev_chg[h] - ev_dis[h]
                
            grid_orig[h] += 0.0  # no storage in baseline

            # add PV (negative ⇒ generation)
            if h < len(pv_vec):
                grid_orig[h] += pv_vec[h]
                grid_opt[h] += pv_vec[h]

        imp = self.grid_agent.import_price if self.grid_agent else 0.05
        exp = self.grid_agent.export_price if self.grid_agent else 0.04

        def _bill(flow):
            """positive = import, negative = export"""
            return np.sum(np.where(flow >= 0, flow*imp, flow*exp))

        cost_o = _bill(grid_orig)  # Original/baseline cost
        cost_n = _bill(grid_opt)   # New/optimized calculated cost
        
        # Use the solver's objective value if provided
        solver_cost = solver_obj if solver_obj is not None else cost_n

        # store day–tag pair
        by_day = self.__dict__.setdefault("costs_by_day", {}).setdefault(tag, {})
        by_day[day_idx] = {"orig": cost_o, "opt": cost_n, "sav": cost_o-cost_n, "solver": solver_obj or cost_n}

        # convenience rolling totals
        self.__dict__[f"total_savings_{tag}"] = \
            sum(d["sav"] for d in by_day.values())
            
        # Also store the solver-based savings if solver objective was provided
        if solver_obj is not None:
            self.__dict__[f"solver_savings_{tag}"] = \
                sum((d["orig"] - d["solver"]) for d in by_day.values())

    def evaluate_performance(self, tag="decentralized"):
        """
        Computes the baseline vs. current cost across all devices for a given optimization type.
        This method now correctly calculates costs by leveraging the robust `_evaluate_costs_for_day` method.
        """
        total_savings = 0.0
        total_original_cost = 0.0
        total_optimized_cost = 0.0

        # Get all unique days from the first device's data
        if not self.devices:
            return {"baseline_cost": 0, "current_cost": 0, "cost_reduction_percentage": 0}

        all_days = self.devices[0].data['day'].unique()

        for day in all_days:
            day_mask = self.devices[0].data['day'] == day
            day_prices = self.devices[0].data[day_mask]['price_per_kwh'].values

            pv_vec = None
            if self.pv_agent:
                pv_vec = self.pv_agent.get_hourly_forecast_pv(day)

            self._evaluate_costs_for_day(
                day_idx=day,
                price_vec=day_prices,
                pv_vec=pv_vec,
                tag=tag
            )

            if tag in self.costs_by_day and day in self.costs_by_day[tag]:
                day_costs = self.costs_by_day[tag][day]
                total_original_cost += day_costs.get('orig', 0)
                total_optimized_cost += day_costs.get('opt', 0)

        total_savings = total_original_cost - total_optimized_cost
        reduction_pct = (total_savings / total_original_cost) * 100 if total_original_cost > 0 else 0

        return {
            "baseline_cost": total_original_cost,
            "current_cost": total_optimized_cost,
            "cost_reduction_percentage": reduction_pct
        }

    def save_results(self, filename: str):
        """
        Save the offline optimization results along with global battery data.
        Modifications:
          - Include battery global arrays and battery agent history.
          - Fix handling of battery attributes to ensure they're always properly initialized.
        """
        results = {
            "iteration_history": self.iteration_history,
            "best_iteration": self.best_iteration,
            "best_savings": self.best_savings,
            "devices": {}
        }
        
        # Determine total_hours from either a class attribute or device data length
        total_hours = 24  # Default fallback
        if hasattr(self, 'total_hours'):
            total_hours = self.total_hours
        elif len(self.devices) > 0:
            total_hours = len(self.devices[0].data)
            
        # Save the global battery data if available
        if self.battery_agent is not None:
            # Create or ensure battery attributes exist
            # battery_soc
            if not hasattr(self, 'battery_soc'):
                self.battery_soc = np.full(total_hours, self.battery_agent.current_soc)
            elif self.battery_soc is None:
                self.battery_soc = np.full(total_hours, self.battery_agent.current_soc)
            
            # battery_charge
            if not hasattr(self, 'battery_charge'):
                self.battery_charge = np.zeros(total_hours)
            elif self.battery_charge is None:
                self.battery_charge = np.zeros(total_hours)
            
            # battery_discharge
            if not hasattr(self, 'battery_discharge'):
                self.battery_discharge = np.zeros(total_hours)
            elif self.battery_discharge is None:
                self.battery_discharge = np.zeros(total_hours)
                
            # Initialize global SoC, charge, discharge arrays if specified for backward compatibility
            if hasattr(self, 'battery_soc_global') and self.battery_soc_global is not None:
                results["battery"] = {
                    "soc": self.battery_soc_global,
                    "charge": self.battery_charge_global,
                    "discharge": self.battery_discharge_global,
                    "soc_global": self.battery_soc_global,  # For backward compatibility
                    "charge_global": self.battery_charge_global,
                    "discharge_global": self.battery_discharge_global,
                    "soc_history": self.battery_agent.soc_history,
                    "charge_history": self.battery_agent.charge_history,
                    "discharge_history": self.battery_agent.discharge_history,
                    "estimated_capacity": self.battery_agent.estimated_capacity,
                    "max_charge_rate": self.battery_agent.max_charge_rate,
                    "max_discharge_rate": self.battery_agent.max_discharge_rate,
                    "cycle_count": self.battery_agent.cycle_count
                }
            else:
                results["battery"] = {
                    "soc": self.battery_soc,
                    "charge": self.battery_charge,
                    "discharge": self.battery_discharge,
                    "soc_history": self.battery_agent.soc_history,
                    "charge_history": self.battery_agent.charge_history,
                    "discharge_history": self.battery_agent.discharge_history,
                    "estimated_capacity": self.battery_agent.estimated_capacity,
                    "max_charge_rate": self.battery_agent.max_charge_rate,
                    "max_discharge_rate": self.battery_agent.max_discharge_rate,
                    "cycle_count": self.battery_agent.cycle_count
                }
                
        # Save the global EV data if available
        if self.ev_agent is not None:
            # Create or ensure EV attributes exist
            # ev_soc
            if not hasattr(self, 'ev_soc_global') or self.ev_soc_global is None:
                self.ev_soc_global = np.full(total_hours, self.ev_agent.current_soc)
            
            # ev_charge
            if not hasattr(self, 'ev_charge_global') or self.ev_charge_global is None:
                self.ev_charge_global = np.zeros(total_hours)
            
            # ev_discharge
            if not hasattr(self, 'ev_discharge_global') or self.ev_discharge_global is None:
                self.ev_discharge_global = np.zeros(total_hours)
                
            # Save EV data
            results["ev"] = {
                "soc": self.ev_soc_global,
                "charge": self.ev_charge_global,
                "discharge": self.ev_discharge_global,
                "soc_global": self.ev_soc_global,
                "charge_global": self.ev_charge_global,
                "discharge_global": self.ev_discharge_global,
                "capacity": self.ev_agent.capacity,
                "max_charge_rate": self.ev_agent.max_charge_rate,
                "max_discharge_rate": getattr(self.ev_agent, 'max_discharge_rate', 0.0),
                "current_soc": self.ev_agent.current_soc,
                "must_be_full_by_hour": self.ev_agent.spec.get("must_be_full_by_hour", 7)
            }
            
            # Save history arrays if they exist
            if hasattr(self.ev_agent, 'soc_history'):
                results["ev"]["soc_history"] = self.ev_agent.soc_history
            if hasattr(self.ev_agent, 'charge_history'):
                results["ev"]["charge_history"] = self.ev_agent.charge_history
            if hasattr(self.ev_agent, 'discharge_history'):
                results["ev"]["discharge_history"] = self.ev_agent.discharge_history
            if hasattr(self.ev_agent, 'cycle_count'):
                results["ev"]["cycle_count"] = self.ev_agent.cycle_count
        
        for dev in self.devices:
            # Initialize basic device data
            device_data = {
                "optimized_consumption": dev.optimized_consumption,
                "shifts": dev.shifts,
                "iteration_consumption": dev.iteration_consumption,
            }
            
            # Add nextday_optimized_schedule if it exists
            if hasattr(dev, 'nextday_optimized_schedule') and dev.nextday_optimized_schedule is not None:
                device_data["nextday_optimized_schedule"] = dev.nextday_optimized_schedule
            
            # Save individual device battery usage data if available
            if hasattr(dev, 'battery_soc') and dev.battery_soc is not None:
                device_data["battery_soc"] = dev.battery_soc
                device_data["battery_charge"] = dev.battery_charge
                device_data["battery_discharge"] = dev.battery_discharge
            
            results["devices"][dev.device_name] = device_data
        
        with open(filename, "wb") as f:
            pickle.dump(results, f)
        logging.info(f"Saved results to {filename}")

    def load_results(self, filename: str):
        """
        Load offline optimization results and restore global battery data.
        Modifications:
          - Restore global battery arrays and battery agent history.
        """
        with open(filename, "rb") as f:
            data = pickle.load(f)
        self.iteration_history = data.get("iteration_history", [])
        self.best_iteration = data.get("best_iteration")
        self.best_savings = data.get("best_savings")
        
        # Load battery data if available
        if "battery" in data and self.battery_agent is not None:
            battery_data = data["battery"]
            if "soc_global" in battery_data:
                self.battery_soc_global = battery_data.get("soc_global")
                self.battery_charge_global = battery_data.get("charge_global")
                self.battery_discharge_global = battery_data.get("discharge_global")
            
            # Restore battery agent history
            self.battery_agent.soc_history = battery_data.get("soc_history", [])
            self.battery_agent.charge_history = battery_data.get("charge_history", [])
            self.battery_agent.discharge_history = battery_data.get("discharge_history", [])
            self.battery_agent.cycle_count = battery_data.get("cycle_count", 0)
            
        # Load EV data if available
        if "ev" in data and self.ev_agent is not None:
            ev_data = data["ev"]
            if "soc_global" in ev_data:
                self.ev_soc_global = ev_data.get("soc_global")
                self.ev_charge_global = ev_data.get("charge_global")
                self.ev_discharge_global = ev_data.get("discharge_global")
            
            # Restore EV agent state
            if "current_soc" in ev_data:
                self.ev_agent.current_soc = ev_data.get("current_soc")
                
            # Restore EV agent history if exists
            if hasattr(self.ev_agent, 'soc_history') and "soc_history" in ev_data:
                self.ev_agent.soc_history = ev_data.get("soc_history", [])
            if hasattr(self.ev_agent, 'charge_history') and "charge_history" in ev_data:
                self.ev_agent.charge_history = ev_data.get("charge_history", [])
            if hasattr(self.ev_agent, 'discharge_history') and "discharge_history" in ev_data:
                self.ev_agent.discharge_history = ev_data.get("discharge_history", [])
            if hasattr(self.ev_agent, 'cycle_count') and "cycle_count" in ev_data:
                self.ev_agent.cycle_count = ev_data.get("cycle_count", 0)
        
        devdict = data.get("devices", {})
        for dev in self.devices:
            dd = devdict.get(dev.device_name, {})
            dev.optimized_consumption = dd.get("optimized_consumption", dev.optimized_consumption)
            dev.shifts = dd.get("shifts", dev.shifts)
            dev.iteration_consumption = dd.get("iteration_consumption", dev.iteration_consumption)
            dev.nextday_optimized_schedule = dd.get("nextday_optimized_schedule", dev.nextday_optimized_schedule)
            
            # Load device battery data if available
            if "battery_soc" in dd:
                dev.battery_soc = dd.get("battery_soc")
                dev.battery_charge = dd.get("battery_charge")
                dev.battery_discharge = dd.get("battery_discharge")
        
        logging.info(f"Loaded results from {filename}")
        
    def optimize_with_advanced_coordination(self):
        """
        Run optimization with enhanced device-battery coordination
        This method utilizes the advanced coordination features in devices
        and battery agent to achieve better overall savings.
        
        Returns:
            List of iteration history
        """
        import random

        logging.info("\n\nOptimizing with Advanced Coordination...")
        original_bldg_cost = 0.0
        optimized_bldg_cost = 0.0

        # Reset devices and clear iteration history
        for device in self.devices:
            device.iterations = []
            device.iteration_consumption = {}
            device.optimized_consumption = device.original_consumption.copy()

        # Get the maximum data length for battery arrays
        data_length = 24  # Default fallback
        if len(self.devices) > 0:
            data_length = len(self.devices[0].data)
        
        # Reset global building load tracker
        self.global_layer.hourly_load = np.zeros(24)
        self.iteration_history = []
        
        # If we have price data, make it available to the global layer for coordination
        if len(self.devices) > 0 and 'price_per_kwh' in self.devices[0].data.columns:
            self.global_layer.set_price_data(self.devices[0].data[['utc_timestamp', 'price_per_kwh']])
        
        if self.battery_agent is not None:
            # Starting battery state with correct size
            self.battery_soc = np.full(data_length, self.battery_agent.current_soc)
            self.battery_charge = np.zeros(data_length)
            self.battery_discharge = np.zeros(data_length)

        # Randomize device order for each pass:
        for iteration in range(self.max_iterations):
            logging.info(f"\nIteration {iteration+1}/{self.max_iterations}")
            random.shuffle(self.devices)
            iteration_devices = []
            
            # Take a snapshot of the load profile at the start of this iteration
            self.global_layer.snapshot_load_profile()
            
            for device in self.devices:
                logging.info(f"  Optimizing {device.device_name} with advanced coordination...")
                day_indices = []
                days = device.data['day'].unique()
                
                for day in days:
                    day_mask = (device.data['day'] == day)
                    first_idx = device.data[day_mask].index[0] if len(device.data[day_mask]) > 0 else None
                    if first_idx is not None:
                        day_indices.append(first_idx)
                
                for day_idx in day_indices:
                    # Use the new enhanced optimization method
                    device.optimize_with_global_constraints(
                        day_idx, 
                        use_battery=(self.battery_agent is not None),
                        advanced_coordination=True
                    )
                    
                    if self.battery_agent is not None:
                        # Update global battery state from device optimization
                        try:
                            day_mask = (device.data['day'] == device.data.loc[day_idx, 'day'])
                            day_idxs = device.data[day_mask].index
                            
                            # Make sure we don't go out of bounds
                            for idx in day_idxs:
                                if idx < len(self.battery_soc):
                                    self.battery_soc[idx] = device.battery_soc[idx]
                                    self.battery_charge[idx] = device.battery_charge[idx]
                                    self.battery_discharge[idx] = device.battery_discharge[idx]
                            
                            # Update battery agent's current state with the final SoC value
                            if len(day_idxs) > 0 and day_idxs[-1] < len(self.battery_soc):
                                final_soc = self.battery_soc[day_idxs[-1]]
                                # CRITICAL FIX: Ensure SOC stays within bounds to prevent infeasible subsequent days
                                final_soc = min(max(final_soc, self.battery_agent.soc_min), self.battery_agent.soc_max)
                                self.battery_agent.current_soc = final_soc
                        except Exception as e:
                            logging.warning(f"Error updating battery state: {e}")
    
                iteration_devices.append({
                    'device_name': device.device_name,
                    'savings': device.savings
                })
            
            # Save iteration history
            self.iteration_history.append(iteration_devices)
            
            # Print costs at this iteration
            eval_metrics = self.evaluate_performance()
            original_cost = eval_metrics.get("baseline_cost", 0)
            optimized_cost = eval_metrics.get("current_cost", 0)
            original_bldg_cost = original_cost
            optimized_bldg_cost = optimized_cost
            logging.info(f" ===> Advanced coordination iteration {iteration+1} costs: original={original_cost:.2f}, optimized={optimized_cost:.2f}, savings={(original_cost-optimized_cost):.2f} ({100 * (original_cost-optimized_cost) / original_cost:.1f}%)")
        
        # Final cost calculation
        eval_metrics = self.evaluate_performance()
        original_bldg_cost = eval_metrics.get("baseline_cost", 0)
        optimized_bldg_cost = eval_metrics.get("current_cost", 0)
        savings = original_bldg_cost - optimized_bldg_cost
        savings_pct = 100 * savings / original_bldg_cost if original_bldg_cost > 0 else 0
        
        logging.info(f"\nAdvanced coordination optimization complete!")
        logging.info(f"  Original cost: ${original_bldg_cost:.2f}")
        logging.info(f"  Optimized cost: ${optimized_bldg_cost:.2f}")
        logging.info(f"  Savings: ${savings:.2f} ({savings_pct:.1f}%)")
        
        # Save the results with a special filename
        self.save_results("advanced_coordination_optimization.pkl")
        
        return self.iteration_history

    def validate_battery_coordination(self, day, hourly_battery_plans):
        """
        Validate battery coordination to ensure our battery business rules are being followed
        """
        if not hourly_battery_plans or day not in hourly_battery_plans:
            return False
        
        violations = []
        
        # Check for rule violations
        for hour in range(24):
            plan = hourly_battery_plans[day][hour]
            
            # Rule 1: If charging, must be at max rate and exclusive
            if plan['charge'] > 0.001:
                # Check if charging at max rate (allow 5% tolerance)
                max_rate = self.battery_agent.max_charge_rate
                is_max_rate = abs(plan['charge'] - max_rate) <= 0.05 * max_rate
                
                # Check exclusivity - cannot discharge when charging
                is_exclusive = plan['discharge'] < 0.001
                
                if not is_max_rate or not is_exclusive:
                    violations.append(f"Hour {hour}: Charging at {plan['charge']:.2f} kW (max={max_rate:.2f} kW), "
                                    f"Discharging at {plan['discharge']:.2f} kW. "
                                    f"{'Not at max rate' if not is_max_rate else ''} "
                                    f"{'Not exclusive' if not is_exclusive else ''}")
        
        # Print validation results
        if violations:
            # print(f"[BATTERY VALIDATION] Day {day} has {len(violations)} coordination violations:")
            for v in violations:
                print(f"  - {v}")
            return False
        else:
            print(f"[BATTERY VALIDATION] Day {day} battery coordination is valid.")
            return True

    def update_device_constraints(self, devices):
        """
        Update the optimizer's internal device constraints.
        
        This method refreshes the optimizer's knowledge of device constraints,
        ensuring that any modifications made to device specifications (particularly
        'allowed_hours') are properly reflected in the optimization process.
        
        Parameters:
        -----------
        devices : list
            List of device objects whose constraints should be updated in the optimizer.
        
        Returns:
        --------
        bool
            True if constraints were successfully updated, False otherwise.
        """
        # Store original devices for constraint creation
        self.devices = devices
        
        # Clear existing device constraints
        if hasattr(self, 'device_constraints'):
            self.device_constraints = {}
        
        # Rebuild device constraint cache
        for device in devices:
            if not hasattr(device, 'spec'):
                continue
                
            device_id = device.device_name
            
            # Store the complete spec as reference
            if not hasattr(self, 'device_specs'):
                self.device_specs = {}
            
            # Deep copy to avoid reference issues
            self.device_specs[device_id] = device.spec.copy()
            
            # Log the update for debugging
            phases = device.spec.get('phases', [])
            allowed_hours = device.spec.get('allowed_hours', list(range(24)))
            
            print(f"Updated constraints for {device_id}:")
            print(f"  - Phases: {len(phases)}")
            print(f"  - Allowed hours: {allowed_hours}")
        
        # Mark constraints as needing rebuild
        self.constraints_initialized = False
        
        return True