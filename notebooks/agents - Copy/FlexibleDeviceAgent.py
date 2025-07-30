import numpy as np
import pandas as pd
import logging
import datetime
from datetime import timedelta
from typing import Dict, Any, List, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from pulp import LpProblem, LpVariable, lpSum, LpMinimize, PULP_CBC_CMD, LpStatus

###########################################################
# Helper functions 
###########################################################
def forecast_discount(forecast_value):
    # up to 40% discount if forecast_value ~1
    return 0.4 / (1 + np.exp(-10 * (forecast_value - 0.5)))

def calculate_preference_penalty(device, hour, penalty_weight=None):
    """
    Calculate a penalty for scheduling a device at hours that don't align with learned preferences.
    
    Args:
        device: The device object which may have 'hour_probability' attribute
        hour: The hour (0-23) to calculate the penalty for
        penalty_weight: Weight of the preference penalty relative to economic costs
        
    Returns:
        float: A penalty value (higher for less preferred hours)
    """
    # Check if device has probability data
    if not hasattr(device, 'hour_probability') or not device.hour_probability:
        return 0.0
    
    # Get probability for this hour
    prob = device.hour_probability.get(hour, 0.0)
    
    # Get maximum probability for normalization
    max_prob = max(device.hour_probability.values()) if device.hour_probability else 0.0
    
    # Calculate normalized probability (0-1 range)
    norm_prob = prob / max_prob if max_prob > 0 else 0.0

    if penalty_weight is None:
        penalty_weight = getattr(device, "PREFERENCE_PENALTY_WEIGHT", 0.0)
    
    # Calculate penalty (higher when probability is lower)
    # print("AAAAAAAAAAA=====> penalty_weight: ", penalty_weight)
    return penalty_weight * (1.0 - norm_prob)

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
        
def get_building_pv_column(device):
    """
    Derive the building PV column name from the device name.
    Assumes the device name is in the format "BuildingID_deviceType"
    (or "BuildingID_subID_deviceType") so that the building id is everything
    before the last underscore.
    """
    parts = device.device_name.split('_')
    if len(parts) >= 2:
        building_id = '_'.join(parts[:-1])
    else:
        building_id = device.device_name
    pv_col = building_id + '_pv'
    # Check if this column exists in the data
    if pv_col in device.data.columns:
        return pv_col
    else:
        # Fallback: search for any column containing "pv" (ignoring grid imports/exports)
        pv_candidates = [col for col in device.data.columns if 'pv' in col.lower() and 'grid' not in col.lower()]
        if pv_candidates:
            return pv_candidates[0]
        else:
            return None

def determine_pv_columns(self):
    """
    Determines all PV production columns in self.data.
    It searches for columns that contain 'pv' (case-insensitive) and excludes those that mention 'grid'.
    Returns a list of matching column names.
    """
    pv_candidates = [col for col in self.data.columns 
                     if 'pv' in col.lower() and 'grid' not in col.lower()]
    return pv_candidates


# =============================================================================
# New Helper Functions for Warm Start Discretization
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt

def discretize_profile_with_phases_multi(
    continuous_profile: np.ndarray,
    device_spec: dict,
    max_runs: int = 2,
    device_name: str = "",
    debug_plot: bool = True
) -> (np.ndarray, list):
    """
    Convert a continuous warm-start profile into multiple consecutive blocks,
    each block having length = sum of device_spec["phases"].duration.
    
    This attempts to find up to 'max_runs' disjoint blocks (e.g. dishwasher
    might run 2 separate times in the day). For each run, we pick the block
    of length total_duration that yields the largest sum of continuous_profile,
    set that block aside, then continue searching for the next run.
    
    Returns:
      discrete_profile: array of same length as continuous_profile, with
                        multiple disjoint blocks of usage.
      start_indices: list of the hour indices where each run starts.
    
    NOTE: This is a greedy approach. For more complex scheduling, you might
          do a specialized dynamic programming or a small MILP.
    """
    import logging
    import numpy as np
    import matplotlib.pyplot as plt

    phases = device_spec.get("phases", [])
    if not phases:
        logging.warning(f"{device_name} has no 'phases' => returning zeros.")
        return np.zeros_like(continuous_profile), []

    # sum up total hours from phases
    total_duration = sum(int(round(ph.get("duration", 1))) for ph in phases)
    total_duration = min(total_duration, len(continuous_profile))
    if total_duration < 1:
        return np.zeros_like(continuous_profile), []

    # We'll pick up to max_runs disjoint blocks
    discrete_profile = np.zeros_like(continuous_profile)
    used_indices = np.zeros_like(continuous_profile, dtype=bool)
    start_indices = []

    # local copy so we can zero out used blocks
    local_profile = continuous_profile.copy()

    for run_i in range(max_runs):
        best_sum = -1e9
        best_start = None

        # find best consecutive window ignoring used slots
        # We'll skip any window that overlaps used_indices
        for start_idx in range(0, len(local_profile) - total_duration + 1):
            # check if this window is free (no used indices)
            window_slice = slice(start_idx, start_idx + total_duration)
            if used_indices[window_slice].any():
                # some overlap => skip
                continue
            block_sum = local_profile[window_slice].sum()
            if block_sum > best_sum:
                best_sum = block_sum
                best_start = start_idx

        if best_start is None or best_sum <= 0:
            # no good window found
            break

        # we found a block
        start_indices.append(best_start)
        # fill discrete_profile
        offset = 0
        for ph in phases:
            d = int(round(ph.get("duration", 1)))
            e_kwh = float(ph.get("energy_kwh", 0.0))
            for h in range(d):
                idx = best_start + offset + h
                discrete_profile[idx] = e_kwh
            offset += d

        # mark that block as used so we don't pick it again
        used_indices[best_start : best_start + total_duration] = True

    # optional debug plot
    if debug_plot:
        hours = np.arange(len(continuous_profile))
        plt.figure(figsize=(9,4))
        plt.plot(hours, continuous_profile, 'o-', label="Continuous Profile")
        plt.step(hours, discrete_profile, where='post', label="Discrete Profile")
        plt.title(f"Multiple-Run Discretization for {device_name}")
        plt.xlabel("Hour")
        plt.ylabel("kWh")
        plt.grid(True)
        plt.legend()
        plt.show()

    return discrete_profile, start_indices


def discretize_profile(continuous_profile: np.ndarray, device_spec: dict,
                        off_value: float = 0.0) -> np.ndarray:
    """
    Discretize a continuous warm start profile into a discrete profile that respects the 
    device specification for initial scheduling.
    
    Instead of simply rounding to a binary 1.0, this function uses the device specification 
    to determine the total required on–hours (by summing the 'duration' fields in device_spec["phases"])
    and the total required energy (by summing the 'energy_kwh' fields). It then computes the average 
    energy required per on–hour. Finally, it selects exactly that many hours (those with the highest 
    continuous values) and assigns them the computed energy value, while the remaining hours are set 
    to off_value.
    
    Args:
        continuous_profile (np.ndarray): The continuous warm start profile (e.g., hourly optimized consumption).
        device_spec (dict): The device specification; must contain a key "phases", where each phase is a dict 
                            with keys "duration" (hours) and "energy_kwh" (total energy for that phase).
        off_value (float): The value to assign for “off” hours (default 0.0).
    
    Returns:
        np.ndarray: A discrete profile (of the same length as continuous_profile) with exactly the required 
                    number of on–hours set to the average required energy and off–hours set to off_value.
    """
    # Determine total required on–hours and total energy from device spec
    if "phases" in device_spec and device_spec["phases"]:
        total_required_hours = sum(float(phase.get("duration", 1)) for phase in device_spec["phases"])
        total_required_energy = sum(float(phase.get("energy_kwh", 0)) for phase in device_spec["phases"])
    else:
        # Fallback: if no phase info, assume device must be on all hours and use the average continuous value.
        total_required_hours = len(continuous_profile)
        total_required_energy = continuous_profile.mean() * len(continuous_profile)
    
    # Round required hours to an integer and ensure it does not exceed available periods.
    required_on_hours = int(round(total_required_hours))
    required_on_hours = min(required_on_hours, len(continuous_profile))
    
    # Compute the average energy per on-hour.
    average_on_energy = total_required_energy / required_on_hours if required_on_hours > 0 else 0.0
    
    # Select the indices with the highest continuous values.
    sorted_indices = np.argsort(continuous_profile)[::-1]
    on_indices = sorted_indices[:required_on_hours]
    
    # Build the discrete profile: assign average energy for selected indices, 0 for others.
    discrete_profile = np.full_like(continuous_profile, off_value, dtype=float)
    discrete_profile[on_indices] = average_on_energy
    return discrete_profile

def plot_discretization(continuous_profile: np.ndarray, discrete_profile: np.ndarray, title: str = "Warm Start Discretization"):
    """
    Plot the continuous warm start profile alongside the discretized profile for visual inspection.
    
    Args:
        continuous_profile (np.ndarray): The original continuous profile.
        discrete_profile (np.ndarray): The profile produced by discretize_profile.
        title (str): Title for the plot.
    """
    import matplotlib.pyplot as plt
    hours = np.arange(len(continuous_profile))
    plt.figure(figsize=(10, 5))
    plt.plot(hours, continuous_profile, label="Continuous Profile", marker="o")
    plt.step(hours, discrete_profile, label="Discrete Profile", where="mid", linewidth=2)
    plt.xlabel("Hour")
    plt.ylabel("Energy (kWh)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


###########################################################
#                FlexibleDevice
###########################################################
class FlexibleDevice:
    """
    A flexible device that can do:
      - Offline partial usage MILP with advanced battery SoC constraints
      - Next-day discrete phased MILP scheduling (also with optional battery constraints)
      - Optionally supports RL actions, though not shown in detail here
    """

    # Flex model-based constraints for min on/off times using device flexibility models
    MIN_ON_TIME_PER_FLEX_MODEL = {
        "discrete_phase": 1,    # Discrete devices need minimum on-time
        "partial_usage": 0,     # Partial usage can start/stop at any time  
        "fixed": 24             # Fixed devices are always on
    }
    MIN_OFF_TIME_PER_FLEX_MODEL = {
        "discrete_phase": 1,    # Discrete devices need minimum off-time
        "partial_usage": 0,     # Partial usage can start/stop at any time
        "fixed": 0              # Fixed devices never turn off
    }
    MAX_ON_TIME_PER_FLEX_MODEL = {
        "discrete_phase": 14,   # Discrete devices have limited operation time
        "partial_usage": 18,    # Partial usage can run longer but with limits
        "fixed": 24             # Fixed devices run continuously
    }
    PREFERENCE_PENALTY_WEIGHT = 10.0

    def __init__(self,
                 data: pd.DataFrame,
                 device_name: str,
                 category: str,
                 power_rating: float,
                 global_layer: Any,
                 max_shift_hours: int = 6,
                 is_flexible: bool = True,
                 battery_agent: Optional[Any] = None,
                 pv_agent: Optional[Any] = None,
                 spec: Optional[Dict[str,Any]] = None):
        """
        Combine advanced battery SoC approach + next-day phases + partial usage MILP

        Args:
            data: DataFrame with columns ['utc_timestamp', device_name, 'price_per_kwh', ...]
            device_name: string name for device (e.g. 'dishwasher')
            category: e.g. 'Partially Flexible', 'Highly Flexible'
            power_rating: maximum device power rating (kW)
            global_layer: global building load layer
            max_shift_hours: how many hours we can shift in partial usage MILP
            is_flexible: can we shift the device consumption
            battery_agent: advanced battery modeling class with .get_piecewise_segments() etc
            spec: dictionary with "phases" and "allowed_hours" if using next-day scheduling
        """
        self.device_name = device_name
        self.category = category
        self.power_rating = power_rating
        self.is_flexible = is_flexible
        self.battery_agent = battery_agent

        self.global_layer = global_layer
        self.max_shift_hours = max_shift_hours

        self.shifts = []
        self.savings = 0.0
        self.savings_total = 0.0
        self.original_consumption = None
        self.optimized_consumption = None
        self.iteration_consumption = {}

        # For partial usage vs. next-day mode
        self.weekday_optimized_schedule = None
        self.nextday_optimized_schedule = None
        self.offline_savings = 0.0
        self.nextday_savings = 0.0

        # Learning parameters
        self.conflict_count = {h: 0 for h in range(24)}
        self.success_rates = {h: {'success': 0, 'total': 0} for h in range(24)}
        self.price_sensitivity = 1.0
        self.preferred_hours = []
        # self.hour_probability = {h: 1.0/24.0 for h in range(24)}  # Initialize uniform probability distribution
        self.observation_count = 0
        self.last_entropy = 1.0  # Maximum entropy for uniform distribution
        self.estimated_preferred_hour = None
        self.probability_updates = []
        self.last_update_day = None

        # These lines store the battery solution after each solve
        if battery_agent is not None:
            self.battery_soc = np.full(len(data), battery_agent.current_soc)
        else:
            self.battery_soc = np.zeros(len(data))
        self.battery_charge = np.zeros(len(data))   # Charge in kW
        self.battery_discharge = np.zeros(len(data))# Discharge in kW

        # Save the device spec for later (this is needed for discretization).
        self.spec = spec if spec is not None else {}

        # If we have device phases, store them (for next-day discrete scheduling).
        if spec is not None:
            self.phases = spec.get("phases", [])
            self.allowed_hours = spec.get("allowed_hours", list(range(24)))
        else:
            self.phases = []
            self.allowed_hours = list(range(24))


        # Prepare data
        self._setup_milp_data(data)
        self.pv_columns = determine_pv_columns(self)
        self.pv_agent = pv_agent
        self.pv_forecast_nominal = 0.0
        self.pv_forecast_used = 0.0

    def _setup_milp_data(self, data: pd.DataFrame):
        """Set up necessary data columns, baseline consumption, etc."""
        self.data = data.copy()
        self.data['utc_timestamp'] = pd.to_datetime(self.data['utc_timestamp'])
        self.data['hour'] = self.data['utc_timestamp'].dt.hour
        self.data['day'] = self.data['utc_timestamp'].dt.date
        # optional: weekday, season, etc.:
        self.data['weekday'] = self.data['utc_timestamp'].dt.day_name()
        self.data['season'] = self.data['utc_timestamp'].apply(get_season)

        self.original_consumption = self.data[self.device_name].values.copy()
        self.optimized_consumption = self.original_consumption.copy()

        # For conflict resolution, we see if there's a typical max load
        pos_vals = self.original_consumption[self.original_consumption > 0]
        if len(pos_vals) > 0:
            self.max_load = np.percentile(pos_vals, 95)
        else:
            self.max_load = 0.001

    def update_preferred_hours(self):
        """Recalc after each iteration how well shifting to each hour has worked."""
        success_rates = {
            h: sr['success'] / max(1, sr['total']) for h, sr in self.success_rates.items()
        }
        adjusted = {
            h: success_rates[h] / (1 + self.conflict_count[h])
            for h in success_rates
        }
        self.preferred_hours = sorted(
            adjusted.keys(),
            key=lambda h: (adjusted[h], -self.conflict_count[h]),
            reverse=True
        )

    def learn_from_shift(self, shift: Dict[str,Any], success: bool):
        """Ingest shift result => update success rates & conflict counts => update price_sensitivity."""
        hour = shift['to_hour']
        self.success_rates[hour]['total'] += 1
        if success:
            self.success_rates[hour]['success'] += 1
        else:
            self.conflict_count[hour] += 1

        tot_atts = sum(sr['total'] for sr in self.success_rates.values())
        tot_succ = sum(sr['success'] for sr in self.success_rates.values())
        ratio = tot_succ / max(1, tot_atts)

        if ratio < 0.5:
            self.price_sensitivity *= 1.05
        else:
            self.price_sensitivity *= 0.95
        self.price_sensitivity = max(0.1, min(self.price_sensitivity, 5.0))
        self.update_preferred_hours()

    ###########################################################
    # Offline partial usage MILP with SoC-based battery
    ###########################################################
    def optimize_day(self, day: datetime.date, effective_prices: np.ndarray, pv_forecast: Optional[np.ndarray],
                     battery_state: Optional[dict] = None, grid_info: Optional[dict] = None) -> list:
        """
        Optimizes the device schedule for a given day using the MILP.
        Uses the PV forecast (from PVAgent.get_hourly_forecast_pv) as the expected PV production.
        The PV forecast is adjusted by multiplying by the number of PV columns (total PV capacity).
        After solving, it compares the adjusted forecast with the measured PV (summed over all PV columns)
        to compute a forecast error penalty. This penalty (using the actual price for each hour) is subtracted
        from the nominal savings to yield adjusted savings.
        Returns the list of shifts from the MILP.
        """
        import logging
        mask = (self.data['day'] == day)
        idxs = self.data[mask].index
        if len(idxs) < 1:
            logging.warning(f"No data for {self.device_name} on {day}")
            return []
        
        day_consumption = self.original_consumption[idxs]
        if len(effective_prices) == len(self.data):
            day_prices = effective_prices[idxs]
        else:
            day_prices = effective_prices[:len(idxs)]
        day_hours = self.data.loc[idxs, 'hour'].values

        # Add this at the beginning of optimize_day
        # Add this at the beginning of optimize_day
        if len(set(day_prices)) == 1:  # All prices are the same
            base_price = day_prices[0]
            day_prices = np.array(day_prices)  # Ensure it's a numpy array
            
            # Create realistic pattern with multiple peaks and valleys
            for h in range(len(day_prices)):
                if h < 6:  # Early morning (0-5) - very low
                    day_prices[h] = base_price * 0.7
                elif 6 <= h < 9:  # Morning ramp (6-8) - rising
                    day_prices[h] = base_price * 1.3
                elif 9 <= h < 14:  # Midday peak (9-13)
                    day_prices[h] = base_price * 1.1
                elif 14 <= h < 17:  # Afternoon dip (14-16)
                    day_prices[h] = base_price * 0.85
                elif 17 <= h < 21:  # Evening peak (17-20) - highest
                    day_prices[h] = base_price * 1.4
                else:  # Late night (21-23) - low
                    day_prices[h] = base_price * 0.7
        # Update the prices at the source for this specific day
        for i, idx in enumerate(idxs):
            self.data.loc[idx, 'price_per_kwh'] = day_prices[i]
        
        # Log that we've updated prices
        logging.info(f"Updated flat prices for {day} with synthetic pattern. Range: {day_prices.min():.4f}-{day_prices.max():.4f}")
        
        self.shifts = []
            
        # Adjust forecast by multiplying by the number of PV columns (if available).
        if pv_forecast is not None and self.pv_agent:
            num_pv = len(self.pv_columns)
            # Compute the nominal forecast for this device (scaled by the number of PV arrays)
            forecast_nominal = pv_forecast * num_pv  
            z_alpha = 1.645  # 95% confidence
            pv_error_std = self.pv_agent.compute_hourly_error_std(day) if self.pv_agent else 0.0
            Delta = z_alpha * pv_error_std  # error margin
            forecast_adjusted = forecast_nominal - Delta
        
            # Store the forecasts as attributes for later extraction
            self.pv_forecast_nominal = forecast_nominal
            self.pv_forecast_used = forecast_adjusted
        else:
            forecast_adjusted = pv_forecast
            self.pv_forecast_nominal = np.zeros(24)
            self.pv_forecast_used = np.zeros(24)
    
        # Use the adjusted forecast in the MILP model.
        shifts = self._solve_milp(consumption=day_consumption,
                                  prices=day_prices,
                                  hours=day_hours,
                                  indices=idxs,
                                  pv_profile=forecast_adjusted,
                                  battery_state=battery_state,
                                  grid_info=grid_info)
        self.shifts.extend(shifts)
    
        # Compute forecast error penalty based on the adjusted forecast.
        if forecast_adjusted is not None and self.pv_columns:
            # Instead of summing over the raw indices (which might be incomplete),
            # group the measured PV production by the hour of day and reindex to ensure 24 hours.
            df_day = self.data.loc[idxs, self.pv_columns].copy()
            # Also extract the hour information (ensure it is integer 0-23)
            df_day['hour'] = self.data.loc[idxs, 'hour'].astype(int)
            # Sum the PV production over all PV columns for each hour:
            measured_pv_series = df_day.groupby('hour').sum().sum(axis=1)
            # Reindex to cover all 24 hours; fill missing hours with 0.
            measured_pv = measured_pv_series.reindex(range(24), fill_value=0).values
            # Now compute forecast error: both arrays are length 24.
            forecast_error = abs(forecast_adjusted - measured_pv)
            
            # Ensure day_prices is also length 24 to match forecast_error
            if len(day_prices) != 24:
                # Reindex day_prices to 24 hours using the hour information
                day_hours_series = pd.Series(day_prices, index=day_hours)
                day_prices_24h = day_hours_series.reindex(range(24), fill_value=day_prices.mean()).values
            else:
                day_prices_24h = day_prices
            
            penalty_cost = sum(forecast_error * day_prices_24h)
            # print(f"[{self.device_name}] Penalty cost: {penalty_cost:.4f}")
            # # print("Day prices:", day_prices)
            # print("Forecast error:", forecast_error)
            self.forecast_error_penalty = penalty_cost
            self.adjusted_savings = self.savings - penalty_cost
        else:
            self.forecast_error_penalty = 0.0
            self.adjusted_savings = self.savings

        # Set optimized_schedule from optimized_consumption for the day
        if hasattr(self, 'optimized_consumption') and self.optimized_consumption is not None:
            # Extract optimized schedule for the specific day
            day_indices = idxs
            if len(day_indices) == 24:
                self.optimized_schedule = self.optimized_consumption[day_indices].tolist()
            else:
                # If not exactly 24 hours, pad or truncate to 24 hours
                day_consumption = self.optimized_consumption[day_indices]
                self.optimized_schedule = np.pad(day_consumption, (0, max(0, 24 - len(day_consumption))), 'constant')[:24].tolist()
        else:
            # Fallback: use original consumption if optimization failed
            self.optimized_schedule = [0.0] * 24
    
        return shifts
    
    
    def _solve_milp(self,
                    consumption: np.ndarray,
                    prices: np.ndarray,
                    hours: np.ndarray,
                    indices: np.ndarray,
                    pv_profile: Optional[np.ndarray] = None,
                    battery_state: Optional[Dict[str, Any]] = None,
                    grid_info: Optional[Dict[str, Any]] = None,
                    coord_signals: Optional[List[Dict]] = None,
                    battery_preferences: Optional[Dict] = None,
                    force_arbitrage: bool = True) -> List[Dict]:
        """
        CRITICAL FIX: Added force_arbitrage parameter to directly enforce charging at low prices
        and discharging at high prices when needed.
        
        The implementation uses a combination of:
        1. Binary variable y[t] to prevent simultaneous charging/discharging
        2. Direct economic incentive in the objective function
        3. Adaptive constraints based on battery state and price signals
        """
        """
        Merged partial usage MILP that now:
          - Allows shifting of consumption (x[t, h])
          - Incorporates advanced battery SoC constraints (using piecewise segments if available)
          - Enforces global load constraints
          - Computes grid export/inport based on excess PV (and battery flows)
          - Adds a degradation cost term (proportional to battery throughput)
        
        New variables:
          - g_plus[t]: grid import (nonnegative)
          - g_minus[t]: grid export (nonnegative)
          
        They satisfy:
            g_plus[t] - g_minus[t] = consumption[t] + charge[t] - discharge[t] + pv_term[t]
        where pv_term[t] = -pv_profile[t] (since pv_profile is negative when generating)
        
        The objective now minimizes:
            sum_t [ adjusted_price[t]*x[t, h] + degradation_cost*(charge[t] + discharge[t])
                    + g_plus[t]*price[t] - g_minus[t]*export_price ]
        (export_price is taken from self.grid_agent.get_grid_info()['export_price'])
        """
        from pulp import LpProblem, LpVariable, lpSum, LpMinimize, PULP_CBC_CMD, LpStatus
    
        n_periods = len(indices)
        if n_periods < 2:
            return []
    
        max_shift = min(self.max_shift_hours, n_periods - 1)
        if pv_profile is None:
            pv_profile = np.zeros(n_periods)
    
        # Create the LP
        prob = LpProblem(f"{self.device_name}_MILP", LpMinimize)
    
        # Decision variables for shifting consumption
        x = LpVariable.dicts("x",
                             ((t, h) for t in range(n_periods)
                              for h in range(-max_shift, max_shift + 1)
                              if 0 <= t + h < n_periods),
                             lowBound=0)
    
        # Optional binary variables for all-or-nothing constraints based on flex_model
        device_on = None
        flex_model = self.spec.get("flex_model", "fixed") if hasattr(self, 'spec') and self.spec else "fixed"
        
        if flex_model in self.MIN_ON_TIME_PER_FLEX_MODEL:
            device_on = LpVariable.dicts(f"{self.device_name}_on", range(n_periods), cat="Binary")
            min_on_time = self.MIN_ON_TIME_PER_FLEX_MODEL[flex_model]
            
            # Only apply constraints if min_on_time > 0
            if min_on_time > 0:
                for t in range(n_periods):
                    for h in range(-max_shift, max_shift + 1):
                        if (t, h) in x:
                            prob += x[t, h] <= consumption[t] * device_on[t], f"AllOrNothing_{t}_{h}"
                for t in range(n_periods - min_on_time + 1):
                    for i in range(1, min_on_time):
                        prob += device_on[t] <= device_on[t + i], f"MinOn_{t}_{i}"
                        
            if flex_model in self.MIN_OFF_TIME_PER_FLEX_MODEL:
                min_off_time = self.MIN_OFF_TIME_PER_FLEX_MODEL[flex_model]
                if min_off_time > 0:
                    for t in range(n_periods - min_off_time):
                        for i in range(1, min_off_time + 1):
                            prob += (1 - device_on[t]) <= (1 - device_on[t + i]), f"MinOff_{t}_{i}"
        obj_terms = []
        # Prepare battery variables if battery_state is provided
        charge = {}
        discharge = {}
        soc = {}
        if battery_state is not None and self.battery_agent is not None:
            charge = LpVariable.dicts("charge", range(n_periods), lowBound=0)
            discharge = LpVariable.dicts("discharge", range(n_periods), lowBound=0)
            soc = LpVariable.dicts("soc", range(n_periods),
                                lowBound=battery_state['soc_min'],
                                upBound=battery_state['soc_max'])
            
            # Add binary variable y[t] = 1 if charging, 0 if discharging to prevent simultaneous charging/discharging
            y = LpVariable.dicts("y", range(n_periods), cat="Binary")
            
            # Add all battery constraints using the centralized function
            prob, battery_cost_terms = self.battery_agent.add_battery_constraints_to_milp(
                prob=prob,
                battery_state=battery_state,
                n_periods=n_periods,
                charge=charge,
                discharge=discharge,
                soc=soc,
                prices=prices,
                y=y,
                cost_terms=obj_terms,
                force_arbitrage=force_arbitrage,
                problem_type="standard"
            )
            
            # Update cost_terms with battery-related costs
            obj_terms = battery_cost_terms
            
            # Battery hourly_plan constraints are now handled by add_battery_constraints_to_milp
            
            # All binary operation constraints and SOC evolution constraints are now handled by add_battery_constraints_to_milp
    
        # End battery variable preparation
    
        # ------------------ NEW: Grid Export Variables ------------------
        # We no longer use grid_import/export from grid_info.
        # Instead, we compute grid net flow from device consumption, battery flows, and PV production.
        # Define:
        #   g_plus[t] >= 0 (grid import)
        #   g_minus[t] >= 0 (grid export)
        # with the constraint:
        #   g_plus[t] - g_minus[t] = consumption[t] + charge[t] - discharge[t] + pv_term[t]
        # where pv_term[t] = - pv_profile[t]   (since pv_profile is negative when producing)
        g_plus = {t: LpVariable(f"g_plus_{t}", lowBound=0) for t in range(n_periods)}
        g_minus = {t: LpVariable(f"g_minus_{t}", lowBound=0) for t in range(n_periods)}
        for t in range(n_periods):
            consumption_at_hour_t = lpSum(x[t_, h_] for (t_, h_) in x if t_ + h_ == t)
            pv_term = -pv_profile[t]  # pv_term is positive when there is PV production
            prob += g_plus[t] - g_minus[t] == consumption_at_hour_t + (charge[t] if battery_state is not None else 0) \
                                                  - (discharge[t] if battery_state is not None else 0) + pv_term, f"GridBalance_{t}"
    
        # ------------------ End NEW Grid Export ------------------
    
        # Conflict-based price scaling (for the shifting variables)
        total_conflicts = sum(self.conflict_count.values()) or 1
        conflict_factor = 0.5
        adjusted_prices = np.array([
            prices[i] * self.price_sensitivity * (1 + conflict_factor * self.conflict_count[hours[i]] / total_conflicts)
            for i in range(n_periods)
        ])
    
        # Build objective:
        # - cost from shifting consumption (as before),
        # - plus degradation cost for battery cycling (if battery_state provided),
        # - plus cost for grid import and revenue for grid export.
        degradation_cost = battery_state.get('degradation_cost', 0.0) if battery_state is not None else 0.0
        export_price = self.global_layer.export_price if hasattr(self.global_layer, 'export_price') else 0.0
    
        # Cost from shifting (as before)
        obj_terms.append(lpSum(adjusted_prices[t+h] * x[t, h]
                               for t in range(n_periods)
                               for h in range(-max_shift, max_shift + 1)
                               if (t, h) in x))
        # Battery degradation cost is now handled by add_battery_constraints_to_milp
        # No additional battery cost terms needed here
        # Grid cost/revenue: cost for grid import minus revenue for grid export
        
        if battery_state is not None:
            # Grid cost term (excluding battery components which are handled by battery_constraints_to_milp)
            grid_term = lpSum(g_plus[t] * prices[t] - g_minus[t] * export_price for t in range(n_periods))
            obj_terms.append(grid_term)
        else:
            # Standard grid cost if no battery
            obj_terms.append(lpSum(g_plus[t] * prices[t] - g_minus[t] * export_price for t in range(n_periods)))
        prob += lpSum(obj_terms), "TotalCost"
    
        # Add constraints for shifting variables as before:
        for t in range(n_periods):
            consumption_at_hour_t = lpSum(x[t_, h_] for (t_, h_) in x if t_ + h_ == t)
            for h in range(-max_shift, max_shift + 1):
                if (t, h) in x:
                    prob += x[t, h] <= consumption_at_hour_t, f"MaxX_{t}_{h}"
            est = self.global_layer.get_average_load([hours[t]])[0]
            current_load = self.global_layer.hourly_load[hours[t]]
            max_allow = self.global_layer.max_building_load - current_load
            prob += (lpSum(x[t-h, h] for h in range(-max_shift, max_shift+1) if (t-h, h) in x)  + (charge[t] if battery_state is not None else 0) - (discharge[t] if battery_state is not None else 0) ) <= max_allow, f"GlobalLoad_{t}"
            # shifted_load = lpSum(x[t_, h_] for (t_, h_) in x if t_ == t)
            # Link the shifted load with battery flows (if battery present)
            # prob += shifted_load == consumption[t], f"LoadCons_{t}"
            # if battery_state is not None and self.battery_agent is not None:
            #     SoC_incr_var = f"SoC_incr_{t}"
            #     SoC_decr_var = f"SoC_decr_{t}"
            #     SoC_incr_lp = [v for v in prob.variables() if v.name == SoC_incr_var]
            #     SoC_decr_lp = [v for v in prob.variables() if v.name == SoC_decr_var]
            #     if SoC_incr_lp and SoC_decr_lp:
            #         prob += shifted_load + SoC_decr_lp[0] - SoC_incr_lp[0] == consumption[t], f"LoadCons_{t}"
            #     else:
            #         prob += shifted_load == consumption[t], f"LoadCons_{t}_NoBatteryPiecewise"
            # elif battery_state is not None:
            #     prob += (shifted_load + discharge[t] - charge[t]) == consumption[t], f"LoadCons_{t}"
            # else:
            #     prob += shifted_load == consumption[t], f"LoadCons_{t}"
            # 1. Keep the definition of shifted_load (representing consumption originating from hour t)
            shifted_load = lpSum(x[t_, h_] for (t_, h_) in x if t_ == t)

            # 2. Add a global conservation constraint for the entire day's energy (do this only once)
            if t == 0:  # Only add this constraint once
                total_original = lpSum(consumption)
                total_shifted = lpSum(x[t_, h_] for (t_, h_) in x)
                prob += total_shifted == total_original, f"TotalLoadConservation"
                
           
    
        # Solve the MILP
        prob.solve(PULP_CBC_CMD(msg=False, timeLimit=300, presolve='on', cuts='on'))
        if LpStatus[prob.status] != 'Optimal':
            # print(f"[{self.device_name}] MILP not optimal: {LpStatus[prob.status]}")
            return []
    
        # Process solution and record shifts
        self.optimized_consumption[indices] = 0.0
        shifts = []
        for t in range(n_periods):
            for h in range(-max_shift, max_shift + 1):
                if (t, h) in x:
                    val = x[t, h].varValue
                    if val and val > 0:
                        target_idx = t + h
                        orig_cost = prices[t] * val
                        new_cost = prices[target_idx] * val
                        # We no longer discount PV in the objective since it is now explicitly modeled via grid export.
                        shift_savings = orig_cost - new_cost
                        # self.savings += shift_savings
                        self.optimized_consumption[indices[target_idx]] += val
                        sdict = {
                            'device': self.device_name,
                            'from_hour': hours[t],
                            'to_hour': hours[target_idx],
                            'amount': val,
                            'savings': 0.0,
                            'price': prices[target_idx],
                            'success': True
                        }
                        shifts.append(sdict)
                        self.learn_from_shift(sdict, True)

        if LpStatus[prob.status] == 'Optimal':
            # 1) Compute the new total cost from the MILP solution
        
            # Grid cost: sum of grid imports minus exports
            export_price = 0.0
            if grid_info and 'export_price' in grid_info:
                export_price = grid_info['export_price']
        
            new_cost_grid = 0.0
            for t in range(n_periods):
                gp = g_plus[t].varValue if g_plus[t].varValue else 0.0
                gm = g_minus[t].varValue if g_minus[t].varValue else 0.0
                new_cost_grid += (gp * prices[t])      # pay for grid import
                new_cost_grid -= (gm * export_price)   # revenue from export
        
            # Battery degradation cost (if you have a degrade rate or cost)
            new_cost_degrade = 0.0
            degradation_cost = battery_state.get('degradation_cost', 0.0) if battery_state else 0.0
            if battery_state and degradation_cost > 0:
                for t in range(n_periods):
                    cval = charge[t].varValue if charge and charge[t].varValue else 0.0
                    dval = discharge[t].varValue if discharge and discharge[t].varValue else 0.0
                    new_cost_degrade += degradation_cost * (cval + dval)
        
            total_new_cost = new_cost_grid + new_cost_degrade
        
            # 2) Compute the old (baseline) cost if we had NOT shifted or used the battery
            #    => i.e., sum of day_consumption[t] * prices[t].
            day_baseline_cost = 0.0
            for t in range(n_periods):
                day_baseline_cost += consumption[t] * prices[t]
        
            # 3) The day’s net savings
            day_savings = day_baseline_cost - total_new_cost
            self.savings += day_savings

            for sdict in shifts:
                sdict["savings"] = day_savings
        
            if battery_state is not None:
                logging.info(
                    f"[{self.device_name}] Partial usage day (with battery) => oldCost={day_baseline_cost:.2f}, "
                    f"newCost={total_new_cost:.2f}, daySavings={day_savings:.2f}")
            else:
                logging.info(
                    f"[{self.device_name}] Partial usage day (without battery) => oldCost={day_baseline_cost:.2f}, "
                    f"newCost={total_new_cost:.2f}, daySavings={day_savings:.2f}")
        else:
            logging.warning(f"[{self.device_name}] MILP did not solve optimally => {LpStatus[prob.status]}")
            return []

        # Record battery state and update history for plotting
        if battery_state is not None:
            for t in range(n_periods):
                soc_val = soc[t].varValue if soc[t].varValue is not None else battery_state['current_soc']
                charge_val = charge[t].varValue if charge[t].varValue is not None else 0.0
                discharge_val = discharge[t].varValue if discharge[t].varValue is not None else 0.0
                # print(f"[Device: {self.device_name} | Time index {t}] SOC: {soc_val:.2f} kWh, Charge: {charge_val:.2f} kW, Discharge: {discharge_val:.2f} kW")
                self.battery_soc[indices[t]] = soc_val
                self.battery_charge[indices[t]] = charge_val
                self.battery_discharge[indices[t]] = discharge_val
    
            day_throughput = sum((charge[t].varValue or 0) + (discharge[t].varValue or 0) for t in range(n_periods))
            self.battery_agent.cycle_count += day_throughput / self.battery_agent.estimated_capacity
            total_charge = sum(charge[t].varValue or 0 for t in range(n_periods))
            total_discharge = sum(discharge[t].varValue or 0 for t in range(n_periods))
            self.battery_agent.charge_history.append(total_charge)
            self.battery_agent.discharge_history.append(total_discharge)
            final_soc = soc[n_periods - 1].varValue if soc[n_periods - 1].varValue is not None else battery_state['current_soc']
            self.battery_agent.current_soc = final_soc
            self.battery_agent.soc_history.append(final_soc)
            # print(f"[Device: {self.device_name}] End-of-day SOC: {final_soc:.2f} kWh")
        return shifts

    ###########################################################
    # Next-day aggregator pass (discrete phases)
    ###########################################################
    # def optimize_aggregated_day(self,
    #                             agg_data: pd.DataFrame,
    #                             pv_profile: Optional[np.ndarray] = None,
    #                             pv_forecast: Optional[np.ndarray] = None,
    #                             weather_forecasts: Optional[dict] = None,
    #                             battery_state: Optional[Dict] = None,
    #                             grid_info: Optional[Dict] = None,
    #                             max_runs: int = 2) -> List[Dict]:
    #     """
    #     Next-day optimization using discrete phases (from second snippet).
    #     We reuse the same advanced battery constraints if battery_state is not None.
    #     """
    #     if len(agg_data) != 24:
    #         raise ValueError("Aggregated data must have exactly 24 rows.")

    #     # Warm start
    #     if len(self.optimized_consumption) == len(self.data):
    #         warm_df = pd.DataFrame({'hour': self.data['hour'],
    #                                 'warm_consumption': self.optimized_consumption})
    #         warm_agg = warm_df.groupby('hour')['warm_consumption'].mean().reset_index()
    #         agg_data[self.device_name] = warm_agg['warm_consumption'].values
    #         self.weekday_optimized_schedule = warm_agg['warm_consumption'].values
    #     else:
    #         self.weekday_optimized_schedule = agg_data[self.device_name].values

    #     # print(agg_data)
    #     # self.weekday_optimized_schedule = agg_data["optimized"].values
        
    #     discrete_warm, start_indices = discretize_profile_with_phases_multi(
    #         continuous_profile=self.weekday_optimized_schedule,
    #         device_spec=self.spec,
    #         max_runs=max_runs,
    #         device_name=self.device_name,
    #         debug_plot=False
    #     )
    #     # For now, choose the first run's start index as the MILP warm start.
    #     if start_indices:
    #         warm_start_index = start_indices[0]
    #     else:
    #         warm_start_index = 0

    #     # # Use the device's spec (which should be provided in self.spec or similar)
    #     # discrete_warm_start = discretize_profile(self.weekday_optimized_schedule, self.spec)
    #     # # Optionally, plot for debugging:
    #     # plot_discretization(self.weekday_optimized_schedule, discrete_warm_start,
    #     #                     title=f"Warm Start Discretization for {self.device_name}")
        
    #     self.weekday_optimized_schedule = discrete_warm

    #     if np.all(np.abs(self.weekday_optimized_schedule) < 1e-6):
    #         logging.warning(f"weekday_optimized_schedule for {self.device_name} contains all zeros. Check optimization.")


    #     day_prices = agg_data['price_per_kwh'].values
    #     day_hours = agg_data['hour'].values
    #     idxs = np.arange(24)

    #     self.shifts = []
        
    #     # call the phased MILP with battery
    #     shifts = self._solve_milp_phases(
    #         prices=agg_data['price_per_kwh'].values,
    #         hours=agg_data['hour'].values,
    #         indices=np.arange(24),
    #         pv_profile=pv_profile,
    #         pv_forecast=pv_forecast,
    #         weather_forecasts=weather_forecasts,
    #         battery_state=battery_state,
    #         grid_info=grid_info,
    #         warm_start_index=warm_start_index  # <--- new param
    #     )
    #     self.shifts.extend(shifts)
    #     self.nextday_optimized_schedule = self.optimized_consumption[idxs].copy()
    #     self.nextday_pv_forecast = pv_forecast
    #     if np.all(np.abs(self.nextday_optimized_schedule) < 1e-6):
    #         logging.warning(f"nextday_optimized_schedule for {self.device_name} contains all zeros. Check optimization.")

    #     logging.info(f"{self.device_name} Next-Day schedule => {self.nextday_optimized_schedule}")
    #     logging.info(f"{self.device_name} Next-Day iteration => {self.savings:.4f} € savings")
    #     self.nextday_savings = self.savings
    #     self.savings = 0.0
    #     return shifts

    def _get_device_data_for_day(self, day_index):
        """
        Helper method to get device data for a specific day
        
        Args:
            day_index: Index in the data for the day to get
            
        Returns:
            DataFrame with day's data or None if not enough data
        """
        day = self.data.loc[day_index, 'day']
        day_mask = (self.data['day'] == day)
        device_data = self.data[day_mask].copy()
        
        if len(device_data) < 24:
            return None
            
        return device_data
    
    def optimize(self, day_index, use_battery=True):
        """
        Original optimization method for compatibility
        
        Args:
            day_index: Index of the day to optimize
            use_battery: Whether to use battery in optimization
            
        Returns:
            List of shifts if successful, None otherwise
        """
        day = self.data.loc[day_index, 'day']
        
        # Get battery state if needed
        battery_state = None
        if use_battery and self.battery_agent is not None:
            battery_state = self.battery_agent.get_battery_state()
            
        # Get PV forecast if available
        pv_forecast = None
        if self.pv_agent is not None:
            pv_forecast = self.pv_agent.get_hourly_forecast_pv(day)
            
        # Call optimize_day with the day
        return self.optimize_day(day, 
                                self.data['price_per_kwh'].values, 
                                pv_forecast, 
                                battery_state, 
                                None)
    
    def optimize_with_global_constraints(self, day_index, use_battery=True, advanced_coordination=True):
        """
        Enhanced optimization with stronger global load awareness and battery coordination
        
        Args:
            day_index: Index of the day to optimize
            use_battery: Whether to use battery in optimization
            advanced_coordination: Whether to use enhanced coordination features
            
        Returns:
            List of shifts if successful, None otherwise
        """
        # If not using advanced coordination, fall back to existing method
        if not advanced_coordination:
            return self.optimize(day_index, use_battery)
        
        # Get the current global load profile and coordination signals
        day = self.data.loc[day_index, 'day']
        mask = (self.data['day'] == day)
        day_idxs = self.data[mask].index
        
        if len(day_idxs) < 24:
            return None
            
        # Get device-specific data
        device_data = self._get_device_data_for_day(day_index)
        if device_data is None or len(device_data) < 24:
            return None
            
        # Get global coordination signals for each hour
        day_hours = self.data.loc[day_idxs, 'hour'].values
        coord_signals = [self.global_layer.get_device_coordination_signal(h) for h in day_hours]
        
        # Extract prices for the day
        day_prices = self.data.loc[day_idxs, 'price_per_kwh'].values
        
        # Initialize battery coordination if available
        battery_state = None
        battery_preferences = None
        
        if use_battery and self.battery_agent is not None:
            battery_state = self.battery_agent.get_battery_state()
            # Get battery preferences for charging and discharging
            charge_prefs = self.battery_agent.get_price_aware_charge_preferences(day_prices)
            discharge_prefs = self.battery_agent.get_price_aware_discharge_preferences(day_prices)
            battery_preferences = {
                'charge': charge_prefs,
                'discharge': discharge_prefs
            }
        
        # Get PV forecast if available
        pv_profile = None
        if self.pv_agent is not None:
            pv_profile = self.pv_agent.get_hourly_forecast_pv(day)
        
        # Call our original optimization but with additional parameters
        shifts = self._solve_milp(
            consumption=device_data[self.device_name].values,
            prices=day_prices, 
            hours=day_hours,
            indices=day_idxs,
            pv_profile=pv_profile,
            battery_state=battery_state,
            grid_info=None,  # We'll handle grid interaction differently
            coord_signals=coord_signals,  # New parameter for enhanced coordination
            battery_preferences=battery_preferences  # New parameter for battery coordination
        )
        
        return shifts
    
    # def _solve_milp_phases(self,
    #                         prices: np.ndarray,
    #                         hours: np.ndarray,
    #                         indices: np.ndarray,
    #                         pv_profile: Optional[np.ndarray] = None,
    #                         pv_forecast: Optional[np.ndarray] = None,
    #                         weather_forecasts: Optional[dict] = None,
    #                         battery_state: Optional[Dict] = None,
    #                         grid_info: Optional[Dict] = None,
    #                         warm_start_index: Optional[int] = None,
    #                         force_arbitrage: bool = True) -> List[Dict]:
    #         """
    #         CRITICAL FIX: Added force_arbitrage parameter to directly enforce battery arbitrage when needed.
            
    #         The implementation uses a combination of:
    #         1. Binary variable y[t] to prevent simultaneous charging/discharging
    #         2. Direct economic incentive in the objective function 
    #         3. Adaptive constraints based on battery state to avoid infeasibility
    #         4. Additional objective terms to encourage proper arbitrage behavior
    #         """
    #         """
    #         Discrete on/off phases approach for next-day scheduling with robust battery integration.
    #         In addition to scheduling device phases, this MILP incorporates battery constraints using
    #         charge, discharge, and state-of-charge (SoC) variables. A binary variable y[t] enforces that
    #         the battery cannot charge and discharge simultaneously. Battery degradation cost and a bonus term 
    #         that rewards high SoC during low-price hours are included in the objective.
    #         """
    #         from pulp import LpProblem, LpMinimize, LpVariable, LpStatus, lpSum, PULP_CBC_CMD
    #         import numpy as np
    #         import logging

    #         cost_terms = []

    #         n = len(indices)
    #         if n < 1:
    #             return []
    #         phases = self.phases
    #         P = len(phases)
    #         if P == 0:
    #             return []

    #         prob = LpProblem(f"{self.device_name}_PHASE_MILP", LpMinimize)

    #         # Binary variables X[(p,k)] => phase p starts at hour k.
    #         X = {}
    #         for p in range(P):
    #             for k in range(n):
    #                 X[(p, k)] = LpVariable(f"X_{self.device_name}_p{p}_k{k}", cat="Binary")
    #         # Enforce allowed hours.
    #         for p in range(P):
    #             for k in range(n):
    #                 if hours[k] not in self.allowed_hours:
    #                     prob += X[(p, k)] == 0, f"AllowedHours_{p}_{k}"

    #         # --- Battery Variables ---
    #         # We create battery variables only if battery_state is provided.
    #         if battery_state is not None:
    #             charge = LpVariable.dicts("charge", range(n), lowBound=0)
    #             discharge = LpVariable.dicts("discharge", range(n), lowBound=0)
    #             soc = LpVariable.dicts("soc", range(n),
    #                                 lowBound=battery_state['soc_min'],
    #                                 upBound=battery_state['soc_max'])
    #             # Binary variable y[t]: y[t]=1 means charging, y[t]=0 means discharging.
    #             y = LpVariable.dicts("y", range(n), cat="Binary")
                
    #             # Add all battery constraints using the centralized function
    #             prob, battery_cost_terms = self.battery_agent.add_battery_constraints_to_milp(
    #                 prob=prob,
    #                 battery_state=battery_state,
    #                 n_periods=n,
    #                 charge=charge,
    #                 discharge=discharge,
    #                 soc=soc,
    #                 prices=prices,
    #                 y=y,
    #                 cost_terms=cost_terms,
    #                 force_arbitrage=force_arbitrage,
    #                 problem_type="phases"
    #             )
                
    #             # Update cost_terms with battery-related costs
    #             cost_terms = battery_cost_terms
                
    #             # All battery constraints are now handled by the centralized function
    #             # No additional battery constraints needed here
    #         # --- End Battery Variables ---

    #         # --- Objective Function for Phases and Battery ---
    #         # Phase scheduling cost terms - we'll add to the existing battery cost terms
    #         # Do NOT reset cost_terms here
    #         on_penalty = 0.01  # small penalty for turning device on
    #         for p in range(P):
    #             dur = phases[p]["duration"]
    #             e_kwh = phases[p]["energy_kwh"]
    #             for k in range(n):
    #                 run_hours = range(k, min(k + dur, n))
    #                 base_cost = sum(prices[rh] for rh in run_hours) * e_kwh
    #                 # Apply PV forecast discount if available.
    #                 if pv_forecast is not None:
    #                     potential = min(abs(pv_forecast[k]) / 32704.0, 1.0)
    #                     discount = forecast_discount(potential)
    #                 else:
    #                     discount = 0.0
    #                 # Optionally incorporate weather effects.
    #                 factor = 1.0
    #                 if weather_forecasts is not None:
    #                     for col_name, arr24 in weather_forecasts.items():
    #                         val = arr24[k]
    #                         if ('temperature' in col_name.lower()) and val < 25:
    #                             # Reward low temperature (cheaper hours) with slight bonus.
    #                             factor -= 0.01 * (25 - val)
    #                 adj_cost = base_cost * (1 - discount) * factor
    #                 cost_terms.append((adj_cost + on_penalty) * X[(p, k)])
    #         # Combine all cost terms.
    #         prob += lpSum(cost_terms), "TotalCost"

    #         # --- Phase Scheduling Constraints ---
    #         # Each phase must start exactly once.
    #         for p in range(P):
    #             prob += lpSum(X[(p, k)] for k in range(n)) == 1, f"PhaseStart_{p}"
    #         # Consecutive ordering: enforce that phase p+1 starts after phase p.
    #         for p in range(P - 1):
    #             durp = phases[p]["duration"]
    #             prob += lpSum(k * X[(p + 1, k)] for k in range(n)) == lpSum((k + durp) * X[(p, k)] for k in range(n)), f"Consecutive_{p}"

    #         # --- Building Load Constraints ---
    #         for t in range(n):
    #             est = self.global_layer.get_average_load([hours[t]])[0]
    #             max_allow = self.global_layer.max_building_load - est
    #             active_terms = []
    #             for p in range(P):
    #                 durp = phases[p]["duration"]
    #                 peakp = phases[p]["peak_kw"]
    #                 for k in range(n):
    #                     if t >= k and t < k + durp:
    #                         active_terms.append(peakp * X[(p, k)])
    #             if battery_state is not None:
    #                 prob += lpSum(active_terms) + charge[t] - discharge[t] <= max_allow, f"bldgLoad_{t}"
    #             else:
    #                 prob += lpSum(active_terms) <= max_allow, f"bldgLoad_{t}"

    #         # Optional sequencing constraints for phases.
    #         for p in range(P - 1):
    #             durp = phases[p]["duration"]
    #             for k1 in range(n):
    #                 for k2 in range(n):
    #                     if k2 < k1 + durp:
    #                         prob += X[(p, k1)] + X[(p + 1, k2)] <= 1, f"seq_{p}_{k1}_{k2}"

    #         # Warm start if provided.
    #         if warm_start_index is not None:
    #             offset = 0
    #             for p in range(P):
    #                 d = int(round(phases[p].get("duration", 1)))
    #                 if warm_start_index + offset < n:
    #                     X[(p, warm_start_index + offset)].setInitialValue(1.0)
    #                 offset += d

    #         # Solve the MILP.
    #         try:
    #             solver = PULP_CBC_CMD(msg=False, timeLimit=300, presolve='on', cuts='on')
    #             prob.solve(solver)
    #         except Exception as e:
    #             debug_filename = f"{self.device_name}_debug.lp"
    #             prob.writeLP(debug_filename)
    #             logging.error(f"Error solving MILP for device {self.device_name}. Model written to {debug_filename}. Exception: {e}")
    #             return []

    #         if LpStatus[prob.status] != 'Optimal':
    #             logging.warning(f"MILP for device {self.device_name} returned status {LpStatus[prob.status]}.")
    #             debug_filename = f"{self.device_name}_nonoptimal.lp"
    #             prob.writeLP(debug_filename)
    #             logging.warning(f"Non-optimal model written to {debug_filename}.")
    #             return []

    #         # Process results: reconstruct phase schedule.
    #         final_usage = np.zeros(n)
    #         shifts = []
    #         for p in range(P):
    #             durp = phases[p]["duration"]
    #             e_kwh = phases[p]["energy_kwh"]
    #             for k in range(n):
    #                 if X[(p, k)].varValue > 0.5:
    #                     # Distribute the energy evenly over the phase duration.
    #                     for hh in range(k, min(k + durp, n)):
    #                         final_usage[hh] += e_kwh / durp
    #                     orig_cost = prices[k] * (e_kwh / durp)
    #                     new_cost = orig_cost
    #                     if pv_forecast is not None:
    #                         potential = min(abs(pv_forecast[k]) / 32704.0, 1.0)
    #                         disc = forecast_discount(potential)
    #                         new_cost *= (1 - disc)
    #                     shift_savings = orig_cost - new_cost
    #                     self.savings += shift_savings
    #                     self.optimized_consumption[indices[k]] += (e_kwh / durp)
    #                     sdict = {
    #                         'device': self.device_name,
    #                         'phase': p,
    #                         'start_hour': hours[k],
    #                         'to_hour': hours[k],
    #                         'duration': durp,
    #                         'energy_kwh': e_kwh,
    #                         'savings': shift_savings,
    #                         'price': prices[k],
    #                         'success': True
    #                     }
    #                     shifts.append(sdict)
    #                     self.learn_from_shift(sdict, True)
    #         self.nextday_optimized_schedule = final_usage.copy()

    #         # Process battery results if battery_state is provided.
    #         if battery_state is not None:
    #             # For each hour, update battery variables in device's arrays.
    #             for t in range(n):
    #                 # Assuming indices[t] gives the corresponding time index.
    #                 idx = indices[t]
    #                 if idx != -1:
    #                     soc_val = soc[t].varValue if soc[t].varValue is not None else battery_state['current_soc']
    #                     charge_val = charge[t].varValue if charge[t].varValue is not None else 0.0
    #                     discharge_val = discharge[t].varValue if discharge[t].varValue is not None else 0.0
    #                     self.battery_soc[idx] = soc_val
    #                     self.battery_charge[idx] = charge_val
    #                     self.battery_discharge[idx] = discharge_val
    #             day_throughput = sum((charge[t].varValue or 0) + (discharge[t].varValue or 0) for t in range(n))
    #             self.battery_agent.cycle_count += day_throughput / self.battery_agent.estimated_capacity
    #             total_charge = sum(charge[t].varValue or 0 for t in range(n))
    #             total_discharge = sum(discharge[t].varValue or 0 for t in range(n))
    #             self.battery_agent.charge_history.append(total_charge)
    #             self.battery_agent.discharge_history.append(total_discharge)
    #             final_soc = soc[n - 1].varValue if soc[n - 1].varValue is not None else battery_state['current_soc']
    #             self.battery_agent.current_soc = final_soc
    #             self.battery_agent.soc_history.append(final_soc)

    #         return shifts

    def optimize_aggregated_day(self,
                                agg_data: pd.DataFrame,
                                pv_profile: Optional[np.ndarray] = None,
                                pv_forecast: Optional[np.ndarray] = None,
                                weather_forecasts: Optional[dict] = None,
                                battery_state: Optional[Dict] = None,
                                grid_info: Optional[Dict] = None) -> List[Dict]:
        """
        Next-day optimization using discrete phases (from second snippet).
        We reuse the same advanced battery constraints if battery_state is not None.
        """
        if len(agg_data) != 24:
            raise ValueError("Aggregated data must have exactly 24 rows.")

        # Warm start
        # if len(self.optimized_consumption) == len(self.data):
        #     warm_df = pd.DataFrame({'hour': self.data['hour'],
        #                             'warm_consumption': self.optimized_consumption})
        #     warm_agg = warm_df.groupby('hour')['warm_consumption'].mean().reset_index()
        #     agg_data[self.device_name] = warm_agg['warm_consumption'].values
        #     self.weekday_optimized_schedule = warm_agg['warm_consumption'].values
        # else:
        #     self.weekday_optimized_schedule = agg_data[self.device_name].values

        # print(agg_data)
        self.weekday_optimized_schedule = agg_data["optimized"].values

        day_prices = agg_data['price_per_kwh'].values
        day_hours = agg_data['hour'].values
        idxs = np.arange(24)

        self.shifts = []
        # call the phased MILP with battery
        self.nextday_optimized_schedule = self.optimized_consumption[idxs].copy()
        shifts = self._solve_milp_phases(prices=day_prices,
                                         hours=day_hours,
                                         indices=idxs,
                                         pv_profile=pv_profile,
                                         pv_forecast=pv_forecast,
                                         battery_state=battery_state,
                                         grid_info=grid_info)
        self.shifts.extend(shifts)
        # self.nextday_optimized_schedule = self.optimized_consumption[idxs].copy()

        logging.info(f"{self.device_name} Next-Day schedule => {self.nextday_optimized_schedule}")
        logging.info(f"{self.device_name} Next-Day iteration => {self.savings:.4f} € savings")
        self.nextday_savings = self.savings
        self.savings = 0.0
        return shifts
        
    def _solve_milp_phases(self,
                        prices: np.ndarray,
                        hours: np.ndarray,
                        indices: np.ndarray,
                        pv_profile: Optional[np.ndarray] = None,
                        pv_forecast: Optional[np.ndarray] = None,
                        weather_forecasts: Optional[dict] = None,
                        battery_state: Optional[Dict] = None,
                        grid_info: Optional[Dict] = None,
                        warm_start_index: Optional[int] = None,
                        force_arbitrage: bool = True) -> List[Dict]:
        """
        Discrete on/off phases approach for next-day scheduling.
        We add optional battery SoC constraints similarly.
        This is more complicated, so each device's phases are scheduled in blocks.
        If we want to model battery in phases, we'd replicate the SoC logic with time steps = 24.
        """
        from pulp import LpProblem, LpMinimize, LpVariable, LpStatus, lpSum, PULP_CBC_CMD

        n = len(indices)
        if n < 1:
            return []
        phases = self.phases
        P = len(phases)
        if P == 0:
            return []

        prob = LpProblem(f"{self.device_name}_PHASE_MILP", LpMinimize)

        # Binary X[(p,k)] => start phase p at hour k
        X = {}
        for p in range(P):
            for k in range(n):
                X[(p,k)] = LpVariable(f"X_{self.device_name}_p{p}_k{k}", cat="Binary")

        # enforce allowed hours
        for p in range(P):
            for k in range(n):
                if hours[k] not in self.allowed_hours:
                    prob += X[(p,k)] == 0, f"AllowedHours_{p}_{k}"

        # Battery variables if we want SoC constraints in next-day phases
        # We'll do a simpler approach: 1-hour resolution, same logic as partial usage
        charge = {}
        discharge = {}
        soc = {}
        if battery_state is not None:
            charge = LpVariable.dicts("charge", range(n), lowBound=0)
            discharge = LpVariable.dicts("discharge", range(n), lowBound=0)
            soc = LpVariable.dicts("soc", range(n),
                                   lowBound=battery_state['soc_min'],
                                   upBound=battery_state['soc_max'])

            # For piecewise efficiency:
            if self.battery_agent is not None:
                segments = self.battery_agent.get_piecewise_segments()
                seg_bin = {}
                for t in range(n):
                    for s_i in range(len(segments)):
                        seg_bin[(t,s_i)] = LpVariable(f"segBin_{t}_{s_i}", cat="Binary")
                for t in range(n):
                    prob += lpSum(seg_bin[(t,s_i)] for s_i in range(len(segments))) == 1, f"OneSeg_{t}"

                bigM = battery_state['soc_max']*1.1
                charge_eff = {}
                discharge_eff = {}
                for t in range(n):
                    charge_eff[t] = LpVariable(f"chargeEff_{t}", lowBound=0, upBound=1)
                    discharge_eff[t] = LpVariable(f"dischargeEff_{t}", lowBound=0, upBound=1)
                    for s_i, (upper_frac, effc, effd) in enumerate(segments):
                        prob += soc[t] - (upper_frac*battery_state['soc_max']) <= bigM*(1 - seg_bin[(t,s_i)])
                    prob += charge_eff[t] == lpSum(segments[s_i][1]*seg_bin[(t,s_i)] for s_i in range(len(segments)))
                    prob += discharge_eff[t] == lpSum(segments[s_i][2]*seg_bin[(t,s_i)] for s_i in range(len(segments)))

                for t in range(n):
                    SoC_incr = LpVariable(f"SoC_incr_{t}", lowBound=0)
                    SoC_decr = LpVariable(f"SoC_decr_{t}", lowBound=0)
                    prob += charge[t] <= battery_state['max_charge_rate']
                    prob += discharge[t] <= battery_state['max_discharge_rate']

                    prob += SoC_incr <= charge[t]
                    prob += SoC_incr <= charge_eff[t]*battery_state['max_charge_rate']
                    prob += SoC_incr >= charge[t] - battery_state['max_charge_rate']*(1-charge_eff[t])

                    prob += SoC_decr <= discharge[t]
                    prob += SoC_decr <= discharge_eff[t]*battery_state['max_charge_rate']
                    prob += SoC_decr >= discharge[t] - battery_state['max_charge_rate']*(1-discharge_eff[t])

                    if t == 0:
                        prob += soc[t] == battery_state['current_soc'] + SoC_incr - SoC_decr
                    else:
                        prob += soc[t] == soc[t-1] + SoC_incr - SoC_decr

                    prob += charge[t] + discharge[t] <= battery_state['max_charge_rate']
            else:
                # simpler approach
                for t in range(n):
                    if t == 0:
                        prob += soc[t] == battery_state['current_soc'] + charge[t] - discharge[t]
                    else:
                        prob += soc[t] == soc[t-1] + charge[t] - discharge[t]
                    prob += charge[t] <= battery_state['max_charge_rate']
                    prob += discharge[t] <= battery_state['max_discharge_rate']
                    prob += charge[t] + discharge[t] <= battery_state['max_charge_rate']

        # Objective function => sum of cost for each phase + potential battery cost terms
        cost_terms = []
        on_penalty = 0.01
        for p in range(P):
            dur = phases[p]["duration"]
            e_kwh = phases[p]["energy_kwh"]
            for k in range(n):
                run_hours = range(k, min(k+dur, n))
                base_cost = sum(prices[rh] for rh in run_hours)*e_kwh
                # if pv_forecast, discount
                if pv_forecast is not None:
                    potential = min(abs(pv_forecast[k])/32704.0, 1.0)
                    discount = forecast_discount(potential)
                else:
                    discount = 0.0
                adj_cost = base_cost*(1-discount)
                cost_terms.append( (adj_cost+on_penalty)*X[(p,k)] )

        # If you want to add battery usage cost or something, you can do it as well:
        # for t in range(n):
        #     cost_terms.append( some battery usage penalty? )
        # but let's keep it minimal.

        prob += lpSum(cost_terms), "TotalCost"

        # Each phase must start exactly once
        for p in range(P):
            prob += lpSum(X[(p,k)] for k in range(n)) == 1, f"PhaseStart_{p}"

        # consecutive ordering
        for p in range(P-1):
            durp = phases[p]["duration"]
            prob += ( lpSum(k*X[(p+1,k)] for k in range(n)) ==
                      lpSum((k+durp)*X[(p,k)] for k in range(n)) ), f"Consecutive_{p}"

        # building load constraints
        for t in range(n):
            est = self.global_layer.get_average_load([hours[t]])[0]
            max_allow = self.global_layer.max_building_load - est
            active_terms = []
            for p in range(P):
                durp = phases[p]["duration"]
                peakp = phases[p]["peak_kw"]
                for k in range(n):
                    if t >= k and t < k+durp:
                        active_terms.append( peakp*X[(p,k)] )
            # battery can offset or add load => if you want that, do:
            # net_load = sum of active_terms + charge[t] - discharge[t]
            # must be <= max_allow
            if battery_state is not None:
                prob += (lpSum(active_terms) + charge[t] - discharge[t]) <= max_allow, f"bldgLoad_{t}"
            else:
                prob += lpSum(active_terms) <= max_allow, f"bldgLoad_{t}"

        # Optionally, sequence constraints for phases
        for p in range(P-1):
            durp = phases[p]["duration"]
            for k1 in range(n):
                for k2 in range(n):
                    if k2 < k1+durp:
                        prob += X[(p,k1)] + X[(p+1,k2)] <= 1, f"seq_{p}_{k1}_{k2}"

        # Solve
        prob.solve(PULP_CBC_CMD(msg=False, timeLimit=300, presolve='on', cuts='on'))
        status = LpStatus[prob.status]
        if status != "Optimal":
            logging.error(f"Phased MILP not optimal => {status}")
            return []

        # process results
        final_usage = np.zeros(n)
        shifts = []
        for p in range(P):
            durp = phases[p]["duration"]
            e_kwh = phases[p]["energy_kwh"]
            for k in range(n):
                if X[(p,k)].varValue > 0.5:
                    # fill usage
                    for hh in range(k, min(k+durp, n)):
                        final_usage[hh] += e_kwh/durp
                    orig_cost = prices[k]*(e_kwh/durp)
                    new_cost = prices[k]*(e_kwh/durp)
                    # apply forecast discount
                    if pv_forecast is not None:
                        potential = min(abs(pv_forecast[k])/32704.0, 1.0)
                        disc = forecast_discount(potential)
                        new_cost *= (1 - disc)
                    shift_savings = orig_cost - new_cost
                    self.savings += shift_savings
                    # self.optimized_consumption[indices[k]] += (e_kwh/durp)
                    sdict = {
                        'device': self.device_name,
                        'phase': p,
                        'start_hour': hours[k],
                        'to_hour': hours[k],
                        'duration': durp,
                        'energy_kwh': e_kwh,
                        'savings': shift_savings,
                        'price': prices[k],
                        'success': True
                    }
                    shifts.append(sdict)
                    self.learn_from_shift(sdict, True)

        self.nextday_optimized_schedule[indices] = final_usage
        return shifts
    
    def _solve_milp_phases_nbew(self,
                        prices: np.ndarray,
                        hours: np.ndarray,
                        indices: np.ndarray,
                        pv_profile: Optional[np.ndarray] = None,
                        pv_forecast: Optional[np.ndarray] = None,
                        weather_forecasts: Optional[dict] = None,
                        battery_state: Optional[Dict] = None,
                        grid_info: Optional[Dict] = None,
                        warm_start_index: Optional[int] = None,
                        force_arbitrage: bool = True) -> List[Dict]:
        """        
        The implementation uses a combination of:
        1. Binary variable y[t] to prevent simultaneous charging/discharging
        2. Direct economic incentive in the objective function 
        3. Adaptive constraints based on battery state to avoid infeasibility
        4. Additional objective terms to encourage proper arbitrage behavior
        """
        """
        Discrete on/off phases approach for next-day scheduling with robust battery integration.
        In addition to scheduling device phases, this MILP incorporates battery constraints using
        charge, discharge, and state-of-charge (SoC) variables. A binary variable y[t] enforces that
        the battery cannot charge and discharge simultaneously. Battery degradation cost and a bonus term 
        that rewards high SoC during low-price hours are included in the objective.
        """
        from pulp import LpProblem, LpMinimize, LpVariable, LpStatus, lpSum, PULP_CBC_CMD
        import numpy as np
        import logging

        # Add at the beginning of _solve_milp_phases:
        print(f"DEBUG: {self.device_name} - allowed_hours = {self.allowed_hours}")
        print(f"DEBUG: hours parameter = {hours}")

        cost_terms = []

        n = len(indices)
        if n < 1:
            return []
        phases = self.phases
        P = len(phases)
        if P == 0:
            return []

        prob = LpProblem(f"{self.device_name}_PHASE_MILP", LpMinimize)

        # Binary variables X[(p,k)] => phase p starts at hour k.
        X = {}
        for p in range(P):
            for k in range(n):
                X[(p, k)] = LpVariable(f"X_{self.device_name}_p{p}_k{k}", cat="Binary")
        # Enforce allowed hours.
        for p in range(P):
            for k in range(n):
                if hours[k] not in self.allowed_hours:
                    prob += X[(p, k)] == 0, f"AllowedHours_{p}_{k}"

        # --- Battery Variables ---
        # We create battery variables only if battery_state is provided.
        if battery_state is not None and self.battery_agent is not None:
            charge = LpVariable.dicts("charge", range(n), lowBound=0)
            discharge = LpVariable.dicts("discharge", range(n), lowBound=0)
            soc = LpVariable.dicts("soc", range(n),
                                lowBound=battery_state['soc_min'],
                                upBound=battery_state['soc_max'])
            # Binary variable y[t]: y[t]=1 means charging, y[t]=0 means discharging.
            y = LpVariable.dicts("y", range(n), cat="Binary")
            
            # Add all battery constraints using the centralized function
            prob, battery_cost_terms = self.battery_agent.add_battery_constraints_to_milp(
                prob=prob,
                battery_state=battery_state,
                n_periods=n,
                charge=charge,
                discharge=discharge,
                soc=soc,
                prices=prices,
                y=y,
                cost_terms=cost_terms,
                force_arbitrage=force_arbitrage,
                problem_type="phases"
            )
            
            # Update cost_terms with battery-related costs
            cost_terms = battery_cost_terms
            
            # All battery constraints are now handled by the centralized function
            # No additional battery constraints needed here
        # --- End Battery Variables ---

        # --- Objective Function for Phases and Battery ---
        # Phase scheduling cost terms - we'll add to the existing battery cost terms
        # Do NOT reset cost_terms here
        on_penalty = 0.01  # small penalty for turning device on
        for p in range(P):
            dur = phases[p]["duration"]
            e_kwh = phases[p]["energy_kwh"]
            for k in range(n):
                run_hours = range(k, min(k + dur, n))
                base_cost = sum(prices[rh] for rh in run_hours) * e_kwh
                # Apply PV forecast discount if available.
                if pv_forecast is not None:
                    potential = min(abs(pv_forecast[k]) / 32704.0, 1.0)
                    discount = forecast_discount(potential)
                else:
                    discount = 0.0
                # Optionally incorporate weather effects.
                factor = 1.0
                if weather_forecasts is not None:
                    for col_name, arr24 in weather_forecasts.items():
                        val = arr24[k]
                        if ('temperature' in col_name.lower()) and val < 25:
                            # Reward low temperature (cheaper hours) with slight bonus.
                            factor -= 0.01 * (25 - val)
                adj_cost = base_cost * (1 - discount) * factor
                cost_terms.append((adj_cost + on_penalty) * X[(p, k)])
        # Combine all cost terms.
        prob += lpSum(cost_terms), "TotalCost"

        # --- Phase Scheduling Constraints ---
        # Each phase must start exactly once.
        for p in range(P):
            prob += lpSum(X[(p, k)] for k in range(n)) == 1, f"PhaseStart_{p}"
        # Consecutive ordering: enforce that phase p+1 starts after phase p.
        for p in range(P - 1):
            durp = phases[p]["duration"]
            prob += lpSum(k * X[(p + 1, k)] for k in range(n)) == lpSum((k + durp) * X[(p, k)] for k in range(n)), f"Consecutive_{p}"

        # --- Building Load Constraints ---
        for t in range(n):
            est = self.global_layer.get_average_load([hours[t]])[0]
            max_allow = self.global_layer.max_building_load - est
            active_terms = []
            for p in range(P):
                durp = phases[p]["duration"]
                peakp = phases[p]["peak_kw"]
                for k in range(n):
                    if t >= k and t < k + durp:
                        active_terms.append(peakp * X[(p, k)])
            if battery_state is not None:
                prob += lpSum(active_terms) + charge[t] - discharge[t] <= max_allow, f"bldgLoad_{t}"
            else:
                prob += lpSum(active_terms) <= max_allow, f"bldgLoad_{t}"

        # Optional sequencing constraints for phases.
        for p in range(P - 1):
            durp = phases[p]["duration"]
            for k1 in range(n):
                for k2 in range(n):
                    if k2 < k1 + durp:
                        prob += X[(p, k1)] + X[(p + 1, k2)] <= 1, f"seq_{p}_{k1}_{k2}"

        # Warm start if provided.
        if warm_start_index is not None:
            offset = 0
            for p in range(P):
                d = int(round(phases[p].get("duration", 1)))
                if warm_start_index + offset < n:
                    X[(p, warm_start_index + offset)].setInitialValue(1.0)
                offset += d

        # Solve the MILP.
        try:
            solver = PULP_CBC_CMD(msg=False, timeLimit=300, presolve='on', cuts='on')
            prob.solve(solver)
        except Exception as e:
            debug_filename = f"{self.device_name}_debug.lp"
            prob.writeLP(debug_filename)
            logging.error(f"Error solving MILP for device {self.device_name}. Model written to {debug_filename}. Exception: {e}")
            return []

        if LpStatus[prob.status] != 'Optimal':
            logging.warning(f"MILP for device {self.device_name} returned status {LpStatus[prob.status]}.")
            debug_filename = f"{self.device_name}_nonoptimal.lp"
            prob.writeLP(debug_filename)
            logging.warning(f"Non-optimal model written to {debug_filename}.")
            return []

        # Process results: reconstruct phase schedule.
        final_usage = np.zeros(n)
        shifts = []
        for p in range(P):
            durp = phases[p]["duration"]
            e_kwh = phases[p]["energy_kwh"]
            for k in range(n):
                if X[(p, k)].varValue > 0.5:
                    # Distribute the energy evenly over the phase duration.
                    for hh in range(k, min(k + durp, n)):
                        final_usage[hh] += e_kwh / durp
                    orig_cost = prices[k] * (e_kwh / durp)
                    new_cost = orig_cost
                    if pv_forecast is not None:
                        potential = min(abs(pv_forecast[k]) / 32704.0, 1.0)
                        disc = forecast_discount(potential)
                        new_cost *= (1 - disc)
                    shift_savings = orig_cost - new_cost
                    self.savings += shift_savings
                    # self.optimized_consumption[indices[k]] += (e_kwh / durp)
                    sdict = {
                        'device': self.device_name,
                        'phase': p,
                        'start_hour': hours[k],
                        'to_hour': hours[k],
                        'duration': durp,
                        'energy_kwh': e_kwh,
                        'savings': shift_savings,
                        'price': prices[k],
                        'success': True
                    }
                    shifts.append(sdict)
                    self.learn_from_shift(sdict, True)
        self.nextday_optimized_schedule = final_usage.copy()

        # Process battery results if battery_state is provided.
        if battery_state is not None:
            # For each hour, update battery variables in device's arrays.
            for t in range(n):
                # Assuming indices[t] gives the corresponding time index.
                idx = indices[t]
                if idx != -1:
                    soc_val = soc[t].varValue if soc[t].varValue is not None else battery_state['current_soc']
                    charge_val = charge[t].varValue if charge[t].varValue is not None else 0.0
                    discharge_val = discharge[t].varValue if discharge[t].varValue is not None else 0.0
                    self.battery_soc[idx] = soc_val
                    self.battery_charge[idx] = charge_val
                    self.battery_discharge[idx] = discharge_val
            day_throughput = sum((charge[t].varValue or 0) + (discharge[t].varValue or 0) for t in range(n))
            self.battery_agent.cycle_count += day_throughput / self.battery_agent.estimated_capacity
            total_charge = sum(charge[t].varValue or 0 for t in range(n))
            total_discharge = sum(discharge[t].varValue or 0 for t in range(n))
            self.battery_agent.charge_history.append(total_charge)
            self.battery_agent.discharge_history.append(total_discharge)
            final_soc = soc[n - 1].varValue if soc[n - 1].varValue is not None else battery_state['current_soc']
            self.battery_agent.current_soc = final_soc
            self.battery_agent.soc_history.append(final_soc)

        return shifts


