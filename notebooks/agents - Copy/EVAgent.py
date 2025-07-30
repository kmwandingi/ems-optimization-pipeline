"""
EVAgent.py - Electric Vehicle Agent class

This module implements an EV Agent that inherits from BatteryAgent
but behaves as a charge-only battery with additional EV-specific constraints:
- No discharging capability
- Must be fully charged by a specified hour (e.g., 7 AM)
- Preserves all SOC, efficiency, and battery state dynamics from BatteryAgent
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Union
from pulp import LpVariable, LpProblem, LpMaximize, LpMinimize, lpSum, LpStatus

from .BatteryAgent import BatteryAgent

class EVAgent(BatteryAgent):
    """
    Electric Vehicle Agent that inherits from BatteryAgent but functions as
    a charge-only battery with a "must be fully charged by" constraint.
    
    All SOC dynamics, efficiency calculations, degradation factors, and other
    battery state logic are preserved from the BatteryAgent parent class.
    """
    def __init__(self, 
                 data: pd.DataFrame = None,
                 device_name: str = "ev",
                 category: str = "Partially Flexible",
                 power_rating: float = 7.4,  # kW
                 global_layer = None,
                 max_shift_hours: int = 24,
                 is_flexible: bool = True,
                 pv_agent = None,
                 spec: dict = None,
                 must_be_full_by_hour: int = 7,  # Default: 7 AM
                 usage_windows: List[Tuple[int, int]] = None,  # List of (start_hour, end_hour) tuples for expected usage
                 required_soc_for_trips: List[float] = None,  # Required SOC for each usage window
                 max_charge_rate: float = None, 
                 initial_soc: float = None, 
                 soc_min: float = None, 
                 soc_max: float = None,
                 capacity: float = 60.0,  # EV battery capacity in kWh
                 degradation_rate: float = 0.00005,
                 temperature_coefficient: float = 1.0,
                 max_ramp_rate: float = None,
                 efficiency_charge: float = 0.95,
                 efficiency_discharge: float = 0.95,
                 self_discharge_rate: float = 0.001,
                 degradation_cost: float = 0.01,
                 degradation_factor: float = 1.0,
                 phases: list = None,
                 max_discharge_rate: float = None,
                 allowed_hours: List[int] = None,
                 preference_penalty_weight: float = 0.0):
        """
        Initialize the EVAgent with all BatteryAgent parameters plus EV-specific additions.
        
        Args:
            data: DataFrame containing historical data
            device_name: Name of the device ("ev" by default)
            category: Device category (default "Partially Flexible")
            power_rating: Maximum charging power in kW
            global_layer: GlobalConnectionLayer instance for building-level coordination
            max_shift_hours: Maximum number of hours to shift operation
            is_flexible: Whether the device is flexible (should be True)
            pv_agent: Optional PV agent for coordination
            spec: Device specification dictionary from device_specs
            must_be_full_by_hour: Hour by which the EV must be fully charged (default 7 AM)
            max_charge_rate: Maximum charging rate (will use power_rating if None)
            initial_soc: Initial state of charge (kWh)
            soc_min: Minimum state of charge (kWh)
            soc_max: Maximum state of charge (kWh)
            degradation_rate: Battery degradation rate per kWh
            temperature_coefficient: Temperature effect multiplier
            max_ramp_rate: Maximum charge rate change per hour (kW/h)
            efficiency_charge: Charging efficiency (0-1)
            efficiency_discharge: Discharging efficiency (0-1)
            self_discharge_rate: Rate of self-discharge per hour (0-1)
            degradation_cost: Cost per kWh of degradation
            degradation_factor: Multiplier for degradation cost
            max_discharge_rate: Maximum discharge rate (kW)
            allowed_hours: List of hours when charging/discharging is allowed
            preference_penalty_weight: Weight for preference penalty (default 0.0)
        """
        # Initialize default values
        self.spec = spec or {}
        
        # Process spec overrides
        must_be_full_by_hour = self.spec.get('must_be_full_by_hour', must_be_full_by_hour)
        soc_min = self.spec.get('soc_min', soc_min or 2.0)
        soc_max = self.spec.get('soc_max', soc_max or 40.0)
        initial_soc = self.spec.get('initial_soc', initial_soc or 5.0)
        capacity = self.spec.get('capacity', capacity)  # Get capacity from spec or use parameter
        efficiency_charge = self.spec.get('efficiency_charge', efficiency_charge)
        efficiency_discharge = self.spec.get('efficiency_discharge', efficiency_discharge)
        self_discharge_rate = self.spec.get('self_discharge_rate', self_discharge_rate)
        degradation_cost = self.spec.get('degradation_cost', degradation_cost)
        degradation_factor = self.spec.get('degradation_factor', degradation_factor)
        max_discharge_rate = self.spec.get('max_discharge_rate', max_discharge_rate or power_rating)
        allowed_hours = self.spec.get('allowed_hours', allowed_hours or list(range(24)))
        max_charge_rate = self.spec.get('max_charge_rate', max_charge_rate or power_rating)
        
        # Store EV-specific parameters
        self.must_be_full_by_hour = must_be_full_by_hour
        self.allowed_hours = allowed_hours
        self.max_discharge_rate = max_discharge_rate
        self.PREFERENCE_PENALTY_WEIGHT = preference_penalty_weight
        
        # Store EV-specific parameters
        self.device_name = device_name
        self.category = category
        self.power_rating = power_rating
        self.global_layer = global_layer
        self.max_shift_hours = max_shift_hours
        self.is_flexible = is_flexible
        self.pv_agent = pv_agent
        self.spec = spec or {}
        
        # Initialize usage windows and required SOCs
        self.usage_windows = usage_windows or []
        self.required_soc_for_trips = required_soc_for_trips or []
        self.phases = phases or []  # Store phases attribute
        
        # Ensure max_discharge_rate is 0.0 to prevent discharging
        max_discharge_rate = 0.0

        # Determine capacity to pass to BatteryAgent
        # EVAgent uses soc_max as its capacity if not otherwise specified in spec
        ev_capacity = self.spec.get('capacity', soc_max) # Use soc_max derived from spec or default
        
        # Initialize the BatteryAgent parent with battery-related parameters
        super().__init__(
            data=data,
            device_name=device_name,
            category=category,
            power_rating=power_rating,
            global_layer=global_layer,
            max_shift_hours=max_shift_hours,
            is_flexible=is_flexible,
            pv_agent=pv_agent,
            spec=spec,
            max_charge_rate=max_charge_rate or power_rating,
            max_discharge_rate=max_discharge_rate or 0.0,  # Default to no discharge for EV
            initial_soc=initial_soc,
            soc_min=soc_min,
            soc_max=soc_max,
            capacity=capacity,
            degradation_rate=degradation_rate,
            temperature_coefficient=temperature_coefficient,
            max_ramp_rate=max_ramp_rate,
            efficiency_charge=efficiency_charge,
            efficiency_discharge=efficiency_discharge,
            self_discharge_rate=self_discharge_rate,
            degradation_cost=degradation_cost,
            degradation_factor=degradation_factor,
            phases=phases,
            allowed_hours=allowed_hours,
            preference_penalty_weight=preference_penalty_weight
        )
        
        # Ensure we have the _last_updated_day attribute for tracking daily updates
        if not hasattr(self, '_last_updated_day'):
            self._last_updated_day = None
        
        # Explicitly set capacity attribute to ensure it's accessible
        self.capacity = capacity
        self.self_discharge_rate = self_discharge_rate
        
        # Add hourly arrays for optimization results (in addition to parent class arrays)
        self.hourly_charge = [0.0] * 24
        self.hourly_discharge = [0.0] * 24
        self.hourly_soc = [initial_soc] * 24
        
        # For optimization results
        self.nextday_optimized_schedule = None
        self.nextday_optimized_soc = None
        
        # Set category for reporting
        self.category = 'ev'
        
        logging.info(
            f"Initialized EV {device_name} with "
            f"SOC: {initial_soc:.1f}-{soc_max:.1f} kWh, "
            f"Charge/Discharge: {max_charge_rate:.1f}/{self.max_discharge_rate:.1f} kW, "
            f"Full by hour: {must_be_full_by_hour or 'Not specified'}"
        )
        
        # Store allowed charging hours for EV
        self.allowed_hours = allowed_hours or list(range(24))
        
        # For battery state tracking
        self.battery_soc_day = []
        self.battery_charge_day = []
        self.battery_discharge_day = []
        
        # Save when this agent was created for debugging
        self.creation_time = pd.Timestamp.now()
        
        logging.info(f"EVAgent initialized for {device_name} with must_be_full_by_hour={must_be_full_by_hour}")
        logging.info(f"  Max charge rate: {self.max_charge_rate} kW")
        logging.info(f"  Battery capacity: {self.soc_max} kWh")
        logging.info(f"  Allowed hours: {self.allowed_hours}")
    
    def get_available_discharge_capacity(self):
        """
        Override to return the available discharge capacity.
        """
        return self.soc_max - self.current_soc
    
    def add_battery_constraints_to_milp(self, prob, battery_state, n_periods, charge, discharge, soc,
                                           prices=None, y=None, cost_terms=None, force_arbitrage=True,
                                           problem_type='standard', name_prefix=None):
        """
        Add battery constraints to the MILP problem, including EV-specific constraints.
        
        This method extends the parent class's method to add EV-specific constraints:
        1. Must-be-full-by-hour constraint
        2. Allowed hours for charging/discharging
        3. Maintains all parent class battery constraints
        
        Args:
            prob: PuLP problem instance
            battery_state: Dictionary with battery state variables
            n_periods: Number of time periods
            charge: Dictionary of charge variables
            discharge: Dictionary of discharge variables
            soc: Dictionary of state of charge variables
            prices: Optional price vector for cost terms
            y: Optional binary variables for on/off states
            cost_terms: List to append cost terms to
            force_arbitrage: Whether to force arbitrage constraints
            problem_type: Type of optimization problem ('standard', 'phases', or 'centralized')
            
        Returns:
            Tuple of (updated problem, updated battery_state)
        """
        # First call the parent class's method to set up standard battery constraints
        prob, cost_terms = super().add_battery_constraints_to_milp(
            prob, battery_state, n_periods, 
            charge, discharge, soc,
            prices, y, cost_terms, 
            force_arbitrage, problem_type,
            name_prefix
        )
        
        # Set up prefix for constraint names
        prefix = (name_prefix or getattr(self, "device_name", "EV")).strip()
        if prefix and not prefix.endswith("_"):
            prefix += "_"
        
        # 1. Add must-be-full-by-hour constraint if specified
        if self.must_be_full_by_hour is not None and self.must_be_full_by_hour < n_periods:
            # Calculate the minimum SOC required at the must_be_full_by_hour
            # This ensures the EV has enough charge for any trips
            required_soc = max(
                battery_state['soc_min'],  # At least the minimum SOC
                battery_state['soc_max'] * 0.95  # Or 95% of max, whichever is higher
            )
            
            # Add the constraint with prefixed variable name
            prob += soc[self.must_be_full_by_hour] >= required_soc, \
                   f"{prefix}must_be_full_by_hour_{self.must_be_full_by_hour}"
            
            # Log the constraint for debugging
            logging.info(
                f"EV {self.device_name} must have at least {required_soc:.2f} kWh "
                f"by hour {self.must_be_full_by_hour}"
            )
        
        # 2. Add allowed hours constraints if specified
        if hasattr(self, 'allowed_hours') and self.allowed_hours is not None:
            allowed_hours_set = set(self.allowed_hours)
            for t in range(n_periods):
                if t not in allowed_hours_set:
                    # If hour is not in allowed hours, force charge and discharge to zero
                    prob += charge[t] == 0, f"{prefix}charge_not_allowed_hour_{t}"
                    prob += discharge[t] == 0, f"{prefix}discharge_not_allowed_hour_{t}"
        
        # 3. Add usage window constraints if specified (for trip requirements)
        if hasattr(self, 'usage_windows') and self.usage_windows:
            for i, (start, end) in enumerate(self.usage_windows):
                if end < n_periods:  # Only add if the window is within optimization horizon
                    required_soc = self.required_soc_for_trips[i] if i < len(self.required_soc_for_trips) else 0.2 * battery_state['soc_max']
                    prob += soc[end] - soc[start] >= required_soc, f"{prefix}RequiredSOC_Window_{i}"
        if self.usage_windows:
            logging.info(f"Processing {len(self.usage_windows)} usage windows for EV {self.device_name}")
            
            for i, (start_hour, end_hour) in enumerate(self.usage_windows):
                # Get required SOC for this trip (energy needed)
                required_soc = self.required_soc_for_trips[i]
                
                # Cap required SOC to available capacity if needed
                available_capacity = battery_state['soc_max'] - battery_state['soc_min']
                if required_soc > available_capacity:
                    logging.warning(f"Trip {i+1} requires {required_soc} kWh but battery capacity is only {available_capacity} kWh. Capping required SOC.")
                    required_soc = available_capacity * 0.9  # 90% of capacity to ensure feasibility
                
                # Apply constraints for every day in the optimization horizon
                for day_offset in range((n_periods + 23) // 24):  # Ceiling division to get days
                    # Calculate period indices for this trip on this day
                    t_start = start_hour + day_offset * 24
                    t_end = end_hour + day_offset * 24
                    
                    # Skip if outside optimization horizon
                    if t_end >= n_periods or t_start < 0:
                        continue
                    
                    # Ensure sufficient charge before departure
                    departure_period = max(0, t_start - 1)  # Period right before departure
                    min_needed_soc = battery_state['soc_min'] + required_soc
                    
                    prob += soc[departure_period] >= min_needed_soc, f"{prefix}MinSOCBeforeTrip_{i}_day{day_offset}"
                    logging.info(f"Added constraint: SOC ≥ {min_needed_soc:.1f} kWh before trip {i+1} at hour {start_hour} on day {day_offset+1}")
                    
                    # Create virtual usage flag for this trip's period
                    # Flag is 1 during trip periods, 0 otherwise
                    trip_periods = list(range(t_start, t_end+1))
                    
                    # Add constraint for SOC reduction after the trip
                    # Allow some flexibility in the exact energy consumption
                    if t_end < n_periods:
                        # SOC after trip should be lower than before trip by approximately required_soc
                        prob += soc[t_end] <= soc[departure_period] - 0.9 * required_soc, f"{prefix}SOCAfterTrip_{i}_day{day_offset}"
                        logging.info(f"Added constraint: SOC decreases by ~{required_soc:.1f} kWh after trip {i+1} on day {day_offset+1}")
                        
                        # Ensure SOC doesn't go below minimum after trip
                        prob += soc[t_end] >= battery_state['soc_min'], f"{prefix}MinSOCAfterTrip_{i}_day{day_offset}"
                        
                        # Prevent charging during trip
                        for t in trip_periods:
                            if t < n_periods:
                                prob += charge[t] == 0, f"{prefix}NoChargeDuringTrip_{i}_t{t}"
        
        # If we're not in a usage window and not at the must_be_full_by_hour,
        # let the optimizer freely decide when to charge based on prices
        if cost_terms is None:
            cost_terms = []
        
        # Return the updated problem and cost terms
        return prob, cost_terms
    
    def optimize_day(self, prices, global_constraints=None):
        """
        Optimize charging schedule for the EV for the next day.
        
        Args:
            prices: Hourly electricity prices
            global_constraints: Optional global constraints from GlobalConnectionLayer
            
        Returns:
            Optimized schedule as a numpy array
        """
        # This method will be called by the centralized optimizer
        # The actual optimization is handled by the GlobalOptimizer
        logging.info(f"EV Agent {self.device_name} optimize_day called")
        
        # Log the EV's charging requirements
        if self.must_be_full_by_hour is not None:
            logging.info(f"Full charge deadline: {self.must_be_full_by_hour}:00")
            
        # Log usage windows if specified
        if self.usage_windows:
            for i, (start_hour, end_hour) in enumerate(self.usage_windows):
                logging.info(f"Usage window {i+1}: {start_hour}:00 - {end_hour}:00, "  
                             f"Required SOC: {self.required_soc_for_trips[i]:.1f} kWh")
        
        # Return empty schedule - actual optimization happens in centralized optimization
        return np.zeros(24)
    
    def get_device_info(self):
        """
        Return information about the EV for the GlobalOptimizer.
        
        Returns:
            Dictionary with EV parameters
        """
        return {
            'device_name': self.device_name,
            'category': self.category,
            'flex_model': 'battery',  # Special type for EV
            'power_rating': self.power_rating,
            'max_charge_rate': self.max_charge_rate,
            'max_discharge_rate': 0.0,  # Always 0 for EV
            'soc_min': self.soc_min,
            'soc_max': self.soc_max,
            'current_soc': self.current_soc,
            'must_be_full_by_hour': self.must_be_full_by_hour,
            'allowed_hours': self.allowed_hours
        }
    
    def get_battery_state(self):
        """
        Override to include EV-specific parameters.
        
        Returns:
            Dictionary with EV/battery state for optimization
        """
        battery_state = super().get_battery_state()
        battery_state['must_be_full_by_hour'] = self.must_be_full_by_hour
        battery_state['allowed_hours'] = self.allowed_hours
        battery_state['device_type'] = 'ev'
        return battery_state

    def update_daily_soc(self, current_date=None):
        """
        Update the EV's state of charge for the next day.
        Only updates soc_history once per unique day.
        
        Args:
            current_date: The current date (datetime.date or str). If None, uses today's date.
        """
        import datetime
        
        # Convert current_date to date object if it's a string
        if current_date is not None and not isinstance(current_date, datetime.date):
            if isinstance(current_date, str):
                current_date = datetime.datetime.strptime(current_date, '%Y-%m-%d').date()
            elif isinstance(current_date, datetime.datetime):
                current_date = current_date.date()
        elif current_date is None:
            current_date = datetime.date.today()
        
        # Only update if we haven't already processed this day
        if not hasattr(self, '_last_updated_day') or self._last_updated_day != current_date:
            # Log the update for debugging
            old_soc = getattr(self, 'initial_soc', getattr(self, 'current_soc', 0.0))
            
            # Ensure we have current_soc attribute
            if not hasattr(self, 'current_soc') or self.current_soc is None:
                self.current_soc = old_soc
            
            # Update the initial_soc to the current_soc for the next day
            self.initial_soc = self.current_soc
            
            # Initialize soc_history if it doesn't exist
            if not hasattr(self, 'soc_history'):
                self.soc_history = []
            
            # Only append if we have a valid current_soc
            if hasattr(self, 'current_soc') and self.current_soc is not None:
                self.soc_history.append(self.current_soc)
                
                # Update the last updated day
                self._last_updated_day = current_date
                
                logging.info(f"Updated EV {getattr(self, 'device_name', '')} daily SoC for {current_date}: {old_soc:.2f} -> {self.current_soc:.2f} kWh")
            else:
                logging.warning(f"Cannot update daily SoC for EV {getattr(self, 'device_name', '')}: current_soc is {getattr(self, 'current_soc', 'not set')}")
        else:
            logging.debug(f"Skipping duplicate daily SoC update for EV {getattr(self, 'device_name', '')} on {current_date}")
