
import numpy as np
import pandas as pd
import logging
###########################################################
#                   BatteryAgent
###########################################################
class BatteryAgent:
    """
    Encapsulates battery data loading, capacity/rate detection, and usage.
    Designed to support both MILP-based and RL-based workflows.
    For RL, we define a 3-discrete-action interface internally (0=discharge, 1=hold, 2=charge).
    """
    def __init__(self, 
                 data: pd.DataFrame = None,
                 battery_charge_cols=None, 
                 battery_discharge_cols=None,
                 max_charge_rate: float = None, 
                 max_discharge_rate: float = None, 
                 initial_soc: float = None, 
                 soc_min: float = None, 
                 soc_max: float = None,
                 degradation_rate: float = 0.00005,     # REDUCED: capacity lost per unit energy throughput (or cycle)
                 temperature_coefficient: float = 1.0,    # multiplier based on temperature
                 max_ramp_rate: float = None,         # maximum change in charge/discharge per time step (kW/h)
                 capacity: float = None,
                 **kwargs):         # <--- Modified to correctly handle **kwargs
        """
        Args:
            data (pd.DataFrame): May contain historical battery charging/discharging columns.
            battery_charge_cols (list, optional): Columns for 'storage_charge'.
            battery_discharge_cols (list, optional): Columns for 'storage_decharge'.
            If these lists are None, we detect them automatically from the data.
            
            If no data is provided, we assume direct user-provided values for max_charge_rate, etc.
        """
        logging.info("\nInitializing Battery Agent...")
        self.action_size = 3  # For RL usage: (0=discharge,1=hold,2=charge)
        self.data = data  # Store data as an instance attribute

        # If self.data is None, we rely entirely on provided parameters
        if self.data is None:
            logging.info("No data provided; using direct battery parameters only.")
            if any(param is None for param in [
                max_charge_rate, max_discharge_rate, 
                initial_soc, soc_min, soc_max
            ]):
                raise ValueError("If no data is provided, you must specify all battery parameters (rates, SOC, etc.)!")

            # Set capacity and related variables
            self.capacity = capacity
            self.soc_max = soc_max or capacity
            self.soc_min = soc_min or (0.2 * self.soc_max)  # Default to 20% min SOC
            self.initial_soc = initial_soc or (0.5 * (self.soc_max - self.soc_min) + self.soc_min)
            # Ensure estimated_capacity is not None for cycle counting
            self.estimated_capacity = self.capacity if self.capacity is not None else self.soc_max  # Used for cycle counting
            self.max_charge_rate = max_charge_rate
            self.max_discharge_rate = max_discharge_rate

        else:
            # We have historical data, attempt to auto-detect columns
            self.data = self.data.copy() # Operate on the instance attribute
            if battery_charge_cols is None:
                battery_charge_cols = [c for c in self.data.columns if 'storage_charge' in c.lower()]
            if battery_discharge_cols is None:
                battery_discharge_cols = [c for c in self.data.columns if 'storage_decharge' in c.lower()]

            logging.info(f"Detected battery charge columns => {battery_charge_cols}")
            logging.info(f"Detected battery discharge columns => {battery_discharge_cols}")

            # Make sure we have datetime index
            if not isinstance(self.data.index, pd.DatetimeIndex):
                if 'utc_timestamp' in self.data.columns:
                    self.data['utc_timestamp'] = pd.to_datetime(self.data['utc_timestamp'])
                    self.data.set_index('utc_timestamp', inplace=True)
                else:
                    raise ValueError("Data must have 'utc_timestamp' or a DatetimeIndex.")

            # Calculate net charge from those columns
            if len(battery_charge_cols) == 0 and len(battery_discharge_cols) == 0:
                logging.warning("No battery columns found. We'll treat this as no data scenario.")
                # fallback to user-provided param
                if any(param is None for param in [
                    max_charge_rate, max_discharge_rate, 
                    initial_soc, soc_min, soc_max
                ]):
                    raise ValueError("No battery columns found and missing direct parameters.")
                self.max_charge_rate = max_charge_rate
                self.max_discharge_rate = max_discharge_rate
                self.initial_soc = initial_soc
                self.soc_min = soc_min
                self.soc_max = soc_max
                self.estimated_capacity = soc_max
            else:
                # Summation of columns
                net_charge_series = self.data[battery_charge_cols].sum(axis=1) - self.data[battery_discharge_cols].sum(axis=1)
                logging.info(f"Battery net charge stats => mean:{net_charge_series.mean():.2f}, max:{net_charge_series.max():.2f}, min:{net_charge_series.min():.2f}")

                daily_net = net_charge_series.groupby(self.data.index.date).sum()
                self.estimated_capacity = float(np.percentile(abs(daily_net), 95))
                logging.info(f"Estimated capacity from historical net charge => {self.estimated_capacity:.2f} kWh")

                # Set min c-rate
                MIN_C_RATE = 0.2
                if max_charge_rate is None:
                    # guess from max of charge columns
                    if len(battery_charge_cols) > 0:
                        cmax = float(self.data[battery_charge_cols].max().max())
                        c_inferred = max(cmax, self.estimated_capacity*MIN_C_RATE)
                        self.max_charge_rate = c_inferred
                    else:
                        self.max_charge_rate = self.estimated_capacity * MIN_C_RATE
                else:
                    self.max_charge_rate = max_charge_rate

                if max_discharge_rate is None:
                    # guess from discharge columns
                    if len(battery_discharge_cols) > 0:
                        dmax = float(self.data[battery_discharge_cols].max().max())
                        d_inferred = max(dmax, self.estimated_capacity*MIN_C_RATE)
                        self.max_discharge_rate = d_inferred
                    else:
                        self.max_discharge_rate = self.estimated_capacity * MIN_C_RATE
                else:
                    self.max_discharge_rate = max_discharge_rate

                logging.info(f"Maximum charging rate => {self.max_charge_rate:.2f} kW")
                logging.info(f"Maximum discharging rate => {self.max_discharge_rate:.2f} kW")

                # Estimate initial SOC from last 24 hours net
                recent_hours = min(24, len(net_charge_series))
                recent_sum = net_charge_series.iloc[-recent_hours:].sum()
                soc_init_guess = 0.5*self.estimated_capacity
                if initial_soc is None:
                    # Attempt a correction
                    corrected = soc_init_guess + recent_sum
                    # clamp
                    corrected = max(0.2*self.estimated_capacity, min(corrected, 0.8*self.estimated_capacity))
                    self.initial_soc = corrected
                    logging.info(f"Auto-chosen initial SOC => {self.initial_soc:.2f} kWh")
                else:
                    self.initial_soc = initial_soc

                if soc_min is None:
                    self.soc_min = 0.2*self.estimated_capacity  # Increased minimum SOC to have more usable capacity
                else:
                    self.soc_min = soc_min
                if soc_max is None:
                    self.soc_max = self.estimated_capacity
                else:
                    self.soc_max = soc_max
                
        # Initialize history lists
        self.soc_history = []       # List of final SOC values at the end of each day
        self.charge_history = []    # List of total energy charged each day (kWh)
        self.discharge_history = [] # List of total energy discharged each day (kWh)
        self._last_updated_day = None  # Track the last day we updated soc_history
                    
        # Default: 25°C, safe temperature to start with
        self.temperature_coefficient = temperature_coefficient
        self.max_ramp_rate = max_ramp_rate 
        
        # Initialize dynamic state
        self.current_soc = initial_soc
        self.charge_history = []
        self.discharge_history = []
        
        # Initialize with the initial SOC if provided
        if initial_soc is not None:
            self.soc_history = [initial_soc]
        self.cycle_count = 0
        self.degradation_rate = degradation_rate
        
        # Add hourly arrays for optimization results
        self.hourly_charge = [0.0] * 24
        self.hourly_discharge = [0.0] * 24
        self.hourly_soc = [initial_soc] * 24

        logging.info("BatteryAgent ready with the following parameters:")
        logging.info(f"  capacity range => {self.soc_min:.1f} to {self.soc_max:.1f} kWh")
        logging.info(f"  charge rate => {self.max_charge_rate:.2f} kW, discharge rate => {self.max_discharge_rate:.2f} kW")
        logging.info(f"  initial SOC => {self.initial_soc:.2f} kWh")

    def update_daily_soc(self, current_date=None):
        """
        Update the battery's state of charge for the end of the day.
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
            old_soc = getattr(self, 'initial_soc', self.current_soc)
            
            # Update the initial_soc to the current_soc for the next day
            self.initial_soc = self.current_soc
            
            # Append to soc_history only if it's a new day
            if not hasattr(self, 'soc_history'):
                self.soc_history = []
            
            # Only append if we have a valid current_soc
            if hasattr(self, 'current_soc') and self.current_soc is not None:
                self.soc_history.append(self.current_soc)
                
                # Update the last updated day
                self._last_updated_day = current_date
                
                logging.info(f"Updated battery daily SoC for {current_date}: {old_soc:.4f} -> {self.current_soc:.4f}")
            else:
                logging.warning(f"Cannot update daily SoC: current_soc is {getattr(self, 'current_soc', 'not set')}")
        else:
            logging.debug(f"Skipping duplicate daily SoC update for {current_date}")

    def add_battery_constraints_to_milp(self, prob, battery_state, n_periods, charge, discharge, soc, 
                                       prices=None, y=None, cost_terms=None, force_arbitrage=True, 
                                       problem_type="standard", name_prefix=None):
        """
        Add the standard battery constraints to `prob`.
        A unique `name_prefix` is prepended to every constraint so that
        multiple storage devices (battery + EV) can coexist in one MILP.

        Add battery-related constraints to a MILP optimization problem.
        
        Args:
            prob: PuLP problem object to add constraints to
            battery_state: Dictionary containing battery parameters
            n_periods: Number of time periods (hours) in the optimization
            charge: Dictionary of LP variables for charging power
            discharge: Dictionary of LP variables for discharging power
            soc: Dictionary of LP variables for battery state of charge
            prices: Optional array of electricity prices for arbitrage incentives
            y: Optional dictionary of binary variables for charge/discharge mode
            cost_terms: List of objective function terms to append to (will create new if None)
            force_arbitrage: Whether to enforce price-based arbitrage constraints
            problem_type: Type of optimization problem ("standard", "phases", or "centralized")
            
        Returns:
            prob: Updated PuLP problem with battery constraints added
            cost_terms: List of objective function terms related to battery operation
        """
        from pulp import lpSum, LpVariable
        import numpy as np
        import logging

        prefix = (name_prefix or getattr(self, "device_name", "Batt")).strip()
        if prefix and not prefix.endswith("_"):
            prefix += "_"
        
        # Initialize or use provided cost_terms list
        if cost_terms is None:
            cost_terms = []
        
        # Create binary mode variables if not provided
        if y is None:
            y = LpVariable.dicts("y", range(n_periods), cat="Binary")
        
        # 1. Basic constraints: binary operation (no simultaneous charge/discharge)
        for t in range(n_periods):
            prob += charge[t] <= battery_state['max_charge_rate'] * y[t], f"{prefix}ChargeMode_{t}"
            prob += discharge[t] <= battery_state['max_discharge_rate'] * (1 - y[t]), f"{prefix}DischargeMode_{t}"
        
        # 2. SOC evolution with efficiency modeling
        # Check if we have piecewise efficiency segments available
        if hasattr(self, 'get_piecewise_segments') and callable(getattr(self, 'get_piecewise_segments')):
            # Use piecewise efficiency model (more accurate)
            segments = self.get_piecewise_segments()
            seg_bin = {}
            
            # Initialize efficiency dictionaries
            charge_eff = {}
            discharge_eff = {}
            
            # Create binary variables for segment selection
            for t in range(n_periods):
                for s_i in range(len(segments)):
                    seg_bin[(t, s_i)] = LpVariable(f"{prefix}segBin_{t}_{s_i}", cat="Binary")
                    
                # Ensure exactly one segment is active for each time period
                prob += lpSum(seg_bin[(t, s_i)] for s_i in range(len(segments))) == 1, f"{prefix}OneSegment_{t}"
            
            # Set up piecewise efficiency variables
            bigM = battery_state['soc_max'] * 1.1
            
            for t in range(n_periods):
                # Create efficiency variables with unique prefixed names
                charge_eff[t] = LpVariable(f"{prefix}chargeEff_{t}", lowBound=0, upBound=1)
                discharge_eff[t] = LpVariable(f"{prefix}dischargeEff_{t}", lowBound=0, upBound=1)
                
                # Set SOC segment constraints 
                for s_i, (upper_frac, eff_c, eff_d) in enumerate(segments):
                    prob += soc[t] - (upper_frac * battery_state['soc_max']) <= bigM * (1 - seg_bin[(t, s_i)]), f"{prefix}SoCseg_{t}_{s_i}"
                
                prob += charge_eff[t] == lpSum(segments[s_i][1] * seg_bin[(t, s_i)] for s_i in range(len(segments))), f"{prefix}ChargeEff_{t}"
                prob += discharge_eff[t] == lpSum(segments[s_i][2] * seg_bin[(t, s_i)] for s_i in range(len(segments))), f"{prefix}DischargeEff_{t}"
            
            # SOC evolution with piecewise efficiencies
            for t in range(n_periods):
                SoC_incr = LpVariable(f"{prefix}SoC_incr_{t}", lowBound=0)
                SoC_decr = LpVariable(f"{prefix}SoC_decr_{t}", lowBound=0)
                
                # Constraints for incremental SOC change based on charging efficiency
                prob += SoC_incr <= charge[t], f"{prefix}SoC_incr_le_charge_{t}"
                prob += SoC_incr <= charge_eff[t] * battery_state['max_charge_rate'], f"{prefix}SoC_incr_le_eff_{t}"
                prob += SoC_incr >= charge[t] - battery_state['max_charge_rate'] * (1 - charge_eff[t]), f"{prefix}SoC_incr_ge_{t}"
                
                # Constraints for decremental SOC change based on discharging efficiency
                prob += SoC_decr <= discharge[t], f"{prefix}SoC_decr_le_discharge_{t}"
                prob += SoC_decr <= discharge_eff[t] * battery_state['max_discharge_rate'], f"{prefix}SoC_decr_le_eff_{t}"
                prob += SoC_decr >= discharge[t] - battery_state['max_discharge_rate'] * (1 - discharge_eff[t]), f"{prefix}SoC_decr_ge_{t}"
                
                # SOC evolution equation
                if t == 0:
                    prob += soc[t] == battery_state['current_soc'] + SoC_incr - SoC_decr, f"{prefix}SoC0_{t}"
                else:
                    prob += soc[t] == soc[t-1] + SoC_incr - SoC_decr, f"{prefix}SoC_{t}"
        else:
            # Simple fixed efficiency model (fallback)
            charge_eff_value = battery_state.get('charge_efficiency', 0.95)
            discharge_eff_value = battery_state.get('discharge_efficiency', 0.95)
            
            for t in range(n_periods):
                # SOC evolution with fixed efficiencies
                if t == 0:
                    # For first time period, use the current SOC
                    if problem_type == "centralized":
                        # Centralized optimization uses a different efficiency formula
                        prob += soc[t] == battery_state['current_soc'] + charge[t] * charge_eff_value - discharge[t] * (1/discharge_eff_value), f"{prefix}Soc0_{t}"
                    else:
                        # Standard and phases use this formula
                        prob += soc[t] == battery_state['current_soc'] + charge[t] * charge_eff_value - discharge[t] / discharge_eff_value, f"{prefix}Soc0_{t}"
                else:
                    # For subsequent periods, use the previous SOC
                    if problem_type == "centralized":
                        prob += soc[t] == soc[t-1] + charge[t] * charge_eff_value - discharge[t] * (1/discharge_eff_value), f"{prefix}SoC_{t}"
                    else:
                        prob += soc[t] == soc[t-1] + charge[t] * charge_eff_value - discharge[t] / discharge_eff_value, f"{prefix}SoC_{t}"
        
        # 3. Basic constraints: rate limits
        for t in range(n_periods):
            prob += charge[t] <= battery_state['max_charge_rate'], f"{prefix}MaxCh_{t}"
            prob += discharge[t] <= battery_state['max_discharge_rate'], f"{prefix}MaxDisch_{t}"
        
        # 4. Ramp rate constraints (if defined)
        if battery_state.get('max_ramp_rate') is not None:
            for t in range(1, n_periods):
                prob += charge[t] - charge[t-1] <= battery_state['max_ramp_rate'], f"{prefix}RampRateCharge_{t}"
                prob += charge[t-1] - charge[t] <= battery_state['max_ramp_rate'], f"{prefix}RampRateChargeNeg_{t}"
                prob += discharge[t] - discharge[t-1] <= battery_state['max_ramp_rate'], f"{prefix}RampRateDischarge_{t}"
                prob += discharge[t-1] - discharge[t] <= battery_state['max_ramp_rate'], f"{prefix}RampRateDischargeNeg_{t}"

                Δin  = charge[t]    - charge[t-1]
                Δout = discharge[t] - discharge[t-1]
                # net ramp = Δin - Δout
                prob += Δin - Δout <= battery_state['max_ramp_rate'], f"{prefix}RampUp_{t}"
                prob += Δout - Δin <= battery_state['max_ramp_rate'], f"{prefix}RampDown_{t}"
        
        # 5. Add economic terms and arbitrage constraints if prices are provided
        if prices is not None:
            # Get battery parameters for economic terms
            degradation_rate = battery_state.get('degradation_rate', 0.00005)
            arbitrage_scale = battery_state.get('arbitrage_scale', 50.0)
            
            # Calculate price statistics for arbitrage strategy
            price_min = np.min(prices)
            price_max = np.max(prices)
            price_range = price_max - price_min if price_max > price_min else 1.0
            
            # Find optimal periods for charging/discharging
            sorted_price_indices = np.argsort(prices)
            lowest_price_indices = sorted_price_indices[:n_periods//4]  # Bottom 25%
            highest_price_indices = sorted_price_indices[-n_periods//4:]  # Top 25%
            
            # Economic incentives for price-based arbitrage
            for t in range(n_periods):
                # Core economic drivers for arbitrage
                if problem_type == "centralized":
                    # For centralized optimization, use these economic terms
                    price_value = prices[t]
                    # Direct economic incentive based on price
                    cost_terms.append(price_value * (charge[t] - discharge[t]))
                    # Degradation cost
                    cost_terms.append(degradation_rate * (charge[t] + discharge[t]))
                    
                    # Normalized price position for incentives (0=lowest, 1=highest)
                    norm_price = (prices[t] - price_min) / price_range
                    
                    # Incentives based on price position
                    charge_incentive = arbitrage_scale * (1.0 - norm_price) * charge[t]
                    discharge_incentive = arbitrage_scale * norm_price * discharge[t]
                    
                    cost_terms.append(-charge_incentive)  # Negative cost = incentive to charge at low prices
                    cost_terms.append(-discharge_incentive)  # Negative cost = incentive to discharge at high prices
                else:
                    # For standard and phases optimization
                    # 1. Direct price-based incentive (1.2 multiplier prioritizes discharge at high prices)
                    cost_terms.append(prices[t] * (charge[t] - discharge[t] * 1.2))
                    
                    # 2. Degradation cost
                    cost_terms.append(degradation_rate * (charge[t] + discharge[t]))
                    
                    # 3. Price-position based incentives
                    norm_price = (prices[t] - price_min) / price_range
                    
                    charge_incentive = arbitrage_scale * (1.0 - norm_price) * charge[t]
                    discharge_incentive = arbitrage_scale * norm_price * discharge[t]
                    
                    cost_terms.append(-charge_incentive)
                    cost_terms.append(-discharge_incentive)
            
            # Enhanced arbitrage strategy with more sophisticated constraints
            if force_arbitrage:
                # Calculate price quartiles for more nuanced control
                sorted_prices = sorted(prices)
                very_low_price = sorted_prices[min(len(sorted_prices)//10, len(sorted_prices)-1)]  # 10th percentile
                low_price = sorted_prices[min(len(sorted_prices)//4, len(sorted_prices)-1)]        # 25th percentile
                median_price = sorted_prices[min(len(sorted_prices)//2, len(sorted_prices)-1)]     # 50th percentile
                high_price = sorted_prices[min(3*len(sorted_prices)//4, len(sorted_prices)-1)]     # 75th percentile
                very_high_price = sorted_prices[min(9*len(sorted_prices)//10, len(sorted_prices)-1)]  # 90th percentile
                
                # 1. Absolutely prohibit charging during highest price periods
                for t in highest_price_indices:
                    prob += charge[t] == 0, f"{prefix}NoChargeHighPrice_{t}"
                
                # 2. Absolutely prohibit discharging during lowest price periods
                for t in lowest_price_indices:
                    prob += discharge[t] == 0, f"{prefix}NoDischargeEconomic_{t}"
                
                # 3. Encourage charging during very low price periods
                for t in range(n_periods):
                    if prices[t] <= very_low_price:
                        # Strong incentive to charge during very low price periods
                        # Create a binary variable to track charging decision
                        should_charge_t = LpVariable(f"should_charge_vlow_{t}", cat="Binary")
                        
                        # Link the binary variable to actual charging
                        M = battery_state['max_charge_rate'] * 2  # Big-M constant
                        prob += charge[t] <= battery_state['max_charge_rate'] * should_charge_t, f"{prefix}ChargeEnableVLow_{t}"
                        prob += charge[t] >= 0.1 * battery_state['max_charge_rate'] * should_charge_t, f"{prefix}MinChargeVLow_{t}"
                        
                        # Add significant incentive in objective function
                        charge_bonus = -10.0 * should_charge_t  # Large negative cost = incentive to charge
                        cost_terms.append(charge_bonus)
                
                # 4. Encourage discharging during very high price periods
                for t in range(n_periods):
                    if prices[t] >= very_high_price:
                        # Strong incentive to discharge during very high price periods
                        should_discharge_t = LpVariable(f"should_discharge_vhigh_{t}", cat="Binary")
                        
                        # Link binary variable to actual discharging
                        M = battery_state['max_discharge_rate'] * 2
                        prob += discharge[t] <= battery_state['max_discharge_rate'] * should_discharge_t, f"{prefix}DischargeEnableVHigh_{t}"
                        prob += discharge[t] >= 0.1 * battery_state['max_discharge_rate'] * should_discharge_t, f"{prefix}MinDischargeVHigh_{t}"
                        
                        # Add significant incentive in objective function
                        discharge_bonus = -10.0 * should_discharge_t  # Large negative cost = incentive to discharge
                        cost_terms.append(discharge_bonus)
                
                # 5. Safety constraint: ensure sufficient capacity for discharging
                for t in range(1, n_periods):
                    prob += discharge[t] <= soc[t-1] - battery_state['soc_min'], f"{prefix}DischargeLimit_{t}"
                
                # 6. NEW: Ensure minimum daily throughput for battery value
                # Calculate minimum charge/discharge throughput target based on usable capacity
                usable_capacity = battery_state['soc_max'] - battery_state['soc_min']
                min_daily_throughput = usable_capacity * 0.2  # Target 20% of usable capacity daily
                
                # Add soft constraint to encourage meeting throughput target
                total_throughput = lpSum(charge[t] + discharge[t] for t in range(n_periods))
                throughput_deficit = LpVariable("throughput_deficit", lowBound=0)
                prob += throughput_deficit >= min_daily_throughput - total_throughput, f"{prefix}DeficitDefinition"
                
                # Add penalty for not meeting throughput target
                throughput_penalty = throughput_deficit * 0.5  # Moderate penalty
                cost_terms.append(throughput_penalty)
        
        # For the hourly_plan integration (specific to standard optimization)
        if problem_type == "standard" and 'hourly_plan' in battery_state:
            hourly_plan = battery_state['hourly_plan']
            
            for t in range(n_periods):
                hour = t % 24  # Ensure hour is 0-23
                if hour in hourly_plan:
                    # Get planned charging and discharging
                    planned_charge = hourly_plan[hour]['charge']
                    planned_discharge = hourly_plan[hour]['discharge']
                    
                    # RULE 1: If battery is not being used at all, provide flexibility
                    if planned_charge < 0.001 and planned_discharge < 0.001:
                        # Check price to determine if charging is advantageous
                        if prices is not None:
                            avg_price = sum(prices) / len(prices)
                            price_threshold = avg_price * 0.85
                            
                            # If price is low, enable optional charging
                            if prices[t] <= price_threshold:
                                should_charge = LpVariable(f"should_charge_{t}", cat="Binary")
                                
                                # For first period, calculate safe charge rate
                                if t == 0:
                                    remaining_capacity = battery_state['soc_max'] - battery_state['current_soc']
                                    safe_charge_rate = min(battery_state['max_charge_rate'], remaining_capacity)
                                    prob += charge[t] == safe_charge_rate * should_charge, f"{prefix}SafeChargeRate_{t}"
                                else:
                                    # For subsequent periods, use constraints
                                    prob += charge[t] <= battery_state['max_charge_rate'] * should_charge, f"{prefix}MaxRateLimit_{t}"
                                    
                                    # Capacity constraint
                                    M = battery_state['max_charge_rate'] * 2
                                    prob += charge[t] <= battery_state['soc_max'] - soc[t-1] + M * (1 - should_charge), f"{prefix}CapacityLimit_{t}"
                                    
                                    # Minimum charge amount
                                    prob += charge[t] >= 0.1 * should_charge, f"{prefix}MinChargeAmount_{t}"
                                
                                # Prevent discharging if charging
                                prob += discharge[t] <= battery_state['max_discharge_rate'] * (1 - should_charge), f"{prefix}NoDischargeIfCharging_{t}"
                                
                                # Add incentive to charge when price is low
                                price_ratio = min(1.0, prices[t] / avg_price)
                                charge_incentive = -5.0 * (1.0 - price_ratio) * battery_state['max_charge_rate'] * should_charge
                                cost_terms.append(charge_incentive)
                    
                    # RULE 2: If battery is already charging, prevent additional operations
                    elif planned_charge > 0.001:
                        prob += charge[t] == 0, f"{prefix}NoAdditionalCharge_{t}"
                        prob += discharge[t] == 0, f"{prefix}NoDischargeWhenCharging_{t}"
                    
                    # RULE 3: If battery is already discharging, prevent charging
                    elif planned_discharge > 0.001:
                        # Calculate remaining discharge capacity
                        remaining_discharge = min(
                            battery_state['max_discharge_rate'] - planned_discharge,
                            (battery_state['current_soc'] - battery_state['soc_min']) - planned_discharge
                        )
                        
                        # Limit discharge based on remaining capacity
                        if remaining_discharge > 0:
                            prob += discharge[t] <= remaining_discharge, f"{prefix}RemainingDischarge_{t}"
                        else:
                            prob += discharge[t] == 0, f"{prefix}NoDischargeLeft_{t}"
                        
                        # Prevent charging when discharging
                        prob += charge[t] == 0, f"{prefix}NoChargeWhenDischarging_{t}"
                    
                    # Update SOC limits based on planned operations
                    current_soc_estimate = battery_state['current_soc'] + planned_charge - planned_discharge
                    soc_upper_limit = min(battery_state['soc_max'], current_soc_estimate + battery_state['max_charge_rate'])
                    soc_lower_limit = max(battery_state['soc_min'], current_soc_estimate - battery_state['max_discharge_rate'])
                    
                    # Apply updated SOC constraints
                    prob += soc[t] <= soc_upper_limit, f"SOCUpperWithHistory_{t}"
                    prob += soc[t] >= soc_lower_limit, f"SOCLowerWithHistory_{t}"
                
                # Original: No end-of-day SOC constraints - let arbitrage drive charging naturally

        
        return prob, cost_terms


    # ----------------------- In BatteryAgent.py -----------------------
    def get_battery_state(self) -> dict:
        """
        Return dictionary summarizing battery's relevant parameters for MILP usage,
        including ramp rate, degradation cost, efficiency effects, and arbitrage parameters.
        
        ELEGANT ARBITRAGE SOLUTION: Optimized parameters for effective price-based arbitrage.
        """
        # Compute effective maximum capacity (degrade linearly with cycle count)
        effective_soc_max = self.soc_max * (1 - self.degradation_rate * self.cycle_count)
        
        # Calculate usable energy for arbitrage operations
        usable_energy = effective_soc_max - self.soc_min
        
        return {
            # Basic battery parameters
            'max_charge_rate': self.max_charge_rate * self.temperature_coefficient,
            'max_discharge_rate': self.max_discharge_rate * self.temperature_coefficient,
            'current_soc': self.current_soc,
            'soc_min': self.soc_min,
            'soc_max': effective_soc_max,
            'max_ramp_rate': self.max_ramp_rate,  # if None, no ramp constraint is applied
            
            # Efficiency parameters
            'charge_efficiency': 0.95,    # AC-to-DC conversion efficiency
            'discharge_efficiency': 0.95, # DC-to-AC conversion efficiency
            
            # Arbitrage optimization parameters
            'degradation_cost': 0.00005,  # Significantly reduced to prioritize economic arbitrage
            'arbitrage_scale': 5.0,      # Strong incentive scale for price-based behavior
            'arbitrage_priority': True,   # Flag to indicate arbitrage should be prioritized
            'force_arbitrage': True,      # Flag to activate arbitrage constraints and incentives
            
            # Battery status information
            'usable_energy': usable_energy, # For reference in arbitrage calculations
            'soc_current_percent': (self.current_soc - self.soc_min) / max(0.1, usable_energy), # Current SOC as percentage
            'cycle_count': self.cycle_count, # Current cycle count
            
            # Battery characteristics
            'temperature_coefficient': getattr(self, 'temperature_coefficient', 1.0),  # Default to 1.0 if not set
            'estimated_capacity': getattr(self, 'estimated_capacity', getattr(self, 'capacity', self.soc_max))  # Use capacity or soc_max as fallback
        }
    
    def get_piecewise_segments(self):
        """
        Returns a list of (upper_bound, eff_charge, eff_discharge) defining
        piecewise segments for battery efficiency based on SoC.
        Each tuple is (SoC fraction upper bound, charging_eff, discharging_eff).
        
        Example below: 3 segments:
        - Segment 1: SoC up to 0.33 -> (eff_charge=0.90, eff_discharge=0.90)
        - Segment 2: SoC up to 0.66 -> (0.95, 0.95)
        - Segment 3: SoC up to 1.00 -> (0.88, 0.88)
        """
        return [
            (0.33, 0.90, 0.90),
            (0.66, 0.95, 0.95),
            (1.00, 0.88, 0.88)
        ]


