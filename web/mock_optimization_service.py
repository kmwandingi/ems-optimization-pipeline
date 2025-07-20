"""
Mock implementation of OptimizationService for Streamlit demo
"""
import random
import sys
from pathlib import Path
# Ensure we can import device_specs regardless of running location
project_root = Path(__file__).resolve().parent.parent.parent
spec_path = project_root / "notebooks" / "utils"
if str(spec_path) not in sys.path:
    sys.path.insert(0, str(spec_path))
from device_specs import device_specs
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple

class MockBatteryAgent:
    """Mock battery agent that returns random SoC values"""
    
    def __init__(self):
        self.hourly_soc = [random.uniform(0.3, 0.9) for _ in range(24)]
        

class MockFlexibleDevice:
    """Mock flexible device that returns random schedules"""
    
    def __init__(self, device_name: str):
        self.device_name = device_name
        self.nextday_optimized_schedule = [random.uniform(0, 2) for _ in range(24)]
        

class MockDatabaseService:
    """Mock database service for the mock optimization service"""
    
    def __init__(self):
        self.building_data_cache = {"default_building": {}}  # Minimal implementation


class MockProbabilityModelAgent:
    """Mock probability model agent for the mock optimization service"""
    
    def __init__(self):
        pass
        
    def update_model_with_actuals(self, building_id, target_date, actual_usage):
        """Mock implementation of updating model with actuals"""
        # Just print to console to show it was called
        print(f"Updating model for {building_id} on {target_date} with actuals")
        print(f"Actuals: {actual_usage}")
        return True


class MockOptimizationService:
    """
    Mock implementation of OptimizationService for UI demonstration
    """
    
    def __init__(self):
        """Initialize the mock optimization service"""
        self._battery_agents = {}
        self.db_service = MockDatabaseService()
        self.prob_agent = MockProbabilityModelAgent()
        self.device_types = [
            "dishwasher",
            "washing_machine",
            "dryer",
            "ev_charger",
            "heat_pump"
        ]
        
        # Default PMFs for different device types
        self._default_pmfs = {
            "washing_machine": [0.05, 0.05, 0.05, 0.05, 0.1, 0.15, 0.2, 0.1, 0.05, 0.05, 0.05, 0.1],
            "dishwasher": [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.15, 0.2, 0.15],
            "tumble_dryer": [0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.15, 0.15, 0.1, 0.1, 0.1, 0.05],
            "ev_charger": [0.15, 0.15, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.15]
        }
        
        # Store device-specific PMFs that will be updated with learning
        self._device_pmfs = {}
        
        # Store complete history of PMFs for each device
        # Format: {device_name: [pmf_day_1, pmf_day_2, ...]} where each pmf is a list of probabilities
        self._pmf_history = {}
        
        # Maximum history length to maintain
        self._max_history_length = 20
        
        # Learning rate for PMF updates (higher = faster adaptation)
        self.learning_rate = 0.15
        
    def next_day(self, building_id: str, device_constraints: dict, baseline_usage: dict) -> tuple:
        """Generate a schedule for the next day that aligns with price variations
        
        Args:
            building_id: Building ID
            device_constraints: Dictionary of device constraints
            baseline_usage: Dictionary of device baseline usage
            
        Returns:
            Tuple of (schedule_dict, price_curve)
            - schedule_dict: Dictionary of device schedules and battery SoC
            - price_curve: List of hourly prices (€/kWh)
        """
        # Create a mock battery agent
        battery_agent = self.get_battery_agent(building_id)
        
        schedule: Dict[str, List[float]] = {}
        
        # Create a mock price curve if not testing with a real one
        # The curve has low prices at night (0-6), moderate during day, high in evening (18-21)
        price_curve = []
        for hour in range(24):
            if 0 <= hour < 6:  # Night: lowest prices
                price_curve.append(0.15 + random.uniform(-0.03, 0.03))
            elif 18 <= hour < 22:  # Evening peak: highest prices
                price_curve.append(0.38 + random.uniform(-0.05, 0.05))
            else:  # Rest of day: moderate prices
                price_curve.append(0.26 + random.uniform(-0.04, 0.04))
        
        for device_name, constraints in device_constraints.items():
            spec = device_specs.get(device_name)
            if not spec:
                # Unknown device, skip
                continue

            earliest_hour = constraints.get("earliest_hour", 0)
            latest_hour = constraints.get("latest_hour", 23)

            allowed_hours = [h for h in spec.get("allowed_hours", list(range(24))) if earliest_hour <= h <= latest_hour]
            if not allowed_hours:
                # No hours available – return zero schedule
                schedule[device_name] = [0.0] * 24
                continue

            # For discrete_phase devices schedule one contiguous run of all phases
            phases = spec.get("phases", [])
            total_duration = int(sum(int(p.get("duration", 1)) for p in phases))
            if not total_duration:
                total_duration = 2  # Default if no phases defined

            # Feasible start hours are those where every hour of the block is allowed
            feasible_starts = [h for h in allowed_hours 
                             if all(((h + offset) in allowed_hours) for offset in range(total_duration)) 
                             and h + total_duration - 1 <= latest_hour]

            if not feasible_starts:
                schedule[device_name] = [0.0] * 24
                continue
            
            # Calculate the average price for each possible start time window
            start_costs = {}
            for start_hour in feasible_starts:
                window_prices = price_curve[start_hour:start_hour+total_duration]
                avg_price = sum(window_prices) / len(window_prices)
                start_costs[start_hour] = avg_price
            
            # Sort by price (lowest first)
            sorted_starts = sorted(start_costs.keys(), key=lambda h: start_costs[h])
            
            # Select from the cheapest 40% of options to add some randomness while favoring cheap hours
            # If few options, just take the cheapest
            cheap_options = sorted_starts[:max(1, len(sorted_starts) // 3 + 1)]
            
            # The optimizer's goal is to find the cheapest feasible time.
            # We select from the cheapest 33% of options to add some variability while strongly favoring low prices.
            start_hour = random.choice(cheap_options)
            
            device_schedule = [0.0] * 24
            hour_ptr = start_hour
            for phase in phases:
                dur = int(phase.get("duration", 1))
                energy = phase.get("energy_kwh", 0.0)
                for _ in range(dur):
                    if hour_ptr < 24:
                        device_schedule[hour_ptr] = energy
                    hour_ptr += 1
            
            # If no phases defined, create a basic schedule
            if not phases:
                # Default energy usage if not specified
                energy = 1.0
                for i in range(total_duration):
                    if start_hour + i < 24:
                        device_schedule[start_hour + i] = energy
                        
            schedule[device_name] = device_schedule
        
                # Add battery SoC
        schedule["battery_soc"] = battery_agent.hourly_soc

        # Return both the schedule and the price curve
        return schedule, price_curve
    
    def get_battery_agent(self, building_id: str) -> MockBatteryAgent:
        """Get a mock battery agent"""
        if building_id not in self._battery_agents:
            self._battery_agents[building_id] = MockBatteryAgent()
        return self._battery_agents[building_id]
    
    def optimize_single_day(
        self, 
        building_id: str, 
        target_date,
        updated_specs: Dict = None,
        battery_agent = None,
        use_weather: bool = True
    ) -> Tuple[List[MockFlexibleDevice], Any, bool]:
        """Generate mock optimization results"""
        # Create mock devices with random schedules
        devices = [
            MockFlexibleDevice(device_name=device_type) 
            for device_type in self.device_types
        ]
        
        # Create a mock optimizer (we'll just return None)
        mock_optimizer = None
        
        # Mock whether PV is available
        has_pv = True
        
        return devices, mock_optimizer, has_pv
    
    def update_with_actuals(self, date_str: str, actual_usage: Dict[str, List[float]]) -> None:
        """Update the model with actual usage and save it to a file.

        Args:
            date_str: Date string in ISO format
            actual_usage: Dictionary of device names to actual usage arrays
        """
        # Ensure a directory for schedules and actuals exists
        schedules_dir = Path("schedules")
        schedules_dir.mkdir(exist_ok=True)

        # Save the actual usage to a corresponding JSON file
        actuals_file = schedules_dir / f"{date_str}_actuals.json"
        try:
            with open(actuals_file, "w") as f:
                import json
                json.dump(actual_usage, f, indent=2)
            print(f"Successfully saved actuals to {actuals_file}")
        except Exception as e:
            print(f"Error saving actuals to {actuals_file}: {e}")

        # Store the actuals for reference
        self._last_actuals = actual_usage
        
        # Update the PMF for each device based on actual usage
        for device, values in actual_usage.items():
            # Calculate block sums (2-hour blocks)
            block_sums = [
                sum(values[0:2]),
                sum(values[2:4]),
                sum(values[4:6]),
                sum(values[6:8]),
                sum(values[8:10]),
                sum(values[10:12]),
                sum(values[12:14]),
                sum(values[14:16]),
                sum(values[16:18]),
                sum(values[18:20]),
                sum(values[20:22]),
                sum(values[22:24])
            ]
            
            # Check if there's any usage to avoid division by zero
            if sum(block_sums) > 0:
                # Create normalized probabilities from the actual usage
                new_probs = []
                total = sum(block_sums) + 0.001  # Small offset to avoid division by zero
                
                for block_sum in block_sums:
                    prob = block_sum / total
                    new_probs.append(prob)
                
                # Normalize again to ensure sum is 1
                total = sum(new_probs)
                new_probs = [p / total for p in new_probs]
                
                # If this is the first update for this device, initialize with default
                if device not in self._device_pmfs:
                    if device in self._default_pmfs:
                        self._device_pmfs[device] = self._default_pmfs[device].copy()
                    else:
                        # Generic default
                        self._device_pmfs[device] = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.15, 0.2, 0.15]
                
                # Initialize history for this device if it doesn't exist
                if device not in self._pmf_history:
                    self._pmf_history[device] = []
                
                # Add current PMF to history before updating
                self._pmf_history[device].append(self._device_pmfs[device].copy())
                
                # Trim history if it exceeds maximum length
                if len(self._pmf_history[device]) > self._max_history_length:
                    self._pmf_history[device].pop(0)  # Remove oldest entry
                
                # Apply learning: blend current PMF with new observed probabilities
                # using learning rate to control adaptation speed
                for i in range(len(self._device_pmfs[device])):
                    self._device_pmfs[device][i] = (1 - self.learning_rate) * self._device_pmfs[device][i] + \
                                                self.learning_rate * new_probs[i]
                
                # Log the update
                print(f"Updated PMF for {device}")
            else:
                print(f"No usage detected for {device}, PMF not updated")

    def get_device_pmf(self, device_name: str) -> Dict[str, List]:
        """Get the probability mass function for a device.

        Args:
            device_name: Device name

        Returns:
            Dict with time_blocks, current_probabilities, and pmf_history
        """
        # Get PMF history. This is the source of truth for the latest PMF.
        pmf_history = self._pmf_history.get(device_name, [])

        # The current PMF is the most recent entry in the history.
        # If history is empty, fall back to the stored PMF or a default.
        if pmf_history:
            current_pmf = pmf_history[-1]
        elif device_name in self._device_pmfs:
            current_pmf = self._device_pmfs[device_name]
        elif device_name in self._default_pmfs:
            current_pmf = self._default_pmfs[device_name]
        else:
            # Generic default if no history and no specific default exists
            current_pmf = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.15, 0.2, 0.15]

        # Ensure history is not empty for the chart by adding the current PMF if needed.
        if not pmf_history:
            pmf_history = [current_pmf.copy()]

        # Create time blocks ("00-02", "02-04", etc.)
        time_blocks = [f"{i:02d}-{i+2:02d}" for i in range(0, 24, 2)]

        return {
            "time_blocks": time_blocks,
            "current_probabilities": current_pmf,
            "pmf_history": pmf_history
        }

    def initialize_pmf_with_baseline(self, device_name: str, baseline_hour: int):
        """Initialize a device's PMF with a strong peak at the user's typical start hour."""
        # Create a new PMF with a strong peak at the baseline hour
        # The distribution is a simple peak, not a sophisticated model
        new_pmf = [0.01] * 12  # Base probability for all 2-hour blocks
        block_index = baseline_hour // 2
        new_pmf[block_index] = 0.89 # Strong peak for the selected block

        # Normalize to ensure it sums to 1
        total = sum(new_pmf)
        new_pmf = [p / total for p in new_pmf]

        # Set this as the current PMF for the device
        self._device_pmfs[device_name] = new_pmf

        # Also add it to the history, so get_device_pmf will use it immediately
        if device_name not in self._pmf_history:
            self._pmf_history[device_name] = []
        self._pmf_history[device_name].append(new_pmf)
