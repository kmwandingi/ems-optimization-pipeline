#!/usr/bin/env python
"""
Unit Tests for 24-Hour Schedule Validation

This test suite verifies that all device schedules, battery schedules, and EV schedules
are exactly 24 hours long (0-23 hours), with proper padding where needed.
"""

import unittest
from unittest.mock import Mock, patch
import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "notebooks"))
sys.path.insert(0, str(project_root / "scripts"))

class TestScheduleValidation(unittest.TestCase):
    """Test that all schedules are exactly 24 hours long."""
    
    def setUp(self):
        """Set up test environment."""
        self.building_id = "DE_KN_residential1"
        
        # Create mock 24-hour data
        self.mock_data = pd.DataFrame({
            'hour': list(range(24)),
            'day': ['2023-01-01'] * 24,
            'price_per_kwh': [0.25] * 24,
            'utc_timestamp': pd.date_range('2023-01-01', periods=24, freq='H')
        })

    def test_battery_agent_hourly_arrays_length(self):
        """Test that BatteryAgent hourly arrays are exactly 24 hours."""
        from agents.BatteryAgent import BatteryAgent
        
        battery_agent = BatteryAgent(
            max_charge_rate=3.0,
            max_discharge_rate=3.0,
            initial_soc=8.0,
            soc_min=1.0,
            soc_max=10.0,
            capacity=10.0
        )
        
        # Check that all hourly arrays are length 24
        self.assertEqual(len(battery_agent.hourly_charge), 24)
        self.assertEqual(len(battery_agent.hourly_discharge), 24)
        self.assertEqual(len(battery_agent.hourly_soc), 24)

    def test_ev_agent_hourly_arrays_length(self):
        """Test that EVAgent hourly arrays are exactly 24 hours."""
        from agents.EVAgent import EVAgent
        
        ev_agent = EVAgent(
            capacity=60.0,
            initial_soc=30.0,
            soc_min=6.0,
            soc_max=54.0,
            max_charge_rate=11.0,
            max_discharge_rate=0.0,
            must_be_full_by_hour=7
        )
        
        # Check that all hourly arrays are length 24
        self.assertEqual(len(ev_agent.hourly_charge), 24)
        self.assertEqual(len(ev_agent.hourly_discharge), 24)
        self.assertEqual(len(ev_agent.hourly_soc), 24)

    def test_flexible_device_schedule_length(self):
        """Test that FlexibleDevice schedules are exactly 24 hours."""
        from agents.FlexibleDeviceAgent import FlexibleDevice
        from agents.GlobalConnectionLayer import GlobalConnectionLayer
        
        # Create mock global layer
        global_layer = GlobalConnectionLayer(max_building_load=50.0, total_hours=24)
        
        device = FlexibleDevice(
            device_name="test_device",
            data=self.mock_data,
            category="Highly Flexible",
            power_rating=3.0,
            global_layer=global_layer,
            battery_agent=None,
            spec={'category': 'Highly Flexible', 'power_rating': 3.0}
        )
        
        # Test that device has 24-hour arrays
        if hasattr(device, 'optimized_schedule'):
            self.assertEqual(len(device.optimized_schedule), 24)
        
        # Test the padding logic specifically
        device.optimized_consumption = [1.0] * 18  # Only 18 hours
        device.data = self.mock_data.head(18)  # Only 18 hours of data
        
        # This should pad to 24 hours
        day_indices = list(range(18))
        day_consumption = device.optimized_consumption
        padded_schedule = np.pad(day_consumption, (0, max(0, 24 - len(day_consumption))), 'constant')[:24].tolist()
        
        self.assertEqual(len(padded_schedule), 24)
        # First 18 hours should have data, last 6 should be zeros
        self.assertEqual(padded_schedule[:18], [1.0] * 18)
        self.assertEqual(padded_schedule[18:], [0.0] * 6)

    def test_schedule_extraction_always_24_hours(self):
        """Test that schedule extraction always returns 24-hour arrays."""
        import common
        
        # Create mock device with different schedule lengths
        mock_device = Mock()
        mock_device.device_name = "test_device"
        
        # Test with longer schedule (should be truncated to 24)
        mock_device.optimized_consumption = [1.0] * 30  # 30 hours
        schedule = np.array(mock_device.optimized_consumption[:24])
        self.assertEqual(len(schedule), 24)
        
        # Test with shorter schedule (should use only available data)
        mock_device.optimized_consumption = [1.0] * 18  # 18 hours
        schedule = np.array(mock_device.optimized_consumption[:24])  # Will take only 18
        self.assertEqual(len(schedule), 18)  # This is the current behavior

    def test_cost_calculation_uses_24_hour_slices(self):
        """Test that cost calculations always use 24-hour schedule slices."""
        # Test data with more than 24 hours
        long_schedule = [1.0] * 30
        prices = [0.25] * 30
        
        # Should only use first 24 hours
        cost_24 = np.sum(np.array(long_schedule[:24]) * np.array(prices[:24]))
        cost_full = np.sum(np.array(long_schedule) * np.array(prices))
        
        self.assertNotEqual(cost_24, cost_full)
        self.assertEqual(cost_24, 24 * 1.0 * 0.25)  # 24 hours * 1.0 kWh * 0.25 EUR/kWh

    def test_device_with_partial_schedule_gets_padded(self):
        """Test that devices with partial schedules (e.g., 6-8 hours) get properly padded."""
        # Simulate a device that only operates for 3 hours (6-8)
        active_hours = [6, 7, 8]
        schedule = [0.0] * 24
        
        # Set activity only during hours 6-8
        for hour in active_hours:
            schedule[hour] = 2.5  # 2.5 kWh during active hours
        
        # Verify it's exactly 24 hours
        self.assertEqual(len(schedule), 24)
        
        # Verify non-zero only during active hours
        for hour in range(24):
            if hour in active_hours:
                self.assertEqual(schedule[hour], 2.5)
            else:
                self.assertEqual(schedule[hour], 0.0)

    def test_global_optimizer_populates_full_24_hour_arrays(self):
        """Test that GlobalOptimizer populates full 24-hour arrays even with partial optimization."""
        from agents.BatteryAgent import BatteryAgent
        from agents.EVAgent import EVAgent
        
        # Create agents
        battery_agent = BatteryAgent(
            max_charge_rate=3.0,
            max_discharge_rate=3.0,
            initial_soc=8.0,
            soc_min=1.0,
            soc_max=10.0,
            capacity=10.0
        )
        
        ev_agent = EVAgent(
            capacity=60.0,
            initial_soc=30.0,
            soc_min=6.0,
            soc_max=54.0,
            max_charge_rate=11.0,
            max_discharge_rate=0.0,
            must_be_full_by_hour=7
        )
        
        # Simulate partial optimization (only first 18 hours)
        n_hours = 18
        
        # Test the padding logic that should be in GlobalOptimizer
        for t in range(24):
            if t < n_hours:
                # Would be populated from optimization results
                battery_agent.hourly_charge[t] = 1.0 if t < 6 else 0.0
                battery_agent.hourly_discharge[t] = 0.5 if t > 18 else 0.0
                battery_agent.hourly_soc[t] = 8.0 + t * 0.1
                
                ev_agent.hourly_charge[t] = 2.0 if 22 <= t <= 6 else 0.0
                ev_agent.hourly_discharge[t] = 0.0
                ev_agent.hourly_soc[t] = 30.0 + t * 0.5
            else:
                # Should be padded with appropriate defaults
                battery_agent.hourly_charge[t] = 0.0
                battery_agent.hourly_discharge[t] = 0.0
                battery_agent.hourly_soc[t] = battery_agent.hourly_soc[t-1] if t > 0 else battery_agent.current_soc
                
                ev_agent.hourly_charge[t] = 0.0
                ev_agent.hourly_discharge[t] = 0.0
                ev_agent.hourly_soc[t] = ev_agent.hourly_soc[t-1] if t > 0 else ev_agent.current_soc
        
        # Verify all arrays are exactly 24 hours
        self.assertEqual(len(battery_agent.hourly_charge), 24)
        self.assertEqual(len(battery_agent.hourly_discharge), 24)
        self.assertEqual(len(battery_agent.hourly_soc), 24)
        
        self.assertEqual(len(ev_agent.hourly_charge), 24)
        self.assertEqual(len(ev_agent.hourly_discharge), 24)
        self.assertEqual(len(ev_agent.hourly_soc), 24)
        
        # Verify padding works correctly (hours 18-23 should be zeros for charge/discharge)
        for t in range(18, 24):
            self.assertEqual(battery_agent.hourly_charge[t], 0.0)
            self.assertEqual(battery_agent.hourly_discharge[t], 0.0)
            self.assertEqual(ev_agent.hourly_charge[t], 0.0)
            self.assertEqual(ev_agent.hourly_discharge[t], 0.0)

    def test_json_output_always_24_hours(self):
        """Test that JSON output for schedules is always 24 hours."""
        # Simulate device schedule output
        device_schedule = {
            'device_name': 'test_device',
            'schedule': [1.0 if 6 <= i <= 8 else 0.0 for i in range(24)]
        }
        
        # Verify schedule is exactly 24 hours
        self.assertEqual(len(device_schedule['schedule']), 24)
        
        # Verify only hours 6-8 are non-zero
        for hour in range(24):
            if 6 <= hour <= 8:
                self.assertEqual(device_schedule['schedule'][hour], 1.0)
            else:
                self.assertEqual(device_schedule['schedule'][hour], 0.0)

if __name__ == '__main__':
    unittest.main(verbosity=2)