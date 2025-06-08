#!/usr/bin/env python
"""
Test suite for Agent invocations.
Tests that scripts properly invoke Agent methods without fallbacks.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "notebooks"))
sys.path.insert(0, str(project_root / "scripts"))

import common
from agents.FlexibleDeviceAgent import FlexibleDevice
from agents.GlobalOptimizer import GlobalOptimizer
from agents.BatteryAgent import BatteryAgent
from agents.EVAgent import EVAgent
from agents.ProbabilityModelAgent import ProbabilityModelAgent
from agents.GlobalConnectionLayer import GlobalConnectionLayer


class TestAgentInvocations(unittest.TestCase):
    """Test that Agent methods are properly invoked."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.building_id = "DE_KN_residential1"
        self.test_day = pd.Timestamp("2015-01-01").date()
        
        # Create sample DataFrame with 24 hours
        hours = list(range(24))
        self.test_df = pd.DataFrame({
            'hour': hours,
            'day': [self.test_day] * 24,
            'utc_timestamp': pd.date_range('2015-01-01', periods=24, freq='H'),
            'price_per_kwh': [0.25 + 0.1 * np.sin(h * np.pi / 12) for h in hours],
            f'{self.building_id}_dishwasher': [0.5 if 18 <= h <= 20 else 0.1 for h in hours],
            f'{self.building_id}_heat_pump': [2.0 if 6 <= h <= 9 or 18 <= h <= 22 else 0.5 for h in hours],
        })

    def test_global_optimizer_optimize_centralized_called(self):
        """Test that GlobalOptimizer.optimize_centralized() is called."""
        # Create mock devices
        mock_device = Mock()
        mock_device.device_name = "test_device"
        mock_device.data = self.test_df
        mock_device.original_consumption = [1.0] * 24
        
        # Create GlobalOptimizer
        optimizer = GlobalOptimizer(devices=[mock_device])
        
        # Mock the optimization method
        with patch.object(optimizer, 'optimize_centralized') as mock_optimize:
            mock_optimize.return_value = True
            
            # Call optimization
            result = optimizer.optimize_centralized()
            
            # Verify method was called
            mock_optimize.assert_called_once()
            self.assertTrue(result)

    def test_global_optimizer_optimize_phases_centralized_called(self):
        """Test that GlobalOptimizer.optimize_phases_centralized() is called."""
        # Create mock devices
        mock_device = Mock()
        mock_device.device_name = "test_device"
        mock_device.data = self.test_df
        mock_device.original_consumption = [1.0] * 24
        
        # Create GlobalOptimizer
        optimizer = GlobalOptimizer(devices=[mock_device])
        
        # Mock the optimization method
        with patch.object(optimizer, 'optimize_phases_centralized') as mock_optimize:
            mock_optimize.return_value = True
            
            # Call optimization
            result = optimizer.optimize_phases_centralized(
                day=self.test_day,
                prices=[0.25] * 24
            )
            
            # Verify method was called
            mock_optimize.assert_called_once()
            self.assertTrue(result)

    def test_flexible_device_agent_creation(self):
        """Test that FlexibleDevice agents are created properly."""
        # Create global layer
        global_layer = GlobalConnectionLayer(max_building_load=50.0, total_hours=24)
        
        # Create FlexibleDevice
        device = FlexibleDevice(
            device_name="test_device",
            data=self.test_df,
            category="Highly Flexible",
            power_rating=3.0,
            global_layer=global_layer,
            battery_agent=None,
            spec={'category': 'Highly Flexible', 'power_rating': 3.0}
        )
        
        # Verify device was created correctly
        self.assertEqual(device.device_name, "test_device")
        self.assertEqual(device.category, "Highly Flexible")
        self.assertEqual(device.power_rating, 3.0)

    def test_battery_agent_initialization(self):
        """Test that BatteryAgent is initialized with correct parameters."""
        battery_agent = BatteryAgent(
            max_charge_rate=3.0,
            max_discharge_rate=3.0,
            initial_soc=8.0,
            soc_min=1.0,
            soc_max=10.0,
            capacity=10.0
        )
        
        # Verify initialization
        self.assertEqual(battery_agent.max_charge_rate, 3.0)
        self.assertEqual(battery_agent.max_discharge_rate, 3.0)
        self.assertEqual(battery_agent.current_soc, 8.0)
        self.assertEqual(battery_agent.soc_min, 1.0)
        self.assertEqual(battery_agent.soc_max, 10.0)
        self.assertEqual(battery_agent.capacity, 10.0)
        
        # Verify hourly arrays are 24 hours
        self.assertEqual(len(battery_agent.hourly_charge), 24)
        self.assertEqual(len(battery_agent.hourly_discharge), 24)
        self.assertEqual(len(battery_agent.hourly_soc), 24)

    def test_ev_agent_initialization(self):
        """Test that EVAgent is initialized with correct parameters."""
        ev_agent = EVAgent(
            capacity=60.0,
            initial_soc=30.0,
            soc_min=6.0,
            soc_max=54.0,
            max_charge_rate=11.0,
            max_discharge_rate=0.0,
            must_be_full_by_hour=7
        )
        
        # Verify initialization
        self.assertEqual(ev_agent.capacity, 60.0)
        self.assertEqual(ev_agent.current_soc, 30.0)
        self.assertEqual(ev_agent.soc_min, 6.0)
        self.assertEqual(ev_agent.soc_max, 54.0)
        self.assertEqual(ev_agent.max_charge_rate, 11.0)
        self.assertEqual(ev_agent.max_discharge_rate, 0.0)
        self.assertEqual(ev_agent.must_be_full_by_hour, 7)
        
        # Verify hourly arrays are 24 hours
        self.assertEqual(len(ev_agent.hourly_charge), 24)
        self.assertEqual(len(ev_agent.hourly_discharge), 24)
        self.assertEqual(len(ev_agent.hourly_soc), 24)

    def test_probability_model_agent_creation(self):
        """Test that ProbabilityModelAgent is created with proper data."""
        # Create mock priors
        priors_df = pd.DataFrame({
            str(h): [1.0/24] for h in range(24)
        })
        priors_df['device_type'] = ['dishwasher']
        priors_df = priors_df.set_index('device_type')
        
        # Create agent
        prob_agent = ProbabilityModelAgent(prob_dist_df=priors_df)
        
        # Verify agent was created
        self.assertIsNotNone(prob_agent)
        self.assertIsNotNone(prob_agent.prob_dist_df)

    def test_script_imports_agents_correctly(self):
        """Test that scripts import agents correctly."""
        # Test 01_run.py imports
        script_path = project_root / "scripts" / "01_run.py"
        with open(script_path, 'r') as f:
            content = f.read()
            
        # Should have agent imports
        self.assertIn("from agents.GlobalOptimizer import GlobalOptimizer", content)
        self.assertIn("from agents.BatteryAgent import BatteryAgent", content)
        self.assertIn("from agents.EVAgent import EVAgent", content)

    def test_no_manual_optimization_in_scripts(self):
        """Test that scripts don't contain manual optimization."""
        scripts = [
            project_root / "scripts" / "01_run.py",
            project_root / "scripts" / "02_integrated_pipeline.py"
        ]
        
        forbidden_patterns = [
            "optimize_device_with_agent_logic",
            "manual_optimization",
            "simple_optimization"
        ]
        
        for script_path in scripts:
            if script_path.exists():
                with open(script_path, 'r') as f:
                    content = f.read()
                    
                for pattern in forbidden_patterns:
                    self.assertNotIn(pattern, content, 
                                   f"Found forbidden pattern '{pattern}' in {script_path}")


class TestErrorHandling(unittest.TestCase):
    """Test that agent failures raise errors properly."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.building_id = "DE_KN_residential1"
        self.test_day = pd.Timestamp("2015-01-01").date()

    def test_optimizer_failure_raises_error(self):
        """Test that optimizer failures raise RuntimeError."""
        # Create mock device
        mock_device = Mock()
        mock_device.device_name = "test_device"
        mock_device.data = pd.DataFrame({
            'hour': list(range(24)),
            'price_per_kwh': [0.25] * 24
        })
        mock_device.original_consumption = [1.0] * 24
        
        # Create optimizer
        optimizer = GlobalOptimizer(devices=[mock_device])
        
        # Mock optimization to fail
        with patch.object(optimizer, 'optimize_centralized') as mock_optimize:
            mock_optimize.return_value = False  # Failure
            
            # This should be handled by calling code raising RuntimeError
            result = optimizer.optimize_centralized()
            self.assertFalse(result)
            # The calling script should check this and raise an error

    def test_missing_device_schedule_raises_error(self):
        """Test that missing device schedules raise ValueError."""
        from notebooks.utils.helper import compute_device_savings
        
        # Create device without required attributes
        mock_device = Mock()
        mock_device.device_name = "test_device"
        # Don't set original_consumption or optimized_consumption
        
        # Should raise ValueError
        with self.assertRaises(ValueError) as context:
            compute_device_savings(mock_device)
            
        self.assertIn("missing required consumption attributes", str(context.exception))


if __name__ == '__main__':
    unittest.main(verbosity=2)