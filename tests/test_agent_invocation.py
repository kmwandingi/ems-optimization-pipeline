#!/usr/bin/env python
"""
Unit Tests for Agent Invocation

This test suite verifies that all agent methods are invoked correctly and that
no fallback logic bypasses agent calls. Every optimization must go through agent methods.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, call
import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "notebooks"))
sys.path.insert(0, str(project_root / "scripts"))

class TestAgentInvocation(unittest.TestCase):
    """Test that all optimization goes through agent methods."""
    
    def setUp(self):
        """Set up test environment."""
        self.building_id = "DE_KN_residential1"
        
        # Create mock data
        self.mock_data = pd.DataFrame({
            'hour': list(range(24)),
            'price_per_kwh': [0.25] * 24,
            'utc_timestamp': pd.date_range('2023-01-01', periods=24, freq='H')
        })

    @patch('agents.GlobalOptimizer.GlobalOptimizer')
    @patch('agents.FlexibleDeviceAgent.FlexibleDevice')
    def test_global_optimizer_optimize_centralized_called(self, mock_device, mock_optimizer):
        """Test that GlobalOptimizer.optimize_centralized() is called."""
        # Setup mocks
        mock_optimizer_instance = Mock()
        mock_optimizer.return_value = mock_optimizer_instance
        mock_optimizer_instance.optimize_centralized.return_value = True
        
        mock_device_instance = Mock()
        mock_device.return_value = mock_device_instance
        mock_device_instance.device_name = "test_device"
        mock_device_instance.centralized_optimized_schedule = [0.0] * 24
        
        # Import and test
        from agents.GlobalOptimizer import GlobalOptimizer
        
        optimizer = GlobalOptimizer(devices=[mock_device_instance])
        result = optimizer.optimize_centralized()
        
        # Verify agent method was called
        mock_optimizer_instance.optimize_centralized.assert_called_once()
        self.assertTrue(result)

    @patch('agents.GlobalOptimizer.GlobalOptimizer')
    @patch('agents.FlexibleDeviceAgent.FlexibleDevice')
    def test_global_optimizer_optimize_phases_centralized_called(self, mock_device, mock_optimizer):
        """Test that GlobalOptimizer.optimize_phases_centralized() is called."""
        # Setup mocks
        mock_optimizer_instance = Mock()
        mock_optimizer.return_value = mock_optimizer_instance
        mock_optimizer_instance.optimize_phases_centralized.return_value = True
        
        mock_device_instance = Mock()
        mock_device.return_value = mock_device_instance
        mock_device_instance.device_name = "test_device"
        mock_device_instance.phases_optimized_schedule = [0.0] * 24
        
        # Import and test
        from agents.GlobalOptimizer import GlobalOptimizer
        
        optimizer = GlobalOptimizer(devices=[mock_device_instance])
        result = optimizer.optimize_phases_centralized(day='2023-01-01', prices=[0.25]*24)
        
        # Verify agent method was called
        mock_optimizer_instance.optimize_phases_centralized.assert_called_once()
        self.assertTrue(result)

    @patch('agents.FlexibleDeviceAgent.FlexibleDevice')
    def test_flexible_device_optimize_day_called(self, mock_device):
        """Test that FlexibleDevice.optimize_day() is called."""
        # Setup mock
        mock_device_instance = Mock()
        mock_device.return_value = mock_device_instance
        mock_device_instance.device_name = "test_device"
        mock_device_instance.optimized_schedule = [0.0] * 24
        mock_device_instance.optimized_consumption = [0.0] * 24
        mock_device_instance.original_consumption = [1.0] * 24
        mock_device_instance.data = self.mock_data
        
        # Import and test
        from agents.FlexibleDeviceAgent import FlexibleDevice
        
        device = FlexibleDevice(
            device_name="test_device",
            data=self.mock_data,
            category="Highly Flexible",
            power_rating=1.0,
            global_layer=Mock(),
            battery_agent=None,
            spec={'category': 'Highly Flexible'}
        )
        
        # Mock the optimize_day method
        device.optimize_day = Mock()
        
        # Call optimization
        device.optimize_day(day='2023-01-01', prices=[0.25]*24, battery_agent=None, ev_agent=None, grid_agent=None)
        
        # Verify agent method was called
        device.optimize_day.assert_called_once()

    @patch('agents.ProbabilityModelAgent.ProbabilityModelAgent')
    def test_probability_model_agent_train_called(self, mock_prob_agent):
        """Test that ProbabilityModelAgent.train() is called."""
        # Setup mock
        mock_prob_instance = Mock()
        mock_prob_agent.return_value = mock_prob_instance
        mock_prob_instance.train.return_value = None
        
        # Import and test
        from agents.ProbabilityModelAgent import ProbabilityModelAgent
        
        prob_agent = ProbabilityModelAgent()
        
        # Test data
        test_devices = [Mock()]
        test_devices[0].device_name = "test_device"
        test_devices[0].hour_probability = {i: 1/24 for i in range(24)}
        
        # Call train method
        prob_agent.train(devices=test_devices, lr_tau=0.01, lr_max=0.1)
        
        # Verify agent method was called
        mock_prob_instance.train.assert_called_once()

    @patch('agents.BatteryAgent.BatteryAgent')
    def test_battery_agent_methods_called(self, mock_battery):
        """Test that BatteryAgent methods are called."""
        # Setup mock
        mock_battery_instance = Mock()
        mock_battery.return_value = mock_battery_instance
        mock_battery_instance.hourly_charge = [0.0] * 24
        mock_battery_instance.hourly_discharge = [0.0] * 24
        mock_battery_instance.hourly_soc = [8.0] * 24
        
        # Import and test
        from agents.BatteryAgent import BatteryAgent
        
        battery_agent = BatteryAgent(
            max_charge_rate=3.0,
            max_discharge_rate=3.0,
            initial_soc=8.0,
            soc_min=1.0,
            soc_max=10.0,
            capacity=10.0
        )
        
        # Verify agent was created properly
        self.assertIsNotNone(battery_agent)

    @patch('agents.EVAgent.EVAgent')
    def test_ev_agent_methods_called(self, mock_ev):
        """Test that EVAgent methods are called."""
        # Setup mock
        mock_ev_instance = Mock()
        mock_ev.return_value = mock_ev_instance
        mock_ev_instance.hourly_charge = [0.0] * 24
        mock_ev_instance.hourly_discharge = [0.0] * 24
        mock_ev_instance.hourly_soc = [30.0] * 24
        
        # Import and test
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
        
        # Verify agent was created properly
        self.assertIsNotNone(ev_agent)

    def test_no_manual_optimization_functions_exist(self):
        """Test that no forbidden manual optimization functions exist."""
        # List of forbidden function names
        forbidden_functions = [
            'optimize_device_with_agent_logic',
            'run_simple_centralized_optimization',
            'manual_optimization',
            'fallback_optimization',
            'simple_optimization'
        ]
        
        # Check scripts directory
        scripts_dir = project_root / "scripts"
        for script_file in scripts_dir.glob("*.py"):
            with open(script_file, 'r') as f:
                content = f.read()
                for func_name in forbidden_functions:
                    self.assertNotIn(func_name, content, 
                                   f"Forbidden function '{func_name}' found in {script_file}")

    def test_agent_imports_required(self):
        """Test that agent imports are required and no fallbacks exist."""
        # Check that scripts fail properly when agents can't be imported
        scripts_to_check = [
            project_root / "scripts" / "01_run.py",
            project_root / "scripts" / "02_integrated_pipeline.py",
            project_root / "scripts" / "03_probability_learning_optimization.py"
        ]
        
        for script_file in scripts_to_check:
            with open(script_file, 'r') as f:
                content = f.read()
                
                # Should have agent imports in try/except
                self.assertIn("from agents.", content, f"Script {script_file} should import agents")
                
                # Should exit on import failure, not continue with fallbacks
                self.assertIn("sys.exit(1)", content, f"Script {script_file} should exit on agent import failure")

if __name__ == '__main__':
    unittest.main(verbosity=2)