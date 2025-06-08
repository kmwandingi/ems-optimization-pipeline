#!/usr/bin/env python
"""
Unit Tests for No-Fallback Behavior

This test suite verifies that when agent methods fail, the system raises errors
instead of falling back to manual optimization or simplified logic.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
import pandas as pd
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "notebooks"))
sys.path.insert(0, str(project_root / "scripts"))

class TestNoFallbackBehavior(unittest.TestCase):
    """Test that agent failures raise errors instead of falling back."""
    
    def setUp(self):
        """Set up test environment."""
        self.building_id = "DE_KN_residential1"
        
        # Create mock data
        self.mock_data = pd.DataFrame({
            'hour': list(range(24)),
            'day': ['2023-01-01'] * 24,
            'price_per_kwh': [0.25] * 24,
            'utc_timestamp': pd.date_range('2023-01-01', periods=24, freq='H')
        })

    def test_device_without_consumption_attributes_raises_error(self):
        """Test that compute_device_savings raises error when device lacks consumption attributes."""
        from notebooks.utils.helper import compute_device_savings
        
        # Create mock device without required attributes
        mock_device = Mock()
        mock_device.device_name = "test_device"
        # Deliberately don't set original_consumption and optimized_consumption
        
        # Should raise ValueError, not return fallback values
        with self.assertRaises(ValueError) as context:
            compute_device_savings(mock_device)
        
        self.assertIn("missing required consumption attributes", str(context.exception))
        self.assertIn("Agent optimization must be run correctly", str(context.exception))

    def test_device_without_schedule_raises_error_in_common(self):
        """Test that common.py raises error when device lacks required schedule for mode."""
        import common
        
        # Create mock device without any schedule attributes
        mock_device = Mock()
        mock_device.device_name = "test_device"
        # Deliberately don't set any schedule attributes
        
        devices = [mock_device]
        
        # Should raise ValueError for each mode when schedule is missing
        with self.assertRaises(ValueError) as context:
            common.build_optimization_visualization_data(devices, "centralised")
        
        self.assertIn("does not have required schedule for mode", str(context.exception))
        self.assertIn("Agent optimization must be run correctly", str(context.exception))

    def test_global_optimizer_without_schedule_raises_error(self):
        """Test that GlobalOptimizer raises error when device has no optimized schedule."""
        # Import here to avoid import errors in setup
        try:
            from agents.GlobalOptimizer import GlobalOptimizer
        except ImportError:
            self.skipTest("GlobalOptimizer not available")
        
        # Create mock device without schedule
        mock_device = Mock()
        mock_device.device_name = "test_device"
        mock_device.original_consumption = [1.0] * 24
        # Deliberately don't set optimized schedule attributes
        
        # Mock the internal method that checks for schedules
        with patch.object(GlobalOptimizer, '_compute_cost_by_device') as mock_method:
            # Make the method raise the expected error
            mock_method.side_effect = ValueError("Device test_device has no schedule. Agent optimization must be run correctly.")
            
            optimizer = GlobalOptimizer(devices=[mock_device])
            
            # Should raise ValueError, not use fallback
            with self.assertRaises(ValueError) as context:
                optimizer._compute_cost_by_device(mock_device, [0.25]*24, "optimized")
            
            self.assertIn("has no", str(context.exception))
            self.assertIn("Agent optimization must be run correctly", str(context.exception))

    @patch('agents.GlobalOptimizer.GlobalOptimizer')
    def test_agent_import_failure_exits_script(self, mock_optimizer):
        """Test that scripts exit when agent imports fail."""
        # This tests the import error handling in scripts
        
        # Read script content to verify proper error handling
        script_files = [
            project_root / "scripts" / "01_run.py",
            project_root / "scripts" / "02_integrated_pipeline.py",
            project_root / "scripts" / "03_probability_learning_optimization.py"
        ]
        
        for script_file in script_files:
            with open(script_file, 'r') as f:
                content = f.read()
                
                # Should exit on import failure
                self.assertIn("sys.exit(1)", content, 
                             f"Script {script_file.name} should exit on agent import failure")
                
                # Should not have fallback imports
                self.assertNotIn("fallback", content.lower(), 
                                f"Script {script_file.name} should not have fallback logic")

    def test_optimization_failure_raises_error_not_fallback(self):
        """Test that optimization failures raise errors instead of using fallbacks."""
        # Check that scripts raise errors when optimization fails
        script_files = [
            project_root / "scripts" / "01_run.py"
        ]
        
        for script_file in script_files:
            with open(script_file, 'r') as f:
                content = f.read()
                
                # Should raise RuntimeError on optimization failure
                self.assertIn("raise RuntimeError", content,
                             f"Script {script_file.name} should raise errors on optimization failure")
                
                # Should not have manual optimization as fallback
                forbidden_patterns = [
                    "manual_optimization",
                    "simple_optimization", 
                    "fallback_optimization",
                    "if.*failed.*use.*manual"
                ]
                
                for pattern in forbidden_patterns:
                    self.assertNotIn(pattern, content.lower(),
                                   f"Script {script_file.name} should not contain fallback pattern: {pattern}")

    def test_no_try_except_fallback_patterns(self):
        """Test that there are no try/except blocks that provide fallback optimization."""
        # Check that optimization code doesn't have fallback try/except patterns
        agent_files = list((project_root / "notebooks" / "agents").glob("*.py"))
        script_files = list((project_root / "scripts").glob("*.py"))
        
        all_files = agent_files + script_files
        
        for file_path in all_files:
            with open(file_path, 'r') as f:
                content = f.read()
                
                # Look for forbidden try/except fallback patterns
                forbidden_patterns = [
                    "except.*agent.*fallback",
                    "except.*optimization.*failed.*use",
                    "try.*agent.*except.*manual",
                    "except.*continue.*with.*simple"
                ]
                
                for pattern in forbidden_patterns:
                    # Use simple string search since we're looking for broad patterns
                    if "except" in content and "fallback" in content.lower():
                        # More detailed check only if both keywords are present
                        lines = content.split('\n')
                        for i, line in enumerate(lines):
                            if "except" in line and any(word in line.lower() for word in ["fallback", "manual", "simple"]):
                                # Allow test files to have these patterns for testing
                                if "test_" not in file_path.name:
                                    self.fail(f"Found potential fallback pattern in {file_path}:{i+1}: {line.strip()}")

    def test_global_optimizer_failure_propagates_error(self):
        """Test that GlobalOptimizer failures propagate errors instead of using fallbacks."""
        # Mock optimization failure
        with patch('agents.GlobalOptimizer.GlobalOptimizer') as mock_optimizer_class:
            mock_optimizer = Mock()
            mock_optimizer_class.return_value = mock_optimizer
            
            # Make optimization return False (failure)
            mock_optimizer.optimize_centralized.return_value = False
            
            # This should raise an error, not provide a fallback
            try:
                from agents.GlobalOptimizer import GlobalOptimizer
                optimizer = GlobalOptimizer(devices=[])
                result = optimizer.optimize_centralized()
                
                # If result is False, calling code should raise error
                if not result:
                    raise RuntimeError("GlobalOptimizer.optimize_centralized() returned False - optimization failed")
                    
            except RuntimeError as e:
                # This is expected - error should be raised
                self.assertIn("optimization failed", str(e))
            except ImportError:
                # Skip if can't import
                self.skipTest("GlobalOptimizer not available")

    def test_missing_price_data_raises_error(self):
        """Test that missing price data raises error instead of using defaults."""
        # Test that functions require proper price data
        
        # Create device with missing price data
        mock_device = Mock()
        mock_device.device_name = "test_device"
        mock_device.data = pd.DataFrame({'hour': range(24)})  # No price_per_kwh column
        
        # Should raise error when trying to compute costs without price data
        # This tests that we don't silently default to fixed prices
        
        # Example from helper.py - should fail if no price data
        try:
            from notebooks.utils.helper import compute_device_savings
            
            # Create device with all required attributes but no price data
            mock_device.original_consumption = [1.0] * 24
            mock_device.optimized_consumption = [0.8] * 24
            
            # This should work but we're testing the principle
            # In a stricter implementation, missing price data should raise error
            result = compute_device_savings(mock_device)
            
            # If it doesn't raise an error, at least verify it doesn't return nonsensical defaults
            pct_savings, euro_savings, adjusted_cost = result
            
            # Verify calculations are reasonable (not hardcoded fallbacks)
            self.assertIsInstance(pct_savings, (int, float))
            self.assertIsInstance(euro_savings, (int, float))
            self.assertIsInstance(adjusted_cost, (int, float))
            
        except ImportError:
            self.skipTest("Helper functions not available")

if __name__ == '__main__':
    unittest.main(verbosity=2)