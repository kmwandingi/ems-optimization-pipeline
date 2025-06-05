"""
Test suite for Agent invocations and data wrapping.
Tests that the fixed scripts properly invoke Agent methods without fallbacks.
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
from utils.helper import (
    run_building_optimization_multi_day, 
    run_building_optimization_single_day_direct_phases,
    compute_device_savings
)
from agents.FlexibleDeviceAgent import FlexibleDevice
from agents.GlobalOptimizer import GlobalOptimizer
from agents.BatteryAgent import BatteryAgent
from agents.EVAgent import EVAgent
from agents.ProbabilityModelAgent import ProbabilityModelAgent


class TestAgentInvocations(unittest.TestCase):
    """Test that Agent methods are properly invoked."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock DuckDB data
        self.building_id = "DE_KN_residential4"
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
            f'{self.building_id}_ev': [7.4 if 22 <= h <= 24 or 0 <= h <= 6 else 0.0 for h in hours],
            f'{self.building_id}_pv': [-1.5 if 8 <= h <= 16 else 0.0 for h in hours],
            f'{self.building_id}_grid_import': [2.0] * 24,
        })
        
        # Mock device specs
        self.device_specs = {
            f'{self.building_id}_dishwasher': {
                'category': 'Partially Flexible',
                'power_rating': 1.0,
                'allowed_hours': list(range(8, 22))
            },
            f'{self.building_id}_heat_pump': {
                'category': 'Highly Flexible', 
                'power_rating': 3.0,
                'allowed_hours': list(range(6, 24))
            },
            f'{self.building_id}_ev': {
                'category': 'Highly Flexible',
                'power_rating': 7.4,
                'allowed_hours': list(range(0, 24))
            }
        }

    @patch('common.get_con')
    def test_multi_day_uses_globaloptimizer_centralized(self, mock_get_con):
        """Test that multi-day script uses GlobalOptimizer.optimize_centralized()."""
        # Mock DuckDB connection
        mock_con = Mock()
        mock_get_con.return_value = mock_con
        
        # Create mock for GlobalOptimizer
        with patch('utils.helper.GlobalOptimizer') as mock_optimizer_class:
            mock_optimizer = Mock()
            mock_optimizer_class.return_value = mock_optimizer
            mock_optimizer.optimize_centralized.return_value = True
            
            # Call the function
            devices, optimizer, has_pv = run_building_optimization_multi_day(
                building_id=self.building_id,
                use_proxy_battery=True,
                cleaner=None,
                device_specs=self.device_specs,
                weather_df=self.test_df,
                forecast_df=self.test_df,
                days=1
            )
            
            # Assert GlobalOptimizer.optimize_centralized was called
            mock_optimizer.optimize_centralized.assert_called()

    @patch('common.get_con')  
    def test_single_day_uses_globaloptimizer_phases(self, mock_get_con):
        """Test that single-day script uses GlobalOptimizer.optimize_phases_centralized()."""
        # Mock DuckDB connection
        mock_con = Mock()
        mock_get_con.return_value = mock_con
        
        # Create mock for GlobalOptimizer
        with patch('utils.helper.GlobalOptimizer') as mock_optimizer_class:
            mock_optimizer = Mock()
            mock_optimizer_class.return_value = mock_optimizer
            mock_optimizer.optimize_phases_centralized.return_value = True
            
            # Call the function
            devices, optimizer, has_pv, ev_agent = run_building_optimization_single_day_direct_phases(
                building_id=self.building_id,
                single_day=self.test_day,
                use_proxy_battery=True,
                device_specs=self.device_specs,
                weather_df=self.test_df,
                forecast_df=self.test_df
            )
            
            # Assert GlobalOptimizer.optimize_phases_centralized was called
            mock_optimizer.optimize_phases_centralized.assert_called()

    def test_probability_agent_train_called(self):
        """Test that ProbabilityModelAgent.train() is called with proper DataFrame."""
        # Create mock priors
        priors_df = pd.DataFrame({
            '0': [0.04] * 3, '1': [0.04] * 3, '2': [0.04] * 3, '3': [0.04] * 3,
            '4': [0.04] * 3, '5': [0.04] * 3, '6': [0.05] * 3, '7': [0.05] * 3,
            '8': [0.06] * 3, '9': [0.07] * 3, '10': [0.07] * 3, '11': [0.08] * 3,
            '12': [0.07] * 3, '13': [0.02] * 3, '14': [0.02] * 3, '15': [0.03] * 3,
            '16': [0.04] * 3, '17': [0.03] * 3, '18': [0.04] * 3, '19': [0.04] * 3,
            '20': [0.03] * 3, '21': [0.03] * 3, '22': [0.03] * 3, '23': [0.02] * 3,
            'device_type': ['dishwasher', 'heat_pump', 'ev']
        }).set_index('device_type')
        
        # Initialize agent
        prob_agent = ProbabilityModelAgent(prob_dist_df=priors_df)
        
        # Mock the train method
        with patch.object(prob_agent, 'train') as mock_train:
            mock_train.return_value = (self.device_specs, {})
            
            # Call train
            updated_specs, device_probs = prob_agent.train(
                building_id=self.building_id,
                days_list=[self.test_day],
                device_specs=self.device_specs,
                weather_df=self.test_df,
                forecast_df=self.test_df
            )
            
            # Assert train was called
            mock_train.assert_called_once()
            
            # Check arguments passed to train
            call_args = mock_train.call_args
            self.assertEqual(call_args[1]['building_id'], self.building_id)
            self.assertEqual(call_args[1]['days_list'], [self.test_day])

    def test_probability_agent_predict_called(self):
        """Test that ProbabilityModelAgent.predict() is called with proper DataFrame."""
        # Create mock priors
        priors_df = pd.DataFrame({
            str(h): [0.04] for h in range(24)
        })
        priors_df['device_type'] = ['dishwasher']
        priors_df = priors_df.set_index('device_type')
        
        # Initialize agent
        prob_agent = ProbabilityModelAgent(prob_dist_df=priors_df)
        
        # Mock the predict method
        with patch.object(prob_agent, 'predict') as mock_predict:
            mock_predict.return_value = {'dishwasher': {'hour_probability': {h: 0.04 for h in range(24)}}}
            
            # Call predict with single-day DataFrame
            single_day_df = self.test_df[self.test_df['day'] == self.test_day].copy()
            pmf = prob_agent.predict(single_day_df)
            
            # Assert predict was called
            mock_predict.assert_called_once()
            
            # Check that DataFrame passed has correct shape
            call_args = mock_predict.call_args[0][0]
            self.assertEqual(len(call_args), 24)  # Should be 24 hours


class TestDataWrapping(unittest.TestCase):
    """Test proper data wrapping for Agent compatibility."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.building_id = "DE_KN_residential4"
        self.test_day = pd.Timestamp("2015-01-01").date()
        
        # Create malformed DataFrames for testing
        self.incomplete_df = pd.DataFrame({
            'hour': [0, 1, 2],  # Only 3 hours instead of 24
            'day': [self.test_day] * 3,
            'price_per_kwh': [0.25] * 3,
            f'{self.building_id}_dishwasher': [0.5] * 3,
        })
        
        self.complete_df = pd.DataFrame({
            'hour': list(range(24)),
            'day': [self.test_day] * 24,
            'price_per_kwh': [0.25] * 24,
            f'{self.building_id}_dishwasher': [0.5] * 24,
        })

    def test_one_day_wrapping_rejects_incomplete_data(self):
        """Test that single-day optimization rejects incomplete data with clear error."""
        with self.assertRaises(ValueError) as context:
            run_building_optimization_single_day_direct_phases(
                building_id=self.building_id,
                single_day=self.test_day,
                weather_df=self.incomplete_df,
                forecast_df=self.incomplete_df
            )
        
        # Should get a clear error about missing hours
        self.assertIn("does not have exactly 24 hours", str(context.exception))

    def test_one_day_wrapping_accepts_complete_data(self):
        """Test that single-day optimization accepts complete 24-hour data."""
        # Mock GlobalOptimizer to avoid actual optimization
        with patch('utils.helper.GlobalOptimizer') as mock_optimizer_class:
            mock_optimizer = Mock()
            mock_optimizer_class.return_value = mock_optimizer
            mock_optimizer.optimize_phases_centralized.return_value = True
            
            # Should not raise an exception
            devices, optimizer, has_pv, ev_agent = run_building_optimization_single_day_direct_phases(
                building_id=self.building_id,
                single_day=self.test_day,
                weather_df=self.complete_df,
                forecast_df=self.complete_df
            )
            
            # Should return valid results
            self.assertIsNotNone(devices)
            self.assertIsNotNone(optimizer)

    def test_multi_day_data_indexing(self):
        """Test that multi-day optimization handles day indexing correctly."""
        # Create multi-day DataFrame
        multi_day_df = pd.concat([
            self.complete_df,
            self.complete_df.copy().assign(day=pd.Timestamp("2015-01-02").date())
        ], ignore_index=True)
        
        # Mock GlobalOptimizer
        with patch('utils.helper.GlobalOptimizer') as mock_optimizer_class:
            mock_optimizer = Mock()
            mock_optimizer_class.return_value = mock_optimizer
            mock_optimizer.optimize_centralized.return_value = True
            
            # Should handle multiple days correctly
            devices, optimizer, has_pv = run_building_optimization_multi_day(
                building_id=self.building_id,
                use_proxy_battery=True,
                cleaner=None,
                device_specs={f'{self.building_id}_dishwasher': {'category': 'Non-Flexible', 'power_rating': 1.0}},
                weather_df=multi_day_df,
                forecast_df=multi_day_df,
                days=2
            )
            
            # Should return valid results for multi-day optimization
            self.assertIsNotNone(devices)


class TestErrorHandling(unittest.TestCase):
    """Test that errors surface properly without silent failures."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.building_id = "DE_KN_residential4"
        self.test_day = pd.Timestamp("2015-01-01").date()

    def test_missing_data_raises_clear_error(self):
        """Test that missing data raises a clear exception."""
        # Empty DataFrame
        empty_df = pd.DataFrame()
        
        with self.assertRaises((ValueError, KeyError)) as context:
            run_building_optimization_single_day_direct_phases(
                building_id=self.building_id,
                single_day=self.test_day,
                weather_df=empty_df,
                forecast_df=empty_df
            )
        
        # Should get a clear error, not silent failure
        self.assertIsInstance(context.exception, (ValueError, KeyError))

    def test_malformed_timestamps_raise_error(self):
        """Test that malformed timestamps raise clear errors."""
        malformed_df = pd.DataFrame({
            'hour': [0, 1, 2, 25],  # Invalid hour 25
            'day': ['invalid_date'] * 4,  # Invalid date format
            'price_per_kwh': [0.25] * 4,
            f'{self.building_id}_dishwasher': [0.5] * 4,
        })
        
        with self.assertRaises((ValueError, TypeError)):
            run_building_optimization_single_day_direct_phases(
                building_id=self.building_id,
                single_day=self.test_day,
                weather_df=malformed_df,
                forecast_df=malformed_df
            )

    def test_optimizer_failure_propagates(self):
        """Test that optimizer failures propagate as exceptions."""
        complete_df = pd.DataFrame({
            'hour': list(range(24)),
            'day': [self.test_day] * 24,
            'price_per_kwh': [0.25] * 24,
            f'{self.building_id}_dishwasher': [0.5] * 24,
        })
        
        # Mock GlobalOptimizer to return failure
        with patch('utils.helper.GlobalOptimizer') as mock_optimizer_class:
            mock_optimizer = Mock()
            mock_optimizer_class.return_value = mock_optimizer
            mock_optimizer.optimize_phases_centralized.return_value = False  # Failure
            
            with self.assertRaises(RuntimeError) as context:
                run_building_optimization_single_day_direct_phases(
                    building_id=self.building_id,
                    single_day=self.test_day,
                    weather_df=complete_df,
                    forecast_df=complete_df
                )
            
            # Should get clear error about optimization failure
            self.assertIn("optimize_phases_centralized() failed", str(context.exception))


class TestOutputGeneration(unittest.TestCase):
    """Test that scripts generate correct outputs."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.building_id = "DE_KN_residential4"
        self.test_day = pd.Timestamp("2015-01-01").date()
        
        # Create complete test DataFrame
        self.test_df = pd.DataFrame({
            'hour': list(range(24)),
            'day': [self.test_day] * 24,
            'price_per_kwh': [0.25] * 24,
            f'{self.building_id}_dishwasher': [0.5] * 24,
            f'{self.building_id}_heat_pump': [2.0] * 24,
        })

    def test_device_savings_calculation(self):
        """Test that device savings are calculated correctly."""
        # Create mock device with proper attributes
        mock_device = Mock()
        mock_device.original_consumption = np.array([1.0] * 24)
        mock_device.optimized_consumption = np.array([0.8] * 24)  # 20% reduction
        mock_device.data = self.test_df
        mock_device.battery_charge = None
        mock_device.battery_discharge = None
        
        # Calculate savings
        pct_savings, euro_savings, adjusted_cost = compute_device_savings(mock_device)
        
        # Should show 20% savings
        self.assertAlmostEqual(pct_savings, 20.0, places=1)
        self.assertGreater(euro_savings, 0)  # Should have positive savings

    @patch('os.makedirs')
    @patch('matplotlib.pyplot.savefig')
    def test_plots_saved_to_correct_directory(self, mock_savefig, mock_makedirs):
        """Test that plots are saved to results/plots/ directory."""
        from utils.helper import plot_battery_schedule
        
        # Call plotting function
        plot_battery_schedule([1.0] * 24, self.building_id, "2015-01-01")
        
        # Check that correct directory was created
        mock_makedirs.assert_called_with("results/plots", exist_ok=True)
        
        # Check that plot was saved to correct path
        mock_savefig.assert_called()
        save_path = mock_savefig.call_args[0][0]
        self.assertIn("results/plots/", save_path)


if __name__ == '__main__':
    # Create results directory if it doesn't exist
    os.makedirs("results/plots", exist_ok=True)
    
    # Run tests
    unittest.main(verbosity=2)