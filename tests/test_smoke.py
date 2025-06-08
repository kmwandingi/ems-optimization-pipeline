"""
Smoke tests for the EMS pipeline.
Quick tests to verify basic functionality works end-to-end.
"""

import unittest
import subprocess
import sys
import os
from pathlib import Path
import tempfile
import duckdb
import pandas as pd
import numpy as np

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "notebooks"))
sys.path.insert(0, str(project_root / "scripts"))

import common


class TestSmokeTests(unittest.TestCase):
    """Quick smoke tests to verify basic functionality."""
    
    def setUp(self):
        """Set up minimal test environment."""
        self.building_id = "DE_KN_residential4"
        
        # Create temporary DuckDB for testing
        self.test_db_fd, self.test_db_path = tempfile.mkstemp(suffix='.duckdb')
        os.close(self.test_db_fd)  # Close the file descriptor
        os.unlink(self.test_db_path)  # Remove the empty file so DuckDB can create it
        
        # Create minimal test data
        self.create_test_database()
        
        # Patch the DuckDB path for testing
        self.original_db_path = common.DB_PATH
        common.DB_PATH = self.test_db_path

    def tearDown(self):
        """Clean up test environment."""
        # Restore original DB path
        common.DB_PATH = self.original_db_path
        
        # Remove test database
        try:
            os.unlink(self.test_db_path)
        except:
            pass

    def create_test_database(self):
        """Create a minimal test database with sample data."""
        con = duckdb.connect(self.test_db_path)
        
        # Create sample building data for 3 days
        dates = pd.date_range('2015-01-01', periods=3, freq='D')
        all_data = []
        
        for date in dates:
            daily_data = []
            for hour in range(24):
                timestamp = date + pd.Timedelta(hours=hour)
                daily_data.append({
                    'hour': hour,
                    'day': date.date(),
                    'utc_timestamp': timestamp,
                    'price_per_kwh': 0.25 + 0.1 * np.sin(hour * np.pi / 12),
                    f'{self.building_id}_dishwasher': 0.5 if 18 <= hour <= 20 else 0.1,
                    f'{self.building_id}_heat_pump': 2.0 if 6 <= hour <= 9 or 18 <= hour <= 22 else 0.5,
                    f'{self.building_id}_ev': 7.4 if 22 <= hour <= 24 or 0 <= hour <= 6 else 0.0,
                    f'{self.building_id}_pv_roof': -1.5 if 8 <= hour <= 16 else 0.0,
                    f'{self.building_id}_grid_import': 2.0,
                    'DE_temperature': 20.0,
                    'DE_radiation_direct_horizontal': 100.0 if 8 <= hour <= 16 else 0.0,
                    'DE_radiation_diffuse_horizontal': 50.0 if 8 <= hour <= 16 else 0.0,
                    'total_consumption': 3.0,
                    'flexibility_category': ['Partially Flexible', 'Highly Flexible'],
                    'power_rating': [1.0, 3.0],
                    'net_energy_usage': 2.5,
                    'cost_without_generation': 0.6,
                    'cost_with_generation': 0.4,
                    'pv_forecast': -1.2 if 8 <= hour <= 16 else 0.0,
                    'year': 2015.0
                })
            all_data.extend(daily_data)
        
        # Create DataFrame and insert into DuckDB
        df = pd.DataFrame(all_data)
        con.execute(f"CREATE TABLE {self.building_id}_processed_data AS SELECT * FROM df")
        
        # Create device hourly probabilities table
        device_probs = []
        for device_type in ['dishwasher', 'heat_pump', 'ev']:
            prob_row = {str(h): 1/24 for h in range(24)}  # Uniform probabilities
            prob_row['device_type'] = device_type
            device_probs.append(prob_row)
        
        probs_df = pd.DataFrame(device_probs)
        con.execute("CREATE TABLE device_hourly_probabilities AS SELECT * FROM probs_df")
        
        con.close()

    def test_duckdb_connection_works(self):
        """Test that DuckDB connection and data loading works."""
        con = common.get_con()
        
        # Test that we can query the test data
        result = con.execute(f"SELECT COUNT(*) FROM {self.building_id}_processed_data").fetchone()
        self.assertEqual(result[0], 72)  # 3 days * 24 hours
        
        # Test device hourly probabilities
        probs = con.execute("SELECT COUNT(*) FROM device_hourly_probabilities").fetchone()
        self.assertEqual(probs[0], 3)  # 3 device types

    def test_helper_functions_run_without_error(self):
        """Test that helper functions can be imported and run."""
        from utils.helper import (
            compute_device_savings,
            validate_dataframe_for_agents,
            get_jads_color_palette
        )
        
        # These should import without error
        self.assertTrue(callable(compute_device_savings))
        self.assertTrue(callable(validate_dataframe_for_agents))
        self.assertTrue(callable(get_jads_color_palette))

    def test_agent_classes_can_be_imported(self):
        """Test that all required Agent classes can be imported."""
        from agents.FlexibleDeviceAgent import FlexibleDevice
        from agents.GlobalOptimizer import GlobalOptimizer
        from agents.BatteryAgent import BatteryAgent
        from agents.EVAgent import EVAgent
        from agents.ProbabilityModelAgent import ProbabilityModelAgent
        
        # All should import without error
        self.assertTrue(callable(FlexibleDevice))
        self.assertTrue(callable(GlobalOptimizer))
        self.assertTrue(callable(BatteryAgent))
        self.assertTrue(callable(EVAgent))
        self.assertTrue(callable(ProbabilityModelAgent))

    def test_probability_model_agent_initialization(self):
        """Test that ProbabilityModelAgent can be initialized with DuckDB data."""
        from agents.ProbabilityModelAgent import ProbabilityModelAgent
        
        con = common.get_con()
        priors_df = con.execute("SELECT * FROM device_hourly_probabilities").df()
        priors_df = priors_df.set_index('device_type')
        
        # Should initialize without error
        prob_agent = ProbabilityModelAgent(prob_dist_df=priors_df)
        self.assertIsNotNone(prob_agent)
        self.assertIsInstance(prob_agent.latest_distributions, dict)

    def test_battery_agent_initialization(self):
        """Test that BatteryAgent can be initialized."""
        from agents.BatteryAgent import BatteryAgent
        
        battery_params = {
            'max_charge_rate': 3.0,
            'max_discharge_rate': 3.0,
            'initial_soc': 5.0,
            'soc_min': 1.0,
            'soc_max': 10.0,
            'capacity': 10.0,
            'efficiency_charge': 0.95,
            'efficiency_discharge': 0.95
        }
        
        # Should initialize without error
        battery_agent = BatteryAgent(**battery_params)
        self.assertIsNotNone(battery_agent)
        self.assertEqual(battery_agent.max_charge_rate, 3.0)

    def test_cost_comparison_makes_sense(self):
        """Test that optimized costs are less than or equal to original costs."""
        from utils.helper import compute_device_savings
        
        # Create simple class instead of Mock to avoid attribute issues
        class SimpleDevice:
            def __init__(self):
                self.original_consumption = np.array([1.0] * 24)  # Original consumption
                self.optimized_consumption = np.array([0.8] * 24)  # 20% reduction
                
                # Create realistic price data
                prices = [0.25 + 0.1 * np.sin(h * np.pi / 12) for h in range(24)]
                self.data = pd.DataFrame({
                    'price_per_kwh': prices
                })
        
        device = SimpleDevice()
        
        # Calculate savings
        pct_savings, euro_savings, adjusted_cost = compute_device_savings(device)
        
        # Optimized should cost less than original
        self.assertGreaterEqual(pct_savings, 0)  # Should have non-negative savings
        self.assertGreaterEqual(euro_savings, 0)  # Should have positive euro savings

    def test_plotting_functions_create_directories(self):
        """Test that plotting functions create output directories."""
        from utils.helper import plot_battery_schedule
        from unittest.mock import patch
        
        with patch('matplotlib.pyplot.savefig'), patch('matplotlib.pyplot.close'):
            # Should create directory and not raise error
            plot_battery_schedule([1.0] * 24, self.building_id, "2015-01-01")
            
            # Check that results/plots directory exists
            self.assertTrue(Path("results/plots").exists())


class TestLintChecks(unittest.TestCase):
    """Test for forbidden patterns that indicate manual optimization."""
    
    def test_no_forbidden_optimization_patterns(self):
        """Test that scripts don't contain forbidden manual optimization patterns."""
        forbidden_patterns = [
            'run_simple_centralized_optimization',
            'optimize_device_with_agent_logic',
            'greedy',
            'manual.*optimization',
            'fallback.*loop'
        ]
        
        target_files = [
            'notebooks/multi_day_optimisation_with_plots.py',
            'notebooks/next_day_scheduling_with_learned_preferences_training_and_updating_and_plots.py'
        ]
        
        for file_path in target_files:
            if Path(file_path).exists():
                with open(file_path, 'r') as f:
                    content = f.read().lower()
                
                for pattern in forbidden_patterns:
                    self.assertNotIn(pattern.lower(), content, 
                                   f"Found forbidden pattern '{pattern}' in {file_path}")

    def test_no_direct_parquet_access(self):
        """Test that scripts don't directly access parquet files."""
        forbidden_patterns = [
            'pd.read_parquet',
            'read_parquet',
            '.parquet'
        ]
        
        target_files = [
            'notebooks/multi_day_optimisation_with_plots.py',
            'notebooks/next_day_scheduling_with_learned_preferences_training_and_updating_and_plots.py'
        ]
        
        for file_path in target_files:
            if Path(file_path).exists():
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # Check for direct parquet access (allow comments and variable names)
                lines = content.split('\n')
                for line_num, line in enumerate(lines):
                    if '#' in line:
                        line = line[:line.index('#')]  # Remove comments
                    
                    for pattern in forbidden_patterns:
                        if pattern in line and 'parquet_path' not in line and 'parquet_dir' not in line:
                            # Allow variable names but not actual calls
                            if 'pd.read_parquet(' in line or '.parquet"' in line or ".parquet'" in line:
                                self.fail(f"Found direct parquet access '{pattern}' in {file_path}:{line_num+1}")

    def test_uses_duckdb_connection(self):
        """Test that scripts use DuckDB connection."""
        required_patterns = [
            'common.get_con()',
            'con.execute('
        ]
        
        target_files = [
            'notebooks/multi_day_optimisation_with_plots.py',
            'notebooks/next_day_scheduling_with_learned_preferences_training_and_updating_and_plots.py'
        ]
        
        for file_path in target_files:
            if Path(file_path).exists():
                with open(file_path, 'r') as f:
                    content = f.read()
                
                for pattern in required_patterns:
                    self.assertIn(pattern, content, 
                                f"Missing required DuckDB pattern '{pattern}' in {file_path}")


if __name__ == '__main__':
    # Create results directory if it doesn't exist
    os.makedirs("results/plots", exist_ok=True)
    
    # Run tests
    unittest.main(verbosity=2)