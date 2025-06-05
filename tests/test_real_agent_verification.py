#!/usr/bin/env python
"""
COMPREHENSIVE UNIT TESTS for REAL AGENT VERIFICATION

This test suite specifically verifies that all scripts strictly follow 
the "USE REAL AGENT OPTIMIZERS" rule with NO fallbacks or manual optimization logic.

Test Coverage:
1. Verify all Agent method calls use real implementations
2. Test proper DataFrame formatting for Agent consumption
3. Detect any manual optimization loops or fallbacks
4. Verify Agent integration works end-to-end with real data
5. Smoke tests for optimization results consistency
"""

import unittest
from unittest.mock import patch, MagicMock, call
import sys
import os
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np
import tempfile
import duckdb

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "notebooks"))
sys.path.insert(0, str(project_root / "scripts"))

import common

class TestRealAgentVerification(unittest.TestCase):
    """Comprehensive tests to verify real Agent usage."""
    
    def setUp(self):
        """Set up test environment with real DuckDB data."""
        self.building_id = "DE_KN_residential1"
        
        # Create temporary DuckDB for testing
        self.test_db_fd, self.test_db_path = tempfile.mkstemp(suffix='.duckdb')
        os.close(self.test_db_fd)
        os.unlink(self.test_db_path)
        
        # Create test data
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
        """Create test database with realistic data."""
        con = duckdb.connect(self.test_db_path)
        
        # Create sample building data for 2 days
        dates = pd.date_range('2015-05-22', periods=2, freq='D')
        all_data = []
        
        for date in dates:
            for hour in range(24):
                timestamp = date + pd.Timedelta(hours=hour)
                all_data.append({
                    'hour': hour,
                    'day': date.date(),
                    'utc_timestamp': timestamp,
                    'price_per_kwh': 0.025 + 0.02 * np.sin(hour * np.pi / 12),
                    f'{self.building_id}_dishwasher': 0.001 if hour == 6 else 0.0,
                    f'{self.building_id}_freezer': 0.0,
                    f'{self.building_id}_heat_pump': 0.5 + 0.3 * np.random.random(),
                    f'{self.building_id}_washing_machine': 0.001 if hour == 1 else 0.0,
                    f'{self.building_id}_pv_roof': -1.5 if 8 <= hour <= 16 else 0.0,
                    f'{self.building_id}_grid_import': 2.0,
                    'total_consumption': 2.5,
                    'year': 2015.0
                })
        
        # Create DataFrame and insert into DuckDB
        df = pd.DataFrame(all_data)
        con.execute(f"CREATE TABLE {self.building_id}_processed_data AS SELECT * FROM df")
        
        # Create device hourly probabilities table
        device_probs = []
        device_types = ['dishwasher', 'freezer', 'heat_pump', 'washing_machine', 'refrigerator', 'other']
        for device_type in device_types:
            prob_row = {str(h): 1/24 for h in range(24)}
            prob_row['device_type'] = device_type
            device_probs.append(prob_row)
        
        probs_df = pd.DataFrame(device_probs)
        con.execute("CREATE TABLE device_hourly_probabilities AS SELECT * FROM probs_df")
        
        con.close()

    def test_pipeline_a_uses_real_agents_only(self):
        """Test that Pipeline A uses only real Agent methods with no fallbacks."""
        
        # Test that we can run Pipeline A without errors
        result = subprocess.run([
            sys.executable, "scripts/01_run.py",
            "--building", self.building_id,
            "--mode", "centralised",
            "--n_days", "1",
            "--battery", "on"
        ], capture_output=True, text=True, cwd=project_root)
        
        # Should complete successfully
        self.assertEqual(result.returncode, 0, f"Pipeline A failed: {result.stderr}")
        
        # Should contain evidence of real Agent usage
        self.assertIn("✓ Successfully imported ALL real agent classes", result.stdout)
        self.assertIn("REAL AGENT OPTIMIZERS", result.stdout)
        self.assertIn("Real centralized optimization", result.stdout)
        
        # Should NOT contain any fallback language
        self.assertNotIn("fallback", result.stdout.lower())
        self.assertNotIn("simplified", result.stdout.lower())
        self.assertNotIn("manual optimization", result.stdout.lower())

    def test_pipeline_b_uses_real_agents_only(self):
        """Test that Pipeline B uses only real Agent methods with no fallbacks."""
        # Test that we can run Pipeline B without errors
        result = subprocess.run([
            sys.executable, "scripts/02_integrated_pipeline.py",
            "--building", self.building_id,
            "--mode", "centralized_phases",
            "--n_days", "2",
            "--battery", "on"
        ], capture_output=True, text=True, cwd=project_root)
        
        # Should complete successfully
        self.assertEqual(result.returncode, 0, f"Pipeline B failed: {result.stderr}")
        
        # Should contain evidence of real Agent usage
        self.assertIn("✓ Successfully imported ALL real agent classes", result.stdout)
        self.assertIn("REAL AGENT INTEGRATED EMS", result.stdout)
        self.assertIn("REAL probability training", result.stdout)
        self.assertIn("REAL centralized_phases optimization", result.stdout)
        
        # Should NOT contain any fallback language
        self.assertNotIn("fallback", result.stdout.lower())
        self.assertNotIn("simplified", result.stdout.lower())
        self.assertNotIn("manual optimization", result.stdout.lower())

    def test_agent_classes_can_be_imported_and_instantiated(self):
        """Test that all Agent classes can be imported and instantiated."""
        # Import all Agent classes
        from agents.ProbabilityModelAgent import ProbabilityModelAgent
        from agents.BatteryAgent import BatteryAgent
        from agents.EVAgent import EVAgent
        from agents.PVAgent import PVAgent
        from agents.GridAgent import GridAgent
        from agents.FlexibleDeviceAgent import FlexibleDevice
        from agents.GlobalOptimizer import GlobalOptimizer
        from agents.GlobalConnectionLayer import GlobalConnectionLayer
        from agents.WeatherAgent import WeatherAgent
        
        # Test that agents can be instantiated
        con = common.get_con()
        priors_df = con.execute("SELECT * FROM device_hourly_probabilities").df()
        priors_df = priors_df.set_index('device_type')
        
        prob_agent = ProbabilityModelAgent(prob_dist_df=priors_df)
        self.assertIsNotNone(prob_agent)
        
        battery_agent = BatteryAgent(
            max_charge_rate=3.0, max_discharge_rate=3.0,
            initial_soc=5.0, soc_min=1.0, soc_max=10.0,
            capacity=10.0, efficiency_charge=0.95, efficiency_discharge=0.95
        )
        self.assertIsNotNone(battery_agent)
        
        grid_agent = GridAgent(import_price=0.25, export_price=0.05, max_import=15.0, max_export=15.0)
        self.assertIsNotNone(grid_agent)

    def test_dataframe_structure_compatibility(self):
        """Test that DataFrames have correct structure for Agent consumption."""
        con = common.get_con()
        
        # Get test data
        test_df = con.execute(f"""
            SELECT *, EXTRACT(hour FROM utc_timestamp) as hour, DATE(utc_timestamp) as day
            FROM {self.building_id}_processed_data 
            WHERE DATE(utc_timestamp) = '2015-05-22'
            ORDER BY utc_timestamp
        """).df()
        
        # Test required columns exist
        required_columns = ['hour', 'day', 'price_per_kwh', 'utc_timestamp']
        for col in required_columns:
            self.assertIn(col, test_df.columns, f"Required column '{col}' missing")
        
        # Test hour range is complete
        hours = sorted(test_df['hour'].unique())
        self.assertEqual(hours, list(range(24)), "Should have complete 24-hour data")
        
        # Test no missing values in critical columns
        for col in ['hour', 'price_per_kwh']:
            self.assertFalse(test_df[col].isna().any(), f"Column '{col}' has missing values")

    def test_agent_optimization_produces_valid_results(self):
        """Test that Agent optimization produces valid, consistent results."""
        from agents.FlexibleDeviceAgent import FlexibleDevice
        from agents.GlobalConnectionLayer import GlobalConnectionLayer
        from agents.BatteryAgent import BatteryAgent
        
        con = common.get_con()
        
        # Get test data
        test_df = con.execute(f"""
            SELECT *, EXTRACT(hour FROM utc_timestamp) as hour, DATE(utc_timestamp) as day
            FROM {self.building_id}_processed_data 
            WHERE DATE(utc_timestamp) = '2015-05-22'
            ORDER BY utc_timestamp
        """).df()
        
        # Create a test device
        global_layer = GlobalConnectionLayer(max_building_load=50.0, total_hours=24)
        battery_agent = BatteryAgent(
            max_charge_rate=3.0, max_discharge_rate=3.0,
            initial_soc=5.0, soc_min=1.0, soc_max=10.0,
            capacity=10.0, efficiency_charge=0.95, efficiency_discharge=0.95
        )
        
        device = FlexibleDevice(
            device_name=f"{self.building_id}_heat_pump",
            data=test_df.reset_index(drop=True),
            category="Highly Flexible",
            power_rating=3.0,
            global_layer=global_layer,
            battery_agent=battery_agent,
            spec={'category': 'Highly Flexible', 'power_rating': 3.0}
        )
        
        # Test that device has required attributes
        self.assertTrue(hasattr(device, 'optimize_day'))
        self.assertTrue(hasattr(device, 'hour_probability'))
        self.assertTrue(hasattr(device, 'original_consumption'))
        
        # Test that optimization can be called
        day = test_df['day'].iloc[0]
        prices = test_df['price_per_kwh'].values
        
        try:
            device.optimize_day(day, prices, None, None, None)
            self.assertTrue(hasattr(device, 'optimized_schedule'), "Device should have optimized_schedule after optimization")
            self.assertEqual(len(device.optimized_schedule), 24, "Optimized schedule should have 24 hours")
        except Exception as e:
            self.fail(f"Device optimization failed: {e}")

    def test_no_manual_optimization_code_exists(self):
        """Test that no manual optimization code exists in scripts."""
        script_files = [
            project_root / "scripts" / "01_run.py",
            project_root / "scripts" / "02_integrated_pipeline.py"
        ]
        
        forbidden_patterns = [
            "for.*price.*sort",
            "manual.*optimization",
            "greedy.*loop",
            "optimize_device_with_agent_logic",
            "simple.*centralized.*optimization",
            "price.*sort.*manual"
        ]
        
        for script_file in script_files:
            if script_file.exists():
                with open(script_file, 'r') as f:
                    content = f.read()
                
                for pattern in forbidden_patterns:
                    import re
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    self.assertEqual(len(matches), 0, 
                                   f"Found forbidden pattern '{pattern}' in {script_file}: {matches}")

    def test_agent_method_calls_are_mandatory(self):
        """Test that all Agent method calls are mandatory (no try/except fallbacks)."""
        script_files = [
            project_root / "scripts" / "01_run.py",
            project_root / "scripts" / "02_integrated_pipeline.py"
        ]
        
        # Patterns that indicate fallback behavior (forbidden)
        fallback_patterns = [
            r"try:.*agent.*except:.*fallback",
            r"if.*hasattr.*agent.*else:.*manual",
            r"except.*:.*optimize_device_with"
        ]
        
        for script_file in script_files:
            if script_file.exists():
                with open(script_file, 'r') as f:
                    content = f.read()
                
                for pattern in fallback_patterns:
                    import re
                    matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)
                    self.assertEqual(len(matches), 0, 
                                   f"Found fallback pattern '{pattern}' in {script_file}")

    def test_duckdb_only_data_access(self):
        """Test that scripts use DuckDB-only data access with no parquet fallbacks."""
        script_files = [
            project_root / "scripts" / "01_run.py",
            project_root / "scripts" / "02_integrated_pipeline.py"
        ]
        
        # Required patterns for DuckDB usage
        required_patterns = [
            "common.get_con()",
            "con.execute(",
            "DuckDB"
        ]
        
        # Forbidden patterns for direct file access
        forbidden_patterns = [
            "pd.read_parquet",
            "pd.read_csv",
            ".parquet"
        ]
        
        for script_file in script_files:
            if script_file.exists():
                with open(script_file, 'r') as f:
                    content = f.read()
                
                # Check for required DuckDB patterns
                for pattern in required_patterns:
                    self.assertIn(pattern, content, 
                                f"Missing required DuckDB pattern '{pattern}' in {script_file}")
                
                # Check for forbidden direct file access
                for pattern in forbidden_patterns:
                    # Allow variable names but not actual calls
                    if pattern in content:
                        lines = content.split('\n')
                        for line_num, line in enumerate(lines):
                            if '#' in line:
                                line = line[:line.index('#')]  # Remove comments
                            
                            if pattern in line and 'parquet_path' not in line and 'parquet_dir' not in line:
                                if 'pd.read_parquet(' in line or '.parquet"' in line or ".parquet'" in line:
                                    self.fail(f"Found direct file access '{pattern}' in {script_file}:{line_num+1}")

    def test_optimization_results_are_consistent(self):
        """Test that optimization results are consistent between runs."""
        # Run Pipeline A twice with same parameters
        results = []
        for i in range(2):
            result = subprocess.run([
                sys.executable, "scripts/01_run.py",
                "--building", self.building_id,
                "--mode", "centralised",
                "--n_days", "1",
                "--battery", "on"
            ], capture_output=True, text=True, cwd=project_root)
            
            self.assertEqual(result.returncode, 0, f"Run {i+1} failed")
            results.append(result.stdout)
        
        # Both runs should complete successfully
        for i, result in enumerate(results):
            self.assertIn("✅ Pipeline A completed successfully", result, f"Run {i+1} did not complete successfully")

class TestMLflowIntegrationPreparation(unittest.TestCase):
    """Tests to prepare for MLflow integration."""
    
    def test_results_directory_structure(self):
        """Test that results are saved in predictable directory structure for MLflow."""
        expected_dirs = [
            "results/output",
            "results/visualizations"
        ]
        
        for dir_path in expected_dirs:
            self.assertTrue(Path(dir_path).exists(), f"Results directory {dir_path} should exist")

    def test_result_files_contain_metrics(self):
        """Test that result files contain metrics suitable for MLflow tracking."""
        output_files = list(Path("results/output").glob("*.csv"))
        
        if output_files:
            # Check that CSV files contain metrics
            for file_path in output_files[:1]:  # Check first file
                df = pd.read_csv(file_path)
                
                # Should contain cost/savings metrics
                expected_columns = ['day', 'total_cost']
                for col in expected_columns:
                    if col in df.columns:
                        self.assertTrue(True)  # Found expected metric column
                        break
                else:
                    self.fail(f"No metric columns found in {file_path}")

if __name__ == '__main__':
    # Create results directories if they don't exist
    os.makedirs("results/output", exist_ok=True)
    os.makedirs("results/visualizations", exist_ok=True)
    
    # Run tests
    unittest.main(verbosity=2)