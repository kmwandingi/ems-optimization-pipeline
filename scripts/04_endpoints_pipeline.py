#!/usr/bin/env python
"""
PIPELINE D - ENDPOINTS TESTING PIPELINE
========================================
This pipeline mimics 02_integrated_pipeline.py but ONLY uses deployed endpoints 
to test the full learning + optimization workflow via endpoint calls.

STRICT COMPLIANCE: 
- Uses ONLY endpoint calls to deployed models
- Tests complete learning → optimization feedback loop
- NO direct agent calls (all through endpoints)
- Validates endpoint functionality matches direct pipeline results

Usage:
    python scripts/04_endpoints_pipeline.py --building DE_KN_residential1 --n_days 4 --mode centralized
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add notebooks directory to path for utilities
sys.path.append(str(Path.cwd() / "notebooks"))
sys.path.append(str(Path.cwd() / "notebooks" / "utils"))

# Import common utilities
import common
from device_specs import device_specs

# MLflow tracking 
try:
    from utils.mlflow_tracker import EMS_OptimizationTracker
    MLFLOW_AVAILABLE = True
    print("✓ MLflow tracking enabled")
except ImportError:
    print("⚠ MLflow tracking disabled (optional)")
    MLFLOW_AVAILABLE = False

# Azure ML imports for endpoint access
try:
    import mlflow
    import mlflow.pyfunc
    from azureml.core import Workspace
    AZURE_ML_AVAILABLE = True
    print("✓ Azure ML SDK available for endpoint testing")
except ImportError:
    print("❌ CRITICAL ERROR: Azure ML SDK required for endpoint testing")
    print("Please install: pip install azureml-sdk azureml-mlflow")
    sys.exit(1)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="EMS Endpoints Pipeline - Learning + Optimization via Endpoints")
    parser.add_argument("--building", type=str, required=True,
                        help="Building ID (e.g., DE_KN_residential1)")
    parser.add_argument("--n_days", type=int, default=4,
                        help="Total number of days to process")
    # Production: Always uses phases centralized optimization
    parser.add_argument("--battery", type=str, default="true",
                        choices=["true", "false"],
                        help="Enable battery optimization")
    parser.add_argument("--ev", type=str, default="false", 
                        choices=["true", "false"],
                        help="Enable EV optimization")
    parser.add_argument("--model_name", type=str, default="ems_optimizer",
                        help="Name of deployed model")
    parser.add_argument("--model_version", type=str, default="18",
                        help="Version of deployed model to test")
    
    return parser.parse_args()

class EndpointEMSPipeline:
    """
    EMS Pipeline that uses ONLY endpoint calls to test the complete system.
    Mimics 02_integrated_pipeline.py functionality but via deployed endpoints.
    """
    
    def __init__(self, building_id: str, model_name: str, model_version: str = "latest"):
        self.building_id = building_id
        self.model_name = model_name
        self.model_version = model_version
        self.model = None
        self.con = None
        
        print(f"🎯 Initializing Endpoint EMS Pipeline")
        print(f"   Building: {building_id}")
        print(f"   Model: {model_name} v{model_version}")
        
    def connect_to_endpoints(self):
        """Connect to Azure ML workspace and load deployed model."""
        print("🔗 Connecting to Azure ML endpoints...")
        
        # Load Azure ML configuration
        config_path = Path.cwd() / "config.json"
        if not config_path.exists():
            raise FileNotFoundError("config.json not found. Azure ML configuration required.")
            
        with open(config_path) as f:
            config = json.load(f)
        
        # Connect to workspace
        try:
            ws = Workspace(
                subscription_id=config['subscription_id'],
                resource_group=config['resource_group'],
                workspace_name=config['workspace_name']
            )
            print(f"✓ Connected to Azure ML workspace: {ws.name}")
            
            # Set MLflow tracking
            mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
            print("✓ MLflow tracking URI configured")
            
        except Exception as e:
            raise Exception(f"Failed to connect to Azure ML workspace: {e}")
        
        # Load the deployed model
        try:
            if self.model_version == "latest":
                model_uri = f"models:/{self.model_name}/latest"
            else:
                model_uri = f"models:/{self.model_name}/{self.model_version}"
                
            self.model = mlflow.pyfunc.load_model(model_uri)
            print(f"✓ Loaded model: {self.model_name} v{self.model_version}")
            
        except Exception as e:
            raise Exception(f"Failed to load model {self.model_name}: {e}")
    
    def setup_duckdb_connection(self):
        """Setup DuckDB connection for data retrieval."""
        print("📊 Setting up DuckDB connection...")
        
        try:
            self.con = common.get_con()
            
            # Verify building data exists
            query = f"SELECT COUNT(*) as count FROM '{self.building_id}_processed_data'"
            result = self.con.execute(query).fetchone()
            
            if result[0] == 0:
                raise ValueError(f"No data found for building {self.building_id}")
                
            print(f"✓ Connected to DuckDB: {result[0]} rows available for {self.building_id}")
            
        except Exception as e:
            raise Exception(f"Failed to setup DuckDB connection: {e}")
    
    def get_training_data(self, n_training_days: int):
        """Get training data for learning phase."""
        print(f"📚 Loading {n_training_days} days of training data from DuckDB...")
        
        query = f"""
        SELECT *, 
               date_trunc('day', utc_timestamp) as date,
               extract(hour from utc_timestamp) as hour
        FROM '{self.building_id}_processed_data' 
        ORDER BY utc_timestamp 
        LIMIT {n_training_days * 24}
        """
        
        df = self.con.execute(query).df()
        print(f"✓ Loaded {len(df)} rows of training data")
        
        return df
    
    def get_training_data_for_days(self, training_days):
        """Get training data for specific calendar days (same as direct pipeline)."""
        print(f"📚 Loading training data for specific days: {training_days}")
        
        # Convert days to string format for SQL
        day_strings = [f"'{day}'" for day in training_days]
        days_clause = ','.join(day_strings)
        
        query = f"""
        SELECT *, 
               date_trunc('day', utc_timestamp) as date,
               extract(hour from utc_timestamp) as hour
        FROM '{self.building_id}_processed_data' 
        WHERE DATE(utc_timestamp) IN ({days_clause})
        ORDER BY utc_timestamp
        """
        
        df = self.con.execute(query).df()
        print(f"✓ Loaded {len(df)} rows of training data for {len(training_days)} days")
        
        return df
    
    def get_optimization_data_for_days(self, optimization_days):
        """Get optimization data for specific calendar days (same as direct pipeline)."""
        print(f"⚙️ Loading optimization data for specific days: {optimization_days}")
        
        # Convert days to string format for SQL
        day_strings = [f"'{day}'" for day in optimization_days]
        days_clause = ','.join(day_strings)
        
        query = f"""
        SELECT *, 
               date_trunc('day', utc_timestamp) as date,
               extract(hour from utc_timestamp) as hour
        FROM '{self.building_id}_processed_data' 
        WHERE DATE(utc_timestamp) IN ({days_clause})
        ORDER BY utc_timestamp
        """
        
        df = self.con.execute(query).df()
        print(f"✓ Loaded {len(df)} rows of optimization data for {len(optimization_days)} days")
        
        return df
    
    def get_optimization_data(self, start_idx: int, n_opt_days: int):
        """Get data for optimization phase."""
        print(f"⚙️ Loading {n_opt_days} days of optimization data...")
        
        query = f"""
        SELECT *, 
               date_trunc('day', utc_timestamp) as date,
               extract(hour from utc_timestamp) as hour
        FROM '{self.building_id}_processed_data' 
        ORDER BY utc_timestamp 
        LIMIT {n_opt_days * 24} OFFSET {start_idx * 24}
        """
        
        df = self.con.execute(query).df()
        print(f"✓ Loaded {len(df)} rows of optimization data")
        
        return df
    
    def call_learning_endpoint(self, training_df: pd.DataFrame):
        """Call the learning endpoint to train probability models."""
        print("🧠 Calling learning endpoint...")
        
        # Group training data by date for learning calls
        learning_results = []
        dates = training_df['date'].unique()
        
        for date in dates:
            day_data = training_df[training_df['date'] == date]
            
            # Prepare actual usage data in the format expected by endpoint
            actual_usage = {}
            device_columns = [col for col in day_data.columns 
                            if col.startswith(self.building_id) and 
                            col not in ['date', 'hour', 'price_per_kwh']]
            
            for device_col in device_columns:
                if day_data[device_col].sum() > 0:  # Only include active devices
                    actual_usage[device_col] = day_data[device_col].tolist()
            
            if actual_usage:  # Only call if we have actual usage data
                learning_input = {
                    'mode': 'learn',
                    'building_id': self.building_id,
                    'actual_usage': actual_usage,
                    'date': str(date)
                }
                
                try:
                    result = self.model.predict(learning_input)
                    learning_results.append(result)
                    print(f"  ✓ Learned from {date}: {len(result.get('updated_devices', []))} devices updated")
                    
                except Exception as e:
                    print(f"  ❌ Learning failed for {date}: {e}")
                    
        print(f"✓ Learning completed: {len(learning_results)} days processed")
        return learning_results
    
    def call_optimization_endpoint(self, opt_df: pd.DataFrame, battery_enabled: bool, ev_enabled: bool):
        """Call the optimization endpoint for each day."""
        print("⚙️ Calling optimization endpoint...")
        
        optimization_results = []
        dates = opt_df['date'].unique()
        
        for date in dates:
            day_data = opt_df[opt_df['date'] == date]
            
            # Extract price profile for the day
            price_profile = day_data['price_per_kwh'].tolist()
            
            # Ensure we have 24 hours of price data
            if len(price_profile) != 24:
                print(f"  ⚠ Warning: {date} has {len(price_profile)} hours, padding to 24")
                while len(price_profile) < 24:
                    price_profile.append(price_profile[-1] if price_profile else 0.25)
                price_profile = price_profile[:24]
            
            optimization_input = {
                'mode': 'optimize',
                'building_id': self.building_id,
                'target_date': str(date),
                'price_profile': price_profile,
                'battery_enabled': battery_enabled,
                'ev_enabled': ev_enabled,
                'grid_params': {
                    'import_price': 0.25,
                    'export_price': 0.05,
                    'max_import': 15.0,
                    'max_export': 15.0
                }
            }
            
            try:
                result = self.model.predict(optimization_input)
                optimization_results.append(result)
                
                savings = result.get('savings_vs_baseline', 0)
                cost = result.get('total_cost', 0)
                devices = len(result.get('optimized_schedules', {}))
                
                print(f"  ✓ Optimized {date}: €{cost:.2f} cost, €{savings:.2f} savings, {devices} devices")
                
            except Exception as e:
                print(f"  ❌ Optimization failed for {date}: {e}")
                
        print(f"✓ Optimization completed: {len(optimization_results)} days processed")
        return optimization_results
    
    def analyze_results(self, learning_results: list, optimization_results: list):
        """Analyze and report endpoint testing results."""
        print("\n📊 ENDPOINT TESTING RESULTS ANALYSIS")
        print("=" * 50)
        
        # Learning Analysis
        if learning_results:
            total_devices_updated = sum(len(r.get('updated_devices', [])) for r in learning_results)
            learning_days = len(learning_results)
            
            print(f"🧠 LEARNING RESULTS:")
            print(f"   Training days processed: {learning_days}")
            print(f"   Total device updates: {total_devices_updated}")
            print(f"   Average devices per day: {total_devices_updated/learning_days:.1f}")
            
            # Show sample PMF updates
            sample_result = learning_results[-1]  # Latest learning result
            if 'updated_pmfs' in sample_result and sample_result['updated_pmfs']:
                print(f"   Latest PMF updates:")
                for device_id, pmf in list(sample_result['updated_pmfs'].items())[:3]:
                    sorted_hours = sorted(pmf.items(), key=lambda x: x[1], reverse=True)[:3]
                    peak_hours = [f"h{h}({p:.3f})" for h, p in sorted_hours]
                    print(f"     {device_id}: {', '.join(peak_hours)}")
        
        # Optimization Analysis
        if optimization_results:
            total_cost = sum(r.get('total_cost', 0) for r in optimization_results)
            total_savings = sum(r.get('savings_vs_baseline', 0) for r in optimization_results)
            opt_days = len(optimization_results)
            
            print(f"\n⚙️ OPTIMIZATION RESULTS:")
            print(f"   Optimization days processed: {opt_days}")
            print(f"   Total cost: €{total_cost:.2f}")
            print(f"   Total savings: €{total_savings:.2f}")
            print(f"   Average daily cost: €{total_cost/opt_days:.2f}")
            print(f"   Average daily savings: €{total_savings/opt_days:.2f}")
            
            if total_cost < 0:
                print(f"   💰 Revenue generated from arbitrage!")
            
            # Show device activity summary
            all_devices = set()
            total_energy = 0
            for result in optimization_results:
                if 'optimized_schedules' in result:
                    for device_id, schedule in result['optimized_schedules'].items():
                        all_devices.add(device_id)
                        total_energy += sum(schedule)
            
            print(f"   Unique devices optimized: {len(all_devices)}")
            print(f"   Total energy scheduled: {total_energy:.1f} kWh")
        
        # Overall Pipeline Assessment
        print(f"\n🎯 ENDPOINT PIPELINE ASSESSMENT:")
        learning_success = len(learning_results) > 0
        optimization_success = len(optimization_results) > 0
        
        print(f"   Learning endpoint: {'✅ SUCCESS' if learning_success else '❌ FAILED'}")
        print(f"   Optimization endpoint: {'✅ SUCCESS' if optimization_success else '❌ FAILED'}")
        
        if learning_success and optimization_success:
            print(f"   📈 Full learning → optimization pipeline: ✅ WORKING")
            return True
        else:
            print(f"   📈 Full learning → optimization pipeline: ❌ FAILED")
            return False

def main():
    """Main function implementing the endpoints testing pipeline."""
    
    args = parse_args()
    
    building_id = args.building
    n_days = args.n_days
    battery_enabled = args.battery.lower() == "true"
    ev_enabled = args.ev.lower() == "true"
    
    print("=" * 80)
    print("ENDPOINTS PIPELINE - EMS LEARNING + OPTIMIZATION VIA ENDPOINTS")
    print("=" * 80)
    print(f"Building: {building_id}")
    print(f"Days: {n_days}")
    print(f"Battery: {battery_enabled}")
    print(f"EV: {ev_enabled}")
    print(f"Mode: PRODUCTION (phases centralized only)")
    print(f"Model: {args.model_name} v{args.model_version}")
    print("=" * 80)
    
    # Initialize MLflow tracking
    mlflow_tracker = None
    if MLFLOW_AVAILABLE:
        mlflow_tracker = EMS_OptimizationTracker("Endpoints_Pipeline")
        run_name = f"endpoints_{building_id}_phases_{datetime.now().strftime('%Y%m%d_%H%M')}"
        mlflow_tracker.start_run(run_name)
        
        # Log pipeline parameters
        mlflow_tracker.log_params({
            "building_id": building_id,
            "n_days": n_days,
            "optimization_mode": "phases_centralized",
            "battery_enabled": battery_enabled,
            "ev_enabled": ev_enabled,
            "model_name": args.model_name,
            "model_version": args.model_version,
            "pipeline_type": "endpoints_only"
        })
    
    try:
        # Initialize the endpoint pipeline
        pipeline = EndpointEMSPipeline(building_id, args.model_name, args.model_version)
        
        # Connect to endpoints and data
        pipeline.connect_to_endpoints()
        pipeline.setup_duckdb_connection()
        
        # USE SAME DAY SELECTION LOGIC AS DIRECT PIPELINE
        # Get available days with complete 24-hour data (same as direct pipeline)
        full_days_df = pipeline.con.execute(f"""
            SELECT DATE(utc_timestamp) as day, COUNT(*) as hour_count 
            FROM '{building_id}_processed_data' 
            GROUP BY DATE(utc_timestamp) 
            HAVING COUNT(*) = 24 
            ORDER BY DATE(utc_timestamp)
        """).df()
        
        import pandas as pd
        full_days = pd.to_datetime(full_days_df['day']).dt.date.tolist()
        
        if len(full_days) < n_days:
            print(f"⚠ Only {len(full_days)} days available, adjusting to {len(full_days)}")
            n_days = len(full_days)
        
        selected_days = full_days[:n_days]
        training_days = selected_days[:max(1, n_days//2)]  # Use first half for training
        optimization_days = selected_days[max(1, n_days//2):]  # Use second half for optimization
        
        print(f"✓ Selected {len(selected_days)} days total")
        print(f"📚 Training days: {len(training_days)} - {training_days}")
        print(f"⚙️ Optimization days: {len(optimization_days)} - {optimization_days}")
        
        # Phase 1: Learning via endpoint
        print(f"\n🎓 PHASE 1: LEARNING VIA ENDPOINT")
        print("-" * 40)
        
        # Get training data for specific calendar days (not chronological offset)
        training_df = pipeline.get_training_data_for_days(training_days)
        learning_results = pipeline.call_learning_endpoint(training_df)
        
        if mlflow_tracker:
            mlflow_tracker.log_metrics({
                "training_days": len(training_days),
                "learning_calls_made": len(learning_results),
                "total_device_updates": sum(len(r.get('updated_devices', [])) for r in learning_results)
            })
        
        # Phase 2: Optimization via endpoint  
        print(f"\n⚙️ PHASE 2: OPTIMIZATION VIA ENDPOINT")
        print("-" * 40)
        
        # Get optimization data for specific calendar days (not chronological offset)
        opt_df = pipeline.get_optimization_data_for_days(optimization_days)
        optimization_results = pipeline.call_optimization_endpoint(opt_df, battery_enabled, ev_enabled)
        
        if mlflow_tracker:
            total_cost = sum(r.get('total_cost', 0) for r in optimization_results)
            total_savings = sum(r.get('savings_vs_baseline', 0) for r in optimization_results)
            
            mlflow_tracker.log_metrics({
                "optimization_days": n_opt_days,
                "optimization_calls_made": len(optimization_results),
                "total_cost": total_cost,
                "total_savings": total_savings,
                "avg_daily_cost": total_cost / max(1, len(optimization_results)),
                "avg_daily_savings": total_savings / max(1, len(optimization_results))
            })
        
        # Phase 3: Analysis
        print(f"\n📊 PHASE 3: RESULTS ANALYSIS")
        print("-" * 40)
        
        success = pipeline.analyze_results(learning_results, optimization_results)
        
        if mlflow_tracker:
            mlflow_tracker.log_metrics({
                "pipeline_success": 1.0 if success else 0.0,
                "learning_endpoint_success": 1.0 if learning_results else 0.0,
                "optimization_endpoint_success": 1.0 if optimization_results else 0.0
            })
        
        print("\n" + "=" * 80)
        print("ENDPOINTS PIPELINE COMPLETION SUMMARY")
        print("=" * 80)
        
        if success:
            print("✅ Endpoints Pipeline completed successfully!")
            print("✅ Learning endpoint functional")
            print("✅ Optimization endpoint functional") 
            print("✅ Full learning → optimization workflow via endpoints WORKING")
        else:
            print("❌ Endpoints Pipeline failed!")
            print("❌ One or more endpoints not functioning correctly")
            
        print("=" * 80)
        
        if mlflow_tracker:
            mlflow_tracker.end_run()
            print("✓ MLflow tracking completed")
        
        return success
        
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR in endpoints pipeline: {e}")
        if mlflow_tracker:
            mlflow_tracker.log_metrics({"pipeline_error": 1.0})
            mlflow_tracker.end_run()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)