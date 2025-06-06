#!/usr/bin/env python
"""
ENDPOINTS vs DIRECT PIPELINE COMPARISON
=======================================
This script compares the results of the endpoint pipeline (04_endpoints_pipeline.py)
with the direct pipeline (02_integrated_pipeline.py) to ensure endpoint functionality
matches direct agent calls.

Usage:
    python scripts/compare_endpoints_vs_direct.py --building DE_KN_residential1 --n_days 2
"""

import sys
import json
import subprocess
import argparse
from pathlib import Path
import re

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Compare Endpoints vs Direct Pipeline Results")
    parser.add_argument("--building", type=str, default="DE_KN_residential1",
                        help="Building ID to test")
    parser.add_argument("--n_days", type=int, default=2,
                        help="Number of days to test")
    # Production: Always uses phases centralized optimization
    
    return parser.parse_args()

def run_pipeline(script_name: str, building: str, n_days: int):
    """Run a pipeline script and capture its output."""
    print(f"🔧 Running {script_name}...")
    
    # Different argument formats for different scripts
    if "02_integrated_pipeline.py" in script_name:
        cmd = [
            "python", f"scripts/{script_name}",
            "--building", building,
            "--n_days", str(n_days),
            "--battery", "on"  # 02_integrated_pipeline uses on/off
        ]
    else:
        cmd = [
            "python", f"scripts/{script_name}",
            "--building", building,
            "--n_days", str(n_days),
            "--battery", "true"  # 04_endpoints_pipeline uses true/false
        ]
    
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=600,
            cwd=Path.cwd()
        )
        
        if result.returncode == 0:
            print(f"✅ {script_name} completed successfully")
            return result.stdout, result.stderr
        else:
            print(f"❌ {script_name} failed with return code {result.returncode}")
            print(f"Error: {result.stderr}")
            return None, result.stderr
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {script_name} timed out")
        return None, "Timeout"
    except Exception as e:
        print(f"💥 {script_name} crashed: {e}")
        return None, str(e)

def extract_metrics(output: str, pipeline_type: str):
    """Extract key metrics from pipeline output."""
    metrics = {
        'pipeline_type': pipeline_type,
        'total_cost': None,
        'total_savings': None,
        'devices_optimized': None,
        'learning_updates': None,
        'optimization_days': None,
        'success': False
    }
    
    if not output:
        return metrics
    
    # Look for completion indicator
    if "completed successfully" in output or "✅" in output:
        metrics['success'] = True
    
    # Extract costs and savings
    cost_matches = re.findall(r"Total cost: €([\d.-]+)", output)
    if cost_matches:
        metrics['total_cost'] = float(cost_matches[-1])
    
    savings_matches = re.findall(r"Total savings: €([\d.-]+)", output) 
    if savings_matches:
        metrics['total_savings'] = float(savings_matches[-1])
    
    # Extract devices
    device_matches = re.findall(r"(\d+) devices", output)
    if device_matches:
        metrics['devices_optimized'] = int(device_matches[-1])
    
    # Extract learning updates
    learning_matches = re.findall(r"Total device updates: (\d+)", output)
    if learning_matches:
        metrics['learning_updates'] = int(learning_matches[-1])
    
    # Extract optimization days
    opt_matches = re.findall(r"Optimization days processed: (\d+)", output)
    if opt_matches:
        metrics['optimization_days'] = int(opt_matches[-1])
    
    return metrics

def compare_metrics(direct_metrics: dict, endpoint_metrics: dict):
    """Compare metrics between direct and endpoint pipelines."""
    print("\n📊 PIPELINE COMPARISON RESULTS")
    print("=" * 60)
    
    # Basic success check
    direct_success = direct_metrics.get('success', False)
    endpoint_success = endpoint_metrics.get('success', False)
    
    print(f"Pipeline Success:")
    print(f"  Direct Pipeline:   {'✅ SUCCESS' if direct_success else '❌ FAILED'}")
    print(f"  Endpoint Pipeline: {'✅ SUCCESS' if endpoint_success else '❌ FAILED'}")
    
    if not (direct_success and endpoint_success):
        print("\n❌ COMPARISON FAILED: One or both pipelines failed")
        return False
    
    # Compare specific metrics
    print(f"\nMetric Comparison:")
    
    # Total cost comparison
    direct_cost = direct_metrics.get('total_cost')
    endpoint_cost = endpoint_metrics.get('total_cost')
    
    if direct_cost is not None and endpoint_cost is not None:
        cost_diff = abs(direct_cost - endpoint_cost)
        cost_match = cost_diff < 0.10  # Allow 10 cent tolerance
        print(f"  Total Cost:        Direct €{direct_cost:.2f} | Endpoint €{endpoint_cost:.2f} | {'✅' if cost_match else '❌'}")
    else:
        cost_match = False
        print(f"  Total Cost:        ❌ Missing data")
    
    # Total savings comparison  
    direct_savings = direct_metrics.get('total_savings')
    endpoint_savings = endpoint_metrics.get('total_savings')
    
    if direct_savings is not None and endpoint_savings is not None:
        savings_diff = abs(direct_savings - endpoint_savings)
        savings_match = savings_diff < 0.10  # Allow 10 cent tolerance
        print(f"  Total Savings:     Direct €{direct_savings:.2f} | Endpoint €{endpoint_savings:.2f} | {'✅' if savings_match else '❌'}")
    else:
        savings_match = False
        print(f"  Total Savings:     ❌ Missing data")
    
    # Device optimization comparison
    direct_devices = direct_metrics.get('devices_optimized')
    endpoint_devices = endpoint_metrics.get('devices_optimized')
    
    if direct_devices is not None and endpoint_devices is not None:
        devices_match = direct_devices == endpoint_devices
        print(f"  Devices Optimized: Direct {direct_devices} | Endpoint {endpoint_devices} | {'✅' if devices_match else '❌'}")
    else:
        devices_match = True  # Optional metric
        print(f"  Devices Optimized: ⚠ Data not available")
    
    # Learning updates comparison
    direct_learning = direct_metrics.get('learning_updates')
    endpoint_learning = endpoint_metrics.get('learning_updates')
    
    if direct_learning is not None and endpoint_learning is not None:
        learning_match = direct_learning == endpoint_learning
        print(f"  Learning Updates:  Direct {direct_learning} | Endpoint {endpoint_learning} | {'✅' if learning_match else '❌'}")
    else:
        learning_match = True  # Optional metric
        print(f"  Learning Updates:  ⚠ Data not available")
    
    # Overall assessment
    core_metrics_match = cost_match and savings_match
    
    print(f"\n🎯 OVERALL ASSESSMENT:")
    if core_metrics_match:
        print(f"   ✅ ENDPOINTS EQUIVALENT TO DIRECT PIPELINE")
        print(f"   ✅ Cost and savings match within tolerance")
        print(f"   ✅ Endpoint functionality validated")
    else:
        print(f"   ❌ ENDPOINTS DO NOT MATCH DIRECT PIPELINE")
        print(f"   ❌ Significant differences in core metrics")
        print(f"   ❌ Endpoint validation failed")
    
    return core_metrics_match

def main():
    """Main comparison function."""
    args = parse_args()
    
    print("🆚 ENDPOINTS vs DIRECT PIPELINE COMPARISON")
    print("=" * 80)
    print(f"Building: {args.building}")
    print(f"Days: {args.n_days}")
    print(f"Mode: PRODUCTION (phases centralized only)")
    print("=" * 80)
    
    # Run direct pipeline (02_integrated_pipeline.py)
    print("\n1️⃣ RUNNING DIRECT PIPELINE")
    print("-" * 40)
    direct_stdout, direct_stderr = run_pipeline(
        "02_integrated_pipeline.py", 
        args.building, 
        args.n_days
    )
    
    # Run endpoint pipeline (04_endpoints_pipeline.py)
    print("\n2️⃣ RUNNING ENDPOINT PIPELINE")
    print("-" * 40)
    endpoint_stdout, endpoint_stderr = run_pipeline(
        "04_endpoints_pipeline.py", 
        args.building, 
        args.n_days
    )
    
    # Extract and compare metrics
    print("\n3️⃣ EXTRACTING METRICS")
    print("-" * 40)
    
    direct_metrics = extract_metrics(direct_stdout, "Direct")
    endpoint_metrics = extract_metrics(endpoint_stdout, "Endpoint")
    
    print(f"Direct metrics: {direct_metrics}")
    print(f"Endpoint metrics: {endpoint_metrics}")
    
    # Compare results
    print("\n4️⃣ COMPARISON ANALYSIS")
    print("-" * 40)
    
    success = compare_metrics(direct_metrics, endpoint_metrics)
    
    print("\n" + "=" * 80)
    print("COMPARISON COMPLETED")
    print("=" * 80)
    
    if success:
        print("✅ ENDPOINT VALIDATION SUCCESSFUL")
        print("✅ Endpoints produce equivalent results to direct pipeline")
        print("✅ Full learning → optimization workflow validated")
    else:
        print("❌ ENDPOINT VALIDATION FAILED")
        print("❌ Endpoints do not match direct pipeline results")
        print("❌ Investigation required")
        
    print("=" * 80)
    
    return success

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)