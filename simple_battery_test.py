#!/usr/bin/env python3
"""
Simple test to trace battery optimization behavior
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os
from pathlib import Path

# Add notebooks directory to path
notebooks_dir = Path(__file__).parent / 'notebooks'
sys.path.insert(0, str(notebooks_dir))

print("Starting battery behavior test...")
print(f"Python path includes: {notebooks_dir}")

try:
    from agents.BatteryAgent import BatteryAgent
    print("✓ BatteryAgent imported successfully")
except ImportError as e:
    print(f"✗ Failed to import BatteryAgent: {e}")
    sys.exit(1)

try:
    from utils.config import BATTERY_PARAMS
    print("✓ BATTERY_PARAMS imported successfully")
    print(f"Battery params: {BATTERY_PARAMS}")
except ImportError as e:
    print(f"✗ Failed to import BATTERY_PARAMS: {e}")
    # Use default params
    BATTERY_PARAMS = {
        'estimated_capacity': 10.0,
        'max_charge_rate': 5.0,
        'max_discharge_rate': 5.0,
        'charge_efficiency': 0.95,
        'discharge_efficiency': 0.95,
        'soc_min': 0.1,
        'soc_max': 0.9,
        'current_soc': 0.5,
        'degradation_rate': 0.00005
    }

def test_battery_cost_calculation():
    """Test the battery cost calculation directly"""
    
    print("\n" + "="*60)
    print("TESTING BATTERY COST CALCULATION")
    print("="*60)
    
    # Create battery agent
    battery_agent = BatteryAgent(**BATTERY_PARAMS)
    
    # Test scenario 1: Flat prices
    flat_prices = [0.20] * 24
    print(f"\nTest 1 - Flat prices: {flat_prices[:3]}...{flat_prices[-3:]}")
    
    # Test scenario 2: Varied prices  
    varied_prices = []
    for h in range(24):
        if 6 <= h < 9 or 17 <= h < 21:  # Peak
            varied_prices.append(0.25)
        elif 10 <= h < 16:  # Solar midday
            varied_prices.append(0.15)
        else:  # Off-peak
            varied_prices.append(0.20)
    
    print(f"Test 2 - Varied prices: {varied_prices[:6]}...{varied_prices[-6:]}")
    
    # Create a simple MILP problem to test cost terms
    try:
        from pulp import LpProblem, LpVariable, LpMinimize, lpSum
        print("✓ PuLP imported successfully")
        
        for test_name, prices in [("FLAT", flat_prices), ("VARIED", varied_prices)]:
            print(f"\n--- Testing {test_name} prices ---")
            
            # Create simple MILP problem
            prob = LpProblem(f"Battery_Test_{test_name}", LpMinimize)
            
            # Create battery variables
            n_hours = 24
            charge = LpVariable.dicts("charge", range(n_hours), lowBound=0, upBound=5.0)
            discharge = LpVariable.dicts("discharge", range(n_hours), lowBound=0, upBound=5.0)
            soc = LpVariable.dicts("soc", range(n_hours), lowBound=1.0, upBound=9.0)
            y = LpVariable.dicts("y", range(n_hours), cat="Binary")
            
            # Get battery state
            battery_state = battery_agent.get_battery_state()
            print(f"Battery state: {battery_state}")
            
            # Add battery constraints and get cost terms
            cost_terms = []
            prob, updated_cost_terms = battery_agent.add_battery_constraints_to_milp(
                prob=prob,
                battery_state=battery_state,
                n_periods=n_hours,
                charge=charge,
                discharge=discharge,
                soc=soc,
                prices=prices,
                y=y,
                cost_terms=cost_terms,
                force_arbitrage=False,
                problem_type="centralized",
                name_prefix="Test"
            )
            
            print(f"Number of cost terms added: {len(updated_cost_terms)}")
            print(f"Number of constraints: {len(prob.constraints)}")
            
            # Analyze the cost terms
            print("Cost terms analysis:")
            for i, term in enumerate(updated_cost_terms[:10]):  # Show first 10 terms
                print(f"  Term {i}: {term}")
            
            if len(updated_cost_terms) > 10:
                print(f"  ... and {len(updated_cost_terms) - 10} more terms")
                
    except ImportError as e:
        print(f"✗ Failed to import PuLP: {e}")
        return

def test_price_variation_detection():
    """Test how price variation is detected and handled"""
    
    print("\n" + "="*60)
    print("TESTING PRICE VARIATION DETECTION")
    print("="*60)
    
    # Test different price patterns
    test_cases = [
        ("Completely flat", [0.20] * 24),
        ("Small variation", [0.20, 0.201, 0.199, 0.20] * 6),
        ("Large variation", [0.15, 0.20, 0.25, 0.20] * 6),
        ("Peak pattern", [0.15 if 10 <= h < 16 else 0.25 if 17 <= h < 21 else 0.20 for h in range(24)])
    ]
    
    for name, prices in test_cases:
        price_array = np.array(prices)
        std_dev = price_array.std()
        min_price = price_array.min()
        max_price = price_array.max()
        range_ratio = (max_price - min_price) / price_array.mean() if price_array.mean() > 0 else 0
        
        print(f"\n{name}:")
        print(f"  Prices: {prices[:6]}...{prices[-6:]}")
        print(f"  Std dev: {std_dev:.6f}")
        print(f"  Range: {min_price:.3f} - {max_price:.3f}")
        print(f"  Range ratio: {range_ratio:.3f}")
        print(f"  Unique values: {len(set(prices))}")

def main():
    """Main test function"""
    
    print("Simple Battery Behavior Test")
    print("="*40)
    
    # Test 1: Price variation detection
    test_price_variation_detection()
    
    # Test 2: Battery cost calculation
    test_battery_cost_calculation()
    
    print("\n" + "="*60)
    print("TEST COMPLETED")
    print("="*60)

if __name__ == "__main__":
    main()
