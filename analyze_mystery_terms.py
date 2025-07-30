#!/usr/bin/env python3
"""
Focused analysis of the mystery cost terms in battery optimization
"""

import numpy as np
import sys
from pathlib import Path

# Add notebooks directory to path
notebooks_dir = Path(__file__).parent / 'notebooks'
sys.path.insert(0, str(notebooks_dir))

try:
    from pulp import LpProblem, LpVariable, LpMinimize, lpSum
    from agents.BatteryAgent import BatteryAgent
    from utils.config import BATTERY_PARAMS
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

def analyze_cost_terms_in_detail(prices, test_name):
    """Analyze every single cost term to understand the mystery terms"""
    
    print(f"\n{'='*60}")
    print(f"DETAILED COST TERM ANALYSIS: {test_name}")
    print(f"Prices: {prices[:6]}...{prices[-6:]}")
    print(f"{'='*60}")
    
    # Create battery agent and get state
    battery_agent = BatteryAgent(**BATTERY_PARAMS)
    battery_state = battery_agent.get_battery_state()
    
    print(f"Battery state: {battery_state}")
    
    # Create MILP problem
    prob = LpProblem(f"Analysis_{test_name}", LpMinimize)
    n_hours = len(prices)
    
    # Create battery variables
    charge = LpVariable.dicts("charge", range(n_hours), lowBound=0, upBound=battery_state['max_charge_rate'])
    discharge = LpVariable.dicts("discharge", range(n_hours), lowBound=0, upBound=battery_state['max_discharge_rate'])
    soc = LpVariable.dicts("soc", range(n_hours), lowBound=battery_state['soc_min'], upBound=battery_state['soc_max'])
    y = LpVariable.dicts("y", range(n_hours), cat="Binary")
    
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
    
    print(f"\nTotal cost terms: {len(updated_cost_terms)}")
    
    # Categorize and analyze EVERY term
    categories = {
        'price_arbitrage': [],
        'degradation': [],
        'incentive_charge': [],
        'incentive_discharge': [],
        'binary_bonus': [],
        'unknown': []
    }
    
    for i, term in enumerate(updated_cost_terms):
        term_str = str(term)
        
        # Check for price arbitrage terms (price * (charge - discharge))
        if any(f"{price}*charge_" in term_str and f"- {price}*discharge_" in term_str for price in prices):
            categories['price_arbitrage'].append((i, term))
        # Check for degradation terms (small coefficient * (charge + discharge))
        elif "0.00005" in term_str or ("charge_" in term_str and "discharge_" in term_str and "+" in term_str):
            categories['degradation'].append((i, term))
        # Check for charge incentive terms (negative coefficient * charge)
        elif "charge_" in term_str and term_str.startswith("-"):
            categories['incentive_charge'].append((i, term))
        # Check for discharge incentive terms (negative coefficient * discharge)
        elif "discharge_" in term_str and term_str.startswith("-"):
            categories['incentive_discharge'].append((i, term))
        # Check for binary variable bonuses
        elif "should_charge" in term_str or "should_discharge" in term_str:
            categories['binary_bonus'].append((i, term))
        else:
            categories['unknown'].append((i, term))
    
    # Report findings
    print(f"\nCOST TERM CATEGORIZATION:")
    print(f"{'Category':<20} {'Count':<8} {'Sample Terms'}")
    print("-" * 80)
    
    for category, terms in categories.items():
        sample_terms = [str(term) for _, term in terms[:3]]
        print(f"{category:<20} {len(terms):<8} {sample_terms}")
        
        if category == 'unknown' and terms:
            print(f"\n⚠️  UNKNOWN TERMS DETECTED - These might be the mystery terms!")
            for i, (idx, term) in enumerate(terms[:10]):
                print(f"    [{idx}] {term}")
            if len(terms) > 10:
                print(f"    ... and {len(terms) - 10} more unknown terms")
    
    # Calculate expected vs actual term counts
    expected_price_terms = n_hours  # One per hour: price * (charge - discharge)
    expected_degradation_terms = n_hours * 2  # One per hour for charge, one for discharge
    
    print(f"\nEXPECTED vs ACTUAL:")
    print(f"Price arbitrage terms: Expected {expected_price_terms}, Got {len(categories['price_arbitrage'])}")
    print(f"Degradation terms: Expected {expected_degradation_terms}, Got {len(categories['degradation'])}")
    print(f"Incentive terms: Charge {len(categories['incentive_charge'])}, Discharge {len(categories['incentive_discharge'])}")
    print(f"Binary bonus terms: {len(categories['binary_bonus'])}")
    print(f"Unknown terms: {len(categories['unknown'])}")
    
    # Check if we have the problematic incentive terms
    total_incentive_terms = len(categories['incentive_charge']) + len(categories['incentive_discharge'])
    if total_incentive_terms > 0:
        print(f"\n🔍 FOUND INCENTIVE TERMS!")
        print(f"These might be creating artificial 'profits' that drive unrealistic behavior.")
        
        # Show some incentive terms
        if categories['incentive_charge']:
            print(f"\nCharge incentive terms:")
            for i, (idx, term) in enumerate(categories['incentive_charge'][:5]):
                print(f"  [{idx}] {term}")
        
        if categories['incentive_discharge']:
            print(f"\nDischarge incentive terms:")
            for i, (idx, term) in enumerate(categories['incentive_discharge'][:5]):
                print(f"  [{idx}] {term}")
    
    return categories

def main():
    """Main analysis function"""
    
    print("Mystery Cost Terms Analysis")
    print("="*40)
    
    # Test flat prices
    flat_prices = [0.20] * 24
    flat_categories = analyze_cost_terms_in_detail(flat_prices, "FLAT_PRICES")
    
    # Test varied prices
    varied_prices = []
    for h in range(24):
        if 6 <= h < 9 or 17 <= h < 21:  # Peak
            varied_prices.append(0.25)
        elif 10 <= h < 16:  # Solar midday
            varied_prices.append(0.15)
        else:  # Off-peak
            varied_prices.append(0.20)
    
    varied_categories = analyze_cost_terms_in_detail(varied_prices, "VARIED_PRICES")
    
    # Compare the two scenarios
    print(f"\n{'='*60}")
    print("COMPARISON BETWEEN SCENARIOS")
    print(f"{'='*60}")
    
    print(f"{'Category':<20} {'Flat Count':<12} {'Varied Count':<12} {'Difference'}")
    print("-" * 60)
    
    for category in flat_categories.keys():
        flat_count = len(flat_categories[category])
        varied_count = len(varied_categories[category])
        diff = varied_count - flat_count
        print(f"{category:<20} {flat_count:<12} {varied_count:<12} {diff:+d}")
    
    # Identify the root cause
    flat_incentives = len(flat_categories['incentive_charge']) + len(flat_categories['incentive_discharge'])
    varied_incentives = len(varied_categories['incentive_charge']) + len(varied_categories['incentive_discharge'])
    
    print(f"\n🎯 ROOT CAUSE ANALYSIS:")
    print(f"Total incentive terms - Flat: {flat_incentives}, Varied: {varied_incentives}")
    
    if flat_incentives != varied_incentives:
        print(f"⚠️  DIFFERENT INCENTIVE STRUCTURES between scenarios!")
        print(f"This could explain the different battery behavior.")
    else:
        print(f"✓ Same number of incentive terms in both scenarios.")
        print(f"The difference must be in the term values or constraints.")

if __name__ == "__main__":
    main()
