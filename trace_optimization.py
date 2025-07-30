#!/usr/bin/env python3
"""
Trace the actual optimization behavior step by step
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add notebooks directory to path
notebooks_dir = Path(__file__).parent / 'notebooks'
sys.path.insert(0, str(notebooks_dir))

print("Tracing optimization behavior...")

try:
    from pulp import LpProblem, LpVariable, LpMinimize, lpSum, PULP_CBC_CMD, value
    from agents.BatteryAgent import BatteryAgent
    from utils.config import BATTERY_PARAMS
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

def create_minimal_optimization_problem(prices, test_name):
    """Create and solve a minimal battery optimization problem"""
    
    print(f"\n{'='*50}")
    print(f"TESTING: {test_name}")
    print(f"Prices: {prices}")
    print(f"Price std: {np.std(prices):.6f}")
    print(f"{'='*50}")
    
    # Create battery agent
    battery_agent = BatteryAgent(**BATTERY_PARAMS)
    battery_state = battery_agent.get_battery_state()
    
    # Create MILP problem
    prob = LpProblem(f"Battery_Test_{test_name}", LpMinimize)
    
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
    
    # Set objective
    if updated_cost_terms:
        prob += lpSum(updated_cost_terms), "TotalCost"
    
    print(f"Problem setup:")
    print(f"  Variables: {len(prob.variables())}")
    print(f"  Constraints: {len(prob.constraints)}")
    print(f"  Cost terms: {len(updated_cost_terms)}")
    
    # Analyze cost terms
    print(f"\nCost terms breakdown:")
    price_terms = []
    degradation_terms = []
    other_terms = []
    
    for i, term in enumerate(updated_cost_terms):
        term_str = str(term)
        if any(f"charge_{h}" in term_str or f"discharge_{h}" in term_str for h in range(n_hours)):
            if any(f"{price}" in term_str for price in prices):
                price_terms.append((i, term))
            elif "0.00005" in term_str or "degradation" in term_str.lower():
                degradation_terms.append((i, term))
            else:
                other_terms.append((i, term))
    
    print(f"  Price-related terms: {len(price_terms)}")
    print(f"  Degradation terms: {len(degradation_terms)}")
    print(f"  Other terms: {len(other_terms)}")
    
    # Show first few terms of each type
    if price_terms:
        print(f"  Sample price terms:")
        for i, (idx, term) in enumerate(price_terms[:3]):
            print(f"    [{idx}] {term}")
    
    if degradation_terms:
        print(f"  Sample degradation terms:")
        for i, (idx, term) in enumerate(degradation_terms[:3]):
            print(f"    [{idx}] {term}")
    
    if other_terms:
        print(f"  Sample OTHER terms (MYSTERY):")
        for i, (idx, term) in enumerate(other_terms[:5]):
            print(f"    [{idx}] {term}")
        if len(other_terms) > 5:
            print(f"    ... and {len(other_terms) - 5} more mystery terms")
    
    # Solve the problem
    print(f"\nSolving optimization...")
    try:
        solver = PULP_CBC_CMD(msg=False)
        prob.solve(solver)
        
        status = prob.status
        print(f"Status: {status}")
        
        if status == 1:  # Optimal
            objective_value = value(prob.objective)
            print(f"Objective value: {objective_value:.6f}")
            
            # Extract solution
            solution = {
                'charge': [charge[t].varValue or 0.0 for t in range(n_hours)],
                'discharge': [discharge[t].varValue or 0.0 for t in range(n_hours)],
                'soc': [soc[t].varValue or battery_state['current_soc'] for t in range(n_hours)]
            }
            
            print(f"\nSolution:")
            print(f"  Total charge: {sum(solution['charge']):.3f} kWh")
            print(f"  Total discharge: {sum(solution['discharge']):.3f} kWh")
            print(f"  Net battery flow: {sum(solution['charge']) - sum(solution['discharge']):.3f} kWh")
            
            # Show hourly breakdown
            print(f"\nHourly breakdown:")
            print(f"{'Hour':<4} {'Price':<6} {'Charge':<6} {'Discharge':<8} {'SOC':<6}")
            print("-" * 35)
            for t in range(min(n_hours, 8)):  # Show first 8 hours
                print(f"{t:<4} {prices[t]:<6.3f} {solution['charge'][t]:<6.3f} {solution['discharge'][t]:<8.3f} {solution['soc'][t]:<6.3f}")
            if n_hours > 8:
                print("...")
                for t in range(max(8, n_hours-2), n_hours):  # Show last 2 hours
                    print(f"{t:<4} {prices[t]:<6.3f} {solution['charge'][t]:<6.3f} {solution['discharge'][t]:<8.3f} {solution['soc'][t]:<6.3f}")
            
            # Calculate cost components
            total_arbitrage_cost = sum(prices[t] * (solution['charge'][t] - solution['discharge'][t]) for t in range(n_hours))
            total_degradation_cost = sum(battery_state.get('degradation_rate', 0.00005) * (solution['charge'][t] + solution['discharge'][t]) for t in range(n_hours))
            
            print(f"\nCost breakdown:")
            print(f"  Arbitrage cost: {total_arbitrage_cost:.6f} €")
            print(f"  Degradation cost: {total_degradation_cost:.6f} €")
            print(f"  Total calculated: {total_arbitrage_cost + total_degradation_cost:.6f} €")
            print(f"  Solver objective: {objective_value:.6f} €")
            
            return solution, objective_value
        else:
            print(f"Optimization failed with status: {status}")
            return None, None
            
    except Exception as e:
        print(f"Solver error: {e}")
        return None, None

def main():
    """Main test function"""
    
    print("Battery Optimization Tracing")
    print("="*40)
    
    # Test case 1: Completely flat prices
    flat_prices = [0.20] * 24
    solution_flat, obj_flat = create_minimal_optimization_problem(flat_prices, "FLAT_PRICES")
    
    # Test case 2: Varied prices
    varied_prices = []
    for h in range(24):
        if 6 <= h < 9 or 17 <= h < 21:  # Peak
            varied_prices.append(0.25)
        elif 10 <= h < 16:  # Solar midday
            varied_prices.append(0.15)
        else:  # Off-peak
            varied_prices.append(0.20)
    
    solution_varied, obj_varied = create_minimal_optimization_problem(varied_prices, "VARIED_PRICES")
    
    # Compare results
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")
    
    if solution_flat and solution_varied:
        print(f"{'Metric':<25} {'Flat Prices':<15} {'Varied Prices':<15}")
        print("-" * 60)
        print(f"{'Total Charge (kWh)':<25} {sum(solution_flat['charge']):<15.3f} {sum(solution_varied['charge']):<15.3f}")
        print(f"{'Total Discharge (kWh)':<25} {sum(solution_flat['discharge']):<15.3f} {sum(solution_varied['discharge']):<15.3f}")
        print(f"{'Net Battery Flow (kWh)':<25} {sum(solution_flat['charge']) - sum(solution_flat['discharge']):<15.3f} {sum(solution_varied['charge']) - sum(solution_varied['discharge']):<15.3f}")
        print(f"{'Objective Value (€)':<25} {obj_flat:<15.6f} {obj_varied:<15.6f}")
        
        # Check if there's unrealistic behavior
        flat_discharge_total = sum(solution_flat['discharge'])
        varied_discharge_total = sum(solution_varied['discharge'])
        
        print(f"\nAnalysis:")
        if flat_discharge_total > varied_discharge_total * 2:
            print(f"⚠️  ANOMALY DETECTED: Flat prices show {flat_discharge_total/varied_discharge_total:.1f}x more discharge")
        else:
            print(f"✓ Battery behavior appears consistent between scenarios")
    
    print(f"\nTest completed!")

if __name__ == "__main__":
    main()
