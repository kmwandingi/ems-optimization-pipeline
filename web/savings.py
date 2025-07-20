"""
savings.py — stateless cost helpers
"""
from typing import Dict, List, Tuple
import streamlit as st    # touches session_state only

__all__ = [
    "calculate_schedule_cost",
    "calculate_baseline_cost",
    "update_savings_tracking",
]

# ── original code lifted verbatim from app.py ───────────────
def calculate_schedule_cost(schedule: Dict[str, List[float]],
                            prices: List[float]) -> float:
    total_cost = 0.0
    hourly = [0.0]*24
    for dev, loads in schedule.items():
        if dev == "battery_soc":
            continue
        for h, kwh in enumerate(loads):
            hourly[h] += kwh
    for h, kwh in enumerate(hourly):
        total_cost += kwh * prices[h]
    return total_cost


def calculate_baseline_cost(schedule: Dict[str, List[float]],
                            baseline_usage: Dict[str, int],
                            prices: List[float]) -> float:
    cost = 0.0
    for dev, loads in schedule.items():
        if dev == "battery_soc":
            continue
        pattern = [k for k in loads if k > 0]
        start   = baseline_usage.get(dev, 0)
        for j, kwh in enumerate(pattern):
            cost += kwh * prices[(start+j) % 24]
    return cost


def update_savings_tracking(schedule: Dict[str, List[float]],
                            actual_usage: Dict[str, List[float]],
                            prices: List[float],
                            date_str: str,
                            baseline_usage: Dict[str, int]
                            ) -> Tuple[float, float]:
    base  = calculate_baseline_cost(schedule, baseline_usage, prices)
    opt   = calculate_schedule_cost(schedule, prices)
    act   = calculate_schedule_cost(actual_usage, prices)

    pot = base - opt
    real = base - act

    st.session_state.total_potential_savings += pot
    st.session_state.total_actual_savings    += real
    st.session_state.daily_savings[date_str] = {
        "baseline_cost":  base,
        "optimised_cost": opt,
        "actual_cost":    act,
        "potential":      pot,
        "actual":         real,
    }
    return pot, real
