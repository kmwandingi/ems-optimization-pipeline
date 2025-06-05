# device_specs.py
# we assume each device can be modeled as multiple 1‑hour phases,
# one per hour of the day, with energy_kwh matching the *average* load,
# and peak_kw matching the device’s peak load.

# Added `flex_model` for each device: one of 'discrete_phase', 'partial_usage', or 'fixed'.

device_specs = {
    # -------------------------------------------------------
    # Device with phases and flexibility models
    # -------------------------------------------------------
    "_charge": {
        "category": "Highly Flexible",
        "power_rating": 4.0,
        "allowed_hours": list(range(0, 24)),
        "phases": [
            {"duration": 1, "energy_kwh": 4.0, "peak_kw": 4.0}
        ],
        "flex_model": "partial_usage"  # EV charging-style flexibility
    },
    "dishwasher": {
        "category": "Partially Flexible",
        "power_rating": 1.0,
        "allowed_hours": list(range(8, 22)),
        "device_on": {"on": 1, "at_time": 7},
        "phases": [
            {"duration": 1, "energy_kwh": 0.35, "peak_kw": 1.0},
            {"duration": 1, "energy_kwh": 0.25, "peak_kw": 1.0},
            {"duration": 1, "energy_kwh": 0.2,  "peak_kw": 1.0}
        ],
        "flex_model": "discrete_phase"
    },
    "washing_machine": {
        "category": "Partially Flexible",
        "power_rating": 2.0,
        "allowed_hours": list(range(8, 22)),
        "phases": [
            {"duration": 1, "energy_kwh": 0.6, "peak_kw": 2.0},
            {"duration": 1, "energy_kwh": 0.8, "peak_kw": 2.0}
        ],
        "flex_model": "discrete_phase"
    },
    "heat_pump": {
        "category": "Highly Flexible",
        "power_rating": 3.0,
        "allowed_hours": list(range(6, 24)),
        "phases": [
            {"duration": 2, "energy_kwh": 2.8, "peak_kw": 3.0},
            {"duration": 2, "energy_kwh": 3.0, "peak_kw": 3.0},
            {"duration": 2, "energy_kwh": 2.8, "peak_kw": 3.0}
        ],
        "flex_model": "partial_usage"
    },
    "refrigerator": {
        "category": "Highly Flexible",
        "power_rating": 1.0,
        "allowed_hours": list(range(0, 24)),
        "phases": [
            {"duration": 1, "energy_kwh": 0.025, "peak_kw": 1.0}
            for _ in range(24)
        ],
        "flex_model": "fixed"  # thermostat cycles; minimal user-driven flexibility
    },
    "freezer": {
        "category": "Highly Flexible",
        "power_rating": 1.0,
        "allowed_hours": list(range(0, 24)),
        "phases": [
            {"duration": 1, "energy_kwh": 0.025, "peak_kw": 1.0}
            for _ in range(24)
        ],
        "flex_model": "fixed"
    },
    "circulation_pump": {
        "category": "Highly Flexible",
        "power_rating": 3.0,
        "allowed_hours": list(range(0, 24)),
        "phases": [
            {"duration": 1, "energy_kwh": 0.05, "peak_kw": 3.0}
        ],
        "flex_model": "partial_usage"
    },
    "ev": {
        "category": "Partially Flexible",
        "power_rating": 5.5,
        "allowed_hours": list(range(0, 24)),
        "phases": [
            {"duration": 1, "energy_kwh": 5.5, "peak_kw": 5.0},  # First hour: highest charging rate
            {"duration": 1, "energy_kwh": 4.0, "peak_kw": 5.0},  # Second hour: reduced rate
            {"duration": 1, "energy_kwh": 3.0, "peak_kw": 5.0},  # Third hour: further reduced
            {"duration": 1, "energy_kwh": 5.5, "peak_kw": 5.0},  # Fourth hour: lowest rate
            {"duration": 1, "energy_kwh": 5.5, "peak_kw": 5.0},  # Fourth hour: lowest rate
            {"duration": 1, "energy_kwh": 5.5, "peak_kw": 5.0},  # Fourth hour: lowest rate
        ],
        "flex_model": "partial_usage"
    },
    # -------------------------------------------------------
    # New entries for common household devices
    # -------------------------------------------------------
    "dryer": {
        "category": "Partially Flexible",
        "power_rating": 2.5,
        "allowed_hours": list(range(8, 22)),
        "phases": [
            {"duration": 1, "energy_kwh": 1.8, "peak_kw": 2.5},
            {"duration": 1, "energy_kwh": 1.8, "peak_kw": 2.5}
        ],
        "flex_model": "discrete_phase"  # fixed cycle dryer
    },
    "water_heater": {
        "category": "Highly Flexible",
        "power_rating": 4.5,
        "allowed_hours": list(range(0, 24)),
        "phases": [
            {"duration": 1, "energy_kwh": 3.0, "peak_kw": 4.5}
            for _ in range(3)
        ],
        "flex_model": "partial_usage"  # can spread heating load
    },
    "oven": {
        "category": "Partially Flexible",
        "power_rating": 3.0,
        "allowed_hours": list(range(6, 22)),
        "phases": [
            {"duration": 1, "energy_kwh": 2.0, "peak_kw": 3.0}
        ],
        "flex_model": "discrete_phase"
    },
    "coffee_maker": {
        "category": "Partially Flexible",
        "power_rating": 1.2,
        "allowed_hours": list(range(6, 10)),
        "phases": [
            {"duration": 0.25, "energy_kwh": 0.3, "peak_kw": 1.2}
        ],
        "flex_model": "discrete_phase"
    },
    "lighting": {
        "category": "Always On",
        "power_rating": 0.1,
        "allowed_hours": list(range(17, 24)) + list(range(0, 6)),
        "phases": [
            {"duration": 1, "energy_kwh": 0.1, "peak_kw": 0.1}
            for _ in range(6)
        ],
        "flex_model": "fixed"
    },
    "computer": {
        "category": "Always On",
        "power_rating": 0.2,
        "allowed_hours": list(range(0, 24)),
        "phases": [
            {"duration": 1, "energy_kwh": 0.15, "peak_kw": 0.2}
            for _ in range(8)
        ],
        "flex_model": "fixed"
    },
    # "tumble_dryer": {
    #     "category": "Partially Flexible",
    #     "power_rating": 2.0,
    #     "allowed_hours": list(range(8, 22)),
    #     "phases": [
    #         {"duration": 1, "energy_kwh": 1.0, "peak_kw": 2.0},
    #         {"duration": 1, "energy_kwh": 1.0, "peak_kw": 2.0}
    #     ],
    #     "flex_model": "discrete_phase"
    # },
    "fridge": {
        "category": "Fixed",
        "power_rating": 0.08,
        "allowed_hours": list(range(0, 24)),
        "phases": [
            {"duration": 1, "energy_kwh": 0.08, "peak_kw": 0.1}
            for _ in range(24)
        ],
        "flex_model": "fixed"
    },
    "pump": {
        "category": "Highly Flexible",
        "power_rating": 1.5,
        "allowed_hours": list(range(0, 24)),
        "phases": [
            {"duration": 1, "energy_kwh": 0.05, "peak_kw": 1.5}
        ],
        "flex_model": "partial_usage"
    },

}
