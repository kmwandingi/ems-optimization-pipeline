import numpy as np

"""
Unified configuration file that collects all project parameters and default values.
This file is imported across the project to ensure complete consistency.
"""

# --------------------------
# Device Agents
# --------------------------
BATTERY_PARAMS = {
    "capacity": 15.0,
    "max_charge_rate": 5.0,
    "max_discharge_rate": 5.0,
    "initial_soc": 8.0,
    "soc_min": 0.6,
    "soc_max": 14.0,
    "degradation_rate": 0.00001,
    "temperature_coefficient": 1.0,
    "max_ramp_rate": 1.0,
    "efficiency_charge": 0.95,
    "efficiency_discharge": 0.95
}


EV_PARAMS = {
    "capacity": 60.0,
    "efficiency_discharge": 0.92,
    "efficiency_charge": 0.92,
    "max_discharge_rate": 11.0,
    "max_charge_rate": 11.0,
    "initial_soc": 20,
    "soc_min":3,
    "soc_max": 57,
    "power_rating": 11.0,
    "must_be_full_by_hour": 7,
    "degradation_cost": 0.007
}

# --------------------------
# Battery Tuning Parameters
# --------------------------
BATTERY_TUNING_PARAMETERS = [
    {
        "max_charge_rate": 5.0,
        "max_discharge_rate": 5.0,
        "initial_soc": 5.0,
        "soc_min": 0.4,
        "soc_max": 6.0,
        "degradation_rate": 0.00001,
        "temperature_coefficient": 1.0,
        "max_ramp_rate": 1.0
    },
    {
        "max_charge_rate": 6.0,
        "max_discharge_rate": 6.0,
        "initial_soc": 5.0,
        "soc_min": 0.5,
        "soc_max": 6.0,
        "degradation_rate": 0.00001,
        "temperature_coefficient": 1.0,
        "max_ramp_rate": 1.0
    },
    {
        "max_charge_rate": 10.0,
        "max_discharge_rate": 10.0,
        "initial_soc": 5.0,
        "soc_min": 0.6,
        "soc_max": 7.0,
        "degradation_rate": 0.00001,
        "temperature_coefficient": 1.0,
        "max_ramp_rate": 1.0
    },
    {
        "max_charge_rate": 12.0,
        "max_discharge_rate": 12.0,
        "initial_soc": 7.0,
        "soc_min": 0.6,
        "soc_max": 7.0,
        "degradation_rate": 0.00001,
        "temperature_coefficient": 1.0,
        "max_ramp_rate": 1.0
    },
    {
        "max_charge_rate": 14.0,
        "max_discharge_rate": 14.0,
        "initial_soc": 10.0,
        "soc_min": 0.6,
        "soc_max": 12.0,
        "degradation_rate": 0.00001,
        "temperature_coefficient": 1.0,
        "max_ramp_rate": 1.0
    }
]

# --------------------------
# Flexible Device Parameters
# --------------------------
FLEXIBLE_DEVICE_PARAMS = {
    "max_shift_hours": 6,
    "is_flexible": True,
    "min_on_time_per_device": {
        "dishwasher": 1,
        "washing_machine": 1,
        "heat_pump": 1
    },
    "min_off_time_per_device": {
        "dishwasher": 1,
        "washing_machine": 1,
        "heat_pump": 1
    },
    "max_on_time_per_device": {
        "dishwasher": 14,
        "washing_machine": 13,
        "heat_pump": 18
    }
}

# --------------------------
# Grid Agent Parameters
# --------------------------
GRID_AGENT_PARAMS = {
    "import_price": 0.25,
    "export_price": 0.10,
    "max_import": 50.0,
    "max_export": 20.0
}

# --------------------------
# Global Connection Layer Parameters
# --------------------------
GLOBAL_CONNECTION_PARAMS = {
    "max_building_load": 100.0
    # total_hours is set at runtime (24 hours per day)
}

# --------------------------
# Global Optimizer Parameters
# --------------------------
GLOBAL_OPTIMIZER_PARAMS = {
    "max_iterations": 5,
    "online_iterations": 3,
    "monte_carlo_num_simulations": 10,
    "monte_carlo_z_alpha": 1.645
}

# --------------------------
# PV Agent Parameters
# --------------------------
PV_AGENT_PARAMS = {
    "profile_cols": None,
    "forecast_cols": None
}

# --------------------------
# Weather Agent Parameters
# --------------------------
WEATHER_AGENT_PARAMS = {
    # Currently no additional parameters
}

# --------------------------
# Building List
# --------------------------
BUILDINGS = [
    "DE_KN_residential1",
    "DE_KN_residential3",
    "DE_KN_residential4",
    "DE_KN_residential6"
]

# --------------------------
# Logging & Output
# --------------------------
LOGGING_LEVEL = "INFO"
OPTIMIZATION_RESULTS_FILE = "all_building_results_without_simulation.json"

# --------------------------
# Derived Settings
# --------------------------
IMPORT_PRICE       = GRID_AGENT_PARAMS["import_price"]
EXPORT_PRICE       = GRID_AGENT_PARAMS["export_price"]
MAX_BUILDING_LOAD  = GLOBAL_CONNECTION_PARAMS["max_building_load"]

# Parameters passed via **kwargs
GRID_PARAMS       = {"import_price": GRID_AGENT_PARAMS["import_price"], "export_price": GRID_AGENT_PARAMS["export_price"], "max_import": GRID_AGENT_PARAMS["max_import"], "max_export": GRID_AGENT_PARAMS["max_export"] }
FLEXIBLE_PARAMS   = FLEXIBLE_DEVICE_PARAMS
PV_PARAMS         = PV_AGENT_PARAMS
EV_PARAMS         = EV_PARAMS

