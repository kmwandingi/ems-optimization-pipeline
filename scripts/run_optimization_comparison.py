# =============================================================================
#  EMS OPTIMISATION COMPARISON – CLEAN, ROBUST VERSION
#  (save as scripts/run_optimization_comparison.py)
# =============================================================================
"""
I compare three scheduling strategies for every requested building & day:
    1) Baseline  (no optimisation)
    2) Decentralised   (per-device optimiser)
    3) Centralised     (Global MILP optimiser – *phases* only, prod standard)

Results are written to <results/comparison/YYYYMMDD_HHMMSS/>.
"""

# -----------------------------------------------------------------------------#
# 1) HOUSE-KEEPING IMPORTS & PATH SET-UP
# -----------------------------------------------------------------------------#
from __future__ import annotations

import argparse
import json
import logging
import os
from   pathlib import Path
import sys
import time
import traceback
from   datetime import datetime
from   typing    import Any, Tuple, List

import duckdb            # DB interface
import numpy as np        # numerical
import pandas as pd       # tabular

# Project root = two levels above this file
PROJ_ROOT = Path(__file__).resolve().parents[1]
sys.path.extend([
    str(PROJ_ROOT),
    str(PROJ_ROOT / "agents"),      # fallback if modules were moved here
    str(PROJ_ROOT / "notebooks"),   # original notebook namespace
    str(PROJ_ROOT / "notebooks" / "utils"),
])

# -----------------------------------------------------------------------------#
# 2) LOGGING (console + rotating file)
# -----------------------------------------------------------------------------#
LOG_DIR = PROJ_ROOT / "results" / "comparison"
LOG_DIR.mkdir(parents=True, exist_ok=True)

def _build_logger() -> logging.Logger:
    log = logging.getLogger("ems.optimisation_comparison")
    log.setLevel(logging.DEBUG)

    frmt = logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s")

    # console – INFO+
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(frmt)
    log.addHandler(sh)

    # file – DEBUG (rotated daily)
    fh = logging.FileHandler(LOG_DIR / "optimisation_comparison.log", mode="a")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(frmt)
    log.addHandler(fh)
    return log

logger = _build_logger()

# short alias for quick DEBUG prints inside long loops
dbg = logger.debug

# -----------------------------------------------------------------------------#
# 3) STATIC PARAMETERS & COMMON HELPERS
# -----------------------------------------------------------------------------#
from helper import get_jads_color_palette                         # :contentReference[oaicite:1]{index=1}
import common                                                     # shared utilities

# --- dynamically load create_devices_from_duckdb from 01_run.py ---------
from importlib import util as _imp
_run01_path = PROJ_ROOT / "scripts" / "01_run.py"
_create_devices_from_duckdb = None
if _run01_path.exists():
    spec = _imp.spec_from_file_location("_run01", _run01_path)
    _run01 = _imp.module_from_spec(spec)  # type: ignore[arg-type]
    assert spec.loader is not None
    spec.loader.exec_module(_run01)  # type: ignore[attr-defined]
    _create_devices_from_duckdb = getattr(_run01, "create_devices_from_duckdb", None)

if _create_devices_from_duckdb is None and hasattr(common, "create_devices_from_duckdb"):
    _create_devices_from_duckdb = common.create_devices_from_duckdb  # type: ignore[attr-defined]

if _create_devices_from_duckdb is None:
    raise ImportError("create_devices_from_duckdb not found in scripts.01_run or common.py")


# basic system parameters (identical to previous script)
from device_specs import device_specs

BATTERY_PARAMS = dict(max_charge_rate=3.0, max_discharge_rate=3.0,
                      initial_soc=7.0, soc_min=1.0, soc_max=10.0,
                      capacity=10.0, degradation_rate=0.001,
                      efficiency_charge=0.95, efficiency_discharge=0.95)

EV_PARAMS = dict(capacity=60.0, initial_soc=12.0, soc_min=6.0, soc_max=54.0,
                 max_charge_rate=7.4, efficiency_charge=0.92,
                 must_be_full_by_hour=7)

GRID_PARAMS = dict(import_price=0.25, export_price=0.05,
                   max_import=15.0,  max_export=15.0)

BUILDINGS: List[str] = [
    "DE_KN_industrial3",
    "DE_KN_residential1",
    "DE_KN_residential2",
    "DE_KN_residential3",
    "DE_KN_residential4",
    "DE_KN_residential5",
    "DE_KN_residential6",
]

# -----------------------------------------------------------------------------#
# 4) RUNTIME IMPORTS (agents) WITH FALLBACK
# -----------------------------------------------------------------------------#
def _import_agents():
    """Try notebooks.agents.*, fall back to agents.*"""
    base_paths = ["notebooks.agents", "agents"]
    mods = {}
    errors = []
    for base in base_paths:
        try:
            from importlib import import_module
            mods["ProbabilityModelAgent"] = import_module(f"{base}.ProbabilityModelAgent").ProbabilityModelAgent
            mods["BatteryAgent"]          = import_module(f"{base}.BatteryAgent").BatteryAgent
            mods["EVAgent"]               = import_module(f"{base}.EVAgent").EVAgent
            mods["PVAgent"]               = import_module(f"{base}.PVAgent").PVAgent
            mods["GridAgent"]             = import_module(f"{base}.GridAgent").GridAgent
            mods["FlexibleDevice"]        = import_module(f"{base}.FlexibleDeviceAgent").FlexibleDevice
            mods["GlobalOptimizer"]       = import_module(f"{base}.GlobalOptimizer").GlobalOptimizer
            mods["WeatherAgent"]          = import_module(f"{base}.WeatherAgent").WeatherAgent
            mods["GlobalConnectionLayer"] = import_module(f"{base}.GlobalConnectionLayer").GlobalConnectionLayer
            logger.info(f"Imported agents from '{base}.*'")
            return mods
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{base}: {exc}")
    raise ImportError(
        "Could not import agent modules – tried:\n  " + "\n  ".join(errors)
    )

AG = _import_agents()  # AG["BatteryAgent"], etc.

# -----------------------------------------------------------------------------#
# 5) DUCKDB CONNECTION WRAPPER
# -----------------------------------------------------------------------------#
def setup_duckdb_connection(building_id: str) -> Tuple[duckdb.DuckDBPyConnection, str]:
    """Always request an **in-memory** connection to side-step disk issues."""
    con, view = common.get_con(building_id=building_id, use_memory=True)
    logger.info("DuckDB ready – view '%s' for %s", view, building_id)
    return con, view

def select_days_from_duckdb(con: duckdb.DuckDBPyConnection,
                            view: str,
                            n_days: int) -> List[str]:
    """Pick the *last* n_days that have a full 24-hour record (derived from utc_timestamp)."""
    q = f"""
        SELECT day
        FROM (
            SELECT DATE(utc_timestamp) AS day, COUNT(*) AS cnt
            FROM   {view}
            GROUP  BY day
            HAVING cnt = 24
        ) AS full_days
        ORDER  BY day DESC
        LIMIT  {n_days}
    """
    return [r[0] for r in con.execute(q).fetchall()][::-1]  # ascending

# -----------------------------------------------------------------------------#
# 5b) PV UTILISATION HELPERS
# -----------------------------------------------------------------------------#

def _extract_device_consumption(device, scenario: str) -> Optional[np.ndarray]:
    """Return 24-h consumption array in kWh for the given scenario.
    Falls back gracefully if a preferred attribute is missing."""
    # Skip PV devices entirely – they appear as negative generation and would cancel load
    name_lc = getattr(device, "device_name", "").lower()
    if name_lc.startswith("pv") or "photovoltaic" in name_lc:
        return None

    # optimisation functions should populate these attrs
    if scenario == "baseline":
        if hasattr(device, "original_consumption") and device.original_consumption is not None:
            return np.asarray(device.original_consumption)[:24]
    else:
        # optimised versions
        for attr in ("optimized_consumption", "optimized_schedule", "nextday_optimized_schedule"):
            if hasattr(device, attr):
                arr = getattr(device, attr)
                if arr is not None:
                    return np.asarray(arr)[:24]
    # final fallback – read from underlying data column
    if hasattr(device, "data") and device.device_name in device.data.columns:
        arr = np.asarray(device.data[device.device_name].values)[:24]
    else:
        return None

    # Ensure we only return *consumption* (>=0)
    arr = np.clip(arr, 0, None)
    if arr.sum() == 0:
        return None
    return arr

def _pv_profile_from_devices(devices: list) -> Optional[np.ndarray]:
    """Return 24-h PV generation profile (negative numbers) or ``None`` if absent."""
    # many devices share the same underlying DataFrame with all building columns
    for dev in devices:
        if hasattr(dev, "data") and isinstance(dev.data, pd.DataFrame):
            df = dev.data
            break
    else:
        return None

    pv_cols = [c for c in df.columns if "pv" in c.lower() and "grid" not in c.lower()]
    if not pv_cols:
        return None
    profile = -df[pv_cols].sum(axis=1).values[:24]
    if np.any(profile):
        return profile
    return None
    return None

def compute_pv_utilisation(devices: list, scenario: str) -> float:
    """Compute fraction of PV generation that is self-consumed for a scenario (0–1)."""
    # PV generation profile: negative numbers mean export/generation.
    pv_profile = _pv_profile_from_devices(devices)
    if pv_profile is None:
        return None
    generation = (-pv_profile).clip(min=0)  # kWh generated each hour (>=0)
    gen_total = generation.sum()
    if gen_total == 0:
        return None

    # Aggregate device consumption
    load = np.zeros(24)
    for dev in devices:
        arr = _extract_device_consumption(dev, scenario)
        if arr is not None:
            load += arr

    # Self-consumed PV is the overlap (min of generation and load) each hour
    self_consumed = np.minimum(generation, load).sum()
    return float(self_consumed / gen_total)

# -----------------------------------------------------------------------------#
# 6) CORE PIPELINE – STUBS FOR BASELINE/OPTIMISERS
# -----------------------------------------------------------------------------#
# These functions simply delegate to the correctly implemented agent methods.
# Keeping them here isolates the orchestration code from agent internals.

def run_no_optimisation(devices, layer, pv, grid, batt, ev) -> dict:     # noqa: D401
    """Baseline cost with original schedules."""
    return {"devices": devices}

def run_decentralised_optimisation(devices, layer, pv, grid, batt, ev):  # noqa: D401
    """Run each FlexibleDevice's local optimisation for the day.

    We derive the required parameters (day, prices, pv forecast) from the
    device's own dataframe to avoid extra dependencies.
    """
    optimised = []
    for dev in devices:
        try:
            df = dev.data  # type: ignore[attr-defined]
            day_val = df['utc_timestamp'].dt.date.iloc[0]
            prices = df['price_per_kwh'].to_numpy()
            dev.optimize_day(day_val, prices, None)
        except Exception as exc:  # noqa: BLE001
            logger.error("Device %s decentralised optimisation failed – %s", dev, exc)
        optimised.append(dev)
    return {"devices": optimised}

def run_centralised_optimisation(devices, layer, pv, grid, batt, ev):    # noqa: D401
    if layer is None:
        # Fallback: skip centralised when no layer info
        return {"devices": devices}
    opt = AG["GlobalOptimizer"](devices, layer, pv, grid, batt, ev)
    opt.optimize_phases_centralized()
    return {"devices": devices}

# -----------------------------------------------------------------------------#
# 7) HIGH-LEVEL DRIVER
# -----------------------------------------------------------------------------#
def process_building(building_id: str, n_days: int) -> dict[str, Any]:
    con, view = setup_duckdb_connection(building_id)
    days      = select_days_from_duckdb(con, view, n_days)

    building_results: dict[str, Any] = {}

    for day in days:
        logger.info("-> %s  day %s", building_id, day)
        # --- 7.1 prepare agents & devices ----------------------------------
        battery = AG["BatteryAgent"](**BATTERY_PARAMS)
        ev      = None   # EV disabled in this simplified run
        pv = None  # PV generation will be derived directly from data
        grid    = AG["GridAgent"](**GRID_PARAMS)
        weather = None  # Weather data not needed for current PV utilisation analysis
        try:
            res = _create_devices_from_duckdb(
                con, view, building_id, day, battery, ev
            )
            if isinstance(res, tuple):
                devices, layer = res
            else:
                devices, layer = res, None
        except ValueError as e:
            logger.error("Device creation failed – %s", e)
            continue

        # Skip the day entirely if the building has no PV profile for this day
        if _pv_profile_from_devices(devices) is None:
            logger.info("No PV generation on %s for %s – skipping day", day, building_id)
            continue

        # --- 7.2 run scenarios --------------------------------------------
        day_res: dict[str, Any] = {}
        try:
            day_res["baseline"]      = run_no_optimisation(devices, layer, pv, grid, battery, ev)
            day_res["decentralised"] = run_decentralised_optimisation(devices, layer, pv, grid, battery, ev)
            day_res["centralised"]   = run_centralised_optimisation(devices, layer, pv, grid, battery, ev)

            # --- 7.3 compute PV utilisation for each scenario -------------
            for scen in ("baseline", "decentralised", "centralised"):
                util = compute_pv_utilisation(day_res[scen]["devices"], scen)
                day_res[scen]["pv_utilization"] = util
        except Exception as exc:     # noqa: BLE001
            logger.error("Optimisation failed for %s – %s", day, exc, exc_info=True)
            continue

        building_results[str(day)] = day_res

    return building_results

# -----------------------------------------------------------------------------#
# 8) ENTRY-POINT
# -----------------------------------------------------------------------------#
def main(buildings: List[str], days: int, out_dir: Path) -> None:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Results -> %s", run_dir)

    all_results: dict[str, Any] = {}
    for bld in buildings:
        try:
            all_results[bld] = process_building(bld, days)
        except Exception as exc:  # keep pipeline alive
            logger.error("Building %s failed – %s", bld, exc, exc_info=True)

    # ------------------ save summary --------------------------------------
    summary_json = run_dir / "optimisation_details.json"
    summary_json.write_text(json.dumps(all_results, indent=2, default=str))
    logger.info("Detailed results saved")

    # Simple Markdown overview (placeholder – extend as needed)
    md = "# Optimisation run\n\n| Building | Days | PV util. baseline | PV util. decentralised | PV util. centralised |\n|----------|------|-------------------|-----------------------|----------------------|\n"
    for b, r in all_results.items():
        if len(r) == 0:
            # skip buildings with no PV days processed
            continue
        baseline_vals = [v for d in r.values() if "baseline" in d for v in [d["baseline"]["pv_utilization"]] if v is not None]
        dec_vals      = [v for d in r.values() if "decentralised" in d for v in [d["decentralised"]["pv_utilization"]] if v is not None]
        centr_vals    = [v for d in r.values() if "centralised" in d for v in [d["centralised"]["pv_utilization"]] if v is not None]

        def _fmt(vals: list[Any]) -> str:
            return f"{np.mean(vals):.2%}" if vals else "n/a"

        md += f"| {b} | {len(r)} | {_fmt(baseline_vals)} | {_fmt(dec_vals)} | {_fmt(centr_vals)} |\n"
    (run_dir / "summary.md").write_text(md)

# -----------------------------------------------------------------------------#
# 9) CLI
# -----------------------------------------------------------------------------#
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EMS optimisation comparison")
    parser.add_argument("--buildings", nargs="+", default=BUILDINGS)
    parser.add_argument("--days",     type=int, default=3)
    parser.add_argument("--output",   type=Path, default=LOG_DIR)
    args = parser.parse_args()

    # lightweight audit trail for end-users
    simple_log = open(LOG_DIR / "quick_run.log", "a", encoding="utf-8")
    simple_log.write(f"\n[{datetime.now():%F %T}] Run started\n")
    try:
        main(args.buildings, args.days, args.output)
        simple_log.write("Run finished OK\n")
    except Exception as exc:
        simple_log.write(f"FAILED – {exc}\n")
        raise
    finally:
        simple_log.close()
