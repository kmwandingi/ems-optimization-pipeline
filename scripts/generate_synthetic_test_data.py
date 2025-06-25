#!/usr/bin/env python3
"""Generate a small synthetic dataset for unit-testing the EMS pipeline.

The file is written to `processed_data/TEST_building_processed_data.parquet`
and contains exactly three consecutive days (72 rows, hourly resolution) with:
    • 5 device load columns (kWh):
        - washing_machine
        - dishwasher
        - tumble_dryer
        - heat_pump
        - ev_charger
    • pv_generation  (negative numbers = generation)
    • price_per_kwh  (simple TOU: 0.30 €/kWh 08-19h else 0.15 €/kWh)

This is sufficient for quick runs of both *decentralised* and *centralised*
optimisation – no phase-level optimisation is needed.

Run:
    python scripts/generate_synthetic_test_data.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta, timezone

# -----------------------------------------------------------------------------
OUT_DIR = Path(__file__).resolve().parents[1] / "processed_data"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "TEST_building_processed_data.parquet"

START = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)
HOURS = 72  # 3 days
idx = pd.date_range(start=START, periods=HOURS, freq="H", tz="UTC")

# ----- device profiles --------------------------------------------------------
np.random.seed(42)

def _spiky(base: float, prob_on: float) -> np.ndarray:
    """Return base kWh with sparse spikes (for washing, dishwasher, etc.)."""
    on = np.random.rand(HOURS) < prob_on
    return np.where(on, base, 0.0)

data = pd.DataFrame({
    "utc_timestamp": idx,
    # small household appliances ~1.0 kWh cycles
    "washing_machine": _spiky(1.0, 0.15),
    "dishwasher":      _spiky(1.2, 0.15),
    "tumble_dryer":    _spiky(1.5, 0.10),
    # heat pump modest continuous load with noise
    "heat_pump":       0.5 + 0.2 * np.sin(np.linspace(0, 3*np.pi, HOURS)) + 0.05*np.random.randn(HOURS),
    # overnight EV charging demand (~7 kW for 2 hours)
    "ev_charger":      np.where((idx.hour >= 0) & (idx.hour < 2), 7.0, 0.0),
    # PV generation profile (negative, peak at noon)
    "pv_generation":  -np.maximum(0, 3.0 * np.sin((idx.hour - 6) * np.pi / 12)),
})

# ensure no negatives in load columns
for col in ["washing_machine", "dishwasher", "tumble_dryer", "heat_pump", "ev_charger"]:
    data[col] = data[col].clip(lower=0.0)

# price column (simple TOU)
prices = np.where((idx.hour >= 8) & (idx.hour < 20), 0.30, 0.15)
data["price_per_kwh"] = prices

# -----------------------------------------------------------------------------
# Save – parquet keeps schema compact and is the format used by pipeline
# -----------------------------------------------------------------------------
print(f"Writing synthetic dataset → {OUT_FILE.relative_to(Path.cwd())}")
data.to_parquet(OUT_FILE, index=False)
print("Done – rows:", len(data))
