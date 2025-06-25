#!/usr/bin/env python
"""
Collect performance metrics for all buildings by running scripts/01_run.py.

Usage (default 3 days, battery on):
    python scripts/collect_performance.py

Optional flags:
    --n_days 5               # number of days to optimise
    --battery on|off         # battery mode
    --ev on|off              # ev mode
    --buildings B1 B2 ...    # explicit list of building IDs; if omitted, scan data directory

The script will:
1. Discover processed parquet files ( *_processed_data.parquet ) in data/ and notebooks/data/
2. Execute 01_run.py sequentially for each building
3. After each run, read the <building>_metrics.json dropped by 01_run.py
4. Aggregate into results/performance_summary.csv and .md
"""

import argparse
import subprocess
import sys
import json
import os
from pathlib import Path
from datetime import datetime
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent  # project root
DATA_DIRS = [ROOT / "data", ROOT / "notebooks" / "data"]
OUTPUT_DIR = ROOT / "results" / "output"
SUMMARY_CSV = ROOT / "results" / "performance_summary.csv"
SUMMARY_MD = ROOT / "results" / "performance_summary.md"


def discover_buildings():
    """Return set of building IDs with *_processed_data.parquet available."""
    ids = set()
    for d in DATA_DIRS:
        if d.exists():
            for parquet in d.glob("*_processed_data.parquet"):
                bid = parquet.stem.replace("_processed_data", "")
                ids.add(bid)
    return sorted(ids)


def run_building(bid: str, n_days: int, battery: str, ev: str):
    """Run 01_run.py for one building and return metrics dict (or None on fail)."""
    cmd = [sys.executable, "scripts/01_run.py", "--building", bid, "--n_days", str(n_days), "--battery", battery]
    if ev:
        cmd.extend(["--ev", ev])
    print("\n==============================")
    print(f"Running optimisation for {bid} ...")
    print("==============================")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(
        cmd,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    # Echo output for debugging
    print(result.stdout)
    if result.returncode != 0:
        print(f"❌ optimisation failed for {bid}")
        return None
    metrics_file = OUTPUT_DIR / f"{bid}_metrics.json"
    if metrics_file.exists():
        with open(metrics_file, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        metrics["building_id"] = bid
        metrics["run_timestamp"] = datetime.now().isoformat(timespec='seconds')
        return metrics
    else:
        print(f"⚠ metrics file not found for {bid}: {metrics_file}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_days", type=int, default=3)
    ap.add_argument("--battery", choices=["on", "off"], default="on")
    ap.add_argument("--ev", choices=["on", "off"], default="off")
    ap.add_argument("--buildings", nargs="*", help="Explicit list of building IDs to process (default: discover)")
    args = ap.parse_args()

    buildings = args.buildings or discover_buildings()
    if not buildings:
        print("No buildings found. Ensure parquet files exist in data/ or notebooks/data/")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = []
    for bid in buildings:
        metrics = run_building(bid, args.n_days, args.battery, args.ev)
        if metrics:
            summary.append(metrics)

    if not summary:
        print("No successful runs -> summary not written")
        return

    df = pd.DataFrame(summary)
    df.to_csv(SUMMARY_CSV, index=False)
    with open(SUMMARY_MD, "w", encoding="utf-8") as md:
        md.write(df.to_markdown(index=False))
    print(f"\n✓ Performance summary written to:\n  - {SUMMARY_CSV}\n  - {SUMMARY_MD}")


if __name__ == "__main__":
    main()
