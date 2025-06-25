#!/usr/bin/env python3
"""
Generate validated optimisation comparison tables (no plots).

For each building we compute 24-hour KPIs for a configurable number of days
with two configurations:
    • With Battery ( + default EV )
    • No Battery  ( + default EV )

We rely exclusively on the existing `scripts.calculate_kpis.calculate_kpis`
function so we stay compatible with the current agent implementations
(battery, EV, PV, etc.).

Outputs (all written to project root):
    1) <building>_daily_savings_table.csv        – detailed per-day costs & savings
    2) all_buildings_savings_summary.csv         – one-row per building summary
    3) paper_style_building_summary.csv          – consolidated “paper-style” table

All tables are also printed to stdout for quick inspection.

Run directly:
    python scripts/generate_optimization_tables.py --days 50
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import pandas as pd

# ----------------------------------------------------------------------------
# Import KPI helper (add project paths first)
# ----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks"))  # for agent modules

from scripts.calculate_kpis import calculate_kpis  # noqa: E402

# ----------------------------------------------------------------------------
# CONSTANTS / CONFIG
# ----------------------------------------------------------------------------
BUILDINGS: List[str] = [
    "DE_KN_industrial3",
    "DE_KN_residential1",
    "DE_KN_residential2",
    "DE_KN_residential3",
    "DE_KN_residential4",
    "DE_KN_residential5",
    "DE_KN_residential6",
]

DEFAULT_DAYS = 30  # change with CLI

# ----------------------------------------------------------------------------
# CORE HELPERS
# ----------------------------------------------------------------------------

def run_two_configs(building_id: str, n_days: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return KPI DataFrames (no_battery, with_battery)."""
    kpi_no_batt   = calculate_kpis(building_id, n_days=n_days, use_battery=False, use_ev=True)
    kpi_with_batt = calculate_kpis(building_id, n_days=n_days, use_battery=True,  use_ev=True)
    return kpi_no_batt, kpi_with_batt


def build_daily_table(no_batt: pd.DataFrame, batt: pd.DataFrame) -> pd.DataFrame:
    """Merge the two KPI frames into a daily-savings table (centralised costs)."""
    merged = (no_batt[['day', 'centralised_cost']]
              .rename(columns={'centralised_cost': 'No Battery (€)'})
              .merge(batt[['day', 'centralised_cost']]
                     .rename(columns={'centralised_cost': 'With Battery (€)'}),
                     on='day', how='outer'))
    merged = merged.sort_values('day').reset_index(drop=True)
    merged['Difference (€)'] = merged['With Battery (€)'] - merged['No Battery (€)']
    merged['No Battery (%)'] = 100 * merged['No Battery (€)'] / merged['No Battery (€)']
    merged['With Battery (%)'] = 100 * merged['With Battery (€)'] / merged['No Battery (€)']
    merged['Improvement (%)'] = merged['With Battery (%)'] - merged['No Battery (%)']
    return merged


def summarise_daily_table(df: pd.DataFrame) -> Dict:
    """Return dict row with totals/averages for summary table."""
    return dict(
        no_battery_total_savings   = df['No Battery (€)'].sum(),
        with_battery_total_savings = df['With Battery (€)'].sum(),
        savings_improvement        = df['Difference (€)'].sum(),
        no_battery_pct             = df['No Battery (%)'].mean(),
        with_battery_pct           = df['With Battery (%)'].mean(),
        pct_improvement            = df['Improvement (%)'].mean(),
    )

# ----------------------------------------------------------------------------
# MAIN PIPELINE
# ----------------------------------------------------------------------------

def main(days: int = DEFAULT_DAYS):
    summary_rows = {}
    paper_rows   = []  # placeholder if we later extend with more metrics

    for b_id in BUILDINGS:
        print(f"\n{'='*80}\nProcessing {b_id} ({days} days)\n{'='*80}")
        nb_df, wb_df = run_two_configs(b_id, days)

        daily_tbl = build_daily_table(nb_df, wb_df)
        csv_daily = f"{b_id}_daily_savings_table.csv"
        daily_tbl.to_csv(csv_daily, index=False)
        print(f"Saved daily table → {csv_daily}")

        # display first/last 3 rows
        disp_cols = ['day', 'No Battery (€)', 'With Battery (€)', 'Difference (€)']
        print(daily_tbl[disp_cols].head(3).to_string(index=False))
        print("   ...")
        print(daily_tbl[disp_cols].tail(3).to_string(index=False))

        # summary row for overall comparison
        summary_rows[b_id] = dict(building_id=b_id, **summarise_daily_table(daily_tbl))

    # ---------------------------------------------------------------------
    # Overall building comparison
    # ---------------------------------------------------------------------
    summary_df = pd.DataFrame.from_dict(summary_rows, orient='index')
    summary_df = summary_df.round(2)
    print(f"\n{'='*80}\nOverall Building Comparison\n{'='*80}")
    print(summary_df.to_string(index=False))
    summary_df.to_csv('all_buildings_savings_summary.csv', index=False)
    print("✔ Saved → all_buildings_savings_summary.csv")

    # placeholder for paper-style table (if needed later)
    paper_df = pd.DataFrame(paper_rows)
    if not paper_df.empty:
        paper_df.to_csv('paper_style_building_summary.csv', index=False)
        print("✔ Saved → paper_style_building_summary.csv")


# ----------------------------------------------------------------------------
# CLI ENTRY
# ----------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate optimisation comparison tables (no plots)')
    parser.add_argument('--days', type=int, default=DEFAULT_DAYS, help='Number of days per building')
    args = parser.parse_args()
    main(days=args.days)
