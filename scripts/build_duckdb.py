#!/usr/bin/env python
"""
I create (or refresh) a DuckDB file 'ems_data.duckdb'
and register every *_processed_data.parquet and priors parquet
as *views* – zero copy, so disk footprint stays minimal.
Run me once, or whenever new parquet files appear.
"""
from pathlib import Path
import duckdb      # pip already available in repo image

DB_PATH   = "ems_data.duckdb"
DATA_DIR  = Path(__file__).parent.parent / "notebooks" / "data"
PRIOR_DIR = Path(__file__).parent.parent / "notebooks" / "probabilities"

con = duckdb.connect(DB_PATH)

# ── register building parquet files ──
for p in sorted(DATA_DIR.glob("*_processed_data.parquet")):
    view = p.stem            # e.g. DE_KN_residential1_processed_data
    con.execute(f"""
        CREATE OR REPLACE VIEW {view} AS
        SELECT * FROM read_parquet('{p.as_posix()}');
    """)
    print("✓ view", view)

# ── register priors (device_hourly_probabilities) ──
for p in sorted(PRIOR_DIR.glob("*.parquet")):
    view = p.stem
    con.execute(f"""
        CREATE OR REPLACE VIEW {view} AS
        SELECT * FROM read_parquet('{p.as_posix()}');
    """)
    print("✓ view", view)

con.close()
print("🎉  DuckDB ready:", DB_PATH)