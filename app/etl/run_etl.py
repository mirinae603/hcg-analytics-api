"""End-to-end ETL: raw Excel -> curated facts + dimensions -> KPI aggregates.

Usage:
    python -m app.etl.run_etl              # full rebuild
    python -m app.etl.run_etl --stage curated   # only facts + dims
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from app.core.config import settings
from app.etl import ingest, dimensions

CURATED = Path(settings.CURATED_DIR)
KPI = Path(settings.KPI_DIR)


def _save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"[etl] wrote {path.name}: {len(df):,} rows")


def build_curated() -> dict:
    t0 = time.time()
    print("=== Ingesting raw extracts ===")
    consumption = ingest.load_consumption()
    inventory = ingest.load_inventory()
    grn = ingest.load_grn()
    po = ingest.load_po()

    print("=== Building dimensions ===")
    dim_plant = dimensions.build_dim_plant(po, inventory, consumption)
    dim_material = dimensions.build_dim_material(inventory, consumption)
    dim_sloc = dimensions.build_dim_sloc(inventory)
    dim_vendor = dimensions.build_dim_vendor(po, grn)
    dim_costcenter = dimensions.build_dim_costcenter(consumption)

    for name, df in {
        "fact_consumption": consumption,
        "fact_inventory": inventory,
        "fact_grn": grn,
        "fact_po": po,
        "dim_plant": dim_plant,
        "dim_material": dim_material,
        "dim_sloc": dim_sloc,
        "dim_vendor": dim_vendor,
        "dim_costcenter": dim_costcenter,
    }.items():
        _save(df, CURATED / f"{name}.parquet")

    report = {
        "consumption_rows": int(len(consumption)),
        "inventory_rows": int(len(inventory)),
        "grn_rows": int(len(grn)),
        "po_rows": int(len(po)),
        "plants": int(len(dim_plant)),
        "materials": int(len(dim_material)),
        "vendors": int(len(dim_vendor)),
        "cost_centers": int(len(dim_costcenter)),
        "months_consumption": sorted(consumption["month_num"].dropna().unique().tolist()),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (CURATED / "_dq_report.json").write_text(json.dumps(report, indent=2))
    print("=== Curated layer report ===")
    print(json.dumps(report, indent=2))
    return {
        "consumption": consumption, "inventory": inventory, "grn": grn, "po": po,
        "dim_plant": dim_plant, "dim_material": dim_material, "dim_sloc": dim_sloc,
        "dim_vendor": dim_vendor, "dim_costcenter": dim_costcenter,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["curated", "kpi", "all"], default="all")
    args = ap.parse_args()

    curated = build_curated()

    if args.stage in ("kpi", "all"):
        from app.etl import transforms  # imported lazily; built in Phase 2
        transforms.build_all(curated)


if __name__ == "__main__":
    main()
