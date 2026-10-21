"""Merge the split material codes ('101313' and '101313.0') in the sales parquets.

Why this exists: until ingest_sales used norm_code, the four OP months read from .xlsb
files (Dec-25 to Mar-26) keyed every product by the float form of its code, '101313.0',
while the .xlsx months used '101313'. So each material-keyed sales table holds most
products twice, and the '.0' half has no category because it never matched dim_material.
KEYTRUDA, for example, is two rows (Rs 45.77 Cr and Rs 1.71 Cr) instead of one Rs 47.48 Cr
row, and its monthly trend is understated for Dec to Mar.

Every column in these tables is a plain sum, so merging the two codes and adding them up
gives exactly what a full re-run of ingest_sales would produce, without re-reading
~1.2 GB of patient-level billing files. Only these four tables carry a material key:

    sales_by_material           key: material
    sales_by_material_month     key: period, material
    sales_by_material_hospital  key: hospital, material
    sales_by_material_mfr       key: manufacturer, material

The other sales parquets (totals, monthly, by hospital, by manufacturer) have no material
key and are left alone.

The group comes from dim_material on the cleaned code, the same lookup ingest_sales does,
falling back to a group the table already had. The description prefers the clean row's.

Run from the repo root. A dry run (the default) changes nothing and prints what would:
    python -m app.etl.repair_sales_codes
    python -m app.etl.repair_sales_codes --write

--write copies the four originals to a timestamped folder under SALES_REPAIR_BACKUP
(default ~/Documents/HCG/hcg-work/snapshots) first, writes each repaired table to
'<name>.parquet.tmp' and then swaps it in, so a crash never leaves a half-written file.
Every table must pass the checks (totals unchanged, no '.0' codes left, no duplicate keys)
before anything is written; one failure stops the run with nothing changed.
"""
from __future__ import annotations

import os
import shutil
import sys
import time

import duckdb

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KPI = os.path.join(ROOT, "app", "data", "kpi")
DIM = os.path.join(ROOT, "app", "data", "curated", "dim_material.parquet")
BACKUP = os.path.expanduser(os.environ.get("SALES_REPAIR_BACKUP") or "~/Documents/HCG/hcg-work/snapshots")

# table -> the key columns besides `material`
TABLES = {
    "sales_by_material": [],
    "sales_by_material_month": ["period"],
    "sales_by_material_hospital": ["hospital"],
    "sales_by_material_mfr": ["manufacturer"],
}
MEASURES = ("revenue", "cost", "qty", "lines")
# A rupee of drift on hundreds of crores is rounding in the float sums, not a lost row.
TOLERANCE = 1.0


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _lit(path: str) -> str:
    return "'" + path.replace("'", "''") + "'"


def repaired_sql(c, path: str, extra: list, dim_path: str) -> str:
    """SELECT that returns the table at `path` with its codes merged: same columns, same
    order, same types."""
    desc = c.execute(f"DESCRIBE SELECT * FROM read_parquet({_lit(path)})").fetchall()
    types = {r[0]: r[1] for r in desc}
    cols = [r[0] for r in desc]
    expected = set(extra) | {"material", "desc", "group", *MEASURES}
    if set(cols) != expected:
        # A column this script does not know how to merge would be silently dropped or
        # summed wrongly. Stop and let a person look.
        raise SystemExit(f"[repair] {os.path.basename(path)}: unexpected columns {sorted(cols)}; "
                         f"expected {sorted(expected)}. Nothing was written.")
    keys = "".join(f"{_q(x)}, " for x in extra)
    sums = ", ".join(f"CAST(sum({_q(m)}) AS {types[m]}) AS {_q(m)}" for m in MEASURES)
    order = ", ".join([_q(x) for x in extra] + ["material"])
    return f"""
    WITH src AS (
        SELECT *, regexp_replace(trim(material), '\\.0+$', '') AS m,
               regexp_matches(trim(material), '\\.0+$') AS was_float
        FROM read_parquet({_lit(path)})),
    dm AS (
        SELECT regexp_replace(trim(CAST(material AS VARCHAR)), '\\.0+$', '') AS m,
               any_value(material_group) AS dim_group
        FROM read_parquet({_lit(dim_path)}) GROUP BY 1),
    agg AS (
        SELECT {keys}m AS material,
               first("desc" ORDER BY was_float, length("desc") DESC) AS "desc",
               max(NULLIF("group", '')) AS seen_group,
               {sums}
        FROM src GROUP BY {keys}m),
    fin AS (
        SELECT a.* EXCLUDE (seen_group),
               CAST(COALESCE(dm.dim_group, a.seen_group, '') AS {types['group']}) AS "group"
        FROM agg a LEFT JOIN dm ON dm.m = a.material)
    SELECT {', '.join(_q(x) for x in cols)} FROM fin ORDER BY {order}"""


def _check(c, name: str, path: str, q: str, extra: list) -> tuple:
    """(rows before, rows after, revenue) after asserting the repair loses nothing."""
    agg = ", ".join(f"sum({_q(m)})" for m in MEASURES)
    b = c.execute(f"SELECT count(*), {agg} FROM read_parquet({_lit(path)})").fetchone()
    a = c.execute(f"SELECT count(*), {agg}, "
                  f"count(*) FILTER (WHERE regexp_matches(material, '\\.0+$')) FROM ({q})").fetchone()
    key = ", ".join([_q(x) for x in extra] + ["material"])
    dup = c.execute(f"SELECT count(*) FROM (SELECT {key} FROM ({q}) "
                    f"GROUP BY ALL HAVING count(*) > 1)").fetchone()[0]
    drift = [MEASURES[i - 1] for i in range(1, 5) if abs((b[i] or 0) - (a[i] or 0)) >= TOLERANCE]
    if drift or a[5] or dup:
        raise SystemExit(f"[repair] {name}: check failed (totals moved: {drift or 'no'}; "
                         f"'.0' codes left: {a[5]}; duplicate keys: {dup}). Nothing was written.")
    return b[0], a[0], a[1] or 0.0


def repair(kpi_dir: str = KPI, dim_path: str = DIM, write: bool = False,
           backup_root: str = BACKUP, tables: dict = TABLES) -> dict:
    """Check every table, then (only with write=True) back up and replace them.
    Returns {table: (rows before, rows after)}."""
    c = duckdb.connect()
    # This machine runs the dashboards too; the largest table is ~70k rows.
    c.execute("SET memory_limit='400MB'; SET threads=2")
    plan, out = [], {}
    for name, extra in tables.items():
        p = os.path.join(kpi_dir, name + ".parquet")
        if not os.path.exists(p):
            raise SystemExit(f"[repair] {p} not found. Nothing was written.")
        q = repaired_sql(c, p, extra, dim_path)
        before, after, revenue = _check(c, name, p, q, extra)
        print(f"[repair] {name}: rows {before:,} -> {after:,}; revenue Rs {revenue / 1e7:,.2f} Cr unchanged",
              flush=True)
        plan.append((name, p, q, after))
        out[name] = (before, after)
    if not write:
        print("[repair] dry run: nothing written. Re-run with --write to apply.")
        return out

    # Back up all four before touching any, so a failure part-way leaves a full set to
    # restore from.
    bdir = os.path.join(backup_root, "sales-parquet-" + time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(bdir, exist_ok=True)
    for _, p, _, _ in plan:
        shutil.copy2(p, bdir)
    for name, p, q, after in plan:
        tmp = p + ".tmp"   # not '*.parquet', so the AI warehouse never loads it
        c.execute(f"COPY ({q}) TO {_lit(tmp)} (FORMAT PARQUET)")
        n = c.execute(f"SELECT count(*) FROM read_parquet({_lit(tmp)})").fetchone()[0]
        if n != after:
            os.remove(tmp)
            raise SystemExit(f"[repair] {name}: wrote {n:,} rows, expected {after:,}. Originals are in {bdir}.")
        os.replace(tmp, p)
    print(f"[repair] written. Originals backed up to {bdir}")
    print("[repair] Next: restart the API (or POST /admin/refresh-data) so the dashboards and the "
          "AI warehouse re-read the tables, then commit the four parquets.")
    return out


if __name__ == "__main__":
    repair(write="--write" in sys.argv[1:])
