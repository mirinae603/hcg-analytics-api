# tests/test_sales_material_codes.py — one product, one material code.
#
# pyxlsb reads every number as a float, so the four .xlsb OP months (Dec-25 to Mar-26) keyed
# KEYTRUDA as '101313.0' while the .xlsx months used '101313'. Every material-keyed sales
# table then held the product twice, and the '.0' half had no category. The trend the AI
# reported for KEYTRUDA's December was Rs 9.29 Cr instead of Rs 9.68 Cr.
#
# These tests run the repair on a tiny synthetic copy in tmp_path. They never read or write
# app/data, and deliberately do not assert that the real parquets are clean yet: the repair
# is run by hand (see RUNBOOK_HCG.md), after which these same rules hold for the real files.
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import pytest

from app.etl import repair_sales_codes as R

KEYTRUDA = "101313"

_ROWS = [
    # (extra key, material, desc, group, revenue, cost, qty, lines)
    ("2025-12", "101313", "KEYTRUDA 100MG INJ VIAL", "M065-INJECTIONS", 100.0, 60.0, 2.0, 1.0),
    ("2025-12", "101313.0", "KEYTRUDA 100MG INJ VIAL.", "", 50.0, 30.0, 1.0, 1.0),
    ("2026-01", "101313.0", "KEYTRUDA 100MG INJ VIAL.", "", 40.0, 20.0, 1.0, 1.0),
    ("2026-01", "200001", "PARACETAMOL TAB", "M001-TABLETS", 10.0, 5.0, 4.0, 2.0),
    # a code that only ever appeared in float form: still merged, and given its group
    ("2026-02", "300002.0", "SYRINGE 5ML", "", 7.0, 3.0, 10.0, 3.0),
]
_EXTRA = {"sales_by_material_month": "period", "sales_by_material_hospital": "hospital",
          "sales_by_material_mfr": "manufacturer"}
_DIM = {"101313": "M065-INJECTIONS", "200001": "M001-TABLETS", "300002": "M020-CONSUMABLES"}


def _write_tables(root: Path) -> tuple[Path, Path]:
    kpi = root / "kpi"
    kpi.mkdir()
    for name in R.TABLES:
        recs = []
        for key, mat, desc, grp, rev, cost, qty, lines in _ROWS:
            row = {"material": mat, "desc": desc, "group": grp,
                   "revenue": rev, "cost": cost, "qty": qty, "lines": lines}
            if name in _EXTRA:
                # Same column order ingest_sales writes: the extra key first, except the
                # month table, which puts `period` after `material`.
                val = key if name == "sales_by_material_month" else "KABHK"
                row = ({"material": mat, "period": val, **{k: v for k, v in row.items() if k != "material"}}
                       if name == "sales_by_material_month" else {_EXTRA[name]: val, **row})
            recs.append(row)
        pd.DataFrame(recs).to_parquet(kpi / f"{name}.parquet", index=False)
    dim = root / "dim_material.parquet"
    pd.DataFrame({"material": list(_DIM), "material_group": list(_DIM.values())}).to_parquet(dim, index=False)
    return kpi, dim


def _read(path: Path):
    c = duckdb.connect()
    return c.execute(f"SELECT * FROM read_parquet('{path}')").df()


def _schema(path: Path):
    c = duckdb.connect()
    return [(r[0], r[1]) for r in c.execute(f"DESCRIBE SELECT * FROM read_parquet('{path}')").fetchall()]


def test_a_dry_run_changes_nothing(tmp_path):
    kpi, dim = _write_tables(tmp_path)
    before = {p.name: p.read_bytes() for p in kpi.iterdir()}
    out = R.repair(kpi_dir=str(kpi), dim_path=str(dim), write=False, backup_root=str(tmp_path / "bk"))
    assert {p.name: p.read_bytes() for p in kpi.iterdir()} == before
    assert not (tmp_path / "bk").exists()
    # the plan it reports: 5 rows become 3 in the 6-month total, 4 in the month table
    assert out["sales_by_material"] == (5, 3)
    assert out["sales_by_material_month"] == (5, 4)


def test_the_repair_merges_the_float_code_into_the_clean_one(tmp_path):
    kpi, dim = _write_tables(tmp_path)
    R.repair(kpi_dir=str(kpi), dim_path=str(dim), write=True, backup_root=str(tmp_path / "bk"))

    total = _read(kpi / "sales_by_material.parquet").set_index("material")
    assert not any(m.endswith(".0") for m in total.index)
    assert total.loc[KEYTRUDA, "revenue"] == 190.0          # 100 + 50 + 40, one row
    assert total.loc[KEYTRUDA, "group"] == "M065-INJECTIONS"
    assert total.loc[KEYTRUDA, "desc"] == "KEYTRUDA 100MG INJ VIAL"   # the clean row's spelling
    assert total.loc["300002", "group"] == "M020-CONSUMABLES"        # float-only code, now grouped

    month = _read(kpi / "sales_by_material_month.parquet")
    k = month[month.material == KEYTRUDA].set_index("period")["revenue"].to_dict()
    assert k == {"2025-12": 150.0, "2026-01": 40.0}      # December now carries its OP half


@pytest.mark.parametrize("name", list(R.TABLES))
def test_totals_columns_and_keys_survive_the_repair(tmp_path, name):
    kpi, dim = _write_tables(tmp_path)
    path = kpi / f"{name}.parquet"
    before, schema = _read(path), _schema(path)
    R.repair(kpi_dir=str(kpi), dim_path=str(dim), write=True, backup_root=str(tmp_path / "bk"))
    after = _read(path)

    for m in ("revenue", "cost", "qty", "lines"):
        assert after[m].sum() == pytest.approx(before[m].sum())
    assert _schema(path) == schema                       # same columns, order and types
    keys = R.TABLES[name] + ["material"]
    assert not after.duplicated(subset=keys).any()
    assert not after["material"].str.contains(r"\.0+$", regex=True).any()


def test_write_backs_up_the_originals_first(tmp_path):
    kpi, dim = _write_tables(tmp_path)
    originals = {p.name: p.read_bytes() for p in kpi.iterdir()}
    R.repair(kpi_dir=str(kpi), dim_path=str(dim), write=True, backup_root=str(tmp_path / "bk"))
    (bdir,) = list((tmp_path / "bk").iterdir())
    assert {p.name: p.read_bytes() for p in bdir.iterdir()} == originals
    assert not list(kpi.glob("*.tmp"))                    # nothing half-written left behind


def test_an_unexpected_column_stops_the_run_before_anything_is_written(tmp_path):
    kpi, dim = _write_tables(tmp_path)
    p = kpi / "sales_by_material.parquet"
    df = pd.read_parquet(p)
    df["surprise"] = 1.0
    df.to_parquet(p, index=False)
    before = {q.name: q.read_bytes() for q in kpi.iterdir()}
    with pytest.raises(SystemExit):
        R.repair(kpi_dir=str(kpi), dim_path=str(dim), write=True, backup_root=str(tmp_path / "bk"))
    assert {q.name: q.read_bytes() for q in kpi.iterdir()} == before


def test_the_default_paths_are_the_real_tables_and_dry_run_is_the_default():
    # Guards the CLI contract the RUNBOOK documents: no flag = dry run.
    src = Path(R.__file__).read_text()
    assert '"--write" in sys.argv' in src
    assert R.KPI.endswith(str(Path("app") / "data" / "kpi"))


# ── the ETL itself must not produce the split again ───────────────────────────────────────
def _ingest_src() -> str:
    return (Path(__file__).resolve().parents[1] / "app" / "etl" / "ingest_sales.py").read_text()


def test_ingest_sales_normalises_material_codes():
    src = _ingest_src()
    assert "norm_code(" in src
    assert 'str(g("mat"))' not in src          # the line that wrote '101313.0'


def test_ingest_sales_aborts_on_float_shaped_codes():
    src = _ingest_src()
    assert "float-formatted" in src
    from app.etl.ingest_sales import float_codes
    assert float_codes(["101313", "101313.0", "200001.00", "M-1.0A"]) == ["101313.0", "200001.00"]


def test_ingest_sales_has_no_hard_coded_developer_path():
    src = _ingest_src()
    assert "/Users/" not in src
    assert "SALES_RAW" in src


def test_norm_code_handles_both_readers():
    from app.etl.formulary.classify import norm_code
    assert norm_code(101313.0) == "101313"      # pyxlsb
    assert norm_code(101313) == "101313"        # openpyxl
    assert norm_code(" 101313.0 ") == "101313"
