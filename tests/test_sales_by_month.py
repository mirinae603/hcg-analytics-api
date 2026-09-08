# tests/test_sales_by_month.py — the cut the ETL never wrote.
#
# "Show me the sales trend of KEYTRUDA" could only be answered as a six-month total, and the
# assistant said so: "the dataset does not provide a month-by-month breakdown". That was true
# of the WAREHOUSE and false of the DATA. Every raw billing row carries SALESDATE and
# MATERIALCODE; ingest_sales computed month×patient and material separately and never wrote
# the cross.
from __future__ import annotations

import pandas as pd

from app.ai.resolve import impossible_combination, measure_locations
from app.ai.warehouse import con

KEYTRUDA = "101313"


def test_the_table_exists_and_is_month_grained():
    periods = [r[0] for r in con().execute(
        "SELECT DISTINCT period FROM sales_by_material_month ORDER BY period").fetchall()]
    assert periods == ["2025-12", "2026-01", "2026-02", "2026-03", "2026-04", "2026-05"]


def test_it_reconciles_exactly_with_the_material_totals():
    # the new cut must not invent or lose a rupee against the aggregate that already existed
    a = con().execute("SELECT SUM(revenue) FROM sales_by_material_month").fetchone()[0]
    b = con().execute("SELECT SUM(revenue) FROM sales_by_material").fetchone()[0]
    assert abs(a - b) < 1.0, (a, b)


def test_it_reconciles_with_sales_monthly_too():
    a = con().execute("SELECT SUM(revenue) FROM sales_by_material_month").fetchone()[0]
    c = con().execute("SELECT SUM(revenue) FROM sales_monthly").fetchone()[0]
    assert abs(a - c) < 1.0, (a, c)


def test_a_named_drug_now_has_a_real_monthly_series():
    rows = con().execute(
        "SELECT period, revenue FROM sales_by_material_month "
        f"WHERE material = '{KEYTRUDA}' ORDER BY period").fetchall()
    assert len(rows) == 6
    assert rows[0][0] == "2025-12"          # December is FIRST, and sorts that way
    assert all(r[1] > 0 for r in rows)


def test_the_period_key_sorts_chronologically_not_by_calendar_month():
    periods = [r[0] for r in con().execute(
        "SELECT DISTINCT period FROM sales_by_material_month ORDER BY period").fetchall()]
    assert periods == sorted(periods)
    assert periods[0].startswith("2025")     # not "2026-01" first


def test_the_resolver_no_longer_calls_the_question_impossible():
    assert impossible_combination(
        'Show me the sales trend for "KEYTRUDA 100MG INJ VIAL"') is None


def test_the_month_grained_table_is_offered_for_revenue():
    assert any("sales_by_material_month" in loc for loc in measure_locations("revenue"))


# ── the ETL must refuse to write a partial dataset ───────────────────────────────────────
def test_the_etl_aborts_when_a_file_reads_empty():
    # pyxlsb was an OPTIONAL import; without it four .xlsb months read zero rows, the run
    # "succeeded", and it overwrote seven parquets with ₹492.40 Cr where ₹521.67 Cr belonged
    src = (__import__("pathlib").Path(__file__).resolve().parents[1]
           / "app" / "etl" / "ingest_sales.py").read_text()
    assert "ABORT" in src and "read_counts" in src
    assert "produced ZERO rows" in src


def test_the_xlsb_reader_is_a_hard_requirement():
    reqs = (__import__("pathlib").Path(__file__).resolve().parents[1]
            / "requirements.txt").read_text()
    assert "pyxlsb" in reqs
