# tests/test_source_completeness.py — nothing in the source may be silently absent.
#
# `pd.read_excel(f, engine="openpyxl")` with no sheet_name reads sheet 0. Three of the six PO
# workbooks carry a SECOND sheet — "Capex", "Capex PO", "Dom Capital PO" — with the identical
# 43-column schema. 645 purchase-order lines worth ₹76.76 Cr had never been ingested, and
# nothing anywhere said so: not a filter, not a note, just a tab nobody opened.
#
# They are ingested now and TAGGED, so excluding capital purchases stays a decision rather
# than an accident. The KPIs deliberately keep their existing operational-only scope: pulling
# capex into them would have moved the headline from ₹649.91 Cr to ₹726.67 Cr and silently
# changed every dashboard number the client has reviewed.
from __future__ import annotations

from app.ai.warehouse import con


def test_capex_purchase_orders_are_ingested():
    rows = dict(con().execute(
        "SELECT po_type, COUNT(*) FROM fact_po GROUP BY 1").fetchall())
    assert rows.get("capex", 0) == 645
    assert rows.get("operational", 0) == 249_884


def test_the_two_scopes_are_distinguishable():
    v = dict(con().execute(
        "SELECT po_type, ROUND(SUM(total_value_wo_tax)/1e7, 2) FROM fact_po GROUP BY 1"
    ).fetchall())
    assert abs(v["operational"] - 649.91) < 0.05
    assert abs(v["capex"] - 76.76) < 0.05


def test_the_procurement_kpi_keeps_its_operational_scope():
    # the dashboard figure must not move because a sheet started being read
    total = con().execute(
        "SELECT SUM(purchase_value)/1e7 FROM kpi_purchase_value").fetchone()[0]
    assert abs(total - 649.91) < 0.05, total


def test_the_etl_reads_every_sheet_not_just_the_first():
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "app" / "etl" / "ingest.py").read_text()
    po = src[src.index("def load_po()"):]
    assert "sheet_name=None" in po, "load_po must read all sheets"
    assert "po_type" in po


def test_the_kpi_builder_filters_deliberately_not_accidentally():
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "app" / "etl" / "transforms.py").read_text()
    assert 'po_type"] == "operational"' in src


def test_the_assistant_knows_capex_is_a_separate_scope():
    from app.ai.resolve import brief
    b = brief("how much did we spend on capex?")
    assert "po_type" in b and "operational" in b
