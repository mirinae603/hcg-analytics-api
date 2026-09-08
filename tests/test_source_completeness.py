# tests/test_source_completeness.py — the unread sheet that must STAY unread.
#
# Three of the six PO workbooks carry a second sheet ("Capex", "Capex PO", "Dom Capital PO")
# holding 645 lines worth ₹76.76 Cr in the identical 43-column schema, and `pd.read_excel`
# with no sheet_name never opens it. That looks exactly like data silently dropped on the
# floor. I ingested it, and it was wrong.
#
# Every one of those 645 rows has an EXACT twin in the main sheet — same po_no, material,
# quantity and value — and all 486 distinct PO numbers already appear there. The tabs are a
# convenience view of capital POs that are already in the extract. Ingesting them inflates
# procurement from ₹649.91 Cr to ₹726.67 Cr, an 11.8% double-count, and counts `Dom Capital
# PO` once on the main sheet and twice on the tab.
#
# These tests exist so the next person who notices the unread tab does not "fix" it either.
from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.ai.warehouse import con

RAW = Path("/Users/shivanshdarshan/Documents/Profession/bidezy/codebase/Bidezy-2/"
           "PO details Dec 25 to May 26")
CAPEX_TABS = [("01.PO details for Dec 2025 extr 02.01.2026.xlsx", "Capex"),
              ("02.PO details for Jan 2026 extr 01.02.2026.xlsx", "Capex PO"),
              ("03.PO details for Feb 2026 extr 03.03.2026.xlsx", "Dom Capital PO")]


def test_the_capex_tabs_are_duplicates_of_rows_already_ingested():
    """The load-bearing fact. If this ever fails, the tabs became real data."""
    if not RAW.exists():
        return                                  # raw extracts not present in this checkout
    def norm(po_no, material, qty):
        """One key shape for both sides: SAP ids as digits, quantity as a float."""
        def ident(v):
            # None (DuckDB) and NaN (pandas) are the same absence and must key the same,
            # or every service PO — which carries no material — looks like a new row
            t = str(v).strip()
            if t in ("", "None", "nan", "NaN", "NaT", "<NA>"):
                return None
            return t[:-2] if t.endswith(".0") else t
        try:
            q = round(float(qty), 3)
        except (TypeError, ValueError):
            q = None
        return (ident(po_no), ident(material), q)

    main = con().execute("SELECT po_no, material, po_qty FROM fact_po").fetchall()
    have = {norm(*r) for r in main}
    unseen = []
    for fname, sheet in CAPEX_TABS:
        d = pd.read_excel(RAW / fname, sheet_name=sheet, engine="openpyxl")
        d.columns = [str(c).strip() for c in d.columns]
        for _, r in d.iterrows():
            k = norm(r.get("PO No"), r.get("Material"), r.get("PO Quantity"))
            if k not in have:
                unseen.append(k)
    assert not unseen, (
        f"{len(unseen)} capex rows are NOT already in fact_po — they are real data now, "
        f"e.g. {unseen[:3]}. Re-open the ingestion question.")


def test_procurement_stays_at_the_verified_total():
    # ingesting the tabs moved this to ₹726.67 Cr; it must not move again by accident
    total = con().execute("SELECT SUM(total_value_wo_tax)/1e7 FROM fact_po").fetchone()[0]
    assert abs(total - 649.91) < 0.05, total


def test_the_kpi_agrees_with_the_fact_table():
    a = con().execute("SELECT SUM(total_value_wo_tax)/1e7 FROM fact_po").fetchone()[0]
    b = con().execute("SELECT SUM(purchase_value)/1e7 FROM kpi_purchase_value").fetchone()[0]
    assert abs(a - b) < 0.05, (a, b)


def test_capital_purchases_are_a_doc_type_not_a_table():
    # they are already inside the ₹649.91 Cr, identified by doc_type
    rows = dict(con().execute(
        "SELECT doc_type, ROUND(SUM(total_value_wo_tax)/1e7, 2) FROM fact_po "
        "WHERE doc_type LIKE '%apital%' GROUP BY 1").fetchall())
    assert rows.get("Dom Capital PO", 0) > 100          # ₹121.75 Cr, already counted


def test_the_reason_is_recorded_where_someone_would_change_it():
    src = (Path(__file__).resolve().parents[1] / "app" / "etl" / "ingest.py").read_text()
    po = src[src.index("def load_po()"):]
    assert "EXACT twin" in po and "double-count" in po
    assert "sheet_name=None" not in po


def test_the_assistant_calls_capex_a_subset_not_an_addition():
    from app.ai.resolve import brief
    b = brief("how much did we spend on capex?")
    assert "doc_type" in b and "SUBSET" in b
