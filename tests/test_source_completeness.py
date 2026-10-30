# tests/test_source_completeness.py — the unread sheet that must STAY unread.
#
# Six of the PO workbooks carry a second sheet ("Capex", "Capex PO", "Dom Capital PO",
# "Capex deatils", "Capital PO", "Capex": 1,110 lines worth ₹158.48 Cr) in the identical
# 43-column schema, and the loader never opens it. That looks exactly like data silently
# dropped on the floor. I ingested it, and it was wrong.
#
# Every one of those rows has an EXACT twin in the main sheet — same order, material,
# quantity and value — and every order on a tab is on the main sheet too. The tabs are a
# convenience view of capital POs that are already in the extract. Ingesting them
# double-counts every capital line (on the first three tabs alone it inflated procurement
# from ₹649.91 Cr to ₹726.67 Cr, an 11.8% double-count).
#
# These tests exist so the next person who notices the unread tab does not "fix" it either.
from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.ai.warehouse import con
from app.etl import ingest
from app.etl.formulary import excel as X
from app.etl.formulary.discover import BuildError


def test_the_capex_tabs_are_duplicates_of_rows_already_ingested():
    """The load-bearing fact. If this ever fails, the tabs became real data.

    It reads HCG's raw PO workbooks, so it runs only when asked (HCG_RAW_TESTS=1, through the
    memory-guarded runner) and only where the extracts are. A line is keyed on plant,
    material, quantity, value and PO date — never on the order number, which fact_po does
    not keep — and only the lines fact_po keeps (not deleted, with a material) are checked.
    """
    if not os.environ.get("HCG_RAW_TESTS"):
        pytest.skip("reads HCG's raw PO workbooks: set HCG_RAW_TESTS=1 to run it")
    try:
        folder = ingest.find_source_dir(ingest.PO_DIR_RE, "'PO details ...'")
    except BuildError:
        pytest.skip("raw extracts not present in this checkout")

    def key(plant, material, qty, value, when):
        return (plant, material, round(float(qty or 0), 3), round(float(value or 0), 2),
                str(when)[:10])

    have = {key(*r) for r in con().execute(
        "SELECT plant, material, po_qty, total_value_wo_tax, po_date FROM fact_po").fetchall()}
    tabs, unseen = 0, []
    for path in ingest._workbooks(folder):
        with X.Workbook(path) as wb:
            for sheet in wb.sheet_names[1:]:
                try:
                    got = ingest.read_po_sheet(wb, sheet)
                except BuildError:                       # July's 'GL details': not order lines
                    continue
                if got is None:
                    continue
                tabs += 1
                lines = got[0]
                lines = lines[lines["material"].notna() & ~ingest.po_deleted(lines)]
                unseen += [k for k in (key(*r) for r in lines[
                    ["plant", "material", "po_qty", "total_value_wo_tax", "po_date"]]
                    .itertuples(index=False)) if k not in have]
    assert tabs == 6, f"expected the six capex tabs, read {tabs}"
    assert not unseen, (
        f"{len(unseen)} capex rows are NOT already in fact_po — they are real data now, "
        f"e.g. {unseen[:3]}. Re-open the ingestion question.")


def test_procurement_stays_at_the_verified_total():
    # Dec-25..Sep-26 after the import fix (app/etl/ingest.py, 2026-10-23): deleted lines (Del
    # Flag 'L', Rs 34.60 Cr in Aug-26 alone), lines without a material (services, civil
    # works, capital items without a code) and return lines (a negative quantity: Rs -0.33 Cr,
    # mostly Sep-26) are no longer counted. It was ₹649.91 Cr over Dec-25..May-26 with the
    # first two counted (ingesting the tabs moved that to ₹726.67 Cr), and ₹806.91 Cr with the
    # returns still in. It must not move again by accident.
    total = con().execute("SELECT SUM(total_value_wo_tax)/1e7 FROM fact_po").fetchone()[0]
    assert abs(total - 807.24) < 0.05, total


def test_the_kpi_agrees_with_the_fact_table():
    a = con().execute("SELECT SUM(total_value_wo_tax)/1e7 FROM fact_po").fetchone()[0]
    b = con().execute("SELECT SUM(purchase_value)/1e7 FROM kpi_purchase_value").fetchone()[0]
    assert abs(a - b) < 0.05, (a, b)


def test_capital_purchases_are_a_doc_type_not_a_table():
    # they are already inside the ₹807.24 Cr, identified by doc_type: ₹82.18 Cr of capital
    # lines that carry a material code (capital lines without one are not order lines of a
    # material and are left out with the services; the import report counts them)
    rows = dict(con().execute(
        "SELECT doc_type, ROUND(SUM(total_value_wo_tax)/1e7, 2) FROM fact_po "
        "WHERE doc_type LIKE '%apital%' GROUP BY 1").fetchall())
    assert abs(rows.get("Dom Capital PO", 0) - 82.18) < 0.05, rows


def test_the_reason_is_recorded_where_someone_would_change_it():
    # the first-sheet rule lives in the per-workbook reader that load_po calls
    src = (Path(__file__).resolve().parents[1] / "app" / "etl" / "ingest.py").read_text()
    po = src[src.index("def _read_po_file("):]
    assert "EXACT twin" in po and "double-count" in po
    assert "sheet_names[0]" in po and "sheet_name=None" not in po


def test_the_assistant_calls_capex_a_subset_not_an_addition():
    from app.ai.resolve import brief
    b = brief("how much did we spend on capex?")
    assert "doc_type" in b and "SUBSET" in b
