# tests/test_tax_columns.py — the columns worth adding, and the ones that were not.
#
# 86 source columns were never read by the ETL. Adding all of them takes fact_grn from 24 to
# 69 columns and puts five plausible money columns where there were two — and today's wrong
# answers are already mostly wrong-column answers. So the set was chosen on evidence:
# density measured per column, dead ones excluded.
#
# EXCLUDED after measuring: Selling Price and Total Sellling price (100% filled, 100% ZERO),
# Free Qty and Customer (0% filled), Formulary on GRN (all zero — the real one is on
# dim_material), and PR No / PR Date / PO TAT (4-6% filled; a requisition-cycle answer built
# on 6% of rows misleads more than it informs).
from __future__ import annotations

from app.ai.resolve import brief
from app.ai.warehouse import con


def test_the_gst_split_is_queryable():
    c, s, i = con().execute(
        "SELECT SUM(cgst_value)/1e7, SUM(sgst_value)/1e7, SUM(igst_value)/1e7 "
        "FROM fact_po").fetchone()
    assert abs((c + s + i) - 48.79) < 0.1, (c, s, i)


def test_tax_does_not_disturb_the_ex_tax_spend_figure():
    # the number every dashboard shows must not move because tax columns arrived
    total = con().execute("SELECT SUM(total_value_wo_tax)/1e7 FROM fact_po").fetchone()[0]
    assert abs(total - 649.91) < 0.05, total


def test_receipt_side_tax_is_there_too():
    t = con().execute("SELECT SUM(tax_amount)/1e7 FROM fact_grn").fetchone()[0]
    assert t > 1, t


def test_discounts_are_queryable_though_sparse():
    v, n = con().execute(
        "SELECT SUM(discount_value)/1e7, COUNT(*) FILTER (WHERE discount_value > 0) "
        "FROM fact_grn").fetchone()
    assert v > 0 and n > 1000


def test_dead_columns_were_not_imported():
    cols = {r[0] for r in con().execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name IN ('fact_grn','fact_po','fact_inventory')").fetchall()}
    for dead in ("selling_price", "free_qty", "total_sellling_price", "customer"):
        assert dead not in cols, f"{dead} is 100% empty or zero — it should not be here"


def test_sparse_requisition_columns_were_not_imported():
    cols = {r[0] for r in con().execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = 'fact_grn'").fetchall()}
    for sparse in ("pr_no", "pr_date", "po_tat"):
        assert sparse not in cols, f"{sparse} is filled on 4-6% of rows"


# ── the guidance that keeps tax out of the spend figure ─────────────────────────────────
def test_a_spend_question_is_told_the_basis_is_ex_tax():
    assert "EX-TAX" in brief("what is our total procurement spend?")


def test_a_tax_question_resolves_at_all():
    # it names no entity, measure or grain, so the brief used to come back EMPTY and the
    # guidance never reached the model
    b = brief("how much GST did we pay?")
    assert b and "EX-TAX" in b


def test_mrp_is_flagged_as_neither_revenue_nor_spend():
    b = brief("what is the MRP value received?")
    assert "total_mrp_value" in b and "Never" in b


# ── the scope guard was inverted for cities ─────────────────────────────────────────────
def test_an_alias_does_not_count_as_a_filter():
    # `SELECT SUM(total_value_wo_tax) AS bangalore_hospital_procurement FROM fact_po`
    # contains "bangalore", passed the scope check, and reported the whole network's
    # ₹649.91 Cr as one city's spend. An alias is a label the model chose, not a filter.
    from app.ai.scope import missing_entity_scope
    assert missing_entity_scope(
        "SELECT SUM(total_value_wo_tax) AS bangalore_hospital_procurement FROM fact_po",
        ["Bangalore"])


def test_filtering_a_city_by_plant_code_counts_as_scoped():
    # city names live only in dim_plant, so a correct query filters by CODE and never says
    # "Bangalore" — the old check blocked exactly the right query and allowed the wrong one
    from app.ai.scope import missing_entity_scope
    assert missing_entity_scope(
        "SELECT SUM(total_value_wo_tax) FROM fact_po "
        "WHERE plant IN ('HC01','HC05','HC06','HC40')", ["Bangalore"]) is None


def test_an_item_filter_still_satisfies_it():
    from app.ai.scope import missing_entity_scope
    assert missing_entity_scope(
        "SELECT * FROM t WHERE material_desc LIKE '%KEYTRUDA%'", ["KEYTRUDA"]) is None


def test_a_wholly_unscoped_query_is_still_refused():
    from app.ai.scope import missing_entity_scope
    assert missing_entity_scope("SELECT SUM(qty) FROM t", ["KEYTRUDA"])
