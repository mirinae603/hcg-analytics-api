# tests/test_scope.py — the scope contract. See app/ai/scope.py for the three live wrong
# answers that motivated it; these lock in the checks that make them impossible.
from __future__ import annotations

from app.ai.scope import missing_entity_scope, sql_literals


KEYTRUDA = ["101313", "KEYTRUDA 100MG INJ VIAL", "KEYTRUDA", "VIAL"]

# The exact query that produced "KEYTRUDA's" December revenue of ₹89.52 Cr. It is
# sales_monthly's total for EVERY material; the item's real revenue across all six
# months is ₹47.48 Cr. sales_monthly has no `material` column at all, so the filter
# could not be expressed and was silently dropped.
SCOPE_LOSS = (
    "SELECT month, SUM(revenue) AS revenue, SUM(revenue - cost) AS margin "
    "FROM sales_monthly WHERE month IN ('2025-12','2026-01') GROUP BY month ORDER BY month"
)


def test_catches_the_dropped_entity_filter():
    msg = missing_entity_scope(SCOPE_LOSS, KEYTRUDA)
    assert msg and "not scoped" in msg
    # the message has to name what it should have been scoped to, or it cannot be acted on
    assert "101313" in msg


def test_a_properly_scoped_query_passes_on_the_code():
    assert missing_entity_scope("SELECT SUM(revenue) FROM sales_by_material WHERE material='101313'", KEYTRUDA) is None


def test_a_properly_scoped_query_passes_on_the_name():
    assert missing_entity_scope(
        "SELECT * FROM sales_by_material_hospital WHERE material_desc ILIKE '%KEYTRUDA%'", KEYTRUDA) is None


def test_scoping_via_a_join_or_subquery_still_passes():
    # the check asks "is this entity mentioned", never "is it mentioned in a WHERE" —
    # a guard that dictates query SHAPE would reject legitimate SQL and get switched off
    assert missing_entity_scope(
        "SELECT h.hospital, SUM(h.revenue) FROM sales_by_material_hospital h "
        "WHERE h.material IN (SELECT material FROM dim_material WHERE material='101313') GROUP BY 1",
        KEYTRUDA) is None


def test_no_bound_entity_means_no_constraint():
    # a genuinely portfolio-wide question binds nothing and must stay unrestricted
    assert missing_entity_scope(SCOPE_LOSS, []) is None


def test_short_tokens_cannot_satisfy_the_check():
    # a two-letter fragment would match almost any SQL by accident and make the guard a no-op
    assert missing_entity_scope("SELECT 1 FROM t", ["AB"]) is not None


# ── literal extraction: the input to the wrong-dimension hunt ──────────────────
def test_literals_ignore_values_that_carry_no_scope():
    lits = sql_literals("SELECT * FROM t WHERE vendor_name='MSD' AND is_active='Y' AND month='2025-12'")
    assert "MSD" in lits
    assert "Y" not in lits          # a flag scopes nothing
    assert "2025-12" not in lits    # nor does a date


def test_literals_handle_escaped_quotes():
    assert sql_literals("SELECT * FROM t WHERE name='O''Brien Pharma'") == ["O'Brien Pharma"]
