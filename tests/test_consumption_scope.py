# tests/test_consumption_scope.py — consumption is two events; the table holds one.
#
# "How does the consumption trend of Keytruda look?" was answered "there is no recorded
# internal consumption trend", for a drug with 2,193 units billed and ₹47.48 Cr of revenue.
# The query was correct. The TABLE was wrong: fact_consumption records store issues only,
# and 13,928 of 25,166 materials are dispensed against a patient's bill instead — they have
# zero rows there permanently, by construction.
from __future__ import annotations

from app.ai.resolve import brief, resolve, spelling_suggestions
from app.ai.scope import billed_not_internal

_TREND_SQL = ("SELECT date_trunc('month', posting_date) m, SUM(qty) FROM fact_consumption "
              "WHERE upper(material_desc) LIKE '%KEYTRUDA%' GROUP BY 1")


def test_an_empty_consumption_result_for_a_billed_item_is_corrected():
    w = billed_not_internal(_TREND_SQL, {"row_count": 0})
    assert w and w.startswith("WRONG TABLE")
    assert "2,193 units" in w                      # the real figure, not a pointer
    assert "kpi_billable_consumption" in w


def test_it_forbids_reporting_the_item_as_unused():
    w = billed_not_internal(_TREND_SQL, {"row_count": 0})
    assert "Do NOT report that this item has no consumption" in w


def test_it_warns_that_no_monthly_trend_exists_at_material_grain():
    w = billed_not_internal(_TREND_SQL, {"row_count": 0})
    assert "no month column" in w and "not" in w


def test_a_non_empty_result_is_left_alone():
    assert billed_not_internal(_TREND_SQL, {"row_count": 12}) is None


def test_other_tables_are_not_touched():
    assert billed_not_internal(
        "SELECT SUM(qty) FROM sales_by_material WHERE material_desc LIKE '%KEYTRUDA%'",
        {"row_count": 0}) is None


def test_an_item_with_real_internal_issues_is_not_redirected():
    # only items that are billed-and-not-issued should trigger the correction
    sql = ("SELECT SUM(qty) FROM fact_consumption "
           "WHERE upper(material_desc) LIKE '%NO SUCH ITEM XYZZY%'")
    assert billed_not_internal(sql, {"row_count": 0}) is None


# ── spelling ────────────────────────────────────────────────────────────────────────────
def test_a_one_letter_typo_is_recognised():
    q = "how does the consumption trend of keytuda look?"
    s = spelling_suggestions(q, resolve(q))
    assert any(x["typed"] == "keytuda" and x["meant"] == "keytruda" for x in s), s


def test_a_generic_name_typo_is_recognised_too():
    q = "trend for pembrolzumab"
    s = spelling_suggestions(q, resolve(q))
    assert any(x["meant"] == "pembrolizumab" for x in s), s


def test_correct_spelling_produces_no_suggestion():
    q = "how does the consumption trend of keytruda look?"
    assert spelling_suggestions(q, resolve(q)) == []


def test_a_short_token_is_never_fuzzy_matched():
    # 'msd' vs 'mds' vs 'msdc' — the STICKER-MSDS failure lives at this length
    q = "what do we buy from mds?"
    assert all(len(x["typed"]) >= 5 for x in spelling_suggestions(q, resolve(q)))


def test_the_brief_tells_it_to_proceed_not_to_ask():
    b = brief("how does the consumption trend of keytuda look?")
    assert "LIKELY MISSPELLING" in b
    assert "do not stop and ask" in b


def test_a_single_match_family_is_stated_as_an_identification():
    b = brief("how does the consumption trend of keytruda look?")
    assert "IS KEYTRUDA 100MG INJ VIAL" in b
    assert "Treat it as identified" in b


# ── the data-model fix: one honest place to ask the question ────────────────────────────
def test_consumption_all_holds_both_scopes():
    from app.ai.warehouse import con
    rows = con().execute(
        "SELECT scope, COUNT(DISTINCT material) FROM consumption_all GROUP BY 1").fetchall()
    scopes = {r[0]: r[1] for r in rows}
    assert scopes.get("internal", 0) > 10_000
    assert scopes.get("billed", 0) > 10_000


def test_it_covers_far_more_materials_than_fact_consumption_alone():
    from app.ai.warehouse import con
    both = con().execute("SELECT COUNT(DISTINCT material) FROM consumption_all").fetchone()[0]
    internal = con().execute(
        "SELECT COUNT(DISTINCT material) FROM fact_consumption").fetchone()[0]
    assert both > internal * 2, (both, internal)


def test_keytruda_is_findable_there():
    from app.ai.warehouse import con
    rows = con().execute(
        "SELECT scope, qty FROM consumption_all WHERE material = '101313'").fetchall()
    assert rows and rows[0][0] == "billed"
    assert abs(rows[0][1] - 2193) < 1


def test_every_row_names_its_scope():
    # a total that silently mixes issued-from-stores with billed-to-patient is worse than
    # either number alone
    from app.ai.warehouse import con
    bad = con().execute(
        "SELECT COUNT(*) FROM consumption_all WHERE scope NOT IN ('internal','billed')"
    ).fetchone()[0]
    assert bad == 0


def test_the_cost_column_is_not_offered_as_revenue():
    # it was called `value`, which matches the revenue pattern, so a COST column was listed
    # as somewhere revenue lives
    from app.ai.resolve import measure_locations
    assert not any("consumption_all" in loc for loc in measure_locations("revenue"))


def test_consumption_resolves_to_the_view_first():
    from app.ai.resolve import measure_locations
    assert measure_locations("consumption")[0].startswith("consumption_all")


def test_internal_only_tables_are_not_offered_as_measure_homes():
    # `_pydf_*` are pandas frames registered as an implementation detail
    from app.ai.resolve import measure_locations
    for m in ("consumption", "revenue", "stock", "lead_time"):
        assert not any(loc.startswith("_pydf") for loc in measure_locations(m)), m
