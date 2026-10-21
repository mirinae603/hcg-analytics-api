# tests/test_ai_retrieval_fixes.py — "get me the sales trend of keytruda".
#
# The assistant answered "the sales data for Keytruda does not have a monthly breakdown —
# would you like the 6-month total instead?" while sales_by_material_month held all six
# months. Three prompt lines taught that reply word for word, the month table was the one
# sales view left with its '.0' duplicate codes (so even a correct query undercounted
# December by ₹39 L), and nothing handed the model the query that answers it.
#
# Also covered: the shared-connection race (an unlocked execute+fetch reads another thread's
# rows), the per-turn rebuild of the schema prompt, and routing that retried a failing
# embedding endpoint on every question.
from __future__ import annotations

import re
import sys
import threading
from pathlib import Path

from app.ai import resolve, routing, semantics, warehouse, windows
from app.ai.resolve import brief, item_trend_sql

KEYTRUDA = "101313"
_AI = Path(__file__).resolve().parents[1] / "app" / "ai"


# ── the prompt no longer teaches the wrong reply ────────────────────────────────────────
def test_the_guide_lists_the_month_table():
    assert "sales_by_material_month" in semantics.GUIDE


def test_gotcha_8_no_longer_says_an_item_has_no_monthly_sales():
    assert "no table carries item AND a sales time axis" not in semantics.GOTCHAS
    assert "sales_by_material_month" in semantics.GOTCHAS


def test_there_is_a_worked_item_trend_example_without_literal_figures():
    i = semantics.EXAMPLES.index("Sales trend of <item X>")
    block = semantics.EXAMPLES[i:i + 700]
    assert "FROM sales_by_material_month" in block and "GROUP BY period" in block
    # figures in the prompt go stale on the next refresh (test_no_stale_facts.py)
    assert not re.search(r"9\.68|9\.29|₹\s?\d", block)


def test_the_window_rule_names_the_jun_aug_reports_as_the_exception():
    assert "<<REPORTS_RANGE>>" in semantics.RULES and "formulary-compliance" in semantics.RULES
    assert windows.values()["REPORTS_RANGE"] in windows.fill(semantics.RULES)      # read from the data


# ── the month view is de-duplicated like the other three ────────────────────────────────
def test_the_month_view_has_no_float_material_codes():
    n = warehouse.query(
        "SELECT count(*) FROM sales_by_material_month WHERE material LIKE '%.0'", one=True)[0]
    assert n == 0


def test_keytruda_december_includes_both_code_formats():
    v = warehouse.query(
        "SELECT SUM(revenue) FROM sales_by_material_month "
        "WHERE material = ? AND period = '2025-12'", [KEYTRUDA], one=True)[0]
    assert abs(v / 1e7 - 9.68) < 0.01, v / 1e7          # 9.29 before the merge


def test_the_month_view_still_reconciles_with_the_company_total():
    a = warehouse.query("SELECT SUM(revenue) FROM sales_by_material_month", one=True)[0]
    b = warehouse.query("SELECT SUM(revenue) FROM sales_totals", one=True)[0]
    assert abs(a - b) < 1.0, (a, b)


# ── the ready-made query ────────────────────────────────────────────────────────────────
def test_the_user_question_gets_the_month_query():
    sql = item_trend_sql("get me the sales trend of keytruda")
    assert sql and "sales_by_material_month" in sql and "KEYTRUDA" in sql
    assert "GROUP BY period ORDER BY period" in sql


def test_the_ready_made_query_runs_and_gives_the_true_series():
    res = warehouse.run_sql(item_trend_sql("get me the sales trend of keytruda"))
    assert [r["period"] for r in res["rows"]] == [
        "2025-12", "2026-01", "2026-02", "2026-03", "2026-04", "2026-05"]
    assert abs(res["rows"][0]["revenue"] / 1e7 - 9.68) < 0.01
    assert abs(res["rows"][1]["revenue"] / 1e7 - 9.40) < 0.01
    assert all(r["margin"] is not None and r["qty"] > 0 for r in res["rows"])


def test_an_exact_description_is_filtered_by_code():
    # the sales feed spells some products differently from the item master, so the code is
    # the safe filter for an exact name
    sql = item_trend_sql("monthly sales of KEYTRUDA 100MG INJ VIAL")
    assert sql and f"material IN ('{KEYTRUDA}')" in sql


def test_a_molecule_reaches_its_brand():
    sql = item_trend_sql("sales trend of pembrolizumab")
    assert sql and f"'{KEYTRUDA}'" in sql


def test_a_small_family_is_one_product_question():
    sql = item_trend_sql("monthly revenue of tecentriq")
    assert sql and "TECENTRIQ" in sql


def test_questions_it_must_not_answer():
    for q in ("what is our margin by vendor?",            # no month, no product
              "sales trend of KABHK",                      # a hospital, not a product
              "consumption trend of keytruda",             # not a sales measure
              "monthly sales of keytruda by hospital",     # month view has no hospital
              "sales forecast of keytruda by month",       # projections are not history
              "sales trend of keytruda and perjeta",       # two products, one summed series
              "keytruda sales trend in Bangalore",         # a city scope the view cannot carry
              "what were total sales last month?",         # no product at all
              # A word the question merely contains is not "the product". These bound
              # "network" (network switches), "wise" (₹75 L of WISE-named products) and
              # "split" to one-token families and were handed a filtered series as the
              # company's trend.
              "Show the network-wide monthly sales trend",
              "What is the monthly margin trend across the network?",
              "monthly revenue trend against target",
              "What is the monthly sales trend split by IP and OP?",
              "month wise sales trend",
              # A hospital by its short form resolves to nothing, so these looked
              # company-wide; no table holds one product by hospital AND month.
              "sales trend of keytruda at HCG KR",
              "sales trend of keytruda at hcg kr",
              "monthly sales of keytruda at KR",
              "sales trend of keytruda at Panda"):        # a one-token hospital family
        assert item_trend_sql(q) is None, q


def test_the_product_can_still_be_named_before_the_measure():
    for q in ("keytruda sales trend", "keytruda's monthly revenue",
              "monthly sales trend for 'keytruda'"):
        sql = item_trend_sql(q)
        assert sql and "KEYTRUDA" in sql, q


def test_a_family_token_must_be_the_object_of_the_question():
    assert resolve._names_as_the_object("get me the sales trend of keytruda", "keytruda")
    assert resolve._names_as_the_object("keytruda sales trend", "keytruda")
    assert not resolve._names_as_the_object("Show the network-wide monthly sales trend",
                                            "network")
    assert not resolve._names_as_the_object("margin trend across the network", "network")


def test_site_short_forms_are_read_from_the_plant_names():
    short = resolve._site_short_names()
    assert {"kr", "dr", "nmr"} <= short
    assert "hcg" not in short and "llp" not in short
    assert resolve._names_a_site("sales trend of keytruda at HCG KR")
    assert not resolve._names_a_site("get me the sales trend of keytruda")
    assert not resolve._names_a_site("monthly sales trend across the HCG network")


def test_the_brief_says_a_hospital_product_trend_does_not_exist():
    b = brief("sales trend of keytruda at HCG KR")
    assert "MONTHLY SALES EXIST" not in b
    assert "NOT AVAILABLE AT THIS GRAIN" in b and "ALL hospitals" in b


def test_the_brief_hands_the_query_over():
    b = brief("get me the sales trend of keytruda")
    assert "THIS PRODUCT'S MONTHLY SALES EXIST" in b and "sales_by_material_month" in b
    assert "NOT AVAILABLE AT THIS GRAIN" not in b


def test_the_brief_does_not_offer_it_where_it_would_be_wrong():
    assert "MONTHLY SALES EXIST" not in brief("monthly sales of keytruda by hospital")


# ── consumption and formulary guidance point at what exists ─────────────────────────────
def test_the_consumption_note_points_at_the_billed_monthly_series():
    note = resolve._note_text(resolve._MEASURE_NOTES["consumption"])
    assert "sales_by_material_month" in note
    assert "cannot give a monthly" not in note


def test_a_compliance_question_is_sent_to_the_report_tool():
    b = brief("what is the formulary compliance at HCG KR in August?")
    assert "formulary-compliance" in b and "get_kpi" in b


def test_the_item_master_flag_is_still_offered_for_the_sales_share():
    # eval l4-nonformulary-share answers from dim_material.formulary_status
    b = brief("What share of our sales revenue comes from non-formulary items?")
    assert "formulary_status" in b


def test_a_report_word_without_formulary_also_gets_the_note():
    assert "attis-saving" in brief("how much did we save with Attis at KR?")


def test_the_reporting_window_names_the_report_window():
    w = resolve.reporting_window()
    assert w.startswith(windows.sales_months()[0]) and "<<REPORTS_RANGE>>" in w


# ── small resolver fixes ────────────────────────────────────────────────────────────────
def test_median_and_today_are_stop_words_again():
    # "median', 'today" was ONE string, so neither word was actually a stop word
    assert "median" in resolve._STOP and "today" in resolve._STOP
    assert "median', 'today" not in resolve._STOP


def test_resolve_is_cached_but_hands_out_copies():
    q = "get me the sales trend of keytruda"
    first = resolve.resolve(q)
    first["families"].clear()
    first["grains"].append("nonsense")
    again = resolve.resolve(q)
    assert again["families"] and "nonsense" not in again["grains"]
    assert resolve._resolve_cached.cache_info().hits >= 1


# ── one connection, one lock ────────────────────────────────────────────────────────────
def test_the_lock_is_reentrant():
    assert type(warehouse._lock) is type(threading.RLock())
    with warehouse._lock:                     # con() takes the lock inside a locked block
        assert warehouse.query("SELECT 1", one=True)[0] == 1


def test_two_threads_never_read_each_others_rows():
    warehouse.con()
    bad: list = []

    def via_run_sql():
        for i in range(500):
            rows = warehouse.run_sql(f"SELECT {i} AS n")["rows"]
            if rows != [{"n": i}]:
                bad.append(("run_sql", i, rows))

    def via_query():
        for i in range(500):
            row = warehouse.query(f"SELECT {i}, 2", one=True)
            if tuple(row) != (i, 2):
                bad.append(("query", i, row))

    threads = [threading.Thread(target=via_run_sql), threading.Thread(target=via_query)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not bad, bad[:5]


def test_owned_modules_never_execute_and_fetch_on_the_bare_connection():
    for name in ("resolve", "scope", "coverage", "ontology", "semantics", "routing"):
        src = (_AI / f"{name}.py").read_text()
        assert not re.search(r"con\(\)\s*\.execute\(", src), name


def test_the_documented_ontology_rebuild_command_exists():
    assert 'if __name__ == "__main__":' in (_AI / "ontology.py").read_text()


# ── the prompt context is built once ────────────────────────────────────────────────────
def test_context_is_cached(monkeypatch):
    calls = {"schema": 0, "grain": 0}

    def schema():
        calls["schema"] += 1
        return "fake schema"

    def grain():
        calls["grain"] += 1
        return "fake grain"

    monkeypatch.setattr(warehouse, "schema_text", schema)
    monkeypatch.setattr(warehouse, "grain_text", grain)
    semantics.context.cache_clear()
    try:
        a = semantics.context()
        b = semantics.context()
        assert a == b and "fake schema" in a
        assert calls == {"schema": 1, "grain": 1}
    finally:
        semantics.context.cache_clear()     # never leave the fake text cached


def test_schema_and_grain_text_are_cached():
    assert hasattr(warehouse.schema_text, "cache_clear")
    assert hasattr(warehouse.grain_text, "cache_clear")
    assert warehouse.grain_text() is warehouse.grain_text()


# ── routing fails fast and remembers it ─────────────────────────────────────────────────
class _FailingEmbeddings:
    def __init__(self):
        self.calls = 0

    def create(self, **kw):
        self.calls += 1
        raise RuntimeError("deployment not found")


class _FakeClient:
    def __init__(self):
        self.embeddings = _FailingEmbeddings()
        self.options: list[dict] = []

    def with_options(self, **kw):
        self.options.append(kw)
        return self


def test_a_failed_corpus_embed_is_not_retried_on_every_question(monkeypatch):
    monkeypatch.setattr(routing, "_corpus_vecs", None)
    monkeypatch.setattr(routing, "_corpus_failed_at", 0.0)
    client = _FakeClient()
    q = "are we overpaying for stents?"
    first = routing.hints_for(client, q)
    second = routing.hints_for(client, q)
    assert client.embeddings.calls == 1
    assert first == second and "REQUIRED APPROACH" in first      # the regex lock survives
    assert client.options and client.options[0] == {
        "timeout": routing.EMBED_TIMEOUT_S, "max_retries": 0}


def test_routing_has_a_product_trend_exemplar_and_no_doctor_refusal():
    hints = " ".join(h for _, h in routing._CORPUS)
    assert "ITEM/CATEGORY SALES TREND → sales_by_material_month" in hints
    assert not any("doctor" in ex for ex, h in routing._CORPUS if h.startswith("OUT OF SCOPE"))
    assert "get_kpi(\"formulary-compliance\"" in hints


# ── a total is not a trend ──────────────────────────────────────────────────────────────
def test_a_total_question_is_not_handed_the_month_query():
    # "Keytruda total revenue for the 6-month period" was the follow-up the assistant itself
    # offered; "6-month" resolved the month grain and the brief said "Run exactly this" over
    # the month-by-month query.
    for q in ("Keytruda total revenue for the 6-month period",
              "overall sales of keytruda across the six months",
              "total sales of keytruda over 6 months"):
        assert item_trend_sql(q) is None, q
    assert "MONTHLY SALES EXIST" not in brief("Keytruda total revenue for the 6-month period")


def test_trend_wording_wins_over_total_wording():
    for q in ("total monthly revenue of keytruda", "6-month sales trend of keytruda",
              "total sales of keytruda by month"):
        sql = item_trend_sql(q)
        assert sql and "KEYTRUDA" in sql, q


def test_a_hospital_total_is_not_called_unavailable():
    # a product's whole-window total per hospital exists (sales_by_material_hospital)
    assert not resolve.site_trend_unavailable(
        "total sales of keytruda at HCG KR over the 6 months")
    assert resolve.site_trend_unavailable("sales trend of keytruda at HCG KR")


# ── "last month" is the last month in the data ──────────────────────────────────────────
def test_relative_note_wording(monkeypatch):
    monkeypatch.setattr(resolve, "_sales_window_months", lambda: ("2025-12", "2026-05"))
    monkeypatch.setattr(resolve, "_latest_report_month", lambda: "2026-08")
    assert resolve.relative_date_note("procurement spend last month") == (
        "RELATIVE DATE: The data does not run to today. 'Last/latest month' = the latest "
        "month in the data: May-2026 (2026-05) for sales, purchasing and consumption; "
        "Aug-2026 (2026-08) for the formulary/spend/brand-rule reports. Say which month you "
        "used. Never report an empty recent month as zero.")
    for q in ("revenue this quarter", "sales YTD", "what did we buy yesterday",
              "previous month's consumption", "current FY spend", "MTD revenue"):
        assert resolve.relative_date_note(q), q
    for q in ("sales over the last 6 months", "get me the sales trend of keytruda",
              "which month was keytruda highest"):
        assert resolve.relative_date_note(q) == "", q


def test_fy_to_date_is_counted_in_the_data(monkeypatch):
    monkeypatch.setattr(resolve, "_latest_report_month", lambda: None)
    monkeypatch.setattr(resolve, "_sales_window_months", lambda: ("2025-12", "2026-05"))
    n = resolve.relative_date_note("what is our revenue this FY?")
    assert "FY-to-date in the data is Apr-2026..May-2026" in n
    assert "calendar YTD is Jan-2026..May-2026" in n
    assert "formulary" not in n                    # no report month, no report clause
    # an FY that began before the data did starts where the data starts
    monkeypatch.setattr(resolve, "_sales_window_months", lambda: ("2025-12", "2026-02"))
    assert "FY-to-date in the data is Dec-2025..Feb-2026" in resolve.relative_date_note(
        "revenue YTD")


def test_the_relative_note_reads_its_months_from_the_data():
    first, last = resolve._sales_window_months()
    assert last == warehouse.query("SELECT MAX(month) FROM sales_monthly", one=True)[0]
    assert resolve._month_label("2026-05") == "May-2026"
    b = brief("what was procurement spend last month?")
    assert f"{resolve._month_label(last)} ({last}) for sales" in b
    assert "Never report an empty recent month as zero" in b
    rep = resolve._latest_report_month()
    if rep:
        assert f"{resolve._month_label(rep)} ({rep}) for the formulary" in b


def test_a_question_with_only_a_relative_date_still_gets_a_brief():
    assert "RELATIVE DATE" in brief("what happened yesterday?")
    assert "RELATIVE DATE" not in brief("get me the sales trend of keytruda")


# ── IP/OP ───────────────────────────────────────────────────────────────────────────────
def test_ip_op_wording_gets_the_split_note():
    for q in ("OP sales of keytruda", "inpatient revenue by hospital",
              "IPD vs OPD consumption", "out-patient margin by category"):
        assert "IP/OP split exists ONLY company-wide" in brief(q), q
    for q in ("top items by revenue", "what is our total revenue by hospital?"):
        assert "IP/OP split" not in brief(q), q


# ── month labels ────────────────────────────────────────────────────────────────────────
def test_gotcha_6_names_the_period_key_and_the_prose_format():
    g = semantics.GOTCHAS
    assert "period" in g and "with NO exception" not in g
    assert 'Mon-YYYY ("Jan-2026")' in g and "ORDER BY period" in g


# ── a data refresh also forgets the AI's cached prompt ──────────────────────────────────
def test_refresh_data_resets_the_ai_caches_when_loaded(monkeypatch):
    from app.api import dashboard_summary as ds
    from app.core.config import settings
    calls = []
    monkeypatch.setattr(ds.da, "refresh_cache", lambda: None)
    monkeypatch.setattr(warehouse, "reset_caches", lambda: calls.append(1))
    assert ds.refresh_data(x_admin_token=settings.ADMIN_REFRESH_TOKEN) == {
        "status": "cache cleared"}
    assert calls == [1]


def test_refresh_data_never_imports_the_ai_side(monkeypatch):
    from app.api import dashboard_summary as ds
    from app.core.config import settings
    monkeypatch.setattr(ds.da, "refresh_cache", lambda: None)
    monkeypatch.delitem(sys.modules, "app.ai.warehouse")
    ds.refresh_data(x_admin_token=settings.ADMIN_REFRESH_TOKEN)
    assert "app.ai.warehouse" not in sys.modules
