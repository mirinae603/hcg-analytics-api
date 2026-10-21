# tests/test_kpi_registry.py — the canonical KPI tool the chatbot uses instead of
# re-deriving named metrics from ad hoc SQL. See app/ai/kpi_registry.py's module
# docstring for the live-audit bug this exists to fix.
from __future__ import annotations

from pathlib import Path

import pytest

from app.ai import kpi_registry as kr
from app.core.config import settings

REPORT_KEYS = ("formulary-compliance", "brand-rules", "scheme-brand-saving", "attis-saving", "cash-tpa-margin-loss")


def _built(entry: dict) -> bool:
    """The Jun–Aug 2026 report tables are published by their own ETL and are not in git: a clean
    checkout has none, and those entries answer 'not loaded'. Call them only where they exist."""
    need = entry.get("needs")
    if not need:
        return True
    folder = settings.FORMULARY_DATA_DIR if need == "formulary" else settings.SPEND_DATA_DIR
    return any(Path(folder).glob("*.parquet"))


def test_every_registered_kpi_can_be_called_unfiltered():
    # This is the exact class of bug that shipped once already: calling a route handler
    # directly (bypassing FastAPI's own request-time Query() resolution) leaves any OTHER
    # Query(...)-defaulted parameter beyond Plant/Category as a live fastapi.params.Query
    # object, not its real default — reorder_priority's `band` param raised
    # `TypeError: int() argument must be ... not 'Query'` the first time this was tried.
    # _neutral_kwargs is supposed to catch this generically for every KPI, present and
    # future — this test is the regression guard for that promise.
    for key, entry in kr.KPI_REGISTRY.items():
        if not _built(entry):
            continue
        result = kr.call_kpi(key, plant=None, category=None)
        assert result["_kpi_key"] == key
        assert result["_canonical"] is True
        assert isinstance(result["data"], dict)
        assert result["data"], f"{key} returned an empty dict"
        assert "error" not in result["data"], f"{key}: {result['data']}"
        assert "_ignored_args" not in result


def test_call_kpi_scopes_to_a_specific_hospital():
    # Direct proof the registry actually threads Plant= through — the exact call that
    # resolves the audit's false-capability-claim finding ("ITR isn't available by
    # hospital"): it demonstrably is, HC05's figure differs from the company-wide one.
    company = kr.call_kpi("inventory-turnover-ratio", plant=None)["data"]
    hc05 = kr.call_kpi("inventory-turnover-ratio", plant="HC05")["data"]
    assert company["totals"]["portfolio_itr"] != hc05["totals"]["portfolio_itr"]
    assert hc05["totals"]["total_skus"] < company["totals"]["total_skus"]


def test_unknown_kpi_key_raises():
    with pytest.raises(KeyError):
        kr.call_kpi("not-a-real-kpi")


def test_tool_schema_enum_matches_registry_exactly():
    # If these ever drift apart, the model could be told a KPI exists (in its own past
    # training/memory) that the enum doesn't actually allow, or vice versa. tool_schema()
    # is supposed to generate the enum FROM the registry so this can't happen by
    # construction — this test just confirms that promise holds.
    schema = kr.tool_schema()
    assert schema["function"]["name"] == "get_kpi"
    enum = set(schema["function"]["parameters"]["properties"]["kpi"]["enum"])
    assert enum == set(kr.KPI_REGISTRY)


def test_revenue_margin_takes_no_plant_or_category():
    # revenue_insights() takes neither filter — confirm call_kpi doesn't try to pass one
    # (which would raise a TypeError: unexpected keyword argument).
    result = kr.call_kpi("revenue-margin", plant="HC05", category="Injections")
    assert result["_canonical"] is True
    assert "revenue" in result["data"]["totals"]


def test_consumption_by_department_takes_no_category():
    # dept_insights() takes only Plant, not Category — confirm passing category doesn't
    # raise even though it's silently ignored (matches the registry's own `category: False`
    # declaration for this entry).
    result = kr.call_kpi("consumption-by-department", plant="All Plants", category="Injections")
    assert result["_canonical"] is True


# ---------- extra arguments (the Jun–Aug 2026 report keys) ----------

def _capture(monkeypatch, key: str) -> dict:
    """Swap the entry's function for one that records what call_kpi passed, so these tests
    need no data at all."""
    seen: dict = {}

    def fake(**kwargs):
        seen.update(kwargs)
        return {"totals": {"x": 1}}
    monkeypatch.setitem(kr.KPI_REGISTRY, key, {**kr.KPI_REGISTRY[key], "fn": fake})
    return seen


def test_extra_args_reach_only_the_entries_that_take_them(monkeypatch):
    seen = _capture(monkeypatch, "formulary-compliance")
    r = kr.call_kpi("formulary-compliance", plant="HC05", month=" Aug ", view="", doctor=None,
                    measure="restricted_scheme", payer="GOV")
    assert seen == {"Plant": "HC05", "Category": None, "month": "Aug", "payer": "GOV"}
    # measure belongs to brand-rules: reported back, never silently dropped
    assert r["_ignored_args"] == ["measure"]
    assert r["data"] == {"totals": {"x": 1}} and r["_canonical"] is True


def test_a_month_passed_to_a_dec_may_kpi_is_reported_as_ignored(monkeypatch):
    # The failure this guards: 'fill rate for July' answered for the whole Dec–May window with
    # nothing in the result saying the month did nothing.
    seen = _capture(monkeypatch, "fill-rate")
    r = kr.call_kpi("fill-rate", plant=None, month="2026-07", view="trend")
    assert "month" not in seen and "view" not in seen
    assert r["_ignored_args"] == ["month", "view"]


def test_no_ignored_args_key_when_nothing_was_ignored(monkeypatch):
    _capture(monkeypatch, "attis-saving")
    assert "_ignored_args" not in kr.call_kpi("attis-saving", month="all", payer=None)


def test_report_entries_declare_window_ratio_and_data():
    for key in REPORT_KEYS:
        e = kr.KPI_REGISTRY[key]
        assert e["ratio"] is True and e["needs"] in ("formulary", "spend"), key
        assert e["window"] == "reports"                        # months read from the data
        assert e["about"].startswith("<<REPORTS>> ONLY"), key
        assert e["hint"].startswith("<<REPORTS>> ONLY"), key
        assert e["args"] and set(e["args"]) <= set(kr.EXTRA_ARGS), key
    # ratio KPIs that already existed: summing per-site figures is wrong for them too
    for key in ("fill-rate", "stock-out-rate", "wastage-rate", "inventory-turnover-ratio", "days-on-hand"):
        assert kr.KPI_REGISTRY[key].get("ratio") is True, key
    assert not kr.KPI_REGISTRY["purchase-value"].get("ratio")


def test_schema_declares_every_extra_argument_with_valid_enums():
    props = kr.tool_schema()["function"]["parameters"]["properties"]
    for name in kr.EXTRA_ARGS:
        assert name in props, name
        if "enum" in props[name]:
            assert props[name]["enum"], f"{name}: an empty enum makes the whole tool schema invalid"
    assert props["payer"]["enum"] == ["CASH", "CASH(TPA)", "GOV"]
    assert {"summary", "trend", "hospitals", "doctors", "items", "measures", "book"} <= set(props["view"]["enum"])


def test_schema_lists_every_key_and_label_but_stays_short():
    desc = kr.tool_schema()["function"]["description"]
    for key, e in kr.KPI_REGISTRY.items():
        assert f"{key} ({e['label']}" in desc, key
    # every `about` in here made the tool 5.4K characters, resent on every tool round
    assert len(desc) < 3600
    assert "ALWAYS call this FIRST" not in desc


def test_revenue_margin_says_it_is_company_wide_only():
    # The Keytruda reply ('no monthly breakdown... total for the 6-month period') came from
    # this route being read as a product's sales trend.
    e = kr.KPI_REGISTRY["revenue-margin"]
    assert e["about"].startswith("Company-wide ONLY")
    assert "sales_by_material_month" in e["about"] and "sales_by_material_month" in e["hint"]


# ---------- plant values a Dec–May KPI cannot resolve ----------
# resolve_plant returns None for anything it does not recognise, and filter_plant reads None
# as "no filter": plant="HC01,HC05,HC06,HC40" on fill-rate came back as the whole network's
# figure, labelled as the four Bangalore hospitals.

def test_a_comma_list_is_refused_by_a_dec_may_kpi(monkeypatch):
    seen = _capture(monkeypatch, "fill-rate")
    with pytest.raises(ValueError, match="ONE hospital per call") as e:
        kr.call_kpi("fill-rate", plant="HC01,HC05,HC06,HC40")
    assert seen == {}, "the handler must not run with a plant it cannot filter on"
    # fill-rate is a ratio: the advice must not invite combining the per-site figures
    msg = str(e.value).lower()
    assert not any(w in msg for w in ("sum", " add", "average")), msg
    # a money KPI may be totalled across sites
    _capture(monkeypatch, "purchase-value")
    with pytest.raises(ValueError, match="total the per-site figures"):
        kr.call_kpi("purchase-value", plant="HC01, HC05")


def test_an_unknown_plant_is_refused_instead_of_answering_for_the_network(monkeypatch):
    seen = _capture(monkeypatch, "stock-out-rate")
    for bad in ("HC5", "Gotham"):
        with pytest.raises(ValueError, match="WHOLE network"):
            kr.call_kpi("stock-out-rate", plant=bad)
    # a city with several hospitals is refused too, naming them: one call per hospital
    with pytest.raises(ValueError, match="covers 3 hospitals"):
        kr.call_kpi("stock-out-rate", plant="Bangalore")
    assert seen == {}


def test_a_padded_or_lower_case_code_and_all_still_work(monkeypatch):
    seen = _capture(monkeypatch, "fill-rate")
    kr.call_kpi("fill-rate", plant=" hc05 ")
    assert seen["Plant"] == "HC05"
    for everything in (None, "", "All Plants", "all hospitals"):
        seen.clear()
        kr.call_kpi("fill-rate", plant=everything)
        assert seen["Plant"] == "All Plants", everything


def test_report_keys_still_take_a_comma_list(monkeypatch):
    # the Jun–Aug reports pool several hospitals themselves; the list must reach them as sent
    seen = _capture(monkeypatch, "formulary-compliance")
    kr.call_kpi("formulary-compliance", plant="HC05,HC06")
    assert seen["Plant"] == "HC05,HC06"


def test_schema_says_comma_lists_are_for_report_keys_only():
    desc = kr.tool_schema()["function"]["parameters"]["properties"]["plant"]["description"]
    assert "Only the <<REPORTS>> report keys take several codes" in desc
    assert "once per code" in desc
