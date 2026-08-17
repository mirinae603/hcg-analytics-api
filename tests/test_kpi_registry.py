# tests/test_kpi_registry.py — the canonical KPI tool the chatbot uses instead of
# re-deriving named metrics from ad hoc SQL. See app/ai/kpi_registry.py's module
# docstring for the live-audit bug this exists to fix.
from __future__ import annotations

import pytest

from app.ai import kpi_registry as kr


def test_every_registered_kpi_can_be_called_unfiltered():
    # This is the exact class of bug that shipped once already: calling a route handler
    # directly (bypassing FastAPI's own request-time Query() resolution) leaves any OTHER
    # Query(...)-defaulted parameter beyond Plant/Category as a live fastapi.params.Query
    # object, not its real default — reorder_priority's `band` param raised
    # `TypeError: int() argument must be ... not 'Query'` the first time this was tried.
    # _neutral_kwargs is supposed to catch this generically for every KPI, present and
    # future — this test is the regression guard for that promise.
    for key in kr.KPI_REGISTRY:
        result = kr.call_kpi(key, plant=None, category=None)
        assert result["_kpi_key"] == key
        assert result["_canonical"] is True
        assert isinstance(result["data"], dict)
        assert result["data"], f"{key} returned an empty dict"


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
