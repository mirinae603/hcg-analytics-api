# tests/test_ai_final_fixes.py — regressions found by the last review of the AI changes.
#
# 1. Units: a KPI's `rate` / `value_rate` are percentages (stock-out), unit_rate / avg_rate
#    are rupee prices, and a percent name ending in a count word (pct_of_lines) is still a
#    percentage. The fast path (orchestrator.infer_kind) and deep mode (engine._kind) agree.
# 2. The scale check compares rupees: "₹45 L" restating revenue_cr = 0.45 is correct, while a
#    real 10x slip is still caught.
# 3. A chat deleted while its answer is being written gives a clear message, not a 500.
from __future__ import annotations

import json

import pytest

from app.ai import orchestrator as orch
from app.ai.deep import engine


@pytest.mark.parametrize("col,kind", [
    ("rate", "pct"), ("value_rate", "pct"), ("growth_rate", "pct"), ("fill_rate", "pct"),
    ("unit_rate", "inr"), ("avg_rate", "inr"), ("purchase_rate", "inr"),
    ("pct_of_lines", "pct"), ("percent_of_units", "pct"), ("share_of_units", "pct"),
    ("purchase_qty", "num"), ("sales_lines", "num"), ("revenue_cr", "num"),
])
def test_fast_and_deep_agree_on_units(col, kind):
    assert orch.infer_kind(col) == kind
    assert engine._kind(col, [{col: 1.5}]) == kind


def test_deep_count_column_holding_text_stays_text():
    assert engine._kind("units", [{"units": "Strip"}]) == "text"


def test_stock_out_payload_keeps_its_percent_signs():
    out = orch._format_kpi_payload({"totals": {"rate": 12.3456, "value_rate": 6.0},
                                    "by_plant": [{"plant": "HC05", "rate": 7.5}]})
    assert str(out["totals"]["rate"]).endswith("%")
    assert str(out["totals"]["value_rate"]).endswith("%")
    assert "₹" not in str(out["totals"]["value_rate"])
    assert str(out["by_plant"][0]["rate"]).endswith("%")


def _res(col, value):
    return [{"sql": "-- test", "columns": [col], "rows": [{col: value}], "row_count": 1}]


def test_prescaled_crore_restated_in_lakh_is_not_flagged():
    assert orch._unsupported_numbers("Revenue was ₹45 L.", _res("revenue_cr", 0.45)) == []


def test_prescaled_lakh_restated_in_crore_is_not_flagged():
    assert orch._unsupported_numbers("Revenue was ₹9.68 Cr.", _res("revenue_lakh", 968)) == []


def test_prescaled_crore_tenfold_slip_is_still_flagged():
    assert orch._unsupported_numbers("Revenue was ₹96.8 Cr.", _res("revenue_cr", 9.68)) == ["₹96.8 Cr"]


def test_rupee_figure_restated_in_lakh_is_not_flagged_but_a_slip_is():
    res = _res("revenue", 300_200_000)                     # formatted ₹30.02 Cr
    assert orch._unsupported_numbers("That is ₹3,002 L.", res) == []
    assert orch._unsupported_numbers("That is ₹300.2 Cr.", res) == ["₹300.2 Cr"]


# ---------- a chat deleted while the answer is being written ----------

def _deleting_answer(db_session_factory, sid_box):
    def _answer(query, history=None, **_kw):
        from app.models import ChatMessage, ChatSession
        db = db_session_factory()
        try:
            db.query(ChatMessage).filter(ChatMessage.session_id == sid_box[0]).delete()
            db.query(ChatSession).filter(ChatSession.id == sid_box[0]).delete()
            db.commit()
        finally:
            db.close()
        yield {"type": "answer", "text": "An answer nobody will see", "verified": "ok", "options": []}
        yield {"type": "done"}
    return _answer


def test_post_message_to_a_chat_deleted_mid_turn_is_a_clear_404(client, admin_headers,
                                                                 db_session_factory, monkeypatch):
    session = client.post("/chat/sessions", json={}, headers=admin_headers).json()
    monkeypatch.setattr("app.ai.orchestrator.answer", _deleting_answer(db_session_factory, [session["id"]]))
    r = client.post(f"/chat/sessions/{session['id']}/messages", json={"query": "hi"}, headers=admin_headers)
    assert r.status_code == 404
    assert "deleted" in r.json()["detail"]


def test_stream_to_a_chat_deleted_mid_turn_ends_with_an_error_event(client, admin_headers,
                                                                    db_session_factory, monkeypatch):
    session = client.post("/chat/sessions", json={}, headers=admin_headers).json()
    monkeypatch.setattr("app.ai.orchestrator.answer", _deleting_answer(db_session_factory, [session["id"]]))
    with client.stream("POST", f"/chat/sessions/{session['id']}/messages/stream",
                       json={"query": "hi"}, headers=admin_headers) as r:
        assert r.status_code == 200
        events = [json.loads(line[5:]) for line in r.iter_lines() if line.startswith("data:")]
    types = [e["type"] for e in events]
    assert "persisted" not in types
    assert types[-1] == "error" or any(e["type"] == "error" and "deleted" in e.get("text", "") for e in events)


# ---------- the user is told the real reason the model could not be reached ----------

class _ApiErr(Exception):
    def __init__(self, status, message="boom"):
        super().__init__(message)
        self.status_code = status
        self.body = {"message": message}


@pytest.mark.parametrize("status,needle", [
    (401, "rejected the key or the endpoint"),
    (403, "rejected the key or the endpoint"),
    (404, "was not found"),
    (400, "refused the request"),
    (429, "rate-limited"),
    (503, "did not respond in time"),
])
def test_llm_error_text_names_the_real_cause(status, needle):
    text = orch._llm_error_text(_ApiErr(status, "Access denied"))
    assert needle in text
    if status != 429:
        assert "rate-limited" not in text


def test_llm_error_text_looks_through_a_failed_stream_open():
    try:
        try:
            raise _ApiErr(401)
        except _ApiErr as inner:
            raise orch._StreamOpenFailed("stream", False) from inner
    except orch._StreamOpenFailed as e:
        assert "rejected the key" in orch._llm_error_text(e)


def test_check_connection_reports_a_missing_key_without_calling_azure(monkeypatch, tmp_path, capsys):
    from app.ai import check_connection
    monkeypatch.chdir(tmp_path)                       # no .env here
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    assert check_connection.main() == 1
    assert "MISSING" in capsys.readouterr().out


# ---------- hospitals named the way HCG says them ("KR", "Nagpur") ----------

def test_every_hospital_alias_points_at_a_mapped_plant():
    from app.ai import hospitals
    mapped = set(hospitals._rows())
    if not mapped:
        pytest.skip("hospital_map.csv not present")
    for alias, codes in hospitals._ALIASES.items():
        assert set(codes) <= mapped, alias


@pytest.mark.parametrize("text,codes", [
    ("KR", ("HC05",)), ("kr", ("HC05",)), ("HCG KR", ("HC05",)), ("HCG KR, Bangalore", ("HC05",)),
    ("DR", ("HC06",)), ("North Bangalore", ("HC40",)), ("Nagpur", ("HC38",)), ("Panda", ("HC13",)),
    ("Bangalore", ("HC05", "HC06", "HC40")), ("Gotham", None),
])
def test_resolve_short_names(text, codes):
    from app.ai import hospitals
    assert hospitals.resolve(text) == codes


@pytest.mark.parametrize("question,expected", [
    ("How much did KR spend on restricted brands for scheme patients in Aug-26?", ["HC05"]),
    ("What was compliance at KR.", ["HC05"]),                      # end of a sentence
    ("Which brands did Dr. Rao prescribe?", []),                   # a doctor, not HCG DR
    ("Compare DR and North Bangalore", ["HC06", "HC40"]),
    ("Which city has the best compliance?", []),                    # "city" is not Curie City
    ("Show the kr trend", []),                                      # lower-case two letters are not KR
])
def test_hospitals_mentioned_in_a_question(question, expected):
    from app.ai import hospitals
    got = [c for _a, codes in hospitals.mentioned(question) if len(codes) == 1 for c in codes]
    assert got == expected


def test_scope_guard_refuses_a_kr_question_asked_without_a_hospital():
    from app.ai.deep import tools
    q = "How much did KR spend on preferred brands for scheme patients in Aug-26?"
    msg = tools._kpi_scope_mismatch("brand-rules", "", q, ratio=True)
    assert msg and msg.startswith("UNSCOPED KPI") and "plant='HC05'" in msg
    assert "sum" not in msg.lower()
    assert tools._kpi_scope_mismatch("brand-rules", "HC05", q, ratio=True) is None


def test_dashboard_kpi_accepts_a_short_name_and_refuses_a_city():
    from app.ai import kpi_registry
    assert kpi_registry._dashboard_plant("inventory-turnover-ratio", "KR", True) == "HC05"
    with pytest.raises(ValueError, match="covers 3 hospitals"):
        kpi_registry._dashboard_plant("inventory-turnover-ratio", "Bangalore", True)


def test_report_plants_accept_short_names():
    from app.ai import hcg_reports
    if not __import__("app.ai.hospitals", fromlist=["_rows"])._rows():
        pytest.skip("hospital_map.csv not present")
    try:
        assert hcg_reports._plants("KR") == ["HC05"]
        assert hcg_reports._plants("KR, Nagpur") == ["HC05", "HC38"]
    except ValueError as e:                    # formulary tables not built in this checkout
        pytest.skip(str(e))


def test_get_kpi_plant_description_lists_the_short_names():
    from app.ai import kpi_registry
    desc = kpi_registry.tool_schema()["function"]["parameters"]["properties"]["plant"]["description"]
    assert "KR=HC05" in desc and "Nagpur=HC38" in desc


def test_saving_is_never_offered_back_as_shaving():
    from app.ai import resolve
    q = "Total scheme saving missed at KR for Aug-26 across all suggested brands?"
    typed = {s["typed"] for s in resolve.spelling_suggestions(q, resolve.resolve(q))}
    assert not typed & {"saving", "missed", "scheme", "suggested", "brands"}


# ---------- spend on restricted brands is brand-rules, not the margin-loss report ----------

def test_restricted_spend_for_cash_tpa_points_at_brand_rules():
    from app.ai import resolve
    b = resolve.brief("How much did KR spend on restricted brands for cash and TPA patients in Aug-26?")
    assert "WHICH REPORT" in b and "restricted_cash_tpa" in b


def test_margin_loss_question_does_not_get_the_restricted_note():
    from app.ai import resolve
    b = resolve.brief("What margin did KR lose on cash & TPA patients in Aug-26?")
    assert "WHICH REPORT" not in b


def test_report_descriptions_tell_the_two_apart():
    from app.ai import kpi_registry
    reg = kpi_registry.KPI_REGISTRY
    assert "restricted_cash_tpa" in reg["brand-rules"]["about"]
    assert "NOT the spend on restricted brands" in reg["cash-tpa-margin-loss"]["about"]


def test_there_is_no_drugs_only_segment():
    from app.ai import kpi_registry, page_context
    assert kpi_registry._EXTRA_SCHEMA["segment"]["enum"] == ["onco", "non-onco", "consumables"]
    ctx = page_context.normalise({"page": "spend", "filters": {"segment": "drugs"}})
    assert "segment" not in (ctx.get("filters") or {})
