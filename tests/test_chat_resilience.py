# tests/test_chat_resilience.py — a turn that fails part of the way through.
#
# A live run lost a good, verified answer: the model sent a list where the chart spec asks
# for one column, building the chart raised after the answer had already reached the user,
# and the chat service then saved the exception's text over the answer — kind "error" with
# verified still "canonical", so the UI showed the trust badge on an error. Two halves here:
# the orchestrator draws a chart from any list-shaped spec, and the chat routes keep what
# the user already has when something fails after the answer. Offline: the engine is a
# scripted generator, the model a scripted fake client.
from __future__ import annotations

import json
import re

import pytest

from app.ai import charts
from app.ai import orchestrator as orch
from tests.test_orchestrator_fixes import FakeClient, offline  # noqa: F401  (offline is a fixture)

# ── the chart spec ────────────────────────────────────────────────────────────
_COLS = ["hospital", "plant", "compliance_pct", "out_of_formulary_cost", "formulary_cost"]
_ROWS = [
    {"hospital": "HCG KR", "plant": "HC05", "compliance_pct": 91.2, "out_of_formulary_cost": 4_200_000.0, "formulary_cost": 41_000_000.0},
    {"hospital": "HCG Ahmedabad", "plant": "HC06", "compliance_pct": 88.4, "out_of_formulary_cost": 3_100_000.0, "formulary_cost": 23_500_000.0},
    {"hospital": "HCG Kolkata", "plant": "HC07", "compliance_pct": 94.9, "out_of_formulary_cost": 1_900_000.0, "formulary_cost": 30_200_000.0},
]
_RESULT = {"sql": "SELECT …", "columns": _COLS, "rows": _ROWS, "row_count": len(_ROWS),
           "truncated": False, "purpose": "compliance by hospital"}

# The four shapes the live run's TypeError came from, and what each must become.
_LIST_SHAPED = [
    ({"type": "bar", "x": ["hospital", "plant"], "y": "out_of_formulary_cost"},
     {"type": "bar", "x": "hospital", "y": "out_of_formulary_cost"}),
    ({"type": "grouped_bar", "x": "hospital", "y": ["out_of_formulary_cost"], "color": ["plant"]},
     {"type": "grouped_bar", "x": "hospital", "y": ["out_of_formulary_cost"], "color": "plant"}),
    ({"type": "combo", "x": "hospital", "y": "out_of_formulary_cost", "y2": ["compliance_pct", "plant"]},
     {"type": "combo", "x": "hospital", "y": "out_of_formulary_cost", "y2": "compliance_pct"}),
    ({"type": "grouped_bar", "x": "hospital", "y": [["out_of_formulary_cost", "formulary_cost"]]},
     {"type": "grouped_bar", "x": "hospital", "y": ["out_of_formulary_cost", "formulary_cost"]}),
]


@pytest.mark.parametrize("spec,expected", _LIST_SHAPED)
def test_a_list_shaped_spec_is_normalised_and_draws(spec, expected):
    norm = orch._normalise_chart_spec(spec)
    assert norm == expected
    assert orch._pick_result([_RESULT], norm) is _RESULT
    fig = charts.build(_ROWS, norm)
    assert fig and fig["data"]


@pytest.mark.parametrize("spec,_", _LIST_SHAPED)
def test_pick_result_reads_the_normalised_spec_without_a_type_error(spec, _):
    with pytest.raises(TypeError):     # what the raw spec did, and why the answer was lost
        orch._pick_result([_RESULT], spec)
    orch._pick_result([_RESULT], orch._normalise_chart_spec(spec))


def test_a_well_formed_spec_comes_back_unchanged():
    spec = {"type": "combo", "x": "hospital", "y": ["out_of_formulary_cost", "formulary_cost"],
            "y2": "compliance_pct", "value_format": "inr", "y2_format": "pct",
            "orientation": "v", "title": "Out-of-formulary spend by hospital"}
    before = json.loads(json.dumps(spec))
    assert orch._normalise_chart_spec(spec) == before
    assert spec == before                      # and the model's own dict is not touched


def test_values_that_name_no_column_are_dropped_and_non_specs_skipped():
    assert orch._normalise_chart_spec({"type": ["pie"], "x": [], "y": [[], ""], "title": 5}) == {"type": "pie"}
    assert orch._chart_specs({"charts": ["a bar chart", None, {}]}) == []
    # one spec sent where the list belongs still counts; `chart` only when `charts` gives none
    assert orch._chart_specs({"charts": {"type": "bar", "x": ["hospital"]}}) == [{"type": "bar", "x": "hospital"}]
    assert orch._chart_specs({"charts": [], "chart": {"type": "pie", "x": "hospital"}}) == [{"type": "pie", "x": "hospital"}]


def test_list_shaped_specs_from_present_reach_the_user_as_charts(offline, monkeypatch):  # noqa: F811
    # Through the real answer loop: the model's present() carries all four shapes; every
    # one becomes a chart event, after the answer, and the turn still ends with "done".
    monkeypatch.setattr(orch.warehouse, "run_sql", lambda sql, *a, **k: {k2: v for k2, v in _RESULT.items() if k2 != "purpose"} | {"sql": sql})
    offline(FakeClient(rounds=[
        ("", [("run_sql", {"sql": "SELECT hospital FROM t", "purpose": "compliance by hospital"})]),
        ("HCG KR has the highest out-of-formulary spend of the three hospitals.",
         [("present", {"charts": [s for s, _ in _LIST_SHAPED]})]),
    ]))
    events = list(orch.answer("which hospital has the most out-of-formulary spend?"))
    types = [e["type"] for e in events]
    assert "error" not in types
    assert types.count("chart") == len(_LIST_SHAPED)
    assert types.index("answer") < types.index("chart")
    assert types[-1] == "done"


# ── the chat routes: the error protocol ───────────────────────────────────────
_FIG_A = {"data": [{"type": "bar", "x": ["KR"], "y": [1]}], "layout": {"title": "A"}}
_FIG_B = {"data": [{"type": "pie", "labels": ["KR"], "values": [1]}], "layout": {"title": "B"}}
_TABLE = {"title": "by hospital", "columns": [{"key": "hospital", "label": "hospital", "kind": "text"}],
          "rows": [{"hospital": "KR"}]}
_ANSWER = {"type": "answer", "text": "KR spent the most.", "verified": "canonical",
           "options": ["Break this down by doctor"], "scope": "formulary by hospital"}
_REF = re.compile(r"\(ref [0-9a-f]{8}\)$")
_SECRET = "duckdb: SELECT * FROM fact_secret_table WHERE key='abc'"


def _engine(monkeypatch, *events, raise_after: bool = False):
    def _answer(query, history=None):
        yield {"type": "step", "text": "Thinking it through"}
        yield from events
        if raise_after:
            raise RuntimeError(_SECRET)
        yield {"type": "done"}
    monkeypatch.setattr("app.ai.orchestrator.answer", _answer)


def _sse(response):
    out = []
    for line in response.iter_lines():
        if line.startswith("data: "):
            out.append(json.loads(line[len("data: "):]))
    return out


def _stream(client, headers, query="q"):
    session = client.post("/chat/sessions", json={}, headers=headers).json()
    with client.stream("POST", f"/chat/sessions/{session['id']}/messages/stream",
                       json={"query": query}, headers=headers) as r:
        assert r.status_code == 200
        events = _sse(r)
    stored = client.get(f"/chat/sessions/{session['id']}", headers=headers).json()["messages"][-1]
    return events, stored["content"]


def _post(client, headers, query="q"):
    session = client.post("/chat/sessions", json={}, headers=headers).json()
    r = client.post(f"/chat/sessions/{session['id']}/messages", json={"query": query}, headers=headers)
    assert r.status_code == 200, r.text
    return r.json()["assistant_message"]["content"]


def test_an_exception_before_the_answer_is_a_generic_error_with_no_badge(client, admin_headers, monkeypatch, caplog):
    _engine(monkeypatch, {"type": "sql", "purpose": "p", "sql": "SELECT 1", "rows": 1, "error": None},
            raise_after=True)
    with caplog.at_level("ERROR", logger="app.services.chat_service"):
        events, saved = _stream(client, admin_headers)
    errors = [e for e in events if e["type"] == "error"]
    assert len(errors) == 1 and "partial" not in errors[0]
    assert errors[0]["text"].startswith("Sorry — something went wrong answering that.")
    assert _REF.search(errors[0]["text"])
    assert _SECRET not in json.dumps(events)
    assert saved["kind"] == "error" and saved["verified"] is None
    assert saved["chart"] is None and saved["charts"] == [] and saved["table"] is None
    assert saved["text"] == errors[0]["text"]
    # the same ref is in the server log, with the traceback that the user never sees
    ref = _REF.search(errors[0]["text"]).group(0)[5:-1]
    logged = [r for r in caplog.records if ref in r.getMessage()]
    assert logged and logged[0].exc_info and _SECRET in str(logged[0].exc_info[1])


def test_an_exception_after_the_answer_keeps_the_answer(client, admin_headers, monkeypatch):
    _engine(monkeypatch, _ANSWER, {"type": "chart", "plotly": _FIG_A},
            {"type": "table", "table": _TABLE, "note": ""}, raise_after=True)
    events, saved = _stream(client, admin_headers)
    errors = [e for e in events if e["type"] == "error"]
    assert len(errors) == 1 and errors[0]["partial"] is True
    assert errors[0]["text"].startswith("Part of this answer couldn't be completed")
    assert _REF.search(errors[0]["text"]) and _SECRET not in json.dumps(events)
    assert saved["kind"] == "answer"
    assert saved["text"] == _ANSWER["text"] and saved["verified"] == "canonical"
    assert saved["options"] == _ANSWER["options"] and saved["scope"] == _ANSWER["scope"]
    assert saved["charts"] == [_FIG_A] and saved["chart"] == _FIG_A
    assert saved["table"]["rows"] == _TABLE["rows"]
    assert saved["notice"] == errors[0]["text"]


def test_an_engine_error_event_after_the_answer_is_a_partial_error(client, admin_headers, monkeypatch):
    _engine(monkeypatch, _ANSWER, {"type": "chart", "plotly": _FIG_A},
            {"type": "error", "text": f"chart failed: {_SECRET}"})
    events, saved = _stream(client, admin_headers)
    errors = [e for e in events if e["type"] == "error"]
    assert len(errors) == 1 and errors[0]["partial"] is True and _SECRET not in json.dumps(events)
    assert saved["kind"] == "answer" and saved["text"] == _ANSWER["text"]
    assert saved["verified"] == "canonical" and saved["charts"] == [_FIG_A]
    assert saved["notice"] == errors[0]["text"]


def test_an_engine_error_event_before_the_answer_keeps_its_own_words(client, admin_headers, monkeypatch):
    # The engine writes these for the user ("isn't configured", "rejected the key").
    _engine(monkeypatch, {"type": "error", "text": "The AI Analyst isn't configured."})
    events, saved = _stream(client, admin_headers)
    assert [e["text"] for e in events if e["type"] == "error"] == ["The AI Analyst isn't configured."]
    assert saved["kind"] == "error" and saved["verified"] is None and "notice" not in saved


def test_the_real_engine_never_words_its_own_error_from_an_exception(client, admin_headers, monkeypatch, offline, caplog):  # noqa: F811
    # The engine's own error words are passed on as they are, so the engine must never build
    # them from an exception: a schema context that fails to build (a DuckDB or file error)
    # once reached the user, and the saved chat, as "AI service unavailable: <its text>".
    def _broken():
        raise RuntimeError(_SECRET)
    monkeypatch.setattr(orch.semantics, "context", _broken)
    with caplog.at_level("ERROR", logger="app.services.chat_service"):
        events, saved = _stream(client, admin_headers)
    errors = [e for e in events if e["type"] == "error"]
    assert len(errors) == 1 and errors[0]["text"].startswith("Sorry — something went wrong answering that.")
    assert _SECRET not in json.dumps(events) and _SECRET not in json.dumps(saved)
    assert saved["kind"] == "error" and saved["verified"] is None
    ref = _REF.search(errors[0]["text"]).group(0)[5:-1]
    assert any(ref in r.getMessage() and r.exc_info for r in caplog.records)
    assert _SECRET not in json.dumps(_post(client, admin_headers))
    assert _SECRET not in json.dumps(_ai_chat(client, admin_headers))


class _ServiceErr(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status_code = status
        self.body = {"message": message}


@pytest.mark.parametrize("status", [400, 401, 404, 418, 429, 503])
def test_the_model_services_error_body_stays_in_the_log(status, caplog):
    # A 400's body once went to the user ("refused the request (error 400): <body>"); it can
    # quote the prompt back. The user gets the wording and a ref; the log gets the body.
    with caplog.at_level("WARNING", logger="app.ai.orchestrator"):
        text = orch._llm_error_text(_ServiceErr(status, _SECRET))
    assert _SECRET not in text and _REF.search(text)
    ref = _REF.search(text).group(0)[5:-1]
    assert any(ref in r.getMessage() and _SECRET in r.getMessage() for r in caplog.records)


def test_every_chart_of_a_turn_is_saved_in_order(client, admin_headers, monkeypatch):
    _engine(monkeypatch, _ANSWER, {"type": "chart", "plotly": _FIG_A}, {"type": "chart", "plotly": _FIG_B})
    for saved in (_stream(client, admin_headers)[1], _post(client, admin_headers)):
        assert saved["charts"] == [_FIG_A, _FIG_B]
        assert saved["chart"] == _FIG_B           # the last one, for readers older than "charts"
        assert saved["kind"] == "answer" and "notice" not in saved


@pytest.mark.parametrize("text", ["", "   ", None])
def test_an_empty_answer_is_no_answer_on_both_routes(client, admin_headers, monkeypatch, text):
    _engine(monkeypatch, {**_ANSWER, "text": text}, {"type": "chart", "plotly": _FIG_A},
            {"type": "table", "table": _TABLE, "note": ""})
    events, streamed = _stream(client, admin_headers)
    posted = _post(client, admin_headers)
    errors = [e for e in events if e["type"] == "error"]
    assert len(errors) == 1 and "partial" not in errors[0] and _REF.search(errors[0]["text"])
    for saved in (streamed, posted):
        assert saved["kind"] == "error" and saved["verified"] is None
        assert saved["charts"] == [] and saved["chart"] is None and saved["table"] is None
        assert saved["text"].startswith("Sorry — something went wrong answering that.")


def test_the_blocking_route_keeps_the_answer_after_an_exception(client, admin_headers, monkeypatch):
    _engine(monkeypatch, _ANSWER, {"type": "chart", "plotly": _FIG_A}, raise_after=True)
    saved = _post(client, admin_headers)
    assert saved["kind"] == "answer" and saved["verified"] == "canonical"
    assert saved["text"] == _ANSWER["text"] and saved["charts"] == [_FIG_A]
    assert saved["notice"].startswith("Part of this answer couldn't be completed") and _SECRET not in json.dumps(saved)


def _ai_chat(client, headers):
    with client.stream("POST", "/ai/chat", json={"query": "q", "history": []}, headers=headers) as r:
        assert r.status_code == 200
        return [e for e in _sse(r) if e["type"] != "end"]


def test_ai_chat_never_sends_the_exception_text(client, admin_headers, monkeypatch):
    _engine(monkeypatch, raise_after=True)
    events = _ai_chat(client, admin_headers)
    errors = [e for e in events if e["type"] == "error"]
    assert len(errors) == 1 and "partial" not in errors[0]
    assert errors[0]["text"].startswith("Sorry — something went wrong answering that.")
    assert _REF.search(errors[0]["text"]) and _SECRET not in json.dumps(events)


def test_ai_chat_sends_a_partial_error_after_the_answer(client, admin_headers, monkeypatch):
    _engine(monkeypatch, _ANSWER, raise_after=True)
    events = _ai_chat(client, admin_headers)
    assert [e["type"] for e in events] == ["step", "answer", "error"]
    assert events[1] == _ANSWER                          # sent on untouched
    assert events[2]["partial"] is True and _SECRET not in json.dumps(events)


def test_an_engine_error_event_with_no_words_is_one_generic_error(client, admin_headers, monkeypatch):
    # It was sent on with text null, and then finish() sent the generic error as well.
    _engine(monkeypatch, {"type": "error", "text": None})
    events, saved = _stream(client, admin_headers)
    errors = [e for e in events if e["type"] == "error"]
    assert len(errors) == 1 and "partial" not in errors[0]
    assert errors[0]["text"].startswith("Sorry — something went wrong answering that.")
    assert _REF.search(errors[0]["text"])
    assert saved["kind"] == "error" and saved["text"] == errors[0]["text"]
    assert len([e for e in _ai_chat(client, admin_headers) if e["type"] == "error"]) == 1


# ── one chart that fails costs only itself ────────────────────────────────────
def test_a_chart_that_fails_leaves_the_other_charts_and_the_table(client, admin_headers, monkeypatch,
                                                                   offline, caplog):  # noqa: F811
    # charts.build raising for one chart used to leave the loop before the table was sent, so
    # the turn lost every chart after it and the table too.
    monkeypatch.setattr(orch.warehouse, "run_sql", lambda sql, *a, **k: {k2: v for k2, v in _RESULT.items() if k2 != "purpose"} | {"sql": sql})
    real_build = charts.build

    def build(rows, spec):
        if spec.get("type") == "pie":
            raise RuntimeError(_SECRET)
        return real_build(rows, spec)
    monkeypatch.setattr(orch.charts, "build", build)
    specs = [{"type": "pie", "x": "hospital", "y": "out_of_formulary_cost"},
             {"type": "bar", "x": "hospital", "y": "out_of_formulary_cost"}]
    offline(FakeClient(rounds=[
        ("", [("run_sql", {"sql": "SELECT hospital FROM t", "purpose": "compliance by hospital"})]),
        ("HCG KR has the highest out-of-formulary spend of the three hospitals.",
         [("present", {"charts": specs})]),
    ]))
    with caplog.at_level("ERROR"):
        events, saved = _stream(client, admin_headers)
    types = [e["type"] for e in events]
    assert types.count("chart") == 1 and types.count("table") == 1 and types.count("error") == 1
    assert types.index("answer") < types.index("chart") < types.index("table") < types.index("error")
    error = events[types.index("error")]
    assert error["partial"] is True and _REF.search(error["text"]) and _SECRET not in json.dumps(events)
    assert events[types.index("chart")]["plotly"]["data"][0]["type"] == "bar"
    assert saved["kind"] == "answer" and saved["verified"] is not None
    assert len(saved["charts"]) == 1 and len(saved["table"]["rows"]) == len(_ROWS)
    assert saved["notice"] == error["text"] and _SECRET not in json.dumps(saved)
    # the ref the user is shown is the one logged with the chart's traceback
    ref = _REF.search(error["text"]).group(0)[5:-1]
    assert any(ref in r.getMessage() and r.exc_info and _SECRET in str(r.exc_info[1])
               for r in caplog.records)
