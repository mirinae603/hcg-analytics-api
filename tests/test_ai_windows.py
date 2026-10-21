# tests/test_ai_windows.py — the AI reads the months it talks about from the data.
#
# Its prompts used to say "Jun–Aug 2026 ONLY", "Dec-2025 … May-2026" and DATE '2026-05-31' in
# so many words, so a September upload would have left the model telling users September does
# not exist. The texts now carry <<TOKENS>> that app/ai/windows.py fills from the published
# tables at the moment a request is sent.
from __future__ import annotations

import io
import re
import tokenize
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from app.ai import windows
from app.core.config import settings

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def data(tmp_path, monkeypatch):
    for name in ("FORMULARY_DATA_DIR", "KPI_DIR", "CURATED_DIR"):
        (tmp_path / name).mkdir()
        monkeypatch.setattr(settings, name, str(tmp_path / name))

    def write(reports=(), sales=(), snapshot=None):
        if reports:
            pd.DataFrame({"period": list(reports)}).to_parquet(tmp_path / "FORMULARY_DATA_DIR" / "formulary_summary.parquet")
        if sales:
            pd.DataFrame({"month": list(sales)}).to_parquet(tmp_path / "KPI_DIR" / "sales_monthly.parquet")
        if snapshot:
            pd.DataFrame({"snapshot_date": [datetime.fromisoformat(snapshot)]}).to_parquet(
                tmp_path / "CURATED_DIR" / "fact_inventory.parquet")
    return write


def test_the_months_come_from_the_tables(data):
    data(reports=["2026-06", "2026-07", "2026-08"],
         sales=["2025-12", "2026-01", "2026-02", "2026-03", "2026-04", "2026-05"], snapshot="2026-05-31")
    v = windows.values()
    assert v["REPORTS"] == "Jun–Aug 2026" and v["REPORTS_RANGE"] == "2026-06..2026-08"
    assert v["REPORTS_LATEST"] == "Aug-2026"
    assert v["SALES"] == "Dec-2025…May-2026" and v["SALES_SHORT"] == "Dec–May" and v["SALES_MONTHS"] == "6"
    assert (v["SALES_START"], v["SALES_END"]) == ("2025-12-01", "2026-06-01")
    assert v["SALES_YEARS"] == "'2025' covers Dec only (1 month), '2026' covers Jan–May only (5 months)"
    assert (v["SNAPSHOT"], v["SNAPSHOT_SQL"]) == ("2026-05-31", "DATE '2026-05-31'")


def test_a_new_month_changes_the_text_with_no_code_change(data):
    data(reports=["2026-06", "2026-07", "2026-08"])
    assert windows.fill("<<REPORTS>> ONLY") == "Jun–Aug 2026 ONLY"
    data(reports=["2026-06", "2026-07", "2026-08", "2026-09"])          # September is published
    assert windows.fill("<<REPORTS>> ONLY") == "Jun–Sep 2026 ONLY"
    assert windows.fill("default (<<REPORTS_LATEST>>)") == "default (Sep-2026)"


def test_a_window_across_years_and_one_month():
    assert windows.span(("2025-12", "2026-01", "2026-02")) == "Dec 2025–Feb 2026"
    assert windows.span(("2026-09",)) == "Sep 2026"
    assert windows.span(()) == ""


def test_missing_tables_give_neutral_words_never_a_date(data):
    v = windows.values()
    assert v["REPORTS"] == "the report months" and v["SALES"] == "the sales months"
    assert v["SNAPSHOT_SQL"] == "(SELECT MAX(snapshot_date)::DATE FROM fact_inventory)"
    assert not re.search(r"20\d\d", " ".join(v.values()))


def test_a_request_is_filled_messages_and_tools(data):
    data(reports=["2026-06", "2026-08"], sales=["2025-12", "2026-05"], snapshot="2026-05-31")
    req = {"model": "m", "messages": [
        {"role": "system", "content": "Reports <<REPORTS>>; sales <<SALES>>; expiry vs <<SNAPSHOT_SQL>>"},
        {"role": "assistant", "content": None, "tool_calls": []},
        {"role": "user", "content": [{"type": "text", "text": "window <<REPORTS_RANGE>>"}]}],
        "tools": [{"type": "function", "function": {"name": "get_kpi", "description": "<<REPORTS>> ONLY",
                                                    "parameters": {"properties": {"month": {"description": "(<<REPORTS_LATEST>>)"}}}}}]}
    out = windows.fill_request(req)
    assert out["messages"][0]["content"] == "Reports Jun–Aug 2026; sales Dec-2025…May-2026; expiry vs DATE '2026-05-31'"
    assert out["messages"][1] == req["messages"][1]
    assert out["messages"][2]["content"][0]["text"] == "window 2026-06..2026-08"
    fn = out["tools"][0]["function"]
    assert fn["description"] == "Jun–Aug 2026 ONLY"
    assert fn["parameters"]["properties"]["month"]["description"] == "(Aug-2026)"
    assert "<<" in req["messages"][0]["content"]                         # the caller's copy is untouched


def test_every_ai_text_the_model_gets_is_filled():
    from app.ai import kpi_registry, orchestrator, resolve, routing, semantics
    texts = [str(kpi_registry.tool_schema()), orchestrator.SYSTEM, orchestrator._FOLLOWUP_SYS,
             str(routing._CORPUS), semantics.RULES + semantics.GUIDE + semantics.GOTCHAS + semantics.EXAMPLES,
             resolve.brief("What was formulary compliance at KR and the sales trend of Keytruda?")]
    filled = windows.fill_request({"messages": [{"role": "system", "content": t} for t in texts]})
    for m in filled["messages"]:
        left = set(windows.TOKEN.findall(m["content"]))
        assert not left, left


# Months written into an AI string (not a comment or docstring) would go stale with the data.
_FIXED = re.compile(r"Jun[–-]Aug|Dec[–-]May|Dec-2025|May-2026|Aug-2026|2026-05-31|2026-06\.\.2026-08|"
                    r"\b(?:Jun|Jul|Aug)-26\b|\b6-month\b|\b6 MONTHS\b")


def _strings(path: Path):
    src = path.read_text()
    toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    kinds = {tokenize.STRING} | ({tokenize.FSTRING_MIDDLE} if hasattr(tokenize, "FSTRING_MIDDLE") else set())
    for i, t in enumerate(toks):
        if t.type not in kinds:
            continue
        j = i - 1
        while j >= 0 and toks[j].type in (tokenize.NL, tokenize.COMMENT):
            j -= 1
        k = i + 1
        while k < len(toks) and toks[k].type in (tokenize.NL, tokenize.COMMENT):
            k += 1
        docstring = (t.type == tokenize.STRING
                     and (j < 0 or toks[j].type in (tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT))
                     and (k >= len(toks) or toks[k].type in (tokenize.NEWLINE, tokenize.ENDMARKER)))
        if not docstring:
            yield t.start[0], t.string


def test_no_month_is_written_into_an_ai_text():
    found = []
    for path in sorted((ROOT / "app" / "ai").rglob("*.py")):
        if path.name == "windows.py":
            continue
        for line, s in _strings(path):
            if _FIXED.search(s):
                found.append(f"{path.relative_to(ROOT)}:{line}: {_FIXED.search(s).group(0)}")
    assert not found, "\n".join(found)
