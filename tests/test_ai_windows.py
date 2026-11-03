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


def test_each_transaction_source_has_its_own_window(data, tmp_path):
    # billing ends first, the receipts later, and consumption has a month with no extract
    data(sales=["2025-12", "2026-01", "2026-02", "2026-03", "2026-04", "2026-05"])
    cur = tmp_path / "CURATED_DIR"
    dates = lambda *ms: pd.to_datetime([f"{m}-15" for m in ms])
    pd.DataFrame({"posting_date": dates("2025-12", "2026-03", "2026-08")}).to_parquet(cur / "fact_grn.parquet")
    pd.DataFrame({"posting_date": dates("2025-12", "2026-07", "2026-09")}).to_parquet(cur / "fact_consumption.parquet")
    v = windows.values()
    assert v["RECEIPTS"] == "Dec-2025…Aug-2026 (no Jan-2026, Feb-2026, Apr-2026, May-2026, Jun-2026, Jul-2026)"
    assert v["CONSUMPTION"].startswith("Dec-2025…Sep-2026 (no Jan-2026") and "Aug-2026)" in v["CONSUMPTION"]
    assert v["ORDERS"] == "the purchase orders months"                     # fact_po not written
    assert [s.label for s, _m in windows.transaction_windows()] == ["sales", "goods receipts", "internal consumption"]
    assert windows.missing_months(("2026-06", "2026-07", "2026-09")) == ("2026-08",)
    assert windows.missing_months(()) == ()
    assert windows.month_end("2026-02") == "2026-02-28" and windows.month_end("2025-12") == "2025-12-31"


def test_the_quoted_totals_come_from_the_tables(data, tmp_path):
    from app.ai import figures
    pd.DataFrame({"purchase_value": [6e9, 2.0690835741e9]}).to_parquet(tmp_path / "KPI_DIR" / "kpi_purchase_value.parquet")
    pd.DataFrame({"cgst_value": [1e7, None], "sgst_value": [1e7, 0.0], "igst_value": [5e6, 0.0]}).to_parquet(
        tmp_path / "CURATED_DIR" / "fact_po.parquet")
    pd.DataFrame({"total_amount_wo_tax": [5e9, 1e9], "total_mrp_value": [1.5e10, 1e9],
                  "discount_value": [0.0, 2e6]}).to_parquet(tmp_path / "CURATED_DIR" / "fact_grn.parquet")
    v = figures.values()
    assert (v["PROCUREMENT_TOTAL"], v["PROCUREMENT_GST"]) == ("₹806.91 Cr", "₹2.50 Cr")
    assert (v["RECEIVED_TOTAL"], v["RECEIVED_MRP"], v["GRN_DISCOUNT"]) == ("₹600.00 Cr", "₹1,600.00 Cr", "₹0.20 Cr")
    assert v["GRN_DISCOUNT_LINES"] == "50.0%"
    assert windows.fill("spend <<PROCUREMENT_TOTAL>>") == "spend ₹806.91 Cr"


def test_the_turnover_recipe_annualises_each_part_over_its_own_months(data, tmp_path):
    # A 31-Aug-26 count: issues in Mar, Apr, Jun, Jul and on the count day (5 months up to it, no
    # May extract) and one in Sep, after it; billing Jan-Mar (3 months). One "x 2.0" would be wrong
    # for both parts; the recipe takes 12/5 for the issues and 12/3 for the billing, as the
    # dashboard's data_access.yearly_cogs does.
    from app.ai import routing, semantics
    data(sales=["2026-01", "2026-02", "2026-03"], snapshot="2026-08-31")
    days = pd.to_datetime(["2026-03-10", "2026-04-15", "2026-06-01", "2026-07-20", "2026-08-31", "2026-09-05"])
    pd.DataFrame({"posting_date": days}).to_parquet(tmp_path / "CURATED_DIR" / "fact_consumption.parquet")
    assert windows.stock_months() == ("2026-03", "2026-04", "2026-06", "2026-07", "2026-08")
    assert windows.values()["STOCK_MONTHS"] == "5"
    recipe = windows.fill(semantics.EXAMPLES)
    assert "(internal.c * 12.0 / 5 + billed.c * 12.0 / 3) / internal.inv AS portfolio_itr" in recipe
    hint = windows.fill(" ".join(h for _, h in routing._CORPUS))
    assert "internal × 12/5, billed × 12/3" in hint
    assert not re.search(r"[×*] ?2\.0", recipe + hint)                 # no single fixed factor left


def test_the_gotchas_name_each_sources_own_window(data, tmp_path):
    # Billing Dec-May (6 months), receipts Dec-Aug, orders Dec-Sep, and a 31-Aug count over the
    # issues of Mar, Apr, Jun, Jul and Aug (5 months, no May extract). The "no rows is not news"
    # gotcha names each source's own months, the non-moving one the 5 months up to the count, and
    # the table guide's sales_by_material the 6 months of billing — not one "6-month window".
    from app.ai import semantics
    data(sales=["2025-12", "2026-01", "2026-02", "2026-03", "2026-04", "2026-05"], snapshot="2026-08-31")
    cur = tmp_path / "CURATED_DIR"
    dates = lambda *ms: pd.to_datetime([f"{m}-15" for m in ms])
    year = [f"2026-{m:02d}" for m in range(1, 10)]                        # Jan..Sep 2026
    pd.DataFrame({"posting_date": dates("2025-12", *year[:8])}).to_parquet(cur / "fact_grn.parquet")
    pd.DataFrame({"posting_date": dates("2025-12", *year)}).to_parquet(cur / "fact_po.parquet")
    pd.DataFrame({"posting_date": dates("2026-03", "2026-04", "2026-06", "2026-07", "2026-08", "2026-09")}
                 ).to_parquet(cur / "fact_consumption.parquet")
    gotchas, guide = windows.fill(semantics.GOTCHAS), windows.fill(semantics.GUIDE)
    assert ("the transaction tables only cover their own windows (billed sales Dec-2025…May-2026, purchase orders "
            "Dec-2025…Sep-2026, goods receipts Dec-2025…Aug-2026, internal consumption Mar-2026…Sep-2026 (no "
            "May-2026)") in gotchas
    assert ('"Non-moving" means ZERO consumption in the 5 months of issues up to the stock count (2026-08-31)'
            in gotchas)
    assert "REVENUE / MARGIN (billed sales, 6mo):" in guide and "by product, 6-MONTH TOTAL" in guide
    assert "6-month analytics window" not in gotchas and "-month window" not in gotchas


def test_the_stock_window_without_a_count_or_without_issues(data, tmp_path):
    assert windows.stock_months() == () and windows.values()["STOCK_MONTHS"] == "few"
    days = pd.to_datetime(["2026-05-10", "2026-09-05"])
    pd.DataFrame({"posting_date": days}).to_parquet(tmp_path / "CURATED_DIR" / "fact_consumption.parquet")
    assert windows.stock_months() == ("2026-05", "2026-09")             # no stock count: every issue


def test_a_window_across_years_and_one_month():
    assert windows.span(("2025-12", "2026-01", "2026-02")) == "Dec 2025–Feb 2026"
    assert windows.span(("2026-09",)) == "Sep 2026"
    assert windows.span(()) == ""


def test_missing_tables_give_neutral_words_never_a_date(data):
    v = windows.values()
    assert v["REPORTS"] == "the report months" and v["SALES"] == "the sales months"
    assert v["SNAPSHOT_SQL"] == "(SELECT MAX(snapshot_date)::DATE FROM fact_inventory)"
    assert (v["RECEIPTS"], v["PROCUREMENT_TOTAL"]) == ("the goods receipts months", "the procurement total")
    assert not re.search(r"20\d\d|₹", " ".join(v.values()))


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
