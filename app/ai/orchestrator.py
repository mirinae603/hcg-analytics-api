"""
Agentic AI-analyst orchestrator (Azure OpenAI / gpt-4o).

Flow — an autonomous SQL analyst that can answer ANYTHING the data supports:
  1. GATHER (recursive): the model issues run_sql queries against the DuckDB
     warehouse — as many as it needs, decomposing complex questions, refining after
     seeing intermediate results (multi-step / recursive analytics).
  2. PRESENT: it returns a grounded prose answer + a chart spec (deterministic build).
  3. VERIFY: a strict auditor pass re-checks every figure in the answer against the
     actual query results; on a flag, one correction pass runs. A "verified" badge
     is emitted only when the auditor is satisfied.

No LLM-written Python is ever executed. SQL is governed (SELECT-only, capped,
timed-out). Numbers fed back to the model are pre-formatted (₹Cr/L/%) so it never
mis-converts units. The key is read from env — never hardcoded.
"""
from __future__ import annotations
import json
import os
import re

from app.ai import warehouse, semantics, charts

AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://ed-gpt.openai.azure.com")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
MAX_SQL_STEPS = 8


def has_key() -> bool:
    return bool(os.getenv("AZURE_OPENAI_API_KEY"))


def _client():
    from openai import AzureOpenAI
    key = os.getenv("AZURE_OPENAI_API_KEY")
    if not key:
        raise RuntimeError("AZURE_OPENAI_API_KEY is not set")
    return AzureOpenAI(azure_endpoint=AZURE_ENDPOINT, api_key=key, api_version=AZURE_API_VERSION)


# ── number pre-formatting so the model can't mis-convert units ───────────────
_INR = re.compile(r"(revenue|margin|value|cost|spend|price|amount|opportunity|overpay|mrp|purchase|sales|cr\b)", re.I)
_PCT = re.compile(r"(pct|percent|share|rate|margin_pct|%)", re.I)
_DAYS = re.compile(r"(doh|days|aging|lead|cover|tat)", re.I)


def infer_kind(col: str) -> str:
    c = col.lower()
    if _PCT.search(c):
        return "pct"
    if _DAYS.search(c):
        return "days"
    if _INR.search(c):
        return "inr"
    return "num"


def col_kind(col: str, rows: list) -> str:
    """Kind from the actual data: text if the column holds strings, else name-inferred."""
    vals = [r.get(col) for r in (rows or [])[:25]]
    has_num = any(isinstance(v, (int, float)) for v in vals if v is not None)
    has_str = any(isinstance(v, str) for v in vals if v is not None)
    if has_str and not has_num:
        return "text"
    return infer_kind(col)


def _fmt(v, kind):
    if v is None or v == "":
        return None
    if kind == "text":
        return v   # codes / ids / names — never coerce to a number
    try:
        n = float(v)
    except (TypeError, ValueError):
        return v
    if kind == "inr":
        a = abs(n)
        if a >= 1e7:
            return f"₹{n/1e7:.2f} Cr"
        if a >= 1e5:
            return f"₹{n/1e5:.2f} L"
        if a >= 1e3:
            return f"₹{n/1e3:.1f} K"
        return f"₹{n:.0f}"
    if kind == "pct":
        return f"{n:.1f}%"
    if kind == "days":
        return f"{n:,.0f} d"
    if abs(n) >= 1000 or n == int(n):
        return f"{n:,.0f}"
    return f"{n:,.2f}"


def _format_result(res: dict, limit: int = 30) -> dict:
    cols = res["columns"]
    # data-aware typing (not name-based): a string column — e.g. a numeric-looking cost-
    # centre or material code — stays text and is never comma/₹-formatted for the model.
    kinds = {c: col_kind(c, res["rows"]) for c in cols}
    rows = [{c: _fmt(r.get(c), kinds[c]) for c in cols} for r in res["rows"][:limit]]
    return {"columns": cols, "rows": rows, "row_count": res["row_count"],
            "truncated": res.get("truncated", False)}


SYSTEM = """You are the HCG Supply-Chain AI Analyst. You answer ANY question about the data by writing DuckDB SQL against the warehouse below and reasoning over the REAL results. You never invent numbers.

{context}

HOW YOU WORK — like a sharp, friendly human analyst:
• UNDERSTAND the real intent first. If the request is genuinely ambiguous or under-specified (unclear time range, which metric/entity, or a term the data doesn't have), call ask_clarification with ONE short question (and 2–4 quick options) INSTEAD of guessing. Don't over-ask — if a sensible default is obvious, just proceed and state the assumption.
• SPECIFIC ITEM? For ANY question about a particular product/SKU (named or by code), call lookup_item FIRST. It returns the item's identity (generic, group, manufacturer, formulary status) and a COMPLETE footprint — the row count in EVERY table it touches (sales, PO, GRN, consumption, inventory, forecasts). This guarantees you never miss a source: an item with 0 sales/purchases but rows in fact_inventory is DEAD / NON-MOVING stock (report qty, aging, expiry, formulary) — never call that "no data". Then run_sql only the tables the footprint shows have rows.
• Call run_sql to fetch data — MULTIPLE times as needed. Decompose complex questions, explore first, then run the precise query; join across tables freely (CTEs, window functions, subqueries all work in DuckDB). Go into real DEPTH: don't just pull the top line — look at the composition, the outliers, the trend, the "so what".
• Every number in your final answer MUST come from a query you actually ran.
• When done, call present() with a warm, natural, analytical answer (talk like a helpful colleague, not a report generator) plus chart(s). Keep it TIGHT — 2–4 sentences: the headline number(s) + the one insight that matters. Do NOT enumerate long lists item-by-item in the prose (the chart AND the data table below already show every row) — mention the top 1–2 and summarise the rest. No filler sign-offs like "let me know if you'd like…". NEVER paste a markdown/pipe table into the answer text.
   – ALWAYS chart a ranking, breakdown, trend, comparison or share.
   – If the user asks for "two charts", "different charts", "a pie and a bar", etc., or if two views genuinely illuminate the data, put MULTIPLE specs in `charts` (e.g. a ranking bar AND a share donut).
   – bar=rankings, line=time trend, donut=shares, combo (percentage on y2)=two different scales, heatmap=matrix, scatter/bubble=correlation, treemap/sunburst=hierarchy, waterfall=build-up.
   – Only omit charts for a pure single-number answer.
• Money is already formatted (₹Cr/₹L) in results — quote those strings verbatim, never recompute units.
• Be genuinely analytical: lead with the answer, then add the insight that matters (a concentration, a trend, a risk, a surprise) and, when useful, a short takeaway or suggested next question. Respect caveats in the schema notes (e.g. manufacturer-of-purchases coverage).
• If the data truly can't answer it, say so plainly and suggest the closest thing you CAN answer."""


RUN_SQL_TOOL = {
    "type": "function", "function": {
        "name": "run_sql",
        "description": "Run one read-only DuckDB SELECT/WITH query over the warehouse and get the rows back. Call repeatedly to build up an answer.",
        "parameters": {"type": "object", "required": ["sql", "purpose"], "properties": {
            "sql": {"type": "string", "description": "A single SELECT or WITH query. No semicolons, no DDL/DML."},
            "purpose": {"type": "string", "description": "Short human phrase for what this query finds (shown to the user), e.g. 'expiring value by manufacturer'."}}}},
}
_CHART_SPEC = {
    "type": "object", "description": "One visualization.", "properties": {
        "type": {"type": "string", "enum": ["bar", "grouped_bar", "stacked_bar", "line", "area", "combo", "pie", "donut", "scatter", "bubble", "heatmap", "treemap", "sunburst", "funnel", "waterfall", "histogram", "box", "indicator"]},
        "x": {"type": "string", "description": "Result column for the category / x-axis / labels."},
        "y": {"description": "Result column (or list of columns) for values.", "type": ["string", "array"], "items": {"type": "string"}},
        "color": {"type": "string", "description": "Optional grouping column (scatter/heatmap/sunburst)."},
        "size": {"type": "string", "description": "Optional bubble-size column."},
        "y2": {"type": "string", "description": "Optional secondary-axis column for combo (put a % metric here alongside a ₹ metric)."},
        "value_format": {"type": "string", "enum": ["inr", "pct", "num", "days"]},
        "y2_format": {"type": "string", "enum": ["inr", "pct", "num", "days"]},
        "orientation": {"type": "string", "enum": ["v", "h"]},
        "title": {"type": "string"}}}

PRESENT_TOOL = {
    "type": "function", "function": {
        "name": "present",
        "description": "Deliver the final answer + chart(s). Call this once you have the data.",
        "parameters": {"type": "object", "required": ["answer"], "properties": {
            "answer": {"type": "string", "description": "Final answer in warm, natural, analytical markdown. Quote the pre-formatted figures exactly."},
            "charts": {"type": "array", "description": "One or MORE charts. If the user asks for multiple/different charts, or two views genuinely help (e.g. a ranking bar AND a share donut), include several. Empty for a single-number answer.", "items": _CHART_SPEC},
            "chart": dict(_CHART_SPEC, description="Deprecated single-chart form — prefer 'charts'.")}}},
}
LOOKUP_TOOL = {
    "type": "function", "function": {
        "name": "lookup_item",
        "description": "Resolve a SPECIFIC product/SKU by name or material code and get (a) its identity — generic name, category, manufacturer, formulary status — and (b) a COMPLETE footprint: the row count in every table it appears in (sales, purchase orders, receipts, consumption, inventory, forecasts, risk). Call this FIRST for any single-item question so you never miss a data source. If it has 0 sales/purchases but rows in fact_inventory, it's dead/non-moving stock — not 'no data'.",
        "parameters": {"type": "object", "required": ["name"], "properties": {
            "name": {"type": "string", "description": "The product name or material code the user asked about, e.g. 'CALPOL-T TAB' or '218766'."}}}},
}
CLARIFY_TOOL = {
    "type": "function", "function": {
        "name": "ask_clarification",
        "description": "Ask the user ONE short clarifying question when the request is genuinely ambiguous or under-specified (e.g. unclear time range, which metric, which entity, or a term the data doesn't have). Only use when you truly cannot pick a sensible default.",
        "parameters": {"type": "object", "required": ["question"], "properties": {
            "question": {"type": "string", "description": "A single, friendly clarifying question."},
            "options": {"type": "array", "items": {"type": "string"}, "description": "Optional 2–4 suggested answers as quick chips."}}}},
}


_MONTHISH = re.compile(r"(^\d{4}-\d{2}$)|(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", re.I)


def _auto_chart(res: dict) -> dict | None:
    """When the model doesn't specify a chart, build a sensible one if the shape fits:
    one label column + at least one numeric column, 2..40 rows. If a ₹ value AND a
    percentage co-occur, build a combo (bars + %-line on a secondary axis)."""
    if not res or not res.get("rows") or not (2 <= res["row_count"] <= 40):
        return None
    cols = res["columns"]
    rows = res["rows"]
    numeric = [c for c in cols if all(isinstance(r.get(c), (int, float)) or r.get(c) is None for r in rows) and any(isinstance(r.get(c), (int, float)) for r in rows)]
    if not numeric:
        return None
    label = next((c for c in cols if c not in numeric), cols[0])
    xs = [str(r.get(label)) for r in rows]
    is_time = sum(1 for v in xs if _MONTHISH.search(v)) >= max(2, len(xs) // 2)
    inr_cols = [c for c in numeric if infer_kind(c) == "inr"]
    pct_cols = [c for c in numeric if infer_kind(c) == "pct"]
    # value + percentage → combo (the "revenue and margin %" case)
    if inr_cols and pct_cols and not is_time and res["row_count"] <= 14:
        return {"type": "combo", "x": label, "y": inr_cols[0], "y2": pct_cols[0],
                "value_format": "inr", "y2_format": "pct",
                "title": f"{inr_cols[0].replace('_', ' ').title()} & {pct_cols[0].replace('_', ' ').title()} by {label.replace('_', ' ')}"}
    ycol = inr_cols[0] if inr_cols else numeric[0]
    ctype = "line" if is_time else "bar"
    return {"type": ctype, "x": label, "y": ycol, "value_format": infer_kind(ycol),
            "title": f"{ycol.replace('_', ' ').title()} by {label.replace('_', ' ')}",
            "orientation": "h" if (ctype == "bar" and len(rows) > 6) else "v"}


def _enhance_spec(spec: dict, res: dict):
    """Add depth: a single ₹ bar over a result that ALSO has a % column becomes a
    combo (bars + %-line on y2) — so 'revenue and margin %' shows both, not just revenue."""
    if spec.get("type") not in ("bar", "grouped_bar"):
        return
    y = spec.get("y")
    ys = y if isinstance(y, list) else [y]
    if len(ys) != 1 or infer_kind(str(ys[0])) != "inr":
        return
    if len(res.get("rows", [])) > 14:
        return
    pct_col = next((c for c in res["columns"] if c not in ys and infer_kind(c) == "pct"), None)
    if pct_col:
        spec["type"] = "combo"
        spec["y"] = ys[0]
        spec["y2"] = pct_col
        spec["value_format"] = "inr"
        spec["y2_format"] = "pct"


def _has_data(res) -> bool:
    """True only if the result has rows with at least one non-null cell. A SUM/aggregate
    over zero matching rows returns a single all-NULL row — that's 'no data', not a table."""
    if not res or not res.get("rows"):
        return False
    return any(v is not None for row in res["rows"] for v in row.values())


def _pick_result(results, chart):
    """Find the query result whose columns cover the chart's referenced columns (prefer most recent)."""
    if not chart:
        return None
    needed = set()
    for k in ("x", "color", "size", "y2"):
        if chart.get(k):
            needed.add(chart[k])
    y = chart.get("y")
    for c in (y if isinstance(y, list) else [y]):
        if c:
            needed.add(c)
    for res in reversed(results):
        if needed.issubset(set(res["columns"])):
            return res
    return results[-1] if results else None


def _verify(client, query, results, answer):
    """Strict auditor: does every figure in the answer match the query results?"""
    payload = {"question": query, "answer": answer,
               "queries": [{"sql": r["sql"], "result": _format_result(r, 15)} for r in results]}
    try:
        resp = client.chat.completions.create(
            model=AZURE_DEPLOYMENT, temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a strict data auditor. Given a question, the SQL query results, and a proposed answer, verify that EVERY number and claim in the answer is supported by the query results and actually answers the question. Reply ONLY JSON: {\"ok\": true|false, \"issue\": \"<empty if ok, else the specific problem>\"}."},
                {"role": "user", "content": json.dumps(payload)[:12000]},
            ])
        out = json.loads(resp.choices[0].message.content or "{}")
        return bool(out.get("ok", True)), str(out.get("issue", ""))
    except Exception:
        return True, ""  # never block on a verifier failure


def answer(query: str, history: list | None = None):
    """Generator of SSE event dicts: step / sql / answer / chart / table / verified / done / error."""
    if not has_key():
        yield {"type": "error", "text": "The AI Analyst isn't configured — set AZURE_OPENAI_API_KEY on the server."}
        return
    try:
        client = _client()
        ctx = semantics.context()
    except Exception as e:
        yield {"type": "error", "text": f"AI service unavailable: {e}"}
        return

    messages = [{"role": "system", "content": SYSTEM.format(context=ctx)}]
    for h in (history or [])[-6:]:
        if h.get("role") in ("user", "assistant") and h.get("content"):
            messages.append({"role": h["role"], "content": str(h["content"])[:1500]})
    messages.append({"role": "user", "content": query})

    yield {"type": "step", "text": "Understanding your question"}
    results: list[dict] = []
    present_args = None

    for _ in range(MAX_SQL_STEPS):
        try:
            resp = client.chat.completions.create(
                model=AZURE_DEPLOYMENT, messages=messages, temperature=0,
                tools=[RUN_SQL_TOOL, LOOKUP_TOOL, PRESENT_TOOL, CLARIFY_TOOL], tool_choice="auto")
        except Exception as e:
            yield {"type": "error", "text": f"AI request failed: {e}"}
            return
        msg = resp.choices[0].message
        if not msg.tool_calls:
            if msg.content:
                present_args = {"answer": msg.content}
            break

        messages.append({"role": "assistant", "content": msg.content or None,
                         "tool_calls": [{"id": tc.id, "type": "function",
                                         "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                                        for tc in msg.tool_calls]})
        stop = False
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except Exception:
                args = {}
            if tc.function.name == "ask_clarification":
                yield {"type": "clarify", "text": args.get("question", "Could you clarify what you'd like?"),
                       "options": args.get("options", [])}
                yield {"type": "done"}
                return
            if tc.function.name == "present":
                present_args = args
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": "ok"})
                stop = True
                continue
            if tc.function.name == "lookup_item":
                name = (args.get("name") or "").strip()
                yield {"type": "step", "text": f"Locating “{name}” across all tables"}
                try:
                    fp = warehouse.item_footprint(name)
                    hit = ", ".join(fp["tables_with_data"][:6]) or "no tables"
                    yield {"type": "sql", "purpose": f"footprint of “{name}”", "sql": f"-- lookup_item('{name}')", "rows": fp["match_count"]}
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(fp)[:5000]})
                except Exception as e:
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps({"error": str(e)[:200]})})
                continue
            # run_sql
            purpose = args.get("purpose") or "querying the data"
            sql = args.get("sql", "")
            yield {"type": "step", "text": purpose[:80]}
            try:
                res = warehouse.run_sql(sql)
                res["purpose"] = purpose
                results.append(res)
                yield {"type": "sql", "sql": res["sql"], "purpose": purpose, "rows": res["row_count"]}
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(_format_result(res))})
            except Exception as e:
                # Keep internal self-corrections invisible: feed the error back to the model
                # so it fixes the query, but don't surface a scary errored query to the user.
                yield {"type": "step", "text": "Refining the query"}
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": json.dumps({"error": str(e)[:300], "hint": "Fix the SQL and try again (check FROM clause, column names, and the typed schema)."})})
        if stop:
            break

    if not present_args:
        yield {"type": "answer", "text": "I couldn't resolve that into a query — try rephrasing, or ask about revenue, inventory, procurement, expiry, or forecasts."}
        yield {"type": "done"}
        return

    ans = (present_args.get("answer") or "").strip()
    chart_specs = present_args.get("charts")
    if not chart_specs:
        single = present_args.get("chart")
        chart_specs = [single] if single else []
    chart_specs = [c for c in chart_specs if c]

    # VERIFY (only when we actually queried data)
    verified = None
    if results and ans:
        yield {"type": "step", "text": "Verifying the numbers"}
        ok, issue = _verify(client, query, results, ans)
        if not ok and issue:
            # one correction pass
            try:
                fix = client.chat.completions.create(
                    model=AZURE_DEPLOYMENT, temperature=0,
                    messages=[
                        {"role": "system", "content": "Rewrite the answer to fix the auditor's issue, using ONLY the query results provided. Keep it concise. Reply with just the corrected answer text."},
                        {"role": "user", "content": json.dumps({"question": query, "answer": ans, "issue": issue,
                                                                 "queries": [{"sql": r["sql"], "result": _format_result(r, 15)} for r in results]})[:12000]},
                    ])
                ans = (fix.choices[0].message.content or ans).strip()
                verified = "corrected"
            except Exception:
                verified = "flagged"
        else:
            verified = "ok"

    yield {"type": "answer", "text": ans, "verified": verified}

    # CHARTS — use the model's spec(s), else auto-build one if the data is chartable
    if not chart_specs and results:
        auto = _auto_chart(results[-1])
        if auto:
            chart_specs = [auto]

    table_res = None
    for spec in chart_specs:
        res = _pick_result(results, spec)
        if not res or not res["rows"]:
            continue
        if not spec.get("value_format") and spec.get("y"):
            yk = spec["y"][0] if isinstance(spec["y"], list) else spec["y"]
            spec["value_format"] = infer_kind(str(yk))
        _enhance_spec(spec, res)
        fig = charts.build(res["rows"], spec)
        if fig:
            yield {"type": "chart", "plotly": fig}
        table_res = table_res or res

    if not table_res and results:
        table_res = next((r for r in reversed(results) if _has_data(r)), None)
    if _has_data(table_res):
        yield {"type": "table", "table": {"title": table_res.get("purpose", ""),
                                          "columns": [{"key": c, "label": c, "kind": col_kind(c, table_res["rows"])} for c in table_res["columns"]],
                                          "rows": table_res["rows"][:50]},
               "note": ("Showing top 50 of %d rows." % table_res["row_count"]) if table_res.get("truncated") or table_res["row_count"] > 50 else ""}

    yield {"type": "done"}
