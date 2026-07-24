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
import time

from app.ai import warehouse, semantics, charts, routing

AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://ed-gpt.openai.azure.com")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
# Follow-up-question suggestions are a lightweight phrasing task, not analytical SQL
# reasoning — gpt-4o is overkill for it. A cheaper, dedicated model call handles this so
# every answer reliably gets 2-3 GENUINELY relevant suggestions (grounded in the actual
# entities/numbers just discussed) instead of depending on the main model remembering to.
MINI_DEPLOYMENT = os.getenv("AZURE_OPENAI_MINI_DEPLOYMENT", "gpt-4o-mini")
MAX_SQL_STEPS = 9
MAX_AUDIT_RETRIES = 3   # times the auditor can bounce a wrong/mis-scoped answer back for re-query
# How many prior messages of conversation the model (and auditor) can see. Was 6 (3 exchanges)
# — too short: a callback to anything ~4+ turns back fell outside the window and the model
# confidently hallucinated a substitute. gpt-4o has a 128K context and answers are short, so a
# generous window is cheap and covers realistic working sessions. Kept in sync with the frontend.
HISTORY_MESSAGES = 24


def has_key() -> bool:
    return bool(os.getenv("AZURE_OPENAI_API_KEY"))


def _client():
    from openai import AzureOpenAI
    key = os.getenv("AZURE_OPENAI_API_KEY")
    if not key:
        raise RuntimeError("AZURE_OPENAI_API_KEY is not set")
    return AzureOpenAI(azure_endpoint=AZURE_ENDPOINT, api_key=key, api_version=AZURE_API_VERSION)


# A single transient Azure hiccup (429 rate-limit, or a 500/502/503/504) used to
# abandon a whole turn — often AFTER real, correct query results had already been
# gathered — surfacing only a raw error (or worse, the model fabricating a
# "permissions issue" excuse). Retry transient failures a few times with short
# exponential backoff before giving up. Only retriable statuses are retried; a
# genuine 400 (bad request) fails fast.
_RETRIABLE_STATUS = {429, 500, 502, 503, 504}
_MAX_LLM_RETRIES = 4


def _is_retriable(e: Exception) -> bool:
    status = getattr(e, "status_code", None) or getattr(getattr(e, "response", None), "status_code", None)
    if status in _RETRIABLE_STATUS:
        return True
    name = type(e).__name__.lower()
    return any(k in name for k in ("ratelimit", "timeout", "connection", "apiconnection", "internalserver"))


def _chat(client, **kwargs):
    """client.chat.completions.create with retry+backoff on transient Azure errors."""
    delay = 2.0
    last = None
    for attempt in range(_MAX_LLM_RETRIES):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as e:  # noqa: BLE001 — we re-raise below if not retriable / out of attempts
            last = e
            if not _is_retriable(e) or attempt == _MAX_LLM_RETRIES - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2.2, 20.0)
    raise last  # unreachable, but keeps type-checkers happy


# ── number pre-formatting so the model can't mis-convert units ───────────────
_INR = re.compile(r"(revenue|margin|value|cost|spend|price|amount|opportunity|overpay|mrp|purchase|sales|cr\b)", re.I)
_PCT = re.compile(r"(pct|percent|share|rate|margin_pct|%)", re.I)
_DAYS = re.compile(r"(doh|days|aging|lead|cover|tat)", re.I)
_YEAR = re.compile(r"^(year|yr|fiscal_year|fy)$", re.I)   # a calendar year is a label, not a quantity


def infer_kind(col: str) -> str:
    c = col.lower()
    if _YEAR.search(c):
        return "year"
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
    if kind == "year":
        try:
            return str(int(float(v)))   # 2025 — no thousands separator, no ₹
        except (TypeError, ValueError):
            return str(v)
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
        if 0 < a < 1:
            return f"₹{n:.2f}"   # a nonzero paisa-level value must never round to "₹0" — indistinguishable from a true zero
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
• UNDERSTAND the real intent first. If the request is ANSWERABLE from the data but genuinely ambiguous or under-specified (unclear time range, which metric/entity), call ask_clarification with ONE short question (and 2–4 quick options) INSTEAD of guessing. Don't over-ask — if a sensible default is obvious, just proceed and state the assumption.
• OUT OF SCOPE: this assistant only covers HCG supply-chain data (sales, procurement, inventory, expiry, consumption, forecasts). If asked for something the data simply doesn't contain — people/roles (e.g. "who is the CEO"), org structure, patient/clinical records, real-world/external facts, weather — do NOT ask a clarifying question and do NOT query. Briefly say it's outside the supply-chain data you have, and point them to what you CAN answer. Decline cleanly in one sentence.
• SPECIFIC ITEM? For ANY question about a particular product/SKU (named or by code), call lookup_item FIRST. It returns the item's identity (generic, group, manufacturer, formulary status) and a COMPLETE footprint — the row count in EVERY table it touches (sales, PO, GRN, consumption, inventory, forecasts). This guarantees you never miss a source: an item with 0 sales/purchases but rows in fact_inventory is DEAD / NON-MOVING stock (report qty, aging, expiry, formulary) — never call that "no data". Then run_sql only the tables the footprint shows have rows.
• BEFORE you write a query, check the DIMENSIONAL MODEL above for the table you're about to use: you may only GROUP BY / filter on a dimension in its "slice by" list, and may only show a time trend if its "time axis" isn't NONE. If the cut the user wants (a dimension × a time axis, or two dimensions) doesn't exist together in any one table, that exact breakdown is NOT available — give the closest correct cut and say so. Never take a broader table's numbers and label them as a narrower entity/period.
• Call run_sql to fetch data — MULTIPLE times as needed. Decompose complex questions, explore first, then run the precise query; join across tables freely (CTEs, window functions, subqueries all work in DuckDB). Go into real DEPTH: don't just pull the top line — look at the composition, the outliers, the trend, the "so what".
• Every number in your final answer MUST come from a query you actually ran.
• SCOPE YOUR CLAIMS TO WHAT YOU ACTUALLY QUERIED — never generalize a narrow or empty result into a broader absolute statement. If a filter applied to one specific list/subset returns nothing, say exactly that ("none of these particular items are X"), never the broader, unverified claim ("there are no X in the dataset") — the broader claim requires its OWN query against the full data, not an inference from a narrower one. Getting this wrong means confidently contradicting yourself the moment the user asks the broader question directly next.
• When done, call present() with a warm, natural, analytical answer (talk like a helpful colleague, not a report generator) plus chart(s). Keep it TIGHT — 2–4 sentences: the headline number(s) + the one insight that matters. Do NOT enumerate long lists item-by-item in the prose (the chart AND the data table below already show every row) — mention the top 1–2 and summarise the rest. No filler sign-offs like "let me know if you'd like…". NEVER paste a markdown/pipe table into the answer text.
   – ALWAYS chart a ranking, breakdown, trend, comparison or share.
   – If the user asks for "two charts", "different charts", "a pie and a bar", etc., or if two views genuinely illuminate the data, put MULTIPLE specs in `charts` (e.g. a ranking bar AND a share donut).
   – bar=rankings, line=time trend, donut=shares, combo (percentage on y2)=two different scales, heatmap=matrix, scatter/bubble=correlation, treemap/sunburst=hierarchy, waterfall=build-up.
   – Only omit charts for a pure single-number answer.
• Money is already formatted (₹Cr/₹L) in results — quote those strings verbatim, never recompute units.
• Be genuinely analytical: lead with the answer, then add the insight that matters (a concentration, a trend, a risk, a surprise). Respect caveats in the schema notes (e.g. manufacturer-of-purchases coverage). Put next-question ideas in follow_ups (clickable chips), not as a trailing question in the prose.
• When your answer is scoped to a specific thing (a category, item, vendor, plant, or a ranked subset), ALWAYS fill present()'s `scope` field with the exact filters you used — this lets a terse follow-up ('which is cheapest', 'and their lead times') correctly inherit your scope instead of resetting to the whole company.
• ⛔ MEMORY HONESTY: you only see the recent part of the conversation. If the user refers back to something "earlier" / "we found before" / "that item/number/figure" and it is NOT actually present in the conversation you can see (nor in the ACTIVE SCOPE), do NOT invent a plausible substitute and label it as the earlier finding — that is a confident hallucination. Instead say briefly that you don't have that earlier turn in view and ask them to restate it (offer to just re-run the analysis fresh). Running a NEW global query and presenting its result as "the one we found earlier" is a serious error.
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
        "x": {"type": "string", "description": "Result column for the category / x-axis / labels. NEVER a raw code/id column (material, vendor_code, plant) — use the readable sibling (material_desc, vendor_name, plant_name) whenever the query selected it."},
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
            "answer": {"type": "string", "description": "Final answer in warm, natural, analytical markdown. Quote the pre-formatted figures exactly. Do NOT end with a trailing rhetorical question ('would you like to explore...?') — put real next-question suggestions in follow_ups instead, which renders as clickable chips."},
            "charts": {"type": "array", "description": "One or MORE charts. If the user asks for multiple/different charts, or two views genuinely help (e.g. a ranking bar AND a share donut), include several. Empty for a single-number answer.", "items": _CHART_SPEC},
            "chart": dict(_CHART_SPEC, description="Deprecated single-chart form — prefer 'charts'."),
            "follow_ups": {"type": "array", "items": {"type": "string"}, "description": "OPTIONAL 2-3 short, concrete drill-down questions a user would naturally ask next about THIS answer (e.g. 'Break this down by hospital', 'Show the monthly trend'). Rendered as clickable chips — clicking one sends it as the next question. Only include ones that are genuinely answerable from this data; omit entirely for a simple/closed answer that doesn't invite a drill-down."},
            "scope": {"type": "string", "description": "OPTIONAL but IMPORTANT for follow-ups: a SHORT machine-usable description of the exact filters THIS answer applied, so a terse next question ('which is cheapest', 'and their lead times', 'just the injection ones', 'the worst one') can inherit them. Include the real filter values you used — e.g. 'category = M065-INJECTIONS (injection drugs)', 'vendor = Vardhman Health Specialities', 'material 101313 KEYTRUDA'. For a RANKING/top-N answer, ALWAYS name the #1 entity so a follow-up like 'the worst/top/first one' resolves to it — e.g. 'vendors ranked by price inconsistency; #1 = D.Vijay Pharma Pvt Ltd', 'top-10 vendors by spend; #1 = Vardhman'. Omit only for a broad, unscoped, whole-company answer."}}}},
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


_AUDITOR_SYS = """You are a STRICT data auditor for a hospital supply-chain analyst. You get the user's question, the exact SQL queries that ran (with their results and stated purpose), and a proposed answer. Catch answers that are WRONG or MISLEADING before they reach the user. Judge whether each number is correct FOR WHAT IT IS LABELLED AS — not whether it is the user's first-choice metric.

FAIL the answer (ok=false) if ANY of these hold:
1. NUMBER MISMATCH — a figure in the answer is not supported by the query results.
2. WRONG SCOPE — the answer attributes a figure to a specific entity (item/brand/category/hospital/vendor/department/plant), but the SQL that produced it did NOT GROUP BY or filter to that entity (e.g. it is actually a company-wide or different-entity total presented as that entity's). This is the most important check. ALSO check `prior_conversation`: if an earlier turn established a scope (a category/type/entity filter — e.g. "pharma injection items") and the CURRENT question is a refinement of that same thing ("remove X", "exclude Y", "excluding zero ones"), the current SQL must still carry that same scope. If the SQL dropped it (e.g. now scans everything instead of just that category) and the answer presents unrelated results (hardware/equipment showing up in what was a drug-only list) as if it were still the same scoped analysis, that is a WRONG SCOPE failure — regardless of whether the current question's SQL alone looks internally consistent.
3. MISLABELLED METRIC/SOURCE — a number is presented as something it is NOT, in a way that changes its meaning: e.g. purchase or consumption figures called "sales" (or vice-versa); an aggregate over the wrong grain (a broader table's total shown as a narrower breakdown); a grand total taken from a table that only partially covers it; a MEAN value reported when the rule requires a MEDIAN. (NOT a defect: calling a correctly-computed MEDIAN "average" or "typical" in prose — the value is the right median, the word choice is colloquial. Only fail if the actual NUMBER is a mean when it should be a median.)
4. UNSUPPORTED CLAIM — a trend/insight/comparison the results don't actually show.
5. DODGES — answers a different question than asked with no explanation.
6. IMPLAUSIBLE PRICE OUTLIER — for a price-deviation / overpay / price-swing / price-increase analysis: if a LOW-VALUE or non-specialty item (a common consumable, stationery, food/grocery, a generic screw/hardware, a cheap disposable) tops the list with a multi-CRORE figure, that is almost certainly a data-entry error in the underlying rows (a price↔quantity transposition, or a ₹0.01 placeholder price), NOT a real finding — FAIL it. fix: rebuild on outlier-clean rows only (net_price>=10; drop rows outside 8x of that item's own median price; require >=5 clean rows before ranking). Do NOT fail a genuinely high-value SPECIALTY item (e.g. an oncology injectable that really costs ₹1–2 lakh/vial) showing a ₹1–3 Cr figure — that can be real; only flag when the item's nature is clearly inconsistent with the magnitude.

PASS (ok=true) when the numbers are SUPPORTED, correctly SCOPED, and correctly LABELLED for what they claim to be. Only fail for a DEFECT THAT MAKES A NUMBER WRONG OR MISLEADING (categories 1–5). Do NOT fail an otherwise-correct answer for any of these — they are all PASS:
  • wording/tone/brevity/rounding; describing a correctly-computed median or typical value as "on average" / "typically";
  • a missing caveat or context (e.g. not noting that top items are non-clinical) — desirable, but its absence is not an error;
  • being incomplete but correct (top N instead of all; one lens of several);
  • an HONEST "this exact breakdown/metric/granularity isn't available" that offers the closest correct, clearly-labelled alternative (e.g. "monthly sales isn't available at item level; here are the item's monthly purchases").
When unsure whether it's a real defect or just imperfect phrasing, PASS. Reserve ok=false for numbers that are actually wrong, mis-scoped, or mislabelled in a way that changes their meaning.

Reply ONLY JSON: {"ok": true|false, "issue": "<empty if ok; else the SPECIFIC defect that makes a number wrong/misleading>", "fix": "<empty if ok; else a concrete instruction: which table/filter/metric/grain to use instead>"}."""


def _verify(client, query, results, answer, history=None):
    """Strict auditor: numbers supported AND correctly scoped/sourced. Returns (ok, issue, fix).
    `history` (prior turns) lets it catch a follow-up that silently drops a scope/filter
    established earlier — invisible from the current turn's query text alone."""
    payload = {"question": query, "answer": answer,
               "prior_conversation": [{"role": h.get("role"), "content": str(h.get("content"))[:400]} for h in (history or [])[-HISTORY_MESSAGES:]],
               "queries": [{"sql": r["sql"], "purpose": r.get("purpose"), "result": _format_result(r, 15)} for r in results]}
    try:
        resp = _chat(client,
            model=AZURE_DEPLOYMENT, temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _AUDITOR_SYS},
                {"role": "user", "content": json.dumps(payload)[:14000]},
            ])
        out = json.loads(resp.choices[0].message.content or "{}")
        return bool(out.get("ok", True)), str(out.get("issue", "")), str(out.get("fix", ""))
    except Exception:
        return True, "", ""  # never block on a verifier failure (already retried inside _chat)


_FOLLOWUP_SYS = """You suggest 2-3 short follow-up questions a hospital supply-chain operator would naturally ask right after getting the answer shown to you. This is a pure phrasing task — you are NOT analyzing data, just proposing what to ask next.

You are given a GRAIN MAP: for each table, exactly which dimensions it can be sliced by and whether it has a time axis. This is ground truth — do NOT suggest a breakdown or trend that isn't actually supported by some table's real dimensions/time axis (e.g. never suggest "by department" for a revenue/sales question if no sales-side table lists a department/cost-centre dimension; never suggest "the trend over N months" for a table whose time axis is NONE). When unsure whether something is really sliceable, prefer a SAFER suggestion instead: a top-N variant, a comparison to a specific already-named entity, an outlier/consistency check, or a breakdown by a dimension you can actually see listed for a relevant table.

Rules:
1. Ground every suggestion in a SPECIFIC entity, number, or comparison that is actually named in the answer (a real item, vendor, manufacturer, category, plant, month, or metric mentioned there) — never a vague "would you like to explore further?" or "let me know if you want more details".
2. Only suggest something genuinely answerable given the GRAIN MAP below. Never suggest anything outside the hospital supply-chain domain (no patient records, staff, org chart, external/competitor data).
3. Keep each suggestion short (under ~9 words), phrased as a natural next question or a concrete drill-down instruction — e.g. "Break this down by hospital", "Which vendor supplies that item?", "Show the monthly trend", "Compare it to last month".
4. Do not repeat the question that was just asked, and do not suggest something the answer already fully covered.
5. If the answer is a simple closed fact with little to drill into, still offer a grounded next angle consistent with the GRAIN MAP — always return AT LEAST 1 suggestion, never an empty list, unless the answer is a decline/out-of-scope message with nothing to build on (then return an empty list).
6. Specific traps to never fall into (these are the mistakes a generic BI assistant makes by default — do NOT make them here): sales/revenue data has NO vendor dimension (vendor only exists on the procurement side — never suggest "which vendor contributes to revenue" or similar) and NO cost-centre/department dimension (that's consumption-side only) and NO plant/region column at all (sales is hospital-based only). This whole dataset covers only a 6-month window — NEVER suggest a year-over-year / "vs last year" / 12-month-trend comparison; a "trend" suggestion must stay within the 6-month window (e.g. "month over month" is fine, "vs last year" is not).

Reply ONLY JSON: {"follow_ups": ["...", "...", "..."]}"""


# Deterministic backstop for the one cross-domain trap the mini model still fell into
# non-deterministically even with an explicit instruction not to (revenue/sales has no
# vendor or cost-centre dimension in this schema — this is a permanent, hard schema fact,
# not a judgment call, so it's cheaper and more reliable to filter it in code than to hope
# a temperature>0 model always remembers the rule).
_REVENUE_WORDS = re.compile(r"\b(revenue|sales|margin)\b", re.I)
_INVALID_CROSS_DOMAIN = re.compile(r"\b(vendor|cost.?cent(er|re)|department|plant)\b", re.I)


def _drop_invalid_cross_domain(suggestions: list[str], query: str, answer: str) -> list[str]:
    if not _REVENUE_WORDS.search(query or "") and not _REVENUE_WORDS.search(answer or ""):
        return suggestions
    return [s for s in suggestions if not _INVALID_CROSS_DOMAIN.search(s)]


def _gen_follow_ups(client, query: str, answer: str, results: list[dict] | None) -> list[str]:
    """Cheap, dedicated follow-up-question generator on gpt-4o-mini (not gpt-4o — this is
    a lightweight wording task, not analytical SQL reasoning). Runs on every real answer so
    suggestions are reliably present and grounded in what was just discussed, rather than
    depending on the main model remembering to propose good ones. The grain map (which
    dimensions/time axis each table really supports) is included so it can't invent an
    unsupported breakdown (e.g. "by department" for revenue, which has no such dimension).
    Fail-open: any error or empty response returns [] and the caller falls back to the main
    model's own follow_ups."""
    if not answer:
        return []
    try:
        queries = [{"purpose": r.get("purpose"), "sql": r.get("sql")} for r in (results or []) if r.get("purpose")][:5]
        payload = {"question": query, "answer": answer[:1200], "queries_run": queries,
                   "grain_map": warehouse.grain_text()}
        resp = _chat(client,
            model=MINI_DEPLOYMENT, temperature=0.1, max_tokens=200,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _FOLLOWUP_SYS},
                {"role": "user", "content": json.dumps(payload)[:10000]},
            ])
        out = json.loads(resp.choices[0].message.content or "{}")
        ups = [str(f).strip() for f in (out.get("follow_ups") or []) if str(f).strip()]
        ups = _drop_invalid_cross_domain(ups, query, answer)
        return ups[:3]
    except Exception:
        return []


_SCOPE_MARK = re.compile(r"\[active scope:\s*(.+?)\s*\]", re.I | re.S)
_WHERE_RE = re.compile(r"\bwhere\b(.+?)(?:\bgroup\s+by\b|\border\s+by\b|\blimit\b|$)", re.I | re.S)
# Refinement cues: a terse follow-up that leans on the PRIOR turn's scope. We only inject
# the prior scope for these — a full, self-contained new question (topic change) must NOT
# inherit stale filters (that bled procurement context into inventory questions).
_REFINE_START = re.compile(r"^\s*(and|also|now|just|only|then|but|what about|how about|excluding|exclude|without|remove|drop|instead|same|of (these|those)|for (these|those|that|it|them))\b", re.I)
_DEICTIC = re.compile(r"\b(that|those|these|this|them|they|it|its|their|there|the (top|cheapest|worst|best|same|first|last|one)|the ones)\b", re.I)


def _is_refinement(query: str) -> bool:
    """True when the query reads as a follow-up leaning on prior scope (terse, or starting
    with a refinement conjunction, or using a back-reference like 'those'/'them'/'it')."""
    q = (query or "").strip()
    if not q:
        return False
    words = q.split()
    if len(words) <= 6:
        return True
    if _REFINE_START.match(q):
        return True
    # a short-ish question that back-references the prior answer
    if len(words) <= 14 and _DEICTIC.search(q):
        return True
    return False


def _derive_scope(results: list[dict]) -> str:
    """Deterministic fallback when the model doesn't self-declare a scope: summarise the
    LAST successful query's FROM table + WHERE filters, so a terse follow-up still inherits
    the concrete context. Not perfect SQL parsing — just enough signal for the next turn."""
    for res in reversed(results or []):
        sql = (res.get("sql") or "").strip()
        if not sql:
            continue
        m = _WHERE_RE.search(sql)
        if not m:
            continue
        where = " ".join(m.group(1).split())[:220]
        fm = re.search(r"\bfrom\s+([a-z_][a-z0-9_]*)", sql, re.I)
        tbl = fm.group(1) if fm else ""
        return (f"previous query used {tbl} filtered on: {where}" if tbl else f"previous filters: {where}")
    return ""


def _latest_scope(history: list | None) -> str:
    """Pull the most recent '[active scope: …]' marker an assistant turn carried
    (the frontend appends it to the assistant content it echoes back). '' if none."""
    for h in reversed(history or []):
        if h.get("role") == "assistant" and h.get("content"):
            m = _SCOPE_MARK.search(str(h["content"]))
            if m:
                return m.group(1).strip()[:300]
    return ""


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
    for h in (history or [])[-HISTORY_MESSAGES:]:
        if h.get("role") in ("user", "assistant") and h.get("content"):
            messages.append({"role": h["role"], "content": str(h["content"])[:1500]})
    # SCOPE THREADING: the last answer may carry an "[active scope: …]" marker (echoed
    # back by the frontend). A terse follow-up must INHERIT that scope, not silently reset
    # to the whole company — this is the multi-turn failure mode live testing kept hitting.
    # Only thread prior scope when THIS question actually reads as a refinement — a full,
    # self-contained new-topic question must not inherit stale filters.
    prior_scope = _latest_scope(history) if _is_refinement(query) else ""
    if prior_scope:
        messages.append({"role": "system", "content":
            f"ACTIVE SCOPE from the previous answer: {prior_scope}. "
            "This new question looks like a refinement/terse follow-up, so re-apply that same "
            "scope/filters and only add the new twist — do NOT rebuild from an unfiltered, "
            "whole-company base. If the user refers to 'the worst/top/first/that one', resolve it "
            "to the specific named entity in that scope and FILTER your query to that entity "
            "(e.g. WHERE vendor_name = '<that vendor>'). (If you judge the user has in fact "
            "changed topic, ignore this.)"})

    # Embedding-based routing hint (fail-open: '' on any error → static examples carry it).
    # When a prior scope is active (a refinement), suppress hard intent-locks so they can't
    # override the vendor/item/category the user is drilling into.
    hint = routing.hints_for(client, query, allow_locks=not prior_scope)
    messages.append({"role": "user", "content": (hint + "\n\n" + query) if hint else query})

    yield {"type": "step", "text": "Understanding your question"}
    results: list[dict] = []
    present_args = None
    present_from_content = False
    verified = None
    present_attempts = 0

    any_sql_failed = False
    for _ in range(MAX_SQL_STEPS):
        try:
            resp = _chat(client,
                model=AZURE_DEPLOYMENT, messages=messages, temperature=0,
                tools=[RUN_SQL_TOOL, LOOKUP_TOOL, PRESENT_TOOL, CLARIFY_TOOL], tool_choice="auto")
        except Exception as e:
            # Only reached after _chat exhausted its retries. If we already gathered real
            # results, hand those back honestly rather than throwing the whole turn away.
            if results:
                break
            yield {"type": "error", "text": "The AI service is briefly overloaded (rate-limited). Please try that again in a moment."}
            return
        msg = resp.choices[0].message
        if not msg.tool_calls:
            if msg.content:
                present_args = {"answer": msg.content}
                present_from_content = True
            break

        messages.append({"role": "assistant", "content": msg.content or None,
                         "tool_calls": [{"id": tc.id, "type": "function",
                                         "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                                        for tc in msg.tool_calls]})
        pending_present = None   # collected during the batch, verified AFTER all calls are acked
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
                pending_present = args
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": "ok"})
                continue
            if tc.function.name == "lookup_item":
                name = (args.get("name") or "").strip()
                yield {"type": "step", "text": f"Locating “{name}” across all tables"}
                try:
                    fp = warehouse.item_footprint(name)
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
                any_sql_failed = True
                yield {"type": "step", "text": "Refining the query"}
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": json.dumps({"error": str(e)[:300], "hint": "Fix the SQL and try again (check FROM clause, column names, and the typed schema)."})})

        # All tool calls in this batch are now acked. If the model asked to present,
        # AUDIT it before accepting — and on a real problem, send it back to re-query.
        if pending_present is not None:
            cand_ans = (pending_present.get("answer") or "").strip()
            ok, issue, fix = True, "", ""
            if results and cand_ans:
                yield {"type": "step", "text": "Cross-checking the answer against the data"}
                ok, issue, fix = _verify(client, query, results, cand_ans, history)
            if ok or present_attempts >= MAX_AUDIT_RETRIES:
                present_args = pending_present
                verified = ("corrected" if present_attempts > 0 else "ok") if (results and cand_ans) else None
                if not ok:
                    verified = "flagged"   # auditor still unhappy after retries → no green badge
                break
            # audit failed and we have a retry left → feed the problem back, let it fix
            present_attempts += 1
            yield {"type": "step", "text": "Correcting the analysis"}
            messages.append({"role": "user", "content":
                "AUDIT FAILED — your last answer was not accepted, do NOT repeat it. Problem: "
                + (issue or "the answer was not correctly supported/scoped.")
                + (" Fix: " + fix if fix else "")
                + " If this is a WRONG SCOPE issue on a follow-up ('remove X', 'exclude Y', etc.): find your MOST RECENT run_sql call earlier in this conversation that established the scope (its WHERE/JOIN conditions for category, item type, entity…) and COPY those same conditions into the new query verbatim, adding only the new exclusion — do not rebuild from an unfiltered base."
                + " Re-run the correct query (right table, filtered to the exact entity/scope the user asked about, right metric) and call present again with corrected numbers. If the data genuinely cannot answer it, say so honestly instead."})

    if not present_args:
        yield {"type": "answer", "text": "I couldn't resolve that into a query — try rephrasing, or ask about revenue, inventory, procurement, expiry, or forecasts."}
        yield {"type": "done"}
        return

    ans = (present_args.get("answer") or "").strip()

    # HONEST FAILURE: if the model gave up as free-text (no present() call) having gathered
    # ZERO successful query results after a SQL error, its prose is an ungrounded excuse —
    # this is where it used to fabricate a plausible "permissions / file access" reason. Do
    # not ship that; say plainly that the query didn't run. (A legitimate "no data" answer
    # always comes through present() WITH results, so this never suppresses a real answer.)
    if present_from_content and not results and (any_sql_failed or not ans):
        yield {"type": "answer",
               "text": "I ran into a repeated error building the query for that and couldn't complete it — please try rephrasing, or ask it a slightly different way.",
               "verified": None, "options": []}
        yield {"type": "done"}
        return

    chart_specs = present_args.get("charts")
    if not chart_specs:
        single = present_args.get("chart")
        chart_specs = [single] if single else []
    chart_specs = [c for c in chart_specs if c]

    # If the answer came back as plain content (model didn't call present), it was never
    # audited inline — audit it now so EVERY data-backed answer gets the same gate.
    if verified is None and results and ans:
        yield {"type": "step", "text": "Cross-checking the answer against the data"}
        ok, _issue, _fix = _verify(client, query, results, ans, history)
        verified = "ok" if ok else "flagged"

    # FLAGGED = the auditor could not confirm the numbers after its retries. Don't let that
    # ship looking identical to a clean answer (the frontend badge alone was proven unreliable):
    # bake an explicit caveat into the answer TEXT so the user always sees it.
    if verified == "flagged" and ans and not ans.lstrip().startswith("⚠️"):
        ans = ("⚠️ _I couldn't fully verify these figures against the data — treat them as "
               "indicative and double-check before acting on them._\n\n") + ans

    # Ship the answer with gpt-4o's own follow_ups (if any) immediately — do NOT block the
    # answer on the dedicated follow-up call below. The better, grounded follow-ups from the
    # mini-model are generated AFTER everything else has already reached the user (see the
    # "followups" patch event near the end) so this feature adds ZERO perceived latency to
    # the answer itself.
    follow_ups = [str(f).strip() for f in (present_args.get("follow_ups") or []) if str(f).strip()][:3]
    # Scope chain: prefer the model's own declaration, else derive from the last query. If BOTH
    # are empty (e.g. a turn that answered from prior data with no new SQL), carry the PRIOR
    # scope forward so a chain of refinements doesn't get severed by one no-query turn.
    scope = str(present_args.get("scope") or "").strip()[:300] or _derive_scope(results) or _latest_scope(history)
    yield {"type": "answer", "text": ans, "verified": verified, "options": follow_ups, "scope": scope}

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

    # Better, grounded follow-ups (dedicated cheap-model call) computed only NOW — after the
    # answer/chart/table have already reached the user — so the extra ~1-2s never delays the
    # answer itself. Patches onto the same message once ready; if it's empty/errors, the
    # gpt-4o-sourced chips already shown (possibly none) simply stay as they are.
    better_follow_ups = _gen_follow_ups(client, query, ans, results)
    if better_follow_ups and better_follow_ups != follow_ups:
        yield {"type": "followups", "options": better_follow_ups}

    yield {"type": "done"}
