"""
AI Analyst orchestrator — Azure OpenAI (gpt-4o) function-calling over the
deterministic catalog. One tool-decision turn + one grounded-answer turn.

Design goals vs. the reference backend (which generated SQL *and* Python plot code
and exec'd it): NO code generation/execution, NO SQL DB, numbers always come from
catalog.py. The model only (a) chooses a tool + params and (b) writes prose over the
real rows. Charts are built deterministically from the tool's own suggestion.

Secrets: the API key is read from AZURE_OPENAI_API_KEY (env / gitignored .env) and
is NEVER hardcoded. Endpoint/version/deployment are non-secret and may default.
"""
from __future__ import annotations
import json
import os

from app.ai import catalog, charts

AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://ed-gpt.openai.azure.com")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
_KEY = os.getenv("AZURE_OPENAI_API_KEY")  # required — no default


def has_key() -> bool:
    return bool(os.getenv("AZURE_OPENAI_API_KEY"))


def _client():
    from openai import AzureOpenAI
    key = os.getenv("AZURE_OPENAI_API_KEY")
    if not key:
        raise RuntimeError("AZURE_OPENAI_API_KEY is not set")
    return AzureOpenAI(azure_endpoint=AZURE_ENDPOINT, api_key=key, api_version=AZURE_API_VERSION)


SYSTEM = """You are the HCG Supply-Chain AI Analyst for a hospital-group analytics platform.
You answer questions about REAL data using the provided tools — never invent numbers.

Data you can reach (all real, last 6 months unless noted):
• revenue — billed IP+OP pharmacy revenue & true margin (MRP−cost); by manufacturer, hospital, category, product, or month.
• procurement — purchase spend by vendor/category/location/month, open purchase orders, price-consolidation savings.
• inventory — stock value, days-of-cover (DOH), aging distribution, non-moving stock, health mix, expiry.
• expiry — 6-band expiry ladder (Expired/0-30d/31-90d/91-180d/181-365d/365d+) and item lists per band.
• stock_risk — replenishment / stock-out risk and reorder lists.
• forecast — forward demand risk radar and fulfillment.
• overview — one-shot portfolio headline numbers.

Rules:
1. ALWAYS call a tool when the question needs data. Pick the single best tool + params.
2. After the tool returns, answer in 2–4 tight sentences. Lead with the number that answers the question.
3. The tool result already gives figures PRE-FORMATTED in ₹Cr / ₹L / %. Quote them EXACTLY as shown — never recompute, rescale, or re-convert units yourself.
4. Quote only figures present in the tool result. If a caveat/note is given, respect it.
5. Be an analyst: add one crisp insight (a share, a concentration, a risk), not just the raw number.
6. If the question is off-topic or not answerable from the data, say so briefly.
Never mention tools, JSON, or internal mechanics to the user."""


TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "revenue",
        "description": "Billed pharmacy revenue and true margin (MRP−cost). Use for sales/revenue/margin questions, top manufacturers, hospitals, products, categories, or revenue trend over months.",
        "parameters": {"type": "object", "properties": {
            "dimension": {"type": "string", "enum": ["manufacturer", "hospital", "category", "material", "month"], "description": "What to break revenue down by. 'material' = individual products."},
            "metric": {"type": "string", "enum": ["revenue", "margin", "margin_pct", "qty"]},
            "top_n": {"type": "integer", "description": "How many rows (default 10)."}}}}},
    {"type": "function", "function": {
        "name": "procurement",
        "description": "Purchase/procurement analytics: vendor spend, spend by category/location/month, open purchase orders, or price-consolidation savings opportunity.",
        "parameters": {"type": "object", "properties": {
            "view": {"type": "string", "enum": ["vendors", "spend", "open_po", "savings"], "description": "'vendors'=top suppliers; 'spend'+dimension; 'open_po'=undelivered orders; 'savings'=overpay vs own median."},
            "dimension": {"type": "string", "enum": ["vendor", "category", "location", "month"]},
            "top_n": {"type": "integer"}}}}},
    {"type": "function", "function": {
        "name": "inventory",
        "description": "Inventory analytics: stock value by category, days-of-cover (DOH), aging distribution, non-moving stock, health mix, or expiry.",
        "parameters": {"type": "object", "properties": {
            "view": {"type": "string", "enum": ["stock_value", "doh", "aging", "non_moving", "health", "expiry"]},
            "top_n": {"type": "integer"}}}}},
    {"type": "function", "function": {
        "name": "expiry",
        "description": "Near-expiry exposure. No slab → the full 6-band ladder by value. With a slab → the item list for that band.",
        "parameters": {"type": "object", "properties": {
            "slab": {"type": "string", "enum": ["Expired", "0-30d", "31-90d", "91-180d", "181-365d", "365d+"]},
            "top_n": {"type": "integer"}}}}},
    {"type": "function", "function": {
        "name": "stock_risk",
        "description": "Replenishment / stock-out risk. No status → risk-band summary. status='stock-out' or 'reorder' → items needing reorder.",
        "parameters": {"type": "object", "properties": {
            "status": {"type": "string", "enum": ["stock-out", "reorder", "overstock"]},
            "top_n": {"type": "integer"}}}}},
    {"type": "function", "function": {
        "name": "forecast",
        "description": "Forward-looking risk radar and fulfillment. view: demand (radar), fulfillment.",
        "parameters": {"type": "object", "properties": {
            "view": {"type": "string", "enum": ["demand", "fulfillment"]}}}}},
    {"type": "function", "function": {
        "name": "overview",
        "description": "One-shot portfolio headline numbers (revenue, margin, stock value, purchase value, expiry). Use for 'how are we doing' / summary questions.",
        "parameters": {"type": "object", "properties": {}}}},
]


def _fmt(v, kind):
    """Pre-format numbers EXACTLY as they should appear, so the model never
    recomputes ₹ units (LLMs are unreliable at crore/lakh conversion)."""
    if v is None or v == "":
        return "—"
    try:
        n = float(v)
    except (TypeError, ValueError):
        return str(v)
    if kind == "inr":
        a = abs(n)
        if a >= 1e7:
            return f"₹{n / 1e7:.2f} Cr"
        if a >= 1e5:
            return f"₹{n / 1e5:.2f} L"
        if a >= 1e3:
            return f"₹{n / 1e3:.1f} K"
        return f"₹{round(n)}"
    if kind == "pct":
        return f"{n:.1f}%"
    if kind == "days":
        return f"{round(n)} d"
    if kind == "num":
        return f"{n:,.0f}"
    return str(v)


def _compact(result: dict) -> dict:
    """Trim + PRE-FORMAT a Result before feeding it back to the model (correct units,
    low tokens). The model must echo these strings verbatim — never recompute."""
    cols = result.get("columns", [])
    kinds = {c["key"]: c.get("kind", "num") for c in cols}
    rows_fmt = []
    for r in result.get("rows", [])[:20]:
        rows_fmt.append({c["label"]: _fmt(r.get(c["key"]), kinds.get(c["key"], "num")) for c in cols})
    # stats: format inr-looking scalars to ₹Cr/L too (heuristic on key name + magnitude)
    stats_fmt = {}
    for k, v in (result.get("stats") or {}).items():
        if isinstance(v, (int, float)):
            kind = "inr" if any(t in k for t in ("value", "revenue", "margin", "spend", "cost", "opportunity", "180d")) and "pct" not in k else \
                   "pct" if "pct" in k or "percent" in k else \
                   "days" if "doh" in k or "days" in k else "num"
            stats_fmt[k] = _fmt(v, kind)
        else:
            stats_fmt[k] = v
    return {
        "title": result.get("title"),
        "stats": stats_fmt,
        "note": result.get("note"),
        "rows": rows_fmt,
        "row_count": len(result.get("rows", [])),
        "_instruction": "All figures above are already formatted (₹Cr/L, %). Quote them EXACTLY as shown — never recompute or rescale.",
    }


def answer(query: str, history: list | None = None):
    """Generator of event dicts: step / token / answer / chart / table / done / error."""
    if not has_key():
        yield {"type": "error", "text": "The AI Analyst isn't configured yet — set AZURE_OPENAI_API_KEY on the server."}
        return
    try:
        client = _client()
    except Exception as e:
        yield {"type": "error", "text": f"AI service unavailable: {e}"}
        return

    yield {"type": "step", "text": "Understanding your question"}
    messages = [{"role": "system", "content": SYSTEM}]
    for h in (history or [])[-6:]:
        if h.get("role") in ("user", "assistant") and h.get("content"):
            messages.append({"role": h["role"], "content": str(h["content"])[:2000]})
    messages.append({"role": "user", "content": query})

    try:
        first = client.chat.completions.create(
            model=AZURE_DEPLOYMENT, messages=messages, tools=TOOL_SCHEMAS,
            tool_choice="auto", temperature=0)
    except Exception as e:
        yield {"type": "error", "text": f"AI request failed: {e}"}
        return

    msg = first.choices[0].message
    if not msg.tool_calls:
        yield {"type": "answer", "text": msg.content or "I can help with revenue, inventory, procurement, expiry and forecast analytics — ask me anything about the numbers."}
        yield {"type": "done"}
        return

    messages.append({"role": "assistant", "content": msg.content or None,
                     "tool_calls": [{"id": tc.id, "type": "function",
                                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                                    for tc in msg.tool_calls]})
    primary = None
    for tc in msg.tool_calls:
        name = tc.function.name
        try:
            args = json.loads(tc.function.arguments or "{}")
        except Exception:
            args = {}
        yield {"type": "step", "text": f"Querying {name.replace('_', ' ')}"}
        res = catalog.run_tool(name, args)
        if primary is None:
            primary = res
        messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(_compact(res))})

    yield {"type": "step", "text": "Composing the answer"}
    full = ""
    try:
        stream = client.chat.completions.create(model=AZURE_DEPLOYMENT, messages=messages,
                                                temperature=0.2, stream=True)
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            piece = getattr(delta, "content", None)
            if piece:
                full += piece
                yield {"type": "token", "text": piece}
    except Exception as e:
        yield {"type": "error", "text": f"Answer generation failed: {e}"}
        return
    yield {"type": "answer", "text": full}

    if primary and primary.get("rows"):
        chart = charts.build(primary)
        if chart:
            yield {"type": "chart", "plotly": chart, "title": primary.get("title", "")}
        yield {"type": "table", "table": {"title": primary.get("title", ""),
                                          "columns": primary.get("columns", []),
                                          "rows": primary.get("rows", [])[:25]},
               "note": primary.get("note", "")}
    yield {"type": "done"}
