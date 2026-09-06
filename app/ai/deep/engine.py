"""Deep mode: a reasoning loop over verified primitives.

Fast mode (app/ai/orchestrator.py) answers in one pass and is untouched by this file. This
is the other setting: for a question where the answer is not a number but an explanation,
and where being right matters more than being quick.

WHY IT IS SHAPED LIKE THIS
--------------------------
The accuracy failures in this app were never SQL-syntax failures. They were semantic:
answering about the whole warehouse when one product was asked about, inventing a month
column that does not exist, filtering `vendor_name` for a manufacturer and reporting the
empty result as fact. One model, one query, one shot, narrating whatever came back.

A human analyst does not work that way. They run twenty small queries and think between
them. So the loop is:

  FRAME       ask the SCHEMA what is answerable before asking the model what to do
  PLAN        decompose into sub-questions, each answerable by one query
  INVESTIGATE run them in parallel, every one scope-guarded
  CORROBORATE re-derive the headline figures by a different route, on a DIFFERENT model
  CRITIQUE    an agent whose only job is to refute, plus one hunting for gaps
  SYNTHESISE  write the brief from confirmed findings only

The precision comes from phases 4 and 5, not from phases 2 and 3. Running more agents that
all reason the same way produces confident consensus on a shared wrong premise — five
agents agreeing beautifully and all wrong together. Independent DERIVATION is what makes a
figure trustworthy; an adversary is what makes a conclusion trustworthy.

ISOLATION
---------
This module and its package own the deep path completely. It imports the warehouse, the
scope guards and the chart builder — all read-only, one-way — and imports nothing from
orchestrator.py. Deep mode cannot regress the two-second path that answers most questions.
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from app.ai import charts, resolve as resolver, scope, warehouse
from app.ai.deep import capability, llm, sanity, schemas, shapes, tools

MAX_SUBQUESTIONS = 6
MAX_ROUNDS = 2          # investigate → critique → (one more investigate) → stop
PARALLEL = 4            # Azure rate limits bite harder than the CPU does
ROW_CAP = 200
MAX_TOOL_STEPS = 8      # a worker that hasn't found it in eight looks isn't going to

# Capitalised words that name nothing in a warehouse — probing these wastes a round-trip
# and binds noise.
_STOPWORDS = {
    "what", "which", "how", "show", "give", "list", "our", "the", "and", "for", "are", "was",
    "top", "most", "least", "best", "worst", "compare", "across", "each", "every", "total",
    "does", "have", "with", "from", "into", "them", "their", "this", "that", "over", "under",
    "why", "when", "where", "who", "much", "many", "make", "made", "than", "then", "also",
}


# ── small helpers ────────────────────────────────────────────────────────────
# A column that identifies a row rather than measuring it. `year` is the one that did the
# damage: it is numeric, so it typed as a measure, got picked as the chart's value, and
# produced twelve identical bars all reading "2,026" — a chart of the year, by month. It
# was also comma-formatted into "2,025", which is not a year anyone writes.
_IDENTIFIER_RE = re.compile(
    r"^(year|yr|month_num|week|quarter|id|.*_id|.*_code|material|plant|po_no|gr_no|invoice.*)$", re.I)

# Months arrive as names and sort alphabetically — April, August, December, February…
# which is not a trend, it is an alphabet. This is the order a reader expects.
_MONTHS = ["january", "february", "march", "april", "may", "june",
           "july", "august", "september", "october", "november", "december"]
_MONTH_INDEX = {m: i for i, m in enumerate(_MONTHS)}
_MONTH_INDEX.update({m[:3]: i for i, m in enumerate(_MONTHS)})


def _kind(col: str, rows: list) -> str:
    """Unit of a column, from its name.

    ORDER MATTERS, and getting it wrong is not cosmetic. `value_share_pct` contains
    "value", so an inr-first check typed it as money and the brief reported "value share
    percentages as high as ₹85" — a percentage rendered as rupees, in the engine whose
    entire purpose is not doing that. The narrower suffixes are therefore tested first;
    "money" is the fallback, never the first guess.
    """
    c = col.lower()
    if _IDENTIFIER_RE.match(c):
        return "id"
    if c.endswith("_pct") or c.endswith("_percent") or c.endswith("_%") or "percent" in c or "share_pct" in c:
        return "pct"
    if c.endswith("_days") or c.endswith("_day") or "lead_time" in c or c.startswith("days_"):
        return "days"
    if any(k in c for k in ("revenue", "cost", "margin", "value", "price", "spend", "amount",
                            "exposure", "mrp")):
        return "inr"
    if "pct" in c:
        return "pct"
    if "day" in c:
        return "days"
    sample = next((r.get(col) for r in rows if r.get(col) is not None), None)
    return "num" if isinstance(sample, (int, float)) else "text"


def _fmt(v, kind: str) -> str:
    if v is None:
        return "—"
    if kind == "text":
        return str(v)
    if kind == "id":
        # a year is 2026, never 2,026; a material code is a label, not a quantity
        try:
            f = float(v)
            return str(int(f)) if f == int(f) else str(v)
        except (TypeError, ValueError):
            return str(v)
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
        return f"₹{n:,.0f}"
    if kind == "pct":
        return f"{n:.1f}%"
    if kind == "days":
        return f"{round(n)} d"
    return f"{n:,.0f}" if abs(n) >= 1000 or n == int(n) else f"{n:,.2f}"


def _chart_title(finding: dict, measure: str) -> str:
    """A short label describing WHAT IS PLOTTED.

    It used to take the planner's sub-question verbatim, which produced "What is the sales
    trend of Keytruda across different plants" sitting above a chart of monthly PURCHASING
    quantities — a heading that contradicted both the data and the brief's own opening
    line. The measure is what is actually on the axis, so it names the chart; the
    sub-question is only a fallback, and never when it says "sales" over non-sales data.
    """
    m = (measure or "").replace("_", " ").strip()
    q = _hospitalise((finding["sub"].get("question") or "").strip().rstrip("?"))
    if m:
        title = m[0].upper() + m[1:]
        # keep the entity if the sub-question named one and the measure alone is generic
        return _hospitalise(title)
    if 3 < len(q) <= 70:
        return q[0].upper() + q[1:]
    return "Result"


def _order_rows(res: dict) -> dict:
    """Put a month/period result back in time order before it is shown or charted."""
    cols = res.get("columns") or []
    rows = list(res.get("rows") or [])
    mcol = next((c for c in cols if c.lower() in ("month", "month_name", "period")), None)
    if not mcol or not rows:
        return res
    ycol = next((c for c in cols if c.lower() in ("year", "yr")), None)

    def key(r):
        raw = str(r.get(mcol) or "").strip().lower()
        idx = _MONTH_INDEX.get(raw, _MONTH_INDEX.get(raw[:3], 99))
        if idx == 99 and "-" in raw:          # already an ISO "2026-01"
            return (0, raw)
        y = 0
        if ycol:
            try:
                y = int(float(r.get(ycol) or 0))
            except (TypeError, ValueError):
                y = 0
        return (y, idx)

    try:
        rows.sort(key=key)
    except Exception:
        return res
    return {**res, "rows": rows}


def _kpi_rows(out: dict) -> dict:
    """A canonical KPI payload as {columns, rows}, so a KPI-backed finding renders exactly
    like a query-backed one — same table, same chart, same formatting."""
    data = ((out.get("payload") or {}).get("data") or {})
    # A KPI payload holds several views of itself and the RIGHT one is the breakdown, not
    # the summary. `near-expiry` keys are totals/buckets/timeline/categories/ladder, and
    # "buckets" was missing from this list — so it fell through to `totals` and the answer
    # reported `exposure` (₹1.98 Cr of TOTAL near-expiry) as "19,807,976 units expiring in
    # 90 days". The banded breakdown is what a 90-day question is actually asking for.
    for key in ("buckets", "bands", "rows", "items", "breakdown", "series",
                "vendors", "plants", "groups", "categories", "timeline", "ladder"):
        v = data.get(key)
        if isinstance(v, list) and v and isinstance(v[0], dict):
            cols = list(v[0].keys())
            return {"columns": cols, "rows": v[:200], "row_count": len(v)}
    # Generic fallback: the largest list of records in the payload. The named list above
    # keeps the RIGHT view winning where two exist (near-expiry: buckets, not categories),
    # but a KPI naming its breakdown something unforeseen used to fall through to `totals`
    # — which is how vendor-volume-contribution, whose payload literally carries
    # top1 = 45.8%, produced "100% of ₹649.91 Cr from a single vendor".
    best_key, best_len = None, 0
    for k, v in data.items():
        if isinstance(v, list) and len(v) > best_len and v and isinstance(v[0], dict):
            best_key, best_len = k, len(v)
    if best_key:
        v = data[best_key]
        return {"columns": list(v[0].keys()), "rows": v[:200], "row_count": len(v)}
    totals = data.get("totals")
    if isinstance(totals, dict) and totals:
        return {"columns": list(totals.keys()), "rows": [totals], "row_count": 1}
    return {"columns": ["metric"], "rows": [{"metric": str(data)[:200]}], "row_count": 1}


def _table_payload(res: dict, title: str) -> dict:
    cols = res.get("columns") or []
    rows = res.get("rows") or []
    kinds = {c: _kind(c, rows) for c in cols}
    # Column labels reach the PDF export as well as the screen, and "plant" was still
    # arriving there long after the prose said "hospital".
    return {"title": _hospitalise(title),
            "columns": [{"key": c, "label": _hospitalise(c), "kind": kinds[c]} for c in cols],
            "rows": rows[:50]}


# An alias carries no unit. `SELECT SUM(line_value) AS total_procurement` produced a column
# called "total_procurement", which matches no money pattern, so ₹174 Cr of Bangalore
# procurement was handed to the writer as the bare integer 1740233400.36 and printed that
# way. The SQL already says what it is: the alias inherits the unit of the column it
# aggregates, whatever the model chose to call it.
_AGG_ALIAS = re.compile(
    r"\b(?:SUM|AVG|MIN|MAX|ROUND|TOTAL)\s*\(\s*(?:DISTINCT\s+)?([A-Za-z_][\w.]*)[^()]*\)"
    r"\s*(?:AS\s+)?([A-Za-z_]\w*)", re.I)
_SQL_WORDS = {"as", "from", "where", "group", "order", "having", "limit", "join", "on",
              "and", "or", "then", "else", "end", "when", "desc", "asc", "by"}


def _alias_units(sql: str) -> dict:
    """{alias: kind} for aggregates whose SOURCE column names a unit."""
    out = {}
    for src, alias in _AGG_ALIAS.findall(sql or ""):
        if alias.lower() in _SQL_WORDS:
            continue
        # a numeric sample is supplied because _kind falls back to "text" when it has no
        # rows to look at, and an alias inheriting "text" would suppress formatting entirely
        col = src.split(".")[-1]
        kind = _kind(col, [{col: 0.0}])
        if kind in ("inr", "pct", "days"):
            out[alias.lower()] = kind
    return out


def _compact(res: dict, limit: int = 25, sql: str = "") -> str:
    """A result as the model should see it: already formatted, so it quotes rather than
    converts. Prose that converts raw rupees itself is where the 10x errors came from."""
    cols = res.get("columns") or []
    rows = res.get("rows") or []
    inherited = _alias_units(sql)
    kinds = {c: (inherited.get(c.lower()) if _kind(c, rows) == "num" else None) or _kind(c, rows)
             for c in cols}
    out = [" | ".join(cols)]
    for r in rows[:limit]:
        out.append(" | ".join(_fmt(r.get(c), kinds[c]) for c in cols))
    if len(rows) > limit:
        out.append(f"… {len(rows) - limit} more rows")
    return "\n".join(out)


# The prompt says "say hospital, never plant" and the model said "at plant AH01" anyway.
# The fast path learned this lesson already (tests/test_output_sanitizer.py records a live
# persona audit where a prompt instruction failed roughly half the time, including right
# after an explicit correction). An instruction is a preference; this is a guarantee.
_PLANT_RE = re.compile(r"\b(plant)(s?)\b", re.I)


def _hospitalise(text: str) -> str:
    def sub(m):
        word = "hospital" + (m.group(2) or "")
        return word.capitalize() if m.group(1)[0].isupper() else word
    return _PLANT_RE.sub(sub, text or "")


# A rupee sign is an assertion that the number is money, so the house format can be applied
# without knowing anything about where it came from. This is the backstop for the alias rule
# in _alias_units: that one needs to recognise the SQL shape, and `SUM(line_value) OVER ()
# AS total` defeated it — leaving "Bangalore hospitals spend ₹1,743,233,400.48" in a brief
# whose every other figure was in crores. Nobody reads nine digits.
_RUPEES = re.compile(r"₹\s?(\d[\d,]*(?:\.\d+)?)(?!\s*(?:Cr|L\b|lakh|crore|k\b))", re.I)


def _rupees_in_scale(text: str) -> str:
    def sub(m):
        try:
            n = float(m.group(1).replace(",", ""))
        except ValueError:
            return m.group(0)
        if n >= 1e7:
            return f"₹{n / 1e7:.2f} Cr"
        if n >= 1e5:
            return f"₹{n / 1e5:.2f} L"
        return m.group(0)
    return _RUPEES.sub(sub, text or "")


# Which measure family a table belongs to. Kept as prefixes so a new mart lands in the
# right family automatically.
_FAMILY_TABLES = {
    "sales":       ("sales_by_", "sales_monthly", "sales_totals", "forecast_sales"),
    "purchasing":  ("fact_po", "fact_grn", "mart_procurement", "kpi_purchase", "kpi_monthly_purchase",
                    "kpi_vendor_volume", "kpi_procurement", "mart_material_vendor", "mart_material_price"),
    "consumption": ("fact_consumption", "kpi_units_consumed", "kpi_billable_consumption",
                    "kpi_consumption_by_department"),
    "stock":       ("fact_inventory", "kpi_stock", "kpi_doh", "kpi_aging", "kpi_near_expiry",
                    "kpi_inventory_aging", "kpi_non_moving"),
}
_FAMILY_WORDS = {
    "sales":       ("sales", "sold", "selling", "revenue", "billed", "turnover"),
    "purchasing":  ("purchase", "purchasing", "procure", "procurement", "bought", "po ", "vendor spend"),
    "consumption": ("consumption", "consumed", "usage", "used", "issued"),
    "stock":       ("stock", "inventory", "on hand", "expiry", "expiring", "aging"),
}


def _family_of_table(sql: str) -> str | None:
    low = (sql or "").lower()
    for fam, prefixes in _FAMILY_TABLES.items():
        if any(p in low for p in prefixes):
            return fam
    return None


def _family_asked(question: str) -> str | None:
    low = (question or "").lower()
    for fam, words in _FAMILY_WORDS.items():
        if any(w in low for w in words):
            return fam
    return None


def _measure_disclosure(question: str, findings: list[dict], primary: dict | None = None) -> str:
    """Say so, in code, when the answer is about a DIFFERENT measure than was asked for.

    This warehouse has no material-by-month SALES grain, so "sales trend of KEYTRUDA" gets
    answered from PURCHASING. That substitution is legitimate and useful — silently
    labelling the result "Sales Trend" is not, and it is a wrong answer however good the
    arithmetic. The synthesis prompt has asked for this disclosure since the first version
    and the model supplies it perhaps half the time, which is exactly the reliability a
    prompt gives you. Plant->Hospital taught the same lesson twice; this is the third.
    """
    asked = _family_asked(question)
    if not asked:
        return ""
    # Judge the HEADLINE, not every table touched. A turn that reads the sales total for
    # context and builds its monthly series from purchasing was staying silent, because
    # "sales" appeared somewhere in the turn — while the number in the first sentence,
    # the one the reader takes away, was purchasing all along.
    if primary is not None:
        fam = _family_of_table(primary.get("sql", ""))
        if not fam or fam == asked:
            return ""
        got = fam
    else:
        used = {f for f in (_family_of_table(x.get("sql", "")) for x in findings) if f}
        if not used or asked in used:
            return ""
        got = " and ".join(sorted(used))
    return (f"There is no {asked} figure available at this level for what you asked; "
            f"what follows is {got.upper()}.\n\n")


_VALUE_KEYS = {"value", "total", "share_pct", "change_pct_first_to_last",
               "swing_pct_trough_to_peak", "top1_share_pct", "top3_share_pct"}


def _format_derived(derived: list[dict]) -> list[dict]:
    """Render every derived number in the unit its measure is actually in."""
    def walk(node, kind):
        if isinstance(node, dict):
            # "_pct" ANYWHERE in the key, and tested first: `change_pct_first_to_last`
            # does not END with _pct, so it fell through to the money branch and a −3.54%
            # change rendered as "₹-4".
            return {k: (_fmt(v, "pct") if "_pct" in k and isinstance(v, (int, float))
                        else _fmt(v, kind) if k in _VALUE_KEYS and isinstance(v, (int, float))
                        else walk(v, kind))
                    for k, v in node.items()}
        if isinstance(node, list):
            return [walk(x, kind) for x in node]
        return node

    out = []
    for d in derived:
        kind = _kind(str(d.get("measure") or ""), [{}])
        if kind in ("id", "text"):
            kind = "num"
        out.append(walk(d, kind))
    return out


# Which column names satisfy a requested grain. "Which products move the most units"
# answered "M070-STATIONARY" — a CATEGORY — and "which manufacturer sells most" answered
# from purchasing. The question says what it wants broken down by; a finding that cannot
# provide it is context, never the headline.
_GRAIN_COLUMNS = {
    # `name` is deliberately NOT here. The units-per-SKU KPI has a column literally called
    # `name` holding CATEGORY labels, so accepting it let "which products move the most
    # units" answer "M070-STATIONARY" and still pass the grain check.
    "material":     re.compile(r"^(material|material_id|material_desc|generic_name|item|sku)$", re.I),
    "hospital":     re.compile(r"^(hospital|plant|plant_name|site)$", re.I),
    "vendor":       re.compile(r"^(vendor|vendor_name|vendor_code)$", re.I),
    "manufacturer": re.compile(r"^(manufacturer|manufacturer_desc)$", re.I),
    "category":     re.compile(r"^(category|material_group|major_group_desc|minor_group_desc|group|name)$", re.I),
    "month":        re.compile(r"^(month|month_name|period|posting_date|year)$", re.I),
    "department":   re.compile(r"^(department|department_name|cost_ctr|costcenter)$", re.I),
}


def _serves_grain(res: dict, grain: str) -> bool:
    pat = _GRAIN_COLUMNS.get(grain)
    if not pat:
        return True
    return any(pat.match(c) for c in (res.get("columns") or []))


def _num_tokens(text: str) -> set[str]:
    return {t.replace(",", "") for t in re.findall(r"\d[\d,]*\.?\d*", text or "")}


# ── the loop ─────────────────────────────────────────────────────────────────
def answer(query: str, history: list | None = None):
    """Same generator contract as orchestrator.answer, so chat_service can swap paths
    with a single branch and the frontend needs no new event vocabulary."""
    if not llm.has_key():
        yield {"type": "answer", "text": "Deep analysis needs AZURE_OPENAI_API_KEY to be set.",
               "verified": None, "options": []}
        yield {"type": "done"}
        return

    cl = llm.client()
    hist = "\n".join(f"{m.get('role')}: {m.get('content')}" for m in (history or [])[-6:])
    findings: list[dict] = []
    queries: list[dict] = []
    entity_tokens: list[str] = []

    # ── PHASE 1 · FRAME ──────────────────────────────────────────────────────
    yield {"type": "step", "text": "Framing the question"}
    footprint = ""
    fp_counts: dict = {}
    m = re.search(r'"([^"]{3,60})"|“([^”]{3,60})”', query)
    if m:
        name = (m.group(1) or m.group(2) or "").strip()
        try:
            fp = warehouse.item_footprint(name)
            for row in (fp.get("matches") or [])[:1]:
                for k in ("material", "material_desc"):
                    v = str(row.get(k) or "").strip()
                    if v:
                        entity_tokens.append(v)
                        entity_tokens += re.findall(r"[A-Za-z]{4,}", v)[:2]
            fp_counts = fp.get("footprint") or {}
            if fp.get("probably_not_an_item") and fp.get("also_found_in"):
                where = ", ".join(f"{e['table']}.{e['column']} ({e['rows']:,} rows)"
                                  for e in fp["also_found_in"][:3])
                lessons.append(f"'{name}' is NOT a product — it is held in {where}. Filter that "
                               f"column, not material_desc. Any item-name match on it is a "
                               f"coincidental substring.")
            footprint = json.dumps(fp)[:1800]
            yield {"type": "sql", "purpose": f"footprint of “{name}”",
                   "sql": f"-- lookup_item('{name}')", "rows": fp.get("match_count", 0)}
        except Exception:
            pass

    lessons: list[str] = []
    lesson_lock = Lock()

    # The user's vocabulary is not the schema's. Seeded before anything is framed, because
    # a missing synonym gets reported as missing DATA.
    lessons.extend(capability.vocabulary())
    # what ONE ROW means: COUNT(*) on a material-per-site table is not an item count
    lessons.extend(capability.grain_notes())

    # TYPED RESOLUTION, before anything is framed or planned. find_value() below says
    # WHERE a literal lives; this says WHAT IT IS — a drug, a city, a manufacturer, a
    # category — and what the question is measuring. Naming the things first is the step a
    # human never skips and this engine had no equivalent of.
    requested_grains: list[str] = []
    requested_measures: list[str] = []
    try:
        from app.ai import resolve as _resolve
        _rb = _resolve.brief(query)
        if _rb:
            lessons.append(_rb)
            _r = _resolve.resolve(query)
            requested_grains = _r.get("grains") or []
            requested_measures = _r.get("measures") or []
            # bind every confidently-typed entity so a query claiming to be about it must
            # actually reference it
            entity_tokens.extend(e["text"] for e in _r["entities"] if e.get("exact"))
            entity_tokens.extend(_r["cities"])
    except Exception:
        pass

    # RESOLVE THE ENTITIES THE QUESTION NAMES, whatever kind they are.
    #
    # Entity binding only ever triggered on a QUOTED item, so "Bangalore", "MSD" and vendor
    # names bound nothing and any query could claim to be about them. Asked for the top
    # selling drugs in Bangalore, the engine returned "KEYTRUDA ₹16.37 Cr in Bangalore
    # hospitals" — and on the previous run ₹10.78 Cr — for a filter that cannot exist:
    # a city lives only in dim_plant.plant_name, which shares no rows with the sales tables.
    #
    # find_value() already knows where any literal lives. Using it on the capitalised words
    # of the question binds every entity kind, not just materials.
    resolved_where: dict[str, list[str]] = {}
    for cand in dict.fromkeys(re.findall(r"\b[A-Z][A-Za-z]{3,}\b", query)):
        if cand.lower() in _STOPWORDS:
            continue
        try:
            hits = tools.find_value(cand).get("found_in") or []
        except Exception:
            continue
        if hits:
            resolved_where[cand] = [f"{h['table']}.{h['column']}" for h in hits[:4]]
    if resolved_where:
        # NOTE them, but do NOT bind them. Binding every capitalised word bound the word
        # "Rank" — capitalised only because it starts the sentence — and then rejected every
        # query that failed to mention it, so "Rank our hospitals by sales revenue" returned
        # "I couldn't establish anything". Binding is now done from resolve.py, which types
        # what it finds and ignores verbs and question words. A lesson saying "'Bangalore' exists only in
        # dim_plant.plant_name" was read past, and the engine still reported "KEYTRUDA
        # ₹10.78 Cr in Bangalore hospitals" from a sales table a city cannot reach. Adding
        # the names to entity_tokens makes missing_entity_scope REJECT any query that does
        # not reference them, so a sales query claiming to be city-filtered cannot run at
        # all — which is the only way this stops being a matter of the model's attention.
        for name, places in resolved_where.items():
            lessons.append(f"'{name}' exists ONLY in {', '.join(places)} — any figure claimed to be "
                           f"about it must come from a query that filters one of those columns. "
                           f"If the measure you need is in a table that cannot reach it, say so.")

    frame = llm.ask_json(
        cl, role="frame",
        system=("You frame analytics questions against a fixed warehouse. You are given the REAL "
                "schema — never assume a table or column that is not listed.\n\n"
                "VOCABULARY — the user's words are not the schema's:\n- "
                + "\n- ".join(capability.vocabulary())
                + "\n\nANSWER SHAPES (pick the one this question really is):\n" + shapes.catalog()
                + "\n\nCANONICAL METRICS (kpi_key — the dashboard's own calculations):\n"
                + ", ".join(tools.kpi_keys())
                + "\n\n" + capability.brief()),
        user=f"Question: {query}\n\nRecent conversation:\n{hist}\n\nItem footprint:\n{footprint}",
        schema_hint=schemas.FRAME)

    # The frame agent's "answerable" verdict is ADVISORY and is deliberately NOT a veto.
    # Asked whether vendor-concentration risk was answerable it said no — while the data to
    # answer it sat in kpi_vendor_volume. A model's opinion about the data is worth less
    # than the data, so the investigation runs and the honest "can't answer this" is
    # reported only if every line of enquiry actually comes back empty (below).
    blocked_reason = frame.get("blocked_reason") if frame.get("answerable") is False else ""

    fam = frame.get("entity_family")
    cap_brief = capability.brief(fam if fam in capability.ENTITY_FAMILIES else None)

    # ── PHASE 2 · PLAN ───────────────────────────────────────────────────────
    # ── PHASE 3 · INVESTIGATE (parallel) ─────────────────────────────────────
    # A LESSON BOARD shared by every worker.
    #
    # An analyst who discovers that sales_by_material has no month column does not
    # rediscover it for the next question — and neither should six agents running in
    # parallel. Each worker writes what it learned about the SHAPE of the data here, and
    # every later worker reads it. One worker's dead end becomes everyone's starting
    # knowledge, which is the cheapest accuracy win in the engine.

    # SEED the board with what the schema already proves, before any worker spends a step
    # discovering it. Four workers independently hunted for a monthly sales grain and
    # tunnelled into forecast_sales; the fact that no sales table carries time is
    # derivable from information_schema in one pass and is exactly the knowledge they
    # needed. Deterministic facts should never cost a model call.
    # GROUND TRUTH FOR THIS ITEM, not a generalisation. lookup_item already counts the
    # rows this material has in EVERY table — and the planner was never shown it, so it
    # planned a "monthly consumption trend" for an item with zero consumption rows
    # (KEYTRUDA is billed-only; fact_consumption and kpi_units_consumed are both 0 while
    # fact_po has 462 and kpi_monthly_purchase_value has 108). A generic lesson said "use
    # purchasing or consumption" and it picked the empty one. Per-item counts remove the
    # guess entirely.
    if fp_counts:
        have = sorted(((t, n) for t, n in fp_counts.items() if n), key=lambda x: -x[1])
        none = sorted(t for t, n in fp_counts.items() if not n)
        if have:
            lessons.append("FOR THIS EXACT ITEM, tables that HAVE rows (row counts): "
                           + ", ".join(f"{t}({n})" for t, n in have[:14]))
        if none:
            lessons.append("FOR THIS EXACT ITEM these tables are EMPTY — never plan against "
                           "them: " + ", ".join(none[:12]))

    try:
        _fam = frame.get("entity_family") if isinstance(frame, dict) else None
        _prof = capability.profile(_fam if _fam in capability.ENTITY_FAMILIES else None)
        _timed = sorted(e["table"] for e in _prof["tables_with_time"])
        _sales_no_time = sorted(e["table"] for e in _prof["tables"]
                                if e["table"].startswith("sales") and not e["time"])
        if _timed:
            lessons.append("Tables carrying BOTH this entity and a time column: " + ", ".join(_timed[:14]))
        # structural traps discovered from the data, not asserted — e.g. the two hospital
        # code systems that look interchangeable and share zero rows
        for _note in capability.joinability():
            lessons.append(_note)
        if _sales_no_time:
            lessons.append("These sales tables have NO time/month column, so no trend can come "
                           "from them: " + ", ".join(_sales_no_time))
        lessons.append("There is no material-by-month SALES grain in this warehouse. Monthly "
                       "figures for an item exist only for PURCHASING (fact_po, fact_grn, "
                       "mart_procurement, kpi_monthly_purchase_value) and CONSUMPTION "
                       "(fact_consumption, kpi_units_consumed). Use one of those and say which.")
    except Exception:
        pass

    shape_name = (frame.get("shape") or "lookup").strip().lower()
    shape = shapes.get(shape_name)
    slot_spec = "\n".join(
        f"- {sl['id']} ({'REQUIRED' if sl.get('required') else 'optional'}): {sl['need']}"
        for sl in shape["slots"])
    yield {"type": "step", "text": f"Planning the investigation ({shape_name})"}
    plan = llm.ask_json(
        cl, role="plan",
        system=("You plan an analytics investigation. This question needs a "
                f"'{shape_name}' answer, which owes the reader: {shape['answer_must']}.\n\n"
                f"SLOTS TO FILL:\n{slot_spec}\n\n"
                "Each sub-question fills one slot and must be answerable by ONE query. Do not "
                "add sub-questions that only restate the question or re-fetch an identifier "
                "you already have.\n\n"
                # The workers had this and the planner did not, so it kept planning around a
                # grain the warehouse does not have — "monthly sales revenue for item X" —
                # and every worker then failed the same way. A plan made in ignorance of what
                # the data can do is why the whole turn came back empty.
                "WHAT THIS DATA CAN AND CANNOT DO — plan within it:\n- "
                + "\n- ".join(lessons) + "\n\nSCHEMA:\n" + cap_brief),
        user=f"Question: {query}\n\nFraming: {json.dumps(frame)[:600]}",
        schema_hint=schemas.PLAN)
    subs = (plan.get("sub_questions") or [])[:MAX_SUBQUESTIONS]
    if not subs:
        subs = [{"id": "q1", "question": query, "why": "direct", "table": ""}]
    # CANONICAL FIRST. If a dashboard metric already answers this, take its figure before
    # any SQL is written. "How much stock is expiring in 90 days" was being re-derived as
    # `expiry_date <= today + 90` — which silently includes stock that expired MONTHS ago
    # and returned 101,005 units against the canonical 45,223. A chatbot that disagrees
    # with the dashboard is worse than one that says nothing, and the definition of
    # "expiring" lives in exactly one place.
    # Deterministic first, the model's suggestion only as a fallback.
    canonical_totals: dict = {}
    kpi_key = capability.kpi_for(query) or (frame.get("kpi_key") or "").strip()
    if kpi_key and kpi_key in tools.kpi_keys():
        yield {"type": "step", "text": f"Taking the canonical figure for {kpi_key}"}
        out = tools.get_kpi(kpi_key)
        if out.get("canonical"):
            res = _kpi_rows(out)
            if res.get("row_count"):
                findings.append({"sub": {"id": "kpi", "question": f"canonical {kpi_key}"},
                                 "sql": f"-- get_kpi('{kpi_key}')", "res": res, "canonical": True,
                                 "purpose": f"{kpi_key} (the dashboard's own calculation)"})
                queries.append({"purpose": f"{kpi_key} (canonical)", "sql": f"-- get_kpi('{kpi_key}')",
                                "rows": res["row_count"]})
                yield {"type": "sql", "purpose": f"{kpi_key} (canonical)",
                       "sql": f"-- get_kpi('{kpi_key}')", "rows": res["row_count"]}
                # still single-threaded here, and _note_lesson is defined further down
                lessons.append(f"The canonical {kpi_key} figure is already retrieved — quote "
                               f"it, do not recompute it in SQL.")
                # The summary block often holds the very fact the question asks for —
                # vendor-volume-contribution carries top1/top5/top10 concentration shares
                # that no SQL needs to rediscover.
                _tot = ((out.get("payload") or {}).get("data") or {}).get("totals")
                if isinstance(_tot, dict) and _tot:
                    # FORMAT THEM. Handed the raw float 604679341.2 the model wrote
                    # "₹604.68 Cr" — a clean 10x overstatement of ₹60.47 Cr, because it
                    # divided by 1e6 instead of 1e7. Fast mode learned this exact lesson
                    # (see _format_kpi_payload and its regression test: ₹300.21 Cr printed
                    # for ₹30.02 Cr); every number handed to the model must already be in
                    # the form it should quote, so it never has to convert anything.
                    _flat = {k: (_fmt(v, _kind(k, [{k: v}])) if isinstance(v, (int, float)) else v)
                             for k, v in _tot.items() if isinstance(v, (int, float, str))}
                    if _flat:
                        lessons.append(f"Canonical {kpi_key} headline figures (authoritative, "
                                       f"quote directly): {json.dumps(_flat, default=str)[:400]}")
                        canonical_totals = {"canonical_metric": kpi_key, "totals": _flat}

    yield {"type": "step", "text": f"Investigating {len(subs)} lines of enquiry"}


    def _note_lesson(text: str) -> None:
        t = (text or "").strip()
        if not t:
            return
        with lesson_lock:
            if t not in lessons and len(lessons) < 24:
                lessons.append(t)

    def investigate(sub: dict, facts: str = "") -> dict:
        """One sub-question, worked the way a person works it: LOOK, then write.

        The previous version generated its final SQL in one shot from a static schema dump
        and had no way to check anything. It wrote `GROUP BY month` against three tables
        that have no month column — all three column lists were in its own prompt. It was
        not reasoning about the warehouse, it was completing a familiar pattern.

        This is a ReAct loop over tools.py: list, describe, sample, find the grain, find
        where a value lives, profile, run, LOOK at what came back, adjust. It ends when the
        model calls finish() with SQL that actually returned rows, or give_up() after
        genuinely checking. Steps are capped so a confused worker costs a bounded amount.
        """
        sys_prompt = (
            "You are a data analyst with direct access to the warehouse. Work like one.\n\n"
            "HARD RULE: never write a WHERE or GROUP BY on a column you have not SEEN, either "
            "in describe_table output or already in this conversation. If you are about to "
            "group by month, first call find_columns('month') and see which tables actually "
            "have one. Assuming a column exists because the question implies it is the single "
            "most common way to produce a confident wrong answer.\n\n"
            "METHOD: look before you write. If a query errors, read the error and change "
            "approach — never resubmit the same shape. If it returns 0 rows, use find_value or "
            "profile_column to find out why before concluding data is missing. If the grain you "
            "want does not exist, find the NEAREST grain that does and finish with that, saying "
            "plainly what you substituted. Sanity-check before finishing: do the row count and "
            "the magnitudes make sense?\n\n"
            "PREFER get_kpi. These are the dashboard's own calculations — correct by "
            "construction and already verified — so a metric taken from there needs no "
            "re-derivation and cannot disagree with what the user sees on their screen. "
            "Only write SQL for things no KPI covers. If you finish on a KPI, pass its key "
            "as the sql argument (e.g. \"-- get_kpi('near-expiry')\").\n\n"
            "Call finish() with the SQL that produced your answer, or give_up() once you have "
            "actually checked and the data cannot answer it.\n\n"
            "SCHEMA (a starting map — verify with describe_table, never trust it over what the "
            "tools return):\n" + cap_brief)

        with lesson_lock:
            known = list(lessons)
        user = (f"Original question: {query}\n"
                + (f"\nESTABLISHED FACTS (concrete values already found — use them, do not re-derive):\n{facts}\n" if facts else "")
                + ("\nWHAT OTHER ANALYSTS ALREADY LEARNED ABOUT THIS DATA:\n- " + "\n- ".join(known) + "\n" if known else "")
                + f"\nYour sub-question: {sub.get('question')}\nWhy it matters: {sub.get('why')}"
                + (f"\nSuggested starting table (a hint, not a constraint): {sub.get('table')}" if sub.get("table") else ""))

        msgs = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user}]
        ctx = {"entity_tokens": entity_tokens, "question": query}
        last_result = None
        last_sql = ""
        last_kpi = None          # a canonical KPI payload, flattened for presentation
        # Observed in the trace: find_value('KEYTRUDA 100MG INJ VIAL') called NINE times in
        # one worker, identical arguments, identical answer. A loop with no memory of what
        # it has already looked at spends its whole step budget re-looking. The cache is
        # the memory; the nudge is what breaks the cycle.
        seen_calls: dict[str, dict] = {}

        for _step in range(MAX_TOOL_STEPS):
            try:
                resp = llm.chat(cl, model=llm.model_for("sql"), temperature=0,
                                messages=msgs, tools=tools.SPECS, tool_choice="auto")
            except Exception as e:
                return {"sub": sub, "skipped": f"model error: {str(e)[:150]}"}
            m = resp.choices[0].message
            calls = getattr(m, "tool_calls", None) or []
            if not calls:
                if last_result is not None:
                    break
                msgs.append({"role": "user", "content":
                             "Do not answer in prose. Use the tools, then call finish() or give_up()."})
                continue

            msgs.append({"role": "assistant", "content": m.content or "",
                         "tool_calls": [{"id": c.id, "type": "function",
                                         "function": {"name": c.function.name,
                                                      "arguments": c.function.arguments}} for c in calls]})
            stop = None
            for c in calls:
                try:
                    args = json.loads(c.function.arguments or "{}")
                except Exception:
                    args = {}
                name = c.function.name

                if name == "finish":
                    sql = (args.get("sql") or last_sql).strip()
                    if sql.startswith("--") and last_kpi is not None:
                        _note_lesson(args.get("what_it_shows", "")[:180])
                        stop = {"sub": sub, "sql": sql, "res": last_kpi, "canonical": True,
                                "purpose": args.get("what_it_shows") or sub.get("question")}
                        break
                    if sql and (last_result is None or sql != last_sql):
                        last_result = tools.run_query(sql, entity_tokens, query)
                        last_sql = sql
                    if last_result and last_result.get("_full") is not None:
                        _note_lesson(args.get("what_it_shows", "")[:180])
                        stop = {"sub": sub, "sql": last_sql, "res": last_result["_full"],
                                "purpose": args.get("what_it_shows") or sub.get("question")}
                        break
                    msgs.append({"role": "tool", "tool_call_id": c.id, "content": json.dumps(
                        {"error": "that SQL returned no rows — you cannot finish on it"})})
                    continue

                if name == "give_up":
                    _note_lesson(args.get("reason", "")[:180])
                    stop = {"sub": sub, "skipped": args.get("reason") or "gave up"}
                    break

                key = name + "|" + json.dumps(args, sort_keys=True, default=str)[:400]
                if key in seen_calls and name != "run_query":
                    msgs.append({"role": "tool", "tool_call_id": c.id, "content": tools.compact(
                        {**seen_calls[key],
                         "note": "You already called this with the same arguments. The answer has "
                                 "not changed. Use it and take a DIFFERENT next step."})})
                    continue
                out = tools.call(name, args, ctx)
                seen_calls[key] = {k: v for k, v in out.items() if not k.startswith("_")}
                if name == "get_kpi" and out.get("canonical"):
                    last_kpi = _kpi_rows(out)
                    last_sql = f"-- get_kpi('{args.get('key')}')"
                if name == "run_query":
                    last_sql = args.get("sql", "")
                    if out.get("_full") is not None:
                        last_result = out
                    elif out.get("error"):
                        err = str(out["error"])
                        if "not found in FROM clause" in err or "does not have a column" in err:
                            tbl = re.search(r"FROM\s+([A-Za-z_][A-Za-z0-9_]*)", last_sql or "", re.I)
                            col = re.search(r'column "?([A-Za-z_][A-Za-z0-9_]*)"?', err)
                            if tbl and col:
                                _note_lesson(f"{tbl.group(1)} has NO '{col.group(1)}' column")
                if name == "describe_table" and out.get("columns"):
                    _note_lesson(f"{out['table']} columns: " + ", ".join(x["name"] for x in out["columns"])[:170])
                msgs.append({"role": "tool", "tool_call_id": c.id, "content": tools.compact(out)})
            if stop:
                return stop

        if last_result is not None and last_result.get("_full") is not None:
            # Check the arithmetic before the result is allowed to be evidence. A number
            # that cannot be true is worse than no number, because it reads as an answer.
            full = last_result["_full"]
            # An unnamed bucket is demoted IN THE ROWS, not just warned about. The warning
            # was there and "our biggest spend category is Uncategorized, ₹173.31 Cr" was
            # written anyway — whatever reads the first row has to see a real category.
            sanity.sink_placeholders(full)
            return {"sub": sub, "sql": last_sql, "res": full,
                    "purpose": sub.get("question"),
                    "warnings": sanity.check(last_sql, full)}
        return {"sub": sub, "skipped": "ran out of steps without a usable result"}


    def _evidence(fs):
        out = []
        for f in fs:
            block = (f"[{f['sub'].get('id','?')}] {_hospitalise(f['purpose'])}\n"
                     f"{_compact(f['res'], sql=f.get('sql') or '')}")
            for w in f.get("warnings") or ():
                block += f"\n!! {w}"
            out.append(block)
        return "\n\n".join(out)

    def _source_words(fs) -> str:
        """The verbs the evidence licenses, from the tables it was actually read out of."""
        tabs = {t for f in fs for t in re.findall(r"\bFROM\s+([A-Za-z_]\w*)|\bJOIN\s+([A-Za-z_]\w*)",
                                                  f.get("sql") or "") for t in (t if isinstance(t, tuple) else (t,)) if t}
        return resolver.source_vocabulary(sorted(tabs))

    def _impossible(f) -> bool:
        return any(w.startswith("IMPOSSIBLE") for w in f.get("warnings") or ())

    # TWO WAVES. Everything that stands alone runs first and in parallel; what depends on
    # knowing a specific entity runs second, with the first wave's results handed to it as
    # concrete values. This is the difference between "what is the top vendor's lead time"
    # being unanswerable and being one query.
    wave1 = [s for s in subs if not (s.get("needs") or "").strip()] or subs
    wave2 = [s for s in subs if s not in wave1]

    def run_wave(batch, facts=""):
        done = []
        with ThreadPoolExecutor(max_workers=PARALLEL) as pool:
            futs = [pool.submit(investigate, s, facts) for s in batch]
            for fut in as_completed(futs):
                try:
                    done.append(fut.result())
                except Exception:
                    pass
        return done

    for out in run_wave(wave1):
        if out.get("res") is not None:
            findings.append(out)
            queries.append({"purpose": out["purpose"], "sql": out["sql"], "rows": out["res"]["row_count"]})
            yield {"type": "sql", "purpose": out["purpose"], "sql": out["sql"], "rows": out["res"]["row_count"]}
        else:
            yield {"type": "step", "text": f"Ruled out: {out['sub'].get('question','')[:60]}"}

    if wave2:
        facts = "\n\n".join(
            f"{f['purpose']}:\n{_compact(f['res'], 10, sql=f.get('sql') or '')}" for f in findings)
        yield {"type": "step", "text": f"Following the {len(wave2)} dependent question(s)"}
        for out in run_wave(wave2, facts):
            if out.get("res") is not None:
                findings.append(out)
                queries.append({"purpose": out["purpose"], "sql": out["sql"], "rows": out["res"]["row_count"]})
                yield {"type": "sql", "purpose": out["purpose"], "sql": out["sql"], "rows": out["res"]["row_count"]}
            else:
                yield {"type": "step", "text": f"Ruled out: {out['sub'].get('question','')[:60]}"}

    # DIRECT FALLBACK. If decomposition produced nothing, answer the question ITSELF
    # before declaring it unanswerable. This is what a person does when their plan falls
    # apart: they stop planning and just write the obvious query.
    #
    # Without it the engine reported "I couldn't establish anything" for "what is our
    # single biggest selling product?" — one ORDER BY over one table — and claimed no
    # table links hospitals to inventory value while fact_inventory carries plant and
    # total_cost. Both were failures of the PLAN, reported as limits of the DATA, which is
    # the most damaging thing this system can do: it teaches the user their data is worse
    # than it is.
    if not findings:
        yield {"type": "step", "text": "Answering it directly instead"}
        direct = investigate({"id": "direct", "question": query,
                              "why": "answer the question as asked, without decomposing it",
                              "table": ""})
        if direct.get("res") is not None:
            findings.append(direct)
            queries.append({"purpose": direct["purpose"], "sql": direct["sql"],
                            "rows": direct["res"]["row_count"]})
            yield {"type": "sql", "purpose": direct["purpose"], "sql": direct["sql"],
                   "rows": direct["res"]["row_count"]}
            evidence = _evidence(findings)

    if not findings:
        text = ((f"That isn't answerable from this data. {blocked_reason}" if blocked_reason else
                 "I couldn't establish anything solid enough to report — every line of enquiry "
                 "either hit a table that can't answer it or returned nothing.")
                + " Try narrowing the question to a specific item, site or period.")
        for ch in text:
            yield {"type": "answer_delta", "text": ch}
        yield {"type": "answer", "text": text, "verified": "flagged", "options": []}
        yield {"type": "done"}
        return

    # Evidence is RESULTS, never the query that produced them. Putting SQL in here is why
    # the brief said "this purchasing data is grouped by year, month, and hospital" and
    # offered "the material ID for Keytruda is 101313" as a driver: given a query, a writer
    # describes the query. The SQL is provenance for the reader (the "N queries run"
    # disclosure), not context for the author.
    evidence = _evidence(findings)

    # ── PHASE 4 · CORROBORATE (different model, different route) ─────────────
    yield {"type": "step", "text": "Re-deriving the key figures independently"}
    corroborations = []
    primary = findings[0]
    got = llm.ask_json(
        cl, role="corroborate",
        system=("You independently CHECK a figure another analyst produced. Compute the same "
                "quantity a DIFFERENT way — a different table, or a sum of parts instead of a "
                "stored total. Do not copy their query.\n\nSCHEMA:\n" + cap_brief),
        user=f"Their query:\n{primary['sql']}\n\nTheir result:\n{_compact(primary['res'], 8)}",
        schema_hint=schemas.CORROBORATE)
    alt_sql = (got.get("sql") or "").strip()
    if alt_sql and not scope.missing_entity_scope(alt_sql, entity_tokens):
        try:
            alt = warehouse.run_sql(alt_sql, row_cap=50)
            if alt.get("row_count"):
                a, b = _num_tokens(_compact(primary["res"], 8)), _num_tokens(_compact(alt, 8))
                agreed = bool(a & b)
                corroborations.append({"agreed": agreed, "sql": alt_sql, "rows": _compact(alt, 8)})
                queries.append({"purpose": "independent re-derivation", "sql": alt_sql,
                                "rows": alt["row_count"]})
                yield {"type": "sql", "purpose": "independent re-derivation", "sql": alt_sql,
                       "rows": alt["row_count"]}
        except Exception:
            pass

    # ── PHASE 5 · CRITIQUE ───────────────────────────────────────────────────
    yield {"type": "step", "text": "Trying to refute the finding"}
    crit = llm.ask_json(
        cl, role="critique",
        system=("You are an adversarial reviewer. Your job is to REFUTE, not to agree. Check: is "
                "every query scoped to what was asked? Is anything double counted? Is a dimension "
                "being used that the table does not carry? Does the conclusion actually follow "
                "from these rows? Default to refuted=false unless you can name a concrete defect."),
        user=f"Question: {query}\n\nEvidence:\n{evidence[:6000]}",
        schema_hint=schemas.CRITIQUE)

    gaps = llm.ask_json(
        cl, role="gaps",
        system="You find what an investigation did not check but should have.\n\nSCHEMA:\n" + cap_brief,
        user=f"Question: {query}\n\nChecked:\n" + "\n".join(f["purpose"] for f in findings),
        schema_hint=schemas.GAPS)
    open_gaps = (gaps.get("gaps") or [])[:3]
    if open_gaps:
        yield {"type": "step", "text": f"Following up {len(open_gaps)} unchecked angle(s)"}
        with ThreadPoolExecutor(max_workers=PARALLEL) as pool:
            known = "\n\n".join(f"{f['purpose']}:\n{_compact(f['res'], 8)}" for f in findings[:4])
            more = [pool.submit(investigate, {"id": f"g{i}", "question": g.get("question", ""),
                                              "why": g.get("why_it_matters", ""),
                                              "table": g.get("table", "")}, known)
                    for i, g in enumerate(open_gaps)]
            for fut in as_completed(more):
                try:
                    out = fut.result()
                except Exception:
                    continue
                if out.get("res") is not None:
                    findings.append(out)
                    queries.append({"purpose": out["purpose"], "sql": out["sql"],
                                    "rows": out["res"]["row_count"]})
                    yield {"type": "sql", "purpose": out["purpose"], "sql": out["sql"],
                           "rows": out["res"]["row_count"]}
        evidence = _evidence(findings)

    # ── PHASE 6 · SYNTHESISE ─────────────────────────────────────────────────
    # Every number that requires arithmetic — direction, % change, peak, trough,
    # concentration, share — is computed HERE, from the rows, before the model writes a
    # word. It gets facts to explain rather than tables to describe, which is the
    # difference between "the trend shows variability across hospitals and months" and
    # "purchasing fell 68% from December to April, peaking in January".
    # When a canonical metric answered the question, derive ONLY from it. Otherwise the
    # headline has two sources competing for it and the model splices them: "101,005 units
    # valued at ₹39.70 L" took the quantity from a SQL re-derivation (which includes stock
    # that expired months ago) and the value from the canonical bands. The other findings
    # stay as EVIDENCE for the drivers; they just stop bidding for the headline figure.
    # A canonical metric owns the headline only when the question is ABOUT the whole
    # business. Asked "what is Vardhman's average lead time" the vendor-lead-time KPI
    # returns company-wide medians with no row for Vardhman — and because canonical
    # findings had been made to own the headline unconditionally, the answer became "there
    # is no lead time figure for Vardhman", for a vendor whose figure is 4.8 days. Same for
    # MSD's ₹47.57 Cr of sales. When a specific entity is named, the KPI is context; the
    # entity-scoped query is the answer.
    entity_specific = bool(entity_tokens)
    # A canonical KPI also loses the headline when it cannot answer at the grain asked for:
    # the units-per-SKU KPI returns CATEGORIES, so "which products move the most units"
    # came back "M070-STATIONARY".
    _want = (requested_grains or [None])[0]
    canonical_findings = [] if entity_specific else [
        f for f in findings if f.get("canonical") and (not _want or _serves_grain(f["res"], _want))]
    # A result that failed the arithmetic check is still shown as evidence, warning attached,
    # because the reader deserves to know a route was tried — but it may not become the
    # figure the brief leads with. ₹649.57 Cr of Bangalore procurement, against ₹478.27 Cr
    # of procurement that exists, was a JOIN fan-out that no wording could have caught.
    sound = [f for f in findings if not _impossible(f)]
    # The grain filter was applied only to CANONICAL findings, so when none of them served
    # the requested level the fallback quietly re-admitted everything — and "which products
    # move the most units" came back "M070-STATIONARY, 7,782,689 units", a category wearing
    # a product's answer. Preferring grain-serving findings here closes that door; the plain
    # `sound` list is still the last resort, because refusing to answer is worse.
    at_grain = [f for f in sound if _serves_grain(f["res"], _want)] if _want else sound
    derived = shapes.derive_all(shape_name, canonical_findings or at_grain or sound or findings)
    if canonical_totals:
        # A KPI's `totals` ARE the headline. Left only in the lesson board they were read
        # past, and the model summed the twelve category rows it could see instead —
        # ₹48.69 Cr of a top-12 breakdown reported as the total stock value of ₹60.47 Cr.
        derived.insert(0, {**canonical_totals,
                           "note": "These are the dashboard's own totals. Use them for the "
                                   "headline figure; do NOT sum the breakdown rows, which "
                                   "are a top-N slice and will undercount."})
    # FORMAT the derived numbers before the writer ever sees them. shapes.py works in raw
    # floats so it stays pure and testable, but handing those straight over produced
    # "sales started at 80,663,366" — a rupee figure written as a bare integer, in the very
    # block that exists to stop the model doing its own arithmetic. Everything it quotes
    # must already be in the form it should quote.
    derived_block = json.dumps(_format_derived(derived), default=str)[:3500] if derived else ""
    # the finding the headline rests on = the first one the shape could actually derive from
    primary_finding = next(
        (f for f in findings if any(d.get("from") == (f.get("purpose") or "")[:80] for d in derived)),
        findings[0] if findings else None)
    if derived:
        yield {"type": "step", "text": "Computing the movement"}
    yield {"type": "step", "text": "Writing the brief"}
    corr_note = ("An independent re-derivation AGREED with the headline figure."
                 if any(c["agreed"] for c in corroborations)
                 else ("An independent re-derivation DISAGREED — say so plainly and do not "
                       "present the figure as settled." if corroborations else
                       "No independent route existed to re-derive the headline figure."))
    crit_note = (f"A reviewer raised: {crit.get('problem')}" if crit.get("refuted") else
                 "A reviewer found no defect that changes the conclusion.")

    disclosure = _measure_disclosure(query, findings, primary_finding)
    if disclosure:
        yield {"type": "answer_delta", "text": disclosure}
    prose = disclosure
    pending = ""      # hold back a partial word so "plant" is never half-emitted
    for tok in llm.stream_text(
        cl, role="synthesise",
        system=("You are a hospital supply-chain analyst writing a short brief for an executive.\n"
                f"THIS ANSWER MUST: {shape['answer_must']}.\n"
                + (f"The question asks for this broken down by {requested_grains[0].upper()} — "
                   f"answer at that level. A category is not a product and a vendor is not a "
                   f"manufacturer; answering one level up is a different question.\n"
                   if requested_grains else "")
                + (_source_words(findings) + "\n" if _source_words(findings) else "")
                + "STRUCTURE: lead with the answer and its number; then WHAT DRIVES IT, ranked; then "
                "WHAT YOU RULED OUT; then the limits.\n"
                "NEVER state a percentage or a share unless that exact percentage appears in "
                "the DERIVED FACTS. Do not compute one from the rows: a query returns the rows it "
                "was asked for, not the whole company, so dividing a number by the rows beside it "
                "produces confident nonsense — '100% of procurement value from one vendor' when the "
                "real figure is 45.8%. If the derived facts carry no share, give the values alone.\n"
                "NEVER describe the data or the query. 'The data is grouped by year, month and "
                "hospital', 'the evidence provides', 'the material ID is 101313' — these are notes "
                "about plumbing, not findings, and they are the difference between an analyst and "
                "a search result. Every line must say something about the BUSINESS.\n"
                "WHAT YOU RULED OUT means a competing explanation you tested and rejected, with "
                "the figure that rejected it. It does not mean a limitation of the dataset — that "
                "belongs under limits.\n"
                "RULES:\n"
                "- Quote figures EXACTLY as formatted in the evidence. Never re-scale or recompute.\n"
                "- Cite the evidence block you took each figure from, as [id].\n"
                "- NAME THE MEASURE YOU ACTUALLY USED. If the question asked for one thing and the "
                "evidence only supports a related one — asked for SALES, evidence is PURCHASING; "
                "asked for revenue, evidence is quantity — say so in the FIRST sentence, plainly: "
                "'There is no monthly sales figure for this item; what follows is monthly "
                "PURCHASING.' Never label a substitute with the name of the thing that was asked "
                "for. A brief headed 'Sales trend' that is actually procurement is a wrong answer, "
                "however good its numbers.\n"
                "- Say 'hospital', never 'plant'. No trailing rhetorical question.\n"
                "- If a line of enquiry was ruled out, say so — that is half the value.\n"
                "- Do not claim anything the evidence does not show."),
        user=(f"Question: {query}\n\n"
              + (f"DERIVED FACTS (computed from the rows — exact, use these for every claim "
                 f"about direction, change, peak, trough, share or concentration; do NOT "
                 f"recompute them):\n{derived_block}\n\n" if derived_block else "")
              + f"EVIDENCE (results only):\n{evidence[:8000]}\n\n"
              f"CORROBORATION: {corr_note}\nREVIEW: {crit_note}")):
        pending += tok
        cut = max(pending.rfind(" "), pending.rfind("\n"))
        if cut >= 0:
            # chunks are cut at a space, so a rupee figure arrives whole and can be rescaled
            ready, pending = _rupees_in_scale(_hospitalise(pending[:cut + 1])), pending[cut + 1:]
            prose += ready
            yield {"type": "answer_delta", "text": ready}
    if pending:
        ready = _rupees_in_scale(_hospitalise(pending))
        prose += ready
        yield {"type": "answer_delta", "text": ready}

    # ── presentation ──────────────────────────────────────────────────────────
    # Which finding to SHOW is a real decision, and picking "the one with the most rows"
    # got it badly wrong: a KEYTRUDA sales question displayed the stock-change table —
    # the very evidence the brief had just explicitly RULED OUT — because it happened to
    # have thirteen rows. The evidence under an answer must be the evidence FOR it.
    #
    # So: prefer findings the synthesis actually leaned on (its [id] citations), then
    # plan order, and require a real measure — not an identifier column.
    cited = {m.group(1) for m in re.finditer(r"\[([A-Za-z]?\d+)\]", prose)}

    def _showable(f):
        res = f["res"]
        rows = res.get("rows") or []
        if len(rows) < 2:
            return None
        cols = res["columns"]
        cat = next((c for c in cols if _kind(c, rows) in ("text", "id")), None)
        val = next((c for c in cols if _kind(c, rows) in ("inr", "pct", "days", "num")), None)
        return (cat, val) if (cat and val) else None

    ranked = sorted(
        [f for f in findings if _showable(f)],
        key=lambda f: (0 if f["sub"].get("id") in cited else 1, findings.index(f)))

    # Prefer a finding that actually provides the grain the question asked for.
    want = (requested_grains or [None])[0]
    if want:
        matching = [f for f in ranked if _serves_grain(f["res"], want)]
        if matching:
            ranked = matching + [f for f in ranked if f not in matching]

    if ranked:
        best = ranked[0]
        res = _order_rows(best["res"])
        cat, val = _showable(best)
        # The chart's title is a title, not the analyst's note-to-self. `purpose` reads
        # "To calculate the stock change for 'keytruda 100mg INJ vial'" — an intention,
        # which is what was being printed above the chart.
        title = _chart_title(best, val)

        # CHART THE ANSWER, NOT THE QUERY.
        #
        # Two problems with always drawing a bar of the raw rows. A time series is a LINE —
        # drawn as bars it reads as a ranking, and sorted by value it stops being a trend at
        # all. And the raw result is often a grid (hospital x month) while the prose
        # describes the series derived FROM it, so the chart and the words disagreed: a
        # "trend" answer sat above a chart of hospitals.
        #
        # The derived series is what the answer is about, so that is what gets drawn, and
        # the shape decides how.
        series = next((d for d in derived if d.get("series")), None)
        if shape_name == "trend" and series:
            rows_for_chart = [{"period": p["period"], series["measure"]: p["value"]}
                              for p in series["series"]]
            spec = {"type": "line", "x": "period", "y": [series["measure"]],
                    "title": title, "value_format": _kind(series["measure"], res["rows"])}
        else:
            rows_for_chart = res["rows"][:14]
            spec = {"type": "bar", "x": cat, "y": [val], "title": title,
                    "value_format": _kind(val, res["rows"]),
                    # a ranking reads top-down; a banded exposure reads in band order
                    "horizontal": cat.lower() not in ("month", "month_name", "period", "year",
                                                      "bucket", "band")}
        try:
            fig = charts.build(rows_for_chart, spec)
            if fig:
                yield {"type": "chart", "plotly": fig}
        except Exception:
            pass

        yield {"type": "table", "table": _table_payload(res, _hospitalise(title)), "note": ""}

    verified = "flagged" if crit.get("refuted") else (
        "ok" if any(c["agreed"] for c in corroborations) else None)
    yield {"type": "answer", "text": prose, "verified": verified, "options": [],
           "scope": f"deep · {len(findings)} lines of enquiry, {len(queries)} queries"}
    yield {"type": "done"}
