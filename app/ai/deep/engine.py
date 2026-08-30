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

from app.ai import charts, scope, warehouse
from app.ai.deep import capability, llm, schemas, tools

MAX_SUBQUESTIONS = 6
MAX_ROUNDS = 2          # investigate → critique → (one more investigate) → stop
PARALLEL = 4            # Azure rate limits bite harder than the CPU does
ROW_CAP = 200
MAX_TOOL_STEPS = 8      # a worker that hasn't found it in eight looks isn't going to


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
    if any(k in c for k in ("revenue", "cost", "margin", "value", "price", "spend", "amount")):
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
    """A short label. `purpose` is the analyst's intent sentence, not a heading."""
    q = (finding["sub"].get("question") or "").strip().rstrip("?")
    if 3 < len(q) <= 70:
        return q[0].upper() + q[1:]
    m = measure.replace("_", " ").strip()
    return (m[0].upper() + m[1:]) if m else "Result"


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
    for key in ("rows", "items", "bands", "breakdown", "series"):
        v = data.get(key)
        if isinstance(v, list) and v and isinstance(v[0], dict):
            cols = list(v[0].keys())
            return {"columns": cols, "rows": v[:200], "row_count": len(v)}
    totals = data.get("totals")
    if isinstance(totals, dict) and totals:
        return {"columns": list(totals.keys()), "rows": [totals], "row_count": 1}
    return {"columns": ["metric"], "rows": [{"metric": str(data)[:200]}], "row_count": 1}


def _table_payload(res: dict, title: str) -> dict:
    cols = res.get("columns") or []
    rows = res.get("rows") or []
    kinds = {c: _kind(c, rows) for c in cols}
    return {"title": title,
            "columns": [{"key": c, "label": c, "kind": kinds[c]} for c in cols],
            "rows": rows[:50]}


def _compact(res: dict, limit: int = 25) -> str:
    """A result as the model should see it: already formatted, so it quotes rather than
    converts. Prose that converts raw rupees itself is where the 10x errors came from."""
    cols = res.get("columns") or []
    rows = res.get("rows") or []
    kinds = {c: _kind(c, rows) for c in cols}
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
            footprint = json.dumps(fp)[:1800]
            yield {"type": "sql", "purpose": f"footprint of “{name}”",
                   "sql": f"-- lookup_item('{name}')", "rows": fp.get("match_count", 0)}
        except Exception:
            pass

    frame = llm.ask_json(
        cl, role="frame",
        system=("You frame analytics questions against a fixed warehouse. You are given the REAL "
                "schema — never assume a table or column that is not listed.\n\n" + capability.brief()),
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
    yield {"type": "step", "text": "Planning the investigation"}
    plan = llm.ask_json(
        cl, role="plan",
        system=("You plan an analytics investigation. Break the question into sub-questions that "
                "TOGETHER explain it — not restatements of it. A good plan tests competing "
                "explanations (price vs volume vs mix vs site vs vendor) so the answer can say "
                "what it is AND what it is not.\n\nSCHEMA:\n" + cap_brief),
        user=f"Question: {query}\n\nFraming: {json.dumps(frame)[:600]}",
        schema_hint=schemas.PLAN)
    subs = (plan.get("sub_questions") or [])[:MAX_SUBQUESTIONS]
    if not subs:
        subs = [{"id": "q1", "question": query, "why": "direct", "table": ""}]
    yield {"type": "step", "text": f"Investigating {len(subs)} lines of enquiry"}

    # ── PHASE 3 · INVESTIGATE (parallel) ─────────────────────────────────────
    # A LESSON BOARD shared by every worker.
    #
    # An analyst who discovers that sales_by_material has no month column does not
    # rediscover it for the next question — and neither should six agents running in
    # parallel. Each worker writes what it learned about the SHAPE of the data here, and
    # every later worker reads it. One worker's dead end becomes everyone's starting
    # knowledge, which is the cheapest accuracy win in the engine.
    lessons: list[str] = []
    lesson_lock = Lock()

    # SEED the board with what the schema already proves, before any worker spends a step
    # discovering it. Four workers independently hunted for a monthly sales grain and
    # tunnelled into forecast_sales; the fact that no sales table carries time is
    # derivable from information_schema in one pass and is exactly the knowledge they
    # needed. Deterministic facts should never cost a model call.
    try:
        _fam = frame.get("entity_family") if isinstance(frame, dict) else None
        _prof = capability.profile(_fam if _fam in capability.ENTITY_FAMILIES else None)
        _timed = sorted(e["table"] for e in _prof["tables_with_time"])
        _sales_no_time = sorted(e["table"] for e in _prof["tables"]
                                if e["table"].startswith("sales") and not e["time"])
        if _timed:
            lessons.append("Tables carrying BOTH this entity and a time column: " + ", ".join(_timed[:14]))
        if _sales_no_time:
            lessons.append("These sales tables have NO time/month column, so no trend can come "
                           "from them: " + ", ".join(_sales_no_time))
        lessons.append("There is no material-by-month SALES grain in this warehouse. Monthly "
                       "figures for an item exist only for PURCHASING (fact_po, fact_grn, "
                       "mart_procurement, kpi_monthly_purchase_value) and CONSUMPTION "
                       "(fact_consumption, kpi_units_consumed). Use one of those and say which.")
    except Exception:
        pass

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
        ctx = {"entity_tokens": entity_tokens}
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
                        last_result = tools.run_query(sql, entity_tokens)
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
            return {"sub": sub, "sql": last_sql, "res": last_result["_full"], "purpose": sub.get("question")}
        return {"sub": sub, "skipped": "ran out of steps without a usable result"}


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
        facts = "\n\n".join(f"{f['purpose']}:\n{_compact(f['res'], 10)}" for f in findings)
        yield {"type": "step", "text": f"Following the {len(wave2)} dependent question(s)"}
        for out in run_wave(wave2, facts):
            if out.get("res") is not None:
                findings.append(out)
                queries.append({"purpose": out["purpose"], "sql": out["sql"], "rows": out["res"]["row_count"]})
                yield {"type": "sql", "purpose": out["purpose"], "sql": out["sql"], "rows": out["res"]["row_count"]}
            else:
                yield {"type": "step", "text": f"Ruled out: {out['sub'].get('question','')[:60]}"}

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

    evidence = "\n\n".join(
        f"[{f['sub'].get('id','?')}] {f['purpose']}\nSQL: {f['sql']}\n{_compact(f['res'])}"
        for f in findings)

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
        evidence = "\n\n".join(
            f"[{f['sub'].get('id','?')}] {f['purpose']}\nSQL: {f['sql']}\n{_compact(f['res'])}"
            for f in findings)

    # ── PHASE 6 · SYNTHESISE ─────────────────────────────────────────────────
    yield {"type": "step", "text": "Writing the brief"}
    corr_note = ("An independent re-derivation AGREED with the headline figure."
                 if any(c["agreed"] for c in corroborations)
                 else ("An independent re-derivation DISAGREED — say so plainly and do not "
                       "present the figure as settled." if corroborations else
                       "No independent route existed to re-derive the headline figure."))
    crit_note = (f"A reviewer raised: {crit.get('problem')}" if crit.get("refuted") else
                 "A reviewer found no defect that changes the conclusion.")

    prose = ""
    pending = ""      # hold back a partial word so "plant" is never half-emitted
    for tok in llm.stream_text(
        cl, role="synthesise",
        system=("You are a hospital supply-chain analyst writing a short brief for an executive.\n"
                "STRUCTURE: lead with the answer and its number; then WHAT DRIVES IT, ranked; then "
                "WHAT YOU RULED OUT; then the limits.\n"
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
        user=(f"Question: {query}\n\nEVIDENCE:\n{evidence[:9000]}\n\n"
              f"CORROBORATION: {corr_note}\nREVIEW: {crit_note}")):
        pending += tok
        cut = max(pending.rfind(" "), pending.rfind("\n"))
        if cut >= 0:
            ready, pending = _hospitalise(pending[:cut + 1]), pending[cut + 1:]
            prose += ready
            yield {"type": "answer_delta", "text": ready}
    if pending:
        ready = _hospitalise(pending)
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

    if ranked:
        best = ranked[0]
        res = _order_rows(best["res"])
        cat, val = _showable(best)
        # The chart's title is a title, not the analyst's note-to-self. `purpose` reads
        # "To calculate the stock change for 'keytruda 100mg INJ vial'" — an intention,
        # which is what was being printed above the chart.
        title = _chart_title(best, val)
        spec = {"type": "bar", "x": cat, "y": [val], "title": title,
                "value_format": _kind(val, res["rows"]),
                # a time series reads left-to-right, a ranking reads top-down
                "horizontal": cat.lower() not in ("month", "month_name", "period", "year")}
        try:
            fig = charts.build(res["rows"][:14], spec)
            if fig:
                yield {"type": "chart", "plotly": fig}
        except Exception:
            pass
        yield {"type": "table", "table": _table_payload(res, title), "note": ""}

    verified = "flagged" if crit.get("refuted") else (
        "ok" if any(c["agreed"] for c in corroborations) else None)
    yield {"type": "answer", "text": prose, "verified": verified, "options": [],
           "scope": f"deep · {len(findings)} lines of enquiry, {len(queries)} queries"}
    yield {"type": "done"}
