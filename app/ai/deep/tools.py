"""The toolbox an analyst actually uses — as callable tools, not a static schema dump.

WHY
---
Asked for a sales trend, the model wrote `GROUP BY month` against `sales_by_material`
three times, on three tables whose column lists were in its own prompt, none of which has
a month column. It was not being stupid. It was GUESSING, because it had been handed a
description of the warehouse and then asked to produce a final query in one shot, blind.

A human does not do that. They open the table, look at five rows, notice there is no month,
go looking for one with find_columns('period'), find `sales_by_material_month`, and answer
from that — and only if no sales table had a month would they fall back to monthly
PURCHASING and say so. That is eight actions with LOOKING between each one, and the
intelligence is in the looking.

So: give it eyes. Every one of these mirrors something an analyst does before committing to
a query — list, describe, peek, find the grain, find where a value lives, profile a column,
try it, check it. The loop that drives them lives in engine.py.

Read-only throughout: every query goes through warehouse.validate() and the scope guards.
"""
from __future__ import annotations

import copy
import json
import re

from app.ai import warehouse
from app.ai.deep import capability

MAX_SAMPLE = 8
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _safe(name: str) -> bool:
    return bool(_IDENT.fullmatch(name or ""))


# ── the tools ────────────────────────────────────────────────────────────────
def list_tables(contains: str = "") -> dict:
    """What is in this warehouse (optionally filtered), with row counts."""
    sch = capability.schema()
    names = [t for t in sorted(sch) if not contains or contains.lower() in t.lower()]
    out = []
    for t in names[:60]:
        try:
            n = warehouse.run_sql(f"SELECT COUNT(*) AS n FROM {t}", row_cap=1)["rows"][0]["n"]
        except Exception:
            n = None
        out.append({"table": t, "rows": n, "columns": len(sch.get(t, []))})
    return {"tables": out, "total": len(names)}


def describe_table(table: str) -> dict:
    """Columns and types. The thing that should have been checked before GROUP BY month."""
    if not _safe(table):
        return {"error": "bad table name"}
    try:
        cols = warehouse.run_sql(
            "SELECT column_name, data_type FROM information_schema.columns "
            f"WHERE table_name = '{table}' ORDER BY ordinal_position", row_cap=200)["rows"]
        n = warehouse.run_sql(f"SELECT COUNT(*) AS n FROM {table}", row_cap=1)["rows"][0]["n"]
    except Exception as e:
        return {"error": str(e)[:200]}
    if not cols:
        return {"error": f"no table named '{table}'"}
    return {"table": table, "rows": n,
            "columns": [{"name": c["column_name"], "type": c["data_type"]} for c in cols]}


def sample_rows(table: str, n: int = 5) -> dict:
    """Actual rows. Column names tell you a column exists; rows tell you what is IN it."""
    if not _safe(table):
        return {"error": "bad table name"}
    try:
        r = warehouse.run_sql(f"SELECT * FROM {table} LIMIT {min(int(n), MAX_SAMPLE)}", row_cap=MAX_SAMPLE)
        return {"table": table, "columns": r["columns"], "rows": r["rows"]}
    except Exception as e:
        return {"error": str(e)[:200]}


def find_columns(name_like: str) -> dict:
    """Which tables have a column like this — i.e. WHICH TABLES CARRY THIS GRAIN.

    "Who has a month?" is the question that would have prevented the whole KEYTRUDA
    failure, and it is one call.
    """
    q = (name_like or "").lower().strip()
    if not q:
        return {"error": "name_like is required"}
    hits = [{"table": t, "column": c}
            for t, cols in capability.schema().items() for c in cols if q in c.lower()]
    return {"looking_for": name_like, "found": hits[:60], "count": len(hits)}


def find_value(value: str) -> dict:
    """Which column actually CONTAINS this value.

    'Bangalore' is not a column, it is a value inside dim_plant.plant_name. Without this,
    the model concluded the warehouse had no location data at all.
    """
    v = (value or "").strip()
    if len(v) < 2:
        return {"error": "value too short"}
    lit = v.replace("'", "''")
    hits = []
    for table, col in capability.entity_columns_for_search():
        try:
            n = warehouse.run_sql(
                f"SELECT COUNT(*) AS n FROM {table} WHERE upper(CAST({col} AS VARCHAR)) "
                f"LIKE '%' || upper('{lit}') || '%'", row_cap=1, timeout_s=3.0)["rows"][0]["n"]
        except Exception:
            continue
        if n:
            hits.append({"table": table, "column": col, "rows": int(n)})
        if len(hits) >= 8:
            break
    hits.sort(key=lambda h: -h["rows"])
    return {"value": v, "found_in": hits, "note": "" if hits else "not found in any entity column"}


def profile_column(table: str, column: str) -> dict:
    """Distinct count, range, nulls, and the commonest values — is this usable as a filter?"""
    if not (_safe(table) and _safe(column)):
        return {"error": "bad identifier"}
    try:
        base = warehouse.run_sql(
            f"SELECT COUNT(*) AS n, COUNT({column}) AS non_null, "
            f"COUNT(DISTINCT {column}) AS distinct_n FROM {table}", row_cap=1)["rows"][0]
        top = warehouse.run_sql(
            f"SELECT CAST({column} AS VARCHAR) AS value, COUNT(*) AS n FROM {table} "
            f"WHERE {column} IS NOT NULL GROUP BY 1 ORDER BY 2 DESC LIMIT 8", row_cap=8)["rows"]
    except Exception as e:
        return {"error": str(e)[:200]}
    return {"table": table, "column": column, **base, "most_common": top}


_COUNTING_WORDS = re.compile(r"\bhow many\b|\bnumber of\b|\bcount of\b", re.I)
_ITEM_WORDS = re.compile(r"\bskus?\b|\bitems?\b|\bproducts?\b|\bdrugs?\b|\bmaterials?\b|\bmedicines?\b", re.I)
_COUNT_STAR = re.compile(r"count\s*\(\s*\*\s*\)", re.I)


def _miscounts_items(sql: str, question: str) -> str | None:
    """COUNT(*) is not an item count on a table keyed by item PER SITE.

    kpi_non_moving holds 16,872 rows and 10,501 distinct materials — one row per material
    per hospital. Asked how many SKUs are non-moving, the answer was 16,872. The grain is
    stated in the lesson board and was read past, which is what a stated preference is
    worth; this makes it a property of the query instead.
    """
    q = question or ""
    if not (_COUNTING_WORDS.search(q) and _ITEM_WORDS.search(q)):
        return None
    if not _COUNT_STAR.search(sql or ""):
        return None
    from app.ai.deep import capability
    low = (sql or "").lower()
    for note in capability.grain_notes():
        table = note.split(":", 1)[0]
        if re.search(rf"\b{re.escape(table)}\b", low):
            return (f"COUNT(*) on {table} counts rows, and {note.split('—', 1)[-1].strip()} "
                    f"The question asks how many ITEMS, so use COUNT(DISTINCT material).")
    return None


def run_query(sql: str, entity_tokens: list[str] | None = None, question: str = "",
              conformance: bool = True) -> dict:
    """Run it and SEE the result — including the error, which is information, not a dead end.

    Two KINDS of check live here and they are not interchangeable:

      INTEGRITY   — is this a valid computation over this warehouse? Disjoint joins,
                    non-additive sums, all-NULL results, a part larger than the whole.
                    True of any query, whatever it was written for.
      CONFORMANCE — does this SQL answer the USER'S question? Item miscounts, the shape the
                    question demands, a placeholder winning the ranking that was asked for.

    A corroborating query is deliberately NOT the user's query — it recomputes one figure a
    different way, so it fails conformance by design while being perfectly valid. Applying
    both sets to it rejected most corroborations (39% of runs down to 14%) and took the
    engine's second opinion away. `conformance=False` runs integrity only.
    """
    from app.ai import scope
    miscount = _miscounts_items(sql, question) if conformance else None
    if miscount:
        return {"error": miscount}
    off = scope.missing_entity_scope(sql, entity_tokens or [])
    if off:
        return {"error": off}
    # A city filter on a sales table cannot be satisfied — the code systems are disjoint.
    # Saying so in the prompt did not stop it; failing the query does.
    from app.ai.deep import sanity as _sanity
    unreachable = _sanity.city_on_unreachable_table(sql)
    if unreachable:
        return {"error": unreachable}
    try:
        r = warehouse.run_sql(sql, row_cap=200)
    except Exception as e:
        return {"error": str(e)[:300]}
    # `ORDER BY spend DESC LIMIT 1` returning "Uncategorized" cannot be repaired by
    # reordering — there is nothing left to promote. Send it back to be re-queried.
    unnamed = _sanity.placeholder_won_a_ranking(sql, r, question) if conformance else None
    if unnamed:
        return {"error": unnamed}
    # An empty fact_consumption result is the expected result for a patient-billed item,
    # not evidence that the item has no consumption.
    billed = scope.billed_not_internal(sql, r)
    if billed:
        return {"error": billed}
    # Partial coverage is not an error — the number is right for what it covers. It is
    # attached to the RESULT so the answer can disclose which half of the estate it
    # describes. fact_inventory holds 26 of 53 hospitals; nothing used to say so.
    from app.ai import coverage as _coverage
    note = _coverage.disclosure(sql)
    if note:
        r["coverage_note"] = note
    # What the QUESTION demands of the SQL. "What share of revenue is non-formulary" was
    # answered "₹29.06 Cr" — the right numerator, never divided, so no share was given.
    from app.ai.deep import constraints as _constraints
    unmet = _constraints.check(question, sql) if conformance else None
    if unmet:
        return {"error": unmet}
    # Checks DERIVED from the ontology rather than typed: non-additive measures, joins
    # between columns measured as sharing no values, and columns whose name means different
    # things in different tables. Rename a column and these follow, because the ontology is
    # rebuilt from the warehouse.
    from app.ai.deep import ontology_checks as _onto
    violation = _onto.check(sql)
    if violation:
        return {"error": violation}
    # SUM() over zero matching rows returns ONE row containing NULL, which looks like a
    # result and is not: "SELECT SUM(revenue) FROM sales_by_manufacturer WHERE manufacturer
    # = 'MSD'" matched nothing (the value is stored 'Msd') and the answer became "there is
    # no sales figure for MSD" for a manufacturer with ₹47.57 Cr.
    rows = r.get("rows") or []
    if r.get("row_count") == 1 and rows and all(v is None for v in rows[0].values()):
        hint = scope.explain_zero_rows(sql)
        return {"row_count": 0, "columns": r["columns"],
                "note": (hint or "") + " Every value came back NULL, which means the filter "
                        "matched no rows — check the spelling and CASE of the value you "
                        "filtered on (use upper()/ILIKE), not the aggregate."}
    if not r.get("row_count"):
        hint = scope.explain_zero_rows(sql)
        return {"row_count": 0, "columns": r["columns"],
                "note": hint or "0 rows — check the filter values with find_value or profile_column"}
    # THE MAGNITUDE CHECK IS NOT APPLIED HERE, AND THAT IS A MEASURED DECISION.
    #
    # app/ai/deep/magnitude.py catches a real error: a join that multiplies rows before a SUM
    # shipped "procurement is dominated by Consumables, Rs 19,825 Cr" against a warehouse
    # holding Rs 478 Cr in total. It fires on that exact shape, it has no false positives on
    # 60 single-table sums, and it reads the ARITHMETIC rather than the query, which is a
    # class no other check here can see.
    #
    # Wired in as a rejection it cost eight points. On the nine cases that got worse:
    # 13/27 with it, 21/27 with it ablated, same build otherwise. And
    # l2-procurement-by-category — the case it was BUILT for — is 0/3 either way, so it never
    # fixed its own target. What it does instead is consume the tool budget: the model retries
    # against a bound it cannot satisfy, and the turn ends in "I did not manage to run a
    # single query" where it used to end in a wrong number.
    #
    # A guard that turns a wrong answer into no answer is not obviously an improvement, and
    # this one turns CORRECT answers into no answer too. The module and its tests stay because
    # the invariant is right and a cheaper application of it may not be; the rejection does
    # not, because the measurement says so.
    return {"row_count": r["row_count"], "columns": r["columns"], "rows": r["rows"][:12],
            "truncated": r["row_count"] > 12, "_full": r}


def _city_plant_codes(question: str) -> list[str]:
    """The plant codes a city in the question resolves to, when there is more than one."""
    try:
        from app.ai import resolve as _resolve
        r = _resolve.resolve(question or "")
    except Exception:
        return []
    out: list[str] = []
    for c in r["cities"]:
        sites = r["city_hospitals"].get(c) or []
        if len(sites) > 1:
            out += [s.split(" ")[0] for s in sites]
    return list(dict.fromkeys(out))


# A percentage is a ratio of sums. Network formulary compliance is sum(formulary cost) over
# sum(drug cost) across the sites, never the sum or the average of each site's percentage —
# and the advice below used to say "SUM them" for every KPI, fill rate and compliance alike.
# The wording avoids the words sum, add and average altogether, so no part of it can be
# read as an instruction to combine the figures. The registry marks days-on-hand, lead time
# and fill rate as ratios too, so it does not say "percentage" alone.
_RATIO_ADVICE = ("This is a RATIO (a percentage, rate, score or days figure), and figures "
                 "from different sites cannot be combined into one by arithmetic. Report each "
                 "site's figure separately.")


def _ratio_advice(key: str, codes: list[str], can_view: bool = False) -> str:
    """The ratio advice, offering only the routes this guard will then let through.

    Fast mode has no per-site fan-out of its own. For "what is the fill rate across Bangalore
    hospitals?" the advice used to be "call get_kpi once per code, or once with no plant and
    view='hospitals'" — and this same guard refused both: each single Bangalore plant is
    UNDER-SCOPED, and the guard never saw `view`, so the no-plant call was UNSCOPED again.
    The Dec–May KPIs do not take `view` at all. The model looped on refusals until it ran
    out of steps. So each route is offered only where it works:
      - `codes` (a city): a Jun–Aug report pools the codes itself; any other KPI is one SQL
        statement over the codes, because get_kpi takes one plant and one is never the city.
      - no codes (named hospitals, no city): one call per code passes, so it is offered.
      - view="hospitals" only for a KPI that takes `view`, and only when the caller hands
        its arguments to the guard (`can_view`), so the call it suggests is not refused.
    """
    view = can_view and _kpi_takes(key, "view")
    view_text = ("call once with no plant and view=\"hospitals\" and quote the rows for "
                 "these sites")
    if codes and _is_report_key(key):
        # the Jun–Aug reports take several codes and compute the pooled ratio themselves
        return (_RATIO_ADVICE + f" For one figure covering all of them, call once with "
                f"plant=\"{','.join(codes)}\" — the report pools them correctly."
                + (f" Or {view_text}." if view else ""))
    if codes:
        return (_RATIO_ADVICE + f" get_kpi takes one plant at a time and one site is not the "
                f"city, so write one SQL statement filtered to plant IN ({', '.join(codes)}) "
                f"that computes the ratio for each plant from its own numerator and "
                f"denominator, and quote one line per site.")
    return (_RATIO_ADVICE + " Call get_kpi once per hospital code"
            + (f", or {view_text}" if view else "")
            + ", or write one SQL statement filtered to those plant codes that computes the "
              "ratio for each plant.")


def _kpi_takes(key: str, arg: str) -> bool:
    """Whether this registry entry accepts an extra argument such as `material`."""
    try:
        from app.ai import kpi_registry
        return arg in ((kpi_registry.KPI_REGISTRY.get(key) or {}).get("args") or ())
    except Exception:
        return False


def _is_report_key(key: str) -> bool:
    """A Jun–Aug report (it needs the formulary or spend data), not a Dec–May dashboard KPI."""
    try:
        from app.ai import kpi_registry
        return bool((kpi_registry.KPI_REGISTRY.get(key) or {}).get("needs"))
    except Exception:
        return False


# Words a question uses to ASK that also happen to begin a product name in the item master.
# The resolver forms a "family" from any rare token, and the first version of the product
# check below counted every family of five or fewer SKUs as one product. On the real
# warehouse it refused the canonical KPI for 27 ordinary company-wide questions from the evals
# and the chat starters: "what should we order FIRST" was about FIRST AID BOX-GEN, "the
# consumption SPLIT" about a SPLIT AC, "how CONCENTRATED is our procurement" about a
# haemodialysis solution — and the refusal then sent the model off to query that item's rows.
# The brand-position rule in _product_family drops most of them (TARGET, BAND, AGAINST,
# NETWORK sit mid-name); this list catches the ones that lead a name. It holds
# business-question vocabulary, not a dictionary. A brand that is also a listed word is not
# caught, which only costs the behaviour from before this check existed; an ordinary word
# that is missing still refuses the KPI, so add it here when one turns up.
_ORDINARY_WORDS = frozenset({
    # order, position and comparison
    "first", "second", "third", "start", "end", "begin", "initial", "final", "early", "late",
    "peak", "bottom", "rank", "ranking", "equal", "multiple", "various", "general", "overall",
    "single", "dual", "double", "triple", "three", "four", "five", "half", "whole", "extreme",
    "superior", "super", "positive", "negative", "neutral", "external", "internal", "national",
    "universal", "natural", "plain", "true", "false", "empty", "solid", "hard", "easy", "light",
    "live", "mobile", "immediate", "interim", "provisional", "preventive", "supplementary",
    # the measures, cuts and shapes analysts ask about
    "split", "share", "mix", "band", "group", "class", "cluster", "segment", "tier", "target",
    "budget", "plan", "actual", "forecast", "variance", "gap", "risk", "loss", "profit",
    "growth", "volume", "value", "cost", "price", "rate", "ratio", "margin", "revenue",
    "order", "supply", "demand", "stock", "excess", "surplus", "shortage", "free", "cash",
    "credit", "net", "gross", "factor", "measure", "performance", "signal", "trace", "pause",
    "block", "back", "ground", "mass", "inflation", "tender", "account", "network", "against",
    "match", "tracking", "patients", "therapeutic", "injection", "emergency", "referral",
    "incident", "information", "resource", "reserve", "custom", "specialty", "subject",
    "option", "display", "register", "file", "text", "sign", "memory", "mind", "welcome",
    "swift", "pathway", "junction", "alliance", "champion", "masters", "contrast", "culture",
    "concentrate", "correction", "extension", "documentation", "justification", "suggestion",
    "changeover", "overtime", "extraction", "examination", "exercise", "tension", "traffic",
    "vertical", "linear", "circular", "lateral", "periodic", "functional", "visual",
    "complimentary", "compensatory", "subjective", "educational",
})
# Inflected English — "concentrated", "tracking", "weekly" — is almost never a brand's first
# word, and listing every such form is the losing game the resolver's stoplists already play.
_INFLECTED = re.compile(r"^[a-z]{3,}(?:ed|ing|ly)$")
_TOKEN = re.compile(r"[A-Za-z0-9]+")


def _ordinary_word(tok: str) -> bool:
    t = (tok or "").lower()
    return (t in _ORDINARY_WORDS or (t.endswith("s") and t[:-1] in _ORDINARY_WORDS)
            or bool(_INFLECTED.match(t)))


def _quoted(tok: str, question: str) -> bool:
    """The user put the word inside double quotes, which is how a name is marked.

    Double quotes only: an apostrophe ("what's our first order's value") would turn half a
    sentence into a false quotation.
    """
    spans = re.findall(r"[\"“]([^\"“”]+)[\"”]", question or "")
    pat = re.compile(rf"(?<![A-Za-z0-9]){re.escape(tok)}(?![A-Za-z0-9])", re.I)
    return any(pat.search(s) for s in spans)


def _leading_word(value: str) -> str:
    """The first identifying word of an item name: KEYTRUDA in "KEYTRUDA 100MG INJ VIAL"."""
    try:
        from app.ai.resolve import _NOISE_TOKENS as noise
    except Exception:
        noise = frozenset()
    for t in _TOKEN.findall(value or ""):
        tl = t.lower()
        if len(tl) >= 3 and not tl.isdigit() and tl not in noise:
            return tl
    return ""


def _product_family(f: dict, question: str) -> bool:
    """Whether a resolver FAMILY is the user naming a product, not a word that happens to
    appear inside a few product names.

    A brand leads its item name — KEYTRUDA 100MG INJ VIAL, OPDYTA 240MG INJ VIAL — while a
    coincidental word sits anywhere in it: CAR-T DUAL TARGET, MULTI BAND LIGATOR, D-LINK
    8 PORT NETWORK SWITCH. So the token must lead every name in the family and must not be an
    ordinary question word, unless the user quoted it.
    """
    tok = (f.get("token") or "").lower()
    ex = f.get("examples") or []
    # a family of a handful of SKUs is still one product (KEYTRUDA is a family of one); a
    # family of seventy sutures is a category
    if f.get("kind") not in ("material", "generic") or f.get("n", 99) > 5 or not ex or not tok:
        return False
    if _quoted(tok, question):
        return True
    return not _ordinary_word(tok) and all(_leading_word(v) == tok for v in ex)


def _named_products(r: dict, question: str) -> list[str]:
    """The products the question names: exact item or molecule entities, and brand families."""
    q = question or ""
    out: list[str] = []
    for e in r.get("entities") or []:
        if e.get("kind") not in ("material", "generic"):
            continue
        text = str(e.get("text") or "")
        if e.get("column") == "material":
            # a code counts only as a whole word in the question: "100" inside "100MG", or a
            # "180" in "180 days", names nothing
            if not re.search(rf"(?<![A-Za-z0-9]){re.escape(text)}(?![A-Za-z0-9])", q):
                continue
        else:
            words = [t for t in _TOKEN.findall(text) if len(t) >= 3]
            if len(words) == 1 and _ordinary_word(words[0]) and not _quoted(words[0], q):
                continue
        out.append(text)
    out += [f["examples"][0] for f in r.get("families") or [] if _product_family(f, q)]
    return out


def _product_scope_mismatch(key: str, r: dict, question: str = "",
                            sql_tool: str = "") -> str | None:
    """A KPI with no product filter, asked about one product.

    "Get me the sales trend of keytruda" was answered from get_kpi('revenue-margin'): its
    per-product rows are six-month totals and its timeline is the WHOLE company, so the reply
    was "the sales data for Keytruda does not have a monthly breakdown" — while
    sales_by_material_month held all six months. None of the dashboard KPIs takes a product,
    so whatever they return for a one-product question is everyone's figure under that
    product's name. The Jun–Aug reports DO take a `material` argument and are left alone.

    `sql_tool` names the caller's SQL tool (run_query in deep mode). Fast mode's is run_sql
    and it calls without one, so the default wording names no tool rather than the wrong one.
    """
    if _kpi_takes(key, "material"):
        return None
    items = _named_products(r, question)
    if not items:
        return None
    sales = "revenue" in key or "sales" in key
    how = f"with {sql_tool}" if sql_tool else "with one SQL query"
    return (f"COMPANY-WIDE KPI — this question is about one product ({items[0]}), and "
            f"get_kpi('{key}') has no product filter: "
            + ("its per-product rows are <<SALES_MONTHS>>-month totals and any timeline it returns is the "
               "WHOLE company. " if sales else
               "whatever it returns is the figure for every product together. ")
            + f"Do not quote it as {items[0]}'s number. Query the product's own rows {how} "
            + ("instead — sales_by_material_month (period × material) for monthly sales, "
               "sales_by_material for its <<SALES_MONTHS>>-month total." if sales else
               "instead, filtered on material or material_desc."))


def _kpi_scope_mismatch(key: str, plant: str, question: str, ratio: bool = False,
                        extra: dict | None = None, sql_tool: str = "") -> str | None:
    """A company-wide KPI must not be handed back as an entity-scoped answer.

    Every other guard in this system inspects SQL. The canonical path produces none — it is
    a function call — so `city_on_unreachable_table`, `part_exceeds_whole` and the
    constraint checker all sat it out. Asked what Bangalore hospitals spend on procurement,
    the engine called get_kpi with no plant and reported ₹649.91 Cr: the total for all 51
    sites, labelled as four. Correct by construction, and the wrong question.

    `ratio` is True for a percentage KPI, where the per-site figures must never be added.
    `extra` is the call's other arguments (month, view …) when the caller forwards them; only
    then can the guard see view="hospitals", so only then is that route offered. `sql_tool`
    is the caller's SQL tool name, for the product advice.
    """
    try:
        from app.ai import resolve as _resolve
        r = _resolve.resolve(question or "")
    except Exception:
        return None

    # The product comes first: passing the right plant does not make a company-wide figure
    # about one drug.
    product = _product_scope_mismatch(key, r, question or "", sql_tool)
    if product:
        return product
    can_view = extra is not None

    # A city covering four hospitals is not answered by ONE of them either. Told only that
    # an unscoped call was wrong, the engine called get_kpi with a single plant and reported
    # ₹16.41 Cr — HC40 alone — against a true citywide ₹173.73 Cr. Under-scoping is as wrong
    # as not scoping, and it looks more convincing.
    multi = [c for c in r["cities"] if len(r["city_hospitals"].get(c) or []) > 1]
    if plant and multi:
        codes = [s.split(" ")[0] for c in multi for s in (r["city_hospitals"].get(c) or [])]
        if plant in codes and len(codes) > 1:
            return (f"UNDER-SCOPED KPI — plant='{plant}' is ONE of the {len(codes)} sites "
                    f"this question covers ({', '.join(codes)}). "
                    + (_ratio_advice(key, codes, can_view) + " " if ratio else
                       "Call get_kpi once per code and SUM them, or write one SQL statement "
                       "filtered to all of them. ")
                    + "Answering for a single hospital understates the city.")
        return None
    if plant:
        return None
    # view="hospitals" returns one row per hospital, so the named sites are in it and are
    # quoted from it. This is the route the ratio advice offers, and it must not be refused.
    if (extra or {}).get("view") == "hospitals" and _kpi_takes(key, "view"):
        return None
    sites =[f"{c}: {', '.join((r['city_hospitals'].get(c) or [])[:6])}" for c in r["cities"]]
    named = [e["text"] for e in r["entities"] if e["kind"] == "hospital"]
    if not (sites or named):
        # A hospital named the way people say it ("KR", "Nagpur", "Panda") is in neither
        # the city list nor the warehouse's codes, so this guard never saw it: asked what KR
        # spent on restricted brands, the model dropped the hospital and reported the
        # network's ₹3.17 Cr as KR's ₹43.47 L (golden fs-kr-*, 10-10-2026).
        from app.ai import hospitals as _hosp
        codes = list(dict.fromkeys(c[0] for _a, c in _hosp.mentioned(question or "") if len(c) == 1))
        if codes:
            said = ", ".join(a for a, c in _hosp.mentioned(question or "") if len(c) == 1)
            head = (f"UNSCOPED KPI — this question names {said} = {_hosp.describe(codes)}, and "
                    f"get_kpi('{key}') without a `plant` argument returns the figure for the WHOLE "
                    f"network. ")
            if len(codes) == 1:
                return head + f"Call get_kpi again with plant='{codes[0]}'."
            if _is_report_key(key):
                return head + (f"Call it with plant='{','.join(codes)}' for these hospitals pooled, "
                               f"or once per code for each hospital's own figure.")
            return head + ("Call get_kpi once per code: " + ", ".join(codes) + "."
                           + (" Quote each site's figure on its own; per-site ratios do not combine."
                              if ratio else ""))
    if not (sites or named):
        return None
    which = "; ".join(sites) or ", ".join(named)
    head = (f"UNSCOPED KPI — this question is about a specific site ({which}), and "
            f"get_kpi('{key}') without a `plant` argument returns the figure for the WHOLE "
            f"network. Reporting it as that site's number is wrong by a factor of the "
            f"estate. ")
    # ONE named hospital needs one argument, not a lecture about cities: this used to say
    # "it covers MORE THAN ONE site" even when exactly one hospital was named.
    several = len(named) > 1 or any(len(r["city_hospitals"].get(c) or []) > 1 for c in r["cities"])
    if not several:
        return head + f"Call get_kpi again with plant=<that hospital's code> ({which})."
    if ratio:
        codes = [s.split(" ")[0] for c in r["cities"] for s in (r["city_hospitals"].get(c) or [])]
        return head + "It covers MORE THAN ONE site. " + _ratio_advice(key, codes, can_view)
    return head + (f"It covers MORE THAN ONE site: call get_kpi once PER CODE listed "
                   f"above and SUM the results, or write one SQL statement filtered to all of "
                   f"those plant codes at once. Answering for a single site is just as wrong as "
                   f"answering for the network — the first attempt at this returned ₹16.41 Cr, "
                   f"one hospital out of four, against a true total of ₹173.73 Cr.")


def get_kpi(key: str, plant: str = "", category: str = "", question: str = "", **extra) -> dict:
    """The dashboard's OWN calculation for a named metric — correct by construction.

    Deep mode was re-deriving everything with hand-written SQL, including metrics the
    application already computes exactly. Asked how much stock expires in 90 days it
    returned "45,223 units" while the canonical KPI — the number on the dashboard card —
    is ₹39.97 L across 45,223 units and 869 items. Not wrong, but re-derived when a
    verified answer was one call away. Every metric taken from here is one fewer LLM
    decision in the chain, which is the only thing that actually compounds reliability.
    """
    from app.ai import kpi_registry
    if key not in kpi_registry.KPI_REGISTRY:
        return {"error": f"unknown kpi '{key}'",
                "available": sorted(kpi_registry.KPI_REGISTRY)[:40]}
    ratio = bool(kpi_registry.KPI_REGISTRY[key].get("ratio"))
    # month, payer, view, doctor … for the Jun–Aug reports. Only real values are forwarded;
    # call_kpi reports any the KPI does not take as `_ignored_args`, so a month passed to a
    # Dec–May KPI is visibly a no-op rather than silently one.
    extra = {k: v for k, v in extra.items() if v not in (None, "")}
    # A product question is refused before the per-site fan-out below: four hospitals'
    # company-wide figures are still not one drug's figures.
    try:
        from app.ai import resolve as _resolve
        product = _product_scope_mismatch(key, _resolve.resolve(question or ""),
                                          question or "", sql_tool="run_query")
    except Exception:
        product = None
    if product:
        return {"error": product}
    # A guard that only refuses leaves the model with nowhere to go: blocking both the
    # unscoped and the single-plant call produced "I couldn't establish anything" and zero
    # queries. Where the right scope is KNOWN — a city resolves to an explicit list of plant
    # codes — fetch every one of them instead of arguing about it.
    codes = _city_plant_codes(question)
    if codes and (not plant or plant in codes) and len(codes) > 1 and _is_report_key(key):
        # The Jun–Aug reports pool several hospitals themselves — the ratio of the summed
        # amounts, which is the city's real figure — so a city is one call, not four figures
        # the model would be tempted to combine. Views that cannot pool raise, and fall
        # through to the per-site calls below.
        try:
            got = kpi_registry.call_kpi(key, ",".join(codes), category or None, **extra)
        except Exception:
            got = None
        if got is not None:
            if _report_error(got):
                return {"error": _report_error(got)[:220]}
            return {"kpi": key, "canonical": True, "scoped_to": codes, "payload": got,
                    "note": (f"One pooled figure for the {len(codes)} sites this question "
                             f"covers ({', '.join(codes)}), computed by the report itself. "
                             f"Quote it as their figure and say which sites it covers.")}
    if codes and (not plant or plant in codes) and len(codes) > 1:
        per_site, failed, reason = {}, [], ""
        for code in codes:
            try:
                got = kpi_registry.call_kpi(key, code, category or None, **extra)
            except Exception as e:
                reason = reason or str(e)
                failed.append(f"{code}: {str(e)[:60]}")
                continue
            if _report_error(got):
                reason = reason or _report_error(got)
                failed.append(f"{code}: {_report_error(got)[:60]}")
            else:
                per_site[code] = got
        # PROVE THE FILTER WORKED. Sales metrics ignore `plant` entirely — their tables use
        # a site code system disjoint from dim_plant — so asking for four hospitals returned
        # the company total four times, identically, labelled as four hospitals. That is
        # worse than the unscoped answer it replaced: it manufactures site-level detail that
        # does not exist. If every site comes back the same, the argument did nothing.
        _sigs = {json.dumps(((p or {}).get("data") or {}).get("totals"), sort_keys=True,
                            default=str) for p in per_site.values()}
        if len(per_site) > 1 and len(_sigs) == 1:
            return {"error": (
                f"KPI '{key}' IGNORES the plant argument — every one of {', '.join(codes)} "
                f"returned an identical figure, which means it is the NETWORK total repeated, "
                f"not a per-site breakdown. This metric cannot be scoped to a city or a "
                f"hospital at all (its source uses a site code system disjoint from "
                f"dim_plant). Do NOT present these numbers as {', '.join(codes)}. Either "
                f"answer for the whole network and say the city cut is unavailable, or use a "
                f"table that genuinely carries the site.")}
        if failed and not per_site:
            # every site failed (data not loaded, a bad argument); the reason is the useful
            # thing to hand back, not a lecture about scoping
            return {"error": f"every site failed ({', '.join(codes)}): {reason[:220]}"}
        if per_site:
            return {"kpi": key, "canonical": True, "scoped_to": codes,
                    "per_site": per_site, "errors": failed or None,
                    "note": (f"This question covers {len(codes)} sites ({', '.join(codes)}), "
                             f"so the metric is returned PER SITE. "
                             + ("It is a RATIO, so these cannot be combined into one city "
                                "figure by arithmetic. Quote each site's figure and say which "
                                "sites are included. "
                                if ratio else
                                "Sum them for the total and say which sites are included. ")
                             + "Reporting one site, or the unscoped network figure, is a "
                               "different answer.")}
    # the extras go in so view="hospitals" — a route the ratio advice offers — is accepted
    mismatch = _kpi_scope_mismatch(key, plant, question, ratio=ratio, extra=extra,
                                   sql_tool="run_query")
    if mismatch:
        return {"error": mismatch}
    try:
        payload = kpi_registry.call_kpi(key, plant or None, category or None, **extra)
    except Exception as e:
        return {"error": str(e)[:220]}
    if _report_error(payload):
        return {"error": _report_error(payload)[:220]}
    return {"kpi": key, "canonical": True, "payload": payload}


def _report_error(payload) -> str:
    """The plain-sentence error a Jun–Aug report returns instead of data, or "".

    A report whose data is not loaded answers {"error": "..."} rather than raising. Passed
    through as a canonical payload, that sentence would become a one-row "finding" that owns
    the headline; returned as a tool error, the model reads it as what it is.
    """
    data = payload.get("data") if isinstance(payload, dict) else None
    if isinstance(data, dict) and set(data) == {"error"}:
        return str(data["error"])
    return ""


def kpi_for_model(out: dict) -> dict:
    """The copy of a get_kpi result the MODEL reads, with every figure already formatted.

    The raw payload holds rupees as floats — 604679341.2 — and handed that, the model does
    its own crore conversion and slips a decimal (₹604.68 Cr for ₹60.47 Cr). Fast mode
    formats the payload before the model sees it; this uses the same function so the two
    modes cannot drift. The raw `out` is untouched: charts and tables are built from it in
    code. Imported here, not at the top, because the orchestrator imports this package.
    """
    try:
        from app.ai.orchestrator import _format_kpi_payload
    except Exception:
        return out
    view = dict(out)
    if isinstance(view.get("payload"), dict):
        view["payload"] = _format_kpi_payload(view["payload"])
    if isinstance(view.get("per_site"), dict):
        view["per_site"] = {k: _format_kpi_payload(v) for k, v in view["per_site"].items()}
    return view


def kpi_keys() -> list[str]:
    from app.ai import kpi_registry
    return sorted(kpi_registry.KPI_REGISTRY)


def kpi_entry(key: str) -> dict:
    """The registry entry for a key, or {} — so callers can read `needs`/`ratio` safely."""
    from app.ai import kpi_registry
    return kpi_registry.KPI_REGISTRY.get(key or "") or {}


# ── OpenAI tool schemas ──────────────────────────────────────────────────────
def _t(name, desc, props, required):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": required}}}


def _get_kpi_spec() -> dict:
    """get_kpi's arguments, taken from the registry's own schema.

    Deep mode kept a hand-written copy with only key/plant/category, so it could never pass
    a month, payer or view to the Jun–Aug reports and always answered for the latest month
    and every hospital. Reading the registry's schema means an argument added there reaches
    this engine without a second list to keep in step. The registry calls the key `kpi`;
    this engine has always called it `key`, so it is renamed rather than changing dispatch.
    """
    from app.ai import kpi_registry
    params = copy.deepcopy(kpi_registry.tool_schema()["function"]["parameters"])
    params["properties"] = {("key" if k == "kpi" else k): v
                            for k, v in (params.get("properties") or {}).items()}
    params["required"] = ["key" if r == "kpi" else r for r in params.get("required") or ["key"]]
    reports = [f"{k} ({e.get('label', k)}): {e.get('about', '')}"
               for k, e in sorted(kpi_registry.KPI_REGISTRY.items()) if e.get("needs")]
    desc = ("The dashboard's OWN calculation for a named metric — correct by construction and "
            "already verified. PREFER THIS over writing SQL whenever the question matches one "
            "of these metrics; only fall back to run_query for things no KPI covers.")
    if reports:
        desc += ("\n<<REPORTS>> REPORTS (no SQL table holds these; they default to the latest "
                 "month and all hospitals, so pass the month, hospital and other arguments the "
                 "question names, and say which month you used):\n- " + "\n- ".join(reports)
                 + "\nEvery other key is a <<SALES>> dashboard KPI and ignores those "
                   "extra arguments.")
    return {"type": "function", "function": {"name": "get_kpi", "description": desc,
                                             "parameters": params}}


def _kpi_extras(a: dict) -> dict:
    """The report arguments (month, payer, view, …) the model passed, by the registry's list."""
    from app.ai import kpi_registry
    names = getattr(kpi_registry, "EXTRA_ARGS", ())
    return {k: a[k] for k in names if a.get(k) not in (None, "")}


SPECS = [
    _t("list_tables", "List warehouse tables with row counts. Start here if unsure what exists.",
       {"contains": {"type": "string", "description": "optional substring filter"}}, []),
    _t("describe_table", "Columns and types of one table. ALWAYS call this before writing a "
       "GROUP BY or WHERE on a column you have not already seen in this conversation.",
       {"table": {"type": "string"}}, ["table"]),
    _t("sample_rows", "A few real rows, so you can see what the values look like.",
       {"table": {"type": "string"}, "n": {"type": "integer"}}, ["table"]),
    _t("find_columns", "Which tables have a column whose name contains this. Use it to find "
       "which tables carry a grain — e.g. find_columns('month') before assuming one has months.",
       {"name_like": {"type": "string"}}, ["name_like"]),
    _t("find_value", "Which table.column actually CONTAINS this value (e.g. 'Bangalore', 'MSD'). "
       "Use it when a filter returns nothing, or to locate a name you were given.",
       {"value": {"type": "string"}}, ["value"]),
    _t("profile_column", "Distinct count, nulls and commonest values of a column.",
       {"table": {"type": "string"}, "column": {"type": "string"}}, ["table", "column"]),
    _get_kpi_spec(),
    _t("run_query", "Run a read-only SELECT and see the rows.",
       {"sql": {"type": "string"}, "purpose": {"type": "string"}}, ["sql", "purpose"]),
    _t("finish", "Call this once you have the rows that answer the sub-question. Pass the SQL "
       "that produced them.",
       {"sql": {"type": "string"}, "what_it_shows": {"type": "string"}}, ["sql", "what_it_shows"]),
    _t("give_up", "Call this only after LOOKING — you have checked which tables carry the grain "
       "and the entity and none does. Say precisely what is missing.",
       {"reason": {"type": "string"}, "closest_available": {"type": "string"}}, ["reason"]),
]

DISPATCH = {
    "list_tables": lambda a, ctx: list_tables(a.get("contains", "")),
    "describe_table": lambda a, ctx: describe_table(a.get("table", "")),
    "sample_rows": lambda a, ctx: sample_rows(a.get("table", ""), a.get("n", 5)),
    "find_columns": lambda a, ctx: find_columns(a.get("name_like", "")),
    "find_value": lambda a, ctx: find_value(a.get("value", "")),
    "profile_column": lambda a, ctx: profile_column(a.get("table", ""), a.get("column", "")),
    "get_kpi": lambda a, ctx: get_kpi(a.get("key", ""), a.get("plant", ""),
                                      a.get("category", ""), ctx.get("question", ""),
                                      **_kpi_extras(a)),
    "run_query": lambda a, ctx: run_query(a.get("sql", ""), ctx.get("entity_tokens"), ctx.get("question", "")),
}


def call(name: str, args: dict, ctx: dict) -> dict:
    fn = DISPATCH.get(name)
    if not fn:
        return {"error": f"no tool named {name}"}
    try:
        return fn(args, ctx)
    except Exception as e:  # a tool must never kill the loop
        return {"error": str(e)[:200]}


def compact(obj: dict, limit: int = 2600) -> str:
    """A tool result as the model reads it, within `limit` characters.

    Cutting the JSON at the limit used to land mid-string, so a KPI's totals or a second
    site's figures simply vanished and the model read a half-object. Whole rows are dropped
    from the longest lists first, with a note of how many, so what arrives is still valid
    JSON with its totals and notes intact. ensure_ascii=False keeps '₹' as one character
    instead of six.
    """
    o = {k: v for k, v in obj.items() if not k.startswith("_")}
    s = json.dumps(o, default=str, ensure_ascii=False)
    if len(s) <= limit:
        return s
    fitted = _drop_rows_to_fit(json.loads(s), limit)
    return fitted if fitted is not None else s[:limit] + " …(truncated)"


def _drop_rows_to_fit(o, limit: int) -> str | None:
    """Halve the biggest list until the JSON fits; None when lists are not what is big."""
    hidden: dict[int, int] = {}

    def lists(node):
        if isinstance(node, list):
            yield node
            for x in node:
                yield from lists(x)
        elif isinstance(node, dict):
            for v in node.values():
                yield from lists(v)

    for _ in range(60):
        cands = [x for x in lists(o) if len(x) - (1 if id(x) in hidden else 0) > 1]
        if not cands:
            return None
        big = max(cands, key=lambda x: len(json.dumps(x, ensure_ascii=False)))
        if id(big) in hidden:
            big.pop()                      # the "… more not shown" marker from last time
        keep = max(1, len(big) // 2)
        hidden[id(big)] = hidden.get(id(big), 0) + len(big) - keep
        del big[keep:]
        big.append(f"… {hidden[id(big)]} more not shown")
        s = json.dumps(o, ensure_ascii=False)
        if len(s) <= limit:
            return s
    return None
