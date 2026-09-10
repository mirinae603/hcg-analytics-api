"""Deterministic SQL, compiled from the ontology — no model writes it.

THE PROBLEM THIS SOLVES

A 3-run measurement of the whole bank found something that reframed the work: NOT ONE CASE
failed all three runs. Fourteen failed at least once, the worst two of three. Every question
is answered correctly sometimes. So the residual error is not missing knowledge — the system
has it — it is ROUTE VARIANCE. The same question takes a different path on different runs
(canonical KPI, decomposition, direct fallback, grain retry) and some paths are wrong.

Guards cannot fix that. A guard rejects a bad route after it is taken, and every guard that
rejects a GOOD one makes it worse — two of those cost three points on their own.

THE ARCHITECTURE

For the questions that have exactly one right answer shape, remove the choice. The model
decides WHAT is being asked; the ontology decides HOW to fetch it, and the SQL is assembled
in code:

    question -> resolve()          measure, grain, entities, qualifiers   (already exists)
             -> plan()             a typed request, or nothing
             -> compile()          one SQL string, built from ontology facts
             -> execute            no model between the plan and the rows

There is exactly one route, so there is no variance. The measure column, its additivity, the
grain column and the table all come from ontology.json, which is rebuilt from the warehouse
— nothing here names a table or a column.

WHY IT IS A ROUTER, NOT A REPLACEMENT

Compiling everything is the semantic-layer trap: a metric compiler is perfectly consistent
on what it models and scores ZERO on anything else, and the only independent clinical-domain
evidence shows pre-specified layers are systematically incomplete. So plan() returns None
the moment anything is ambiguous — two candidate tables, an unknown grain, a comparison, a
question needing more than one measure — and the ReAct engine handles it exactly as now.
Determinism where it is safe, exploration where it is needed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.ai import ontology, resolve


@dataclass
class Plan:
    """A question reduced to something that can be compiled without further decisions."""
    measure_col: str                      # table.column, from the ontology
    table: str
    grain_col: str = ""                   # "" means a single total
    filters: list = field(default_factory=list)   # (column, value) pairs
    aggregate: str = "SUM"
    order_desc: bool = True
    limit: int = 0
    why: str = ""


# Only these shapes are compiled. Anything else goes to the open path.
_RANKING = re.compile(r"\b(top|biggest|largest|highest|most|worst|lowest|smallest|least)\b", re.I)
_TREND = re.compile(r"\btrend\b|\bover time\b|\bmonth(ly|-on-month)?\b|\bby month\b", re.I)
_TOTAL = re.compile(r"\b(total|overall|altogether|in all|sum of)\b", re.I)


# What a question about this measure is actually asking to be counted in. Without it,
# "total procurement spend" matched 40 columns because po_qty and gr_qty are procurement
# measures too — they are just not money.
_WANT_UNIT = {"revenue": "currency", "purchasing": "currency", "margin": "currency",
              "quantity": "quantity", "lead_time": "days"}


def _measure_candidates(measure: str, grain: str) -> list[str]:
    """Ontology columns that record this measure and live beside this grain."""
    if not measure:
        return []
    want_event = {"revenue": "sales", "purchasing": "procurement",
                  "consumption": "consumption", "stock": "stock"}.get(measure, "")
    want_unit = _WANT_UNIT.get(measure, "")
    cols = ontology.load().get("columns") or {}
    grain_tables = {k.split(".", 1)[0] for k in ontology.columns_for_entity(grain)} if grain else set()
    out = []
    for key, spec in cols.items():
        if spec.get("role") != "measure" or not spec.get("additive", True):
            continue
        if want_event and str(spec.get("event", "")).lower() != want_event:
            continue
        if want_unit and str(spec.get("unit", "")).lower() != want_unit:
            continue
        table = key.split(".", 1)[0]
        if table.startswith(("_", "forecast")):
            continue
        if grain and table not in grain_tables:
            continue
        out.append(key)
    # a purpose-built view beats a raw fact beats a pre-aggregated KPI
    out.sort(key=lambda k: (k.startswith("kpi_"), len(k)))
    return out


def _dominant(candidates: list[str]) -> str:
    """The headline measure among several of the same unit and event, or "".

    This is the ONE fact the ontology carries that a column list cannot: which column an
    executive means by a table's headline figure. An earlier version guessed by size —
    take the largest — and compiled `total_mrp_value` as procurement spend, reporting
    ₹1,777 Cr of retail value where ₹649.91 Cr of purchasing belonged. Deterministically,
    confidently, and wrongly, which is worse than the variance it set out to remove.

    Now it asks. A column marked primary wins; several claimants means none of them are, and
    the question goes to the open path.
    """
    tables = {k.split(".", 1)[0] for k in candidates}
    primaries = {ontology.primary_measure(t) for t in tables}
    hits = [k for k in candidates if k in primaries]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        return ""
    # SEVERAL HEADLINE COLUMNS IS NOT AMBIGUITY IF THEY AGREE. sales_totals.revenue,
    # sales_monthly.revenue and sales_by_material.revenue are the same ₹521.67 Cr sliced
    # three ways, so which one answers "total sales revenue" does not matter — the answer is
    # identical. Ambiguity is when they DISAGREE, and that is a question for the data, not a
    # preference order. The smallest table wins the tie because it is the cheapest to read.
    from app.ai import warehouse
    totals = []
    for key in sorted(hits):
        t, c = key.split(".", 1)
        try:
            v, n = warehouse.con().execute(
                f'SELECT SUM("{c}"), COUNT(*) FROM "{t}"').fetchone()
        except Exception:
            return ""
        if v is None:
            return ""
        totals.append((float(v), n, key))
    biggest = max(abs(v) for v, _n, _k in totals)
    if biggest and any(abs(abs(v) - biggest) > biggest * 0.001 for v, _n, _k in totals):
        return ""                          # they disagree — a real ambiguity
    return min(totals, key=lambda x: x[1])[2]


def _filter_column(table: str, kind: str) -> str:
    """A column on this table that can scope to an entity of this kind."""
    if not kind:
        return ""
    # DESCRIPTORS COUNT. columns_for_entity returns identifiers and dimensions only, so
    # `material_desc` — classified a descriptor — was invisible, and the filter landed on
    # `material`, which holds '101313'. Matching 'KEYTRUDA 100MG INJ VIAL' against a code
    # column returns nothing, and an empty result reads as "this hospital sells no
    # Keytruda". A descriptor is exactly what you filter on when a person names a thing.
    cols = [k.split(".", 1)[1] for k, spec in (ontology.load().get("columns") or {}).items()
            if k.split(".", 1)[0] == table
            and str(spec.get("entity", "")).lower() == kind.lower()
            and spec.get("role") in ("identifier", "dimension", "descriptor")]
    if not cols:
        return ""
    # A NAME MATCHES A DESCRIPTION, NOT A CODE. Filtering material = 'KEYTRUDA 100MG INJ
    # VIAL' matches nothing, because that column holds '101313' — and an empty result reads
    # as "this hospital sells no Keytruda" rather than "I filtered the wrong column".
    for c in cols:
        if "desc" in c or "name" in c:
            return c
    return cols[0]


def _grain_column(table: str, grain: str) -> str:
    for key in ontology.columns_for_entity(grain):
        t, c = key.split(".", 1)
        if t == table:
            return c
    return ""


def plan(question: str) -> Plan | None:
    """A compilable request, or None — and None is the common, correct answer."""
    if not question:
        return None
    r = resolve.resolve(question)
    measures, grains = r.get("measures") or [], r.get("grains") or []

    # anything ambiguous goes to the open path, deliberately
    if len(measures) != 1:
        return None
    if r.get("qualifiers"):                       # "high-value" needs a computed threshold
        return None
    if len(grains) > 1:                           # two grains is a cross-tab, not a lookup
        return None
    if re.search(r"\bvs\b|\bversus\b|\bcompare|\bagainst\b|\bwhy\b|\bdriver", question, re.I):
        return None

    measure, grain = measures[0], (grains[0] if grains else "")
    if grain == "month":                          # trends need the period key and shaping
        return None

    cands = _measure_candidates(measure, grain)
    key = cands[0] if len(cands) == 1 else _dominant(cands)
    if not key:                                   # ambiguous -> let the engine look
        return None
    table, mcol = key.split(".", 1)
    gcol = _grain_column(table, grain) if grain else ""
    if grain and not gcol:
        return None

    shape_total = bool(_TOTAL.search(question)) and not grain
    shape_rank = bool(_RANKING.search(question)) and bool(grain)
    if not (shape_total or shape_rank):
        return None
    if _TREND.search(question):
        return None

    # ENTITY FILTERS. Without these the compiler answered "which hospitals sell the most
    # KEYTRUDA" with the top hospital overall — right shape, wrong scope, and confidently
    # so. An entity the question named must reach the WHERE clause or the plan is not a plan
    # for that question. If the chosen table cannot hold the filter, decline and let the
    # engine find a table that can.
    filters = []
    named = [e for e in (r.get("entities") or [])] + [
        {"text": f["examples"][0] if f["n"] == 1 else f["token"], "kind": f["kind"]}
        for f in (r.get("families") or []) if f["n"] == 1]
    for e in named:
        col = _filter_column(table, e.get("kind", ""))
        if not col:
            return None                   # the question names something this table cannot scope
        filters.append((col, str(e["text"])[:60]))
    if r.get("cities"):
        return None                       # a city needs plant codes and a reachability check

    return Plan(measure_col=mcol, table=table, grain_col=gcol, filters=filters,
                aggregate="SUM", limit=5 if shape_rank else 0,
                why=f"{measure} at {grain or 'total'} grain from {key}")


def compile_sql(p: Plan) -> str:
    """One SQL string, assembled from the plan. No model, no choices."""
    m = f'{p.aggregate}("{p.measure_col}")'
    where = ""
    if p.filters:
        clauses = [f'upper(CAST("{c}" AS VARCHAR)) LIKE upper(\'%{v}%\')' for c, v in p.filters]
        where = " WHERE " + " AND ".join(clauses)
    if not p.grain_col:
        return f'SELECT {m} AS "{p.measure_col}" FROM "{p.table}"{where}'
    order = "DESC" if p.order_desc else "ASC"
    limit = f" LIMIT {p.limit}" if p.limit else ""
    return (f'SELECT "{p.grain_col}", {m} AS "{p.measure_col}" FROM "{p.table}"{where} '
            f'GROUP BY 1 ORDER BY 2 {order}{limit}')
