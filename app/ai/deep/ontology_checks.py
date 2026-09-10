"""Query checks DERIVED from the ontology, not typed by hand.

This is the second half of the measured gain. On the OMG Property & Casualty schema an
ontology used purely as CONTEXT took GPT-4 from 16% to 54%; using it additionally to CHECK
generated queries and repair them took it to 72%, with the error rate falling from 45.8% to
20% (arXiv 2405.11706). Context tells the model what exists. Checks stop it acting on a
misunderstanding.

Every check here reads app/ai/ontology.json. None of them contains a table name, a column
name or a business rule — rename a column and the checks follow, because the ontology is
rebuilt from the warehouse rather than edited.

They REPORT. The engine turns a report into a tool error, which is what makes the model
rewrite the query rather than read past a warning.
"""
from __future__ import annotations

import re

from app.ai import ontology

_FROM_JOIN = re.compile(r"\b(?:FROM|JOIN)\s+\"?([A-Za-z_]\w*)\"?", re.I)
_SUM = re.compile(r"\bSUM\s*\(\s*(?:DISTINCT\s+)?([A-Za-z_][\w.]*)", re.I)
_GROUP_BY = re.compile(r"\bGROUP\s+BY\b(.*?)(?:\bORDER\b|\bHAVING\b|\bLIMIT\b|$)", re.I | re.S)


def _tables(sql: str) -> list[str]:
    return list(dict.fromkeys(_FROM_JOIN.findall(sql or "")))


def sums_a_non_additive_measure(sql: str) -> str | None:
    """SUM() over a rate, a price or a pre-computed average.

    Adding percentages produces a number with no meaning, and this is the class that once
    printed "value share percentages as high as ₹85". The ontology marks additivity per
    column, checked in its verify pass, so no list of suffixes has to be maintained here.
    """
    if not sql:
        return None
    bad = {k.split(".", 1)[1]: k for k in ontology.non_additive()}
    if not bad:
        return None
    used = _tables(sql)
    for col in _SUM.findall(sql):
        name = col.split(".")[-1]
        key = bad.get(name)
        if not key:
            continue
        if used and key.split(".", 1)[0] not in used:
            continue                      # a same-named column on a table not in this query
        return (f"SUM({col}) IS NOT MEANINGFUL — the ontology records {key} as "
                f"NON-ADDITIVE: it is a rate, a price or an average that was already "
                f"computed per row. Adding it across rows produces a number that measures "
                f"nothing. Weight it, average it, or recompute the ratio from its "
                f"numerator and denominator instead.")
    return None


def joins_disjoint_columns(sql: str) -> str | None:
    """A join between two columns the ontology has measured as sharing no values.

    dim_plant.plant and sales_by_hospital.hospital are both the hospital and overlap by
    ZERO. A join between them silently returns nothing, or worse, an unfiltered total
    labelled as one site.
    """
    if not sql:
        return None
    used = set(_tables(sql))
    if len(used) < 2:
        return None
    for d in ontology.load().get("disjoint", []):
        a, b = d.get("a"), d.get("b")
        if a in used and b in used:
            other = d.get("other", d.get("column"))
            ent = d.get("entity") or "the same thing"
            return (f"THESE TABLES CANNOT BE JOINED — {a}.{d['column']} and {b}.{other} "
                    f"both identify {ent}, and they share {d['overlap']:.0%} of their "
                    f"values. They use different code systems. This join returns nothing "
                    f"useful, and any total it produces is not what it claims to be. Answer "
                    f"from one side and say the other cannot be reached.")
    return None


def groups_by_an_ambiguous_column(sql: str) -> str | None:
    """GROUP BY a column name that means different things in different tables."""
    if not sql:
        return None
    m = _GROUP_BY.search(sql)
    if not m:
        return None
    grouped = {t.strip().strip('"').split(".")[-1].lower() for t in m.group(1).split(",")}
    used = set(_tables(sql))
    for d in ontology.load().get("disjoint", []):
        col = str(d.get("column", "")).lower()
        if col in grouped and (d.get("a") in used or d.get("b") in used):
            here = d["a"] if d["a"] in used else d["b"]
            there = d["b"] if here == d["a"] else d["a"]
            return (f"\"{col}\" MEANS DIFFERENT THINGS IN DIFFERENT TABLES — the values in "
                    f"{here}.{col} share {d['overlap']:.0%} with {there}.{col}, so they are "
                    f"two separate taxonomies, not one. Name which one you grouped by, "
                    f"because the reader will assume the other.")
    return None


def check(sql: str) -> str | None:
    """The first ontology violation in this query, or None."""
    for fn in (sums_a_non_additive_measure, joins_disjoint_columns,
               groups_by_an_ambiguous_column, averages_a_skewed_measure,
               alias_contradicts_the_ontology):
        hit = fn(sql)
        if hit:
            return hit
    return None


_AVG = re.compile(r"\bAVG\s*\(\s*(?:DISTINCT\s+)?([A-Za-z_][\w.]*)", re.I)
_SKEW_FACTOR = 3.0


def averages_a_skewed_measure(sql: str) -> str | None:
    """AVG() over a measure whose mean is far from its median.

    "Our overall days-on-hand is 203.34 days" — with the median at 16.57. The mean is
    twelve times the median because a few thousand near-dead SKUs carry enormous day counts,
    and quoting it as "overall" describes a warehouse nobody works in. The ontology already
    marks doh_days non-additive; this asks the data whether the mean is safe to report, and
    the answer is a measurement, not a rule of thumb.
    """
    if not sql:
        return None
    non_add = {k.split(".", 1)[1]: k for k in ontology.non_additive()}
    used = _tables(sql)
    for col in _AVG.findall(sql):
        name = col.split(".")[-1]
        key = non_add.get(name)
        if not key:
            continue
        table = key.split(".", 1)[0]
        if used and table not in used:
            continue
        try:
            from app.ai import warehouse
            mean, med = warehouse.con().execute(
                f'SELECT AVG("{name}"), MEDIAN("{name}") FROM "{table}"').fetchone()
        except Exception:
            return None
        if not mean or not med or med == 0:
            return None
        if abs(mean) < abs(med) * _SKEW_FACTOR:
            return None
        return (f"AVG({col}) IS MISLEADING HERE — across {table} its mean is {mean:,.1f} "
                f"and its median {med:,.1f}, a {abs(mean / med):.0f}x gap, so a few extreme "
                f"rows are carrying the average. Report the MEDIAN as the headline figure "
                f"and mention the mean only alongside what skews it.")
    return None


def alias_contradicts_the_ontology(sql: str) -> str | None:
    """An aggregate aliased into a measure family the ontology says it is not.

    `SELECT SUM(amount_lc) AS total_revenue FROM fact_consumption` — a CONSUMPTION cost
    relabelled as revenue, and the prose then faithfully reported "monthly revenue fell
    from ₹10.91 Cr". constraints.misleading_alias missed it because its hand-written pattern
    list has no entry for `amount_lc`; the ontology classified that column's event as
    consumption from its own samples, without anyone typing the column name anywhere.

    This is the argument for the ontology in one function: a hand list only knows the names
    somebody thought of.
    """
    if not sql:
        return None
    ont = ontology.load().get("columns") or {}
    if not ont:
        return None
    used = _tables(sql)

    def event_of(col: str) -> str:
        name = col.split(".")[-1]
        for key, spec in ont.items():
            t, c = key.split(".", 1)
            if c == name and (not used or t in used) and spec.get("role") == "measure":
                return str(spec.get("event", "")).lower()
        return ""

    def alias_claims(alias: str) -> str:
        a = alias.lower()
        for ev, words in (("sales", ("revenue", "sales", "turnover", "billed")),
                          ("procurement", ("purchase", "procure", "spend", "bought")),
                          ("consumption", ("consum", "issued", "dispens", "used")),
                          ("stock", ("stock", "inventory", "on_hand", "holding"))):
            if any(w in a for w in words):
                return ev
        return ""

    for src_col, alias in re.findall(
            r"\b(?:SUM|AVG|MIN|MAX)\s*\(\s*(?:DISTINCT\s+)?([A-Za-z_][\w.]*)[^()]*\)"
            r"\s*(?:AS\s+)?([A-Za-z_]\w*)", sql, re.I):
        src_ev, claim = event_of(src_col), alias_claims(alias)
        if src_ev and claim and src_ev != claim:
            return (f"ALIAS CONTRADICTS THE DATA — `{src_col}` records {src_ev.upper()} "
                    f"according to the ontology, and you have named it `{alias}`, which "
                    f"reads as {claim.upper()}. Those are different business events from "
                    f"different tables, and the answer will use the name you chose. Either "
                    f"query the table that actually records {claim.upper()}, or name the "
                    f"column for what it is.")
    return None
