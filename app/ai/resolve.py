"""Understand the question BEFORE querying it.

WHY
---
Every wrong answer this assistant has produced came from acting before identifying. It
answered about a stationery sticker called STICKER-MSDS when asked about MSD the
manufacturer. It reported a city-filtered sales figure for Bangalore, which cannot exist,
because "Bangalore" lives only in dim_plant.plant_name and that column shares no rows with
the sales tables. It answered "sales trend" from purchasing without saying so. In each case
the entity was never TYPED — the model saw a word and guessed which column it belonged to.

A human analyst does the opposite: read the question, name the things in it (this is a drug,
that is a city, that is a manufacturer, that is a category), decide what is being measured,
and only then write SQL. This module is that first step, done deterministically.

HOW
---
One in-memory index of the warehouse's dimension VALUES, built once and cached. Question
n-grams are matched against it, scored, and returned TYPED — with the table and column that
actually holds each value, so the query has nowhere to go wrong.

Tokens that appear in thousands of values ("INJ", "TAB", "MG") carry no identifying power
and are excluded by document frequency, so "how many tablets" does not resolve to five
thousand materials.

Used by both answer paths: the resolution is a fact about the question, not a property of
whichever engine is running.
"""
from __future__ import annotations

import re
from collections import defaultdict
from functools import lru_cache

# (kind, table, column, max_distinct) — the dimensions worth indexing. A column with tens of
# thousands of values is still fine: the index is inverted, so lookup cost is per QUESTION
# token, not per value.
_SOURCES: list[tuple[str, str, str, int]] = [
    ("material",     "dim_material", "material_desc",     40000),
    ("material",     "dim_material", "material",           40000),
    ("generic",      "dim_material", "generic_name",        8000),
    ("manufacturer", "dim_material", "manufacturer_desc",   5000),
    ("category",     "dim_material", "material_group",       500),
    ("material_type","dim_material", "material_type",         50),
    ("category",     "mart_procurement", "category",          500),
    ("manufacturer", "sales_by_manufacturer", "manufacturer",       5000),
    ("manufacturer", "sales_by_material_mfr", "manufacturer",        5000),
    ("manufacturer", "mart_procurement", "manufacturer_desc",        5000),
    ("vendor",       "dim_vendor",   "vendor_name",         8000),
    ("vendor",       "mart_procurement", "vendor_name",         8000),
    ("hospital",     "dim_plant",    "plant_name",           200),
    ("hospital",     "dim_plant",    "plant",                200),
    ("department",   "dim_costcenter", "department_name",   2000),
]

_WORD = re.compile(r"[A-Za-z0-9]+")
# Tokens that identify nothing: units, dosage forms, and filler that appears across
# thousands of product names.
_NOISE_TOKENS = {
    "inj", "tab", "tabs", "cap", "caps", "mg", "ml", "gm", "gms", "mcg", "iu", "vial",
    "syrup", "susp", "sol", "solution", "amp", "pack", "nos", "set", "kit", "pcs", "no",
    "gen", "pvt", "ltd", "limited", "india", "healthcare", "health", "pharma", "pharms",
    "pharmaceuticals", "medical", "hospital", "hospitals", "hcg", "the", "and", "for", "of",
    "with", "size", "single", "double", "small", "large", "medium", "full", "half",
}
_STOP = {
    "what", "which", "how", "why", "when", "where", "who", "show", "give", "list", "get",
    "tell", "our", "the", "and", "for", "are", "was", "were", "is", "in", "on", "at", "to",
    "by", "of", "from", "with", "we", "us", "me", "you", "top", "most", "least", "best",
    "worst", "much", "many", "compare", "across", "each", "every", "total", "all", "any",
    "do", "does", "did", "have", "has", "had", "can", "could", "should", "would", "please",
    "also", "than", "then", "that", "this", "these", "those", "it", "its", "their", "there",
    "over", "under", "between", "per", "vs", "versus", "into", "out", "up", "down", "now",
    "right", "currently", "actually", "really", "just",
}
_MAX_DF = 250          # a token in more than this many values identifies nothing


@lru_cache(maxsize=1)
def _index() -> tuple[dict, dict]:
    """token -> [(kind, value, table, column)], plus exact-value lookup."""
    from app.ai import warehouse
    postings: dict[str, list[tuple]] = defaultdict(list)
    exact: dict[str, tuple] = {}
    for kind, table, column, cap in _SOURCES:
        try:
            rows = warehouse.run_sql(
                f"SELECT DISTINCT CAST({column} AS VARCHAR) AS v FROM {table} "
                f"WHERE {column} IS NOT NULL AND CAST({column} AS VARCHAR) <> ''",
                row_cap=cap, timeout_s=25.0)["rows"]
        except Exception:
            continue
        for r in rows:
            v = str(r["v"]).strip()
            if not v or len(v) < 2:
                continue
            exact.setdefault(v.upper(), (kind, v, table, column))
            for tok in {t.lower() for t in _WORD.findall(v)}:
                if len(tok) < 3 or tok in _NOISE_TOKENS:
                    continue
                postings[tok].append((kind, v, table, column))
    # drop tokens that appear everywhere — they cannot identify anything
    return ({t: p for t, p in postings.items() if len(p) <= _MAX_DF}, exact)


@lru_cache(maxsize=1)
def cities() -> tuple[str, ...]:
    """City names, which exist only inside hospital names ('HCG KR, Bangalore')."""
    from app.ai import warehouse
    out: set[str] = set()
    try:
        rows = warehouse.run_sql("SELECT DISTINCT plant_name FROM dim_plant WHERE plant_name IS NOT NULL",
                                 row_cap=200)["rows"]
    except Exception:
        return ()
    for r in rows:
        name = str(r["plant_name"])
        if "," in name:
            tail = name.rsplit(",", 1)[-1].strip()
            for part in tail.replace("(", " ").replace(")", " ").split():
                if len(part) > 3 and part.lower() not in _NOISE_TOKENS:
                    out.add(part.title())
    return tuple(sorted(out))


# What is being MEASURED. Ordered: the first match wins, so the more specific phrasings
# come first.
_MEASURES: list[tuple[str, tuple[str, ...]]] = [
    ("margin",       ("margin", "profit", "profitability")),
    ("revenue",      ("revenue", "sales", "sold", "sell", "selling", "turnover", "billed")),
    ("purchasing",   ("purchase", "purchasing", "procure", "procurement", "spend", "spent",
                     "bought", "buy", "buying", "po ", "order", "ordering")),
    ("consumption",  ("consumption", "consumed", "usage", "used", "issued")),
    ("stock",        ("stock", "inventory", "on hand", "holding", "hold", "carrying")),
    ("expiry",       ("expiry", "expiring", "expired", "near expiry", "shelf life")),
    ("lead_time",    ("lead time", "lead-time", "delivery time", "turnaround")),
    ("quantity",     ("quantity", "units", "qty", "volume")),
    ("price",        ("price", "pricing", "rate", "cost per")),
]
_GRAINS: list[tuple[str, tuple[str, ...]]] = [
    ("month",    ("month", "monthly", "per month", "over time", "trend", "month-on-month")),
    ("hospital", ("hospital", "site", "centre", "center", "location", "branch", "plant")),
    ("vendor",   ("vendor", "supplier", "distributor")),
    ("manufacturer", ("manufacturer", "brand", "maker")),
    ("category", ("category", "group", "therapy area", "class", "segment")),
    ("material", ("item", "sku", "product", "drug", "medicine", "material")),
    ("department", ("department", "ward", "cost centre", "cost center")),
]


@lru_cache(maxsize=1)
def _intent_words() -> frozenset[str]:
    """Words that say what is being MEASURED, not what is being talked about.

    "sales trend" matched a vendor literally named "Pharma Sales"; "how many tablets"
    matched a manufacturer named "TABLETS INDIA". A word carrying intent must never be
    used to identify an entity, or every question resolves to whatever company happened
    to put that word in its name.
    """
    out: set[str] = set()
    for _, words in _MEASURES + _GRAINS:
        for w in words:
            out.update(t.lower() for t in _WORD.findall(w))
    return frozenset(out)


def _phrases(question: str, max_len: int = 6) -> list[str]:
    words = _WORD.findall(question or "")
    out = []
    for n in range(min(max_len, len(words)), 0, -1):
        for i in range(len(words) - n + 1):
            out.append(" ".join(words[i:i + n]))
    return out


def resolve(question: str, limit: int = 8) -> dict:
    """Name the things in the question, and say what is being measured.

    Returns entities TYPED and located (table + column), so downstream SQL has nowhere to
    misfile them, plus the measure and grain the question is asking for.
    """
    q = question or ""
    postings, exact = _index()
    qtokens = [t.lower() for t in _WORD.findall(q)]
    intent = _intent_words()
    interesting = [t for t in qtokens
                   if len(t) >= 3 and t not in _STOP and t not in _NOISE_TOKENS and t not in intent]

    # score each candidate value by how much of it the question actually contains
    scores: dict[tuple, float] = defaultdict(float)
    for tok in set(interesting):
        for cand in postings.get(tok, ()):
            scores[cand] += 1.0

    qupper = q.upper()
    entities = []
    for cand, hits in sorted(scores.items(), key=lambda kv: -kv[1]):
        kind, value, table, column = cand
        vtokens = {t.lower() for t in _WORD.findall(value)
                   if len(t) >= 3 and t.lower() not in _NOISE_TOKENS}
        if not vtokens:
            continue
        coverage = hits / len(vtokens)
        # Accept when the question contains the value outright, or covers most of its
        # identifying tokens. A single shared token out of five is a coincidence — that is
        # how "MSD" reached STICKER-MSDS.
        # An inexact match needs TWO identifying tokens. One shared token out of a value's
        # few is a coincidence, and coincidences are how "MSD" reached STICKER-MSDS and
        # "Bangalore" reached one arbitrary hospital out of four.
        if value.upper() in qupper or (coverage >= 0.75 and hits >= 2):
            entities.append({"text": value, "kind": kind, "table": table, "column": column,
                             "confidence": round(min(1.0, coverage), 2),
                             "exact": value.upper() in qupper})
        if len(entities) >= limit * 3:
            break

    # prefer exact, then higher coverage, then longer (more specific) values
    entities.sort(key=lambda e: (-int(e["exact"]), -e["confidence"], -len(e["text"])))
    lines_case_note = False
    merged: dict[tuple[str, str], dict] = {}
    for e in entities:
        key = (e["kind"], e["text"].upper())
        if key in merged:
            # Case varies between tables — 'MSD' in dim_material, 'Msd' in
            # sales_by_manufacturer — and an `=` filter on the wrong one silently matches
            # nothing. Record the spelling each column actually stores.
            loc = f"{e['table']}.{e['column']} (stored as \"{e['text']}\")"
            if loc not in merged[key]["locations"]:
                merged[key]["locations"].append(loc)
            continue
        e = {**e, "locations": [f"{e['table']}.{e['column']} (stored as \"{e['text']}\")"]}
        merged[key] = e
        if len(merged) >= limit:
            break
    deduped = list(merged.values())

    found_cities = [c for c in cities() if re.search(rf"\b{re.escape(c)}\b", q, re.I)]
    low = q.lower()
    measures = [name for name, words in _MEASURES if any(w in low for w in words)]
    grains = [name for name, words in _GRAINS if any(w in low for w in words)]

    # a city is not one hospital — name the ones it covers, so a filter can be written
    city_hospitals: dict[str, list[str]] = {}
    if found_cities:
        from app.ai import warehouse
        for c in found_cities:
            try:
                rows = warehouse.run_sql(
                    "SELECT plant, plant_name FROM dim_plant WHERE upper(plant_name) LIKE '%' || upper(?) || '%'"
                    .replace("?", f"'{c}'"), row_cap=30)["rows"]
                city_hospitals[c] = [f"{x['plant']} ({x['plant_name']})" for x in rows][:8]
            except Exception:
                city_hospitals[c] = []

    return {"entities": deduped, "cities": found_cities, "city_hospitals": city_hospitals,
            "measures": measures, "grains": grains}


def brief(question: str) -> str:
    """The resolution, as a block to put in front of any model that is about to write SQL."""
    r = resolve(question)
    if not (r["entities"] or r["cities"] or r["measures"]):
        return ""
    lines = ["WHAT THIS QUESTION IS ABOUT (resolved against the actual dimension values — "
             "use these exact columns, do not guess where a name lives):"]
    for e in r["entities"]:
        where = ", ".join(e.get("locations") or [f"{e['table']}.{e['column']}"])
        lines.append(f"- \"{e['text']}\" is a {e['kind'].upper()}, held in {where}"
                     + ("" if e["exact"] else f" (confidence {e['confidence']})"))
    for c in r["cities"]:
        sites = r["city_hospitals"].get(c) or []
        which = (" It covers " + ", ".join(sites) + ".") if sites else ""
        lines.append(f"- \"{c}\" is a CITY. Cities appear ONLY inside dim_plant.plant_name, "
                     f"which no sales table can reach.{which}")
    if any(len(e.get("locations") or []) > 1 for e in r["entities"]):
        lines.append("- Spelling and CASE differ between tables. Filter with "
                     "upper(col) LIKE upper('value'), never col = 'value'.")
    if r["measures"]:
        lines.append(f"- MEASURE asked for: {', '.join(r['measures'])}")
    if r["grains"]:
        lines.append(f"- BROKEN DOWN BY: {', '.join(r['grains'])}")
    return "\n".join(lines)
