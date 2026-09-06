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
# A token matching more than this many values is a fragment, not a name.
_FAMILY_MAX = 12

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
    # Ordinary English carries no identifying power either, and family matching made that
    # expensive: "how many units are expiring in the next 90 days" bound "next" to a
    # guidewire called GAIA NEXT 3 and a vendor called Next Radio Ltd. A stoplist is the
    # standard remedy — these are words a question uses to ask, not to name.
    "next", "last", "previous", "prior", "recent", "current", "past", "coming", "upcoming",
    "before", "after", "during", "since", "until", "within", "about", "around", "near",
    "same", "other", "another", "such", "some", "more", "less", "fewer", "only", "still",
    "here", "been", "being", "make", "makes", "made", "take", "takes", "come", "goes",
    "going", "look", "looks", "need", "needs", "want", "wants", "know", "think", "seem",
    "well", "good", "bad", "long", "short", "high", "low", "wide", "deep", "fast", "slow",
    "new", "old", "next-", "year", "years", "month", "months", "week", "weeks", "days",
    "day", "time", "times", "date", "dates", "period", "periods", "level", "levels",
    "thing", "things", "part", "parts", "kind", "kinds", "type", "types", "case", "cases",
    "data", "record", "records", "row", "rows", "table", "tables", "report", "reports",
    # Evaluative adjectives say how much something MATTERS. They never name it — and
    # "which critical items do we buy from one vendor" was answered with CRITICAL REGISTER
    # A4 200 PAGE and CHART-CRITICAL CARE UNIT, at ₹2,700.
    "critical", "essential", "important", "vital", "key", "core", "main", "primary",
    "strategic", "risky", "problem", "problematic", "urgent",
    "line", "lines", "side", "sides", "way", "ways", "point", "points", "area", "areas",
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
# How a measure must be EXPRESSED once entities are being compared. "Which hospitals run the
# thinnest sales margins?" was answered "MPRAT, ₹27,012" — the smallest absolute margin,
# which is just the smallest hospital. Thinness is a rate, and ranking a rate by its
# numerator is a category error no amount of correct SQL will catch.
_MEASURE_NOTES: dict[str, str] = {
    "margin": ("Margin compared ACROSS entities must be a PERCENTAGE — (revenue - cost) / "
               "revenue * 100. An absolute margin ranks by size, so \"thinnest margin\" "
               "would return the smallest site rather than the least profitable one. And "
               "because a percentage on a tiny base is noise, rank margins only above an "
               "explicit revenue floor and SAY what floor you used: -40.1% on \u20b92,927 of "
               "revenue is a rounding error wearing the clothes of a finding."),
    "price":  ("Price compared across items must be per-unit. Summing price across rows is "
               "meaningless — it adds rates, not amounts."),
    "expiry": ("This warehouse buckets expiry as Expired / 0-30d / 31-90d / 91-180d. Stock "
               "that has ALREADY expired is not \"expiring in the next N days\" — it has "
               "expired. A 90-day question is 0-30d + 31-90d only; adding the expired "
               "bucket turns 45,223 units into 101,005. State whether expired stock is "
               "included either way, because both totals are quoted on the dashboard and "
               "the reader cannot tell which one they are looking at."),
    "lead_time": ("Lead time is an average already computed per vendor. Average it with a "
                  "weight, or say it is an unweighted mean of vendor-level figures."),
}

# Words that restrict WHICH ROWS QUALIFY before anything is ranked. "Which high-value drugs
# are we making the worst margin on?" was answered with a ₹2,927 drug: the qualifier was
# dropped, and the worst margin in the whole catalogue is always some near-zero line.
_QUALIFIERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("high-value / large", ("high-value", "high value", "highvalue", "expensive", "costly",
                            "major", "significant", "large", "big-ticket", "top-value")),
    ("low-value / small",  ("low-value", "low value", "cheap", "inexpensive", "minor", "small")),
    ("critical / essential", ("critical", "essential", "important", "vital", "key",
                             "strategic", "core")),
    ("fast-moving",        ("fast-moving", "fast moving", "high-volume", "high volume")),
    ("slow-moving",        ("slow-moving", "slow moving", "non-moving", "non moving", "dead")),
)

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
    def _asks_rather_than_names(t: str) -> bool:
        # the vocabulary lists "item"; the question says "items". Comparing the singular too
        # is what stops "which items do we buy from MSD" resolving "items" to a product.
        stem = t[:-1] if t.endswith("s") and len(t) > 3 else t
        return any(w in bag for w in (t, stem) for bag in (_STOP, _NOISE_TOKENS, intent))

    interesting = [t for t in qtokens if len(t) >= 3 and not _asks_rather_than_names(t)]

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

    # FAMILIES. "What is Vardhman's average lead time?" resolved to nothing and was answered
    # "there is no lead time figure for Vardhman" — while `vendor_avg_lead_time_days` held
    # 4.8 days over 128,357 rows. "Vardhman" covers 1 of the 3 tokens in "Vardhman Health
    # Specialities", so the coverage rule that stops "MSD" reaching STICKER-MSDS rejected it.
    #
    # The distinguishing fact is DOCUMENT FREQUENCY. "MSD" is a fragment of many unrelated
    # values; "vardhman" names a handful of vendors and nothing else. A rare token that
    # matches a small set is a real reference to that SET — which is also the honest answer,
    # because a company with five trading names is five rows, not one. Report the family and
    # the LIKE filter that covers it rather than silently picking its biggest member.
    claimed = {t.lower() for e in deduped for t in _WORD.findall(e["text"])}
    schema_words = _schema_vocabulary()
    families: list[dict] = []
    for tok in interesting:
        # A word the SCHEMA uses to describe things is descriptive vocabulary, not a name.
        # "which high-value drugs have the worst margins" formed a family on "value" and was
        # answered about HIGH VALUE DRUG STICKERS -GEN — the STICKER-MSDS failure again,
        # reintroduced by family matching. The warehouse settles it: "value" appears in
        # line_value, purchase_value, billed_value; "vardhman" appears in no column name.
        if tok in claimed or tok in schema_words or len(tok) < 4:
            continue
        cands = postings.get(tok, ())
        if not (1 <= len(cands) <= _FAMILY_MAX):
            continue
        by_col: dict[tuple[str, str, str], list[str]] = defaultdict(list)
        for kind, value, table, column in cands:
            by_col[(kind, table, column)].append(value)
        for (kind, table, column), values in by_col.items():
            same = next((f for f in families
                         if f["token"] == tok and f["kind"] == kind
                         and f["examples"] == sorted(values)[:6]), None)
            if same:                      # same family, another table holds it too
                same["also_in"].append(f"{table}.{column}")
                continue
            families.append({"token": tok, "kind": kind, "table": table, "column": column,
                             "also_in": [], "n": len(values), "examples": sorted(values)[:6]})
        if len(families) >= limit:
            break

    # A word that names a CLASS is being used as a class. "how many tablets do we stock"
    # matched the category M113-TABLETS and, by coincidence, a manufacturer called TABLETS
    # INDIA — which is how that question used to be answered about one supplier. Neither
    # rarity nor substring frequency separates "tablets" from "vardhman" (both appear in
    # exactly 5 values); what separates them is that the warehouse itself files one as a
    # category label and has no such label for the other. So a category match wins outright
    # for its token. The genuine "how much did we buy from Tablets India" still works,
    # because two matching tokens make it an ENTITY, and families only form from tokens no
    # entity claimed.
    by_token: dict[str, list[dict]] = defaultdict(list)
    for f in families:
        by_token[f["token"]].append(f)
    families = [f for tok, fs in by_token.items()
                for f in (([c for c in fs if c["kind"] == "category"] or fs))]

    # LOOKALIKES. "RELIANCE" resolves cleanly and only as a MANUFACTURER — and the engine
    # still answered "Reliance is a supplier we buy from directly", because it ran its own
    # LIKE '%RELIANCE%' and found "Reliance Pharmaceutical Agencies" and "Reliance Office
    # Mart" in vendor_name. Those are different companies. A substring match is not identity,
    # and the model cannot know that from a list of values it retrieved itself — so the
    # near-misses are named here, with their kind, before it goes looking.
    lookalikes: list[dict] = []
    for e in deduped[:3]:
        if not e["exact"]:
            continue
        target = e["text"].upper()
        toks = [t.lower() for t in _WORD.findall(e["text"])
                if len(t) >= 4 and t.lower() not in _NOISE_TOKENS]
        others: dict[tuple[str, str, str], set] = defaultdict(set)
        for tok in toks:
            for kind, value, table, column in postings.get(tok, ()):
                if kind != e["kind"] and value.upper() != target:
                    others[(kind, table, column)].add(value)
        for (kind, table, column), values in others.items():
            lookalikes.append({"of": e["text"], "kind": kind, "table": table,
                               "column": column, "examples": sorted(values)[:4],
                               "n": len(values)})
    lookalikes = lookalikes[:6]

    found_cities = [c for c in cities() if re.search(rf"\b{re.escape(c)}\b", q, re.I)]
    low = q.lower()
    # WORD boundaries, not substrings. "Which hospital generates the most sales revenue?"
    # was typed as a PRICE question because "generates" contains "rate" — and the brief then
    # told the model, in its most authoritative voice, to think about per-unit pricing. Same
    # for "corporate", "operating", "separate".
    def _asks(words) -> bool:
        # a trailing "s" is allowed so "items" still matches "item" — dropping it cost the
        # material grain on every question that used the plural, which is most of them
        return any(re.search(rf"(?<![a-z]){re.escape(w)}s?(?![a-z])", low) for w in words)

    def _where(words) -> int:
        hits = [m.start() for w in words
                for m in [re.search(rf"(?<![a-z]){re.escape(w)}s?(?![a-z])", low)] if m]
        return min(hits) if hits else 10 ** 6

    measures = [name for name, words in _MEASURES if _asks(words)]
    # Order grains by where the question mentions them. "Which critical ITEMS do we buy from
    # only one VENDOR?" is a question about items; declaration order put vendor first and the
    # size threshold was then computed per vendor — ₹2.86 Cr, which no single item reaches.
    # The subject of a question is almost always the grain it names first.
    grains = sorted((n for n, words in _GRAINS if _asks(words)),
                    key=lambda n: _where(dict(_GRAINS)[n]))
    qualifiers = [name for name, words in _QUALIFIERS if _asks(words)]

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

    return {"entities": deduped, "families": families, "lookalikes": lookalikes,
            "cities": found_cities, "city_hospitals": city_hospitals, "measures": measures,
            "grains": grains, "qualifiers": qualifiers}


def brief(question: str) -> str:
    """The resolution, as a block to put in front of any model that is about to write SQL."""
    r = resolve(question)
    asks_about_time = bool(re.search(
        r"\b(period|window|range|cover|covers|coverage|timeframe|time frame|history)\b",
        question or "", re.I))
    if not (r["entities"] or r["families"] or r["cities"] or r["measures"] or r["grains"]
            or asks_about_time):
        return ""
    lines = ["WHAT THIS QUESTION IS ABOUT (resolved against the actual dimension values — "
             "use these exact columns, do not guess where a name lives):"]
    for e in r["entities"]:
        where = ", ".join(e.get("locations") or [f"{e['table']}.{e['column']}"])
        lines.append(f"- \"{e['text']}\" is a {e['kind'].upper()}, held in {where}"
                     + ("" if e["exact"] else f" (confidence {e['confidence']})"))
    for la in r["lookalikes"]:
        ex = ", ".join(la["examples"])
        lines.append(
            f"- CAUTION: {la['table']}.{la['column']} contains {la['n']} {la['kind'].upper()} "
            f"value(s) whose names merely CONTAIN \"{la['of']}\" — {ex}. Those are DIFFERENT "
            f"entities, not \"{la['of']}\" wearing another hat. A substring match is not "
            f"identity. \"{la['of']}\" itself is a {r['entities'][0]['kind'].upper() if r['entities'] else 'n/a'}, "
            f"and if the question asks what {la['of']} IS, that is the answer.")
    for f in r["families"]:
        ex = ", ".join(f["examples"][:4]) + (" …" if f["n"] > 4 else "")
        lines.append(
            f"- \"{f['token']}\" is not one value: it matches {f['n']} {f['kind'].upper()} "
            f"value(s) in {f['table']}.{f['column']} — {ex}. Cover the whole family with "
            f"upper({f['column']}) LIKE '%{f['token'].upper()}%', and say which ones you "
            f"included. Do not silently answer for just the biggest."
            + (f" Also in {', '.join(f['also_in'])}." if f["also_in"] else ""))
    for c in r["cities"]:
        sites = r["city_hospitals"].get(c) or []
        codes = [s.split(" ")[0] for s in sites]
        if codes:
            lst = ", ".join(f"'{x}'" for x in codes)
            reach, blocked = city_reachability()
            usable = [t for t in reach if not t.startswith("_pydf")][:6]
            lines.append(
                f"- \"{c}\" is a CITY, not a hospital: it covers {', '.join(sites)}. City "
                f"NAMES exist only in dim_plant.plant_name, so filter by CODE — "
                f"plant IN ({lst}) — and never join a fact table to dim_plant just to reach "
                f"the name, which fans rows out and inflates every total.")
            if usable:
                lines.append(f"  Those codes are valid in: {', '.join(usable)}.")
            if blocked:
                lines.append(
                    f"  They are NOT valid in {', '.join(blocked)} — those columns use a "
                    f"DIFFERENT site code system that shares no value with dim_plant, so no "
                    f"SALES figure can be filtered to a city at all. If the question asks "
                    f"for sales by city, say that the data cannot answer it rather than "
                    f"producing a number for some other scope.")
        else:
            lines.append(f"- \"{c}\" is a CITY. Cities appear ONLY inside "
                         f"dim_plant.plant_name, which no sales table can reach.")
    if any(len(e.get("locations") or []) > 1 for e in r["entities"]):
        lines.append("- Spelling and CASE differ between tables. Filter with "
                     "upper(col) LIKE upper('value'), never col = 'value'.")
    for m in r["measures"]:
        # naming the measure is not locating it — say which columns actually hold it
        where = measure_locations(m)
        lines.append(f"- MEASURE asked for: {m}"
                     + (f" — stored in {', '.join(where)}" if where
                        else " — no column in this warehouse stores it"))
        if _MEASURE_NOTES.get(m):
            lines.append(f"  {_MEASURE_NOTES[m]}")
    if r["grains"]:
        lines.append(f"- BROKEN DOWN BY: {', '.join(r['grains'])}")
    if "month" in r["grains"] or re.search(
            r"\b(period|window|range|cover|covers|coverage|timeframe|time frame|history)\b",
            question or "", re.I):
        win = reporting_window()
        if win:
            lines.append(f"- REPORTING PERIOD: {win}")
    for qual in r["qualifiers"]:
        # Hand over the actual number. Asked to "choose a threshold at the top of the
        # distribution" the model chose the median, then Rs 1 lakh; both admit thousands of
        # items and neither is what high-value means.
        cut = None
        if (qual.startswith("high-value") or qual.startswith("critical")) and r["grains"]:
            base = "purchasing" if "purchasing" in r["measures"] else "revenue"
            cut = size_floor(base, r["grains"][0])
        if cut:
            col, floor, n = cut
            lines.append(
                f"- QUALIFIER \"{qual}\": use {col} >= {floor:,.0f}. That is this "
                f"warehouse's own Class-A cut — the {n} largest carry 70% of the total — "
                f"so it is the threshold to filter on and to state in the answer. Do NOT "
                f"pick your own: a median split calls half the catalogue high-value.")
            continue
        lines.append(
            f"- QUALIFIER \"{qual}\": this restricts WHICH ROWS QUALIFY before ranking — it "
            f"is NOT the thing being ranked. Apply it as an explicit threshold in the WHERE "
            f"clause, and state the threshold in the answer. Ranking without it returns the "
            f"smallest rows in the catalogue every time. The threshold must sit at the TOP "
            f"of the distribution — the top decile, or a round absolute floor an executive "
            f"would recognise. A MEDIAN split is not it: calling the upper half "
            f"\"high-value\" admits almost everything and changes nothing.")
    return "\n".join(lines)


# ── WHERE A MEASURE LIVES ────────────────────────────────────────────────────────────────
# Naming a measure is not the same as locating it. "What is Vardhman's average lead time?"
# resolved cleanly — Vardhman is a VENDOR, the measure is lead_time — and was still answered
# "there is no lead time figure for Vardhman", because nothing said lead time is stored in a
# column called `vendor_avg_lead_time_days`. The value was 4.8 days across 128,357 rows.
#
# So measures are resolved against the real schema exactly as entity names are resolved
# against real values: by matching the actual column names, not by hoping the model guesses
# a column it has never been shown. Adding a column later needs no change here.
_MEASURE_COLUMNS: dict[str, tuple[str, ...]] = {
    "margin":      (r"margin", r"profit"),
    "revenue":     (r"revenue", r"sales_value", r"net_sales", r"^sales$", r"turnover",
                    r"billed_value", r"^value$"),
    "purchasing":  (r"line_value", r"po_value", r"purchase_value", r"spend", r"net_value",
                    r"grn_value"),
    "consumption": (r"consum", r"issued", r"usage"),
    "stock":       (r"stock", r"on_hand", r"closing", r"inventory_value", r"total_cost"),
    "expiry":      (r"expiry", r"expir", r"shelf"),
    "lead_time":   (r"lead_time", r"lead_months", r"turnaround"),
    "quantity":    (r"^qty", r"quantity", r"^units", r"_units$", r"volume"),
    "price":       (r"price", r"rate", r"unit_cost", r"mrp"),
}

# What a table actually MEASURES, regardless of what the question called it. `fact_consumption`
# holding a big number for LEAFLET A5 is 1,203,000 units CONSUMED; reporting it as "units sold"
# is a different claim about a different business event, and LEAFLET A5 has no sales rows at
# all. The verb has to come from the source, not from the question's phrasing.
_TABLE_EVENT: list[tuple[str, str, str]] = [
    (r"^sales|_sales|sales_",  "SALES",       "sold / billed to a patient or payer"),
    (r"consum",                "CONSUMPTION", "issued or used internally — NOT sold"),
    (r"procure|^fact_po|^fact_grn|purchase", "PROCUREMENT", "bought from a vendor — NOT sold"),
    (r"inventor|stock|expiry|aging|near_expiry", "STOCK", "held on hand — NOT a flow"),
]


@lru_cache(maxsize=1)
def _schema_columns() -> tuple[tuple[str, str], ...]:
    from app.ai import warehouse
    try:
        rows = warehouse.con().execute(
            "SELECT table_name, column_name FROM information_schema.columns").fetchall()
    except Exception:
        return ()
    return tuple((str(t), str(c)) for t, c in rows)


@lru_cache(maxsize=256)
def measure_locations(measure: str, limit: int = 6) -> tuple[str, ...]:
    """Real `table.column` places a measure is actually stored. Empty if genuinely absent."""
    pats = _MEASURE_COLUMNS.get(measure) or ()
    if not pats:
        return ()
    rx = re.compile("|".join(pats), re.I)
    hits = [f"{t}.{c}" for t, c in _schema_columns() if rx.search(c)]
    # a mart beats a raw fact beats a pre-aggregated KPI, so the first ones offered are the
    # ones a person would actually reach for
    hits.sort(key=lambda h: (h.startswith("kpi_"), len(h)))
    return tuple(hits[:limit])


def event_of(table: str) -> tuple[str, str]:
    """('CONSUMPTION', 'issued or used internally — NOT sold') for a table name."""
    for pat, kind, gloss in _TABLE_EVENT:
        if re.search(pat, table or "", re.I):
            return kind, gloss
    return "", ""


def source_vocabulary(tables) -> str:
    """The verbs the evidence actually licenses, given the tables it came from.

    Injected at synthesis time so wording is bound to the source. Without it the engine
    answered "LEAFLET A5 SIZE SINGLE SIDE PRINT-GEN, 1,203,000 units sold" from
    `fact_consumption` — a product with zero rows in any sales table.
    """
    seen: dict[str, str] = {}
    for t in tables or ():
        kind, gloss = event_of(str(t))
        if kind:
            seen.setdefault(kind, gloss)
    if not seen:
        return ""
    lines = ["WORDS YOUR EVIDENCE LICENSES (the verb must match the source, not the "
             "question's phrasing):"]
    lines += [f"- {k}: {g}" for k, g in seen.items()]
    if "SALES" not in seen:
        lines.append("- No sales table was queried. Do NOT write \"sold\", \"sales\" or "
                     "\"revenue\" about these numbers; say what they actually are.")
    return "\n".join(lines)


@lru_cache(maxsize=1)
def _schema_vocabulary() -> frozenset[str]:
    """Words the warehouse uses to NAME things — its own descriptive vocabulary.

    Derived from table and column names rather than listed by hand, so a column added later
    widens it automatically. This is what separates "value" (line_value, purchase_value,
    billed_value) from "vardhman" (no column anywhere).
    """
    words: set[str] = set()
    for table, column in _schema_columns():
        for part in re.split(r"[^a-z]+", f"{table} {column}".lower()):
            if len(part) >= 4:
                words.add(part)
    return frozenset(words)


# ── HOW BIG IS "BIG" ─────────────────────────────────────────────────────────────────────
# Which columns identify a thing AT a grain. One definition, used by the deep engine to
# check that a finding answers at the level asked for, and here to find the table a size
# threshold should be computed from.
GRAIN_COLUMNS: dict[str, re.Pattern] = {
    # `name` is deliberately absent from `material`: the units-per-SKU KPI has a column
    # literally called `name` holding CATEGORY labels, so accepting it let "which products
    # move the most units" answer "M070-STATIONARY" and still look correct.
    "material":     re.compile(r"^(material|material_id|material_desc|generic_name|item|sku)$", re.I),
    "hospital":     re.compile(r"^(hospital|plant|plant_name|site)$", re.I),
    "vendor":       re.compile(r"^(vendor|vendor_name|vendor_code)$", re.I),
    "manufacturer": re.compile(r"^(manufacturer|manufacturer_desc)$", re.I),
    "category":     re.compile(r"^(category|material_group|major_group_desc|minor_group_desc|group|name)$", re.I),
    "month":        re.compile(r"^(month|month_name|period|posting_date|year)$", re.I),
    "department":   re.compile(r"^(department|department_name|cost_ctr|costcenter)$", re.I),
}


def _table_columns(table: str) -> list[str]:
    return [c for t, c in _schema_columns() if t == table]


@lru_cache(maxsize=64)
def size_floor(measure: str, grain: str, cover: float = 0.70):
    """The value above which rows are "big", computed from the data rather than invented.

    Told only to use a threshold at the top of the distribution, the model picked the MEDIAN
    (half the catalogue), then Rs 1 lakh (still thousands of items) — and answered "which
    high-value drugs have the worst margin" with a Rs 2,927 line either way. So the number is
    computed here instead: the classic ABC Class-A cut, the floor at which qualifying rows
    account for `cover` of the measure. On sales_by_material that is ~150 items carrying
    ~63% of revenue, which is what an executive means by high-value.

    Returns (table.column, floor, n_items), or None when no table serves the grain.
    """
    from app.ai import warehouse
    pat = GRAIN_COLUMNS.get(grain)
    if not pat:
        return None
    for loc in measure_locations(measure, limit=12):
        table, column = loc.split(".", 1)
        if table.startswith("kpi_"):
            continue                      # pre-aggregated: not the population to cut
        key = next((c for c in _table_columns(table) if pat.match(c)), None)
        if not key:
            continue
        try:
            # Aggregate to the grain FIRST. mart_procurement is one row per PO line, so
            # cutting it raw ranks lines, not vendors — 14,194 "big vendors" out of 2,251.
            row = warehouse.con().execute(
                'WITH per_entity AS ('
                f' SELECT "{key}" AS k, SUM("{column}") AS v FROM "{table}"'
                f' WHERE "{column}" > 0 GROUP BY 1),'
                ' ranked AS ('
                ' SELECT v, SUM(v) OVER (ORDER BY v DESC'
                '  ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS run,'
                ' SUM(v) OVER () AS tot FROM per_entity)'
                f' SELECT MIN(v), COUNT(*) FROM ranked WHERE run <= tot * {cover}'
            ).fetchone()
        except Exception:
            continue
        if row and row[0] is not None and row[1]:
            return f"{table}.{column}", float(row[0]), int(row[1])
    return None


@lru_cache(maxsize=1)
def city_reachability() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(reachable, unreachable) site columns, decided by whether their codes actually match.

    A city is a property of dim_plant, so filtering anything by city means matching plant
    CODES. Whether that works is a fact about the data, not about the column's name:
    mart_procurement.plant shares 50 of 51 codes with dim_plant, while
    sales_by_hospital.hospital shares ZERO — it uses a different code system entirely
    (APONG, GJHCA, MHHIK). Told simply to "filter by code", the engine dutifully produced
    "KEYTRUDA, ₹14.38 Cr in Bangalore hospitals" from a table no city can reach.

    Returns column paths, so the brief can name exactly where a city filter is legitimate.
    """
    from app.ai import warehouse
    try:
        codes = {r[0] for r in warehouse.con().execute("SELECT plant FROM dim_plant").fetchall()}
    except Exception:
        return (), ()
    if not codes:
        return (), ()
    ok, no = [], []
    site = GRAIN_COLUMNS["hospital"]
    for table, column in _schema_columns():
        if table == "dim_plant" or not site.match(column):
            continue
        try:
            vals = {r[0] for r in warehouse.con().execute(
                f'SELECT DISTINCT "{column}" FROM "{table}"').fetchall() if r[0]}
        except Exception:
            continue
        if not vals:
            continue
        (ok if len(vals & codes) else no).append(f"{table}.{column}")
    return tuple(sorted(ok)), tuple(sorted(no))


@lru_cache(maxsize=1)
def reporting_window() -> str:
    """The period the TRANSACTIONS cover, which is not the range of every date in the data.

    Asked "what period does our data cover", the engine scanned every column holding a date
    and answered "2020-01-31 to 2026-12-03" — a range built from expiry dates and forecast
    horizons. The transactional window is Dec 2025 to May 2026, and everything a reader
    infers about trend, growth or seasonality depends on knowing that.
    """
    from app.ai import warehouse
    for table, column in (("fact_consumption", "posting_date"), ("fact_grn", "posting_date"),
                          ("sales_monthly", "month")):
        try:
            lo, hi = warehouse.con().execute(
                f'SELECT MIN("{column}"), MAX("{column}") FROM "{table}"').fetchone()
        except Exception:
            continue
        if lo is None or hi is None:
            continue
        return (f"{str(lo)[:10]} to {str(hi)[:10]} (from {table}.{column}). Expiry dates and "
                f"forecast horizons run past this and do NOT extend the reporting period; "
                f"neither does any earlier date sitting in a master table.")
    return ""
