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

import calendar
import copy
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
    # The sales tables use their OWN site codes (APONG, GJHCA, MHHIK) which share no value
    # with dim_plant. They were never indexed, so all 23 of them resolved to nothing —
    # "how much revenue did GJHCA generate" was an unanswerable question about a real site.
    ("hospital",     "sales_by_hospital", "hospital",        200),
    ("hospital",     "sales_by_material_hospital", "hospital", 200),
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
# Business words of HCG's reports. None is a misspelling of a product: "saving" was offered
# back as "shaving" (shaving blades) and the model answered about razors (golden
# fs-kr-scheme-saving-aug, 10-10-2026).
_BUSINESS_WORDS = {
    "saving", "savings", "saved", "save", "missed", "miss", "loss", "losses", "lost", "lose",
    "margin", "margins", "scheme", "schemes", "attis", "healthium", "restricted", "preferred",
    "formulary", "compliance", "compliant", "breach", "breaches", "brand", "brands", "tpa",
    "cash", "payer", "payers", "gov", "government", "doctor", "doctors", "consultant",
    "consultants", "onco", "oncology", "segment", "segments", "suggested", "suggestion",
    "alternative", "alternatives", "sharing", "bifurcation", "rule", "rules", "network",
    "pooled", "potential", "proposed", "spend", "spent", "share", "patients", "patient",
    "inpatient", "outpatient", "consumption", "hospital", "hospitals",
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
    # Common verbs, or a fuzzy matcher offers "generate" -> "generatr" off some product
    # description and calls it a spelling correction.
    "generate", "generates", "generated", "generating", "produce", "produced", "provide",
    "deliver", "delivered", "return", "returns", "returned", "include", "includes",
    "compare", "compared", "increase", "increased", "decrease", "decreased", "reduce",
    "reduced", "improve", "improved", "change", "changed", "happen", "happened",
    "cover", "covers", "covered", "covering", "coverage", "record", "recorded",
    "show", "shows", "shown", "given", "gives", "spent", "spend", "hold", "holds",
    "quarter", "quarters", "annual", "yearly", "monthly", "weekly", "daily", "trend",
    "trends", "overall", "average", "median", "today", "yesterday", "tomorrow",
    "line", "lines", "side", "sides", "way", "ways", "point", "points", "area", "areas",
    # Words that say how WIDE the answer is or how to CUT it. "Show the network-wide monthly
    # sales trend" bound "network" to four D-LINK / HPE network switches, and the brief then
    # handed over a monthly query filtered to them as "this product". "month wise" bound
    # "wise" to products whose names contain WISE, worth ₹75 L of sales. These words ask;
    # they never name.
    "network", "networks", "networkwide", "company", "companywide", "organisation",
    "organization", "business", "whole", "entire", "wise", "split", "splits", "breakdown",
    "against", "combined", "together", "separately", "chain",
}
_MAX_DF = 250          # a token in more than this many values identifies nothing

# How many values a token may match and still be treated as naming a family.
#
# This was 12, which rejected exactly the cases it exists to serve: "vicryl" matches 77
# suture SKUs, "sutures" 66, "catheters" 38, "pentasure" 17 — every one a precise
# identification of a product family, all discarded for being too successful. The index
# has already dropped anything above _MAX_DF as non-identifying, so a token that survives
# indexing is by construction distinctive enough to name something.
_FAMILY_MAX = _MAX_DF


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
    # Drop postings that cannot identify anything — but decide that PER KIND. "ROCHE"
    # appears in 272 material descriptions (every "-ROCHE" suffix) and in exactly ONE
    # manufacturer value. Filtering the token globally deleted the manufacturer posting
    # too, so a major pharma company was invisible precisely BECAUSE it is big. A token
    # that is noise as a product name can still be a perfect manufacturer name.
    kept: dict[str, list[tuple]] = {}
    for tok, plist in postings.items():
        by_kind: dict[str, list[tuple]] = defaultdict(list)
        for post in plist:
            by_kind[post[0]].append(post)
        survivors = [post for kind, group in by_kind.items() if len(group) <= _MAX_DF
                     for post in group]
        if survivors:
            kept[tok] = survivors
    return (kept, exact)


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
    "purchasing": ("Procurement spend is EX-TAX. The <<PROCUREMENT_TOTAL>> figure every dashboard "
                   "shows is fact_po.total_value_wo_tax; GST sits beside it in cgst_value / "
                   "sgst_value / igst_value (<<PROCUREMENT_GST>>) and fact_grn carries "
                   "total_amount_with_tax on the receipt side. Do NOT report a with-tax "
                   "column as procurement spend, and never add tax to the ex-tax total "
                   "unless the question asks for the inclusive figure."),
    "price":  ("Price compared across items must be per-unit. Summing price across rows is "
               "meaningless — it adds rates, not amounts."),
    # A CALLABLE, not a string. The counts here are facts about the data, and writing them
    # into the prompt as literals means the model is told stale numbers the moment the
    # parquet is refreshed — the same "hand-maintained list" failure that hid the disjoint
    # hospital codes. Everything countable is counted at call time.
    "consumption": lambda: (
        "USE `consumption_all` — it is the only table that holds both scopes. Consumption "
        "here is TWO events: `fact_consumption` is materials ISSUED FROM STORES "
        f"({_count_distinct('fact_consumption', 'material')} materials), and materials "
        "dispensed against a patient's bill are not in it at all. consumption_all unions "
        "them at material grain with a `scope` column ('internal' or 'billed'), covering "
        f"{_count_distinct('consumption_all', 'material')} materials. Always report which "
        "scope a figure is, and never read an empty fact_consumption result as proof an "
        "item is unused. NOTE: consumption_all has no plant and no date. For a MONTHLY "
        "trend take the billed side from sales_by_material_month (SUM(qty), SUM(cost) by "
        "period) and the internal side from fact_consumption / kpi_units_consumed, and "
        "report them as two labelled series. A per-plant billed split does not exist "
        "(sales hospital codes do not map to plants)."),
    "expiry": ("This warehouse buckets expiry as Expired / 0-30d / 31-90d / 91-180d. Stock "
               "that has ALREADY expired is not \"expiring in the next N days\" — it has "
               "expired. A 90-day question is 0-30d + 31-90d only; adding the expired "
               "bucket roughly doubles the figure. State whether expired stock is "
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

# Where a grain word means more than one real column, and picking silently is how two
# defensible answers to the same question get produced on different runs.
_GRAIN_NOTES: dict[str, str] = {
    "category": ("\"Category\" is THREE different taxonomies here, and they give different "
                 "answers: dim_material.material_group is the SUPPLY form (139 values — "
                 "M065-INJECTIONS, M113-TABLETS); major_group_desc is the THERAPEUTIC class "
                 "(1,221 — ANTINEOPLASTIC, ALKYLATING AGENT); minor_group_desc is the "
                 "MOLECULE (3,237 — ABIRATERONE). Pick the one the question means, and name "
                 "which you used. For clinical or spend questions the therapeutic class is "
                 "usually intended; for stores and inventory, the supply form."),
}

# Notes tied to a WORD in the question rather than to its grain. The formulary warning was
# briefly attached to the material grain, which meant it appeared on "what is our biggest
# selling product" — true, irrelevant, and one more line to read past.
#
# Two different things answer to the word "formulary", and the old note sent every one of
# them to the item-master flag. Formulary COMPLIANCE, restricted/preferred brands, scheme and
# Attis savings and cash & TPA margin loss are the Jun–Aug 2026 HCG reports, which have no
# SQL table at all — so a compliance question answered from dim_material.formulary_status is
# a different measure over a different window. The flag stays the answer for "is item X on
# the formulary" and the Dec–May sales share by formulary status.
_REPORTS_NOTE = (
    "Formulary COMPLIANCE %, out-of-formulary spend, restricted/preferred brand use, "
    "scheme/Attis/Healthium savings and cash & TPA margin loss are <<REPORTS>> reports "
    "reachable ONLY through get_kpi keys formulary-compliance, brand-rules, "
    "scheme-brand-saving, attis-saving, cash-tpa-margin-loss — there is no SQL table for "
    "them, and they do not overlap the <<SALES>> sales window.")

_KEYWORD_NOTES: tuple[tuple[str, str], ...] = (
    (r"\bformular",
     _REPORTS_NOTE + " dim_material.formulary_status is only for \"is item X on the "
     "formulary\" and the <<SALES_SHORT>> sales share by formulary status. For those: "
     "use dim_material.formulary_status, NOT `formulary`. The raw column spells three "
     "concepts eleven ways (NON FORMULARY, NON FORMUL, NON-FORMULARY, NON FORMUALRY, "
     "NONFORMULARY, NON  FORMULARY, OUT OF FOR …), so an equality filter on it silently "
     "drops 123 items and a GROUP BY returns eleven buckets for three things. The clean "
     "values are FORMULARY (4,024), NON FORMULARY (7,090), OUT OF FORMULARY (3,504), "
     "UNSPECIFIED (10,313)."),
    # "Restricted brands for cash and TPA patients" was answered with the cash & TPA MARGIN
    # LOSS (₹78.08 L for KR, Aug-26) instead of the SPEND on restricted brands (₹4.52 Cr):
    # the words "cash and TPA" pulled it to the wrong report (golden
    # fs-kr-restricted-cash-tpa-aug, 10-10-2026).
    (r"\b(?:restricted|preferred)\b.*\b(?:cash|tpa|scheme|gov|government)\b"
     r"|\b(?:cash|tpa|scheme|gov|government)\b.*\b(?:restricted|preferred)\b",
     "WHICH REPORT: spend on RESTRICTED or PREFERRED brands is get_kpi('brand-rules') with "
     "measure = restricted_cash_tpa (cash & TPA patients), restricted_scheme or preferred_scheme "
     "(scheme / GOV patients): their cost and share of all drug spend. It is NOT "
     "get_kpi('cash-tpa-margin-loss'), a different figure: the margin lost when a cash or TPA "
     "patient got a brand HCG's list suggests replacing."),
    # The same reports asked for without the word "formulary".
    (r"(?<!formulary )\bcompliance\b|\b(?:restricted|preferred) brands?\b|\bbrand[- ]rules?\b"
     r"|\battis\b|\bhealthium\b|\bscheme (?:brand )?sav|\bcash\s*(?:&|and)\s*tpa\b",
     _REPORTS_NOTE),
    # "OP sales of keytruda" resolved nothing about IP/OP — the tokens are two letters, below
    # every matching threshold — so the model reported the combined figure and the prose
    # called it OP. The split exists at exactly two levels, and dim_sloc's ward class is a
    # different thing that only looks like it.
    (r"\b(?:ip|op|ipd|opd|in-?patients?|out[- ]?patients?)\b",
     "IP/OP split exists ONLY company-wide for <<SALES_SHORT>> (sales_totals.patient and "
     "sales_monthly.patient, which has the month too) and in the <<REPORTS>> report tools "
     "(get_kpi patient_type). No <<SALES_SHORT>> item, category or hospital table has IP/OP: for "
     "those give the combined figure and say the IP/OP split is not available at that "
     "level. dim_sloc.ip_op_class is the STORE's ward type (mostly OTHER) — never use it as "
     "the patient's IP/OP."),
    (r"\btax\b|\bgst\b|\bcgst\b|\bsgst\b|\bigst\b|\bduty\b",
     "Tax is SEPARATE from spend. Procurement spend is EX-TAX by definition — the "
     "<<PROCUREMENT_TOTAL>> figure is fact_po.total_value_wo_tax — and GST sits beside it in "
     "cgst_value / sgst_value / igst_value (<<PROCUREMENT_GST>>). fact_grn carries tax_amount, tax_pct and "
     "total_amount_with_tax on the receipt side. NEVER add tax to the spend figure or report "
     "total_amount_with_tax as procurement spend; say which basis you used."),
    (r"\bdiscounts?\b|\brebates?\b",
     "Discounts are fact_grn.discount_value — <<GRN_DISCOUNT>> in total, non-zero on "
     "<<GRN_DISCOUNT_LINES>> of receipt lines. Report the covered share, not an average across "
     "all lines."),
    (r"\bmrp\b|\bretail price\b|\bmargin at receipt\b",
     "fact_grn.total_mrp_value (<<RECEIVED_MRP>>) is the RETAIL value of goods received, not "
     "revenue and not spend — set it beside the <<RECEIVED_TOTAL>> received at cost. Never "
     "quote it as sales."),
    (r"\bcapex\b|\bcapital\b|\bequipment purchase",
     "Capital purchases are identified by doc_type — 'Dom Capital PO', 'Imp capital PO', "
     "'Service PO', 'CMC', 'AMC' — NOT by a separate table. They are already inside the "
     "<<PROCUREMENT_TOTAL>> procurement total, so a capex figure is a SUBSET of procurement, "
     "never an addition to it. Only order lines WITH a material are in fact_po: services, "
     "civil works and capital items without a code are left out by the import, so the figure "
     "covers coded capital purchases only — say so. Say which doc_types you counted."),
    (r"\bgeneric\b|\bmolecule\b|\bsalt\b",
     "The molecule is dim_material.minor_group_desc (3,237 values) or generic_name; "
     "major_group_desc is the therapeutic CLASS, not the molecule."),
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

    Cached. One fast turn asked for the same resolution five or more times — brief, the
    impossible-grain check inside it, the spelling check, the KPI scope guard, and again
    for every query the model ran — and a question naming a city paid a dim_plant query
    each time. The answer depends only on the question and the warehouse snapshot. The
    copy is deliberate: a caller that edits the result must not change what the next
    caller (or the next turn) is handed from the cache.
    """
    return copy.deepcopy(_resolve_cached(question or "", limit))


@lru_cache(maxsize=256)
def _resolve_cached(question: str, limit: int) -> dict:
    return _resolve_uncached(question, limit)


def _resolve_uncached(question: str, limit: int = 8) -> dict:
    q = question or ""
    postings, exact = _index()
    qtokens = [t.lower() for t in _WORD.findall(q)]
    intent = _intent_words()
    def _asks_rather_than_names(t: str) -> bool:
        # The vocabulary lists "item" and "consumed"; questions say "items" and "consume".
        # Without matching those forms, "how much X did we consume" bound "consume" as a
        # NAME — which then suppressed the misspelling check for X, because that only runs
        # when nothing resolved. One stray verb hid a real typo correction.
        forms = _forms(t)
        # _NOISE_TOKENS exists to strip dosage forms out of PRODUCT NAMES, so that "INJ"
        # and "SYRUP" inside a description carry no identifying weight. Applying it to the
        # QUESTION as well made "how many syrups do we stock" unanswerable — while
        # M119-SYRUPS is a real category. A word the warehouse files as a CLASS is a
        # legitimate thing to ask about, whatever it does inside a product name.
        if any(post[0] == "category" for f in forms for post in postings.get(f, ())):
            return False
        return any(f in bag for f in forms for bag in (_STOP, _NOISE_TOKENS, intent))

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
    def _pattern(w: str) -> str:
        """Match a vocabulary word and its ordinary inflections.

        The list holds "consumption" and "consumed"; a user writes "consume". Matching only
        the literals meant "how much keytruda did we consume?" resolved NO measure at all,
        so the consumption guidance — which table holds both scopes — was never shown. A
        word list that has to enumerate every tense is a list that will always be one tense
        behind, so words of six letters or more also match on their stem.
        """
        esc = re.escape(w)
        if len(w) >= 6 and w.isalpha():
            return rf"(?<![a-z]){re.escape(w[:5])}[a-z]{{0,6}}(?![a-z])"
        return rf"(?<![a-z]){esc}s?(?![a-z])"

    def _asks(words) -> bool:
        # a trailing "s" is allowed so "items" still matches "item" — dropping it cost the
        # material grain on every question that used the plural, which is most of them
        return any(re.search(_pattern(w), low) for w in words)

    def _where(words) -> int:
        hits = [m.start() for w in words
                for m in [re.search(_pattern(w), low)] if m]
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


@lru_cache(maxsize=512)
def generic_materials(generic: str, limit: int = 8) -> tuple[str, ...]:
    """The branded materials a molecule name covers.

    "How much pembrolizumab did we consume?" resolved PEMBROLIZUMAB correctly as a GENERIC
    and was answered "there is no direct match for PEMBROLIZUMAB in the catalog" — because
    no fact table has a generic_name column, and nothing said the bridge to material_desc
    exists. It does: dim_material carries both, and PEMBROLIZUMAB is KEYTRUDA 100MG INJ
    VIAL. This is the synonym problem people reach for embeddings to solve, already
    answered by the warehouse's own dimension.
    """
    from app.ai import warehouse
    try:
        rows = warehouse.query(
            "SELECT DISTINCT material_desc FROM dim_material "
            "WHERE upper(CAST(generic_name AS VARCHAR)) = upper(?) "
            "   OR upper(CAST(minor_group_desc AS VARCHAR)) = upper(?) "
            "ORDER BY 1 LIMIT ?", [generic, generic, limit])
    except Exception:
        return ()
    return tuple(str(r[0]) for r in rows if r[0])


@lru_cache(maxsize=128)
def _count_distinct(table: str, column: str) -> str:
    """A live COUNT(DISTINCT), formatted — so a note never quotes a stale number."""
    from app.ai import warehouse
    try:
        n = warehouse.query(
            f'SELECT COUNT(DISTINCT "{column}") FROM "{table}"', one=True)[0] or 0
    except Exception:
        return "many"
    return f"{n:,}"


def _note_text(note) -> str:
    """A note is a string, or a callable that computes one from the current data.

    Returns "" for a missing note. Without that guard str(None) produced the literal string
    "None", which is truthy — so every measure WITHOUT a note (revenue, purchasing, stock,
    quantity: most of them) appended a bare "None" line to the brief the model reads.
    """
    if note is None:
        return ""
    try:
        return (note() if callable(note) else str(note)) or ""
    except Exception:
        return ""


def _weight_note(fam: dict) -> str:
    """Warn when a family's members are wildly unequal, with the actual counts."""
    # Weight by the TRANSACTION table, not the master list. dim_vendor has 11 rows for
    # Vardhman Health Specialities and 2 for Medisales — a 69/12 split that says nothing.
    # mart_procurement has 128,357 against 3,593, which is the split that matters.
    candidates = [fam["table"]] + [a.split(".", 1)[0] for a in fam.get("also_in") or []]
    scored = []
    for table in candidates:
        w = family_weights(table, fam["column"], fam["token"])
        if len(w) >= 2:
            scored.append((sum(n for _, n in w), table, w))
    if not scored:
        return ""
    for total, table, weights in sorted(scored, reverse=True):
        total = total or 1
        top_name, top_n = weights[0]
        if top_n / total < 0.6:
            continue
        rest = ", ".join(f"{v} ({n:,})" for v, n in weights[1:4])
        return (f" THEY ARE NOT EQUAL: {top_name} has {top_n:,} of the {total:,} rows "
                f"({100 * top_n / total:.0f}%); then {rest}. If the measure is already an "
                f"average per entity, do NOT take a plain mean across them — that gives a "
                f"one-row vendor the same weight as a hundred-thousand-row one. Weight by "
                f"row count, or report the dominant member and say so.")
    return ""


def brief(question: str) -> str:
    """The resolution, as a block to put in front of any model that is about to write SQL."""
    r = resolve(question)
    asks_about_time = bool(re.search(
        r"\b(period|window|range|cover|covers|coverage|timeframe|time frame|history)\b",
        question or "", re.I))
    typos = spelling_suggestions(question, r)
    # A keyword note may be the ONLY thing a question needs. "How much GST did we pay?"
    # resolves no entity, no measure and no grain, so the brief returned empty and the tax
    # guidance — the note that stops tax being added to spend — never reached the model.
    kw_hits = [n for pat, n in _KEYWORD_NOTES if re.search(pat, question or "", re.I)]
    # The same for "what happened last month?": nothing resolves, and the one thing it needs
    # is which month "last month" is.
    rel_note = relative_date_note(question)
    # "KR", "Nagpur": hospital names as HCG says them, which no dimension value spells that
    # way (and two-letter words never reach the resolver at all).
    from app.ai import hospitals as _hosp
    hosp = [(a, c) for a, c in _hosp.mentioned(question) if len(c) == 1
            and not any(e["kind"] == "hospital" for e in r["entities"])]
    if not (r["entities"] or r["families"] or r["cities"] or r["measures"] or r["grains"]
            or asks_about_time or typos or kw_hits or rel_note or hosp):
        return ""
    lines = ["WHAT THIS QUESTION IS ABOUT (resolved against the actual dimension values — "
             "use these exact columns, do not guess where a name lives):"]
    for alias, (code,) in hosp:
        his = _hosp.his_code(code)
        lines.append(f"- \"{alias}\" is the HOSPITAL {_hosp.name(code)}: SAP plant {code} (inventory, "
                     f"purchasing, consumption, and plant= for get_kpi)"
                     + (f", sales code {his} (the `hospital` column of the sales_* tables)" if his else "")
                     + ". Scope every figure to it; never answer with the network total.")
    for e in r["entities"]:
        where = ", ".join(e.get("locations") or [f"{e['table']}.{e['column']}"])
        lines.append(f"- \"{e['text']}\" is a {e['kind'].upper()}, held in {where}"
                     + ("" if e["exact"] else f" (confidence {e['confidence']})"))
        if e["kind"] == "generic":
            # No fact table has a generic_name column. Without the bridge the engine
            # searched material_desc for "PEMBROLIZUMAB", found nothing, and asked the user
            # for a brand name it could have looked up itself.
            brands = generic_materials(e["text"])
            if brands:
                lines.append(
                    f"  A GENERIC/molecule name appears in NO fact table. \"{e['text']}\" "
                    f"is sold as: {', '.join(brands)}. Query those material_desc values (or "
                    f"join dim_material on generic_name) — never report the molecule as "
                    f"absent from the catalogue.")
    for t in typos:
        lines.append(
            f"- LIKELY MISSPELLING: \"{t['typed']}\" matches nothing, but \"{t['meant']}\" "
            f"does ({t['similarity']:.0%} similar) — a {'/'.join(k.upper() for k in t['kinds'])}"
            f", e.g. {', '.join(t['examples'][:2])}. Proceed on that reading and SAY you read "
            f"it that way; do not stop and ask, and do not report the item as unknown.")
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
        if f["n"] == 1:
            # ONE match is an identification. "keytruda" covers 1 of the 4 tokens in
            # "KEYTRUDA 100MG INJ VIAL" and so arrives here rather than as an entity —
            # but there is exactly one such item, and hedging about it helps nobody.
            lines.append(
                f"- \"{f['token']}\" IS {f['examples'][0]} — a {f['kind'].upper()} in "
                f"{f['table']}.{f['column']}. Treat it as identified; filter with "
                f"upper({f['column']}) LIKE '%{f['token'].upper()}%'."
                + (f" Also in {', '.join(f['also_in'])}." if f["also_in"] else ""))
            continue
        lines.append(
            f"- \"{f['token']}\" is not one value: it matches {f['n']} {f['kind'].upper()} "
            f"value(s) in {f['table']}.{f['column']} — {ex}. Cover the whole family with "
            f"upper({f['column']}) LIKE '%{f['token'].upper()}%', and say which ones you "
            f"included. Do not silently answer for just the biggest."
            + (f" Also in {', '.join(f['also_in'])}." if f["also_in"] else "")
            + _weight_note(f))
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
        note = _note_text(_MEASURE_NOTES.get(m))
        if note:
            lines.append(f"  {note}")
    if r["grains"]:
        lines.append(f"- BROKEN DOWN BY: {', '.join(r['grains'])}")
        for g in r["grains"]:
            gnote = _note_text(_GRAIN_NOTES.get(g))
            if gnote:
                lines.append(f"  {gnote}")
    for pat, note in _KEYWORD_NOTES:
        if re.search(pat, question or "", re.I):
            lines.append(f"- {note}")
    # Say it BEFORE the model spends four queries discovering it. A question asking for a
    # grain no table has (sales by hospital x month) used to be answered "I couldn't
    # establish anything solid enough to report", which reads as the assistant being weak
    # when the truth is a fact about the data.
    impossible = impossible_combination(question)
    if impossible:
        lines.append(f"- NOT AVAILABLE AT THIS GRAIN: {impossible}")
    # And the opposite: say it when the grain DOES exist. "get me the sales trend of
    # keytruda" was answered "there is no monthly breakdown — would you like the 6-month
    # total instead?" while sales_by_material_month held all six months. Handing over the
    # finished query leaves no room for that.
    canon = item_trend_sql(question)
    if canon:
        lines.append("- THIS PRODUCT'S MONTHLY SALES EXIST. Run exactly this, then answer from "
                     "it (no <<SALES_MONTHS>>-month total, no purchasing substitute): " + canon)
    elif not impossible and site_trend_unavailable(question):
        # The same product question at ONE hospital has no table. The prompt teaches the
        # month view for product trends, so without this line the company-wide series is
        # the natural thing to run — and to present as the hospital's.
        lines.append(
            "- NOT AVAILABLE AT THIS GRAIN: one product's MONTHLY sales exist only for the "
            "whole company (sales_by_material_month has no hospital), and its hospital split "
            "(sales_by_material_hospital) has no month. Say this hospital's month-by-month "
            "figures do not exist; offer the company-wide monthly series, labelled as ALL "
            "hospitals, or the product's whole-window total by hospital. Never present the "
            "company-wide series as this hospital's.")
    # "biggest selling" is revenue to a CFO and units to a storekeeper, and both readings
    # have a real answer: KEYTRUDA at ₹47.48 Cr, EXAMINATION GLOVES at 1,347,643 units.
    if re.search(r"\b(best|biggest|top|highest)[- ]?(selling|seller)\b|\bmoves? the most\b",
                 question or "", re.I) and not re.search(
                     r"\b(revenue|value|units|quantity|volume|\u20b9)\b", question or "", re.I):
        lines.append(
            "- AMBIGUOUS: \"biggest selling\" is REVENUE or UNITS, and they give different "
            "answers — KEYTRUDA leads on revenue, EXAMINATION GLOVES on units. Pick one, say "
            "which, and mention the other exists.")
    if "month" in r["grains"] or re.search(
            r"\b(period|window|range|cover|covers|coverage|timeframe|time frame|history)\b",
            question or "", re.I):
        win = reporting_window()
        if win:
            lines.append(f"- REPORTING PERIOD: {win}")
    if rel_note:
        lines.append(f"- {rel_note}")
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
# The one table that should answer a measure, when the warehouse has a purpose-built one.
# Pattern-matching column names alone ranked `kpi_doh.consumption_qty` above the view that
# actually holds both consumption scopes.
_MEASURE_PREFERRED: dict[str, tuple[str, ...]] = {
    "consumption": ("consumption_all.qty", "consumption_all.cost"),
    # for a TREND question the month-grained table is the one that can answer it
    "revenue": ("sales_by_material_month.revenue", "sales_by_material.revenue"),
}

_MEASURE_COLUMNS: dict[str, tuple[str, ...]] = {
    "margin":      (r"margin", r"profit"),
    "revenue":     (r"revenue", r"sales_value", r"net_sales", r"^sales$", r"turnover",
                    r"billed_value", r"^value$"),
    # sales_by_material_month is the only table with revenue at material x month. It exists
    # because every raw billing row carries SALESDATE and MATERIALCODE, and the ETL had
    # simply never written that cross — so "show me the sales trend of KEYTRUDA" was
    # answerable as a six-month total and nothing else.
    "purchasing":  (r"line_value", r"po_value", r"purchase_value", r"spend", r"net_value",
                    r"grn_value"),
    "consumption": (r"consum", r"issued", r"usage", r"billed_qty", r"internal_units"),
    # consumption_all.qty is the answer to almost every consumption question; it sorts
    # first because measure_locations prefers marts and short names over kpi_ tables.
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
        rows = warehouse.query(
            "SELECT table_name, column_name FROM information_schema.columns")
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
    # `_pydf_*` are pandas frames registered into DuckDB as an implementation detail of the
    # billable/non-billable rebuild. Offering them as places a measure "lives" sent the
    # model at internals instead of at the real table.
    hits = [f"{t}.{c}" for t, c in _schema_columns()
            if rx.search(c) and not t.startswith("_pydf")]
    preferred = _MEASURE_PREFERRED.get(measure, ())
    # a purpose-built view beats a mart beats a raw fact beats a pre-aggregated KPI
    hits.sort(key=lambda h: (h not in preferred, h.startswith("kpi_"), len(h)))
    for p in reversed(preferred):
        if p not in hits and any(f"{t}.{c}" == p for t, c in _schema_columns()):
            hits.insert(0, p)
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
    # `period` first: it is the only key that sorts correctly across a year boundary
    "month":        re.compile(r"^(period|month|month_num|month_name|posting_date|year)$", re.I),
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
            row = warehouse.query(
                'WITH per_entity AS ('
                f' SELECT "{key}" AS k, SUM("{column}") AS v FROM "{table}"'
                f' WHERE "{column}" > 0 GROUP BY 1),'
                ' ranked AS ('
                ' SELECT v, SUM(v) OVER (ORDER BY v DESC'
                '  ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS run,'
                ' SUM(v) OVER () AS tot FROM per_entity)'
                f' SELECT MIN(v), COUNT(*) FROM ranked WHERE run <= tot * {cover}',
                one=True)
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
        codes = {r[0] for r in warehouse.query("SELECT plant FROM dim_plant")}
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
            vals = {r[0] for r in warehouse.query(
                f'SELECT DISTINCT "{column}" FROM "{table}"') if r[0]}
        except Exception:
            continue
        if not vals:
            continue
        (ok if len(vals & codes) else no).append(f"{table}.{column}")
    return tuple(sorted(ok)), tuple(sorted(no))


def reporting_window() -> str:
    """The period each TRANSACTION source covers, which is not the range of every date in the data.

    Asked "what period does our data cover", the engine scanned every column holding a date
    and answered "2020-01-31 to 2026-12-03" — a range built from expiry dates and forecast
    horizons. The sources also end in different months (billing before the receipts, the
    receipts before the orders, and consumption has a month with no extract), so one window
    for all of them told the model that sales ran as late as the orders. Each source's window
    is read from the data (windows.transaction_windows), sales first.
    """
    from app.ai import windows
    parts = []
    for src, months in windows.transaction_windows():
        gaps = windows.missing_months(months)
        parts.append(f"{months[0]}-01 to {windows.month_end(months[-1])} for {src.label} "
                     f"({src.table}.{src.column})" + (f", with no data for {', '.join(gaps)}" if gaps else ""))
    if not parts:
        return ""
    # The Jun–Aug 2026 HCG reports are NOT in this window and NOT in any table, so a bare
    # "the window is Dec–May" told the model a "formulary compliance trend" could not exist.
    # Name them as the one exception.
    return ("; ".join(parts) + ". Each source covers its own months: a figure that sets two "
            "of them side by side (purchases against sales, internal consumption against "
            "billing) must keep to the months both have. Expiry dates and forecast horizons "
            "run past this and do NOT extend the reporting period; neither does any earlier "
            "date sitting in a master table. The formulary compliance, brand-rule, "
            "scheme/Attis saving and cash & TPA margin-loss reports cover <<REPORTS_RANGE>> "
            "(a separate window, reachable only through get_kpi).")


# ── "LAST MONTH" IS THE LAST MONTH IN THE DATA ───────────────────────────────────────────
# The data stops months before today, and nothing told the model so. "Procurement spend last
# month" could be answered for the calendar month (empty, so "0" or "no recorded spend" —
# which reads as zero spend), or by quietly substituting May-2026 without saying so. The
# months are read from the data, so the note moves when the data does.
_RELATIVE_DATE = re.compile(
    r"\b(?:last|this|previous|current|latest|past)\s+"
    r"(?:month|quarter|year|week|fy|financial\s+year|fiscal\s+year)\b"
    r"|\b(?:ytd|mtd|qtd|today|yesterday)\b", re.I)
_FY_WORDING = re.compile(r"\b(?:fy|ytd|financial\s+year|fiscal\s+year)\b", re.I)


def _month_label(ym: str) -> str:
    """'2026-05' -> 'May-2026', the form answers write months in."""
    y, m = ym.split("-")
    return f"{calendar.month_abbr[int(m)]}-{y}"


def _sales_window_months() -> tuple[str, str] | None:
    """(first, last) month of the sales window as 'YYYY-MM' — the same months the REPORTING
    PERIOD line leads with (windows.sales_months)."""
    from app.ai import windows
    s = windows.sales_months()
    return (s[0], s[-1]) if s else None


def _source_latest() -> list[tuple[str, list[str], list[str]]]:
    """[(latest month 'YYYY-MM', the sources ending in it, notes on months they lack)], one
    entry per latest month in SOURCES order: each source's 'last month' is its own."""
    from app.ai import windows
    out: dict[str, tuple[list[str], list[str]]] = {}
    for src, months in windows.transaction_windows():
        labels, gaps = out.setdefault(months[-1], ([], []))
        labels.append(src.label)
        missing = windows.missing_months(months)
        if missing:
            gaps.append(f"{src.label} has no {', '.join(_month_label(m) for m in missing)}")
    return [(ym, labels, gaps) for ym, (labels, gaps) in out.items()]


def _and(words: list[str]) -> str:
    return words[0] if len(words) == 1 else f"{', '.join(words[:-1])} and {words[-1]}"


def _latest_report_month() -> str | None:
    """Newest month of the Jun–Aug formulary/spend reports ('2026-08'), from the published
    formulary summary; None when it is not built or cannot be read."""
    try:
        from app.ai.page_context import latest_report_month
        return latest_report_month()
    except Exception:
        return None


def relative_date_note(question: str) -> str:
    """The note for 'last month', 'this year', 'YTD', 'today' and the like, or ""."""
    if not _RELATIVE_DATE.search(question or ""):
        return ""
    sales, report = _sales_window_months(), _latest_report_month()
    latest = [f"{_month_label(ym)} ({ym}) for {_and(labels)}" + (f" ({'; '.join(gaps)})" if gaps else "")
              for ym, labels, gaps in _source_latest()]
    if report:
        latest.append(f"{_month_label(report)} ({report}) for the formulary/spend/brand-rule "
                      f"reports")
    note = ("RELATIVE DATE: The data does not run to today, and each source ends in its own "
            "month. 'Last/latest month' = the latest month that source has"
            + (": " + "; ".join(latest) if latest else "") + ". ")
    if sales and _FY_WORDING.search(question or ""):
        # HCG's financial year runs April to March, so "this FY" / "YTD" in the data is a
        # few months at most; say how many rather than let it read as a full year. The
        # sales months are the ones counted: the purchase side runs later (its own latest
        # month is named above).
        first, last = sales
        y, m = (int(x) for x in last.split("-"))
        fy_start = max(f"{y if m >= 4 else y - 1}-04", first)
        cal_start = max(f"{y}-01", first)
        note += (f"FY runs Apr–Mar, so FY-to-date in the sales data is {_month_label(fy_start)}.."
                 f"{_month_label(last)}; calendar YTD is {_month_label(cal_start)}.."
                 f"{_month_label(last)}. ")
    return note + "Say which month you used. Never report an empty recent month as zero."


# ── SPELLING ─────────────────────────────────────────────────────────────────────────────
# "how does the consumption trend of keytuda look?" resolved to nothing at all. One dropped
# letter, and an index built on exact tokens has no opinion whatsoever — while a person
# reads it as Keytruda without pausing. The two strings score 0.98 on Jaro-Winkler.
#
# This runs ONLY when a token matched nothing exactly, so it can never override a real
# match, and it returns a SUGGESTION rather than a binding: the whole point of this module
# is that a near-miss is not an identity. "MSD" reaching STICKER-MSDS is the failure this
# system exists to prevent, and a fuzzy matcher that binds silently would rebuild it.
_MIN_FUZZY_LEN = 5      # shorter tokens are too easy to confuse: 'msd' vs 'mds' vs 'msdc'
_MIN_SIMILARITY = 0.86



def _inflections(t: str) -> set[str]:
    """Only real grammatical inflections: plural and tense.

    Narrower than _forms on purpose. _forms also strips a bare trailing "e" so that
    "consume" reaches "consumed"; using that same set to decide "this is merely an
    inflection, not a typo" threw away "Rochee" -> "roche", where the doubled letter IS
    the typo.
    """
    out = {t}
    for suf in ("s", "es", "ed", "ing"):
        if t.endswith(suf) and len(t) - len(suf) >= 3:
            out.add(t[: -len(suf)])
    out.update({t + "s", t + "es", t + "ed", t + "ing"})
    return out


def _forms(t: str) -> set[str]:
    """A token and its ordinary inflections — "items"/"item", "consume"/"consumed"."""
    out = {t}
    for suf in ("s", "es", "ed", "ing", "d", "e"):
        if t.endswith(suf) and len(t) - len(suf) >= 3:
            out.add(t[: -len(suf)])
    out.update({t + "d", t + "s", t + "es", t + "ed"})
    return out

@lru_cache(maxsize=1)
def _vocabulary() -> tuple[str, ...]:
    """Every distinct token that appears in an indexed dimension value."""
    postings, _ = _index()
    return tuple(postings.keys())


@lru_cache(maxsize=512)
def _did_you_mean(token: str, limit: int = 3) -> tuple[tuple[str, float], ...]:
    from difflib import SequenceMatcher
    tok = (token or "").lower()
    if len(tok) < _MIN_FUZZY_LEN:
        return ()
    # prefilter before scoring: a real misspelling keeps its first letter and its length
    near = [w for w in _vocabulary()
            if abs(len(w) - len(tok)) <= 2 and w[:1] == tok[:1] and w != tok]
    scored = []
    for w in near:
        ratio = SequenceMatcher(None, tok, w).ratio()
        if ratio >= _MIN_SIMILARITY:
            scored.append((w, round(ratio, 3)))
    scored.sort(key=lambda x: -x[1])
    return tuple(scored[:limit])


def spelling_suggestions(question: str, resolved: dict) -> list[dict]:
    """Near-miss tokens worth offering back, for a question that resolved to nothing."""
    # Judged PER TOKEN, not per question. Gating on "nothing resolved at all" meant a
    # single incidental match anywhere in the sentence silenced a real misspelling — the
    # word "consume" did exactly that, and so did "last quarter". A token that matched
    # nothing is worth a suggestion even when its neighbours matched something.
    claimed = {t.lower() for e in resolved.get("entities", []) for t in _WORD.findall(e["text"])}
    claimed |= {f["token"] for f in resolved.get("families", [])}
    intent = _intent_words()
    out = []
    for tok in {t.lower() for t in _WORD.findall(question or "")}:
        forms = _forms(tok)
        if (len(tok) < _MIN_FUZZY_LEN or tok in claimed
                or any(f in bag for f in forms
                       for bag in (_STOP, _NOISE_TOKENS, _BUSINESS_WORDS, intent, _schema_vocabulary()))):
            continue
        postings, _ = _index()
        if postings.get(tok):
            continue                       # it matched exactly; not a misspelling
        for word, score in _did_you_mean(tok):
            if word in _inflections(tok):
                continue          # "vendors" -> "vendor" is an inflection, not a typo
            cands = postings.get(word) or ()
            if not cands:
                continue
            kinds = sorted({c[0] for c in cands})
            examples = sorted({c[1] for c in cands})[:3]
            out.append({"typed": tok, "meant": word, "similarity": score,
                        "kinds": kinds, "examples": examples,
                        "n": len({c[1] for c in cands})})
    return out[:4]


# Projections are not history. forecast_sales carries material, month AND a sales value, so
# a naive search says "a monthly sales trend per drug exists" — from forecast rows.
_PROJECTION = re.compile(r"forecast|project|budget|plan|target", re.I)


@lru_cache(maxsize=512)
def family_weights(table: str, column: str, token: str) -> tuple[tuple[str, int], ...]:
    """How many rows each member of a family actually has, biggest first.

    "What is Vardhman's average lead time?" matched five vendors and was answered 2.68 days
    — the unweighted mean of five per-vendor averages, one of which has 128,357 rows and
    another has one. The real figure for the business is 4.77. Averaging pre-averaged
    per-entity values without weights is a category error, and the brief cannot warn about
    it without knowing the members are unequal.
    """
    from app.ai import warehouse
    try:
        rows = warehouse.query(
            f'SELECT "{column}" AS v, COUNT(*) AS n FROM "{table}" '
            f"WHERE upper(CAST(\"{column}\" AS VARCHAR)) LIKE '%' || upper(?) || '%' "
            f'GROUP BY 1 ORDER BY n DESC LIMIT 12', [token])
    except Exception:
        return ()
    return tuple((str(r[0]), int(r[1])) for r in rows if r[0] is not None)


@lru_cache(maxsize=256)
def grain_measure_tables(measure: str, grain: str, allow_projection: bool = False
                         ) -> tuple[str, ...]:
    """Tables that can serve this measure AT this grain — i.e. hold both columns.

    Some combinations simply do not exist. `sales_monthly` has month and revenue but no
    hospital; `sales_by_hospital` has hospital and revenue but no month. So a monthly sales
    trend for one hospital is not a hard question, it is an impossible one — and saying so
    beats "I couldn't establish anything solid enough to report", which reads as a failure
    of the assistant rather than a fact about the data. (Material x month DOES exist:
    sales_by_material_month.)
    """
    pat = GRAIN_COLUMNS.get(grain)
    if not pat:
        return ()
    by_table: dict[str, list[str]] = defaultdict(list)
    for t, c in _schema_columns():
        by_table[t].append(c)
    measure_cols = {loc.split(".", 1)[0]: loc.split(".", 1)[1]
                    for loc in measure_locations(measure, limit=40)}
    out = []
    for table, cs in by_table.items():
        if table.startswith("_pydf") or table not in measure_cols:
            continue
        if _PROJECTION.search(table) and not allow_projection:
            continue
        if any(pat.match(c) for c in cs):
            out.append(f"{table}.{measure_cols[table]}")
    return tuple(sorted(out))


@lru_cache(maxsize=256)
def _tables_serving(measure: str, grains: tuple[str, ...], allow_projection: bool = False
                    ) -> tuple[str, ...]:
    """Tables holding the measure AND every one of these grains at once."""
    sets = [set(grain_measure_tables(measure, g, allow_projection)) for g in grains if g]
    if not sets:
        return ()
    common = set.intersection(*sets) if len(sets) > 1 else sets[0]
    # intersect on TABLE, not on table.column
    names = [{loc.split(".", 1)[0] for loc in s_} for s_ in sets]
    shared = set.intersection(*names) if len(names) > 1 else names[0]
    return tuple(sorted(loc for loc in common if loc.split(".", 1)[0] in shared))


def impossible_combination(question: str) -> str | None:
    """A named reason the question cannot be answered, or None if it can be.

    Returned INSTEAD of a generic "I couldn't establish anything", which tells the reader
    their assistant is weak when the truth is that their warehouse has no such grain.
    """
    r = resolve(question)
    if not (r["measures"] and r["grains"]):
        return None
    measure = r["measures"][0]
    # every grain the question needs AT ONCE — naming a drug and asking for a trend means
    # material AND month (sales_by_material_month); naming a hospital means hospital AND
    # month, which no sales table has
    grains = list(dict.fromkeys(
        r["grains"] + [e["kind"] for e in r["entities"] if e["kind"] in GRAIN_COLUMNS]
        + [f["kind"] for f in r["families"] if f["kind"] in GRAIN_COLUMNS]))
    if not grains:
        return None
    wants_projection = bool(re.search(r"forecast|project|predict|expect", question or "", re.I))
    if _tables_serving(measure, tuple(grains), wants_projection):
        return None
    grain = " x ".join(g.upper() for g in grains)
    # is it the grain or the measure that is missing?
    alt_grains = [g for g in GRAIN_COLUMNS if grain_measure_tables(measure, g)]
    alt_measures = [m for m in _MEASURE_COLUMNS if _tables_serving(m, tuple(grains))]
    parts = [f"No table holds {measure.upper()} at {grain} grain — the combination "
             f"does not exist in this warehouse, so no query can produce it."]
    if alt_grains:
        parts.append(f"{measure.upper()} IS available by: {', '.join(sorted(alt_grains))}.")
    if alt_measures:
        parts.append(f"At {grain} grain you CAN have: {', '.join(sorted(alt_measures))}.")
    parts.append("Say this plainly and offer the closest thing that does exist — do not "
                 "report it as a failure to find anything.")
    return " ".join(parts)


# ── THE ONE CORRECT QUERY FOR "SALES TREND OF <PRODUCT>" ─────────────────────────────────
# Everything needed to answer it is already resolved here: the product (a code, an exact
# description, a molecule, or one rare token naming a handful of SKUs), the measure (revenue
# or margin) and the grain (month). So the query is written here too, once, deterministically
# — the brief hands it to both engines and the orchestrator refuses a clarification while it
# applies. Built on the de-duplicated month view, where KEYTRUDA's December reads 9.68 Cr.
#
# Because the brief then says "Run exactly this" and the orchestrator refuses to let the
# model ask, a WRONG query here is worse than none: it is answered with full confidence.
# Every doubt below therefore resolves to None, and the model works the question out itself.
_TREND_PRODUCT_MAX = 5           # more SKUs than this is a family, not "this product"
_NOT_A_PRODUCT_SCOPE = ("hospital", "vendor", "department")
_PROJECTION_WORDS = re.compile(r"forecast|project|predict|expect|budget|target", re.I)
# A TOTAL that merely mentions months is not a trend. "Keytruda total revenue for the 6-month
# period" — the very follow-up the assistant offered — resolved the month grain from
# "6-month" and was handed the month-by-month query with "Run exactly this". The test is
# total wording WITHOUT trend wording, so "total monthly revenue of keytruda" stays a trend.
_TOTAL_WORDS = re.compile(
    r"\b(?:\d+|one|two|three|four|five|six|twelve)[- ]months?\b|\btotal\b|\boverall\b", re.I)
_TREND_WORDS = re.compile(
    r"\btrends?\b|\bmonthly\b|\bby month\b|\bmonth[- ]on[- ]month\b|\bmonth[- ]by[- ]month\b"
    r"|\bover time\b|\beach month\b|\bper month\b|\bmonth[- ]?wise\b", re.I)


def _asks_for_a_total(question: str) -> bool:
    q = question or ""
    return bool(_TOTAL_WORDS.search(q)) and not _TREND_WORDS.search(q)

# Where a product sits when it IS the thing asked about: after "of" / "for" / "about" / "in"
# ("the sales trend of keytruda"), straight before what is measured ("keytruda sales",
# "keytruda's monthly revenue"), or in quotes. A one-token family anywhere else is a word the
# question happens to contain: "Show the network-wide monthly sales trend" formed a family on
# "network" (four network switches) and was handed their sales as the company's trend.
_OBJECT_BEFORE = r"(?:\b(?:of|for|about|in)\s+|[\"'“‘]\s*)"
_OBJECT_AFTER = (r"(?:['’]s)?[\"'”’]?\s+(?:sales?|sold|sell(?:s|ing)?|"
                 r"revenues?|margins?|profits?|trends?|monthly|months?)\b")

# A hospital named by its short form never resolves. In "sales trend of keytruda at HCG KR",
# HCG is a noise word and KR is two letters, below every matching threshold — so the
# question looked company-wide and was handed the all-hospital series to present as HCG KR's.
# No table holds one product's sales by hospital AND month, so any site wording means None.
# Case-sensitive on purpose: "HCG KR" and "at KR" are names, "HCG network" is not.
_SITE_WORDING = re.compile(r"\bHCG[\s-]+[A-Z]{2,}\b|\bat\s+(?!HCG\b)[A-Z]{2,}\b")
# Upper-case words in hospital names that describe a service or a legal form, not a site.
_SITE_NOT_A_NAME = {"llp", "pet", "pcr", "cdx"}


@lru_cache(maxsize=1)
def _site_short_names() -> frozenset[str]:
    """The short forms hospitals go by — KR, DR, NMR, EKO — read from dim_plant's names.

    Read, not listed: a new hospital's short form is covered the day it appears in dim_plant.
    Lower-case typing ("hcg kr") is why the case-sensitive pattern above is not enough.
    """
    from app.ai import warehouse
    try:
        rows = warehouse.run_sql(
            "SELECT DISTINCT plant_name FROM dim_plant WHERE plant_name IS NOT NULL",
            row_cap=200)["rows"]
    except Exception:
        return frozenset()
    out: set[str] = set()
    for r in rows:
        for w in _WORD.findall(str(r["plant_name"])):
            low = w.lower()
            if (w.isupper() and 2 <= len(w) <= 5 and low not in _NOISE_TOKENS
                    and low not in _SITE_NOT_A_NAME):
                out.add(low)
    return frozenset(out)


def _names_a_site(question: str) -> bool:
    """True when the question points at a hospital in a way resolve() cannot see."""
    q = question or ""
    if _SITE_WORDING.search(q):
        return True
    short = _site_short_names()
    words = [t.lower() for t in _WORD.findall(q)]
    return any(prev in ("hcg", "at", "in") and w in short for prev, w in zip(words, words[1:]))


def _names_as_the_object(question: str, token: str) -> bool:
    t = re.escape(token)
    return bool(re.search(_OBJECT_BEFORE + t + r"\b", question, re.I)
                or re.search(r"\b" + t + _OBJECT_AFTER, question, re.I))


def _sql_str(v) -> str:
    return "'" + str(v).replace("'", "''") + "'"


def _norm_code(code) -> str:
    s = str(code).strip()
    return s[:-2] if s.endswith(".0") and s[:-2].isdigit() else s


def _codes_where(sql: str, params: list) -> list[str]:
    from app.ai import warehouse
    try:
        rows = warehouse.query(sql, params)
    except Exception:
        return []
    return sorted({_norm_code(r[0]) for r in rows if r[0] is not None})


@lru_cache(maxsize=256)
def _sold_materials(where: str) -> int:
    """How many distinct materials with real revenue this filter reaches in the month view.

    The brief tells the model "THIS PRODUCT'S MONTHLY SALES EXIST", so that has to be checked,
    not assumed. It also catches what the item master cannot: a family is counted in
    dim_material by whole token, but the filter is a substring match on the sales feed, so
    "wise" was 2 catalogue SKUs and 9 sold products.
    """
    from app.ai import warehouse
    try:
        row = warehouse.query(
            "SELECT count(DISTINCT material), SUM(revenue) FROM sales_by_material_month "
            f"WHERE {where}", one=True)
    except Exception:
        return 0
    if not row or not row[0] or row[1] is None or row[1] <= 0:
        return 0
    return int(row[0])


def _product_trend_where(q: str, r: dict, any_site: bool = False) -> str | None:
    """The WHERE clause naming the one product a monthly revenue/margin question is about.

    any_site=True skips the hospital checks, so the brief can tell "a product trend for one
    hospital" (which does not exist) from "not a product trend at all".

    A total question is not a trend at all, here or at one hospital: a product's whole-window
    total exists both company-wide (sales_by_material) and per hospital.
    """
    if _asks_for_a_total(q):
        return None
    if ("month" not in r["grains"] or not r["measures"]
            or r["measures"][0] not in ("revenue", "margin")
            or any(g not in ("month", "material") for g in r["grains"])
            or _PROJECTION_WORDS.search(q)):
        return None
    # Families count as scope too: "keytruda at Panda" names the hospital by a single token,
    # which forms a hospital FAMILY, never an entity. And "network" is a vendor family as
    # well as a material one — a word that names two kinds of thing names neither.
    scope_kinds = tuple(k for k in _NOT_A_PRODUCT_SCOPE if not (any_site and k == "hospital"))
    if (not any_site and r["cities"]) or any(
            x["kind"] in scope_kinds for x in r["entities"] + r["families"]):
        return None
    named = ([e for e in r["entities"] if e["kind"] in ("material", "generic")]
             + [f for f in r["families"] if f["kind"] in ("material", "generic")])
    if len(named) != 1:
        return None
    thing = named[0]
    if "token" in thing:
        # A family: one rare token covering a few SKUs ("keytruda" -> KEYTRUDA 100MG INJ
        # VIAL). Filter on the token itself, which is [A-Za-z0-9] only, so it is safe to
        # inline — and it also catches a sold variant the item master has not catalogued.
        # A single token is weak evidence, so it must also be the question's OBJECT and
        # name nothing but products (a manufacturer or category family of the same token
        # means it is a company or a class, not this product).
        tok = thing["token"]
        if (thing["kind"] != "material" or thing["column"] != "material_desc"
                or thing["n"] > _TREND_PRODUCT_MAX
                or any(f["token"] == tok and f["kind"] not in ("material", "generic")
                       for f in r["families"])
                or not _names_as_the_object(q, tok)):
            return None
        where = f"upper(material_desc) LIKE '%{tok.upper()}%'"
    else:
        # An exact entity. Filter by CODE, not description: the sales feed spells 191
        # products differently from the item master, so an exact-description filter can
        # quietly return nothing.
        if thing["kind"] == "generic":
            codes = _codes_where(
                "SELECT DISTINCT material FROM dim_material "
                "WHERE upper(CAST(generic_name AS VARCHAR)) = upper(?) "
                "   OR upper(CAST(minor_group_desc AS VARCHAR)) = upper(?)",
                [thing["text"], thing["text"]])
        elif thing["column"] == "material":
            codes = [_norm_code(thing["text"])]
        else:
            codes = _codes_where(
                "SELECT DISTINCT material FROM dim_material WHERE upper(material_desc) = upper(?)",
                [thing["text"]])
        if not codes or len(codes) > _TREND_PRODUCT_MAX:
            return None
        where = f"material IN ({', '.join(_sql_str(c) for c in codes)})"
    if not 1 <= _sold_materials(where) <= _TREND_PRODUCT_MAX:
        return None
    return where


def _trend_sql(where: str) -> str:
    return ("SELECT period, SUM(revenue) AS revenue, SUM(cost) AS cost, "
            "SUM(revenue - cost) AS margin, SUM(qty) AS qty "
            f"FROM sales_by_material_month WHERE {where} GROUP BY period ORDER BY period")


@lru_cache(maxsize=256)
def item_trend_sql(question: str) -> str | None:
    """The one correct query for 'sales trend of <product>', or None when it does not apply.

    None unless the question names exactly ONE product and asks for its revenue or margin by
    month, with no hospital, city, vendor or department in it (sales_by_material_month has
    none of those, so a query here would silently answer for the whole company) and no
    forecast wording (projections are not history). Two products also give None: one
    summed series would hide the comparison that was asked for. So does a total ("6-month",
    "total", "overall" with no trend word): the month rows are not its answer.

    "No hospital" includes the short forms resolve() cannot see ("at HCG KR"), and a product
    named by one token must be the question's object, not a word it happens to contain.
    """
    q = question or ""
    if _names_a_site(q):
        return None
    where = _product_trend_where(q, resolve(q))
    return _trend_sql(where) if where else None


def site_trend_unavailable(question: str) -> bool:
    """True when the question asks for one product's monthly sales AT one hospital.

    That grain does not exist: the month view has no hospital, the hospital view has no
    month. Saying so up front stops the company-wide series being relabelled as the site's.
    """
    q = question or ""
    r = resolve(q)
    at_site = (_names_a_site(q) or r["cities"]
               or any(x["kind"] == "hospital" for x in r["entities"] + r["families"]))
    return bool(at_site) and _product_trend_where(q, r, any_site=True) is not None
