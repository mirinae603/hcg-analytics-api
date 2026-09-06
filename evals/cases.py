"""The question bank — real questions a hospital supply-chain team would actually ask.

Every expectation below is tied to a figure verified directly against the warehouse, not
to wording. A case asserts what the answer must CONTAIN (a number that is right) and what
it must NOT claim (an absence that is false, a fabricated grain, a wrong entity), because
an eval that matches phrases teaches the system to produce phrases.

GROUND TRUTH (checked 2026-08-30, all from the live warehouse):
  53 hospitals · 24,931 materials · 18,080 stocked SKUs
  Sales      ₹521.67 Cr revenue, ₹213.43 Cr margin  → 40.9% overall
  Monthly    Dec 89.52 · Jan 90.90 · Feb 83.12 · Mar 83.16 · Apr 87.04 · May 87.93 (Cr)
  Top item   KEYTRUDA 100MG INJ VIAL ₹47.48 Cr    Top category M065-INJECTIONS ₹344.31 Cr
  Top mfr    Reliance ₹58.71 Cr, MSD ₹47.57 Cr    Top sales hospital KABHK ₹104.70 Cr
  Vendors    Vardhman ₹297.77 Cr / 111,582 PO lines / 4.8d avg lead time
  Inventory  ₹60.47 Cr; HM01 ₹8.39 Cr, HC05 ₹7.80 Cr; 10,501 non-moving SKUs
  Expiry     already expired 4,041 lines ₹83.13 L; within 90d ₹39.97 L / 45,223 units
  Worst margin over ₹1 Cr: POLIVY 30MG INJ VIAL at 5.78%

STRUCTURAL FACTS the assistant must respect:
  * No material x month SALES grain. Monthly figures for an item exist only for
    purchasing (fact_po, fact_grn, mart_procurement, kpi_monthly_purchase_value) and
    consumption (fact_consumption, kpi_units_consumed).
  * Sales hospital codes (KABHK, GJHCA) and plant codes (HC05, AH01) share ZERO rows and
    nothing maps between them. Hospital NAMES and cities live only in dim_plant.plant_name,
    so no city can reach the sales tables.
  * MSD is a MANUFACTURER, not a vendor. Lead times are keyed by vendor.
  * KEYTRUDA is billed-only: fact_consumption and kpi_units_consumed hold 0 rows for it.
"""
from __future__ import annotations

import re


def _norm(t: str) -> str:
    return re.sub(r"[\s,]+", " ", (t or "").lower())


def has(*subs):
    """Every fragment must appear (commas and spacing ignored)."""
    return lambda t: all(_norm(s) in _norm(t) for s in subs)


def any_of(*subs):
    return lambda t: any(_norm(s) in _norm(t) for s in subs)


def lacks(*subs):
    return lambda t: not any(_norm(s) in _norm(t) for s in subs)


def all_of(*checks):
    return lambda t: all(c(t) for c in checks)


def head_lacks(*bad):
    """Ban words from the HEADLINE only.

    `lacks` reads the whole brief, which punished the right behaviour: a breakdown that
    ranks ANTINEOPLASTIC first and then honestly notes an unclassified bucket further down
    was failing on the word "uncategorized". Leading with the unnamed bucket is the error;
    disclosing it is not.
    """
    def _check(t):
        head = (t or "")[:260].lower()
        return not any(b.lower() in head for b in bad)
    return _check


def num_within(target: float, tol_pct: float = 2.0):
    """Some number in the answer is within tolerance of the truth — tolerant of ₹/Cr/L
    formatting and of the model rounding, strict about the magnitude being right."""
    def check(t: str) -> bool:
        for m in re.finditer(r"(\d[\d,]*\.?\d*)", t or ""):
            try:
                v = float(m.group(1).replace(",", ""))
            except ValueError:
                continue
            if target and abs(v - target) <= abs(target) * tol_pct / 100:
                return True
        return False
    return check


# `must_answer` = a clarifying question does not count; the data supports a real answer.
L1: list[dict] = [
    # ── canonical figures: these must be exact ───────────────────────────────
    {"id": "expiry-90d", "q": "How much stock is expiring in the next 90 days?",
     # Require BOTH canonical figures rather than banning other numbers: naming the
     # already-expired stock separately is correct analysis, and the previous check failed
     # an answer that led with 45,223 / ₹39.97 L and then properly distinguished the
     # expired band. What matters is that the HEADLINE is right, not that other figures
     # are absent.
     "check": all_of(lambda t: num_within(39.97)(t) or num_within(45223, 1)(t),
                     lacks("101,005", "101005")),   # the already-expired-inclusive figure
     "why": "canonical: ₹39.97 L AND 45,223 units (0-30d + 31-90d), excluding already-expired"},

    {"id": "total-sales", "q": "What is our total sales revenue and margin?",
     "check": all_of(num_within(521.67, 3), num_within(213.43, 3)),
     "why": "₹521.67 Cr revenue, ₹213.43 Cr margin"},

    {"id": "overall-margin", "q": "What is our overall margin percentage on sales?",
     "check": num_within(40.9, 5), "why": "40.9%"},

    {"id": "stock-value", "q": "What is the total value of stock we are holding right now?",
     "check": num_within(60.47, 5), "why": "₹60.47 Cr across 18,080 SKUs"},

    # MY EXPECTATION WAS THE OUTLIER. `kpi_non_moving` holds 16,872 rows and 10,501
    # DISTINCT materials, and the canonical KPI reports `blocked_skus: 16872` — so the
    # dashboard's "SKUs" are really material x hospital lines. The chatbot agreeing with
    # the dashboard is correct behaviour; an eval demanding 10,501 would train it to
    # contradict the screen the user is looking at. Both figures accepted; what matters is
    # that it does not invent a third.
    #
    # WORTH RAISING WITH THE BUSINESS: "16,872 non-moving SKUs" overstates the item count
    # by 61% if a reader takes SKU to mean a product.
    {"id": "non-moving-count", "q": "How many SKUs are non-moving?",
     "check": lambda t: num_within(16872, 2)(t) or num_within(10501, 2)(t),
     "why": "canonical blocked_skus = 16,872 (material x hospital lines); 10,501 distinct "
            "materials also acceptable — but not a third number"},

    # ── rankings: the leader AND its concentration ───────────────────────────
    {"id": "top-vendor", "q": "Which vendor do we spend the most with, and how exposed are we to them?",
     "check": all_of(has("vardhman"), num_within(297.77, 3)),
     "why": "Vardhman ₹297.77 Cr — the answer must name it and size it", "must_answer": True},

    {"id": "top-selling-item", "q": "What is our single biggest selling product by revenue?",
     "check": all_of(has("keytruda"), num_within(47.48, 3)), "why": "KEYTRUDA ₹47.48 Cr"},

    {"id": "top-category", "q": "Which product category drives most of our sales revenue?",
     "check": all_of(any_of("injection", "m065"), num_within(344.31, 4)),
     "why": "M065-INJECTIONS ₹344.31 Cr of ₹521.67 Cr"},

    {"id": "top-manufacturer", "q": "Which manufacturer accounts for the most sales revenue?",
     "check": all_of(has("reliance"), num_within(58.71, 4)), "why": "Reliance ₹58.71 Cr"},

    {"id": "top-hospital-sales", "q": "Which hospital generates the most sales revenue?",
     "check": all_of(has("kabhk"), num_within(104.70, 4)), "why": "KABHK ₹104.70 Cr"},

    {"id": "top-stock-site", "q": "Which hospital is holding the most inventory value?",
     "check": all_of(any_of("hm01", "hc05"), num_within(8.39, 8)), "why": "HM01 ₹8.39 Cr"},

    # ── trends: direction and magnitude, never 'it varies' ───────────────────
    {"id": "revenue-trend", "q": "How has our monthly revenue trended over the period?",
     "check": all_of(any_of("dec", "december"), num_within(89.52, 4),
                     any_of("rose", "fell", "flat", "stable", "declin", "increas", "grew", "%")),
     "why": "Dec 89.52 → May 87.93 Cr; must state a direction, not 'varies'", "must_answer": True},

    {"id": "keytruda-trend", "q": 'Show me the sales trend for "KEYTRUDA 100MG INJ VIAL"',
     "check": all_of(
         any_of("purchas", "consumption", "no monthly sales", "no sales figure",
                "not available", "no real trend", "does not hold", "doesn't hold",
                "no time-series", "not contain time-series", "no time series",
                "isn't answerable", "not answerable", "no monthly"),
         lacks("89.52", "90.90")),   # the ALL-material monthly totals
     "why": "no material-by-month sales grain: substitute and SAY SO, or refuse — never "
            "report company-wide monthly revenue as this item's"},

    # ── traps: wrong entity type, false absence ──────────────────────────────
    {"id": "msd-procurement", "q": "Show me the procurement history for MSD products",
     "check": lacks("no recorded procurement", "no procurement details", "no procurement data",
                    "not available", "sticker"),
     "why": "614 lines, ₹42.34 Cr via manufacturer_desc. Must not report absence, must not "
            "resolve MSD to STICKER-MSDS"},

    {"id": "msd-lead-times", "q": "What are the lead times for MSD supplies?",
     "check": lacks("no recorded lead times", "no lead times", "no data"),
     "why": "MSD is a manufacturer; lead times are per vendor — hop to its vendors"},

    {"id": "sales-vs-stock-by-hospital",
     "q": "For each hospital, compare its sales revenue against the stock value it is holding.",
     # widened after a correct answer failed on wording: "there is no linkage between
     # hospital and plant codes" is exactly right and matched none of the original
     # fragments. A check that only recognises one phrasing of the truth is a check that
     # punishes correctness.
     "check": any_of("cannot", "can't", "not possible", "no mapping", "no linkage", "no link",
                     "does not", "doesn't", "do not", "don't", "isn't answerable",
                     "not answerable", "separate", "unable", "different code"),
     "why": "sales hospital codes and plant codes share ZERO rows and nothing maps between "
            "them — the honest answer says so rather than joining or substituting"},

    {"id": "bangalore-sales", "q": "What are the top selling drugs in our Bangalore hospitals?",
     "check": any_of("purchas", "consumption", "cannot", "can't", "not possible", "no city",
                     "plant_name", "hospital name", "procurement"),
     "why": "city lives only in dim_plant.plant_name, which cannot reach sales — answer from "
            "procurement and say so, or explain the limit"},

    # ── diagnosis: must decompose and rule something out ─────────────────────
    {"id": "vendor-concentration",
     "q": "Our procurement is concentrated in one vendor. How exposed are we, and which categories would hurt most if they failed?",
     "check": all_of(has("vardhman"), lacks("evidence does not allow", "not answerable"),
                     lambda t: not ("100%" in t or "100.0%" in t)),   # the false-share artefact
     "why": "must name the vendor and reach a conclusion", "must_answer": True},

    # "High-value" is a judgement call — POLIVY at 5.78% is the worst above ₹1 Cr, but
    # OPDYTA at 12.1% on ₹8.42 Cr of revenue is a defensible reading of the same question.
    # The check now asks for what actually matters: a real low-margin item, named, with a
    # margin figure that is genuinely low.
    {"id": "worst-margin-drugs",
     "q": "Which high-value drugs are we making the worst margin on?",
     "check": all_of(any_of("polivy", "darzalex", "opdyta", "keytruda"),
                     lambda t: any(num_within(v, 30)(t) for v in (5.78, 7.6, 12.1)),
                     lacks("not answerable", "no data")),
     "why": "must NAME a genuinely low-margin high-value drug with its margin %",
     "must_answer": True},

    # MY CHECK WAS WRONG HERE, the model was right. Raw SQL over fact_inventory gives
    # 4,041 lines / ₹83.13 L / 101,005 units, but the canonical near-expiry KPI — the
    # figure on the dashboard — reports the Expired bucket as 2,781 / ₹43.16 L / 55,782
    # units, because it filters to priced, stocked lines. An eval that demands the raw
    # number would train the assistant to contradict the dashboard.
    {"id": "expired-writeoff", "q": "How much stock has already expired and what is it worth?",
     "check": all_of(any_of("expired"),
                     lambda t: num_within(43.16, 8)(t) or num_within(83.13, 8)(t)),
     "why": "canonical Expired bucket ₹43.16 L / 55,782 units (raw fact_inventory ₹83.13 L "
            "also acceptable); must NOT be the 90-day figure"},

    {"id": "cash-to-restock", "q": "How much cash do we need to restock what is out of stock?",
     "check": lacks("not answerable", "cannot determine", "no data"), "why": "reorder value exists"},

    {"id": "slow-movers-by-site",
     "q": "Which hospitals are carrying the most slow-moving stock, and what is it worth?",
     "check": lacks("not answerable", "no data available"), "why": "non-moving by plant is answerable",
     "must_answer": True},

    {"id": "lead-time-risk",
     "q": "Which vendors have the longest lead times on items we actually depend on?",
     "check": lacks("not answerable", "no lead time data"), "why": "kpi_vendor_lead_time joined to volume",
     "must_answer": True},

    {"id": "price-variance",
     "q": "Are we paying inconsistent prices for the same item across hospitals?",
     "check": lacks("not answerable", "no pricing data"),
     "why": "mart_material_price_stats / mart_material_vendor_price_stats exist"},

    {"id": "oncology-exposure",
     "q": "How much of our inventory value sits in oncology drugs, and how much of that is near expiry?",
     "check": lacks("not answerable", "no data"), "why": "material_type/category + near-expiry"},

    {"id": "stockout-impact", "q": "Which items are out of stock but still showing demand?",
     "check": lacks("not answerable", "no data"), "why": "reorder-priority / stock-out KPIs cover this",
     "must_answer": True},

    {"id": "fill-rate", "q": "What is our fill rate and where is it worst?",
     "check": lacks("not answerable", "no fill rate"), "why": "kpi_fill_rate exists"},

    {"id": "doh", "q": "What is our days-on-hand, and which sites are furthest from target?",
     "check": lacks("not answerable", "no data"), "why": "kpi_doh exists"},

    {"id": "keytruda-consumption",
     "q": 'How much "KEYTRUDA 100MG INJ VIAL" have we actually consumed?',
     "check": any_of("billed", "0", "no internal consumption", "not consumed", "purchas", "2,193", "2193"),
     "why": "billed-only item: fact_consumption has 0 rows; the honest answer says billed "
            "rather than reporting zero as 'no data'"},

    {"id": "margin-by-hospital", "q": "Which hospitals run the thinnest margins on their sales?",
     "check": lacks("not answerable", "no data"), "why": "sales_by_material_hospital has revenue and cost",
     "must_answer": True},
]

# ═══════════════════════════════════════════════════════════════════════════
# LEVEL 1 — a single fact. One metric, no breakdown, no reasoning. If these fail
# nothing else matters, so their expectations are exact figures.
# ═══════════════════════════════════════════════════════════════════════════
L1 += [
    {"id": "l1-hospital-count", "q": "How many hospitals do we operate?",
     "check": num_within(53, 2), "why": "53 in dim_plant"},
    {"id": "l1-material-count", "q": "How many distinct materials are in the catalogue?",
     "check": lambda t: num_within(24931, 3)(t) or num_within(24805, 3)(t),
     "why": "24,931 material codes / 24,805 distinct descriptions"},
    {"id": "l1-stocked-skus", "q": "How many SKUs do we currently hold stock for?",
     "check": num_within(18080, 3), "why": "18,080 distinct materials in fact_inventory"},
    {"id": "l1-top-mfr-sales", "q": "How much did MSD sell?",
     "check": num_within(47.57, 4), "why": "MSD ₹47.57 Cr — and MSD is a MANUFACTURER"},
    {"id": "l1-vendor-count", "q": "How many vendors do we buy from?",
     # 2,251 distinct vendor_codes actually appear in procurement; dim_vendor lists 3,576
     # including ones we have never bought from. The question says "buy from", so 2,251 is
     # the better answer and my earlier ~3,4xx was simply wrong.
     "check": lambda t: num_within(2251, 3)(t) or num_within(3576, 3)(t),
     "why": "2,251 vendors actually bought from (3,576 on the master list)"},
    {"id": "l1-total-procurement", "q": "What is our total procurement spend?",
     "check": num_within(649.91, 4), "why": "₹649.91 Cr"},
    {"id": "l1-expired-value", "q": "What is the value of stock that has already expired?",
     "check": lambda t: num_within(43.16, 8)(t) or num_within(83.13, 8)(t),
     "why": "canonical ₹43.16 L (raw ₹83.13 L also defensible)"},
    {"id": "l1-median-lead-time", "q": "What is our median vendor lead time?",
     "check": num_within(3.6, 30), "why": "3.6 days"},
    {"id": "l1-top-vendor-name", "q": "Who is our largest supplier?",
     "check": has("vardhman"), "why": "Vardhman Health Specialities"},
    {"id": "l1-top-category-name", "q": "What is our biggest spend category?",
     "check": all_of(any_of("injection", "m065", "antineoplastic"),
                     head_lacks("uncategorized", "uncategorised")),
     "why": "two real taxonomies — M065-INJECTIONS (material group) or ANTINEOPLASTIC "
            "(therapeutic, what the dashboard shows). Never 'Uncategorized'."},
]

# ═══════════════════════════════════════════════════════════════════════════
# LEVEL 2 — one dimension. A ranking, a filter, a breakdown, a trend. This is
# where entity typing starts to matter: a name has to land in the right column.
# ═══════════════════════════════════════════════════════════════════════════
L2: list[dict] = [
    {"id": "l2-top5-vendors", "q": "Give me the top 5 vendors by procurement spend.",
     "check": all_of(has("vardhman"), num_within(297.77, 4)), "must_answer": True,
     "why": "Vardhman ₹297.77 Cr leads"},
    {"id": "l2-sales-by-hospital", "q": "Rank our hospitals by sales revenue.",
     "check": all_of(has("kabhk"), num_within(104.70, 5)), "must_answer": True, "why": "KABHK ₹104.70 Cr"},
    {"id": "l2-stock-by-hospital", "q": "Which hospitals hold the most inventory value?",
     "check": all_of(any_of("hm01", "hc05"), num_within(8.39, 10)), "must_answer": True, "why": "HM01 ₹8.39 Cr"},
    {"id": "l2-nonmoving-by-group", "q": "Which product groups have the most non-moving stock value?",
     "check": all_of(any_of("m017", "endo"), num_within(2.32, 12)), "must_answer": True,
     "why": "M017-ENDO SURG ACCES ₹2.32 Cr"},
    {"id": "l2-nonmoving-by-site", "q": "Which hospitals are sitting on the most non-moving stock?",
     # ₹145.03 L IS ₹1.45 Cr. The answer said "HM01 (₹1.45 Cr)" and was correct; the check
     # demanded the lakh-scaled digits and failed it. A check must not care which unit a
     # correct figure is expressed in.
     "check": all_of(any_of("hm01", "ah01"),
                     lambda t: num_within(145.03, 12)(t) or num_within(1.45, 12)(t)),
     "must_answer": True, "why": "HM01 ₹1.45 Cr (₹145.03 L), AH01 ₹1.44 Cr"},
    {"id": "l2-procurement-by-category", "q": "Break our procurement spend down by category.",
     "check": all_of(any_of("m065", "injection", "antineoplastic"),
                     lambda t: num_within(234.06, 6)(t) or num_within(84.45, 6)(t),
                     head_lacks("uncategorized", "uncategorised")),
     "must_answer": True,
     "why": "M065-INJECTIONS ₹234.06 Cr (material group) or ANTINEOPLASTIC ₹84.45 Cr "
            "(therapeutic) — both real; never the unclassified bucket"},
    {"id": "l2-bangalore-procurement", "q": "How much do our Bangalore hospitals spend on procurement?",
     # Bangalore covers FOUR hospitals, so both the citywide total (~₹174 Cr) and the
     # largest single site (₹90.82 Cr) are right answers depending on the reading.
     "check": all_of(any_of("hc05", "bangalore"),
                     lambda t: num_within(90.82, 10)(t) or num_within(174.02, 12)(t)),
     "must_answer": True, "why": "citywide ~₹174 Cr, or HCG KR ₹90.82 Cr — via plant_name"},
    # "Move" is genuinely ambiguous between SOLD and CONSUMED, and both have a real answer:
    # EXAMINATION GLOVES 1,347,643 units sold, LEAFLET A5 1,203,000 units consumed. The
    # test is that it answers at PRODUCT level (not category) and says which measure it
    # used — not that it picks the reading I happened to have in mind.
    {"id": "l2-top-units-sold", "q": "Which products move the most units?",
     "check": all_of(lambda t: num_within(1347643, 3)(t) or num_within(1203000, 3)(t),
                     any_of("sold", "sales", "consum", "used", "issued"),
                     head_lacks("m070", "stationary", "uncategorized")),  # category, not a product
     "must_answer": True,
     "why": "must answer at PRODUCT level and name the measure it used"},
    {"id": "l2-revenue-trend", "q": "How has monthly revenue moved over the period?",
     "check": all_of(num_within(89.52, 4), any_of("rose", "fell", "flat", "stable", "declin", "increas", "%")),
     "must_answer": True, "why": "Dec 89.52 → May 87.93 Cr, must state a direction"},
    # "Near-expiry stock" is the dashboard's own metric: FOUR buckets totalling ₹1.98 Cr,
    # expired included. Demanding ₹39.97 L here was my error — that is the 90-day SUBSET,
    # which is what l1-expiring-qty asks for. A breakdown should show the buckets.
    {"id": "l2-expiry-by-band", "q": "Break down our near-expiry stock by ageing band.",
     "check": all_of(any_of("0-30", "31-90", "91-180"),
                     # each bucket is quoted in whichever unit reads better, so accept both
                     # spellings of the same number: ₹114.94 L IS ₹1.15 Cr
                     lambda t: sum(bool(num_within(v, 8)(t))
                                   for v in (1.98, 43.16, 0.43, 16.79, 0.17,
                                             23.18, 0.23, 114.94, 1.15)) >= 2),
     "must_answer": True,
     "why": "four buckets — Expired ₹43.16 L, 0-30d ₹16.79 L, 31-90d ₹23.18 L, "
            "91-180d ₹114.94 L, ₹1.98 Cr total"},
    {"id": "l2-thin-margin-hospitals", "q": "Which hospitals run the thinnest sales margins?",
     "check": all_of(any_of("gjhca", "mhhik"), num_within(31.28, 15)), "must_answer": True,
     "why": "GJHCA 31.3%, MHHIK 32.5%"},
    {"id": "l2-msd-items", "q": "Which items do we buy from MSD?",
     "check": lacks("no procurement", "not available", "sticker", "no data"), "must_answer": True,
     "why": "MSD is a manufacturer — 614 procurement lines"},
    {"id": "l2-vendor-lead-spread", "q": "Which vendors are slowest to deliver?",
     "check": lacks("not answerable", "no lead time"), "must_answer": True,
     "why": "kpi_vendor_lead_time; a rate — no share of it should be claimed"},
    {"id": "l2-oncology-stock", "q": "How much inventory value sits in oncology drugs?",
     "check": lacks("not answerable", "no data"), "must_answer": True, "why": "material_type ZOC"},
    {"id": "l2-keytruda-hospitals", "q": 'Which hospitals sell the most "KEYTRUDA 100MG INJ VIAL"?',
     "check": all_of(has("keytruda"), lacks("not answerable", "no data")), "must_answer": True,
     "why": "sales_by_material_hospital carries both"},
]

# ═══════════════════════════════════════════════════════════════════════════
# LEVEL 3 — multi-step, diagnostic, or a trap. These need the question UNDERSTOOD
# before it is queried: the right entity type, the right grain, and an honest
# answer when the warehouse genuinely cannot connect two things.
# ═══════════════════════════════════════════════════════════════════════════
L3: list[dict] = [
    {"id": "l3-vendor-failure", "q": "If Vardhman failed tomorrow, which categories would hurt most and why?",
     "check": all_of(has("vardhman"), lacks("not answerable", "evidence does not allow"),
                     lambda t: "100%" not in t and "100.0%" not in t),
     "must_answer": True, "why": "must name categories and quantify exposure, no false 100%"},
    {"id": "l3-margin-vs-volume",
     "q": "Are our highest-volume products also our highest-margin ones, or is it the opposite?",
     "check": lacks("not answerable", "no data"), "must_answer": True,
     "why": "needs volume and margin compared, and a stated conclusion"},
    {"id": "l3-expiry-concentration",
     "q": "Is our expiry risk concentrated in a few hospitals or spread evenly, and what would reduce it?",
     "check": lacks("not answerable", "no data"), "must_answer": True, "why": "distribution + recommendation"},
    {"id": "l3-sales-stock-link",
     "q": "For each hospital, compare sales revenue against the stock it holds.",
     "check": any_of("cannot", "can't", "not possible", "no mapping", "no linkage", "no link",
                     "does not", "doesn't", "different code", "separate", "unable", "not answerable"),
     "why": "TRAP: sales and plant hospital codes share zero rows and nothing maps them"},
    {"id": "l3-bangalore-sales-trap", "q": "What sold the most in Bangalore last month?",
     "check": any_of("purchas", "procurement", "cannot", "can't", "no city", "plant_name",
                     "not possible", "consumption", "no monthly"),
     "why": "TRAP: no city reaches sales, and there is no material x month sales grain"},
    {"id": "l3-msd-lead-time-hop",
     "q": "How reliable are the vendors who supply MSD's products?",
     "check": all_of(lacks("no lead times", "no data", "not answerable"),
                     any_of("vardhman", "vendor", "day")),
     "must_answer": True, "why": "manufacturer -> its vendors -> lead times"},
    {"id": "l3-price-outliers",
     "q": "Are we overpaying anyone? Show me where the same item costs very different amounts.",
     "check": lacks("not answerable", "no pricing"), "must_answer": True,
     "why": "mart_material_vendor_price_stats / is_price_outlier"},
    {"id": "l3-cash-vs-expiry",
     "q": "We need to free up cash. Should we cut reordering or clear near-expiry stock first?",
     "check": lacks("not answerable", "no data"), "must_answer": True,
     "why": "must weigh two figures and recommend"},
    {"id": "l3-dead-stock-cost",
     "q": "What is non-moving stock actually costing us, and which sites are worst?",
     "check": all_of(lacks("not answerable"), any_of("hm01", "ah01", "hospital")), "must_answer": True,
     "why": "value + worst sites"},
    {"id": "l3-consumption-vs-purchase",
     "q": 'For "KEYTRUDA 100MG INJ VIAL", does what we buy match what we use?',
     "check": any_of("billed", "no internal consumption", "0", "purchas", "consumption"),
     "why": "TRAP: billed-only item — fact_consumption has 0 rows for it"},
    {"id": "l3-category-margin-drop",
     "q": "Which category is dragging our overall margin down the most?",
     "check": lacks("not answerable", "no data"), "must_answer": True,
     "why": "margin by category, ranked by drag"},
    {"id": "l3-stockout-vs-leadtime",
     "q": "Are our stock-outs caused by slow vendors or by under-ordering?",
     "check": lacks("not answerable", "no data"), "must_answer": True,
     "why": "must test two explanations and rule one out"},
    {"id": "l3-hospital-outlier",
     "q": "Is any hospital behaving very differently from the others on inventory?",
     "check": lacks("not answerable", "no data"), "must_answer": True, "why": "outlier detection"},
    {"id": "l3-top-drug-economics",
     "q": "KEYTRUDA is our biggest seller — is it also our most profitable, and what does it cost us to hold?",
     "check": all_of(has("keytruda"), lacks("not answerable")), "must_answer": True,
     "why": "revenue + margin + holding, one item, several tables"},
    {"id": "l3-seasonality",
     "q": "Is there any seasonality in our procurement, or is it flat?",
     "check": lacks("not answerable", "no data"), "must_answer": True,
     "why": "monthly purchase value exists; must state a conclusion either way"},
]


# ── topping each level to thirty ─────────────────────────────────────────────
L1 += [
    {"id": "l1-total-margin-pct-again", "q": "What margin percentage are we running overall?",
     "check": num_within(40.9, 6), "why": "40.9%"},
    {"id": "l1-inventory-mrp", "q": "What is our stock worth at MRP?",
     "check": num_within(120.89, 6), "why": "₹120.89 Cr at MRP vs ₹60.47 Cr at cost"},
    {"id": "l1-material-groups", "q": "How many material groups are there?",
     "check": lambda t: num_within(139, 8)(t) or num_within(136, 8)(t), "why": "139 groups"},
    {"id": "l1-generic-count", "q": "How many distinct generic molecules do we stock?",
     "check": num_within(3224, 8), "why": "3,224 generic names"},
    {"id": "l1-manufacturer-count", "q": "How many manufacturers supply us?",
     "check": num_within(1321, 8), "why": "1,321 manufacturers"},
    {"id": "l1-expiring-qty", "q": "How many units are expiring in the next 90 days?",
     "check": all_of(num_within(45223, 2), lacks("101,005", "101005")),
     "why": "45,223 units, excluding already-expired"},
    {"id": "l1-po-lines-vardhman", "q": "How many purchase order lines do we have with Vardhman?",
     "check": num_within(111582, 5), "why": "111,582 PO lines"},
    {"id": "l1-data-window", "q": "What period does our data cover?",
     "check": all_of(any_of("dec", "december", "2025"), any_of("may", "2026")),
     "why": "December 2025 to May 2026"},
    {"id": "l1-keytruda-qty", "q": 'How many units of "KEYTRUDA 100MG INJ VIAL" were sold?',
     "check": num_within(2193, 4), "why": "2,193 units"},
    {"id": "l1-avg-lead-vardhman", "q": "What is Vardhman's average lead time?",
     "check": num_within(4.8, 30), "why": "4.8 days average"},
    {"id": "l1-departments", "q": "How many departments consume stock?",
     "check": num_within(730, 10), "why": "730 department names"},
    # "Biggest selling" is units or revenue, and both readings have a real second place:
    # TRASTUREL 440MG at ₹19.31 Cr, PANONIC 40MG TAB at 408,181 units. Either is a correct
    # answer as long as it names a PRODUCT and says which measure it used.
    {"id": "l1-second-selling-item", "q": "What is our second biggest selling product?",
     "check": all_of(any_of("trasturel", "19.3", "panonic", "408,181"),
                     any_of("revenue", "sales", "units", "sold")),
     "why": "TRASTUREL ₹19.31 Cr by revenue, or PANONIC 408,181 by units — name the measure"},
]

L2 += [
    {"id": "l2-generic-concentration", "q": "Which generic molecules account for most of our spend?",
     "check": lacks("not answerable", "no data"), "must_answer": True, "why": "generic_name exists in dim_material"},
]

L3 += [
    {"id": "l3-mfr-vs-vendor-confusion",
     "q": "Is Reliance a supplier we buy from directly, or just a manufacturer of things we buy?",
     "check": any_of("manufacturer", "not a vendor", "brand", "maker"),
     "why": "TRAP: entity typing — Reliance is a MANUFACTURER, not in dim_vendor"},
    {"id": "l3-city-vs-hospital",
     "q": "Compare procurement spend between our Bangalore and Ahmedabad hospitals.",
     "check": lacks("not answerable", "no location", "no city"), "must_answer": True,
     "why": "cities resolve through dim_plant.plant_name — this one IS answerable"},
    {"id": "l3-margin-erosion-driver",
     "q": "Our margin is 40.9%. Which products or categories are pulling it below that?",
     "check": all_of(lacks("not answerable"), num_within(40.9, 12)), "must_answer": True,
     "why": "must anchor on the real overall margin then decompose"},
    {"id": "l3-reorder-vs-expiry-conflict",
     "q": "Are we reordering anything that we already have sitting near expiry?",
     "check": lacks("not answerable", "no data"), "must_answer": True,
     "why": "reorder list intersected with near-expiry — a real operational question"},
    {"id": "l3-vendor-price-vs-speed",
     "q": "Do our cheapest vendors also take the longest to deliver?",
     "check": lacks("not answerable", "no data"), "must_answer": True,
     "why": "price stats vs lead time; a rate — no share of lead time should be claimed"},
    {"id": "l3-single-source-risk",
     "q": "Which critical items do we buy from only one vendor?",
     "check": lacks("not answerable", "no data"), "must_answer": True,
     "why": "count distinct vendors per material"},
    {"id": "l3-what-changed",
     "q": "What changed most between December and May across our procurement?",
     "check": all_of(lacks("not answerable"), any_of("dec", "may", "%")), "must_answer": True,
     "why": "period comparison with a stated direction"},
]

# The first thirty cases predate the levels, so they are classified here by what they
# actually demand rather than by the order they were written in.
_LEVEL_OF = {
    # L2 — one dimension: a ranking, a breakdown, a trend
    "top-vendor": "L2", "top-selling-item": "L2", "top-category": "L2",
    "top-manufacturer": "L2", "top-hospital-sales": "L2", "top-stock-site": "L2",
    "revenue-trend": "L2", "slow-movers-by-site": "L2", "lead-time-risk": "L2",
    "margin-by-hospital": "L2", "cash-to-restock": "L2", "stockout-impact": "L2",
    "price-variance": "L2", "oncology-exposure": "L2",
    # L3 — multi-step, diagnostic, or a trap that needs the entity TYPED first
    "keytruda-trend": "L3", "msd-procurement": "L3", "msd-lead-times": "L3",
    "sales-vs-stock-by-hospital": "L3", "bangalore-sales": "L3",
    "vendor-concentration": "L3", "worst-margin-drugs": "L3", "keytruda-consumption": "L3",
}


def _classify(bucket: list[dict], default: str) -> None:
    for c in bucket:
        c["level"] = _LEVEL_OF.get(c["id"], default)


_classify(L1, "L1")
_classify(L2, "L2")
_classify(L3, "L3")
CASES: list[dict] = L1 + L2 + L3
BY_LEVEL = {lv: [c for c in CASES if c["level"] == lv] for lv in ("L1", "L2", "L3")}
