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
CASES: list[dict] = [
    # ── canonical figures: these must be exact ───────────────────────────────
    {"id": "expiry-90d", "q": "How much stock is expiring in the next 90 days?",
     "check": all_of(num_within(39.97), lacks("83.13", "101,005")),
     "why": "canonical: ₹39.97 L / 45,223 units (0-30d + 31-90d). 83.13 L would mean already-expired stock was included"},

    {"id": "total-sales", "q": "What is our total sales revenue and margin?",
     "check": all_of(num_within(521.67, 3), num_within(213.43, 3)),
     "why": "₹521.67 Cr revenue, ₹213.43 Cr margin"},

    {"id": "overall-margin", "q": "What is our overall margin percentage on sales?",
     "check": num_within(40.9, 5), "why": "40.9%"},

    {"id": "stock-value", "q": "What is the total value of stock we are holding right now?",
     "check": num_within(60.47, 5), "why": "₹60.47 Cr across 18,080 SKUs"},

    {"id": "non-moving-count", "q": "How many SKUs are non-moving?",
     "check": num_within(10501, 3), "why": "10,501 non-moving SKUs"},

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
                     lambda t: num_within(43.16, 8)(t) or num_within(83.13, 8)(t),
                     lacks("39.97")),
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
