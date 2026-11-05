"""Executive overview — HCG's pharmacy on one screen: sales and margin, formulary compliance,
the margin and saving missed, what was paid above the usual cheapest, every unit side by side,
and the insights and actions those figures support.

    GET /executive/overview   -> scope, months, six KPIs (each with its previous month, change,
                                 sparkline and source), the monthly trend, the margin bridge,
                                 IP / OP, the units table, the formulary mix, a treemap and a
                                 flow of sales, the top items and units (and the bottom ones,
                                 by margin %), insights, actions, notes

Nothing here re-derives a figure another section owns. Each one comes from that section's own
handler, called in process the way app/ai/hcg_reports.py calls them, so a figure agrees with
the page its link opens for the same month and filters (each link carries this page's month and
the filters that page takes, LINK_FILTERS; a section's own All months can hold more months than
this page's, Prices every purchase month):

    sales, margin, IP / OP          revenue.revenue_overview           HIS sales lines, at MRP
    compliance, mix, unit split     formulary.formulary_overview       drug cost (units on qty)
    cash & TPA margin missed        spend.spend_cash_overview          sales lines, HCG's FY list
    scheme saving missed            spend.spend_scheme_overview        sales lines, HCG's FY list
    out-of-formulary margin missed  spend.spend_oof_overview           PURCHASES (goods receipts)
    paid above the usual cheapest   purchase_insights.spend_prices_overview   PURCHASES
The treemap, the flow and the top lists group the same sales_items rows the revenue page reads.

The sections hold different months: sales and the spend tables one per sales month, purchases
one per GRN month. A section without the month asked for, or not built on this server, leaves
its KPI, part or column null and says why in `notes` — never an error. Month=ALL takes every
month on the page (the sales months and the out-of-formulary purchase months together); each
section sums the ones it holds, and no change is shown. The previous month a change is taken
against is the previous month with sales, for every KPI alike. An out-of-formulary month built
without HCG's own working is Estimated, and a month whose purchases wait for their Item Master
has no figure yet: the losses KPI, the bridge and the notes say so beside that part.

Basis=qty counts units sold for the sales and formulary measures. Units add tablets, vials and
bottles together, so nothing in units is set against HCG's budget (a drug-cost target), and the
money-only measures — margin, what was missed, extra paid — stay in rupees and say so.

Insights and actions are fixed rules over these figures (no language model, so they read the
same with or without the AI key): exact figures in HCG's number style, months by their label,
and a link to the page that shows them. A rule whose data is missing leaves itself out. An
insight's tone says whether the news is good or bad, not which way the figure moved: "up" =
better (compliance up, a loss down), "down" = worse, "alert" = short of a target or money
missed, "info" = for the record.
"""
from __future__ import annotations

import math
import numbers
import re
from decimal import ROUND_HALF_UP, Decimal
from typing import Callable, NamedTuple, Optional
from urllib.parse import urlencode

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from app.api import formulary as F
from app.api import purchase_insights as PI
from app.api import revenue as RV
from app.api import spend as S
from app.etl.formulary import config as FC
from app.etl.purchase import config as PC
from app.etl.purchase import oof as PO

router = APIRouter()

# ── Rules, thresholds and links ───────────────────────────────────────────────
# The status rule every KPI card draws (hcg-analytics-ui ledger/status.ts `classify`), ported so a
# unit's status here is the one its card shows. HCG's MIS prints whole percentages, so a figure
# meets its budget when it PRINTS at or above it (97.5% meets 98%); up to WITHIN_PTS below that
# is "within", further is "below". Change one, change status.ts with it.
WITHIN_PTS = 2.0                # status.ts WITHIN_PTS: the amber band, agreed with the user 15-10-2026
PRINT_ROUNDING = 0.5            # status.ts PRINT_ROUNDING
# The smallest change that counts as a move (status.ts DELTA_THRESHOLD, formulary summaries.ts
# STEADY_PTS); anything smaller reads as "about the same" and gives no up / down tone.
STEADY_PTS = 0.1                # a % moving less than 0.1 pts
STEADY_PCT = 0.5                # an amount moving less than 0.5%
MAX_INSIGHTS = 6                # the strip's length (contract C6)
MAX_ACTIONS = 4                 # the actions list's length (contract C6)
FLOW_UNITS = 5                  # D11: the flow's five biggest units, then "Other units"
TOP_N = 10                      # top and bottom items and units listed, like HCG's top-10 lists
# The bottom lists (lowest margin %) take an item or unit only when it carries at least this %
# of the selection's sales at MRP, so a line of a few rupees sold at a loss cannot fill them.
BOTTOM_MIN_SHARE_PCT = 0.1
FOLLOW_UP_UNITS = 3             # units the follow-up action names, furthest below budget first
NAMED_UNITS = 3                 # units an insight names before "and N more"

LINKS = {"formulary": "/formulary?view=overview", "revenue": "/revenue",
         "cash": "/spend?section=cash&view=overview", "oof": "/spend?section=oof&view=overview",
         "prices": "/spend?section=prices"}
# The filters each linked page takes, carried in its link beside its section and view, so it
# opens on this page's month (one YYYY-MM, or ALL) and filters; purchases have no patient type.
# The hospital and the Value / Units switch are the header's, app-wide, and need no carrying.
LINK_FILTERS = {"formulary": ("month", "type", "segment"), "revenue": ("month", "type", "segment"),
                "cash": ("month", "type", "segment"), "oof": ("month", "segment"), "prices": ("month", "segment")}
# Opens the Prices page on its list of unusually low purchases to check (/spend/prices/low).
LINK_CHECK = {"check": "1"}
# The sections a link opens, by the source each KPI and note names (scheme is on the cash page).
LINKED_SOURCES = {"sales": "revenue", "formulary": "formulary", "cash": "cash", "oof": "oof", "prices": "prices"}

# Each section in plain words, as the notes name it, and the ones named in the plural.
SOURCE_LABELS = {"sales": "Pharmacy sales", "formulary": "Formulary compliance",
                 "cash": "Cash & TPA margin missed", "scheme": "Scheme saving missed",
                 "oof": "Out-of-formulary purchases", "prices": "Prices paid on purchases"}
PLURAL_SOURCES = ("sales", "oof", "prices")
# The sections a consumables selection leaves out: compliance is drugs only, and HCG's purchase
# scope keeps the drug series only (app/data/reference/purchase_scope_exclusions.csv).
DRUGS_ONLY = ("formulary", "oof", "prices")
TONE_ORDER = {"alert": 0, "down": 1, "up": 2, "info": 3}
BASIS_MEASURE = {"value": "mrp", "qty": "qty"}          # what sales, the treemap and the flow sum
RANK_MEASURE = {"value": "cost", "qty": "qty"}          # what the top lists rank on (formulary's own)
CATEGORY_TYPE = {c: k for k, c in RV.SEGMENTS.items()}  # 'Onco Drugs' -> 'onco'

KPI_LABELS = {"sales": "Revenue (MRP)", "margin_pct": "Margin %", "compliance": "Formulary compliance",
              "oof_share": "Out-of-formulary share", "losses": "Margin & saving missed",
              "extra_paid": "Paid above the usual cheapest"}
UNITS_SOLD_LABEL = "Units sold"
# With one hospital, extra paid is set against that hospital's OWN usual cheapest (the Prices
# page's headline), while its units-table row is set against all hospitals': the label says which.
EXTRA_PAID_OWN_LABEL = "Paid above its own usual cheapest"
# Where a KPI's figure is counted (C6 `source`); the losses KPI lists every part's (`sources`).
LOSS_SOURCES = {"cash": "sales", "scheme": "sales", "oof": "purchases"}

NOTE_SOURCES = ("Every figure is its own section's: revenue and margin from Revenue & margin, compliance and the "
                "mix from Formulary compliance, what was missed from Spend analysis (cash & TPA, scheme, out of "
                "formulary) and extra paid from Prices. A link opens its page on this page's month and filters, and "
                "a figure matches that page for the same month and filters. A section's own All months can cover "
                "more months than this page's (Prices holds every purchase month), so its figures there can differ; "
                "the notes then say so.")
NOTE_PREVIOUS = "Changes are against {label}, the previous month with sales."
NOTE_FIRST = "{label} is the first month with sales, so no change is shown."
NOTE_ALL = ("All months: every month on this page ({span}). Each figure sums the months its section holds, "
            "amounts first and every % taken once, except extra paid, which is set against one usual cheapest "
            "over those months; no change is shown.")
# Under All months a link opens its page on that page's own All months, which holds every month
# of its section: a section with months this page does not show (purchases before the first
# sales month) has other figures there.
NOTE_LINK_MONTHS = ("On {its} own page, All months for {what_lc} covers {span}, more months than here, so {its} "
                    "figures there differ.")
NOTE_PURCHASES = "Out of formulary and extra paid are counted on purchases (goods receipts){same}, not on sales."
SAME_MONTHS = " of the same months"
# Parts of one figure that hold different months (all months, a section built ahead of another).
NOTE_WINDOWS = "The months differ: {groups}."
NOTE_ONE_WINDOW = "{groups} only."
NOTE_PURCHASES_TYPE = "Purchases have no patient type, so the IP/OP filter does not apply to them."
NOTE_UNITS_VIEW = ("Quantity view: sales, compliance, the out-of-formulary share, the mix, the treemap, the flow and "
                   "the top lists count units sold (tablets, vials, bottles and so on added together), and nothing "
                   "counted in units sold is set against HCG's budget. Margin, what was missed and extra paid stay in rupees.")
NOTE_BUDGET_FILTERS = ("HCG sets each unit's compliance budget for its whole drug spend, so it is not compared under "
                       "the IP/OP or segment filter.")
NOTE_CONSUMABLES = ("Consumables have no formulary compliance and are not in the purchase analysis, so compliance, "
                    "the out-of-formulary share, the mix, out of formulary and extra paid are left out.")
NOTE_UNITS_TABLE = ("Units: every unit in HCG's PAN unit list, and any other unit with figures in this selection, "
                    "in HCG's own order. The table ignores the hospital selector and follows the other filters; a "
                    "unit's extra paid is set against the usual cheapest of all hospitals.")
NOTE_STATUS = (f"Status against budget: met when compliance prints at or above it (HCG's MIS prints whole %), "
               f"within when up to {WITHIN_PTS:g} pts below that, below beyond it.")
NOTE_TOP = {"value": ("Top items and units rank consumption the way the formulary pages do, on cost price, not on "
                      "MRP; consumables are included. The treemap and the flow are on MRP."),
            "qty": "Top items and units, the treemap and the flow all count units sold; consumables are included."}
NOTE_BOTTOM = (f"Bottom items and units: the lowest margin % (margin ÷ MRP, in rupees whatever the Value / Units "
               f"switch) among those with at least {BOTTOM_MIN_SHARE_PCT:g}% of the selection's sales at MRP, so a "
               f"line of a few rupees cannot fill the list.")
# D6: what the sales and margin figures are, as the Revenue page says it (revenue.NOTE_GST).
NOTE_GST_KPI = "The HIS file does not say whether its MRP and cost include GST."
# The usual cheapest as the Prices page words it (spend insights.tsx usualCheapestRule); {share} is
# the prices answer's own rule (usual_share_pct), never a literal.
NOTE_EXTRA_PAID = ("Paid above each product's usual cheapest price{at} (the price reached once at least {share} of "
                   "its units have been counted, cheapest purchase first, so one odd cheap purchase cannot set it), "
                   "on purchases.")
NOTE_BLANK_COST_KPI = ("{lines} {noun} with a blank cost price (MRP {mrp}) {count} as cost 0, as HCG's sheet does, "
                       "so that MRP is margin in full.")
NOTE_RULES = ("Insights and actions are fixed rules over these figures, not a language model; actions are "
              "listed by the rupees a switch or a better price would have kept.")
NOTE_NOT_BUILT = "{what} {verb} not built on this server yet, so {it} {is_} left out."
NOTE_NO_MONTH = "{what} {has} no figures for {when} ({it} {holds} {held}), so {it} {is_} left out."
NOTE_PART_MONTHS = "{what} {covers} {held} only: {its} figures are for those months."
# An out-of-formulary month built without HCG's own working (its suggestions come from HCG's other
# months and lists, then the app's match), as the Spend page's out-of-formulary tiles call it.
NOTE_OOF_ESTIMATED = "Out of formulary: Estimated {month} (HCG's working for {span} not received)."

BRIDGE_LABELS = {"earned": "Margin earned on sales", "cash": "Cash & TPA margin missed",
                 "scheme": "Scheme saving missed", "oof": "Out-of-formulary margin missed",
                 "possible": "Margin with every switch"}
NOTE_BRIDGE = ("From the margin earned on sales (MRP − cost) to the margin with every switch HCG suggests. Cash & "
               "TPA and scheme are worked out on the same sales lines at HCG's FY list prices; out of formulary on "
               "purchases (goods receipts){same}, so the last bar is an indication, not one ledger.")
NOTE_BRIDGE_RUPEES = "The bridge is in rupees whatever the Value / Units switch."
NOTE_BRIDGE_LEFT_OUT = "Left out of the bridge, for want of figures for this selection: {parts}."


# ── HCG's number style (hcg-analytics-ui ledger/format.ts, ported) ────────────
LAKH, CRORE = 1e5, 1e7
MINUS = "−"                     # format.ts prints a minus sign, never a hyphen
NO_FIGURE = "—"


def _finite(v) -> bool:
    # numbers.Real takes numpy's scalars too (a pandas sum is numpy.float64 or numpy.int64)
    return isinstance(v, numbers.Real) and not isinstance(v, bool) and math.isfinite(float(v))


def _half_up(x: float) -> int:
    """JavaScript's Math.round for x >= 0 (half up), not Python's half-to-even round()."""
    return int(math.floor(x + 0.5))


def _fixed(x: float, digits: int) -> str:
    """JavaScript's x.toFixed(digits) for x >= 0: an exact tie goes up (Python's format would
    round 10.125 to 10.12; the page prints 10.13)."""
    return str(Decimal(x).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP))


def _indian(n: int) -> str:
    """Whole number in Indian grouping: 12345678 -> '1,23,45,678'."""
    s = str(abs(int(n)))
    head, tail = s[:-3], s[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    if head:
        groups.insert(0, head)
    return ",".join(groups + [tail])


def _with_minus(v: float, body: str) -> str:
    # a figure that rounds to zero never reads "−0"
    return f"{MINUS}{body}" if v < 0 and re.search(r"[1-9]", body) else body


def lakhs(v) -> str:
    """Rupees the way HCG's dashboards print them (format.ts `lakhs`): a crore and above in whole
    lakhs ('₹ 9,431 L'), a lakh up to a crore in lakhs to 2 dp ('₹ 32.43 L'), below a lakh in
    whole rupees ('₹ 31,030'). The band is chosen on the printed value."""
    if not _finite(v):
        return NO_FIGURE
    a = abs(float(v))
    lk = a / LAKH
    if a >= CRORE or _half_up(lk * 100) >= 10000:
        body = f"₹ {_indian(_half_up(lk))} L"
    elif _half_up(a) >= LAKH:
        body = f"₹ {_fixed(lk, 2)} L"
    else:
        body = f"₹ {_indian(_half_up(a))}"
    return _with_minus(v, body)


def unit_price(v) -> str:
    """A unit price as paid (spend insights.tsx `unitPrice`): '₹25,599', '₹18,938.10'."""
    if not _finite(v):
        return NO_FIGURE
    a = abs(float(v))
    whole = abs(a - _half_up(a)) < 0.005
    i, _, frac = _fixed(a, 0 if whole else 2).partition(".")
    return _with_minus(v, "₹" + _indian(int(i)) + (f".{frac}" if frac else ""))


def pct(v) -> str:
    return _with_minus(v, f"{_fixed(abs(float(v)), 2)}%") if _finite(v) else NO_FIGURE


def pts(v) -> str:
    return _with_minus(v, f"{_fixed(abs(float(v)), 2)} pts") if _finite(v) else NO_FIGURE


def budget_text(v) -> str:
    """A budget as HCG writes it: '100%', '97.5%'."""
    return f"{v:g}%" if _finite(v) else NO_FIGURE


def units_text(v) -> str:
    """'44,90,262 units', '1 unit'."""
    if not _finite(v):
        return NO_FIGURE
    n = _half_up(abs(v))
    return f"{_indian(n)} {_plural(n, 'unit', 'units')}"


def _plural(n: int, one: str, many: str) -> str:
    return one if n == 1 else many


def _names_list(names: list) -> str:
    """'A', 'A; B', 'A; B; C; and 2 more'. Semicolons, because hospital names carry commas
    ('HCG Hospitals, Ahmedabad')."""
    shown = names[:NAMED_UNITS]
    rest = len(names) - len(shown)
    return "; ".join(shown) + (f"; and {rest} more" if rest else "")


def product_name(p: dict) -> str:
    """'NIVOLUMAB 100MG INJ', in HCG's molecule name (spend insights.tsx `productName`)."""
    strength = "|".join(x for x in str(p.get("strength") or "").split("|") if x).replace("|", " + ")
    return " ".join(x for x in (p.get("minor_group") or p.get("molecule") or "Unnamed product",
                                strength or None, p.get("form"), p.get("kind")) if x)


# ── Rounding (2 dp money and %, 3 dp quantities; None for anything not finite) ─
def _money(v) -> Optional[float]:
    return round(float(v), 2) + 0.0 if _finite(v) else None


def _qty(v) -> Optional[float]:
    return round(float(v), 3) + 0.0 if _finite(v) else None


def _p2(v) -> Optional[float]:
    return round(float(v), 2) + 0.0 if _finite(v) else None


def _share(part, whole) -> Optional[float]:
    return _p2(float(part) / float(whole) * 100) if _finite(part) and _finite(whole) and whole else None


def _change(value, prev, unit: str) -> Optional[float]:
    """value against prev: points for a %, a % change for money and quantities (None when the
    previous amount is 0 or either side is missing)."""
    if not _finite(value) or not _finite(prev):
        return None
    if unit == "pts":
        return _p2(value - prev)
    return _p2((value - prev) / abs(prev) * 100) if prev else None


def _direction(value, prev, delta, steady: float) -> int:
    """+1 up, -1 down, 0 about the same: a change under `steady` (pts, or % of the previous
    amount) is no move; from a previous amount of 0 any amount is one."""
    if not _finite(value) or not _finite(prev):
        return 0
    if _finite(delta):
        return 0 if abs(delta) < steady else (1 if delta > 0 else -1)
    return 0 if value == prev else (1 if value > prev else -1)


def classify(value, budget) -> Optional[str]:
    """status.ts `classify`: 'met' / 'within' / 'below'; None without both figures."""
    if not _finite(value) or not _finite(budget):
        return None
    if value >= budget - PRINT_ROUNDING:
        return "met"
    if value >= budget - WITHIN_PTS - PRINT_ROUNDING:
        return "within"
    return "below"


# ── Arguments ─────────────────────────────────────────────────────────────────
class Scope(NamedTuple):
    plant: Optional[str]          # SAP plant code; None => all hospitals
    month: Optional[str]          # one 'YYYY-MM'; None => every month on the page (ALL)
    ptype: Optional[str]          # 'IP' / 'OP'; None => both
    segment: Optional[str]        # 'onco' / 'non-onco' / 'consumables'; None => all
    basis: str                    # 'value' / 'qty' (formulary.BASES)


def _try(fn: Callable):
    """fn(), or None when its section is not built (503)."""
    try:
        return fn()
    except HTTPException as e:
        if e.status_code == 503:
            return None
        raise


def _oof_months() -> list:
    """Months the out-of-formulary purchase build holds ([] until it has run)."""
    return (_try(S._oof_periods) or []) if S._has_oof() else []


def _page_months(sales: list) -> list:
    return sorted(set(sales) | set(_oof_months()))


def _month_arg(month, page: list, sales: list) -> Optional[str]:
    """One YYYY-MM on the page, ALL (None), or the latest sales month when not given."""
    s = F._str(month)
    if s is None:
        return sales[-1]
    if s.upper() == "ALL":
        return None
    if not F._PERIOD_RE.match(s):
        F._bad(f"Invalid Month '{s}'. Use one month as YYYY-MM, or ALL")
    if s not in page:
        F._bad(f"No data for {s}. Available months: {', '.join(page)}")
    return s


def _scope(Plant, Month, Type, Segment, Basis) -> Scope:
    sales = RV.sales_periods()              # 503 first: the page starts from the sales tables
    F._periods()                            # built beside them by the same formulary build
    return Scope(plant=F._plant_arg(Plant), month=_month_arg(Month, _page_months(sales), sales),
                 ptype=F._choice(Type, {t: t for t in F.TYPES}, "Type"),
                 segment=F._choice(Segment, RV._SEGMENT_IN, "Segment"), basis=F._basis_arg(Basis))


def version() -> tuple:
    """Identity of every table behind the page: a rebuild of any section changes it."""
    return (RV.version(), S._version(), PI._version(), F._stamp(F._data_dir() / f"{FC.MIS_UNITS}.parquet"))


# ── Months in words ───────────────────────────────────────────────────────────
def _span(months) -> str:
    """'Aug-26', 'Jul-26 to Aug-26', or 'May-26, Jun-26, Aug-26' when not one run of months."""
    months = sorted(months)
    if not months:
        return "no month"
    index = [int(m[:4]) * 12 + int(m[5:7]) for m in months]
    if len(months) > 1 and index[-1] - index[0] == len(months) - 1:
        return f"{F._month_label(months[0])} to {F._month_label(months[-1])}"
    return ", ".join(F._month_label(m) for m in months)


def _when(months) -> str:
    """'in Aug-26', 'from Jul-26 to Aug-26', 'in Jun-26, Aug-26'."""
    span = _span(months)
    return f"from {span}" if " to " in span else f"in {span}"


def _previous(month: Optional[str], sales: list) -> Optional[str]:
    """The sales month before `month` (one month only); None for ALL or the first month."""
    if month is None:
        return None
    before = [p for p in sales if p < month]
    return before[-1] if before else None


def _and(names: list) -> str:
    """'A', 'A and B', 'A, B and C'."""
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def _windows_note(parts: list, whole: tuple) -> Optional[str]:
    """Which months the parts of one figure hold, when they are not all `whole`. `parts`:
    [(name in words, months held)]. All on one other window: 'Cash & TPA and scheme cover
    Jul-26 to Aug-26 only.'; on several: 'The months differ: margin earned covers Jun-26 to
    Aug-26; out of formulary covers Jun-26 to Sep-26.' None when every part holds `whole`."""
    groups: dict = {}
    for words, held in parts:
        groups.setdefault(tuple(held), []).append(words)
    if not groups or list(groups) == [tuple(whole)]:
        return None
    said = [f"{_and(names)} {'covers' if len(names) == 1 else 'cover'} {_span(held)}" for held, names in groups.items()]
    if len(said) == 1:
        return NOTE_ONE_WINDOW.format(groups=said[0][:1].upper() + said[0][1:])
    return NOTE_WINDOWS.format(groups="; ".join(said))


def _link(sc: Scope, key: str, **extra: str) -> str:
    """The app path of a section's page (LINKS) with this page's month and the filters that page
    takes (LINK_FILTERS), so the page it opens shows the same selection: '/spend?section=oof&
    view=overview&month=2026-08&segment=onco'. `extra` adds the page's own parameters."""
    have = {"month": sc.month or "ALL", "type": sc.ptype, "segment": sc.segment}
    params = [(k, have[k]) for k in LINK_FILTERS[key] if have[k]] + list(extra.items())
    base = LINKS[key]
    return f"{base}{'&' if '?' in base else '?'}{urlencode(params)}" if params else base


# ── The sections, called in process ──────────────────────────────────────────
class Part(NamedTuple):
    data: Optional[dict]          # the section's overview of the selection; None when it has none
    months: tuple                 # the selection's months the section holds
    series: Optional[dict]        # an overview to read its every-month trend from (Month aside)
    why: Optional[str]            # plain words: why it has no figure, or which months it covers
    # out of formulary only: its months' state in plain words (estimated, waiting for an Item
    # Master), which the losses KPI and the bridge repeat beside its figure
    status: tuple = ()
    held: tuple = ()              # every month the section holds (its own page's All months)


def _sentence(name: str, template: str, **kw) -> str:
    """A note about one section, its verbs agreeing with its name ('Pharmacy sales have …')."""
    many = name in PLURAL_SOURCES
    label = SOURCE_LABELS[name]
    return template.format(what=label, what_lc=label[:1].lower() + label[1:], verb="are" if many else "is",
                           it="they" if many else "it", is_="are" if many else "is", has="have" if many else "has",
                           covers="cover" if many else "covers", holds="hold" if many else "holds",
                           its="their" if many else "its", **kw)


def _fetch(name: str, held: Optional[list], months: tuple, call: Callable[[str], dict]) -> Part:
    """One section for the selection's months it holds (`call` takes Month=). A section not
    built, without any of those months, or refusing the selection (its 400, plain words) has
    no figure and says why; its every-month trend is still read from its latest month."""
    if held is None:
        return Part(None, (), None, _sentence(name, NOTE_NOT_BUILT))
    have = tuple(m for m in months if m in held)
    try:
        if not have:
            series = call(held[-1]) if held else None
            return Part(None, (), series, _sentence(name, NOTE_NO_MONTH, when=_span(months), held=_span(held)),
                        held=tuple(held))
        data = call(",".join(have))
    except HTTPException as e:
        if e.status_code == 503:
            return Part(None, (), None, _sentence(name, NOTE_NOT_BUILT))
        if e.status_code == 400:
            return Part(None, (), None, f"{SOURCE_LABELS[name]}: {e.detail}", held=tuple(held))
        raise
    partial = len(months) > 1 and len(have) < len(months)
    return Part(data, have, data, _sentence(name, NOTE_PART_MONTHS, held=_span(have)) if partial else None,
                held=tuple(held))


def _sources(sc: Scope, months: tuple) -> dict:
    held = {"sales": RV.sales_periods(), "formulary": F._periods(), "cash": _try(S._periods),
            "oof": _try(S._oof_periods) if S._has_oof() else None, "prices": _try(lambda: PI._periods(PC.PRICES))}
    held["scheme"] = held["cash"]
    drugs = dict(Plant=sc.plant, Segment=sc.segment)
    sales = dict(drugs, Type=sc.ptype)
    calls = {
        "sales": lambda m: RV.revenue_overview(Month=m, **sales),
        "formulary": lambda m: F.formulary_overview(Month=m, Basis=sc.basis, **sales),
        "cash": lambda m: S.spend_cash_overview(Month=m, **sales),
        "scheme": lambda m: S.spend_scheme_overview(Month=m, **sales),
        # purchases have no patient type: Type is not sent (their notes would only say so)
        "oof": lambda m: S.spend_oof_overview(Month=m, **drugs),
        "prices": lambda m: PI.spend_prices_overview(Month=m, **drugs),
    }
    out = {}
    for name, call in calls.items():
        if sc.segment == "consumables" and name in DRUGS_ONLY:
            out[name] = Part(None, (), None, NOTE_CONSUMABLES)
        else:
            out[name] = _fetch(name, held[name], months, call)
    if sc.segment != "consumables":
        out["oof"] = _oof_status(out["oof"], months)
    return out


def _oof_status(part: Part, months: tuple) -> Part:
    """The out-of-formulary part with its months' state, as the Spend page's out-of-formulary
    view says it: an estimated month, built without HCG's own working (the overview's
    headline.estimated_months), and a month of the selection whose purchases wait for their Item
    Master (/spend/meta oof_waiting), which has no figure yet. The sentences follow the part's
    own note (`why`, read into the page's notes) and are kept in `status`."""
    est = ((part.data or {}).get("headline") or {}).get("estimated_months") or []
    waiting = set(_try(S._oof_waiting) or ())
    status = ([NOTE_OOF_ESTIMATED.format(month="month" if len(est) == 1 else "months", span=_span(est))]
              if est else []) + [f"{PO.waiting_note(m)}." for m in months if m in waiting]
    if not status:
        return part
    return part._replace(why=" ".join(([part.why] if part.why else []) + status), status=tuple(status))


def _compliance_before(sc: Scope, prev: Optional[str]) -> Optional[dict]:
    """Each unit's compliance in the previous month, same filters and basis: plant -> pooled %
    (None without A + B). The formulary ranking's own arithmetic (its by_hospital is this
    _rollup over this _cut), without building that month's whole overview again — a quarter of
    a second the page's first load does not need."""
    if prev is None or sc.segment == "consumables" or prev not in F._periods():
        return None
    fsc = F._scope(None, prev, sc.ptype, sc.segment, None, None, None, sc.basis)
    r = F._rollup(F._cut(F._table("formulary_summary"), fsc, plant=False), ["plant"], F._measure(fsc))
    return {str(p): _p2(v) for p, v in zip(r["plant"], r["pooled_pct"])}


def _mis_units() -> dict:
    """HCG's units (hcg_mis_units): plant -> {unit_name, region, pan_order, in_pan}; {} until the
    PAN MIS has been read. in_pan: in the PAN MIS unit list (False for a unit found only in HCG's
    history sheets, such as a centre since closed; True for a table built before the column)."""
    p = F._data_dir() / f"{FC.MIS_UNITS}.parquet"
    stamp = F._stamp(p)
    if stamp is None:
        return {}
    out = {}
    for r in F._read(str(p), stamp).to_dict("records"):
        order, pan = r.get("pan_order"), r.get("in_pan", True)
        out[str(r["plant"])] = {"unit_name": r.get("unit_name") if isinstance(r.get("unit_name"), str) else None,
                                "region": r.get("region") if isinstance(r.get("region"), str) else None,
                                "pan_order": int(order) if order is not None and not pd.isna(order) else None,
                                "in_pan": True if pan is None or pd.isna(pan) else bool(pan)}
    return out


# ── Every-month readings of a section's trend ─────────────────────────────────
def _by_period(rows: Optional[list]) -> dict:
    return {r["period"]: r for r in rows or []}


def _abc_share(r: Optional[dict]) -> Optional[float]:
    """Out of formulary's share of Formulary + Out of formulary + Non formulary."""
    if not r:
        return None
    whole = float(r["formulary"]) + float(r["out_of_formulary"]) + float(r["non_formulary"])
    return _share(r["out_of_formulary"], whole)


class Series(NamedTuple):
    value: Optional[float]        # the selection's figure
    prev: Optional[float]         # the same at the previous sales month
    spark: list                   # every month of the section: [{period, label, value}]


def _series(rows: list, read: Callable[[dict], Optional[float]], value, prev_month: Optional[str],
            rnd: Callable = _money) -> Series:
    """A KPI's figure, the same at `prev_month` (only beside a figure) and every month of its
    section's trend `rows`, each read by `read`."""
    by = _by_period(rows)
    spark = [{"period": r["period"], "label": F._month_label(r["period"]), "value": rnd(read(r))} for r in rows]
    prev = rnd(read(by[prev_month])) if prev_month in by and rnd(value) is not None else None
    return Series(rnd(value), prev, spark)


# ── KPIs ──────────────────────────────────────────────────────────────────────
def _kpi(key: str, label: str, s: Series, unit: str, delta_unit: str, good_when: str, source: str, note: str,
         target=None, status=None, sources: Optional[list] = None, tag: Optional[str] = None) -> dict:
    """One KPI card (C6). `source` is where its figure is counted ('sales' / 'purchases');
    `sources` every place its parts are counted, in order (the losses KPI: sales lines AND
    purchases); `tag` a few words the card shows beside the figure (or None)."""
    return {"key": key, "label": label, "value": s.value, "unit": unit, "prev": s.prev,
            "delta": _change(s.value, s.prev, delta_unit), "delta_unit": delta_unit, "good_when": good_when,
            "target": target, "status": status, "spark": s.spark, "source": source,
            "sources": list(sources or [source]), "note": note, "tag": tag}


# The three parts of what was missed (D2): section, its headline field, its name in a note.
LOSS_PARTS = (("cash", "margin_loss", "Cash & TPA"), ("scheme", "saving_missed", "scheme"),
              ("oof", "margin_loss", "out of formulary"))


def _sales_lines(sc: Scope) -> str:
    """Which sales lines a sales figure counts: 'the HIS pharmacy sales lines, IP and OP', '... of
    IP patients, Onco only'."""
    who = f" of {sc.ptype} patients" if sc.ptype else ", IP and OP"
    return f"the HIS pharmacy sales lines{who}" + (f", {RV.SEGMENT_LABELS[sc.segment]} only" if sc.segment else "")


def _blank_cost(head: dict) -> int:
    """Lines of the selection whose cost price is blank (counted at cost 0, D6)."""
    return int(head.get("cost_missing_lines") or 0)


def _usual_share(rule: Optional[dict]) -> str:
    """'10%': the share of a product's units at which its usual cheapest is reached, from the prices
    answer's rule block; the Prices page's own rule when that section has no answer here."""
    r = (rule or {}).get("usual_share_pct")
    return f"{round(r if _finite(r) else PI._price_rule()['usual_share_pct'], 2):g}%"


def _kpis(sc: Scope, src: dict, prev: Optional[str], judged: bool, months: tuple) -> list:
    qty = sc.basis == "qty"
    rupees = " ₹ only: a money measure stays in rupees in the Quantity view." if qty else ""
    sales, form, prices = src["sales"], src["formulary"], src["prices"]
    m = BASIS_MEASURE[sc.basis]

    def trend(part: Part, key: str = "trend") -> list:
        return (part.series or {}).get(key) or []

    # sales and margin %
    head = (sales.data or {}).get("headline") or {}
    s_sales = _series(trend(sales), lambda r: r[m], head.get(m), prev, _qty if qty else _money)
    s_margin = _series(trend(sales), lambda r: r["margin_pct"], head.get("margin_pct"), prev, _p2)
    lines = _sales_lines(sc)
    whole = "that part of HCG's revenue" if sc.ptype or sc.segment else "HCG's revenue"
    sales_note = (f"Units sold on {lines}: tablets, vials, bottles and so on added together." if qty
                  else f"Total MRP of {lines}: {whole}. {NOTE_GST_KPI}")
    margin_note = ("Margin ÷ MRP, where margin = MRP − cost price, both summed first: MRP less cost, not profit "
                   f"after tax. {NOTE_GST_KPI}")
    blank = _blank_cost(head)
    if blank:
        margin_note += " " + NOTE_BLANK_COST_KPI.format(lines=_indian(blank), noun=_plural(blank, "line", "lines"),
                                                        mrp=lakhs(head.get("mrp_cost_missing")),
                                                        count=_plural(blank, "counts", "count"))
    margin_note += rupees
    margin_tag = f"incl. {_indian(blank)} {_plural(blank, 'line', 'lines')} with no cost" if blank else None

    # compliance and the out-of-formulary share
    fh = (form.data or {}).get("headline") or {}
    s_comp = _series(trend(form), lambda r: r["pooled_pct"], fh.get("pooled_pct"), prev, _p2)
    s_oof = _series(trend(form), _abc_share, _abc_share(fh) if fh else None, prev, _p2)
    target = fh.get("budget_pct") if judged and form.data else None
    on = "units sold" if qty else "drug cost"
    comp_note = f"Formulary ÷ (Formulary + Out of formulary), pooled, on {on}; non formulary is outside the ratio."
    if qty:
        comp_note += " Not set against HCG's budget, which is a drug-cost target."
    elif not sc.plant:
        comp_note += " HCG sets compliance budgets per unit, so all hospitals together have none; the units table " \
                     "sets each unit against its own."
    elif not judged:
        comp_note += " " + NOTE_BUDGET_FILTERS
    elif target is None and form.data:
        comp_note += " " + (F.budget_differs_note(sc.plant, form.months) or "HCG's MIS sets no budget for this unit.")
    else:
        comp_note += " The target is HCG's budget for this unit."
    oof_note = (f"Out of formulary's share of all drug consumption (Formulary + Out of formulary + Non "
                f"formulary), on {on}.")

    # what was missed: the three parts the selection has, like for like across months
    parts = [(name, key, words) for name, key, words in LOSS_PARTS if src[name].data]
    spark_parts = parts or [(n, k, w) for n, k, w in LOSS_PARTS if src[n].series]
    value = sum(float(src[n].data["headline"][k]) for n, k, _ in parts) if parts else None
    rows = {n: _by_period(trend(src[n])) for n, _, _ in spark_parts}
    common = sorted(set.intersection(*(set(r) for r in rows.values()))) if rows else []
    loss_rows = [{"period": p, "value": sum(float(rows[n][p][k]) for n, k, _ in spark_parts)} for p in common]
    s_loss = _series(loss_rows, lambda r: r["value"], value, prev if parts == spark_parts else None)
    loss_note = (" + ".join(f"{words} {lakhs(src[n].data['headline'][k])}" for n, k, words in parts) + "."
                 if parts else "No part of what was missed has figures for this selection.")
    windows = _windows_note([(words, src[n].months) for n, _, words in parts], months)
    if windows:
        loss_note += f" {windows}"
    if src["oof"].data:
        loss_note += " Out of formulary is counted on purchases."
    missing = [words for n, _, words in LOSS_PARTS if not src[n].data]
    if parts and missing:
        loss_note += f" Not included, for want of figures for this selection: {', '.join(missing)}."
    loss_note += "".join(f" {s}" for s in src["oof"].status)
    loss_note += rupees

    loss_sources = list(dict.fromkeys(LOSS_SOURCES[n] for n, _, _ in parts)) or ["sales"]

    ph = (prices.data or {}).get("headline") or {}
    s_extra = _series(trend(prices, "by_month"), lambda r: r["extra_paid"], ph.get("extra_paid_vs_cheapest"), prev)
    at = " at this hospital" if sc.plant else ""
    extra_note = NOTE_EXTRA_PAID.format(at=at, share=_usual_share((prices.data or {}).get("rule")))
    if ph.get("extra_paid_vs_network_cheapest") is not None:
        extra_note += (" Set against the usual cheapest of all hospitals instead, as the units table does, it is "
                       f"{lakhs(ph['extra_paid_vs_network_cheapest'])}.")
    if len(prices.months) > 1:
        # The Prices page's own rule for several months (purchase_insights.NOTE_PRICE_MONTHS).
        extra_note += (f" Over {_span(prices.months)} it is set against one usual cheapest over all of them, so it "
                       "is not the months' own figures (the sparkline) added up.")
    if sc.month is None and prices.data and set(prices.held) - set(months):
        extra_note += " " + _sentence("prices", NOTE_LINK_MONTHS, span=_span(prices.held))
    extra_note += rupees

    # a KPI with no figure says why (its section's own note)
    sales_note, margin_note, comp_note, oof_note, extra_note = (
        note if part.data or not part.why else f"{note} {part.why}"
        for note, part in ((sales_note, sales), (margin_note, sales), (comp_note, form), (oof_note, form),
                           (extra_note, prices)))
    if sc.month is None:                    # ALL: totals only, no change
        s_sales, s_margin, s_comp, s_oof, s_loss, s_extra = (x._replace(prev=None) for x in (
            s_sales, s_margin, s_comp, s_oof, s_loss, s_extra))
    return [
        _kpi("sales", UNITS_SOLD_LABEL if qty else KPI_LABELS["sales"], s_sales, "qty" if qty else "inr", "pct", "up",
             "sales", sales_note),
        _kpi("margin_pct", KPI_LABELS["margin_pct"], s_margin, "pct", "pts", "up", "sales", margin_note,
             tag=margin_tag),
        _kpi("compliance", KPI_LABELS["compliance"], s_comp, "pct", "pts", "up", "sales", comp_note,
             target=target, status=classify(s_comp.value, target)),
        _kpi("oof_share", KPI_LABELS["oof_share"], s_oof, "pct", "pts", "down", "sales", oof_note),
        _kpi("losses", KPI_LABELS["losses"], s_loss, "inr", "pct", "down", loss_sources[0], loss_note,
             sources=loss_sources),
        _kpi("extra_paid", EXTRA_PAID_OWN_LABEL if sc.plant else KPI_LABELS["extra_paid"], s_extra, "inr", "pct",
             "down", "purchases", extra_note),
    ]


# ── The other blocks ──────────────────────────────────────────────────────────
def _trend(src: dict) -> list:
    """Every sales month: IP and OP MRP, cost, margin and its %, units, and compliance."""
    sales = (src["sales"].series or {}).get("trend") or []
    comp = _by_period((src["formulary"].series or {}).get("trend"))
    return [{"period": t["period"], "label": t["label"], "ip_mrp": t["ip"]["mrp"], "op_mrp": t["op"]["mrp"],
             "mrp": t["mrp"], "cost": t["cost"], "margin": t["margin"], "margin_pct": t["margin_pct"],
             "compliance_pct": _p2(comp[t["period"]]["pooled_pct"]) if t["period"] in comp else None,
             "qty": t["qty"]} for t in sales]


def _bridge(sc: Scope, src: dict) -> dict:
    """D3: margin earned on sales, each part of what was missed on top, the margin with every
    switch. A part without figures is left out and the note names it; parts that hold other
    months than the sales (all months, a purchase month built before its sales) are named with
    their months, and the note then no longer says the purchases are "of the same months"."""
    sales = src["sales"].data
    parts = [(words, src[n].months) for n, _, words in LOSS_PARTS if src[n].data]
    windows = _windows_note([("margin earned", src["sales"].months)] + parts, src["sales"].months) if sales else None
    notes = [NOTE_BRIDGE.format(same="" if windows else SAME_MONTHS)] + \
        ([NOTE_BRIDGE_RUPEES] if sc.basis == "qty" else [])
    if not sales:
        return {"steps": [], "note": " ".join(notes + [src["sales"].why or ""]).strip()}
    earned = float(sales["headline"]["margin"])
    steps = [{"key": "earned", "label": BRIDGE_LABELS["earned"], "value": _money(earned), "kind": "total",
              "source": "sales"}]
    total, left_out = earned, []
    for name, key, words in LOSS_PARTS:
        d = src[name].data
        if not d:
            left_out.append(words)
            continue
        v = float(d["headline"][key])
        total += v
        steps.append({"key": name, "label": BRIDGE_LABELS[name], "value": _money(v), "kind": "delta",
                      "source": "purchases" if name == "oof" else "sales"})
    steps.append({"key": "possible", "label": BRIDGE_LABELS["possible"], "value": _money(total), "kind": "total",
                  "source": "sales"})
    if left_out:
        notes.append(NOTE_BRIDGE_LEFT_OUT.format(parts=", ".join(left_out)))
    if windows:
        notes.append(windows)
    notes += src["oof"].status
    return {"steps": steps, "note": " ".join(notes)}


def _ip_op(sc: Scope, src: dict) -> list:
    sales = src["sales"].data
    if not sales:
        return []
    m = BASIS_MEASURE[sc.basis]
    whole = sum(float(t[m]) for t in sales["by_type"])
    return [{"type": t["type"], "label": t["label"], "mrp": t["mrp"], "margin": t["margin"],
             "margin_pct": t["margin_pct"], "qty": t["qty"], "share_pct": _share(t[m], whole)}
            for t in sales["by_type"]]


def _mix(src: dict) -> Optional[dict]:
    f = src["formulary"].data
    if not f:
        return None
    h = f["headline"]
    a, b, c = float(h["formulary"]), float(h["out_of_formulary"]), float(h["non_formulary"])
    whole = a + b + c
    return {"formulary": _money(a), "out_of_formulary": _money(b), "non_formulary": _money(c),
            "total": _money(whole), "formulary_pct": _share(a, whole), "oof_pct": _share(b, whole),
            "nf_pct": _share(c, whole)}


def _rows_by_plant(part: Part, key: str = "by_hospital") -> Optional[dict]:
    return {r["plant"]: r for r in part.data[key]} if part.data else None


def _amount(rows: Optional[dict], plant: str, key: str) -> Optional[float]:
    """A unit's amount from a section's ranking: 0 when the section has the month but not the
    unit (nothing bought or missed there), None when the section has no figures."""
    if rows is None:
        return None
    r = rows.get(plant)
    return _money(r[key]) if r else 0.0


def _units(sc: Scope, src: dict, before: Optional[dict], judged: bool) -> list:
    """Every unit in HCG's PAN list, and any other unit with figures in the selection (sales,
    drug consumption, or a part of what was missed or paid above the cheapest), in HCG's order
    (pan_order, then name): its sales, compliance against budget and change, consumption split,
    and its part of each loss and of extra paid. A unit HCG's history sheets name but the PAN
    list no longer holds (a centre since closed) is left out when it has nothing to show: the
    hospital selector does not know it either. `before`: _compliance_before."""
    mis = _mis_units()
    names = F._sap_names()
    rev = _rows_by_plant(src["sales"])
    form = _rows_by_plant(src["formulary"])
    oof, cash, scheme, prices = (_rows_by_plant(src[n]) for n in ("oof", "cash", "scheme", "prices"))
    amounts = ((oof, "margin_loss"), (cash, "margin_loss"), (scheme, "saving_missed"), (prices, "extra_paid"))
    months = src["formulary"].months

    def has_figures(p: str) -> bool:
        return p in (rev or {}) or p in (form or {}) or any(
            float((rows or {}).get(p, {}).get(key) or 0) for rows, key in amounts)

    listed = {p for p in set(mis) | set(rev or {}) if mis.get(p, {}).get("in_pan") or has_figures(p)}
    out = []
    for p in sorted(listed):
        hcg = mis.get(p, {"unit_name": None, "region": None, "pan_order": None})
        f = (form or {}).get(p)
        comp = _p2(f["compliance_pct"]) if f else None
        # a unit without drug spend still has its budget (judged only on the value basis)
        budget = (f["budget_pct"] if f else (F._budget(p, months)[0] if months else None)) if sc.basis == "value" \
            else None
        # why a budget is blank when HCG does set one: the months selected do not share it
        budget_note = F.budget_differs_note(p, months) if budget is None and months and sc.basis == "value" else None
        r = (rev or {}).get(p)
        his = F._his_name(p)
        out.append({"plant": p, "name": names.get(p) or his or hcg["unit_name"] or p, "his_name": his,
                    "region": hcg["region"], "pan_order": hcg["pan_order"],
                    "mrp": (_money(r["mrp"]) if r else 0.0) if rev is not None else None,
                    "margin_pct": r["margin_pct"] if r else None,
                    "compliance_pct": comp, "budget_pct": _p2(budget), "budget_note": budget_note,
                    "status": classify(comp, budget) if judged else None,
                    "delta_pts": _change(comp, (before or {}).get(p), "pts"),
                    "formulary_share_pct": _p2(f["formulary_share_pct"]) if f else None,
                    "oof_share_pct": _p2(f["oof_share_pct"]) if f else None,
                    "nf_share_pct": _p2(f["nf_share_pct"]) if f else None,
                    "oof_loss": _amount(oof, p, "margin_loss"), "cash_loss": _amount(cash, p, "margin_loss"),
                    "scheme_missed": _amount(scheme, p, "saving_missed"),
                    "extra_paid": _amount(prices, p, "extra_paid"), "selected": p == sc.plant})
    out.sort(key=lambda u: (u["pan_order"] is None, u["pan_order"] or 0, u["name"]))
    return out


def _sales_rows(sc: Scope, src: dict) -> pd.DataFrame:
    """sales_items rows of every unit in the selection's sales months and filters (the hospital
    aside, for the unit ranking), with their C1 bucket and material type."""
    months = src["sales"].months
    if not months:
        return pd.DataFrame(columns=["period", "plant", "material", "material_desc", "generic_name", "_type",
                                     "_bucket", "mrp", "cost", "qty"])
    d = RV._cut(RV.sales_items(), RV.Scope(sc.plant, months, sc.ptype, sc.segment, None), plant=False)
    return d.assign(plant=d["plant"].astype(str), material=d["material"].astype(str),
                    _type=d["category"].astype(str).map(CATEGORY_TYPE), _bucket=RV.bucket_of(d))


def _treemap(d: pd.DataFrame, measure: str, rnd: Callable) -> list:
    """Material type -> HCG's bucket, on MRP (units on qty); empty parts left out."""
    g = d.groupby(["_type", "_bucket"])[measure].sum() if len(d) else pd.Series(dtype=float)
    out = []
    for key, label in RV.SEGMENT_LABELS.items():
        children = [{"key": b, "label": bl, "value": rnd(g[(key, b)])} for b, bl in RV.BUCKETS.items()
                    if (key, b) in g.index and g[(key, b)] != 0]
        if children:
            out.append({"key": key, "label": label, "value": rnd(sum(float(g[(key, c["key"])]) for c in children)),
                        "children": children})
    return out


def _flow(d: pd.DataFrame, measure: str, rnd: Callable, names: dict) -> dict:
    """D11: unit (the FLOW_UNITS biggest, then Other units) -> material type -> HCG's bucket."""
    if d.empty or not float(d[measure].sum()):
        return {"nodes": [], "links": []}
    by_unit = d.groupby("plant")[measure].sum()
    by_unit = by_unit[by_unit != 0].sort_values(ascending=False, kind="mergesort")
    top = list(by_unit.index[:FLOW_UNITS])
    unit = d["plant"].where(d["plant"].isin(top), "other")
    nodes = [{"id": f"unit:{p}", "label": names.get(p) or F._his_name(p) or p, "column": 0} for p in top]
    if len(by_unit) > len(top):
        nodes.append({"id": "unit:other", "label": "Other units", "column": 0})
    t = d.groupby("_type")[measure].sum()
    nodes += [{"id": f"type:{k}", "label": label, "column": 1} for k, label in RV.SEGMENT_LABELS.items()
              if k in t.index and t[k] != 0]
    b = d.groupby("_bucket")[measure].sum()
    nodes += [{"id": f"bucket:{k}", "label": label, "column": 2} for k, label in RV.BUCKETS.items()
              if k in b.index and b[k] != 0]
    links = []
    ut = d.assign(_unit=unit).groupby(["_unit", "_type"])[measure].sum()
    for (u, ty), v in ut.items():
        if v:
            links.append({"source": f"unit:{u}", "target": f"type:{ty}", "value": rnd(v)})
    for (ty, bk), v in d.groupby(["_type", "_bucket"])[measure].sum().items():
        if v:
            links.append({"source": f"type:{ty}", "target": f"bucket:{bk}", "value": rnd(v)})
    order = {n["id"]: i for i, n in enumerate(nodes)}
    links.sort(key=lambda x: (order.get(x["source"], len(order)), order.get(x["target"], len(order))))
    return {"nodes": nodes, "links": links}


def _top(d: pd.DataFrame, every: pd.DataFrame, measure: str, rnd: Callable, names: dict) -> dict:
    """The TOP_N items of the selection and units of every hospital, by consumption (cost on
    value, units on qty), each with its share of the whole."""
    items, units = [], []
    whole = float(d[measure].sum()) if len(d) else 0.0
    if whole:
        g = d.groupby("material").agg(value=(measure, "sum"), qty=("qty", "sum"))
        g = g[g["value"] != 0].sort_values("value", ascending=False, kind="mergesort").head(TOP_N)
        info = (d.sort_values("period").drop_duplicates("material", keep="last")
                .set_index("material")[["material_desc", "generic_name"]])
        for mat, r in g.iterrows():
            desc, generic = info.at[mat, "material_desc"], info.at[mat, "generic_name"]
            items.append({"material": mat, "material_desc": desc if isinstance(desc, str) else None,
                          "generic_name": generic if isinstance(generic, str) else None,
                          "value": rnd(r["value"]), "qty": _qty(r["qty"]), "share_pct": _share(r["value"], whole)})
    network = float(every[measure].sum()) if len(every) else 0.0
    if network:
        g = every.groupby("plant").agg(value=(measure, "sum"), qty=("qty", "sum"))
        g = g[g["value"] != 0].sort_values("value", ascending=False, kind="mergesort").head(TOP_N)
        units = [{"plant": p, "name": names.get(p) or F._his_name(p) or p, "value": rnd(r["value"]),
                  "qty": _qty(r["qty"]), "share_pct": _share(r["value"], network)} for p, r in g.iterrows()]
    return {"items": items, "units": units}


def _lowest_margin(d: pd.DataFrame, key: str) -> pd.DataFrame:
    """`d` summed per `key` (MRP, cost, margin, margin %, share of MRP), the TOP_N with the lowest
    margin % among those with at least BOTTOM_MIN_SHARE_PCT of its MRP; a tie goes to the bigger
    seller."""
    whole = float(d["mrp"].sum()) if len(d) else 0.0
    if whole <= 0:
        return pd.DataFrame(columns=[key, "mrp", "cost", "margin", "margin_pct", "share_pct"])
    g = d.groupby(key).agg(mrp=("mrp", "sum"), cost=("cost", "sum")).reset_index()
    g["share_pct"] = g["mrp"] / whole * 100
    g = g[(g["mrp"] > 0) & (g["share_pct"] >= BOTTOM_MIN_SHARE_PCT)].copy()
    g["margin"] = g["mrp"] - g["cost"]
    g["margin_pct"] = g["margin"] / g["mrp"] * 100
    return g.sort_values(["margin_pct", "mrp", key], ascending=[True, False, True], kind="mergesort").head(TOP_N)


def _bottom(d: pd.DataFrame, every: pd.DataFrame, names: dict) -> dict:
    """The TOP_N items of the selection and units of every hospital with the LOWEST margin %
    (margin ÷ MRP, money whatever the basis; NOTE_BOTTOM): the other end of the top lists."""
    items, units = [], []
    g = _lowest_margin(d, "material")
    if len(g):
        info = (d.sort_values("period").drop_duplicates("material", keep="last")
                .set_index("material")[["material_desc", "generic_name"]])
        for r in g.to_dict("records"):
            desc, generic = info.at[r["material"], "material_desc"], info.at[r["material"], "generic_name"]
            items.append({"material": r["material"], "material_desc": desc if isinstance(desc, str) else None,
                          "generic_name": generic if isinstance(generic, str) else None, "mrp": _money(r["mrp"]),
                          "margin": _money(r["margin"]), "margin_pct": _p2(r["margin_pct"]),
                          "share_pct": _p2(r["share_pct"])})
    units = [{"plant": r["plant"], "name": names.get(r["plant"]) or F._his_name(r["plant"]) or r["plant"],
              "mrp": _money(r["mrp"]), "margin": _money(r["margin"]), "margin_pct": _p2(r["margin_pct"]),
              "share_pct": _p2(r["share_pct"])} for r in _lowest_margin(every, "plant").to_dict("records")]
    return {"items": items, "units": units}


# ── Insights and actions ──────────────────────────────────────────────────────
class Ctx(NamedTuple):
    sc: Scope
    src: dict
    kpis: dict                    # key -> KPI
    units: list
    where: str                    # 'across all hospitals' / 'at <the hospital's name>'
    prev: Optional[str]
    judged: bool


def _short(units: list) -> list:
    """Units below budget, furthest first."""
    return sorted((u for u in units if u["status"] in ("within", "below")),
                  key=lambda u: (u["compliance_pct"] - u["budget_pct"], u["name"]))


def _unit_name(u: dict) -> str:
    return u["his_name"] or u["name"]


def _insight_compliance(c: Ctx) -> Optional[dict]:
    k = c.kpis["compliance"]
    if k["value"] is None:
        return None
    on = " on units sold" if c.sc.basis == "qty" else ""
    when = _when(c.src["formulary"].months)
    text = f"Formulary compliance{on} {c.where} was {pct(k['value'])} {when}"
    if k["target"] is not None:
        gap = k["value"] - k["target"]
        # "within" is the card's rule on the figure as HCG's MIS PRINTS it (whole %), so a raw gap
        # a little over WITHIN_PTS can be within: the words say it is the printed figure
        state = {"met": "met",
                 "within": f"{pts(-gap)} short; HCG's MIS prints it within {WITHIN_PTS:g} pts of the budget",
                 "below": f"{pts(-gap)} short"}[k["status"]]
        text += f", against a budget of {budget_text(k['target'])}: {state}."
    else:
        text += "."
    move = _direction(k["value"], k["prev"], k["delta"], STEADY_PTS)
    if k["prev"] is not None:
        prev = F._month_label(c.prev)
        text += (f" That is {'up' if move > 0 else 'down'} {pts(abs(k['delta']))} from {prev} ({pct(k['prev'])})."
                 if move else f" That is about the same as {prev} ({pct(k['prev'])}).")
    tone = "alert" if k["status"] in ("within", "below") else {1: "up", -1: "down", 0: "info"}[move]
    return {"id": "compliance", "tone": tone, "text": text, "link": _link(c.sc, "formulary")}


def _insight_units_below(c: Ctx) -> Optional[dict]:
    if c.sc.plant or not c.judged:
        return None
    judged = [u for u in c.units if u["status"] is not None]
    if not judged:
        return None
    short = _short(judged)
    when = _when(c.src["formulary"].months)
    if not short:
        every = "The only unit with a budget" if len(judged) == 1 else f"All {len(judged)} units with a budget"
        return {"id": "units_below_budget", "tone": "up", "link": _link(c.sc, "formulary"),
                "text": f"{every} met it {when}."}
    far = short[0]
    # counted by the status each unit's chip shows: on the figure HCG's MIS prints (whole %)
    below = sum(1 for u in short if u["status"] == "below")
    text = (f"{len(short)} of {len(judged)} units with a budget missed it {when}"
            f" ({below} more than {WITHIN_PTS:g} pts short as HCG's MIS prints it). Furthest: {_unit_name(far)} "
            f"at {pct(far['compliance_pct'])} against {budget_text(far['budget_pct'])}, "
            f"{pts(far['budget_pct'] - far['compliance_pct'])} short.")
    return {"id": "units_below_budget", "tone": "alert", "text": text, "link": _link(c.sc, "formulary")}


def _trend_value(part: Part, period: Optional[str], key: str, rows: str = "trend") -> Optional[float]:
    r = _by_period((part.series or {}).get(rows)).get(period) if period else None
    return float(r[key]) if r else None


def _insight_oof(c: Ctx) -> Optional[dict]:
    part = c.src["oof"]
    if not part.data:
        return None
    v = float(part.data["headline"]["margin_loss"])
    when = _when(part.months)
    if v > 0:
        text = f"Out-of-formulary purchases missed {lakhs(v)} of margin {when}"
    elif v < 0:                             # HCG's net figure: the brands bought earned more
        text = f"The out-of-formulary brands bought earned {lakhs(-v)} more margin than the suggested ones {when}"
    else:
        text = f"Out-of-formulary purchases missed no margin {when}"
    prev = _trend_value(part, c.prev, "margin_loss") if c.sc.month else None
    if prev is not None and min(v, prev) < 0:
        prev = None                         # a net gain on either side: "up / down from" would mislead
    delta = _change(v, prev, "pct")
    move = _direction(v, prev, delta, STEADY_PCT)
    if prev is not None:
        label = F._month_label(c.prev)
        if move:
            text += f", {'up' if move > 0 else 'down'} {lakhs(abs(v - prev))} from {label}"
            text += f" ({pct(abs(delta))})" if delta is not None else ""
        else:
            text += f", about the same as {label} ({lakhs(prev)})"
    text += "."
    if not c.sc.plant and v > 0:
        # A share of the units that missed margin, not of the net total: a unit whose brands
        # earned more brings the net down, and a share of it could pass 100%.
        losing = [u for u in part.data["by_hospital"] if u["margin_loss"] > 0]
        if losing:
            top = max(losing, key=lambda u: (u["margin_loss"], u["plant"]))
            share = pct(top["margin_loss"] / sum(u["margin_loss"] for u in losing) * 100)
            text += (f" {top['his_name'] or top['name']} missed the most: {lakhs(top['margin_loss'])} ({share} of "
                     f"the margin missed at units with a loss).")
    tone = {1: "down", -1: "up"}.get(move, "alert" if v > 0 else "up" if v < 0 else "info")
    return {"id": "oof_missed", "tone": tone, "text": text, "link": _link(c.sc, "oof")}


def _insight_cash(c: Ctx) -> Optional[dict]:
    part = c.src["cash"]
    if not part.data:
        return None
    h = part.data["headline"]
    v = float(h["margin_loss"])
    text = f"Cash & TPA patients: {lakhs(v)} of margin missed {_when(part.months)}"
    text += f" ({pct(h['loss_pct'])} of the margin possible)." if h["loss_pct"] is not None else "."
    top = (part.data.get("top_items") or [None])[0]
    if top and top["margin_loss"] > 0:
        text += (f" The biggest single switch: {top['material_desc']} → {top['suggested_desc']}, "
                 f"{lakhs(top['margin_loss'])}.")
    return {"id": "cash_missed", "tone": "alert" if v > 0 else "info", "text": text, "link": _link(c.sc, "cash")}


def _insight_sales(c: Ctx) -> Optional[dict]:
    part = c.src["sales"]
    if not part.data:
        return None
    k, mk = c.kpis["sales"], c.kpis["margin_pct"]
    when = _when(part.months)
    qty = c.sc.basis == "qty"
    text = (f"{units_text(k['value'])} {_plural(_half_up(abs(k['value'])), 'was', 'were')} sold {when}" if qty
            else f"Revenue was {lakhs(k['value'])} at MRP {when}")
    move = _direction(k["value"], k["prev"], k["delta"], STEADY_PCT)
    if k["prev"] is not None:
        label = F._month_label(c.prev)
        if move and k["delta"] is not None:
            text += f", {'up' if move > 0 else 'down'} {pct(abs(k['delta']))} from {label}"
        elif move:
            text += f", {'up' if move > 0 else 'down'} from nothing in {label}"
        else:
            text += f", about the same as {label}"
    if mk["value"] is not None:
        text += f", with a margin of {pct(mk['value'])}" + (f" ({pct(mk['prev'])} in {F._month_label(c.prev)})"
                                                            if mk["prev"] is not None else "")
    text += "."
    ip, op = (next((t for t in part.data["by_type"] if t["type"] == x), None) for x in F.TYPES)
    if not c.sc.ptype and ip and op and ip["margin_pct"] is not None and op["margin_pct"] is not None:
        text += (f" IP earned {pct(ip['margin_pct'])} on {lakhs(ip['mrp'])} and OP {pct(op['margin_pct'])} on "
                 f"{lakhs(op['mrp'])}.")
    tone = {1: "up", -1: "down"}.get(move, "info")
    return {"id": "sales_margin", "tone": tone, "text": text, "link": _link(c.sc, "revenue")}


def _low_words(rule: Optional[dict]) -> str:
    """'half the usual cheapest' (spend insights.tsx `lowRatioWords`)."""
    r = (rule or {}).get("low_ratio_pct")
    if not _finite(r):
        return "far under the usual cheapest"
    return {50: "half the usual cheapest", 25: "a quarter of the usual cheapest"}.get(
        r, f"{r:g}% of the usual cheapest")


def _insight_prices(c: Ctx) -> Optional[dict]:
    part = c.src["prices"]
    if not part.data:
        return None
    h = part.data["headline"]
    extra, low = float(h["extra_paid_vs_cheapest"]), int(h["low_rows"])
    if not h["compared_value"]:
        return None                         # nothing bought at a comparable price: nothing to say
    if extra > 0:
        text = f"{lakhs(extra)} was paid above the usual cheapest price {_when(part.months)}"
        if h["extra_pct"] is not None:
            text += f" ({pct(h['extra_pct'])} of the purchases compared)"
        top = (part.data.get("molecules") or [None])[0]
        if top and top["extra_paid"] > 0:
            text += f", most of it on {product_name(top)} ({lakhs(top['extra_paid'])})"
        text += "."
    else:
        text = f"Nothing was paid above the usual cheapest price {_when(part.months)}."
    if low:
        text += (f" {low} {_plural(low, 'purchase was', 'purchases were')} under "
                 f"{_low_words(part.data.get('rule'))}: worth a check.")
    return {"id": "prices_extra", "tone": "alert" if extra > 0 or low else "info", "text": text,
            "link": _link(c.sc, "prices")}


def _insight_better(c: Ctx) -> Optional[dict]:
    """All hospitals: the units whose brands bought out of formulary earned more than their
    suggestions. One hospital: _insight_oof says it in its own sentence."""
    part = c.src["oof"]
    if not part.data or c.sc.plant:
        return None
    better = [u for u in part.data["by_hospital"] if u["margin_loss"] < 0]
    if not better:
        return None
    better.sort(key=lambda u: (u["margin_loss"], u["plant"]))
    gained = -sum(float(u["margin_loss"]) for u in better)
    n = len(better)
    text = (f"At {n} {_plural(n, 'unit', 'units')} ({_names_list([u['his_name'] or u['name'] for u in better])}) the "
            f"out-of-formulary brands bought earned {lakhs(gained)} more margin than the suggested ones "
            f"{_when(part.months)}.")
    return {"id": "oof_better_units", "tone": "info", "text": text, "link": _link(c.sc, "oof")}


# In KPI order; the tone sorts them (alert, worse, better, for the record) before the cap.
INSIGHT_RULES = (_insight_sales, _insight_compliance, _insight_units_below, _insight_oof, _insight_cash,
                 _insight_prices, _insight_better)


def _insights(c: Ctx) -> list:
    found = [x for x in (rule(c) for rule in INSIGHT_RULES) if x]
    found.sort(key=lambda x: TONE_ORDER[x["tone"]])               # stable: rule order within a tone
    return found[:MAX_INSIGHTS]


def _action_oof(c: Ctx) -> Optional[dict]:
    part = c.src["oof"]
    top = ((part.data or {}).get("top_items") or [None])[0]
    if not top or top["margin_loss"] <= 0:
        return None
    est = " (Estimated)" if top.get("estimated") else ""
    return {"id": "oof_switch", "impact": _money(top["margin_loss"]), "link": _link(c.sc, "oof"),
            "text": f"Buy {top['suggested_desc']} in place of {top['material_desc']}{est}: "
                    f"{lakhs(top['margin_loss'])} of margin missed on out-of-formulary purchases {_when(part.months)}."}


def _action_cash(c: Ctx) -> Optional[dict]:
    part = c.src["cash"]
    top = ((part.data or {}).get("top_items") or [None])[0]
    if not top or top["margin_loss"] <= 0:
        return None
    return {"id": "cash_switch", "impact": _money(top["margin_loss"]), "link": _link(c.sc, "cash"),
            "text": f"Prescribe {top['suggested_desc']} in place of {top['material_desc']} for cash & TPA patients: "
                    f"{lakhs(top['margin_loss'])} of margin missed {_when(part.months)}."}


def _action_follow_up(c: Ctx) -> Optional[dict]:
    short = _short([u for u in c.units if not c.sc.plant or u["selected"]])[:FOLLOW_UP_UNITS]
    if not short:
        return None
    each = [f"{_unit_name(u)} ({pts(u['budget_pct'] - u['compliance_pct'])} short of its "
            f"{budget_text(u['budget_pct'])} budget)" for u in short]
    who = each[0] if len(each) == 1 else ", ".join(each[:-1]) + " and " + each[-1]
    return {"id": "follow_up_units", "impact": None, "link": _link(c.sc, "formulary"),
            "text": f"Follow up on formulary compliance with {who}."}


def _action_low_prices(c: Ctx) -> Optional[dict]:
    part = c.src["prices"]
    if not part.data or not part.data["headline"]["low_rows"]:
        return None
    h = part.data["headline"]
    n = int(h["low_rows"])
    return {"id": "check_low_prices", "impact": None, "link": _link(c.sc, "prices", **LINK_CHECK),
            "text": f"Check the {n} {_plural(n, 'purchase', 'purchases')} bought under "
                    f"{_low_words(part.data.get('rule'))} ({lakhs(h['low_value'])} {_when(part.months)}): a keying "
                    f"slip or a free or bonus supply is likely."}


def _action_negotiate(c: Ctx) -> Optional[dict]:
    part = c.src["prices"]
    top = ((part.data or {}).get("molecules") or [None])[0]
    if not top or top["extra_paid"] <= 0:
        return None
    return {"id": "negotiate_price", "impact": _money(top["extra_paid"]), "link": _link(c.sc, "prices"),
            "text": f"Negotiate the price of {product_name(top)}: {lakhs(top['extra_paid'])} was paid above its "
                    f"usual cheapest of {unit_price(top['cheapest_price'])} a unit {_when(part.months)}."}


ACTION_RULES = (_action_oof, _action_cash, _action_follow_up, _action_low_prices, _action_negotiate)


def _actions(c: Ctx) -> list:
    """Biggest rupees first; a follow-up or a check (no rupees of its own) after them."""
    found = [x for x in (rule(c) for rule in ACTION_RULES) if x]
    found.sort(key=lambda x: (x["impact"] is None, -(x["impact"] or 0.0)))
    return [{"id": x["id"], "text": x["text"], "impact": x["impact"], "link": x["link"]} for x in found[:MAX_ACTIONS]]


# ── Endpoint ──────────────────────────────────────────────────────────────────
@router.get("/executive/overview")
def executive_overview(Plant: str = Query(None), Month: str = Query(None), Type: str = Query(None),
                       Segment: str = Query(None), Basis: str = Query(None)):
    sc = _scope(Plant, Month, Type, Segment, Basis)
    return _overview_cached(sc, version())


@F._cache
def _overview_cached(sc: Scope, ver: tuple) -> dict:
    sales_months = RV.sales_periods()
    oof_months = _oof_months()
    page = sorted(set(sales_months) | set(oof_months))
    months = (sc.month,) if sc.month else tuple(page)
    prev = _previous(sc.month, sales_months)
    judged = sc.basis == "value" and not sc.ptype and not sc.segment
    names = F._sap_names()

    src = _sources(sc, months)
    kpis = _kpis(sc, src, prev, judged, months)
    units = _units(sc, src, _compliance_before(sc, prev) if src["formulary"].data else None, judged)

    qty = sc.basis == "qty"
    rnd = _qty if qty else _money
    every = _sales_rows(sc, src)
    d = every[every["plant"] == sc.plant] if sc.plant else every
    where = f"at {F._his_name(sc.plant) or names.get(sc.plant, sc.plant)}" if sc.plant else "across all hospitals"
    ctx = Ctx(sc, src, {k["key"]: k for k in kpis}, units, where, prev, judged)

    notes = [NOTE_SOURCES]
    if sc.month is None:
        notes.append(NOTE_ALL.format(span=_span(page)))
    elif prev:
        notes.append(NOTE_PREVIOUS.format(label=F._month_label(prev)))
    elif sc.month in sales_months:
        notes.append(NOTE_FIRST.format(label=F._month_label(sc.month)))
    if sc.segment == "consumables":
        notes.append(NOTE_CONSUMABLES)
    notes += list(dict.fromkeys(p.why for p in src.values() if p.why and p.why != NOTE_CONSUMABLES))
    if sc.month is None:
        # a link opens its page on that page's own All months: say where it holds more months
        notes += [_sentence(name, NOTE_LINK_MONTHS, span=_span(p.held)) for name, p in src.items()
                  if name in LINKED_SOURCES and p.data and set(p.held) - set(page)]
    same = bool(src["sales"].data) and all(src[n].months == src["sales"].months
                                           for n in ("oof", "prices") if src[n].data)
    notes.append(NOTE_PURCHASES.format(same=SAME_MONTHS if same else "")
                 + (" " + NOTE_PURCHASES_TYPE if sc.ptype else ""))
    if src["sales"].data:
        # D6: what MRP and cost are, and the lines counted at cost 0, as the Revenue page says them
        notes.append(RV.NOTE_GST)
        head = src["sales"].data["headline"]
        blank = _blank_cost(head)
        notes.append(RV.NOTE_BLANK_COST.format(lines=blank, noun=_plural(blank, "line", "lines"),
                                               verb=_plural(blank, "has", "have"), mrp=head["mrp_cost_missing"])
                     if blank else RV.NOTE_NO_BLANK_COST)
    if qty:
        notes.append(NOTE_UNITS_VIEW)
    elif sc.ptype or sc.segment:
        notes.append(NOTE_BUDGET_FILTERS)
    notes += [NOTE_UNITS_TABLE, NOTE_STATUS, NOTE_TOP[sc.basis], NOTE_BOTTOM, NOTE_RULES]
    if sc.plant and src["sales"].data and not src["sales"].data["headline"]["lines"]:
        notes.append(f"No pharmacy sales for {names.get(sc.plant, sc.plant)} in this selection.")

    return {
        "scope": {"plant": sc.plant or "ALL",
                  "plant_name": names.get(sc.plant, sc.plant) if sc.plant else "All Hospitals",
                  "his_name": F._his_name(sc.plant) if sc.plant else None,
                  "month": sc.month or "ALL", "month_label": F._month_label(sc.month) if sc.month else "All months",
                  "prev_month": prev, "prev_label": F._month_label(prev) if prev else None,
                  "type": sc.ptype or "ALL", "segment": sc.segment or "ALL", "basis": sc.basis,
                  "budget_judged": judged},
        "months": [{"period": p, "label": F._month_label(p), "sales": p in sales_months, "purchases": p in oof_months}
                   for p in page],
        "kpis": kpis,
        "trend": _trend(src),
        "bridge": _bridge(sc, src),
        "ip_op": _ip_op(sc, src),
        "units": units,
        "mix": _mix(src),
        "treemap": _treemap(d, BASIS_MEASURE[sc.basis], rnd),
        "flow": _flow(d, BASIS_MEASURE[sc.basis], rnd, names),
        "top": {**_top(d, every, RANK_MEASURE[sc.basis], rnd, names), "bottom": _bottom(d, every, names)},
        "insights": _insights(ctx),
        "actions": _actions(ctx),
        "notes": notes,
        "data_as_of": F._built_at(),
    }
