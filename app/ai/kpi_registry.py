# app/ai/kpi_registry.py — the ONE canonical source the chatbot uses for any named,
# dashboard-visible KPI.
#
# WHY THIS EXISTS (root-cause fix, not a patch): an 8-persona live audit of the chatbot
# found it silently re-deriving named KPIs (Fill Rate, Procurement Cycle Time, Inventory
# Health Score, Reorder Priority...) from ad hoc SQL against the semantic warehouse,
# instead of using the SAME calculation the dashboard card for that KPI already uses.
# Every one of those calculations has real, hard-won business logic behind it — e.g. the
# stock-out rate excludes plants with zero inventory coverage, ITR is a two-piece
# additive pattern that a naive single SUM overstates ~10x, fill rate needs a specific
# open/ordered-qty definition — and a general-purpose SQL-writing model re-deriving that
# from raw tables on every phrasing is structurally unreliable. Confirmed live: the
# chatbot stated Fill Rate as 39.4% (verified: "ok", no caveat) against a real 91.72%.
#
# THE FIX: every entry below wraps the EXACT SAME Python function the corresponding
# /kpi/*/insights dashboard route calls — not a new query, not a re-implementation.
# There is only one implementation of "what is Fill Rate" in this codebase; both the
# dashboard and the chatbot go through it. This also makes `verified` a provable status
# instead of an LLM's self-assessment (see orchestrator.py's use of `used_canonical`),
# and removes an entire class of "is this queryable" judgment calls from the model — a
# false "inventory turnover ratio isn't available by hospital" claim is impossible once
# the tool just tries Plant="HC05" and gets a real number back (verified: itr_insights
# returns portfolio_itr=80.56 for HC05 on the first try, no reasoning required).
#
# ADDING A NEW KPI: one entry here. No prompt-engineering, no new SQL, no new tool.
from __future__ import annotations

import inspect
from typing import Any, Callable, Optional

from fastapi.params import Query as _QueryParam

from app.ai import hcg_reports
from app.api import legacy_kpi
from app.core import data_access as da

# Every entry: key -> (callable, display label, accepts Plant?, accepts Category?, one-line
# description). The callable is the REAL FastAPI route handler, called directly as a plain
# Python function (verified empirically: Query(None) defaults are never touched because every
# call site here always passes explicit Plant=/Category= — see call_kpi below), or for the
# Jun–Aug 2026 reports an app/ai/hcg_reports function that calls those handlers in-process.
#
# Optional fields:
#   hint   — the short text the get_kpi schema shows next to the label. Only entries whose label
#            alone would mislead the model carry one; `about` is the fuller note for people.
#            (Every `about` in the schema made it 5.4K characters, resent on every tool round.)
#   args   — extra argument names (from EXTRA_ARGS) the entry accepts; call_kpi forwards only
#            these and reports the rest back as _ignored_args.
#   ratio  — the figure is a percentage, average or ratio of sums: never add or average it
#            across hospitals (the scope guards word their advice from this).
#   needs  — "formulary" / "spend": the published data folder the entry reads (tests skip the
#            entry when it is not built).
#   window — "reports": the entry covers HCG's report months, not the warehouse's sales months.
#            Both are read from the data (app/ai/windows.py); no month is written here.
REPORT_WINDOW = "reports"
KPI_REGISTRY: dict[str, dict[str, Any]] = {
    "inventory-turnover-ratio": {
        "fn": legacy_kpi.itr_insights, "label": "Inventory Turnover Ratio", "plant": True, "category": True, "ratio": True,
        "about": "How many times stock cycles through per year (portfolio_itr). Company-wide or scoped to one hospital.",
    },
    "days-on-hand": {
        "fn": legacy_kpi.doh_insights, "label": "Days on Hand", "plant": True, "category": True, "ratio": True,
        "about": "Median/mean days of stock cover, and the moving vs non-moving SKU split.",
    },
    "inventory-health-score": {
        "fn": legacy_kpi.health_insights, "label": "Inventory Health Score", "plant": True, "category": True, "ratio": True,
        "about": "0-100 health score per SKU rolled up into Healthy / Watch / At Risk tiers, with value and average score per tier.",
    },
    "stock-change": {
        "fn": legacy_kpi.stockchange_insights, "label": "Stock Level Change", "plant": True, "category": True,
        "about": "How stock value/quantity has moved.",
    },
    "non-moving-inventory": {
        "fn": legacy_kpi.nonmoving_insights, "label": "Non-Moving Inventory", "plant": True, "category": True,
        "about": "Capital locked in stock with no recent consumption — blocked value, SKU count, and reasons.",
    },
    "inventory-risk": {
        "fn": legacy_kpi.risk_insights, "label": "Inventory Risk Classification", "plant": True, "category": True,
        "about": "SKUs classified High/Medium/Low risk, with count and value per tier.",
    },
    "near-expiry": {
        "fn": legacy_kpi.nearexp_insights, "label": "Near-Expiry Inventory", "plant": True, "category": True,
        "about": "Stock approaching or past expiry, bucketed by days-to-expiry, with value at stake.",
    },
    "stock-out-rate": {
        "fn": legacy_kpi.stock_out_insights, "label": "Stock-Out Rate", "plant": True, "category": True, "ratio": True,
        "about": "Share of actively-moving (hospital, material) pairs currently at zero stock, by count and by consumption value.",
    },
    "wastage-rate": {
        "fn": legacy_kpi.wastage_rate_insights, "label": "Wastage Rate", "plant": True, "category": True, "ratio": True,
        "about": "Value and quantity of stock that expired (wastage), as a percentage of total stock.",
    },
    "aging-distribution": {
        "fn": legacy_kpi.aging_distribution_insights, "label": "Inventory Aging Distribution", "plant": True, "category": True,
        "about": "Stock value bucketed by age (0-30d, 31-90d, 91-180d, 181-365d, 365d+), with fresh/dead percentages.",
    },
    "inventory-valuation": {
        "fn": legacy_kpi.valuation_insights, "label": "Inventory Valuation", "plant": True, "category": True,
        "about": "Total stock value at cost vs MRP, markup percentage, by age and category.",
    },
    "purchase-value": {
        "fn": legacy_kpi.purchase_value_insights, "label": "Purchase Value", "plant": True, "category": True,
        "about": "Total procurement spend, by category/plant breakdown.",
    },
    "procurement-variance": {
        "fn": legacy_kpi.variance_insights, "label": "Procurement Spend Variance", "plant": True, "category": True,
        "about": "Month-over-month procurement spend and its percentage change, per plant/category.",
    },
    "vendor-volume-contribution": {
        "fn": legacy_kpi.vendor_volume_insights, "label": "Vendor Volume Contribution", "plant": True, "category": True,
        "about": "Top vendors ranked by spend, with concentration.", "hint": "top vendors by spend",
    },
    "purchase-by-location": {
        "fn": legacy_kpi.location_insights, "label": "Purchase by Location", "plant": True, "category": True,
        "about": "Procurement spend broken down by hospital/location.", "hint": "procurement spend by hospital",
    },
    "procurement-cycle-time": {
        "fn": legacy_kpi.cycle_insights, "label": "Procurement Cycle Time", "plant": True, "category": True, "ratio": True,
        "about": "Average PO and PR processing time in days (avg_po, avg_pr) — the dashboard's canonical cycle-time definition.",
    },
    "vendor-lead-time": {
        "fn": legacy_kpi.lead_insights, "label": "Vendor Lead Time", "plant": True, "category": True, "ratio": True,
        "about": "Average delivery lead time per vendor.",
    },
    "fill-rate": {
        "fn": legacy_kpi.fill_insights, "label": "Fill Rate", "plant": True, "category": True, "ratio": True,
        "about": "Percentage of ordered quantity actually fulfilled, network-wide and per hospital, with best/worst.",
    },
    "monthly-purchase-value": {
        "fn": legacy_kpi.monthly_purchase_insights, "label": "Monthly Purchase Value", "plant": True, "category": True,
        "about": "Procurement spend trended by month.",
    },
    "unit-sold-per-sku": {
        "fn": legacy_kpi.units_consumed_insights, "label": "Units Consumed per SKU", "plant": True, "category": True,
        "about": "Consumption quantity per SKU.", "hint": "consumption quantity per SKU",
    },
    "consumption-by-department": {
        "fn": legacy_kpi.dept_insights, "label": "Consumption by Department", "plant": True, "category": False,
        "about": "Consumption value broken down by department/cost-centre. Category filter is NOT supported here.",
        "hint": "by department / cost centre; no category filter",
    },
    "demand-forecast": {
        "fn": legacy_kpi.demand_insights, "label": "Demand Forecast", "plant": True, "category": True,
        "about": "3-month forward demand-quantity forecast with confidence bounds. NOT a revenue/sales figure.",
        "hint": "3-month quantity forecast, not revenue",
    },
    "cashflow-forecast": {
        "fn": legacy_kpi.cashflow_insights, "label": "Cashflow / Budget Forecast", "plant": True, "category": True,
        "about": "3-month forward procurement cashflow forecast with confidence bounds.",
        "hint": "3-month procurement cashflow forecast",
    },
    "reorder-priority": {
        "fn": legacy_kpi.reorder_priority, "label": "Reorder Priority", "plant": True, "category": True,
        "about": "Items that need reordering now, ranked by urgency band, with recommended quantity. This is the canonical reorder/replenishment answer — NOT the same as near-expiry or non-moving stock.",
        "hint": "items to reorder now; not near-expiry or non-moving",
    },
    "replenishment-risk": {
        "fn": legacy_kpi.replenishment_insights, "label": "Replenishment & Aging Risk", "plant": True, "category": True,
        "about": "Combined replenishment need and aging-risk view.",
    },
    "revenue-margin": {
        "fn": legacy_kpi.revenue_insights, "label": "Revenue & Margin", "plant": False, "category": False,
        "about": ("Company-wide ONLY — no product filter. Gives revenue/cost/margin by manufacturer, the top "
                  "products as <<SALES_MONTHS>>-month totals, and a WHOLE-COMPANY monthly timeline. For one product's monthly "
                  "sales use run_sql on sales_by_material_month."),
        "hint": ("<<SALES_SHORT>> sales, company-wide ONLY: no product or hospital filter. One product's monthly sales "
                 "→ run_sql on sales_by_material_month"),
    },
    "billable-consumption": {
        "fn": legacy_kpi.billable_split_insights, "label": "Billable vs Non-Billable Consumption", "plant": False, "category": True,
        "about": "Consumption split into billed-to-patient vs internal/non-billable, by value. Plant filter is NOT supported here.",
        "hint": "no hospital filter",
    },
    # ── HCG's Jun–Aug 2026 reports (app/ai/hcg_reports.py) ──────────────────────────────
    # A separate window from everything above: there is no warehouse table for them, so
    # run_sql can never reach these figures. Every one is a ratio of sums (ratio=True).
    "formulary-compliance": {
        "fn": hcg_reports.formulary_compliance, "label": "Formulary Compliance", "plant": True, "category": True,
        "ratio": True, "needs": "formulary", "window": REPORT_WINDOW,
        "args": ("month", "payer", "patient_type", "segment", "method", "view", "doctor", "material",
                 "formulary_group"),
        "about": ("<<REPORTS>> ONLY. HCG's monthly formulary compliance: compliance % = Formulary ÷ (Formulary + "
                  "Out of Formulary) drug cost, out-of-formulary / non-formulary spend, GOV restricted and preferred "
                  "brand figures; by month, hospital, segment, IP/OP, payer, class, doctor and item. NOT for share of "
                  "sales revenue by formulary status (<<SALES_SHORT>>, use run_sql on sales_* joined to "
                  "dim_material.formulary_status)."),
        "hint": ("<<REPORTS>> ONLY: compliance %, out-of-formulary spend; views summary/trend/hospitals/segments/"
                 "types/payers/classes/doctors/items. Not <<SALES_SHORT>> sales by formulary status"),
    },
    "brand-rules": {
        "fn": hcg_reports.brand_rules, "label": "Brand Rules (restricted / preferred brands)", "plant": True,
        "category": True, "ratio": True, "needs": "formulary", "window": REPORT_WINDOW,
        "args": ("month", "patient_type", "segment", "measure", "view", "doctor", "material", "payer"),
        "about": ("<<REPORTS>> ONLY. SPEND on brands that break HCG's brand rules, per payer (measure): restricted / "
                  "preferred / out-of-formulary brands for scheme patients, RESTRICTED BRANDS FOR CASH & TPA PATIENTS "
                  "(restricted_cash_tpa), SCH brands for cash & TPA, -OPD brands on inpatients; each as cost and share "
                  "of ALL drug spend, by month, hospital, doctor and brand, plus the rule book."),
        "hint": ("<<REPORTS>> ONLY: SPEND on restricted / preferred brands per payer, incl. restricted brands for "
                 "cash & TPA patients (measure); views measures/trend/hospitals/doctors/items/book"),
    },
    "scheme-brand-saving": {
        "fn": hcg_reports.scheme_saving, "label": "Scheme Brand Saving Missed", "plant": True, "category": True,
        "ratio": True, "needs": "spend", "window": REPORT_WINDOW,
        "args": ("month", "patient_type", "segment", "view", "doctor", "material", "scheme_group"),
        "about": ("<<REPORTS>> ONLY. Saving missed when scheme (GOV) patients got a brand for which HCG's scheme "
                  "list suggests a cheaper one (every suggestion; Attis and J&J & Covidien → Healthium slices)."),
        "hint": "<<REPORTS>> ONLY: saving missed on scheme patients' brands vs the scheme list (incl. Healthium)",
    },
    "attis-saving": {
        "fn": hcg_reports.attis_saving, "label": "Attis Saving Missed", "plant": True, "category": True,
        "ratio": True, "needs": "spend", "window": REPORT_WINDOW,
        "args": ("month", "patient_type", "segment", "view", "doctor", "material", "payer"),
        "about": ("<<REPORTS>> ONLY. Saving missed when scheme (GOV) patients got a brand for which HCG's scheme "
                  "list suggests an Attis brand; plus Attis consumption across every payer."),
        "hint": "<<REPORTS>> ONLY: saving missed vs the Attis brand for scheme patients; Attis consumption",
    },
    "cash-tpa-margin-loss": {
        "fn": hcg_reports.cash_tpa_margin_loss, "label": "Cash & TPA Margin Loss", "plant": True, "category": True,
        "ratio": True, "needs": "spend", "window": REPORT_WINDOW,
        "args": ("month", "payer", "patient_type", "segment", "view", "doctor", "material"),
        "about": ("<<REPORTS>> ONLY. Margin HCG lost on cash and TPA patients given a brand for which its list "
                  "suggests a higher-margin one; loss % = loss ÷ the suggested brands' margin. NOT the spend on "
                  "restricted brands for cash & TPA patients: that is brand-rules, measure restricted_cash_tpa."),
        "hint": ("<<REPORTS>> ONLY: MARGIN LOST on cash & TPA patients' brands vs the higher-margin suggestion "
                 "(not spend on restricted brands: brand-rules)"),
    },
}

# Arguments beyond plant/category that some entries take (each entry lists its own in `args`).
EXTRA_ARGS: tuple[str, ...] = ("month", "payer", "patient_type", "segment", "method", "view", "doctor",
                               "material", "measure", "formulary_group", "scheme_group")


def _neutral_kwargs(fn: Callable) -> dict[str, Any]:
    """Every OTHER Query(...)-defaulted parameter this function has, beyond Plant/Category
    (e.g. reorder_priority's band/status/q/sort/limit/offset), resolved to its REAL intended
    default value.

    This one bit us empirically while building this registry: calling reorder_priority()
    with only Plant/Category set raised `TypeError: int() argument must be ... not 'Query'`,
    because a parameter's `Query(None)` default is a live fastapi.params.Query OBJECT when
    the function is called directly in Python (bypassing FastAPI's own request-time
    dependency resolution) — not None, and not falsy. `Query(x).default` recovers the real
    x. Doing this generically via signature introspection, rather than hand-listing
    band/status/q/sort/limit/offset for reorder_priority alone, means the NEXT KPI added to
    the registry with its own extra Query-defaulted filters (pagination, sort, whatever)
    works correctly on the first try instead of needing its own bespoke patch here."""
    out: dict[str, Any] = {}
    for name, p in inspect.signature(fn).parameters.items():
        if name in ("Plant", "Category"):
            continue
        if isinstance(p.default, _QueryParam):
            out[name] = p.default.default
    return out


def call_kpi(key: str, plant: Optional[str] = None, category: Optional[str] = None, **extra) -> dict:
    """Call the canonical function for `key` with explicit, real arguments — never the bare
    FastAPI route handler's own Query(None) default. Every call site here passes plant/
    category explicitly, defaulting to "All Plants" / None exactly as the dashboard's own
    default view does, so a canonical answer with no explicit scope matches what a user
    sees loading the dashboard page fresh.
    Raises KeyError for an unknown key — callers should validate against KPI_REGISTRY first
    (the tool schema's enum already constrains the model to real keys, but defend anyway).

    `extra` (month, view, doctor...) reaches the function only when the entry lists the name in
    its `args`. Anything else that was actually set comes back as `_ignored_args`, so a month
    passed to a Dec–May KPI is reported as unused instead of silently answering for all months.

    A plant a Dec–May KPI cannot resolve (a comma list, a city, a typo) raises ValueError
    rather than reaching the handler — see _dashboard_plant for why."""
    entry = KPI_REGISTRY[key]
    kwargs: dict[str, Any] = _neutral_kwargs(entry["fn"])
    if entry["plant"]:
        if entry.get("needs"):
            # the Jun–Aug reports resolve names and comma lists themselves, and raise on a bad one
            kwargs["Plant"] = plant if plant else "All Plants"
        else:
            kwargs["Plant"] = _dashboard_plant(key, plant, bool(entry.get("ratio")))
    if entry["category"]:
        # Fail loudly on a category this KPI cannot actually filter on. filter_category
        # equality-matches the six derived buckets, so anything else (a material_group like
        # "Injections", a drug class, a typo) silently yields an EMPTY frame and the KPI
        # returns a confident 0.0 with no indication anything went wrong. A raised error
        # reaches the model as a tool error it can correct from; a zero does not.
        # NB: resolve_category returns an UNRECOGNISED value unchanged (deliberately — see
        # its docstring; that makes filter_category yield empty rather than silently
        # unfiltered for the dashboard, whose selector is driven off /meta/categories).
        # So `is None` never fires here — membership in CATEGORIES is the real test.
        if category and da.resolve_category(category) not in da.CATEGORIES:
            raise ValueError(
                f"'{category}' is not a material category. Valid categories: {list(da.CATEGORIES)}. "
                f"If you meant a material_group / dosage form (e.g. 'M065-INJECTIONS', 'Tablets') "
                f"or a drug class, use run_sql instead — get_kpi cannot filter on those."
            )
        kwargs["Category"] = category or None
    accepted = set(entry.get("args", ()))
    ignored = []
    for name, value in extra.items():
        if isinstance(value, str):
            value = value.strip()
        if value is None or value == "" or value == [] or value == ():
            continue
        if name in accepted:
            kwargs[name] = value
        else:
            ignored.append(name)
    result = entry["fn"](**kwargs)
    # Provenance tag: how the caller (orchestrator) knows this came from the canonical
    # path and can mark the eventual answer's `verified` status accordingly, rather than
    # trusting the model's own uncertainty judgment (the audit found that self-assessment
    # fired on exactly-correct answers and stayed silent on the worst error found).
    out = {"_kpi_key": key, "_kpi_label": entry["label"], "_canonical": True, "data": result}
    if ignored:
        out["_ignored_args"] = sorted(ignored)
    return out


# What the model sends when it means "every hospital". resolve_plant already treats the first
# two as all; the third is the same intent in the chat's own wording.
_ALL_PLANTS = ("ALL", "ALL PLANTS", "ALL HOSPITALS")


def _dashboard_plant(key: str, plant: Optional[str], ratio: bool) -> str:
    """The Plant= value for a Dec–May dashboard handler: a real plant code, or "All Plants".

    Those handlers filter through da.filter_plant, and a value resolve_plant cannot match
    comes back None — which filter_plant reads as "no filter". That is right for the
    dashboard's region selector, and wrong here: plant="HC01,HC05,HC06,HC40" (the comma list
    the Jun–Aug reports accept), "Bangalore", "hc05" or even "HC05 " all returned the WHOLE
    network's figure, labelled as the hospitals the model asked for, and no scope guard
    noticed because the call did pass a plant. Raising instead reaches the model as a tool
    error it can correct from. The resolved code is what gets forwarded, so a lower-case or
    padded code still works."""
    s = str(plant or "").strip()
    if not s or s.upper() in _ALL_PLANTS:
        return "All Plants"
    if "," in s:
        raise ValueError(
            f"get_kpi('{key}') takes ONE hospital per call, not a list ('{s[:60]}'). Call it once per "
            f"code, then " + ("quote each site's figure on its own: per-site ratios do not combine into "
                              "one." if ratio else "total the per-site figures.")
            + " Comma lists work only for the <<REPORTS>> report keys.")
    code = da.resolve_plant(s) or da.resolve_plant(s.upper())
    if not code:
        from app.ai import hospitals
        alias = hospitals.resolve(s)
        if alias and len(alias) == 1:
            return alias[0]
        if alias:
            raise ValueError(
                f"'{s[:40]}' covers {len(alias)} hospitals: {hospitals.describe(alias)}. get_kpi('{key}') "
                f"takes ONE hospital per call: call it once per code, then "
                + ("quote each site's figure on its own." if ratio else "total the per-site figures."))
    if not code:
        raise ValueError(
            f"plant='{s[:60]}' is not a hospital get_kpi('{key}') recognises, so it would have answered "
            f"for the WHOLE network. Pass one plant code (e.g. 'HC05') or hospital name (a city: one call "
            f"per hospital), or omit plant for the company-wide figure.")
    return code


# Per-argument schema for EXTRA_ARGS. Enums wherever the handlers accept a fixed set, so a wrong
# value is unsendable rather than a 400 (or worse, a silent zero). measure is the six static
# rule keys, not read from formulary_validation.json at import: an empty enum on a server
# without that file would make the whole tool schema invalid.
_EXTRA_SCHEMA: dict[str, dict] = {
    "month": {"type": "string", "description": (
        "<<REPORTS>> report keys only: 'Aug', 'August 2026', '2026-08', a comma list ('2026-07,2026-08') or "
        "'all'. Default = the latest month (<<REPORTS_LATEST>>). State the month(s) you used.")},
    "payer": {"type": "string", "enum": ["CASH", "CASH(TPA)", "GOV"], "description": (
        "Payer bucket; GOV = scheme patients. brand-rules picks its measure from it; cash-tpa-margin-loss takes "
        "CASH / CASH(TPA) only.")},
    "patient_type": {"type": "string", "enum": ["IP", "OP"], "description": "Inpatient or outpatient."},
    "segment": {"type": "string", "enum": ["onco", "non-onco", "consumables"],
                "description": "consumables: spend keys only (compliance covers drugs)."},
    "method": {"type": "string", "enum": ["pooled", "mean"], "description": (
        "formulary-compliance: pooled (default, HCG's current method) or mean of onco and non-onco "
        "(HCG's earlier method).")},
    "view": {"type": "string", "enum": ["summary", "trend", "hospitals", "segments", "types", "payers",
                                        "classes", "doctors", "items", "measures", "book", "consumption"],
             "description": ("Which breakdown of a report key: trend = by month, hospitals = every hospital "
                             "ranked. brand-rules default 'measures'; the others 'summary'.")},
    "doctor": {"type": "string", "description": "Treating doctor's name or doctor_id (report keys)."},
    "material": {"type": "string", "description": "Item / brand name or material code (report keys)."},
    "measure": {"type": "string", "enum": ["restricted_scheme", "preferred_scheme", "oof_scheme",
                                           "restricted_cash_tpa", "opd_brand_inpatient", "scheme_brand_cash_tpa"],
                "description": ("brand-rules: which rule. restricted_scheme / preferred_scheme / oof_scheme = scheme "
                                "(GOV) patients, kept per doctor; restricted_cash_tpa = restricted brands for cash & "
                                "TPA patients; scheme_brand_cash_tpa = scheme brands for cash & TPA; "
                                "opd_brand_inpatient = outpatient-only brands given to inpatients.")},
    "formulary_group": {"type": "string", "enum": ["A", "B", "C"], "description": (
        "formulary-compliance items: A formulary, B out of formulary (default), C non formulary.")},
    "scheme_group": {"type": "string", "enum": ["attis", "healthium"],
                     "description": "scheme-brand-saving: narrow to the Attis or the J&J & Covidien → Healthium slice."},
}


def _hospital_list() -> str:
    from app.ai import hospitals
    return hospitals.prompt_line()


def tool_schema() -> dict:
    """The get_kpi function-calling tool. The `kpi` enum is generated FROM the registry —
    adding a KPI to KPI_REGISTRY above is the only step needed for the model to be able to
    call it; there is no second list to keep in sync (that drift is exactly how "ad hoc SQL
    is the only option" bugs happen for KPIs added after the fact). Every key and label is
    listed; only the entries whose label alone would mislead carry a short hint."""
    keys = sorted(KPI_REGISTRY)

    def one(k: str) -> str:
        e = KPI_REGISTRY[k]
        return f"{k} ({e['label']}" + (f": {e['hint']}" if e.get("hint") else "") + ")"
    reports = [k for k in keys if KPI_REGISTRY[k].get("needs")]
    return {
        "type": "function", "function": {
            "name": "get_kpi",
            "description": (
                "Get the CANONICAL, dashboard-verified figure for a named KPI — the exact calculation the "
                "dashboard uses, not a re-derived approximation. Call this before writing SQL when the question "
                "names or clearly maps to one of these metrics; it takes a hospital (plant) or category, so never "
                "assume a hospital- or category-scoped figure 'isn't available' without trying it here. It has NO "
                "product filter (except `material` on the report keys): for one product's sales, consumption or "
                "trend use run_sql. The <<REPORTS>> report keys (" + ", ".join(reports) + ") take the extra "
                "arguments below and exist ONLY here; other KPIs ignore them (returned as _ignored_args). "
                "Percentages are ratios of sums: never add or average them across hospitals. Known KPIs: "
                + "; ".join(one(k) for k in keys)
            ),
            "parameters": {"type": "object", "required": ["kpi"], "properties": {
                "kpi": {"type": "string", "enum": keys, "description": "Which canonical KPI to fetch."},
                "plant": {"type": "string", "description": (
                    "ONE hospital code (e.g. 'HC05') or name to scope to, or omit for the company-wide figure. "
                    "Only the <<REPORTS>> report keys take several codes ('HC05,HC06'), for their pooled figure; "
                    "any other KPI refuses a list, so call it once per code. Ignored for KPIs that don't "
                    "support it. Short names work as given; the codes are: " + _hospital_list() + ".")},
                # HARD ENUM, not a free string. The old description's own example was
                # "e.g. 'Injections'" — but Injections is a MATERIAL GROUP, not one of the six
                # derived categories filter_category matches on, so following this tool's own
                # documentation returned an empty frame and the KPI answered `median_doh: 0.0,
                # total_skus: 0`. A confident zero, and because the turn was canonical-only the
                # audit was skipped and the UI showed no badge. Constraining the enum makes the
                # wrong value unsendable rather than silently wrong.
                "category": {
                    "type": "string", "enum": list(da.CATEGORIES),
                    "description": ("Derived material category to scope to, or omit for all. These six are the ONLY "
                                    "valid values. NOT the same as material_group — for a dosage-form filter like "
                                    "'M065-INJECTIONS'/'Injections'/'Tablets' use run_sql, not this argument. "
                                    "Ignored for KPIs that don't support it."),
                },
                **{name: _EXTRA_SCHEMA[name] for name in EXTRA_ARGS},
            }}},
    }
