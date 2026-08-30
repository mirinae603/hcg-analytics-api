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

from app.api import legacy_kpi
from app.core import data_access as da

# Every entry: key -> (callable, display label, accepts Plant?, accepts Category?, one-line
# description for the model). The callable is the REAL FastAPI route handler, called directly
# as a plain Python function (verified empirically: Query(None) defaults are never touched
# because every call site here always passes explicit Plant=/Category= — see call_kpi below).
KPI_REGISTRY: dict[str, dict[str, Any]] = {
    "inventory-turnover-ratio": {
        "fn": legacy_kpi.itr_insights, "label": "Inventory Turnover Ratio", "plant": True, "category": True,
        "about": "How many times stock cycles through per year (portfolio_itr). Company-wide or scoped to one hospital.",
    },
    "days-on-hand": {
        "fn": legacy_kpi.doh_insights, "label": "Days on Hand", "plant": True, "category": True,
        "about": "Median/mean days of stock cover, and the moving vs non-moving SKU split.",
    },
    "inventory-health-score": {
        "fn": legacy_kpi.health_insights, "label": "Inventory Health Score", "plant": True, "category": True,
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
        "fn": legacy_kpi.stock_out_insights, "label": "Stock-Out Rate", "plant": True, "category": True,
        "about": "Share of actively-moving (hospital, material) pairs currently at zero stock, by count and by consumption value.",
    },
    "wastage-rate": {
        "fn": legacy_kpi.wastage_rate_insights, "label": "Wastage Rate", "plant": True, "category": True,
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
        "about": "Top vendors ranked by spend, with concentration.",
    },
    "purchase-by-location": {
        "fn": legacy_kpi.location_insights, "label": "Purchase by Location", "plant": True, "category": True,
        "about": "Procurement spend broken down by hospital/location.",
    },
    "procurement-cycle-time": {
        "fn": legacy_kpi.cycle_insights, "label": "Procurement Cycle Time", "plant": True, "category": True,
        "about": "Average PO and PR processing time in days (avg_po, avg_pr) — the dashboard's canonical cycle-time definition.",
    },
    "vendor-lead-time": {
        "fn": legacy_kpi.lead_insights, "label": "Vendor Lead Time", "plant": True, "category": True,
        "about": "Average delivery lead time per vendor.",
    },
    "fill-rate": {
        "fn": legacy_kpi.fill_insights, "label": "Fill Rate", "plant": True, "category": True,
        "about": "Percentage of ordered quantity actually fulfilled, network-wide and per hospital, with best/worst.",
    },
    "monthly-purchase-value": {
        "fn": legacy_kpi.monthly_purchase_insights, "label": "Monthly Purchase Value", "plant": True, "category": True,
        "about": "Procurement spend trended by month.",
    },
    "unit-sold-per-sku": {
        "fn": legacy_kpi.units_consumed_insights, "label": "Units Consumed per SKU", "plant": True, "category": True,
        "about": "Consumption quantity per SKU.",
    },
    "consumption-by-department": {
        "fn": legacy_kpi.dept_insights, "label": "Consumption by Department", "plant": True, "category": False,
        "about": "Consumption value broken down by department/cost-centre. Category filter is NOT supported here.",
    },
    "demand-forecast": {
        "fn": legacy_kpi.demand_insights, "label": "Demand Forecast", "plant": True, "category": True,
        "about": "3-month forward demand-quantity forecast with confidence bounds. NOT a revenue/sales figure.",
    },
    "cashflow-forecast": {
        "fn": legacy_kpi.cashflow_insights, "label": "Cashflow / Budget Forecast", "plant": True, "category": True,
        "about": "3-month forward procurement cashflow forecast with confidence bounds.",
    },
    "reorder-priority": {
        "fn": legacy_kpi.reorder_priority, "label": "Reorder Priority", "plant": True, "category": True,
        "about": "Items that need reordering now, ranked by urgency band, with recommended quantity. This is the canonical reorder/replenishment answer — NOT the same as near-expiry or non-moving stock.",
    },
    "replenishment-risk": {
        "fn": legacy_kpi.replenishment_insights, "label": "Replenishment & Aging Risk", "plant": True, "category": True,
        "about": "Combined replenishment need and aging-risk view.",
    },
    "revenue-margin": {
        "fn": legacy_kpi.revenue_insights, "label": "Revenue & Margin", "plant": False, "category": False,
        "about": "Company-wide billed revenue, cost and margin — by manufacturer, by product, and monthly timeline. Takes NO filters; the function itself returns every breakdown.",
    },
    "billable-consumption": {
        "fn": legacy_kpi.billable_split_insights, "label": "Billable vs Non-Billable Consumption", "plant": False, "category": True,
        "about": "Consumption split into billed-to-patient vs internal/non-billable, by value. Plant filter is NOT supported here.",
    },
}


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


def call_kpi(key: str, plant: Optional[str] = None, category: Optional[str] = None) -> dict:
    """Call the canonical function for `key` with explicit, real arguments — never the bare
    FastAPI route handler's own Query(None) default. Every call site here passes plant/
    category explicitly, defaulting to "All Plants" / None exactly as the dashboard's own
    default view does, so a canonical answer with no explicit scope matches what a user
    sees loading the dashboard page fresh.
    Raises KeyError for an unknown key — callers should validate against KPI_REGISTRY first
    (the tool schema's enum already constrains the model to real keys, but defend anyway)."""
    entry = KPI_REGISTRY[key]
    kwargs: dict[str, Any] = _neutral_kwargs(entry["fn"])
    if entry["plant"]:
        kwargs["Plant"] = plant if plant else "All Plants"
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
    result = entry["fn"](**kwargs)
    # Provenance tag: how the caller (orchestrator) knows this came from the canonical
    # path and can mark the eventual answer's `verified` status accordingly, rather than
    # trusting the model's own uncertainty judgment (the audit found that self-assessment
    # fired on exactly-correct answers and stayed silent on the worst error found).
    return {"_kpi_key": key, "_kpi_label": entry["label"], "_canonical": True, "data": result}


def tool_schema() -> dict:
    """The get_kpi function-calling tool. The `kpi` enum is generated FROM the registry —
    adding a KPI to KPI_REGISTRY above is the only step needed for the model to be able to
    call it; there is no second list to keep in sync (that drift is exactly how "ad hoc SQL
    is the only option" bugs happen for KPIs added after the fact)."""
    keys = sorted(KPI_REGISTRY)
    enum_desc = "; ".join(f"{k} ({KPI_REGISTRY[k]['label']}: {KPI_REGISTRY[k]['about']})" for k in keys)
    return {
        "type": "function", "function": {
            "name": "get_kpi",
            "description": (
                "Get the CANONICAL, dashboard-verified figure for a named KPI — the exact same "
                "calculation the dashboard card for that KPI uses, not a re-derived approximation. "
                "ALWAYS call this FIRST for any question that names or clearly maps to one of these "
                "known metrics, before writing ad hoc SQL. It accepts a specific hospital (Plant) or "
                "category — if a hospital-scoped or category-scoped figure exists, this tool returns "
                "it directly, so never assume a breakdown 'isn't available' without having tried the "
                "matching plant/category argument here first. Known KPIs: " + enum_desc
            ),
            "parameters": {"type": "object", "required": ["kpi"], "properties": {
                "kpi": {"type": "string", "enum": keys, "description": "Which canonical KPI to fetch."},
                "plant": {"type": "string", "description": "A specific hospital code (e.g. 'HC05') to scope to, or omit/leave blank for the company-wide figure. Ignored for KPIs that don't support it."},
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
            }}},
    }
