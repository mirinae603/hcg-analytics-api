"""
Intent routing via embeddings (text-embedding-3-large on Azure).

WHY: the semantic layer's worked examples fix routing WHEN the model attends to the
right one — but as the example set grows and users phrase questions in their own words
("are we bleeding money on any drugs?" vs the canonical "are we overpaying?"), the model
sometimes reaches for the wrong table (sales margin instead of the price mart, raw
fact_po instead of the capex flag). Live testing traced EVERY residual "serious" answer
to a routing miss, not a data bug — the mart is correct whenever it's used.

This module closes that gap by retrieval: a small corpus of intent exemplars, each tied
to a concise ROUTING HINT (which table/mart + which approach). At query time we embed the
user's question, take the nearest exemplars by cosine similarity, and inject their hints
as a focused "MOST RELEVANT PATTERNS FOR THIS QUESTION" block at the top of the prompt —
so the right table is surfaced even for a phrasing nobody pre-wrote.

FAIL-OPEN by design: any embedding error (Azure hiccup, missing deployment) returns an
empty hint block, and the orchestrator falls back to the full static examples exactly as
before — routing can only ever ADD signal, never break a working turn.
"""
from __future__ import annotations
import math
import os
import re
import threading

EMBED_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT", "text-embedding-3-large")
TOP_K = 4
MIN_SIM = 0.32   # below this, the corpus has nothing relevant — inject nothing

# ── Deterministic intent LOCKS ───────────────────────────────────────────────
# Embedding retrieval generalises to unseen phrasings but is probabilistic — on the
# highest-stakes, most-mis-answered intents the model still sometimes ignores the hint
# (e.g. answering "are we overpaying?" with a top-by-value capex list). For those few
# intents a keyword match injects a NON-optional, high-priority directive that removes
# the coin-flip. Kept deliberately small — only for intents proven to mis-route.
_LOCKS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(overpay|over-?pay|overpaying|paying too much|price (inconsisten|leakage|dispersion|swing|variation)|inconsistent price)", re.I),
     "This is a PRICE-DISPERSION question → you MUST use mart_material_price_stats, rank by clean_price_spread (or price_ratio) among clean_n>=5, scoped by category. It is a WRONG answer to return a top-by-VALUE list, biggest purchase orders, capex lines, or a sales-margin/profit ranking."),
    (re.compile(r"\b(genuine|actual|real)\b.{0,25}\bsupply\b|\bsupply\b.{0,25}\bexcluding\b.{0,15}\b(capital|capex|service)|exclud\w*\b.{0,15}\b(capital|capex|service)", re.I),
     "Genuine medical SUPPLY spend → classify on fact_po.doc_type: Supply = doc_type NOT IN ('Service PO','Dom Capital PO','CMC','AMC','Imp capital PO') AND material IS NOT NULL (= ₹429 Cr). Do NOT additionally filter by material_type — NULL-material_type domestic-PO lines are still genuine supply and dropping them undercounts."),
]


def _locks_for(question: str) -> list[str]:
    q = question or ""
    return [d for rx, d in _LOCKS if rx.search(q)]

# ── Intent corpus: (exemplar phrasings, routing hint) ────────────────────────
# Each entry is several natural phrasings of ONE intent (embedded together as one
# vector by joining) mapped to a one-line directive telling the model the right
# table/mart and approach. Keep hints terse and imperative.
_CORPUS: list[tuple[str, str]] = [
    ("are we overpaying; price inconsistency; paying too much; price leakage; which items have inconsistent or swinging purchase prices; procurement price variation",
     "PROCUREMENT PRICE-DISPERSION → mart_material_price_stats, rank by clean_price_spread (or price_ratio) among clean_n>=5; scope with category. NOT sales margin, NOT profit."),
    ("which vendor is cheapest for an item; best price vendor; how much could we save switching vendor; vendor price comparison for a product; consolidate to cheapest supplier",
     "VENDOR PRICE / SAVINGS → mart_material_vendor_price_stats; compare ONLY price_is_stable (clean_n>=5) vendors — a 2-3 line cheap price is an outlier, never an achievable saving."),
    ("which vendors have inconsistent or unreliable pricing; which supplier's prices swing the most; vendor pricing reliability review",
     "VENDOR INCONSISTENCY → mart_procurement, rank vendors by SUM(is_price_outlier) with HAVING count(*)>=20 (repeat volume). price_is_stable=false means 'bought once', NOT 'inconsistent'."),
    ("which plant pays the most for an item; compare price across plants; per-unit cost by plant; is one hospital paying more per unit",
     "PER-UNIT CROSS-PLANT → mart_procurement median(unit_price) GROUP BY plant (show bool_or(is_price_outlier)); never rank by raw SUM(spend). Surface & caveat an anomalous plant, don't hide it."),
    ("what is the purchase price of a named item; how much do we pay for this drug; unit cost of a product",
     "SINGLE-ITEM PRICE → mart_material_price_stats clean_median_price + raw_n; report even if clean_n<5 (never refuse), light caveat on small samples."),
    ("capital vs supply spend; biggest purchase orders; genuine medical supply spend excluding capital and service; is this a capex or a stocked item",
     "CAPEX/SUPPLY SPLIT → classify fact_po.doc_type IN ('Service PO','Dom Capital PO','CMC','AMC','Imp capital PO') OR material IS NULL = Capex/Service; a robot/PET-CT/AMC is capex even with a material code. Never call it a 'medical supply'."),
    ("vendor lead time; how fast does a vendor deliver; delivery time for a supplier; PO to GRN turnaround for a vendor",
     "VENDOR LEAD TIME → kpi_vendor_lead_time by vendor_name (or mart_procurement's pre-joined vendor_median_lead_time_days). For a plant/portfolio cycle time use kpi_cycle_time."),
    ("total revenue; total sales; overall margin; revenue and margin; IP vs OP revenue split",
     "REVENUE/MARGIN → sales_totals for the canonical company total (₹521.67 Cr); margin=revenue-cost. NEVER sales_by_manufacturer for an overall total (it undercounts)."),
    ("revenue by manufacturer; top manufacturers; margin by brand/manufacturer",
     "BY MANUFACTURER → sales_by_manufacturer (revenue side is clean). For by-product revenue use sales_by_material (already de-duplicated)."),
    ("top products by revenue; most profitable drugs; highest margin items; best selling products",
     "BY PRODUCT → sales_by_material (deduped) ORDER BY revenue or (revenue-cost); apply a materiality floor before ranking by margin_pct."),
    ("total procurement spend; how much did we spend on purchasing; overall purchase value",
     "PROCUREMENT SPEND (total) → SUM(kpi_purchase_value.purchase_value) = ₹649.91 Cr (matches the dashboard). Not fact_grn received value."),
    ("procurement spend by category; spend on pharma or injections or a drug type; category-filtered spend",
     "CATEGORY/PHARMA SPEND → fact_po JOIN dim_material on material_type/material_group; kpi_purchase_value has NO material/type column and its 'category' is a raw messy field — never use it for a pharma/category filter."),
    ("which items are high risk; at-risk inventory items; risk classification of stock",
     "INVENTORY RISK CLASSIFICATION → kpi_risk_classification.risk_level ('High'/'Medium'/'Low'). Never filter stock_replenishment_and_aging_risk.aging_risk for 'High' (it holds month-buckets, returns 0)."),
    ("how much inventory is at risk of aging or going stale; slow-moving stock value; value at risk",
     "AGING/SLOW-MOVER VALUE → stock_replenishment_and_aging_risk.inventory_aging_risk=TRUE, or kpi_health_score for the health-tier view. Pick by whether the question is a classification or a value."),
    ("days of cover; DOH; how many days of stock; days of inventory",
     "DAYS OF COVER → MEDIAN(kpi_doh.doh_days) WHERE doh_days>0 for a portfolio figure; never AVG (skewed). Item-level extremes near-zero-consumption are artifacts — caveat them."),
    ("what is expiring; near-expiry stock; items expiring soon; expiry exposure",
     "EXPIRY → fact_inventory, date_diff('day', DATE '2026-05-31', expiry_date); bands per RULES. For cross-plant redistribution check the SAME material at other plants."),
    ("which departments or cost centres consume the most; consumption by department; department usage",
     "CONSUMPTION BY DEPARTMENT → kpi_consumption_by_department by cost_ctr (department names are just codes). For item-level detail within a cost centre use fact_consumption (has material)."),
    ("cost per unit consumed; most expensive item to consume; consumption unit cost",
     "CONSUMPTION UNIT COST → fact_consumption amount_lc/qty but apply the outlier/materiality guard (n>=5, drop near-zero-denominator single lines); a raw n=1 max is an artifact, caveat it."),
    ("demand forecast; predicted demand; cashflow forecast; forecast vs actual",
     "FORECAST → forecast_sales (demand-forecast model only, NOT billed sales). Never use it to answer a revenue question."),
    ("which items will stock out; reorder; replenishment; stockout risk; what to expedite",
     "STOCKOUT/REORDER → stock_replenishment_and_aging_risk (replenishment_quantity>0 = reorder). Flag non-clinical consumables that dominate raw lists."),
    ("out of formulary stock; formulary compliance; non-formulary items",
     "FORMULARY → fact_inventory.formulary (or dim_material.formulary). 'OUT OF FORMULARY' often explains dead stock."),
    ("procurement spend variance month over month; has spend variance improved; monthly spend change",
     "SPEND VARIANCE → recompute from SUM(purchase_value)/SUM(prev_value) on kpi_procurement_variance; NEVER AVG(variance_pct) (unweighted mean is dominated by tiny-base plants → nonsense %)."),
    ("inventory value by ward type; IP OP OT storage; stock by storage location",
     "WARD/SLOC → join dim_sloc ON plant AND sloc (composite key — sloc alone over-counts ~14x)."),
    ("who is the CEO; org chart; patient records; weather; external facts; which doctor requested",
     "OUT OF SCOPE / NOT IN DATA → say plainly it's outside the supply-chain data (no people/roles/patient/doctor/external dimension); do not fabricate or invent a proxy."),
]

_lock = threading.Lock()
_corpus_vecs: list[list[float]] | None = None   # cached embeddings, index-aligned to _CORPUS


def _embed(client, texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model=EMBED_DEPLOYMENT, input=texts)
    return [d.embedding for d in resp.data]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _ensure_corpus(client) -> list[list[float]] | None:
    global _corpus_vecs
    if _corpus_vecs is not None:
        return _corpus_vecs
    with _lock:
        if _corpus_vecs is None:
            try:
                _corpus_vecs = _embed(client, [c[0] for c in _CORPUS])
            except Exception:
                return None
    return _corpus_vecs


def hints_for(client, question: str, allow_locks: bool = True) -> str:
    """Return a focused routing-hint block for `question`, or '' on any failure.
    Never raises — routing is strictly additive. `allow_locks=False` on a scoped refinement
    so a hard entry-point lock can't override the conversation's active scope."""
    q = (question or "").strip()
    if not q:
        return ""
    # Locks are for FRESH entry-point questions. On a scoped refinement (allow_locks=False),
    # the active-scope instruction governs — a hard lock ('MUST use mart_material_price_stats
    # company-wide') would otherwise hijack e.g. "biggest price swings with THEM" away from the
    # vendor the user is drilling into. Suppress them there.
    locks = _locks_for(q) if allow_locks else []
    lock_block = ("⛔ REQUIRED APPROACH (do not deviate):\n" + "\n".join(f"• {d}" for d in locks)) if locks else ""
    try:
        vecs = _ensure_corpus(client)
        if not vecs:
            return lock_block
        qv = _embed(client, [q])[0]
        scored = sorted(((_cosine(qv, v), i) for i, v in enumerate(vecs)), reverse=True)
        picked = [(_CORPUS[i][1]) for sim, i in scored[:TOP_K] if sim >= MIN_SIM]
        hint_block = ""
        if picked:
            lines = "\n".join(f"• {h}" for h in picked)
            hint_block = ("🎯 MOST RELEVANT PATTERNS FOR THIS QUESTION (routing hints — pick the "
                          "table/approach these point to before writing SQL):\n" + lines)
        return "\n\n".join(b for b in (lock_block, hint_block) if b)
    except Exception:
        return lock_block   # locks are pure-regex, always available even if embeddings fail
