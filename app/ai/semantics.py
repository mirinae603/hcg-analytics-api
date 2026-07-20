"""
Semantic layer — the business context that makes the agent's SQL CORRECT.
Assembled into the system prompt: schema + join map + business rules + gotchas +
worked examples. This is what turns "text-to-SQL" into "text-to-RIGHT-SQL".
"""
from __future__ import annotations
from app.ai import warehouse

# ── Business rules (the definitions an analyst must know) ───────────────────
RULES = """BUSINESS DEFINITIONS (use these exactly):
• Currency: format as ₹Cr (crore = 10,000,000) and ₹L (lakh = 100,000). 1 Cr = 100 L.
• Window: all sales / purchase / consumption data covers 6 months (Dec-2025 … May-2026). Inventory & expiry are an as-on SNAPSHOT dated 2026-05-31.
• "Sales" / "revenue" ALWAYS means the billed pharmacy sales in the sales_* tables (sales_totals, sales_by_material, sales_by_manufacturer, sales_by_hospital, sales_by_material_*, sales_monthly). margin = revenue − cost. margin_pct = margin / revenue * 100. REAL billed IP+OP revenue with true margin (billed MRP − actual cost). CANONICAL TOTAL: for an OVERALL revenue/margin figure use sales_totals (SUM over its IP+OP rows) — the authoritative company total (₹521.67 Cr); sales_by_material sums to the same. Do NOT take an overall total from sales_by_manufacturer — it UNDERCOUNTS ~0.4% (materials with no manufacturer mapping drop out); use it ONLY for a by-manufacturer breakdown.
• ⚠️ forecast_sales is NOT billed sales. It is the demand-FORECAST model's own series — sparse and on a different scale (its sales_quantity/sales_value total only ~₹67 Cr vs the real ₹521 Cr). NEVER use forecast_sales (or its sales_quantity/sales_value columns) to answer a "sales"/"revenue" question — those come only from the sales_* tables above. Use forecast_sales ONLY for demand/forecast questions (predicted demand, cashflow forecast, forecast vs actual).
• Procurement margin is only a PROXY: on fact_grn rows where unit_mrp>0 AND net_price>0, mrp_value = gr_qty*unit_mrp, cost_value = gr_qty*net_price, margin_pct = (mrp_value−cost_value)/mrp_value. Call it "MRP-proxy margin", never billed margin.
• Purchase spend = fact_grn.total_amount_wo_tax (or gr_qty*net_price); PO value = fact_po.total_value_wo_tax.
• Days of cover (DOH) = kpi_doh.doh_days; report the MEDIAN over rows where doh_days>0 (the mean is skewed by overstock — never report the mean). In prose call it the "median" or "typical" days of cover — do NOT call it the "average" (a median is not an average). This applies to any metric where you use a median.
• Expiry: compute days-to-expiry as date_diff('day', DATE '2026-05-31', expiry_date) on fact_inventory (qty>0, expiry_date not null). Bands: Expired (<0), 0-30, 31-90, 91-180, 181-365, 365+. "Near-expiry / actionable" = Expired + everything ≤180 days.
• Manufacturer lives in dim_material.manufacturer_desc (join on material) and in sales_by_manufacturer / sales_by_material_mfr for revenue.
• Department: in THIS dataset there are NO readable department names — dim_costcenter.department_name and kpi_consumption_by_department.department_name are both just the cost-centre CODE (a number like '1080101005'). Group consumption by cost_ctr and present it as "Cost centre <code>"; do NOT imply a named department exists. Treat cost_ctr as a text label, never a numeric value.
• Category = material_group (codes like 'M065-INJECTIONS', 'M113-TABLETS') — strip the 'M###-' prefix and Title-Case for display. This is what "category"/"injections"/"tablets"/"which category" ALWAYS means. dim_material also has major_group_desc/minor_group_desc — those are a DIFFERENT thing: the drug's pharmacological/therapeutic CLASS (e.g. 'TAXANE', 'MONOCLONAL ANTIBODY (EGFR)', 'ANTIANDROGEN') and molecule detail. Never filter major_group_desc for a physical-form category like "injections" — it holds drug-class names, not 'Injections'/'Tablets', and will silently match zero rows.
• "Pharma" / "pharmaceutical" / "drug" = dim_material.material_type IN ('ZNOC-Medical Non Onco Drugs', 'ZOC-Medical Onco Drugs'). Everything else (ZMC-Medical Consumables, ZNMC-Non Medical Consumables, ZLR-Laboratory Reagents, ZMA/ZNMA-Assets, lab calibrators/controls) is NOT pharma — it's equipment, consumables, or reagents. ~27% of the catalog has material_type=NULL (unclassified) — mention that coverage gap if it's relevant to the answer. Never guess a different filter for "pharma" — this is the one to use."""

# ── Which table for what + join keys ────────────────────────────────────────
GUIDE = """TABLE GUIDE (pick the right source):
REVENUE / MARGIN (billed sales, 6mo):
  sales_by_manufacturer(manufacturer, revenue, cost, qty)          — revenue/margin by manufacturer
  sales_by_hospital(hospital, revenue, cost, qty)                  — by billing hospital (23 hospital codes)
  sales_by_material(material, material_desc, material_group, revenue, cost, qty)     — by product (6-MONTH TOTAL, no month column)
  sales_by_material_mfr(manufacturer, material, material_desc, material_group, …)    — product×manufacturer (mfr drill)
  sales_by_material_hospital(hospital, material, material_desc, material_group, …)   — product×hospital (hospital drill)
  sales_monthly(patient['IP'|'OP'], month, revenue, cost, qty)     — WHOLE-COMPANY monthly revenue; ONLY columns are patient/month — it has NO material/manufacturer/category/hospital column, so it can NEVER be filtered or attributed to any specific item, brand, category or hospital. Use ONLY for the overall monthly revenue trend or the IP-vs-OP split.
  sales_totals(patient, revenue, cost, qty)                        — grand totals by IP/OP (canonical overall revenue = SUM of these rows)
INVENTORY (snapshot 2026-05-31):
  kpi_stock_value(plant, material, material_desc, material_group, stock_qty, stock_value_cost, stock_value_mrp)
  kpi_doh(plant, material, material_group, stock_qty, avg_daily_consumption, doh_days)
  kpi_aging_distribution(plant, material_group, aging_bucket, stock_value, sku_count)
  kpi_non_moving(material, material_desc, material_group, closing_stock_value, aging_days, last_sale_date, reason)
  kpi_health_score(material, material_group, closing_stock_value, health_score, health_tier, turnover_annualized)
  fact_inventory(plant, material, material_desc, batch, qty, total_cost, total_mrp, manufacturer_desc, material_group, expiry_date, snapshot_date)  — batch/expiry grain
  kpi_near_expiry(material, material_desc, batch, expiry_date, days_to_expiry, expiry_bucket, qty, total_cost)  — capped ≤180d; for full ladder use fact_inventory
PROCUREMENT:
  fact_grn(plant, material, vendor_name, po_no, gr_qty, net_price, unit_mrp, total_amount_wo_tax, major_group, gr_date)  — actual receipts & prices (255k rows)
  fact_po(plant, material, material_desc, vendor_name, po_no, po_qty, open_qty, net_price, total_value_wo_tax, major_group, po_date, year, month)  — orders; open_qty>0 = open PO
  kpi_monthly_purchase_value(year, month, plant, material, material_desc, material_group, monthly_purchase_value, purchase_qty)  — PER-MATERIAL monthly purchase trend (use this for an item's spend over time)
  kpi_purchase_value(plant, vendor_name, category, year, month, purchase_value, purchase_qty)  — by vendor/category, not per-material
  kpi_vendor_volume(plant, vendor_name, vendor_value, value_share_pct)  — per plant; SUM by vendor_name for portfolio
  kpi_vendor_lead_time(vendor_name, avg_lead_time_days, median_lead_time_days)
CONSUMPTION:
  kpi_units_consumed(year, month, plant, material, material_group, total_units, consumption_cost)
  kpi_consumption_by_department(plant, cost_ctr, department_name, month, consumption_qty, consumption_cost)
  fact_consumption(material, plant, cost_ctr, qty, amount_lc, posting_date, month)  — line grain
FORECAST / RISK:
  stock_replenishment_and_aging_risk(plant, material_id, material_desc, material_group, closing_stock, demand_monthly, replenishment_quantity, aging_risk)  — replenishment_quantity>0 = reorder
  kpi_stock_radar(plant, material_id, material_desc, closing_stock, demand_forecast, coverage_months, radar_status)
  kpi_fulfillment(plant, material_id, closing_stock, demand_monthly, fulfillment_rate, coverage_months)
  forecast_sales(material_id, plant, material_group, posting_date, sales_quantity, sales_value, sales_quantity_forecast, cashflow_forecast)  — actual+forecast monthly
DIMENSIONS (join keys):
  dim_material(material → material_desc, material_group[physical category, e.g. 'M065-INJECTIONS'], manufacturer_desc, generic_name, major_group_desc[drug CLASS e.g. 'TAXANE' — NOT a category filter])
  dim_vendor(vendor_code → vendor_name)   dim_plant(plant → plant_name)   dim_costcenter(cost_ctr → department_name)

JOIN KEYS: material (universal — bridges sales/inventory/procurement/consumption to dim_material for manufacturer/generic/category); vendor_name; plant; cost_ctr; po_no (grn↔po).
NOTE: 'material' in sales/kpi tables == 'material' in dim_material == 'material_id' in the risk/radar/fulfillment tables."""

# ── Gotchas that cause WRONG answers ────────────────────────────────────────
GOTCHAS = """CRITICAL GOTCHAS (ignore these and the numbers are wrong):
1. sales_* tables have NO plant/region column — they are hospital-based (23 hospitals ≠ the plant codes). Don't try to filter sales by plant.
2. Inventory/procurement KPI tables are per (plant × material). For a PORTFOLIO number, SUM across plants — don't report a single plant's row.
3. kpi_vendor_volume has one row per vendor×plant — SUM vendor_value GROUP BY vendor_name before ranking vendors.
4. DOH: report MEDIAN(doh_days) WHERE doh_days>0, never AVG (mean is ~1445d nonsense due to overstock tail; median ≈ 121d).
5. Purchases-by-MANUFACTURER via dim_material.manufacturer_desc covers only ~18% of purchase VALUE (master is sparse for high-value items) — if asked, compute it but STATE the coverage caveat. Manufacturer on the REVENUE side (sales_by_manufacturer) is clean.
6. month is a text label ('December','January',…). To sort chronologically use a CASE/ordering, not alphabetical. Prefer sales_monthly.month which is 'YYYY-MM'.
7. Always round money to ₹Cr/₹L in the FINAL answer, but keep raw values in SQL for correctness.
8. ⛔ GRAIN IS LAW (the single most important rule). The DIMENSIONAL MODEL section lists, for EVERY table, exactly which dimensions it can be sliced by and whether it has a time axis. A number is only about an entity if the query that produced it actually GROUPed/filtered a table that carries that dimension. So: (a) only filter/GROUP BY a column in that table's "slice by" list; (b) only show a month/period trend from a table whose "time axis" is not NONE; (c) if the breakdown the user wants needs a dimension crossed with a time axis that DON'T co-exist in any single table, that exact breakdown is NOT AVAILABLE — do NOT approximate it by taking a broader table's numbers and relabelling them to the narrower scope (that is a confident factual error). Give the closest correct cut the model DOES support and say plainly what isn't available. (Illustrative trap: a specific item's month-by-month SALES — no table carries item AND a sales time axis, so report the item's TOTAL sales, and if useful its monthly PURCHASES/CONSUMPTION which DO exist per item×month, each clearly labelled as purchase/consumption — never as sales.)
9. dim_material is the full CATALOG — many SKUs have ZERO transactions (no sales, no purchases). If an item returns no rows in the fact tables, DON'T call it a technical error: say plainly it has no recorded sales/purchases in this 6-month window, and offer its sibling SKUs (same generic_name or material_group that DO have activity, e.g. via dim_material) so the question isn't a dead end.
10. Column names are already clean — use material_desc / material_group (never a bare `desc` or `group`; those reserved words have been aliased away). Just write plain identifiers.
11. NAME LOOKUPS: a product's brand name (e.g. 'CALPOL', 'AUGMENTIN') lives in material_desc. To find a named item, filter material_desc ILIKE '%name%'. generic_name is the MOLECULE ('PARACETAMOL','TRAMADOL') — use it only to group products by active ingredient, NEVER to search for a brand. When the user quotes an item that returns nothing, try a looser material_desc ILIKE on the distinctive word (e.g. 'CALPOL') before concluding there's no data — the exact SKU string ('CALPOL-T TAB') may not exist even though the brand family does.
12. dim_material (24,931 SKUs) is the FULL multi-year item master; sales/PO/GRN/consumption only cover the 6-month analytics window, so most catalog SKUs (~57%) show zero rows there — that alone is not newsworthy. Before telling a user an item "has no data", ALWAYS also check fact_inventory (physical on-hand snapshot, which can hold batches received before the window even started) and fact_consumption. If it HAS inventory, that's the real story — report qty, aging_days, expiry_date (vs snapshot 2026-05-31 — flag if already expired) and formulary status (dim_material.formulary / fact_inventory.formulary: 'OUT OF FORMULARY' often explains why an item stopped moving). A catalog item with stock but no sales/purchase in-window is classic DEAD / NON-MOVING stock — say so plainly instead of a flat "no data".
13. aging_days (kpi_non_moving, fact_inventory) = age of the BATCH since it was received (GRN date), NOT days since last sale. "Non-moving" means ZERO consumption in the 6-month window regardless of how new the batch is — so a freshly-received batch (low aging_days) can still be non-moving. When reporting non-moving stock, lead with the value and the "no consumption in 6 months" reason; if you mention aging_days, call it "batch age", never imply it's time-since-last-sale (that would read as self-contradictory).
14. Reorder / stockout-risk tables (stock_replenishment_and_aging_risk, kpi_stock_radar, kpi_fulfillment) span ALL materials including non-clinical consumables (stationery, kitchen, housekeeping — material_group like 'Stationary'/'Kitchen'). By raw coverage these often top the list. If the top results are non-medicine, SAY SO and, unless the user asked broadly, either focus on drug categories or offer a medicine-only view — a hospital operator asking about "items to reorder" usually means clinical supplies.
15. ⛔ PRICE OUTLIERS ARE REAL AND COMMON in fact_grn/fact_po — 2.8% of materials (668 of 23,955) show a >20x net_price spread from data-entry mistakes: near-zero placeholder prices (₹0.01), price↔qty transposition (one line has price=1/qty=54140, another price=54140/qty=1 for the SAME material/date/vendor), or unit-of-measure inconsistency (one line ₹62 for a milk packet, another ₹40,000 for "1 unit" of the same SKU). A raw MIN/MAX/median-deviation query WILL surface these as a fake multi-crore "impact" for an ordinary low-value item — this is the single most likely way to hand a client an absurd, embarrassing number. For ANY overpay / price-deviation / price-swing / "biggest price increase" analysis, ALWAYS: (a) filter net_price >= 10 (kills placeholder prices); (b) compute a material's median from ALL its clean rows, THEN drop rows where net_price is outside median/8..median*8 (kills transposition/UOM errors — genuine price variance is rarely beyond an 8x band); (c) require at least 5 remaining rows per material before ranking it (a 1-2 line "impact" is not a stable finding). Never skip this — it is exactly what turns a nonsense ₹146 Cr result for a single screw into the honest, defensible answer (a clean top-10 in the ₹1-3 Cr range). If asked, you can mention that some rows were excluded as likely data-entry errors — that is itself a useful, honest finding to hand the client, not something to hide.
16. price_deviation_impact / overpay style metrics should be reported as an UNSIGNED magnitude (SUM(ABS(price - median)*qty)) — "how much this item's pricing swung, cost-wise" — consistently across every phrasing of the question (top-N ranking, min/max breakdown, single-item lookup). Do not switch to a signed formula (price - median without ABS) in a follow-up on the same topic — that flips the sign and looks like a contradiction of your own earlier answer for the same item."""

# ── Worked examples (teach the agent good patterns) ─────────────────────────
EXAMPLES = """WORKED EXAMPLES:
Q: Overall / total revenue and margin (the whole company).
SQL: SELECT SUM(revenue) AS revenue, SUM(revenue-cost) AS margin, SUM(revenue-cost)/SUM(revenue)*100 AS margin_pct FROM sales_totals;
-- (canonical total = ₹521.67 Cr. Do NOT use sales_by_manufacturer for this — it undercounts.)

Q: Top manufacturers by revenue and margin.
SQL: SELECT manufacturer, revenue, revenue-cost AS margin, (revenue-cost)/revenue*100 AS margin_pct
     FROM sales_by_manufacturer ORDER BY revenue DESC LIMIT 10;

Q: Which manufacturers have stock expiring within 90 days, by value?
SQL: SELECT m.manufacturer_desc AS manufacturer, sum(i.total_cost) AS expiring_value, count(DISTINCT i.material) AS items
     FROM fact_inventory i JOIN dim_material m ON i.material=m.material
     WHERE i.qty>0 AND i.expiry_date IS NOT NULL
       AND date_diff('day', DATE '2026-05-31', i.expiry_date) BETWEEN 0 AND 90
     GROUP BY 1 ORDER BY expiring_value DESC LIMIT 10;

Q: For items expiring in 90 days, did we overpay vs the median purchase price? / Which items have the biggest price deviation impact? / Biggest price increases?
Any of these needs the outlier-clean two-pass pattern (gotcha #15) — a first-pass median to find each material's normal price, then drop rows that stray outside an 8x band (data-entry/UOM errors), THEN compute the real metric from the clean rows only:
SQL: WITH raw AS (SELECT * FROM fact_grn WHERE net_price>=10 AND gr_qty>0),
     pass1 AS (SELECT material, median(net_price) AS m1 FROM raw GROUP BY 1),
     clean AS (SELECT r.* FROM raw r JOIN pass1 p USING(material) WHERE r.net_price BETWEEN p.m1/8.0 AND p.m1*8.0),
     med AS (SELECT material, median(net_price) AS med_price, count(*) AS n FROM clean GROUP BY 1 HAVING count(*)>=5)
     SELECT g.material, any_value(g.material_desc) AS item, sum(abs(g.net_price-med.med_price)*g.gr_qty) AS price_deviation_impact
     FROM clean g JOIN med USING(material) GROUP BY 1 ORDER BY price_deviation_impact DESC LIMIT 10;
-- For "did we overpay" specifically (one-directional), swap the final SUM to sum(greatest(g.net_price-med.med_price,0)*g.gr_qty) — still built on the same outlier-clean `clean`/`med` CTEs, never on raw fact_grn directly.

Q: Portfolio days of cover.
SQL: SELECT median(doh_days) AS median_doh FROM kpi_doh WHERE doh_days>0;

Q: Top vendors by spend (dedup across plants).
SQL: SELECT vendor_name, sum(vendor_value) AS spend FROM kpi_vendor_volume GROUP BY 1 ORDER BY spend DESC LIMIT 10;

Q: "Does <item X> have any sales/purchase activity?" and sales_by_material + fact_po + fact_grn all return 0 rows.
Before answering "no data", ALSO check physical inventory (don't stop at sales/purchase):
SQL: SELECT plant, qty, total_cost, aging_days, expiry_date, formulary FROM fact_inventory WHERE material = '<code>' AND qty > 0;
If this returns rows: it's NOT a data gap — it's dead/non-moving stock. Report the qty, how long it's been aging, whether expiry_date is already past the 2026-05-31 snapshot, and the formulary flag (OUT OF FORMULARY commonly explains why it stopped moving). Only say "no data at all" if inventory is ALSO empty."""


def context() -> str:
    return (
        "LIVE SCHEMA (view [rows]: col:type):\n" + warehouse.schema_text()
        + "\n\nDIMENSIONAL MODEL — for each table, the dimensions you may slice by and whether it has a time axis. "
          "A figure is only 'about' a dimension if the query GROUPed/filtered a table that carries it. NEVER attribute a "
          "table's numbers to a dimension not in its 'slice by' list, and never show a time trend from a table whose time axis is NONE:\n"
        + warehouse.grain_text()
        + "\n\n" + RULES + "\n\n" + GUIDE + "\n\n" + GOTCHAS + "\n\n" + EXAMPLES
    )
