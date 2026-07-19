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
• Revenue & margin (billed pharmacy sales): use the sales_* tables. margin = revenue − cost. margin_pct = margin / revenue * 100. This is REAL billed IP+OP revenue with true margin (billed MRP − actual cost).
• Procurement margin is only a PROXY: on fact_grn rows where unit_mrp>0 AND net_price>0, mrp_value = gr_qty*unit_mrp, cost_value = gr_qty*net_price, margin_pct = (mrp_value−cost_value)/mrp_value. Call it "MRP-proxy margin", never billed margin.
• Purchase spend = fact_grn.total_amount_wo_tax (or gr_qty*net_price); PO value = fact_po.total_value_wo_tax.
• Days of cover (DOH) = kpi_doh.doh_days; report the MEDIAN over rows where doh_days>0 (the mean is skewed by overstock — never report the mean).
• Expiry: compute days-to-expiry as date_diff('day', DATE '2026-05-31', expiry_date) on fact_inventory (qty>0, expiry_date not null). Bands: Expired (<0), 0-30, 31-90, 91-180, 181-365, 365+. "Near-expiry / actionable" = Expired + everything ≤180 days.
• Manufacturer lives in dim_material.manufacturer_desc (join on material) and in sales_by_manufacturer / sales_by_material_mfr for revenue.
• Department: in THIS dataset there are NO readable department names — dim_costcenter.department_name and kpi_consumption_by_department.department_name are both just the cost-centre CODE (a number like '1080101005'). Group consumption by cost_ctr and present it as "Cost centre <code>"; do NOT imply a named department exists. Treat cost_ctr as a text label, never a numeric value.
• Category = material_group (codes like 'M065-INJECTIONS'); strip the 'M###-' prefix and Title-Case for display."""

# ── Which table for what + join keys ────────────────────────────────────────
GUIDE = """TABLE GUIDE (pick the right source):
REVENUE / MARGIN (billed sales, 6mo):
  sales_by_manufacturer(manufacturer, revenue, cost, qty)          — revenue/margin by manufacturer
  sales_by_hospital(hospital, revenue, cost, qty)                  — by billing hospital (23 hospital codes)
  sales_by_material(material, material_desc, material_group, revenue, cost, qty)     — by product (6-MONTH TOTAL, no month column)
  sales_by_material_mfr(manufacturer, material, material_desc, material_group, …)    — product×manufacturer (mfr drill)
  sales_by_material_hospital(hospital, material, material_desc, material_group, …)   — product×hospital (hospital drill)
  sales_monthly(patient['IP'|'OP'], month, revenue, cost, qty)     — monthly revenue trend & IP vs OP split (NOT per-material)
  sales_totals(patient, revenue, cost, qty)                        — grand totals by IP/OP
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
  dim_material(material → material_desc, material_group, manufacturer_desc, generic_name, major_group_desc)
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
8. SALES have NO per-material month dimension: sales_by_material is a 6-month TOTAL per product. A per-item SALES trend over time is NOT available — sales_monthly is only IP/OP aggregate. But a per-item PURCHASE trend over months IS available via kpi_monthly_purchase_value (or fact_po/fact_grn by month). If asked for an item's "sales trend", give its total sales + monthly PURCHASE trend, and say the monthly split is on the procurement side.
9. dim_material is the full CATALOG — many SKUs have ZERO transactions (no sales, no purchases). If an item returns no rows in the fact tables, DON'T call it a technical error: say plainly it has no recorded sales/purchases in this 6-month window, and offer its sibling SKUs (same generic_name or material_group that DO have activity, e.g. via dim_material) so the question isn't a dead end.
10. Column names are already clean — use material_desc / material_group (never a bare `desc` or `group`; those reserved words have been aliased away). Just write plain identifiers.
11. NAME LOOKUPS: a product's brand name (e.g. 'CALPOL', 'AUGMENTIN') lives in material_desc. To find a named item, filter material_desc ILIKE '%name%'. generic_name is the MOLECULE ('PARACETAMOL','TRAMADOL') — use it only to group products by active ingredient, NEVER to search for a brand. When the user quotes an item that returns nothing, try a looser material_desc ILIKE on the distinctive word (e.g. 'CALPOL') before concluding there's no data — the exact SKU string ('CALPOL-T TAB') may not exist even though the brand family does.
12. dim_material (24,931 SKUs) is the FULL multi-year item master; sales/PO/GRN/consumption only cover the 6-month analytics window, so most catalog SKUs (~57%) show zero rows there — that alone is not newsworthy. Before telling a user an item "has no data", ALWAYS also check fact_inventory (physical on-hand snapshot, which can hold batches received before the window even started) and fact_consumption. If it HAS inventory, that's the real story — report qty, aging_days, expiry_date (vs snapshot 2026-05-31 — flag if already expired) and formulary status (dim_material.formulary / fact_inventory.formulary: 'OUT OF FORMULARY' often explains why an item stopped moving). A catalog item with stock but no sales/purchase in-window is classic DEAD / NON-MOVING stock — say so plainly instead of a flat "no data"."""

# ── Worked examples (teach the agent good patterns) ─────────────────────────
EXAMPLES = """WORKED EXAMPLES:
Q: Top manufacturers by revenue and margin.
SQL: SELECT manufacturer, revenue, revenue-cost AS margin, (revenue-cost)/revenue*100 AS margin_pct
     FROM sales_by_manufacturer ORDER BY revenue DESC LIMIT 10;

Q: Which manufacturers have stock expiring within 90 days, by value?
SQL: SELECT m.manufacturer_desc AS manufacturer, sum(i.total_cost) AS expiring_value, count(DISTINCT i.material) AS items
     FROM fact_inventory i JOIN dim_material m ON i.material=m.material
     WHERE i.qty>0 AND i.expiry_date IS NOT NULL
       AND date_diff('day', DATE '2026-05-31', i.expiry_date) BETWEEN 0 AND 90
     GROUP BY 1 ORDER BY expiring_value DESC LIMIT 10;

Q: For items expiring in 90 days, did we overpay vs the median purchase price?
SQL: WITH med AS (SELECT material, median(net_price) med_price FROM fact_grn WHERE net_price>0 AND gr_qty>0 GROUP BY 1),
     exp AS (SELECT DISTINCT material FROM fact_inventory WHERE qty>0 AND date_diff('day', DATE '2026-05-31', expiry_date) BETWEEN 0 AND 90)
     SELECT g.material, any_value(g.material_desc) AS item, sum(greatest(g.net_price-med.med_price,0)*g.gr_qty) AS overpay
     FROM fact_grn g JOIN med USING(material) JOIN exp USING(material)
     WHERE g.net_price>0 AND g.gr_qty>0 GROUP BY 1 HAVING overpay>0 ORDER BY overpay DESC LIMIT 10;

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
        "LIVE SCHEMA (view [rows]: columns):\n" + warehouse.schema_text()
        + "\n\n" + RULES + "\n\n" + GUIDE + "\n\n" + GOTCHAS + "\n\n" + EXAMPLES
    )
