# HCG Supply Chain Analytics — Backend Runbook

Parquet-backed FastAPI serving all 28 buildable KPIs from the raw HCG SAP extracts.
No Azure SQL, no chat-AI. See `../KPI_Formula_Workbook.xlsx` for KPI definitions and
`../.../plans/ok-so-you-have-...md` for the full plan.

## 1. Install
```bash
pip install -r requirements.txt        # fastapi, uvicorn, pandas, pyarrow, openpyxl, numpy
```

## 2. Build the data (ETL → curated parquet → KPI aggregates → forecasts)
```bash
# Raw data location defaults to master/ in this checkout (the folder FORMULARY_RAW_DIR uses);
# override with RAW_DATA_DIR.
python -m app.etl.run_etl                # full rebuild (~2 min: about 21 workbooks; the GRN through
                                         # the purchase build's cache in ~/.cache/hcg-analytics/purchase)
python -m app.etl.run_etl --stage curated  # only facts + dimensions
```
Outputs:
- `app/data/curated/*.parquet` — fact_consumption, fact_inventory, fact_grn, fact_po, dim_* (+ `_dq_report.json`,
  which counts per source and month what was read and what was left out)
- `app/data/kpi/*.parquet` — one aggregate per KPI + forecast tables + `forecast_accuracy`

What the import keeps (`app/etl/ingest.py`): GRN receipts only (movement 101; returns are counted,
not kept), PO lines without the deletion flag, with a material and no negative quantity (deleted,
service/no-material and return lines are counted in `_dq_report.json`), Net Price per unit (divided
by Price Unit), and one consumption source per month (the consolidated Dec-25..May-26 extract,
then the monthly files; issues stored as positive quantities). Dates are read day-first. No GR or
PO number is kept: `fact_grn` has no order or receipt number and `fact_po.po_no` is an opaque
per-order key. Stock measures (days on hand, turnover, non-moving, risk) set the 31-May-26 stock
count against the consumption up to that day (`transforms.consumption_until`), live API included.
The capex-tab check reads raw Excel and is opt-in:
`HCG_RAW_TESTS=1 pytest tests/test_source_completeness.py -k capex`.

Formulary, spend and purchases (P1), from HCG's files in `master/` (line-level, never committed), in this order:
```bash
python -m app.etl.formulary   # sales lines -> ~/.cache/hcg-analytics/formulary + app/data/formulary/ (~3 min/month)
python -m app.etl.spend       # that cache + the Main Sheet's SCHEME and CASH&TPA lists -> app/data/spend/
python -m app.etl.purchase    # GRN receipts + Item Masters + HCG's OOF master and workbooks + CASH&TPA prices -> app/data/purchase/
```
All three take `--list` and `--dry-run`. Every figure is computed from these files; nothing typed
from HCG's emails is published. Each build prints KR's figures per month so they can be set
beside HCG's email, and `pytest tests/test_spend_golden.py tests/test_formulary_golden.py`
checks KR Aug-26 against HCG's own numbers (the one place they are written down). Which cash brands HCG bifurcates is `app/data/reference/cash_tpa_bifurcation.csv`
(seeded from HCG's KR Aug-26 report; replace it when HCG sends the full rules), and the cash
brands HCG's report leaves out are `app/data/reference/cash_tpa_excluded.csv` (same source).
HCG's brand rules (Task 4: which Item Master letters are wrong for which payers) are
`app/data/reference/brand_rules.csv`; after editing it, rebuild with
`python -m app.etl.formulary --from-cache` (no Excel re-read), then `python -m app.etl.spend`.

### Out of formulary, price variation and generic vs branded (the purchase build)
The out-of-formulary (OOF) margin follows HCG's purchase (GRN) method, the way HCG's own OOF
workbooks and the KR spend email are built. The earlier sales-based estimate (the spend build's
`oof_switch` table, which matched each OOF brand on the bills to a formulary brand) is retired:
`python -m app.etl.spend` no longer builds it, and `app/etl/spend/oof.py` keeps only the
description parsers (molecule, strength, form, kind) that the purchase build shares.

`python -m app.etl.purchase` (`app/etl/purchase/`):
- Reads every `*.xlsx` in the GRN folder ('GRN Dec 25 to May 2026'), each assigned to its month
  by its GR dates (two files covering one month stop the build). Dates are read day-first; an
  unreadable date is counted in the report, never guessed. No GR, PO or vendor invoice number is
  kept; a per-file cache sits in `~/.cache/hcg-analytics/purchase` (never committed).
- Scope, as in HCG's OOF workings: movement type 101 (receipts) only, material series 1 and 2
  (drugs), without storage location S125 and plants BH* (Milann), HM03 and HM04. The exclusions
  are `app/data/reference/purchase_scope_exclusions.csv` (kind, value, reason): edit it and
  rebuild, no code change.
- Out of formulary: a GRN line counts when that month's Item Master flags its material OUT OF
  FORMULARY (not the GRN's own 'Formulary' column), so OOF months are those with both a GRN and
  an Item Master (Jun, Jul and Aug-26 today; a new month is picked up when its GRN arrives).
  Actual margin = GR qty x (MRP excl GST - Base Price); suggested margin = GR qty x (suggested
  MRP excl GST - suggested HCG Pharma price); margin loss = suggested - actual summed over every
  line (a line whose bought brand earns more lowers it, so a hospital's loss can be negative);
  % = loss / suggested margin.
- The suggested brand, in this order, each line labelled with its source: HCG's OOF worksheet
  of the same month (`hcg_sheet`; reproduces HCG's Summary to the paisa), HCG's pair for that
  brand from its nearest worksheet month (`hcg_sheet_pair`, for months without a worksheet),
  HCG's OOF master (`hcg_list`), the app's own rule (`formulary_rule`: same molecule, strength,
  form and kind among the CASH&TPA sheet's formulary brands; shown as "Estimated"), else `none`
  ("No alternative mapped": its spend counts, it adds nothing to the margins).
- Price variation (unit price after discount = 'Total Amount without tax' / GR qty, compared
  within a molecule, strength and form) and generic vs branded (originator makers in
  `app/data/reference/originator_manufacturers.csv`, per-material corrections in
  `generic_overrides.csv`; an estimated definition until HCG confirms it) cover every GRN month
  in scope.
- Each OOF line has a status: `counted` (in the margins and the loss), `parked` (HCG's 'Need To
  check flaging later' lines: shown, not counted), `no_alternative` (no suggested brand) or
  `unpriced` (a suggested brand no price list prices: shown, not counted).
- Publishes to `app/data/purchase/` (`PURCHASE_DATA_DIR`): `oof_purchase`, HCG's own tables
  `oof_hcg_summary` (per-centre Summary figures, so the app shows HCG's figure beside ours),
  `oof_hcg_pairs` (its worksheet suggestions) and `oof_hcg_parked`, then `purchase_prices`,
  `purchase_generic` and `purchase_validation.json` (rows at each filter step, unreadable dates,
  blank materials, our loss vs HCG's per centre, KR vs the email, the "month without HCG's
  worksheet" simulation).
- Golden guard: for every month with HCG's worksheet, every centre must be within Rs 1 of HCG's
  Summary and KR within Rs 1 of HCG's emailed figures (`app/data/reference/oof_email_figures.csv`:
  add a row when a new email arrives), or nothing is published. A month the email has figures for
  whose HCG workbook was not read stops the build too (the workbook went missing); to publish it
  from sources 2-5 anyway, run `python -m app.etl.purchase --without-hcg-sheet 2026-09`.
  `tests/test_purchase_*.py` hold the golden checks.
- `--months 2026-08` rebuilds those months and keeps the others' rows (HCG's own tables keep a
  month whose workbook was not read); the validation file's build-wide sections always describe
  every month published.
- The API's `/spend/oof/*` views, the Brand filter's out-of-formulary side and the cash overview's
  combined "cash & TPA + out of formulary" line read `oof_purchase`.

### Sales aggregates and the split material codes
The sales tables in `app/data/kpi/sales_*.parquet` come from `app/etl/ingest_sales.py`, which
is not part of `run_etl` and streams ~1.2 GB of patient-level IP/OP billing files. Before
it used `norm_code`, the four `.xlsb` OP months (Dec-25 to Mar-26) wrote every material code
as a float (`101313.0`), so the four material-keyed tables hold most products twice and the
`.0` half has no category (KEYTRUDA shows as Rs 45.77 Cr + Rs 1.71 Cr; its December as
Rs 9.29 Cr instead of Rs 9.68 Cr). Repair the existing parquets instead of re-running the ETL:
```bash
python -m app.etl.repair_sales_codes          # dry run: prints the plan, writes nothing
python -m app.etl.repair_sales_codes --write  # backs up, then rewrites the 4 tables
```
- Expected dry run: `sales_by_material` 21,960 -> 15,171 rows, `sales_by_material_month`
  70,569 -> 56,502, `sales_by_material_hospital` 57,455 -> 41,335, `sales_by_material_mfr`
  21,678 -> 14,968. Revenue, cost, qty and lines do not change (Rs 521.67 Cr total).
- It stops with nothing written if any table would lose a rupee, keep a `.0` code or end up
  with a duplicate key. `--write` copies the originals to
  `~/Documents/HCG/hcg-work/snapshots/sales-parquet-<timestamp>/` first (override with
  `SALES_REPAIR_BACKUP`); to undo, copy them back.
- Afterwards: restart the API (or `POST /admin/refresh-data`), then commit the four parquets.
  The repair is a one-off and safe to run again (a clean table stays the same).
- Numbers that move on the dashboards (all now correct): `/revenue/items` lists 15,171 items
  and KEYTRUDA once at Rs 47.48 Cr; category revenue and billed COGS go up (Onco Rs 297.51 ->
  311.11 Cr; Unclassified Rs 52.62 -> 23.68 Cr); billed days on hand (DOH) goes down; fewer items
  show as non-moving. Grand totals do not change.

To rebuild the sales tables from the raw files instead (needs a few GB of free RAM), point
`SALES_RAW` at the folder that holds `IP Sales` and `OP Sales`:
```bash
SALES_RAW=/path/to/master/<folder> python -m app.etl.ingest_sales
```
It refuses to write if a file reads zero rows or if any material code is still float-shaped.
`kpi_billable_consumption.parquet` has no generator in this repo; it is already clean.

## 3. Run the API
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 4. Key endpoints
- `GET /health`
- `GET /meta/plants` · `GET /meta/material-groups` · `GET /meta/vendors?Plant=` · `GET /meta/kpis`
- `GET /kpi/{key}?Plant=&Material=&MaterialGroup=` — chart data (23 keys; see `/meta/kpis`)
- `GET /kpi/{key}/table?Plant=&page=&page_size=&sort_field=&sort_order=&filter_field_0=&filter_value_0=&global_filter=` — `{data,total}`
- `GET /forecast/sales-forecast?Plant=&Material=` · `GET /forecast/cashflow-forecast` · `GET /forecast/accuracy`
- `GET /inventory/replenishment-data?plant=&material_id=`
- `GET /api/dashboard/all?region={plant}` — Executive Summary cards
- `GET /formulary/meta|overview|doctors|items|scheme/items` · `GET /formulary/validation` (admin)
- `GET /formulary/rules/overview|items|doctors|book` (brand rules per payer, P1/P2 rule book)
- `GET /spend/meta` · `GET /spend/attis/overview|items|doctors|consumption` · `GET /spend/scheme/overview|items|doctors`
  · `GET /spend/cash/overview|items|doctors` · `GET /spend/oof/overview|items|vendors` (purchase data;
  `/spend/oof/doctors` is removed, purchases carry no doctor; `/spend/oof/items` takes `Vendor=`;
  `Type` and `Payer` are accepted and reported as not applied)
  · `GET /spend/prices/overview|molecules` · `GET /spend/generic/overview` (purchase data: the cheapest
  price comes from Plant/Month/Segment, `Vendor=` and `Material=` narrow whose purchases are set against
  it) · `GET /spend/validation` (admin)
- `GET /portfolio/consumption/consumables/overview|departments|items` (3-series consumables per SAP cost centre)
- `POST /admin/refresh-data` (bearer token + header `x-admin-token`) — clears the parquet cache

## 5. KPI key → table map
See `app/api/kpi_generic.py:REGISTRY` (28 KPIs across inventory / procurement / consumption / forecasting / additional).

## Notes
- `Plant` param is the **plant code** (e.g. `HC05`); `ALL` = all plants. Names via `/meta/plants`.
- Forecast accuracy: aggregate ~75%, volume-weighted ~48% (6-month history; see workbook gap log).
- Every route except `/health`, `/signin` and `/signup` needs `Authorization: Bearer <token>`
  (the token comes from `POST /signin`; enforced in `app/api/api_router.py`). Production must set
  `JWT_SECRET_KEY`, `SEED_ADMIN_PASSWORD` and `ADMIN_REFRESH_TOKEN`; browser origins are limited by
  `CORS_ORIGINS` / `CORS_ORIGIN_REGEX` (`app/core/config.py`).
- Old CSV card endpoints (`home.py`, `kpi_cards.py`) are present but not mounted.
