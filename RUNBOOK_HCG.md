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
# Raw data location defaults to ../Bidezy-2 ; override with RAW_DATA_DIR.
python -m app.etl.run_etl                # full rebuild (~2 min: reads 14 Excel files)
python -m app.etl.run_etl --stage curated  # only facts + dimensions
```
Outputs:
- `app/data/curated/*.parquet` — fact_consumption, fact_inventory, fact_grn, fact_po, dim_* (+ `_dq_report.json`)
- `app/data/kpi/*.parquet` — one aggregate per KPI + forecast tables + `forecast_accuracy`

Formulary & spend (P1), from HCG's files in `master/` (line-level, never committed), in this order:
```bash
python -m app.etl.formulary   # sales lines -> ~/.cache/hcg-analytics/formulary + app/data/formulary/ (~3 min/month)
python -m app.etl.spend       # that cache + the Main Sheet's SCHEME and CASH&TPA lists + each month's Item Master -> app/data/spend/
```
Both take `--list` and `--dry-run`. Every figure is computed from these files; nothing typed
from HCG's emails is published. Each build prints KR's figures per month so they can be set
beside HCG's email, and `pytest tests/test_spend_golden.py tests/test_formulary_golden.py`
checks KR Aug-26 against HCG's own numbers (the one place they are written down). Which cash brands HCG bifurcates is `app/data/reference/cash_tpa_bifurcation.csv`
(seeded from HCG's KR Aug-26 report; replace it when HCG sends the full rules), and the cash
brands HCG's report leaves out are `app/data/reference/cash_tpa_excluded.csv` (same source).
HCG's brand rules (Task 4: which Item Master letters are wrong for which payers) are
`app/data/reference/brand_rules.csv`; after editing it, rebuild with
`python -m app.etl.formulary --from-cache` (no Excel re-read), then `python -m app.etl.spend`.
The out-of-formulary margin (`oof_switch`) needs no reference file: HCG has not sent which
formulary brand replaces each out-of-formulary brand, so the build matches it from HCG's own
masters (same Item Master 'Minor Group', strength and form, priced in the CASH&TPA sheet, best
margin; `app/etl/spend/oof.py`). Every match is listed in the app's Out of formulary → Brands tab.

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
  · `GET /spend/cash/overview|items|doctors` · `GET /spend/oof/overview|items|doctors` · `GET /spend/validation` (admin)
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
