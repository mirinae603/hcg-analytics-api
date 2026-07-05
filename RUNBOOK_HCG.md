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
- `POST /admin/refresh-data` (header `x-admin-token`) — clears the parquet cache

## 5. KPI key → table map
See `app/api/kpi_generic.py:REGISTRY` (28 KPIs across inventory / procurement / consumption / forecasting / additional).

## Notes
- `Plant` param is the **plant code** (e.g. `HC05`); `ALL` = all plants. Names via `/meta/plants`.
- Forecast accuracy: aggregate ~75%, volume-weighted ~48% (6-month history; see workbook gap log).
- Auth (`authenticate.py`) and old CSV card endpoints (`home.py`, `kpi_cards.py`) are present but
  not mounted — re-add to `api_router.py` if needed.
