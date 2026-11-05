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
After a new stock count, restart the API or call `POST /admin/refresh-data`: the stock figures and
`GET /meta/windows` (the months and the count date every page label reads) stay cached until then.
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
After any change to the formulary build's code, run `python -m app.etl.formulary --from-cache`
too (seconds): it rebuilds every cached month and also publishes `sales_items.parquet` (the
Revenue & margin page and the sales export). A build that keeps months from the published tables
(`--months`, or `--from-cache` when the line cache has lost a month) stops before it reads any
workbook, a dry run too, when a table lacks one of those months (a table never published, or
`sales_items` restored from an older copy). It names the command that rebuilds every month:
`--from-cache` when the line cache holds them all, else the plain build, which reads every
month's workbook again (after putting back any workbook that has left the sales folder).

### Out of formulary, price variation and generic vs branded (the purchase build)
The out-of-formulary (OOF) margin follows HCG's purchase (GRN) method, the way HCG's own OOF
workbooks and the KR spend email are built. The earlier sales-based estimate (the spend build's
`oof_switch` table, which matched each OOF brand on the bills to a formulary brand) is retired:
`python -m app.etl.spend` no longer builds it, and `app/etl/spend/oof.py` keeps only the
description parsers (molecule, strength, form, kind) that the purchase build shares.

`python -m app.etl.purchase` (`app/etl/purchase/`):
- Reads every `.xlsx` and `.xlsb` workbook in the GRN folder ('GRN Dec 25 to May 2026'), each
  assigned to its month by its GR dates (two files covering one month stop the build); any other
  file there is named as "not read" in the log and the report. Dates are read day-first; an
  unreadable date is counted in the report, never guessed. No GR, PO or vendor invoice number is
  kept; a per-file cache sits in `~/.cache/hcg-analytics/purchase` (never committed).
- Scope, as in HCG's OOF workings: movement type 101 (receipts) only, material series 1 and 2
  (drugs), without storage location S125 and plants BH* (Milann), HM03 and HM04. The exclusions
  are `app/data/reference/purchase_scope_exclusions.csv` (kind, value, reason): edit it and
  rebuild, no code change.
- Out of formulary: a GRN line counts when that month's Item Master flags its material OUT OF
  FORMULARY (not the GRN's own 'Formulary' column), so OOF months are those with both a GRN and
  an Item Master (Jun, Jul and Aug-26 today). A GRN month from Jun-26 on without its Item Master
  waits, with a warning (next section); the GRN months before Jun-26 are history for prices and
  generic only.
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

### When HCG sends a new month (e.g. September)
Nothing in the code names a month: each build takes the months from the files. Put the new files
in place, run the builds in this order, and check what each one prints.

**Files** (under `master/<hash>/`; the dashboard folder is 'Formulary Compliance and Spend
Analysis Dashboard data'):

| File | Folder | Needed for |
|---|---|---|
| IP & OP sales, e.g. `IP & OP Consolidated File September 26.xlsx` | dashboard folder `/IP and OP sales/` | formulary, spend, Revenue & margin, sales export |
| Item Master, e.g. `Item master for formulary complience Sep 26.xlsx` | dashboard folder `/Item Master Sheet/` | formulary (it stops without it) and out of formulary (it waits for it) |
| GRN, e.g. `06.Consolidated GRN Sep 2026.xlsx` | `GRN Dec 25 to May 2026/`: the only folder under `master/` whose name starts with `GRN ` in any case (a second one, or none, stops the build; the rest of the name does not matter) | prices, generic, out of formulary; `run_etl` |
| HCG's out-of-formulary working, sheets `Summary` and `Out of formulary Sep 26 WS` (or `WS Final Sheet`) | `Out of Formulary Spend Analysis @6-27/` | optional: without it the month is Estimated (below) |
| HCG's KR email figures | a row in `app/data/reference/oof_email_figures.csv` | only with the working (the guard checks KR against it) |
| HCG's MIS, `Onco and Non Onco Formulary Compliance Sep 26 - MIS.xlsx` | dashboard folder `/HCG MIS/` | optional: `python -m app.etl.formulary.hcg_mis` |
| Consumption and PO extracts | their own folders | `run_etl` |

The OOF master, the Spend Analysis Main Sheet and the Payer Master stay as they are (the FY
2026-27 lists cover up to Mar-27; April 2027 needs HCG's FY 2027-28 lists, and the build says so).
If HCG sends an updated one, replace the file. A new hospital code needs a row in
`app/data/reference/hospital_map.csv` (the formulary build stops and names it).

Naming rules (tested on the real readers):
- The month is read from the file name for sales, Item Master and MIS files: 'Sep 26', 'Sep26',
  'Sept-26', 'September 2026' and 'September 2026_Updated' all read as 2026-09. "SEP'26",
  '30.09.2026', a month without its year ('September') and two months ('Aug-26 to Sep-26') read
  as no month: the build stops and names the file. A GRN file's month comes from its GR dates and
  an OOF working's from its worksheet, never from the name.
- One file of each kind per month and folder: replace, don't add copies. 'Item Master Sep 26
  (2).xlsx' also reads as Sep, and two files for one month stop the build. Only one
  `*Spend Analysis*` workbook may sit in the dashboard folder (don't drop HCG's own "KR Spend
  Analysis" report there).
- `.xlsx` or `.xlsb` only. An `.xls`, `.csv` or any other file in a source folder is not read:
  the formulary, spend, purchase and MIS builds name it with the reason ("... is not read: an
  Excel 97-2003 workbook (.xls) ...: open it and save it as .xlsx") in their log and in their
  validation report's `warnings`, then carry on without it. (A report published before the
  `warnings` key was added gets it at its build's next run.) Lock, temp and system files
  (`~$...`, `.~lock...#`, `~WRL0001.tmp`, `*.tmp`, `Thumbs.db`, `.DS_Store`) are ignored.
- Files sit directly in their folder. A sub-folder of the sales, Item Master, GRN, OOF master or
  HCG MIS folder, or any sub-folder of the OOF folder except the OOF master's, is named the same
  way ("... is not read: a sub-folder (the build reads only the files directly in ...): move its
  files up ...") and nothing inside it is read.
- HCG's out-of-formulary working needs a sheet named `Summary` and one named `Out of formulary
  <Month> 26 WS` or `WS Final Sheet`; names like "OOF Sep 26 WS", "Summary Sep" or "... Working"
  stop the build.

**Commands, in order** (from this folder; the formulary, spend and purchase builds take `--list`
and `--dry-run`):
1. `python -m app.etl.formulary --list`. Check: September shows its sales file and its Item
   Master, and no "is not read" note. Then `python -m app.etl.formulary --months 2026-09`: the only
   step that opens the big IP/OP sales workbook (about 3 minutes, heavy on memory). Check: the
   KR line against HCG's MIS, and `published to ... sales_items`. If it stops because a table
   lacks a month it keeps (e.g. a table never published), run the command it names (usually
   `python -m app.etl.formulary --from-cache`, seconds), then this step again. When only the
   formulary code changed, `--from-cache` is all that is needed.
2. `python -m app.etl.spend --dry-run`, then `python -m app.etl.spend` (seconds; it reads the
   formulary cache, not the workbook). Check: the KR scheme and cash & TPA lines for September.
3. `python -m app.etl.purchase --list`. Check: the September GRN file and Item Master are listed
   and there is no "not read:" line (a "waiting:" line appears once the new GRN has been read,
   e.g. after the dry run). Then `python -m app.etl.purchase --dry-run` (about a minute to read the
   new GRN once; report in `~/.cache/hcg-analytics/purchase/`) and `python -m app.etl.purchase`.
   Check: `[purchase] 2026-09: OOF ... lines, loss ...`; with HCG's working, `HCG's Summary ...
   (ours - HCG 0.00, worst centre 0.00)`; and every `[purchase] warning:` line. It reads no sales
   file and does not need the formulary build (it shares the formulary build's file discovery,
   so a sales file whose name has no month stops it too).
4. `python -m app.etl.formulary.hcg_mis`, only when a newer MIS file came.
5. `python -m app.etl.run_etl` (consumption, inventory, GRN and PO; about 2 minutes), then restart
   the API or call `POST /admin/refresh-data`. The formulary, spend and purchase tables are picked
   up without a restart.
6. `pytest tests/test_purchase_golden.py tests/test_spend_golden.py tests/test_formulary_golden.py`:
   HCG's Aug-26 figures still hold (no test pins September).

**When HCG's out-of-formulary working for the month is missing**, the purchase build publishes the
month from sources 2-5 (HCG's pairs from its other months, its out-of-formulary list, then the
app's own match) and marks it as estimated: the log says `[purchase] warning: Estimated: HCG's
working for Sep-26 was not received; ...`, the month's report in `purchase_validation.json` has
`hcg_working: false`, `estimated: true` and that warning, `/spend/oof/overview` gives
`estimated: true`, `hcg_working: false` and the note, and every out-of-formulary tile shows the
Estimated badge and "Estimated month". Tried on Aug-26 without HCG's working, this came out 0.72%
below HCG's own figure. If `oof_email_figures.csv` already has a September row, the build stops
instead (the working went missing): restore it, or run `python -m app.etl.purchase
--without-hcg-sheet 2026-09`. When the working arrives later, drop it in the folder, add the email
row and run `python -m app.etl.purchase` again: September becomes HCG's own, checked centre by
centre by the golden guard.

**When the month's Item Master is missing**, the formulary build stops ("no Item Master for
2026-09", naming the file if it came as an `.xls` or `.csv`), so formulary and spend wait for it.
The purchase build does not stop: it warns `Out of formulary for Sep-26 waits for its Item Master;
prices and generic are built`, publishes September's price variation and generic vs branded, and
leaves out of formulary at August. `/spend/meta` lists the month in `oof_waiting`, the
out-of-formulary overview's notes say the same, asking it for September gives a 400 with that
sentence, and its trend tile says "Sep-26 waits for its Item Master". When the Item Master
arrives, put it in the Item Master folder and run steps 1 to 3 again (the full purchase build, not
`--months`).

That is for a month whose out-of-formulary rows have never been published. When a month already
published loses its Item Master from the folder (deleted, renamed, moved into a sub-folder), the
purchase build stops before it builds anything, a dry run too: `2026-07: out of formulary for
Jul-26 is published, built on the Item Master '...', which is no longer in the Item Master
folder; put it back ...`, and `--list` shows the month under `missing:`, not `waiting:`.
Otherwise the month would keep its published rows while its report said it waits, and its prices
and generic would be rebuilt on another month's master. Put the file back, or rebuild only the
other months with `--months`.

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
It refuses to write if a file reads zero rows, if a file's SALESDATE cannot be read on more than
half its rows, or if any material code is still float-shaped.
It keeps the billing months `LEGACY_SALES_WINDOW` (`2025-12..2026-05`, `app/etl/ingest_sales.py`)
on purpose: later months come from the HIS sales build (`sales_items`). `SALES_MONTHS=YYYY-MM..YYYY-MM`
moves that window with no code change; turnover and days on hand then scale the billed part over
the months the tables hold. The run log counts the rows it leaves out, each for what it is: dated
before the window, dated after it (served by `sales_items`), and with no readable SALESDATE (not
counted anywhere).
`kpi_billable_consumption.parquet` has no generator in this repo; it is already clean.

## 3. Run the API
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Voice input in the AI chat

The AI chat has a mic button. The browser records the question (up to a minute), the API sends
the clip to an Azure OpenAI speech-to-text deployment (POST /ai/transcribe, login required,
nothing stored), and the text appears in the chat box for the user to check and send. Turn it on:

1. In the Azure OpenAI resource the chat uses, deploy **gpt-4o-mini-transcribe** (Deployments →
   Deploy model). If that resource's region does not offer it, deploy it on a resource in a
   region that does and set `AZURE_OPENAI_TRANSCRIBE_ENDPOINT` and `AZURE_OPENAI_TRANSCRIBE_API_KEY`
   too (the key as a secret).
2. On the Container App, add `AZURE_OPENAI_TRANSCRIBE_DEPLOYMENT` = the deployment's name.
3. Check: `python -m app.ai.check_connection` prints a `voice ... OK` line. GET /ai/status then
   says `"voice": true` and the mic appears in the chat.

### Keep AI chats and users across deploys

Users and AI Analyst chat sessions live in the database named by `DATABASE_URL`. Its default
is a SQLite file, `app/data/app.db`, which is right on a laptop but NOT in production: the
Azure Container App's disk is thrown away on every deploy (each push to `main`), restart and
scale event, so every chat and every signed-up user except the seeded admin disappears, and a
second replica would keep its own separate file. The API logs a warning at start when it runs
on SQLite inside a container.

Point production at a server database once (no code change; the driver `psycopg` is installed):

1. Create an Azure Database for PostgreSQL – Flexible Server in the backend's resource group
   and region (Burstable B1ms is enough), a database (e.g. `hcg`), and allow the Container
   App to reach it (public access with "Allow Azure services", or the same virtual network).
2. In the Container App `hcg-backend` → Secrets, add `database-url` with the value
   `postgresql+psycopg://<user>:<password>@<server>.postgres.database.azure.com:5432/hcg?sslmode=require`.
3. In Containers → Environment variables, add `DATABASE_URL` referencing that secret, and save
   (a new revision starts).
4. On first start the API creates its tables and re-seeds the admin account. Users sign up
   again once; from then on chats and users survive every deploy and restart.

Chats written before the switch were on the old container disk and cannot be recovered.

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
  · `GET /spend/prices/overview|molecules|low` · `GET /spend/generic/overview` (purchase data: the cheapest
  price comes from Plant/Month/Segment, `Vendor=` and `Material=` narrow whose purchases are set against
  it; every price view carries the rule's thresholds as `rule`) · `GET /spend/validation` (admin)
  · `GET /spend/prices/low` is the whole "to check" list behind the overview's `low_prices`: one row per
  hospital × brand × supplier holding a purchase under the low ratio of the usual cheapest, furthest
  under first, with sums (qty, value, below_cheapest, lines) and facets (hospital, product, supplier)
- `GET /revenue/overview|materials` (HIS pharmacy sales at MRP, cost and margin; each `by_hospital` row
  carries `buckets`, the unit split by formulary bucket with IP and OP beside, adding up to the row)
  · `GET /executive/overview` (one screen; every figure comes from its own section's handler)
- `GET /export/meta|sales|grn` (the sales lines and the GRN lines for download; no patient, bill, GR or
  PO number)
- Paging on every `{data,total}` list: `page` from 0 and `page_size` from 1 (default 25); anything else
  is a 400, never a silently shorter or longer page. The lists the Export CSV pages through take at most
  **5,000** rows a page (`table_query.EXPORT_MAX_PAGE`, the UI's `EXPORT_PAGE` and the AI's `ALL_ROWS`:
  change them together): `/export/sales|grn`, `/revenue/materials`, `/spend/prices/molecules|low`, and
  the formulary, brand-rule and spend item, doctor, vendor, consumption and rule-book lists. Every other
  table (`/kpi/{key}/table`, the legacy and consumables tables) takes up to **100,000**
  (`MAX_PAGE_SIZE`). Page through a longer list.
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
- Known month-bound pieces (documented, not yet read from the data):
  - `app/forecast/engine.py`: `HIST_ORDER` fixes the history to Dec-25..May-26, the forecast months
    start at Jun-26 and `build_backtest` holds out Apr and May. A new stock count leaves the
    forecasts on those months until the engine takes its history from the data. The forecast pages
    count the months of the forecast's own timeline, so their words stay true either way.
  - `kpi_stock_out.parquet` and `kpi_billable_consumption.parquet` are static Dec-25..May-26 tables
    that no build here refreshes. A new stock count or billing window does not move them, while the
    Stock-out page's month count follows `GET /meta/windows`: rebuild them before trusting those
    two pages after a new count.
