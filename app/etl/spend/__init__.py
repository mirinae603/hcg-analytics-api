"""Spend analysis build (Task 2: Attis; the same scheme engine serves Task 3b later).

HCG's Spend Analysis Main Sheet (sheet SCHEME: for each brand, the cheaper brand HCG
suggests for government-scheme patients, with both FY price-list prices) is applied to the
line cache the formulary build keeps (FORMULARY_CACHE_DIR) -> small aggregated tables in
SPEND_DATA_DIR that the API serves. Run by hand AFTER python -m app.etl.formulary:
`python -m app.etl.spend --help`. Reads no sales workbook itself: seconds, not minutes.

  config.py     paths, SCHEME headers, HCG's KR Aug-26 figures the build must reproduce
  masters.py    the SCHEME sheet -> used brand -> suggested brand + prices
  aggregate.py  DuckDB: GOV lines x scheme list, and Attis brand consumption
  build.py      orchestration, validation, atomic publish, CLI
"""
