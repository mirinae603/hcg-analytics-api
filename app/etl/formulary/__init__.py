"""Formulary compliance build (Task 1).

HCG's monthly IP+OP pharmacy sales workbooks, Item Masters and Payer Master (under
FORMULARY_RAW_DIR, line-level patient data, never committed) -> small aggregated tables in
FORMULARY_DATA_DIR that the API serves. Run by hand: `python -m app.etl.formulary --help`.

  discover.py   find the files, pair each sales month with its own Item Master
  excel.py      streaming .xlsx/.xlsb readers and cell normalisers
  extract.py    one month's sales sheet -> FORMULARY_CACHE_DIR/lines_YYYY-MM.parquet
  masters.py    Item Master, Payer Master, hospital_map.csv, payer_overrides.csv
  classify.py   the rules (pure functions; also what the tests pin down)
  aggregate.py  DuckDB: classify + sum the cached lines into the four tables
  validate.py   formulary_validation.json and the abort rules
  build.py      orchestration, atomic publish, CLI
"""
