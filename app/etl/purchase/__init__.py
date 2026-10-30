"""Purchase analysis build: HCG's GRN (goods receipt) extracts and out-of-formulary workbooks
-> small aggregated tables in PURCHASE_DATA_DIR that the API serves.

Every purchase analysis counts the receipts of HCG's out-of-formulary workings (movement type
101, drug series 1 and 2, without storage location S125 and the BH*, HM03, HM04 plants), as
listed in app/data/reference/purchase_scope_exclusions.csv. Run by hand:
`python -m app.etl.purchase --help`. GRN workbooks are read once into PURCHASE_CACHE_DIR.

  config.py    paths, folder / sheet / column names, the scope file, table names
  grn.py       the one GRN reader (also used by app/etl/ingest.py) and the scope funnel
  masters.py   HCG's OOF master, HCG's monthly OOF workbooks (worksheet suggestions, parked
               lines, per-centre Summary), the month's Item Master, the Main Sheet CASH&TPA list
  molecule.py  product keys (molecule | strength | form | kind) for comparing makers
  oof.py       HCG's out-of-formulary method -> oof_purchase + HCG's own tables, golden guard
  prices.py    price variation -> purchase_prices
  generic.py   generic vs branded (originator list + overrides) -> purchase_generic
  build.py     orchestration (TABLE_BUILDERS), validation report, atomic publish, CLI
"""
