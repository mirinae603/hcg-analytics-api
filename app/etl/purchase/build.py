"""Purchase build: HCG's GRN extracts and out-of-formulary workbooks -> small committed tables.

Usage (the GRN workbooks are read once into PURCHASE_CACHE_DIR, about a minute; seconds after):
    python -m app.etl.purchase --list                  # files, months and cache state; no big reads
    python -m app.etl.purchase --dry-run               # build + validate, publish nothing
    python -m app.etl.purchase                         # build every GRN month and publish
    python -m app.etl.purchase --months 2026-08        # rebuild one month (others kept)
    python -m app.etl.purchase --without-hcg-sheet 2026-09   # publish a month HCG's email has
                                                       # figures for although its workbook is missing

Steps: discover the GRN workbooks (each one's month comes from its GR dates) -> read them
through the cache -> keep the purchase scope (purchase_scope_exclusions.csv) -> run every
table builder in TABLE_BUILDERS -> validate -> publish to PURCHASE_DATA_DIR.

A table builder takes the Inputs (read once: the scope's lines, the Item Masters, HCG's OOF
workbooks and master, the Main Sheet) and returns a Built: its tables with the months their
rows replace, a report section per month, its build-wide section (taken on the tables as
published, so a --months build describes every month kept as well) and its abort reasons.
TABLE_BUILDERS:
  oof       oof_purchase + HCG's own tables (oof.py). Its abort reasons are the golden guard:
            with HCG's own worksheet of a month, every centre must equal HCG's Summary and KR
            HCG's email within a rupee, or nothing is published; a month HCG's email has
            figures for whose workbook was not read aborts too (--without-hcg-sheet waives it).
  prices    purchase_prices, price variation over every scope line (prices.py)
  generic   purchase_generic, generic vs branded (generic.py; the originator list and the
            overrides are reference files)

Tables are written to temp files and os.replace'd, purchase_validation.json last. Rebuilding
some months replaces only those months' rows (HCG's own tables only the months whose workbook
was read); every committed table is aggregated and carries no GR, PO or invoice number
(checked before publishing).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Iterable, Optional

import pandas as pd

from app.etl.formulary import classify as K
from app.etl.formulary import config as FC
from app.etl.formulary.build import _check_columns
from app.etl.formulary.discover import BuildError
from app.etl.purchase import config as C
from app.etl.purchase import generic as Gn
from app.etl.purchase import grn as G
from app.etl.purchase import masters as M
from app.etl.purchase import oof as O
from app.etl.purchase import prices as P

# Purchase identifiers no committed table may carry (the sales ones are the formulary
# build's FORBIDDEN_COLUMNS, checked too).
IDENTIFIER_COLUMNS = frozenset({
    "GR_NO", "PO_NO", "PR_NO", "GR_DATE", "PO_DATE", "REFERENCE_DOCUMENT", "VENDOR_INV_NO",
    "SEC_INV_NO", "INVOICE_NO", "BATCH"})
# The GRN read report fields kept in purchase_validation.json (not the cache bookkeeping).
GRN_REPORT_FIELDS = ("file", "sheet", "rows", "lines", "blank_rows", "blank_material", "mvt",
                     "returns", "dates", "value", "qty")


def _log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
# Inputs and table builders
# --------------------------------------------------------------------------- #
@dataclass
class Inputs:
    """Everything a table builder may use, read once per build."""
    scope: C.Scope
    grn: G.GrnRead                  # the GRN workbooks of the built months, with their reports
    lines: pd.DataFrame             # the scope's lines of the built months (grn.read_grn_files columns)
    funnel: pd.DataFrame            # apply_scope's funnel of those months
    periods: list                   # the GRN months built
    item_masters: dict              # period -> masters.item_master_frame, every month with a master
    item_master_files: dict         # period -> its file name
    hcg: dict                       # period -> HcgWorkbook, every HCG OOF workbook
    master: M.OofMaster
    main: M.MainSheet
    email: dict                     # (period, plant) -> oof.EmailFigure
    reference: Optional[str] = None  # REFERENCE_DIR for builders that read reference files
    without_hcg_sheet: frozenset = frozenset()   # months published from sources 2-5 although
                                                 # HCG's email has figures for them
    log: Callable = _log


# A table as published after this build (this build's rows merged with the months kept), or
# None when it was never published: what a builder's build-wide section is taken on.
Published = Callable[[str], Optional[pd.DataFrame]]


def _no_section(_: Published) -> dict:
    return {}


@dataclass
class Built:
    """What one table builder returns."""
    tables: dict = field(default_factory=dict)      # table name -> rows of the built months
    months: dict = field(default_factory=dict)      # period -> its report section
    report: Callable[[Published], dict] = _no_section   # its build-wide report section
    problems: list = field(default_factory=list)    # abort reasons: nothing is published
    replaces: dict = field(default_factory=dict)    # table name -> the months its rows replace
                                                    # (default: every month built)


def build_oof(inputs: Inputs) -> Built:
    """oof_purchase and HCG's own tables (oof_hcg_summary, oof_hcg_pairs, oof_hcg_parked)."""
    r = O.build_tables(lines=inputs.lines, periods=inputs.periods, item_masters=inputs.item_masters,
                       item_master_files=inputs.item_master_files, hcg=inputs.hcg,
                       master=inputs.master, main=inputs.main, email=inputs.email,
                       without_sheet=inputs.without_hcg_sheet, log=inputs.log)
    return Built(tables=r.tables, months=r.months, problems=r.problems, replaces=r.replaces,
                 report=lambda table: O.report_section(inputs.master, inputs.main, inputs.hcg,
                                                       table(C.OOF), inputs.email))


def _list_mrp(inputs: Inputs) -> dict:
    """HCG's CASH&TPA MRP of each brand (incl GST, per unit): price variation's unit check."""
    return {m: b.mrp for m, b in inputs.main.prices.items()}


def build_prices(inputs: Inputs) -> Built:
    """purchase_prices: price variation over the scope's lines of every built month."""
    table, months = P.build_table(inputs.lines, inputs.item_masters, _list_mrp(inputs))
    inputs.log(P.summary_line(months))
    return Built(tables={C.PRICES: table}, months=months, report=lambda _: P.report_section())


def build_generic(inputs: Inputs) -> Built:
    """purchase_generic: generic vs branded over the scope's lines of every built month."""
    definition = Gn.load_definition(inputs.reference)
    priced = P.priced_lines(inputs.lines, inputs.item_masters, _list_mrp(inputs))
    table, months = Gn.build_table(priced, definition, inputs.item_masters)
    inputs.log(Gn.summary_line(months))
    return Built(tables={C.GENERIC: table}, months=months,
                 report=lambda published: Gn.report_section(definition, published(C.GENERIC)))


# (report section, builder), run in this order. Each builder owns its tables (config.TABLES);
# a new analysis adds its builder here.
TABLE_BUILDERS: tuple = (
    ("oof", build_oof),
    ("prices", build_prices),
    ("generic", build_generic),
)


def load_inputs(months=None, *, raw=None, cache=None, reference=None, without_hcg_sheet=(),
                log=_log) -> Inputs:
    """Discover and read everything the builders use. `months` limits the GRN months built
    (every Item Master and HCG workbook is still read: other months' masters fill blank minor
    groups, other months' worksheets give HCG's pairs). `without_hcg_sheet`: months to publish
    from sources 2-5 although HCG's email has figures for them."""
    read = G.read_grn_files(raw=raw, cache=cache, months=months, log=log)
    scope = C.load_scope(reference)
    lines, funnel = G.apply_scope(read.frame, scope, read.blank_lines())
    paths = M.item_master_paths(raw)
    item_masters = {p: M.item_master_frame(M.load_month_item_master(p, paths=paths))
                    for p in sorted(paths)}
    hcg = M.load_hcg_workbooks(raw)
    master = M.load_oof_master(raw=raw)
    main = M.load_main_sheet(raw)
    email = O.load_email_figures(reference)
    log(f"[purchase] GRN months {', '.join(read.periods)} ({len(read.frame):,} lines read, "
        f"{len(lines):,} in scope); Item Masters {', '.join(item_masters) or 'none'}; HCG OOF "
        f"workbooks {', '.join(hcg) or 'none'}; OOF master {len(master.entries)} brands; CASH&TPA "
        f"{len(main.prices):,} priced brands; email figures for {len(email)} hospital-months")
    for w in read.warnings:
        log(f"[purchase] note: {w}")
    return Inputs(scope=scope, grn=read, lines=lines, funnel=funnel, periods=read.periods,
                  item_masters=item_masters,
                  item_master_files={p: paths[p].name for p in item_masters}, hcg=hcg,
                  master=master, main=main, email=email, reference=reference,
                  without_hcg_sheet=frozenset(without_hcg_sheet), log=log)


# --------------------------------------------------------------------------- #
# Validation report
# --------------------------------------------------------------------------- #
def month_base(inputs: Inputs, period: str) -> dict:
    """What every month's report holds before the builders add theirs: the GRN file read and
    the scope funnel (lines and Total Amount without tax left after each step)."""
    rep = next(f for f in inputs.grn.files if f.period == period)
    f = inputs.funnel[inputs.funnel["period"] == period]
    return {"period": period, "label": FC.period_label(period),
            "grn": {k: getattr(rep, k) for k in GRN_REPORT_FIELDS},
            "unparseable_dates": rep.unparseable_dates,
            "blank_material_lines": rep.blank_material_lines,
            "funnel": f[["step", "kind", "label", "lines", "value"]].to_dict("records")}


def full_report(inputs: Inputs, months: dict, sections: dict, data=None) -> dict:
    """This build's months merged into the published validation file."""
    vpath = C.data_dir(data) / C.VALIDATION_FILE
    old = {}
    if vpath.exists():
        try:
            old = json.loads(vpath.read_text())
        except ValueError:
            old = {}
    merged = dict(old.get("months") or {})
    merged.update(months)
    merged = {k: merged[k] for k in sorted(merged)}
    return {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generator": "python -m app.etl.purchase",
        "periods": list(merged),
        "latest": max(merged) if merged else None,
        "scope": {"file": C.SCOPE_FILE, "rows": inputs.scope.report(),
                  "order": "movement type -> material series -> storage location -> plant prefix "
                           "-> plant (HCG's order)"},
        "grn_folder": inputs.grn.folder.name if inputs.grn.folder else None,
        "warnings": list(inputs.grn.warnings),
        **sections,
        "months": merged,
    }


def _save_failed(report: dict, cache=None, log=_log) -> None:
    out = C.cache_dir(cache) / "purchase_validation.failed.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str) + "\n")
    log(f"[purchase] failing report saved to {out}")


# --------------------------------------------------------------------------- #
# Publishing
# --------------------------------------------------------------------------- #
def check_columns(name: str, df: pd.DataFrame) -> None:
    """No identifier and no date column may be published."""
    bad = sorted({c.upper() for c in df.columns} & IDENTIFIER_COLUMNS)
    if bad:
        raise BuildError(f"{name} would publish identifier columns {bad}")
    _check_columns(name, df)


def merge_tables(tables: dict, replaces: dict, data=None) -> dict:
    """Per table: its built rows + the published rows of every month they do not replace
    (replaces: table name -> months)."""
    d = C.data_dir(data)
    merged = {}
    for name, df in tables.items():
        built = set(replaces[name])
        parts = [df]
        path = d / f"{name}.parquet"
        if path.exists():
            old = pd.read_parquet(path)
            parts.append(old[~old["period"].astype(str).isin(built)])
        out = pd.concat([p for p in parts if len(p)] or parts[:1], ignore_index=True)
        check_columns(name, out)
        merged[name] = out.sort_values("period", kind="stable", ignore_index=True)
    return merged


def published_view(merged: dict, data=None) -> Published:
    """A table as it stands after this build: merged, else the file already published."""
    d = C.data_dir(data)

    def table(name: str) -> Optional[pd.DataFrame]:
        if name in merged:
            return merged[name]
        path = d / f"{name}.parquet"
        return pd.read_parquet(path) if path.exists() else None
    return table


def publish(merged: dict, report: dict, data=None) -> dict:
    d = C.data_dir(data)
    d.mkdir(parents=True, exist_ok=True)
    staged = []
    try:
        for name, df in merged.items():
            tmp = d / f".{name}.parquet.tmp"
            df.to_parquet(tmp, index=False, compression="zstd")
            staged.append((tmp, d / f"{name}.parquet"))
        for tmp, final in staged:
            os.replace(tmp, final)
    finally:
        for tmp, _ in staged:
            if tmp.exists():
                tmp.unlink()
    sizes = {name: {"rows": int(len(df)), "kb": round((d / f"{name}.parquet").stat().st_size / 1024, 1)}
             for name, df in merged.items()}
    report["tables"] = sizes
    tmp = d / f".{C.VALIDATION_FILE}.tmp"
    tmp.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    os.replace(tmp, d / C.VALIDATION_FILE)
    return sizes


# --------------------------------------------------------------------------- #
# The build
# --------------------------------------------------------------------------- #
def run(inputs: Inputs, *, data=None, cache=None, dry_run: bool = False,
        builders: tuple = TABLE_BUILDERS) -> dict:
    """Run the table builders on the inputs, validate, publish. Returns the validation
    report; raises BuildError (nothing published) when a builder reports a problem."""
    log = inputs.log
    tables, replaces, problems, built = {}, {}, [], []
    months = {p: month_base(inputs, p) for p in inputs.periods}
    for section, builder in builders:
        b = builder(inputs)
        twice = sorted(set(b.tables) & set(tables))
        if twice:
            raise ValueError(f"tables {twice} built twice (by {section!r} and an earlier builder)")
        tables.update(b.tables)
        replaces.update({name: b.replaces.get(name, inputs.periods) for name in b.tables})
        for p, rep in b.months.items():
            months[p][section] = rep
        built.append((section, b))
        problems += b.problems
    merged = merge_tables(tables, replaces, data)
    published = published_view(merged, data)
    sections = {section: b.report(published) for section, b in built}
    report = full_report(inputs, months, sections, data)
    if problems:
        _save_failed(report, cache, log)
        raise BuildError("validation failed, nothing published:\n  " + "\n  ".join(problems))
    if dry_run:
        out = C.cache_dir(cache) / "purchase_validation.dry-run.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
        log(f"[purchase] dry run: nothing published; report in {out}")
        return report
    sizes = publish(merged, report, data)
    log(f"[purchase] published to {C.data_dir(data)}: "
        + ", ".join(f"{k} {v['rows']:,} rows / {v['kb']} KB" for k, v in sizes.items()))
    return report


def build(months=None, *, raw=None, cache=None, data=None, reference=None, dry_run: bool = False,
          without_hcg_sheet: Iterable[str] = (), log=_log) -> dict:
    t0 = time.time()
    inputs = load_inputs(months, raw=raw, cache=cache, reference=reference,
                         without_hcg_sheet=without_hcg_sheet, log=log)
    report = run(inputs, data=data, cache=cache, dry_run=dry_run)
    log(f"[purchase] done in {time.time() - t0:.0f}s")
    return report


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def list_files(raw=None, cache=None, data=None, reference=None, log=_log) -> None:
    """What the build would read, and which GRN workbooks are cached. Reads no workbook."""
    log(f"raw folder:    {C.raw_dir(raw)}")
    folder = G.find_grn_dir(raw)
    log(f"GRN folder:    {folder.name}  (each workbook's month comes from its GR dates)")
    for p in G.grn_files(raw):
        rep = G.cached_report(p, cache)
        log(f"  {p.name}  |  " + (f"cached: {rep.period} ({FC.period_label(rep.period)}), "
                                  f"{rep.rows:,} lines" if rep else "not cached (read on the next build)"))
    log(f"GRN cache:     {C.cache_dir(cache)}")
    for p, path in sorted(M.item_master_paths(raw).items()):
        log(f"Item Master:   {p} ({FC.period_label(p)})  {path.name}")
    log(f"OOF folder:    {M.find_oof_dir(raw).name}")
    log(f"  OOF master:  {M.find_oof_master(raw).name}")
    for p in M.find_hcg_workbooks(raw):
        log(f"  HCG workbook: {p.name}  (month read from its worksheet)")
    from app.etl.spend.build import find_main_sheet       # the spend build owns the lookup
    log(f"Main Sheet:    {find_main_sheet(raw).name}")
    scope = C.load_scope(reference)
    log(f"scope:         {C.SCOPE_FILE}: " + "; ".join(f"{k} {v}" for k, v, _ in scope.rows))
    email = O.load_email_figures(reference)
    log(f"email figures: {O.EMAIL_FILE}: " + (", ".join(f"{pl} {p}" for p, pl in sorted(email)) or "none"))
    definition = Gn.load_definition(reference)
    log(f"generic:       {Gn.ORIGINATOR_FILE}: {len(definition.originators)} patterns; "
        f"{Gn.OVERRIDES_FILE}: {len(definition.overrides)} materials")
    log(f"output:        {C.data_dir(data)}")


def _months(s: str) -> list:
    out = [x.strip() for x in s.split(",") if x.strip()]
    bad = [x for x in out if not K.is_period(x)]
    if bad:
        raise argparse.ArgumentTypeError(f"months must be YYYY-MM: {', '.join(bad)}")
    return out


def main(argv=None) -> int:
    from dotenv import load_dotenv
    load_dotenv()
    ap = argparse.ArgumentParser(prog="python -m app.etl.purchase",
                                 description="Build the purchase tables (HCG's out-of-formulary "
                                             "method, price variation, generic vs branded) from "
                                             "HCG's GRN extracts.")
    ap.add_argument("--list", action="store_true", help="show the files found and exit")
    ap.add_argument("--dry-run", action="store_true", help="build and validate, publish nothing")
    ap.add_argument("--months", type=_months, help="comma list of YYYY-MM (default: every GRN month)")
    ap.add_argument("--without-hcg-sheet", type=_months, default=[], metavar="MONTHS",
                    help="comma list of YYYY-MM to publish from sources 2-5 although HCG's email "
                         "has figures for them (their HCG out-of-formulary workbook is missing)")
    args = ap.parse_args(argv)
    # .env is loaded above, after app.core.config built `settings`: read the paths now.
    dirs = dict(raw=os.environ.get("FORMULARY_RAW_DIR"),
                cache=os.environ.get("PURCHASE_CACHE_DIR"),
                data=os.environ.get("PURCHASE_DATA_DIR"),
                reference=os.environ.get("REFERENCE_DIR"))
    try:
        if args.list:
            list_files(**dirs)
            return 0
        build(args.months, dry_run=args.dry_run, without_hcg_sheet=args.without_hcg_sheet, **dirs)
    except BuildError as e:
        print(f"[purchase] ABORT: {e}", file=sys.stderr, flush=True)
        return 1
    return 0
