"""Formulary compliance build: HCG's monthly Excel files -> small committed tables.

Usage (run by hand on the machine that has HCG's files; ~3 min per month to read):
    python -m app.etl.formulary --list                   # what was found, no big reads
    python -m app.etl.formulary --dry-run                # read + validate, publish nothing
    python -m app.etl.formulary                          # build every month and publish
    python -m app.etl.formulary --months 2026-08         # rebuild one month (others kept)
    python -m app.etl.formulary --from-cache             # re-classify the cached lines (seconds)
    python -m app.etl.formulary --limit 5000 --months 2026-08   # smoke test, never publishes

Steps per month: stream the consolidated sales sheet into the line cache (outside the
repo) -> classify with that month's Item Master, the Payer Master and our reference files
-> aggregate -> validate. Only when every month passes are the tables published, into
FORMULARY_DATA_DIR: written to temp files, then os.replace'd; rebuilding some months
replaces only those months' rows. formulary_validation.json is written last.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import tempfile
import time
from datetime import date, datetime, timezone

import pandas as pd

from app.etl.formulary import classify as K
from app.etl.formulary import config as C
from app.etl.formulary import validate as V
from app.etl.formulary.aggregate import (MonthResult, aggregate_month, apply_doctor_names,
                                         connect, doctor_name_map)
from app.etl.formulary.discover import BuildError, discover, select_periods
from app.etl.formulary.extract import cache_paths, cached_periods, extract_month, read_sidecar
from app.etl.formulary.masters import (load_brand_rules, load_hospital_map, load_item_master,
                                       load_payer_master, load_payer_overrides)

SORT_KEYS = {
    C.SUMMARY: ["period", "plant", "patient_type", "category", "payer_bucket", "formulary_class"],
    C.BY_DOCTOR: ["period", "plant", "patient_type", "category", "payer_bucket",
                  "formulary_group", "doctor_id"],
    C.BY_ITEM: ["period", "plant", "patient_type", "payer_bucket", "material"],
    C.DOCTOR_ITEM: ["period", "plant", "payer_bucket", "doctor_id", "material"],
    C.RULE_BREACH: ["measure", "period", "plant", "patient_type", "payer_bucket", "doctor_id",
                    "material"],
    C.RULE_BOOK: ["period", "material"],
}
INT_COLS = ("lines",)
FLOAT_COLS = ("cost", "mrp", "qty")


def _log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
# Publishing
# --------------------------------------------------------------------------- #
def _check_columns(name: str, df: pd.DataFrame) -> None:
    bad = sorted({c.upper() for c in df.columns} & C.FORBIDDEN_COLUMNS)
    if bad:
        raise BuildError(f"{name} would publish forbidden columns {bad}")

    def is_date(col) -> bool:
        if pd.api.types.is_datetime64_any_dtype(col):
            return True
        first = col.dropna().head(1).tolist()
        return bool(first) and isinstance(first[0], date)

    dated = [c for c in df.columns if is_date(df[c])]
    if dated:
        raise BuildError(f"{name} would publish date columns {dated}")


def _tidy(name: str, df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in INT_COLS:
        if c in df:
            df[c] = df[c].astype("int64")
    for c in FLOAT_COLS:
        if c in df:
            df[c] = df[c].astype("float64")
    return df.sort_values(SORT_KEYS[name], kind="stable").reset_index(drop=True)


def merge_tables(results: list, data=None) -> dict:
    """New months' rows + the published rows of every other month, with one doctor name
    per (plant, doctor_id) across all of them."""
    d = C.data_dir(data)
    periods = {r.period for r in results}
    merged = {}
    kept = {}
    for name in C.TABLES:
        path = d / f"{name}.parquet"
        old = pd.read_parquet(path) if path.exists() else None
        kept[name] = (old[~old["period"].isin(periods)] if old is not None else None)
    # most frequent spelling: exact line counts for the months being built, the published
    # rows' line counts for the months kept
    counts = [r.doctor_names for r in results]
    old_doc = kept[C.BY_DOCTOR]
    if old_doc is not None and len(old_doc):
        o = old_doc[old_doc["doctor_name"] != C.UNKNOWN_DOCTOR_NAME]
        counts.append(o.groupby(["plant", "doctor_id", "doctor_name"], as_index=False)["lines"]
                       .sum().rename(columns={"lines": "n"}))
    names = doctor_name_map(pd.concat([c for c in counts if len(c)], ignore_index=True)
                            if any(len(c) for c in counts) else pd.DataFrame(
                                columns=["plant", "doctor_id", "doctor_name", "n"]))
    for name in C.TABLES:
        parts = [r.tables[name] for r in results]
        if kept[name] is not None and len(kept[name]):
            parts.append(kept[name])
        df = pd.concat(parts, ignore_index=True)
        if name in (C.BY_DOCTOR, C.DOCTOR_ITEM, C.RULE_BREACH):
            df = apply_doctor_names(df, names)
        _check_columns(name, df)
        merged[name] = _tidy(name, df)
    return merged


def _write_atomic(staged: list) -> None:
    for tmp, final in staged:
        os.replace(tmp, final)


def publish(merged: dict, report: dict, data=None) -> dict:
    d = C.data_dir(data)
    d.mkdir(parents=True, exist_ok=True)
    staged, sizes = [], {}
    try:
        for name, df in merged.items():
            final = d / f"{name}.parquet"
            tmp = d / f".{name}.parquet.tmp"
            df.to_parquet(tmp, index=False, compression="zstd")
            staged.append((tmp, final))
        _write_atomic(staged)
    finally:
        for tmp, _ in staged:
            if tmp.exists():
                tmp.unlink()
    for name in merged:
        sizes[name] = {"rows": int(len(merged[name])),
                       "kb": round((d / f"{name}.parquet").stat().st_size / 1024, 1)}
    report["tables"] = sizes
    vpath = d / C.VALIDATION_FILE
    tmp = d / f".{C.VALIDATION_FILE}.tmp"
    tmp.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    os.replace(tmp, vpath)
    return sizes


def full_report(month_reports: dict, pm, overrides, hospitals, data=None, rules=()) -> dict:
    """The new months' reports merged into the published validation file."""
    vpath = C.data_dir(data) / C.VALIDATION_FILE
    old = {}
    if vpath.exists():
        try:
            old = json.loads(vpath.read_text())
        except ValueError:
            old = {}
    months = dict(old.get("months") or {})
    months.update(month_reports)
    months = {k: months[k] for k in sorted(months)}
    return {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generator": "python -m app.etl.formulary",
        "periods": list(months),
        "latest": max(months) if months else None,
        "rules": {
            "category": "first digit of MATERIALCODE: 1 Onco Drugs, 2 Other Drugs, 3 Consumables",
            "formulary_class": "same month's Item Master: FORMULARY+Category | NON FORMULARY | "
                               "OUT OF FORMULARY | NOT IN MASTER",
            "groups": "A = class starting FORMULARY, B = OUT OF FORMULARY, C = NON FORMULARY",
            "compliance": "pooled = A/(A+B) over drugs; mean = average of the onco and "
                          "non-onco ratios; ratio of sums, never an average of percentages",
            "payer": "blank -> CASH; payer_overrides.csv; Payer Master first row by "
                     "CUSTOMERNAME; else CASH(TPA)",
        },
        # the brand rules the published tables were built with (the API serves these)
        "brand_rules": [r.as_dict() for r in rules],
        "payer_master": V.payer_master_report(pm, overrides),
        "hospital_map": {"file": C.HOSPITAL_MAP_FILE, "hospitals": len(hospitals)},
        "months": months,
    }


# --------------------------------------------------------------------------- #
# The build
# --------------------------------------------------------------------------- #
def build(months=None, *, raw=None, cache=None, data=None, reference=None,
          from_cache: bool = False, dry_run: bool = False, limit: int = 0,
          log=_log) -> dict:
    """Build the given months (default: every month found). Returns the validation
    report; raises BuildError (nothing published) when a check fails. --limit runs read
    into a throwaway cache and never publish."""
    t0 = time.time()
    if limit and from_cache:
        raise BuildError("--limit cannot be combined with --from-cache")
    manifest = discover(raw)
    for w in manifest.warnings:
        log(f"[formulary] note: {w}")
    available = cached_periods(cache) if from_cache else list(manifest.sales)
    periods = select_periods(manifest, months, available)
    hospitals = load_hospital_map(reference)
    overrides = load_payer_overrides(reference)
    pm = load_payer_master(manifest.payer_master)
    rules = load_brand_rules(reference)
    log(f"[formulary] months {', '.join(periods)}; payer master {pm.rows:,} rows, "
        f"{len(pm.names):,} names; {sum(len(v) for v in overrides.values())} overrides; "
        f"{len(rules)} brand-rule measures; "
        f"{len(hospitals)} hospitals mapped")
    publishing = not dry_run and not limit

    with (tempfile.TemporaryDirectory(prefix="formulary-limit-") if limit
          else contextlib.nullcontext(None)) as tmp_cache:
        work_cache = tmp_cache or cache
        results, reports = [], {}
        con = connect()
        try:
            for p in periods:
                if from_cache:
                    sidecar = read_sidecar(work_cache, p)
                    now = manifest.sales.get(p)
                    if now is not None and sidecar.get("source_file") != now.name:
                        log(f"[formulary] note: {p} cache came from {sidecar['source_file']!r}, "
                            f"the raw folder now has {now.name!r}")
                else:
                    sidecar = extract_month(manifest.sales[p], p, work_cache, limit=limit, log=log)
                im = load_item_master(manifest.item_masters[p])
                res: MonthResult = aggregate_month(
                    p, cache=work_cache, item_master=im, payer_master=pm, overrides=overrides,
                    hospitals=hospitals, has_hcg=sidecar["has_hcg_columns"], con=con, rules=rules)
                rep = V.month_report(period=p, sidecar=sidecar, item_master=im, result=res,
                                     rules=rules)
                rep["built_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
                if limit:
                    rep["warnings"].append(f"--limit {limit}: partial read, not published")
                reports[p] = rep
                results.append(res)
                log(summary_line(rep))
                problems = V.abort_reasons(rep)
                if problems:
                    _save_failed(reports, pm, overrides, hospitals, cache, data, rules)
                    raise BuildError("validation failed, nothing published:\n  "
                                     + "\n  ".join(problems))
        finally:
            con.close()

    report = full_report(reports, pm, overrides, hospitals, data, rules)
    if publishing:
        merged = merge_tables(results, data)
        sizes = publish(merged, report, data)
        log(f"[formulary] published to {C.data_dir(data)}: "
            + ", ".join(f"{k} {v['rows']:,} rows / {v['kb']} KB" for k, v in sizes.items()))
    else:
        if not limit:
            out = C.cache_dir(cache) / "formulary_validation.dry-run.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
            log(f"[formulary] dry run: nothing published; report in {out}")
        else:
            log("[formulary] --limit run: nothing published")
    log(f"[formulary] done in {time.time() - t0:.0f}s")
    return report


def _save_failed(reports, pm, overrides, hospitals, cache, data, rules=()) -> None:
    try:
        out = C.cache_dir(cache) / "formulary_validation.failed.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(full_report(reports, pm, overrides, hospitals, data, rules),
                                  indent=1, ensure_ascii=False) + "\n")
        _log(f"[formulary] failing report saved to {out}")
    except Exception:
        pass


def summary_line(rep: dict) -> str:
    kr, al, src, pub = rep["kr"], rep["all_hospitals"], rep["source"], rep["published"]
    ag = rep.get("agreement") or {}

    def a(k):
        return f"{ag[k]['lines_pct']}%" if ag.get(k) else "n/a"

    def pc(v):
        return "n/a" if v is None else f"{v:.4f}%"

    return (f"[formulary] {rep['period']}: {src['rows_read']:,} rows, cost read "
            f"{src['cost_read']:,.2f} / published {pub['cost']:,.2f}; drugs in master "
            f"{rep['drugs']['found_in_master_pct']}%; HCG agreement category {a('category')}, "
            f"payer {a('payer')}, segment {a('segment')}\n"
            f"[formulary] {rep['period']}: KR A {kr['formulary']:,.2f} B {kr['out_of_formulary']:,.2f} "
            f"C {kr['non_formulary']:,.2f} pooled {pc(kr['pooled_pct'])} (onco {pc(kr['onco_pct'])}, "
            f"non-onco {pc(kr['non_onco_pct'])}, mean {pc(kr['mean_pct'])}); all hospitals "
            f"pooled {pc(al['pooled_pct'])}"
            + "".join(f"\n[formulary] {rep['period']}: KR {k} {v['cost']:,.2f} ({v['share_pct']}% of drug spend)"
                      for k, v in (rep.get("brand_rules") or {}).get("kr", {}).items()))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def list_files(raw=None, cache=None, reference=None, data=None, log=_log) -> None:
    m = discover(raw)
    cached = set(cached_periods(cache))
    log(f"raw folder:   {m.raw_dir}")
    log(f"payer master: {m.payer_master.name if m.payer_master else 'MISSING'}")
    log(f"cache:        {C.cache_dir(cache)}")
    log(f"output:       {C.data_dir(data)}")
    for p in sorted(set(m.sales) | set(m.item_masters) | cached):
        s, im = m.sales.get(p), m.item_masters.get(p)
        c = ""
        if p in cached:
            meta = json.loads(cache_paths(cache, p)[1].read_text())
            c = f"cached {meta['rows']:,} rows{' (LIMITED)' if meta.get('limited') else ''}"
        log(f"  {p} ({C.period_label(p)})  sales: {s.name if s else '-'}  |  "
            f"item master: {im.name if im else 'MISSING'}  |  {c or 'not cached'}")
    for w in m.warnings:
        log(f"  note: {w}")
    hm = load_hospital_map(reference)
    ov = load_payer_overrides(reference)
    log(f"reference:    {len(hm)} hospitals in {C.HOSPITAL_MAP_FILE}, "
        f"{sum(len(v) for v in ov.values())} rows in {C.PAYER_OVERRIDES_FILE}")


def _months(s: str) -> list:
    out = [x.strip() for x in s.split(",") if x.strip()]
    bad = [x for x in out if not K.is_period(x)]
    if bad:
        raise argparse.ArgumentTypeError(f"months must be YYYY-MM: {', '.join(bad)}")
    return out


def main(argv=None) -> int:
    from dotenv import load_dotenv
    load_dotenv()
    ap = argparse.ArgumentParser(prog="python -m app.etl.formulary",
                                 description="Build the formulary compliance tables.")
    ap.add_argument("--list", action="store_true", help="show the files found and exit")
    ap.add_argument("--dry-run", action="store_true", help="read and validate, publish nothing")
    ap.add_argument("--months", type=_months, help="comma list of YYYY-MM (default: all)")
    ap.add_argument("--from-cache", action="store_true",
                    help="re-use the cached lines instead of reading the sales workbooks")
    ap.add_argument("--limit", type=int, default=0,
                    help="read only the first N lines per month (never publishes)")
    args = ap.parse_args(argv)
    # .env is loaded above, after app.core.config built `settings`: read the paths now.
    dirs = dict(raw=os.environ.get("FORMULARY_RAW_DIR"),
                cache=os.environ.get("FORMULARY_CACHE_DIR"),
                reference=os.environ.get("REFERENCE_DIR"),
                data=os.environ.get("FORMULARY_DATA_DIR"))
    try:
        if args.list:
            list_files(**dirs)
            return 0
        build(args.months, from_cache=args.from_cache, dry_run=args.dry_run,
              limit=args.limit, **dirs)
    except BuildError as e:
        print(f"[formulary] ABORT: {e}", file=sys.stderr, flush=True)
        return 1
    return 0
