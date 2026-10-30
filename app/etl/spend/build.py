"""Spend analysis build: formulary line cache + HCG's brand lists -> small committed tables.

Usage (after python -m app.etl.formulary has filled the line cache; takes seconds):
    python -m app.etl.spend --list        # the scheme list and cached months found
    python -m app.etl.spend --dry-run     # build + validate, publish nothing
    python -m app.etl.spend               # build every cached month and publish

Every month in the line cache is rebuilt; a month published earlier but no longer cached
keeps its published rows. Both lists (SCHEME and CASH&TPA) come from the one Spend Analysis
Main Sheet; which cash brands HCG bifurcates comes from app/data/reference/
cash_tpa_bifurcation.csv. The out-of-formulary margin is not built here: it follows HCG's
purchase (GRN) method in python -m app.etl.purchase. Each month's KR figures are printed so
they can be set beside HCG's email (tests/test_spend_golden.py checks KR Aug-26 against it).
Tables are written to temp files and os.replace'd, spend_validation.json last.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from app.etl.formulary import config as FC
from app.etl.formulary.aggregate import apply_doctor_names, connect, doctor_name_map
from app.etl.formulary.build import _check_columns
from app.etl.formulary.discover import BuildError, _excel_files, discover
from app.etl.formulary.extract import cached_periods, read_sidecar
from app.etl.formulary.masters import load_hospital_map, load_payer_master, load_payer_overrides
from app.etl.spend import config as C
from app.etl.spend.aggregate import MonthResult, aggregate_month
from app.etl.spend.masters import (BrandList, load_bifurcation, load_brand_blocks, load_cash_exclusions,
                                   load_cash_list, load_scheme_list)

SORT_KEYS = {
    C.SWITCH: ["period", "plant", "patient_type", "category", "doctor_id", "material"],
    C.ATTIS_USE: ["period", "plant", "patient_type", "category", "payer_bucket", "material"],
    C.CASH_SWITCH: ["period", "plant", "patient_type", "category", "payer_bucket", "doctor_id",
                    "material"],
    C.CASH_ALLOWANCE: ["period", "plant", "material"],
    C.BRAND_BLOCKS: ["block_id", "p_rank", "material"],
}
INT_COLS = ("lines", "block_id", "block_size", "p_rank")
FLOAT_COLS = ("qty", "actual_cost", "used_price", "suggested_price", "used_value",
              "suggested_value", "cost", "mrp", "cash_qty", "used_mrp", "used_gst",
              "suggested_mrp", "suggested_gst", "used_cost_value", "used_mrp_value",
              "suggested_cost_value", "suggested_mrp_value", "actual_margin", "potential_margin",
              "margin_loss", "sharing", "group_qty", "allowed_qty", "excess_qty", "block_sharing_sum")
BOOL_COLS = ("attis", "bifurcated", "tie", "scheme_attis")
DOCTOR_TABLES = (C.SWITCH, C.CASH_SWITCH)


def _log(msg: str) -> None:
    print(msg, flush=True)


def find_main_sheet(raw=None) -> Path:
    """The one '*Spend Analysis*' workbook in the dashboard-data folder(s)."""
    root = FC.raw_dir(raw)
    dashboards = sorted(d for d in root.rglob("*")
                        if d.is_dir() and FC.DASHBOARD_DIR_RE.match(d.name))
    found = [p for d in dashboards for p in _excel_files(d) if C.MAIN_SHEET_FILE_RE.search(p.name)]
    if not found:
        raise BuildError(f"no '*Spend Analysis*' workbook in the dashboard-data folder under {root}")
    if len(found) > 1:
        raise BuildError("more than one Spend Analysis workbook: "
                         + ", ".join(repr(p.name) for p in found))
    return found[0]


def _fy_label(fy: tuple) -> str:
    return f"{fy[0]}-{str(fy[1])[2:]}"


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def month_report(res: MonthResult, sidecar: dict) -> dict:
    rep = {"period": res.period, "label": FC.period_label(res.period),
           "source_file": sidecar.get("source_file"), **res.checks, "warnings": []}
    if not res.checks["scheme"]["lines"]:
        rep["warnings"].append("no GOV line uses a brand that has a scheme suggestion")
    return rep


def summary_line(rep: dict) -> str:
    s, a, c, kr = rep["scheme"], rep["attis"], rep["cash"], rep["kr"]
    return (f"[spend] {rep['period']}: scheme used {s['used_value']:,.2f} vs suggested "
            f"{s['suggested_value']:,.2f}; Attis saving missed {a['saving_missed']:,.2f}; "
            f"cash & TPA margin {c['actual_margin']:,.2f} vs {c['potential_margin']:,.2f} "
            f"(loss {c['margin_loss']:,.2f})\n"
            f"[spend] {rep['period']}: KR scheme {kr['scheme']['used_value']:,.2f} / "
            f"{kr['scheme']['suggested_value']:,.2f} / {kr['scheme']['saving_missed']:,.2f}; "
            f"KR Attis {kr['attis']['used_value']:,.2f} / {kr['attis']['suggested_value']:,.2f} "
            f"/ {kr['attis']['saving_missed']:,.2f}; KR Healthium {kr['healthium']['saving_missed']:,.2f}; "
            f"KR cash & TPA {kr['cash']['actual_margin']:,.2f} / {kr['cash']['potential_margin']:,.2f} "
            f"/ {kr['cash']['margin_loss']:,.2f}")


def scheme_report(sl: BrandList) -> dict:
    sw = sl.switches.values()
    return {"file": sl.path.name, "sheet": sl.sheet, "financial_year": _fy_label(sl.fy),
            "brand_rows": sl.rows, "switches": len(sl.switches),
            "attis_switches": sum(1 for s in sw if s.attis),
            "attis_brands": len({s.suggested for s in sw if s.attis}),
            "healthium_switches": sum(1 for s in sw if (s.scheme_group or "").strip().upper()
                                      in C.HEALTHIUM_GROUPS),
            "skipped": sl.skipped}


def exclude(cl: BrandList, excluded: dict) -> dict:
    """Drop the excluded brands from the cash list; {material: note} of those that were in it."""
    hit = {m: n for m, n in excluded.items() if m in cl.switches}
    cl.switches = {m: s for m, s in cl.switches.items() if m not in excluded}
    return hit


def cash_report(cl: BrandList, bifurcation: dict, excluded: dict = None, hit: dict = None) -> dict:
    excluded, hit = excluded or {}, hit or {}
    return {"file": cl.path.name, "sheet": cl.sheet, "financial_year": _fy_label(cl.fy),
            "brand_rows": cl.rows, "switches": len(cl.switches), "skipped": cl.skipped,
            "excluded": {"file": C.EXCLUDED_FILE, "brands": sorted(hit),
                         "not_in_cash_list": sorted(set(excluded) - set(hit)),
                         "source": "brands HCG's KR Aug-26 report leaves out; HCG to confirm"},
            "bifurcation": {"file": C.BIFURCATION_FILE, "brands": len(bifurcation),
                            "not_in_cash_list": sorted(set(bifurcation) - set(cl.switches)),
                            "source": "seeded from HCG KR Aug-26 Spend Analysis; HCG to send the "
                                      "full rules"}}


# --------------------------------------------------------------------------- #
# Publishing
# --------------------------------------------------------------------------- #
def _tidy(name: str, df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in INT_COLS:
        if c in df:
            df[c] = df[c].astype("int64")
    for c in FLOAT_COLS:
        if c in df:
            df[c] = df[c].astype("float64")
    for c in BOOL_COLS:
        if c in df:
            df[c] = df[c].astype(bool)
    return df.sort_values(SORT_KEYS[name], kind="stable").reset_index(drop=True)


def merge_tables(results: list, data=None, undated: dict = None) -> dict:
    """The built months' rows + the published rows of every other month, with one doctor
    name per (plant, doctor_id). Undated tables (FY master data) are replaced whole."""
    d = C.data_dir(data)
    periods = {r.period for r in results}
    merged = {name: df.copy() for name, df in (undated or {}).items()}
    for name in (n for n in C.TABLES if n not in C.UNDATED):
        parts = [r.tables[name] for r in results]
        path = d / f"{name}.parquet"
        if path.exists():
            old = pd.read_parquet(path)
            parts.append(old[~old["period"].astype(str).isin(periods)])
        merged[name] = pd.concat([p for p in parts if len(p)] or parts[:1], ignore_index=True)
    counts = [r.doctor_names for r in results if len(r.doctor_names)]
    for name in DOCTOR_TABLES:
        t = merged[name]
        if "doctor_name" in t.columns:          # kept months already carry a name
            o = t[~t["period"].astype(str).isin(periods)]
            o = o[o["doctor_name"].notna() & (o["doctor_name"] != FC.UNKNOWN_DOCTOR_NAME)]
            if len(o):
                counts.append(o.groupby(["plant", "doctor_id", "doctor_name"], as_index=False)["lines"]
                               .sum().rename(columns={"lines": "n"}))
    names = doctor_name_map(pd.concat(counts, ignore_index=True)) if counts else {}
    for name in DOCTOR_TABLES:
        merged[name] = apply_doctor_names(merged[name], names)
    for name in (n for n in C.TABLES if n in merged):
        _check_columns(name, merged[name])
        merged[name] = _tidy(name, merged[name])
    return merged


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


def full_report(month_reports: dict, sl: BrandList, cl: BrandList, bifurcation: dict,
                data=None, excluded: dict = None, hit: dict = None) -> dict:
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
        "generator": "python -m app.etl.spend",
        "periods": list(months),
        "latest": max(months) if months else None,
        "rules": {
            "scheme_patient": "payer bucket GOV (the formulary build's payer rules), IP and OP",
            "switch": "the line's material has a suggested brand in HCG's SCHEME list",
            "values": "qty x FY price-list price (HCG Pharma selling price excl GST) of the used "
                      "brand and of the suggested brand; saving missed = used - suggested; "
                      "% = saving missed / used",
            "attis": "the suggested brand is an Attis brand ('Attis Brand' in the list)",
            "attis_consumption": "every line whose HIS manufacturer is ATTIS ..., every payer",
            "healthium": "SCHEME rows whose CATEGORY is SUTURES or Endo (HCG's J&J & Covidien vs "
                         "Healthium line)",
            "cash_patient": "payer bucket CASH or CASH(TPA), IP and OP",
            "cash_excluded": f"brands in {C.EXCLUDED_FILE} are not counted (they still count in "
                             f"molecule groups)",
            "cash_qty": "cash/TPA qty; for a bifurcated brand only the excess over SHARING x the "
                        "molecule group's all-payer qty at that hospital in that month, never "
                        "below 0, shared across its rows in proportion to their qty",
            "cash_values": "unit margin = MRP / (1 + GST%) - HCG Pharma price, FY list, for the "
                           "used and the suggested brand; actual margin = qty x used margin, "
                           "potential = qty x suggested margin, loss = potential - actual; "
                           "% = loss / potential",
        },
        "scheme_list": scheme_report(sl),
        "cash_list": cash_report(cl, bifurcation, excluded, hit),
        "months": months,
    }


# --------------------------------------------------------------------------- #
# The build
# --------------------------------------------------------------------------- #
def build(*, raw=None, cache=None, data=None, reference=None, dry_run: bool = False,
          log=_log) -> dict:
    t0 = time.time()
    periods = cached_periods(cache)
    if not periods:
        raise BuildError(f"the line cache {FC.cache_dir(cache)} is empty — run "
                         f"python -m app.etl.formulary first")
    sidecars = {p: read_sidecar(cache, p) for p in periods}     # aborts on a --limit read
    manifest = discover(raw)
    if manifest.payer_master is None:
        raise BuildError("no Payer Master ('*Payer Master*.xls[xb]') found")
    main_sheet = find_main_sheet(raw)
    sl = load_scheme_list(main_sheet)
    cl = load_cash_list(main_sheet)
    for lst in (sl, cl):
        outside = [p for p in periods if not lst.covers(p)]
        if outside:
            raise BuildError(f"the {lst.sheet} list is for FY {_fy_label(lst.fy)}; it does not "
                             f"cover {', '.join(outside)}")
    bifurcation = load_bifurcation(reference)
    blocks = load_brand_blocks(main_sheet, sl, cl)          # before the exclusions: master data
    excluded = load_cash_exclusions(reference)
    hit = exclude(cl, excluded)
    hospitals = load_hospital_map(reference)
    overrides = load_payer_overrides(reference)
    pm = load_payer_master(manifest.payer_master)
    log(f"[spend] months {', '.join(periods)}; scheme list FY {_fy_label(sl.fy)}: "
        f"{len(sl.switches):,} suggestions ({sum(s.attis for s in sl.switches.values())} to an "
        f"Attis brand), {len(sl.skipped)} without a price; cash & TPA list {len(cl.switches):,} "
        f"suggestions, {len(cl.skipped)} without a price; {len(bifurcation)} bifurcated brands "
        f"in {C.BIFURCATION_FILE}; {len(hit)} brands left out per {C.EXCLUDED_FILE}")

    results, reports = [], {}
    con = connect()
    try:
        for p in periods:
            res = aggregate_month(p, cache=cache, scheme=sl, cash=cl, bifurcation=bifurcation,
                                  payer_master=pm, overrides=overrides, hospitals=hospitals, con=con)
            rep = month_report(res, sidecars[p])
            reports[p] = rep
            results.append(res)
            log(summary_line(rep))
    finally:
        con.close()
    report = full_report(reports, sl, cl, bifurcation, data, excluded, hit)
    multi = blocks[blocks["block_size"] > 1]
    report["brand_blocks"] = {
        "sheet": C.CASH_SHEET, "brands": int(len(blocks)), "blocks": int(blocks["block_id"].nunique()),
        "multi_brand_blocks": int(multi["block_id"].nunique()),
        "blocks_with_tied_p1": int(multi.loc[multi["tie"] & (multi["p_rank"] == 1), "block_id"].nunique()),
        "blocks_not_summing_to_1": int(blocks.loc[(blocks["block_sharing_sum"] - 1).abs() > 0.001,
                                                  "block_id"].nunique()),
        "rule": "P1 = the highest '2026-27 SHARING' in a block of interchangeable brands (proposed, "
                "until HCG confirms or sends its Brand Priority Master)"}
    if dry_run:
        out = FC.cache_dir(cache) / "spend_validation.dry-run.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
        log(f"[spend] dry run: nothing published; report in {out}")
    else:
        sizes = publish(merge_tables(results, data, {C.BRAND_BLOCKS: blocks}), report, data)
        log(f"[spend] published to {C.data_dir(data)}: "
            + ", ".join(f"{k} {v['rows']:,} rows / {v['kb']} KB" for k, v in sizes.items()))
    log(f"[spend] done in {time.time() - t0:.0f}s")
    return report


def list_files(raw=None, cache=None, data=None, reference=None, log=_log) -> None:
    path = find_main_sheet(raw)
    sl = load_scheme_list(path)
    cl = load_cash_list(path)
    bif = load_bifurcation(reference)
    excl = load_cash_exclusions(reference)
    hit = exclude(cl, excl)
    log(f"scheme list:  {path.name} (sheet {C.SCHEME_SHEET}, FY {_fy_label(sl.fy)})")
    log(f"              {sl.rows:,} brand rows, {len(sl.switches):,} suggestions, "
        f"{sum(s.attis for s in sl.switches.values())} to an Attis brand, "
        f"{len(sl.skipped)} skipped (no price)")
    log(f"cash list:    sheet {C.CASH_SHEET}, FY {_fy_label(cl.fy)}: {cl.rows:,} brand rows, "
        f"{len(cl.switches) + len(hit):,} suggestions, {len(cl.skipped)} skipped (no price)")
    log(f"bifurcation:  {len(bif)} brands in {C.BIFURCATION_FILE}"
        + (f" ({len(set(bif) - set(cl.switches) - set(hit))} not in the cash list)" if bif else ""))
    log(f"excluded:     {len(hit)} cash brands left out per {C.EXCLUDED_FILE}"
        + (f" ({len(excl) - len(hit)} not in the cash list)" if len(excl) > len(hit) else ""))
    log(f"line cache:   {FC.cache_dir(cache)}")
    for p in cached_periods(cache):
        s = read_sidecar(cache, p)
        log(f"  {p} ({FC.period_label(p)})  {s.get('rows', 0):,} lines from {s.get('source_file')}"
            f"{'' if sl.covers(p) else '  <- outside the list year'}")
    log(f"output:       {C.data_dir(data)}")


def main(argv=None) -> int:
    from dotenv import load_dotenv
    load_dotenv()
    ap = argparse.ArgumentParser(prog="python -m app.etl.spend",
                                 description="Build the spend tables (Attis, scheme and cash & "
                                             "TPA brand switches) from the formulary line cache.")
    ap.add_argument("--list", action="store_true", help="show the scheme list and cached months")
    ap.add_argument("--dry-run", action="store_true", help="build and validate, publish nothing")
    args = ap.parse_args(argv)
    dirs = dict(raw=os.environ.get("FORMULARY_RAW_DIR"),
                cache=os.environ.get("FORMULARY_CACHE_DIR"),
                data=os.environ.get("SPEND_DATA_DIR"))
    try:
        if args.list:
            list_files(reference=os.environ.get("REFERENCE_DIR"), **dirs)
            return 0
        build(dry_run=args.dry_run, reference=os.environ.get("REFERENCE_DIR"), **dirs)
    except BuildError as e:
        print(f"[spend] ABORT: {e}", file=sys.stderr, flush=True)
        return 1
    return 0
