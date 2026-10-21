"""Find HCG's formulary files under FORMULARY_RAW_DIR and pair each sales month with its
own Item Master.

Layout (the top folder is a hash):
  <raw>/**/Formulary Compliance and Spend Analysis Dashboard data/
      IP and OP sales/*.xls[xb]          one consolidated IP+OP file per month
      Item Master Sheet/*.xls[xb]        one Item Master per month
      *Payer Master*.xls[xb]             one Payer Master
The separate '<raw>/**/IP Sales/' and 'OP Sales/' extracts are NOT read: June's raw IP
file carries 7 locations (2 Nashik units, 5 Milann clinics) HCG leaves out of the report.

The month comes from the file name. Each sales month must use the SAME month's Item
Master — the other months' masters give wrong numbers (Aug-26 sales on the Jul-26 master
read 99.06% for KR instead of 98.42%) — so a month without one aborts the build.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from app.etl.formulary import config as C


class BuildError(Exception):
    """The build cannot produce correct tables; nothing has been published."""


@dataclass
class Manifest:
    raw_dir: Path
    sales: dict = field(default_factory=dict)           # period -> Path
    item_masters: dict = field(default_factory=dict)    # period -> Path
    payer_master: Optional[Path] = None
    warnings: list = field(default_factory=list)

    @property
    def periods(self) -> list:
        return sorted(self.sales)


def _excel_files(folder: Path) -> list:
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in C.EXCEL_SUFFIXES
                  and not p.name.startswith("~$"))


def _by_month(files, what: str, errors: list) -> dict:
    out: dict = {}
    for p in files:
        period = C.month_from_name(p.name)
        if period is None:
            errors.append(f"{what}: cannot read a month from the file name {p.name!r}")
        elif period in out:
            errors.append(f"{what}: two files for {period}: {out[period].name!r} and {p.name!r}")
        else:
            out[period] = p
    return out


def discover(raw=None) -> Manifest:
    root = C.raw_dir(raw)
    if not root.is_dir():
        raise BuildError(f"FORMULARY_RAW_DIR does not exist: {root}")
    dashboards = sorted(d for d in root.rglob("*")
                        if d.is_dir() and C.DASHBOARD_DIR_RE.match(d.name))
    if not dashboards:
        raise BuildError(f"no 'Formulary Compliance and Spend Analysis Dashboard data' "
                         f"folder under {root}")
    m = Manifest(raw_dir=root)
    errors: list = []
    sales, masters, payers = [], [], []
    for d in dashboards:
        for sub in sorted(p for p in d.iterdir() if p.is_dir()):
            if C.SALES_DIR_RE.match(sub.name.strip()):
                sales += _excel_files(sub)
            elif C.ITEM_MASTER_DIR_RE.match(sub.name.strip()):
                masters += _excel_files(sub)
        payers += [p for p in _excel_files(d) if C.PAYER_MASTER_FILE_RE.search(p.name)]
    m.sales = _by_month(sales, "sales", errors)
    m.item_masters = _by_month(masters, "item master", errors)
    if len(payers) > 1:
        errors.append("more than one Payer Master: " + ", ".join(repr(p.name) for p in payers))
    m.payer_master = payers[0] if len(payers) == 1 else None
    if errors:
        raise BuildError("\n  ".join(["file discovery failed:"] + errors))
    for period in sorted(set(m.item_masters) - set(m.sales)):
        m.warnings.append(f"Item Master for {period} has no sales file (not used)")
    return m


def select_periods(m: Manifest, months=None, available=None) -> list:
    """The periods to build: every month in `available` (default: the discovered sales
    files), or the requested ones. Aborts on a requested month that is not available or
    any chosen month with no Item Master."""
    available = sorted(m.sales if available is None else available)
    periods = available if not months else sorted(set(months))
    missing = [p for p in periods if p not in available]
    if missing:
        raise BuildError(f"no sales data for {', '.join(missing)} "
                         f"(found: {', '.join(available) or 'none'})")
    if not periods:
        raise BuildError(f"no sales files found under {m.raw_dir}")
    no_master = [p for p in periods if p not in m.item_masters]
    if no_master:
        raise BuildError(f"no Item Master for {', '.join(no_master)}: each sales month needs "
                         f"the same month's master")
    if m.payer_master is None:
        raise BuildError("no Payer Master ('*Payer Master*.xls[xb]') found")
    return periods
