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

Only .xlsx and .xlsb workbooks are read (config.EXCEL_SUFFIXES). Any other file in a source
folder (an .xls, a .csv, a PDF) is listed as a SkippedFile with the reason, in the manifest's
warnings, and the build carries on: a file HCG sent in the wrong format is named, never passed
over in silence. So is a sub-folder of a folder that holds its files directly (the sales and Item
Master folders here; GRN, OOF master and HCG MIS in the other builds): none of its files is read.
Lock, temp and system files are not HCG's and are ignored (config.IGNORED_FILE_RE). The purchase
and spend builds find their own folders with the same _excel_files.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from app.etl.formulary import config as C


class BuildError(Exception):
    """The build cannot produce correct tables; nothing has been published."""


@dataclass(frozen=True)
class SkippedFile:
    """A file in a source folder that the build does not read, and why."""
    folder: str                 # the folder's name, as HCG names it
    name: str
    reason: str

    def __str__(self) -> str:
        return f"{self.folder}/{self.name} is not read: {self.reason}"


@dataclass
class Manifest:
    raw_dir: Path
    sales: dict = field(default_factory=dict)           # period -> Path
    item_masters: dict = field(default_factory=dict)    # period -> Path
    payer_master: Optional[Path] = None
    warnings: list = field(default_factory=list)
    # Files not read (SkippedFile), per folder: the sales and Item Master folders and the
    # dashboard-data folder itself (the Payer Master's; the spend build's Main Sheet is there too).
    skipped_sales: list = field(default_factory=list)
    skipped_item_masters: list = field(default_factory=list)
    skipped_dashboard: list = field(default_factory=list)

    @property
    def periods(self) -> list:
        return sorted(self.sales)

    @property
    def skipped(self) -> list:
        return self.skipped_sales + self.skipped_item_masters + self.skipped_dashboard


def skip_reason(path: Path) -> Optional[str]:
    """Why a file in a source folder is not read; None for a workbook the build reads
    (config.EXCEL_SUFFIXES) and for a lock, temp or system file that is not HCG's."""
    suffix = path.suffix.lower()
    if suffix in C.EXCEL_SUFFIXES or C.IGNORED_FILE_RE.search(path.name):
        return None
    return C.SKIPPED_SUFFIX_REASONS.get(suffix) or C.SKIPPED_OTHER_REASON.format(suffix=suffix or "no file type")


def _excel_files(folder: Path, skipped: Optional[list] = None, *,
                 expected_folders: Optional[tuple] = None) -> list:
    """The workbooks directly in `folder` that the build reads, by name. Every other file of
    HCG's in it goes into `skipped` (SkippedFile) when a list is given, for the build to report.
    expected_folders, for a source folder: the patterns of the sub-folders the caller reads itself,
    () for a folder that holds its files directly (sales, Item Master, GRN). Every other sub-folder
    goes into `skipped` too (config.SKIPPED_FOLDER_REASON): no file inside it is read, so an Item
    Master moved into 'Sep 26/' is named, not lost. None passes sub-folders over, for a folder
    whose sub-folders other lookups read (the dashboard-data folder)."""
    out = []
    for p in sorted(folder.iterdir()):
        if C.IGNORED_FILE_RE.search(p.name):
            continue
        if p.is_dir():
            if (skipped is not None and expected_folders is not None
                    and not any(r.match(p.name.strip()) for r in expected_folders)):
                skipped.append(SkippedFile(folder.name, p.name,
                                           C.SKIPPED_FOLDER_REASON.format(folder=folder.name)))
            continue
        if not p.is_file():
            continue
        reason = skip_reason(p)
        if reason is None:
            out.append(p)
        elif skipped is not None:
            skipped.append(SkippedFile(folder.name, p.name, reason))
    return out


def not_read(skipped, match=None) -> str:
    """'; not read: X.xls (why)' for the skipped files (those whose name `match` finds), to add
    to the message of a lookup that found nothing: the file may be there in the wrong format."""
    hit = [s for s in skipped if match is None or match.search(s.name)]
    return ("; not read: " + "; ".join(f"{s.name!r} ({s.reason})" for s in hit)) if hit else ""


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
                sales += _excel_files(sub, m.skipped_sales, expected_folders=())
            elif C.ITEM_MASTER_DIR_RE.match(sub.name.strip()):
                masters += _excel_files(sub, m.skipped_item_masters, expected_folders=())
        payers += [p for p in _excel_files(d, m.skipped_dashboard) if C.PAYER_MASTER_FILE_RE.search(p.name)]
    m.sales = _by_month(sales, "sales", errors)
    m.item_masters = _by_month(masters, "item master", errors)
    if len(payers) > 1:
        errors.append("more than one Payer Master: " + ", ".join(repr(p.name) for p in payers))
    m.payer_master = payers[0] if len(payers) == 1 else None
    if errors:
        raise BuildError("\n  ".join(["file discovery failed:"] + errors))
    m.warnings += [str(s) for s in m.skipped]
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
                         f"the same month's master{not_read(m.skipped_item_masters)}")
    if m.payer_master is None:
        raise BuildError(payer_master_missing(m))
    return periods


def payer_master_missing(m: Manifest) -> str:
    """The abort message when the dashboard-data folder holds no Payer Master workbook."""
    return ("no Payer Master ('*Payer Master*' as .xlsx or .xlsb) in the dashboard-data folder"
            + not_read(m.skipped_dashboard, C.PAYER_MASTER_FILE_RE))
