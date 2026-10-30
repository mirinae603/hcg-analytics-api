"""Paths, folder and sheet names, GRN columns, the analysis scope and table names for the
purchase (GRN) build.

Directories are resolved at CALL time from app.core.config.settings (never captured at
import), so tests can point a build at tmp folders and never touch the real app/data.
HCG's raw folder and the reference folder are the formulary build's own.

The scope (which receipts every purchase analysis counts) is HCG's out-of-formulary
workings, held as data in app/data/reference/purchase_scope_exclusions.csv so HCG can
change it without a code change:

  kind          value   meaning
  mvt_keep      101     keep only these movement types (101 = goods receipt)
  series_keep   1, 2    keep only material codes starting with these digits (1 onco, 2 other drugs)
  sloc          S125    drop this storage location
  plant_prefix  BH      drop plants whose code starts with this (the Milann units)
  plant         HM03    drop this plant

The filters run in that order (HCG's): the funnel of every month reproduces HCG's sheet
step by step (Jun-26: 43,142 lines -> 42,345 -> 24,941 -> 20,776 -> 19,712 -> 17,657).
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

from app.core.config import settings
from app.etl.formulary import config as FC
from app.etl.formulary.classify import norm_code, norm_text
from app.etl.formulary.discover import BuildError


def raw_dir(d=None) -> Path:
    return FC.raw_dir(d)


def cache_dir(d=None) -> Path:
    return Path(d or settings.PURCHASE_CACHE_DIR).expanduser()


def data_dir(d=None) -> Path:
    return Path(d or settings.PURCHASE_DATA_DIR).expanduser()


def reference_dir(d=None) -> Path:
    return FC.reference_dir(d)


# --- Where HCG's files sit under FORMULARY_RAW_DIR (the top folder has a hash name) -----
# 'GRN Dec 25 to May 2026': one SAP GRN extract per month (it also holds Jun-Aug 26 now, so
# the folder name is not trusted for months: each file's month comes from its GR dates).
GRN_DIR_RE = re.compile(r"^grn\s", re.I)
# 'Out of Formulary Spend Analysis @6-27': HCG's monthly OOF workbooks, and a sub-folder
# 'Out Of formulary Master Sheet (...)' holding HCG's OOF brand -> suggested brand master.
OOF_DIR_RE = re.compile(r"^out\s+of\s+formulary\s+spend\s+analysis", re.I)
OOF_MASTER_DIR_RE = re.compile(r"^out\s+of\s+formulary\s+master", re.I)
# Financial year in the master's file name: '...brand master-2026-27.xlsb'.
FY_IN_NAME_RE = re.compile(r"(20\d\d)\s*-\s*(\d\d)(?!\d)")

# --- GRN extract columns: output column -> header (matched on norm_text: whitespace
# collapsed, trimmed, upper-cased, so '    Base Price' and '     Base Price' agree) --------
# Never read: GR No, PO No, PR No, reference document, Vendor Inv No / Date, Sec Inv No,
# Batch, the GST numbers, Parked / Posted Invoice — no identifier leaves the reader.
GRN_COLUMNS = {
    "plant": "PLANT",
    "sloc": "S.LOC",
    "material": "MATERIAL",
    "material_desc": "MATERIAL DESCRIPTION",
    "vendor_code": "GOODS SUPPLIER",
    "vendor_name": "GOODS SUPPLIER NAME",
    "mvt_type": "MVT TYPE",
    "gr_date": "GR DATE",
    "gr_qty": "GR QTY",
    "uom": "GR UOM",
    "base_price": "BASE PRICE",              # HCG's OOF 'Net price' (before discount)
    "discount_value": "DISCOUNT VALUE",
    "net_price": "NET PRICE",                # after discount
    "total_wo_tax": "TOTAL AMOUNT WITHOUT TAX",
    "tax_pct": "TAX %",
    "unit_mrp": "UNIT MRP",                  # incl GST
    "manufacturer": "MANUFACTURER",
    "material_type": "MATERIAL TYPE",
    "minor_category": "MINOR CATEGORY",
    "major_group": "MAJOR GROUP",
    "minor_group": "MINOR GROUP",
    "grn_formulary": "FORMULARY",             # SAP's own flag: NOT HCG's (the Item Master is)
}
# Optional extra columns a caller may ask for (the legacy curated import keeps them in
# fact_grn). Read when present, None when a file lacks them. Still no identifiers.
GRN_EXTRA_COLUMNS = {
    "po_date": "PO DATE",
    "expiry_date": "EXPIRY DATE",
    "po_to_gr_tat": "PO TO GR TAT",
    "pr_to_gr_tat": "PR TO GR TAT",
    "tax_amount": "TAX AMOUNT",
    "total_with_tax": "TOTAL AMOUNT WITH TAX",
    "total_mrp_value": "TOTAL MRP VALUE W/O FREE",
}
GRN_DATE_COLUMNS = ("gr_date", "po_date", "expiry_date")
GRN_NUMBER_COLUMNS = ("gr_qty", "base_price", "discount_value", "net_price", "total_wo_tax",
                      "tax_pct", "unit_mrp", "po_to_gr_tat", "pr_to_gr_tat", "tax_amount",
                      "total_with_tax", "total_mrp_value")
RECEIPT_MVT = "101"
RETURN_MVTS = ("122", "161")          # return delivery to the vendor; return against the PO
# Bumped whenever the reader's output changes, so every cached file is read again.
GRN_READER_VERSION = 1

# --- HCG's out-of-formulary workbooks (one a month) -----------------------------------
# Sheets, matched on norm_text of the sheet name. HCG's final worksheet is 'WS Final Sheet'
# when the month has one (Jun-26: the WS minus the lines HCG could not price), else the
# 'Out of formulary <Month> 26 WS' sheet.
WS_FINAL_SHEET_RE = re.compile(r"^WS FINAL")
WS_SHEET_RE = re.compile(r"^OUT OF FORMULARY\b.*\bWS\b")
PARKED_SHEET_RE = re.compile(r"^NEED TO CHECK")          # Jul-26 'Need To check flaging later'
SUMMARY_SHEET = "SUMMARY"                                # Jul-26 'Summary ' (trailing space)
# Worksheet header (row 2). The OOF block repeats the GRN headers; the suggested block
# follows the second 'PLANT' (=E): SAP code, SAP NAME, price, MRP incl GST, GST %.
WS_REQUIRED = ("GR DATE", "PLANT", "MATERIAL", "GR QTY", "BASE PRICE", "SAP CODE")
WS_SUG_PRICE_PREFIX = "HCG PHARMA SELLING PRICE"
# Summary header (row 2), with HCG's spellings ('Margi Value', 'Suggesed', 'Differnce').
SUMMARY_CENTRE_RE = re.compile(r"^CENT(ER|RE)$")
SUMMARY_NAME_RE = re.compile(r"^CENT(ER|RE) NAME")
SUMMARY_USED_PP_RE = re.compile(r"USED BRAND PP$")
SUMMARY_USED_MRP_RE = re.compile(r"USED BRAND MRP$")
SUMMARY_SUG_PP_RE = re.compile(r"^SUGG?E\w* FORMULARY BRAND PP$")
SUMMARY_SUG_MRP_RE = re.compile(r"^SUGG?E\w* FORMULARY BRAND MRP$")
SUMMARY_MARGIN_RE = re.compile(r"^MARGI?N? VALUE$")
SUMMARY_LOSS_RE = re.compile(r"^MARGIN VALUE DIFF")
SUMMARY_TOTAL = "GRAND TOTAL"

# --- HCG's OOF master (sub-folder of the OOF folder) ------------------------------------
# Sheet1: a title row, then Material, Material Description, Generic name, (blank),
# MATERIAL, DESCRIPTION, PRICE, MRP, GST — the second MATERIAL is the suggested brand.
OOF_MASTER_REQUIRED = ("MATERIAL", "MATERIAL DESCRIPTION", "GENERIC NAME", "DESCRIPTION",
                       "PRICE", "MRP", "GST")

# --- Scope -----------------------------------------------------------------------------
SCOPE_FILE = "purchase_scope_exclusions.csv"
# Filter kinds in the order they are applied (HCG's order); keep kinds must list >= 1 value.
SCOPE_KINDS = ("mvt_keep", "series_keep", "sloc", "plant_prefix", "plant")
SCOPE_KEEP_KINDS = ("mvt_keep", "series_keep")

# --- Outputs ---------------------------------------------------------------------------
OOF = "oof_purchase"                  # OOF lines aggregated, ours (HCG's method + fallback rule)
HCG_SUMMARY = "oof_hcg_summary"       # HCG's own per-centre Summary figures, per month
HCG_PAIRS = "oof_hcg_pairs"           # HCG's worksheet suggestions: plant x material x base price
HCG_PARKED = "oof_hcg_parked"         # lines HCG parked ('Need To check flaging later')
PRICES = "purchase_prices"            # price variation
GENERIC = "purchase_generic"          # generic vs branded
HCG_TABLES = (HCG_SUMMARY, HCG_PAIRS, HCG_PARKED)    # HCG's own figures, from its OOF workbooks
TABLES = (OOF, *HCG_TABLES, PRICES, GENERIC)
VALIDATION_FILE = "purchase_validation.json"
KR_PLANT = FC.KR_PLANT


@dataclass(frozen=True)
class Scope:
    """The analysis scope read from purchase_scope_exclusions.csv. Values are upper-case
    text: movement types '101', series digits '1', plant / storage-location codes."""
    mvt_keep: frozenset
    series_keep: frozenset
    sloc: frozenset
    plant_prefix: tuple
    plant: frozenset
    rows: tuple = ()                  # (kind, value, reason) as read, for the build report

    def values(self, kind: str):
        """The values of one kind (config.SCOPE_KINDS)."""
        return getattr(self, kind)

    def report(self) -> list:
        return [{"kind": k, "value": v, "reason": r} for k, v, r in self.rows]


def load_scope(ref_dir=None) -> Scope:
    """purchase_scope_exclusions.csv (kind, value, reason) -> Scope. The file must exist and
    list at least one movement type and one material series to keep: an empty keep list
    would silently widen every analysis to all receipts."""
    path = reference_dir(ref_dir) / SCOPE_FILE
    if not path.exists():
        raise BuildError(f"purchase scope file not found: {path}")
    got: dict = {k: [] for k in SCOPE_KINDS}
    rows, seen = [], set()
    with open(path, newline="", encoding="utf-8") as f:
        for n, row in enumerate(csv.DictReader(f), start=2):
            where = f"{path.name} line {n}"
            kind = (row.get("kind") or "").strip().lower()
            if kind not in SCOPE_KINDS:
                raise BuildError(f"{where}: kind must be one of {', '.join(SCOPE_KINDS)}")
            raw = row.get("value")
            value = norm_code(raw) if kind == "mvt_keep" else norm_text(raw)
            if not value:
                raise BuildError(f"{where}: needs a value")
            if kind == "series_keep" and not re.fullmatch(r"\d", value):
                raise BuildError(f"{where}: a material series is one digit, not {value!r}")
            if (kind, value) in seen:
                raise BuildError(f"{where}: {kind} {value} listed twice")
            seen.add((kind, value))
            got[kind].append(value)
            rows.append((kind, value, (row.get("reason") or "").strip()))
    empty = [k for k in SCOPE_KEEP_KINDS if not got[k]]
    if empty:
        raise BuildError(f"{path.name}: needs at least one row of kind {' and '.join(empty)}")
    return Scope(mvt_keep=frozenset(got["mvt_keep"]), series_keep=frozenset(got["series_keep"]),
                 sloc=frozenset(got["sloc"]), plant_prefix=tuple(got["plant_prefix"]),
                 plant=frozenset(got["plant"]), rows=tuple(rows))
