"""Paths, file patterns, column names and thresholds for the formulary build.

Directories are resolved at CALL time from app.core.config.settings (never captured at
import), so tests can point a build at tmp folders and never touch the real app/data.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from app.core.config import settings


def raw_dir(d=None) -> Path:
    return Path(d or settings.FORMULARY_RAW_DIR).expanduser()


def cache_dir(d=None) -> Path:
    return Path(d or settings.FORMULARY_CACHE_DIR).expanduser()


def data_dir(d=None) -> Path:
    return Path(d or settings.FORMULARY_DATA_DIR).expanduser()


def reference_dir(d=None) -> Path:
    return Path(d or settings.REFERENCE_DIR).expanduser()


# --- Where HCG's files sit under FORMULARY_RAW_DIR (the top folder has a hash name) ---
DASHBOARD_DIR_RE = re.compile(r"^formulary compliance", re.I)
SALES_DIR_RE = re.compile(r"^ip\s*(and|&)\s*op\s+sales$", re.I)
ITEM_MASTER_DIR_RE = re.compile(r"^item master", re.I)
PAYER_MASTER_FILE_RE = re.compile(r"payer master", re.I)
EXCEL_SUFFIXES = (".xlsx", ".xlsb")

# Month in a file name: 'June 26', 'July 2026', 'JUne-26', 'August -26'.
MONTHS = {m: i + 1 for i, m in enumerate(
    ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"))}
MONTH_IN_NAME_RE = re.compile(
    r"(?<![a-z])(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[\s._-]*(\d{4}|\d{2})(?!\d)",
    re.I)

# --- Sales workbook columns (matched trimmed and case-insensitively) -------------------
# output column -> source header. Only these are read; patient columns (UHID, PATNAME,
# IP_OPNUMBER), BATCHCODE, ISSUECODE, EXPIRYDATE and the full invoice number never leave
# the reader.
SALES_COLUMNS = {
    "patient_type": "TYPE",
    "hospital_name": "LOCATIONNAME",
    "his_code": "SALESINVOICENO",          # only the prefix before the first '-' is kept
    "sales_date": "SALESDATE",
    "bill_category": "BILLCATEGORY",
    "doctor_id": "TREATINGDOCID",
    "doctor_name": "TREATINGDOCNAME",
    "material": "MATERIALCODE",
    "material_desc": "MATERIALLINEITEM",
    "brand_name": "BRANDNAME",
    "generic_name": "GENERICNAME",
    "qty": "QTY",
    "total_mrp": "TOTALMRP",
    "total_cost": "TOTALCOSTPRICE",
    "payer1": "PAYER1",
    "manufacturer": "MANUFACTURERNAME",
}
# HCG's own lookup columns in the worksheet — read ONLY to check our lookup against.
HCG_CATEGORY_RE = re.compile(r"^FY \d\d-\d\d - CATEGORY$", re.I)
HCG_PAYER_HEADER = "PAYER AS PER UPDATED MASTER SHEET AS PER CREDIT CONTROLE TEAM-NEW"
HCG_SEGMENT_HEADER = "ONCO/NON ONCO"

# --- Masters -------------------------------------------------------------------------
ITEM_MASTER_SHEET = "Sheet1"
PAYER_MASTER_SHEET = "Sheet1"
PAYER_BUCKET_PREFIX = "FINAL AFTER DISCUSSION"   # 'Final after Discussion 5.2.2026'

HOSPITAL_MAP_FILE = "hospital_map.csv"
PAYER_OVERRIDES_FILE = "payer_overrides.csv"
# HCG's brand rules (Task 4): which class x payer x patient type each measure counts.
BRAND_RULES_FILE = "brand_rules.csv"

# --- Outputs -------------------------------------------------------------------------
SUMMARY = "formulary_summary"
BY_DOCTOR = "formulary_by_doctor"
BY_ITEM = "formulary_by_item"
DOCTOR_ITEM = "formulary_doctor_item"
RULE_BREACH = "formulary_rule_breach"     # Task 4: brand-rule measures per doctor x material
RULE_BOOK = "formulary_rules"             # Task 4: every formulary item of each month's master
TABLES = (SUMMARY, BY_DOCTOR, BY_ITEM, DOCTOR_ITEM, RULE_BREACH, RULE_BOOK)
VALIDATION_FILE = "formulary_validation.json"

# --- HCG's monthly PAN formulary-compliance MIS (app/etl/formulary/hcg_mis.py) ----------
# <dashboard>/HCG MIS/'Onco and Non Onco Formulary Compliance Sep 26 - MIS.xlsx', one a
# month; each holds the FY to date and the history, so only the latest is read.
MIS_DIR_RE = re.compile(r"^hcg\s*mis$", re.I)
MIS_FILE_RE = re.compile(r"formulary\s+compl\w*.*\bmis\b", re.I)
MIS_FY_SHEET_RE = re.compile(r"^FY\s*\d{2}\s*-\s*\d{2}\s+FORMULARY COMPLIANCE")   # on norm_text
MIS_HISTORY_SHEETS = {"ONCO FORMULARY CONSUMPTION": "onco", "NON ONCO FORMULARY CONSUMPTION": "non-onco"}
MIS_UNITS = "hcg_mis_units"               # one row per unit: HCG's name, region, PAN order, remark
MIS_COMPLIANCE = "hcg_mis_compliance"     # unit x month: budget, HCG's printed actual, its formula
MIS_MONTHLY = "hcg_mis_monthly"           # unit x month x segment: HCG's rupee measures
MIS_PAN = "hcg_mis_pan"                   # month: HCG's printed PAN %, and recomputed
MIS_TABLES = (MIS_UNITS, MIS_COMPLIANCE, MIS_MONTHLY, MIS_PAN)
MIS_VALIDATION_FILE = "hcg_mis_validation.json"

# Never allowed in a committed table (checked before publishing).
FORBIDDEN_COLUMNS = frozenset({
    "UHID", "PATNAME", "PATIENTNAME", "IP_OPNUMBER", "OPNUMBER", "SALESINVOICENO", "BILLNO",
    "BATCHCODE", "BATCHSERIALCODE", "ISSUECODE", "EXPIRYDATE", "SALES_DATE", "SALESDATE",
    "PAYER1ID",
})

UNKNOWN_DOCTOR_ID = "UNKNOWN"
UNKNOWN_DOCTOR_NAME = "Not recorded"
KR_PLANT = "HC05"                      # HCG KR, Bangalore (HIS KABHK): the reference hospital

# --- Abort thresholds -----------------------------------------------------------------
MAX_COST_DIFF = 1.0                    # rupees, published vs read
MIN_DRUGS_IN_MASTER_PCT = 99.9
MIN_CATEGORY_AGREEMENT_PCT = 99.5

CHUNK_ROWS = 100_000
DUCKDB_MEMORY_LIMIT = "1GB"
DUCKDB_THREADS = 4


def period_label(period: str) -> str:
    """'2026-08' -> 'Aug-26' (HCG's own spelling)."""
    y, m = period.split("-")
    return f"{list(MONTHS)[int(m) - 1].title()}-{y[2:]}"


def month_from_name(name: str) -> Optional[str]:
    """'IP & OP Consolidated File August 26.xlsx' -> '2026-08'. None when the name has no
    month or more than one."""
    found = {(int(y) + (2000 if len(y) == 2 else 0), MONTHS[m.lower()[:3]])
             for m, y in MONTH_IN_NAME_RE.findall(Path(name).stem)}
    if len(found) != 1:
        return None
    y, m = found.pop()
    return f"{y:04d}-{m:02d}"
