"""Paths, file patterns, SCHEME sheet headers and the reference figures for the spend build.

Directories are resolved at CALL time from app.core.config.settings (never captured at
import), so tests can point a build at tmp folders and never touch the real app/data.
The raw folder, line cache and reference folder are the formulary build's own.
"""
from __future__ import annotations

import re
from pathlib import Path

from app.core.config import settings


def data_dir(d=None) -> Path:
    return Path(d or settings.SPEND_DATA_DIR).expanduser()


# --- HCG's Main Sheet, next to the Payer Master in the dashboard-data folder -----------
# 'Copy of Onco Formulary ,Non Onco Formulary and Consumbales Complaince  2026-27 -Spend
# Analysis Main Sheet_11.08.26.xlsx'
MAIN_SHEET_FILE_RE = re.compile(r"spend analysis", re.I)
SCHEME_SHEET = "SCHEME"
CASH_SHEET = "CASH&TPA"
# Title cell above the header: 'FORMULARY FOR THE YEAR 2026-27 ( ONCO - 26-27 ...'
FY_RE = re.compile(r"YEAR\s*(20\d\d)\s*-\s*(\d\d)")

# SCHEME headers (normalised). The used brand comes first, the suggested brand after it,
# and both blocks reuse the names SAP CODE / SAP NAME: the first occurrence is the used
# brand, the second the suggested one.
H_CODE, H_NAME = "SAP CODE", "SAP NAME"
H_GROUP = "CATEGORY"                     # SCHEME: consumable group (SUTURES, Endo, ...)
H_USED_PRICE_RE = re.compile(r"^\d{4}-\d{2} HCG PHARMA SELLING PRICE")   # FY price, excl GST
H_USED_MRP_RE = re.compile(r"^\d{4}-\d{2} MRP")                          # FY MRP, incl GST
H_USED_GST_RE = re.compile(r"^\d{4}-\d{2} GST")
H_USED_CATEGORY_RE = re.compile(r"^\d{4}-\d{2} CATEGORY$")               # P / R / P-SCH ...
H_SHARING_RE = re.compile(r"^\d{4}-\d{2} SHARING$")
H_CASH_MARKER = "UNIT MARGIN VALUE"
H_ATTIS = "ATTIS BRAND/OTHER BRANDS"
H_SUGGESTED_PRICE = "HCG PHARMA SELLING PRICE"                           # prefix, no year
ATTIS_BRAND = "ATTIS BRAND"
# Attis products in the sales lines, by HIS MANUFACTURERNAME ('ATTIS GLOBAL PHARMA').
ATTIS_MANUFACTURER_RE = r"(^|[^A-Z])ATTIS([^A-Z]|$)"

# HCG's 'J&J & Covidien vs Healthium' line = the SCHEME rows of these two groups (not
# 'ENDO SURGERY ACCESSORIES'). Compared upper-case.
HEALTHIUM_GROUPS = ("SUTURES", "ENDO")

# Which cash/TPA brands HCG bifurcates, and each one's molecule group (not in the master).
BIFURCATION_FILE = "cash_tpa_bifurcation.csv"
# Cash/TPA list brands HCG's own report does not count (not in HCG's KR Aug-26 CASH&TPA sheet
# although KR sold them). They still count inside other brands' molecule groups.
EXCLUDED_FILE = "cash_tpa_excluded.csv"

# --- Outputs -------------------------------------------------------------------------
SWITCH = "scheme_switch"
ATTIS_USE = "attis_consumption"
CASH_SWITCH = "cash_switch"
CASH_ALLOWANCE = "cash_allowance"
BRAND_BLOCKS = "brand_blocks"          # Task 4: the CASH&TPA sheet's groups of interchangeable brands
OOF_SWITCH = "oof_switch"              # Task 3c: out-of-formulary brands -> a formulary brand (app/etl/spend/oof.py)
# A formulary brand stands in for an out-of-formulary one only if its MRP is within this factor
# of the OOF brand's MRP per unit on the bills (10 = from a tenth to ten times). Wide on purpose:
# a biosimilar at a third of the formulary brand's MRP is a real switch (that gap IS the margin);
# liposomal vs plain doxorubicin (~15x) or a per-strip vs per-unit price is not.
OOF_PRICE_BAND = 10.0
TABLES = (SWITCH, ATTIS_USE, CASH_SWITCH, CASH_ALLOWANCE, BRAND_BLOCKS, OOF_SWITCH)
UNDATED = (BRAND_BLOCKS,)              # FY master data: replaced whole, not per month
VALIDATION_FILE = "spend_validation.json"
SCHEME_PAYER = "GOV"
CASH_PAYERS = ("CASH", "CASH(TPA)")
