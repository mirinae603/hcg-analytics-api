"""Ingest raw HCG SAP Excel extracts into normalized curated fact tables.

Each loader returns a tidy DataFrame with canonical snake_case columns and
coerced dtypes. Header normalization strips the leading/trailing whitespace
present in the raw exports (e.g. ' Aging', '   Total Cost').
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

from app.core.config import settings

RAW = Path(settings.RAW_DATA_DIR)

# Sub-folders inside Bidezy-2
DIR_CONSUMPTION = "Consumption Dec 25 to May 26"
DIR_INVENTORY = "Inventory report as on 31 May 2026"
DIR_GRN = "GRN Dec 25 to May 2026"
DIR_PO = "PO details Dec 25 to May 26"

MONTH_NAME = {
    1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
    7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December",
}


def _find(folder: str, pattern: str = "*.xlsx") -> list[str]:
    return sorted(glob.glob(str(RAW / folder / pattern)))


def _norm_headers(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _clean_id(s: pd.Series) -> pd.Series:
    """Normalize numeric-ID columns: floats like '100301.0' -> '100301'.

    SAP codes are read inconsistently as int/float/str across files; without this
    a float-read code ('100301.0') won't join to an int-read one ('100301')."""
    out = s.astype(str).str.strip()
    out = out.str.replace(r"\.0$", "", regex=True)
    return out


def _add_period(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    d = pd.to_datetime(df[date_col], errors="coerce")
    df["posting_date"] = d
    df["year"] = d.dt.year.astype("Int64")
    df["month_num"] = d.dt.month.astype("Int64")
    df["month"] = df["month_num"].map(lambda m: MONTH_NAME.get(int(m)) if pd.notna(m) else None)
    return df


# --------------------------------------------------------------------------- #
# Consumption
# --------------------------------------------------------------------------- #
def load_consumption() -> pd.DataFrame:
    files = _find(DIR_CONSUMPTION)
    if not files:
        raise FileNotFoundError(f"No consumption file under {RAW / DIR_CONSUMPTION}")
    df = pd.read_excel(files[0], engine="openpyxl")
    df = _norm_headers(df)

    out = pd.DataFrame({
        "material": _clean_id(df["Material"]),
        "material_desc": df["Material Description"].astype(str).str.strip(),
        "plant": df["Plnt"].astype(str).str.strip(),
        "sloc": df["SLoc"].astype(str).str.strip(),
        "mvt": _to_num(df["MvT"]),
        "cost_ctr": _clean_id(df["Cost Ctr"]),
        "qty": _to_num(df["Quantity"]),
        "amount_lc": _to_num(df["    Amount in LC"]) if "    Amount in LC" in df.columns else _to_num(df["Amount in LC"]),
        "uom": df["EUn"].astype(str).str.strip(),
    })
    d = pd.to_datetime(df["Pstng Date"], errors="coerce")
    out["posting_date"] = d
    out["year"] = d.dt.year.astype("Int64")
    out["month_num"] = d.dt.month.astype("Int64")
    out["month"] = out["month_num"].map(lambda m: MONTH_NAME.get(int(m)) if pd.notna(m) else None)

    # Data-quality (Part-1 finding #11): keep only internal goods-issue (MvT 201),
    # drop rows with unparseable posting date, qty<=0 or null amount.
    before = len(out)
    out = out[(out["mvt"] == 201) & out["posting_date"].notna()].copy()
    out = out[out["qty"].fillna(0) > 0]
    dropped = before - len(out)
    print(f"[ingest] consumption: kept {len(out):,} / {before:,} rows (dropped {dropped:,})")
    return out


# --------------------------------------------------------------------------- #
# Inventory snapshot
# --------------------------------------------------------------------------- #
def load_inventory() -> pd.DataFrame:
    files = _find(DIR_INVENTORY)
    if not files:
        raise FileNotFoundError(f"No inventory file under {RAW / DIR_INVENTORY}")
    df = pd.read_excel(files[0], engine="openpyxl")
    df = _norm_headers(df)

    def g(name, *alts):
        for n in (name, *alts):
            if n in df.columns:
                return df[n]
        return pd.Series([np.nan] * len(df))

    out = pd.DataFrame({
        "plant": g("Plant").astype(str).str.strip(),
        "sloc": g("Storage Location").astype(str).str.strip(),
        "sloc_desc": g("Storage Location Description").astype(str).str.strip(),
        "material": _clean_id(g("Material")),
        "material_desc": g("Material Desc").astype(str).str.strip(),
        "hsn": g("HSN Code").astype(str).str.strip(),
        "batch": g("Batch").astype(str).str.strip(),
        "qty": _to_num(g("    Quantity", "Quantity")),
        "uom": g("Basic UOM").astype(str).str.strip(),
        "total_cost": _to_num(g("   Total Cost", "Total Cost")),
        "total_mrp": _to_num(g("    Total MRP", "Total MRP")),
        "moving_avg_price": _to_num(g("Moving Average Price(Unit)")),
        "aging_days": _to_num(g(" Aging", "Aging")),
        "manufacturer_desc": g("Manufacturer Description").astype(str).str.strip(),
        "generic_name": g("Generic Name").astype(str).str.strip(),
        "formulary": g("Formulary").astype(str).str.strip(),
        "material_type": g("Material Type").astype(str).str.strip(),
        "material_group": g("Material Group").astype(str).str.strip(),
        "major_group_desc": g("Major Group Description").astype(str).str.strip(),
        "minor_group_desc": g("Minor Group Description").astype(str).str.strip(),
    })
    out["expiry_date"] = pd.to_datetime(g("Expiry Date"), errors="coerce")
    out["grn_date"] = pd.to_datetime(g("GRN Date"), errors="coerce")
    out["snapshot_date"] = pd.to_datetime(g("Date"), errors="coerce")
    print(f"[ingest] inventory: {len(out):,} rows, snapshot={out['snapshot_date'].max()}")
    return out


# --------------------------------------------------------------------------- #
# GRN (6 monthly files)
# --------------------------------------------------------------------------- #
def load_grn() -> pd.DataFrame:
    files = _find(DIR_GRN)
    if not files:
        raise FileNotFoundError(f"No GRN files under {RAW / DIR_GRN}")
    frames = []
    for f in files:
        d = _norm_headers(pd.read_excel(f, engine="openpyxl"))
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)

    out = pd.DataFrame({
        "plant": df["Plant"].astype(str).str.strip(),
        "sloc": df["S.Loc"].astype(str).str.strip(),
        "material": _clean_id(df["Material"]),
        "material_desc": df["Material Description"].astype(str).str.strip(),
        "vendor_code": _clean_id(df["Goods supplier"]),
        "vendor_name": df["Goods supplier name"].astype(str).str.strip(),
        "po_no": _clean_id(df["PO No"]),
        "gr_no": _clean_id(df["GR No"]),
        "gr_qty": _to_num(df["GR Qty"]),
        "base_price": _to_num(df["    Base Price"]) if "    Base Price" in df.columns else _to_num(df.get("Base Price")),
        "net_price": _to_num(df["     Net Price"]) if "     Net Price" in df.columns else _to_num(df.get("Net Price")),
        "unit_mrp": _to_num(df["      Unit MRP"]) if "      Unit MRP" in df.columns else _to_num(df.get("Unit MRP")),
        "total_amount_wo_tax": _to_num(df.get("Total Amount without tax")),
        "po_to_gr_tat": _to_num(df["PO to GR TAT"]),
        "pr_to_gr_tat": _to_num(df["PR to GR TAT"]),
        "major_group": df.get("Major Group", pd.Series([np.nan] * len(df))).astype(str).str.strip(),
        "minor_group": df.get("Minor Group", pd.Series([np.nan] * len(df))).astype(str).str.strip(),
    })
    out["po_date"] = pd.to_datetime(df["PO Date"], errors="coerce")
    out["gr_date"] = pd.to_datetime(df["GR Date"], errors="coerce")
    out["expiry_date"] = pd.to_datetime(df["Expiry Date"], errors="coerce")
    out = _add_period(out, "gr_date")
    out = out[out["material"].str.lower() != "nan"].copy()
    print(f"[ingest] grn: {len(out):,} rows from {len(files)} files")
    return out


# --------------------------------------------------------------------------- #
# PO (6 monthly files)
# --------------------------------------------------------------------------- #
def load_po() -> pd.DataFrame:
    files = _find(DIR_PO)
    if not files:
        raise FileNotFoundError(f"No PO files under {RAW / DIR_PO}")
    frames = []
    for f in files:
        d = _norm_headers(pd.read_excel(f, engine="openpyxl"))
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)

    out = pd.DataFrame({
        "plant": df["Plant"].astype(str).str.strip(),
        "plant_name": df["Plant Name"].astype(str).str.strip(),
        "sloc": df["Storage Location"].astype(str).str.strip(),
        "material": _clean_id(df["Material"]),
        "material_desc": df["Material Description"].astype(str).str.strip(),
        "vendor_code": _clean_id(df["PO Vendor"]),
        "vendor_name": df["Vendor Name"].astype(str).str.strip(),
        "po_no": _clean_id(df["PO No"]),
        "doc_type": df["Doc.Type description"].astype(str).str.strip(),
        "po_qty": _to_num(df["PO Quantity"]),
        "open_qty": _to_num(df["Open Qty"]),
        "net_price": _to_num(df["Net Price"]),
        "total_value_wo_tax": _to_num(df["Total value without Tax"]),
        "total_value_tax": _to_num(df.get("Total value with Tax")),
        "major_group": df.get("Major Group", pd.Series([np.nan] * len(df))).astype(str).str.strip(),
        "minor_group": df.get("Minor Group", pd.Series([np.nan] * len(df))).astype(str).str.strip(),
    })
    # PO dates are strings dd.mm.yyyy
    out["po_date"] = pd.to_datetime(df["PO Date"], format="%d.%m.%Y", errors="coerce")
    out["delivery_date"] = pd.to_datetime(df["Delivery Date"], format="%d.%m.%Y", errors="coerce")
    out = _add_period(out, "po_date")
    out = out[out["material"].str.lower() != "nan"].copy()
    print(f"[ingest] po: {len(out):,} rows from {len(files)} files")
    return out
