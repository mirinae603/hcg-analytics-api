"""Build dimension/lookup tables from the curated facts."""
from __future__ import annotations

import re
import pandas as pd


def _ip_op_class(desc: str) -> str:
    if not isinstance(desc, str):
        return "OTHER"
    u = desc.upper()
    has_ip = bool(re.search(r"\bIP\b|IP[-/ ]", u)) or u.startswith("IP")
    has_op = bool(re.search(r"\bOP\b|OP[-/ ]", u)) or u.startswith("OP")
    if has_ip and has_op:
        return "IP/OP"
    if has_ip:
        return "IP"
    if has_op:
        return "OP"
    if "OT" in u:
        return "OT"
    return "OTHER"


def build_dim_plant(po: pd.DataFrame, inventory: pd.DataFrame, consumption: pd.DataFrame) -> pd.DataFrame:
    # Hospital names come from PO 'Plant Name'; union with all plant codes seen anywhere.
    names = (
        po[["plant", "plant_name"]]
        .dropna()
        .query("plant_name != 'nan' and plant != 'nan'")
        .drop_duplicates("plant")
    )
    all_codes = pd.Index(
        pd.concat([po["plant"], inventory["plant"], consumption["plant"]]).dropna().unique()
    )
    all_codes = all_codes[all_codes.astype(str).str.lower() != "nan"]
    dim = pd.DataFrame({"plant": sorted(all_codes)})
    dim = dim.merge(names, on="plant", how="left")
    dim["plant_name"] = dim["plant_name"].fillna(dim["plant"])
    return dim


def build_dim_material(inventory: pd.DataFrame, consumption: pd.DataFrame) -> pd.DataFrame:
    inv = inventory[[
        "material", "material_desc", "material_group", "major_group_desc",
        "minor_group_desc", "generic_name", "manufacturer_desc", "formulary", "material_type",
    ]].drop_duplicates("material")
    cons = consumption[["material", "material_desc"]].drop_duplicates("material")
    dim = cons.merge(inv, on="material", how="outer", suffixes=("", "_inv"))
    dim["material_desc"] = dim["material_desc"].fillna(dim.get("material_desc_inv"))
    if "material_desc_inv" in dim.columns:
        dim = dim.drop(columns=["material_desc_inv"])
    return dim


def build_dim_sloc(inventory: pd.DataFrame) -> pd.DataFrame:
    dim = (
        inventory[["plant", "sloc", "sloc_desc"]]
        .dropna()
        .drop_duplicates(["plant", "sloc"])
        .copy()
    )
    dim["ip_op_class"] = dim["sloc_desc"].map(_ip_op_class)
    return dim


def build_dim_vendor(po: pd.DataFrame, grn: pd.DataFrame) -> pd.DataFrame:
    a = po[["vendor_code", "vendor_name"]].dropna().drop_duplicates("vendor_code")
    b = grn[["vendor_code", "vendor_name"]].dropna().drop_duplicates("vendor_code")
    dim = pd.concat([a, b], ignore_index=True).drop_duplicates("vendor_code")
    dim = dim[dim["vendor_code"].astype(str).str.lower() != "nan"]
    return dim


def build_dim_costcenter(consumption: pd.DataFrame) -> pd.DataFrame:
    # Names not supplied by HCG yet -> name == code (pending master, see gap log).
    codes = consumption["cost_ctr"].dropna().unique()
    dim = pd.DataFrame({"cost_ctr": sorted(c for c in codes if str(c).lower() != "nan")})
    dim["department_name"] = dim["cost_ctr"]
    return dim
