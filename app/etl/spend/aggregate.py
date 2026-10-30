"""One month of cached sales lines -> the spend tables.

scheme_switch: GOV-billed lines (IP and OP, every category) whose material has a suggested
brand in HCG's SCHEME list, summed per hospital, patient type, doctor and material. Values
are HCG's: qty x FY price-list price of the used brand, and qty x the suggested brand's
price — never the bill's TOTALCOSTPRICE, which is kept alongside as actual_cost. Rows are
flagged attis when the suggested brand is an Attis brand (Task 2), and carry the list's
consumable group (SUTURES, Endo, ...) for the Healthium view (Task 3b).

cash_switch: CASH and CASH(TPA) lines whose material has a suggested brand in HCG's CASH&TPA
list (Task 3a), per hospital, patient type, payer, doctor and material. counted qty = the
cash/TPA qty, except for a bifurcated brand: there HCG first allows SHARING x the molecule
group's all-payer qty at that hospital in that month, and only the excess counts; the excess
is shared across the brand's rows in proportion to their qty (cash_allowance keeps the
hospital-month working). Values at FY list prices: actual margin = qty x (MRP excl GST -
price) of the used brand, potential margin = the same for the suggested brand, margin loss =
potential - actual.

attis_consumption: every line of an Attis product (HIS manufacturer ATTIS ...), every
payer, summed per hospital, patient type, payer and material.

The out-of-formulary margin is not a sales table: it follows HCG's purchase (GRN) method in
app/etl/purchase.

Payer bucket and hospital use the formulary build's own lookups, so GOV / CASH / CASH(TPA)
here are exactly those of the formulary tables.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app.etl.formulary import config as FC
from app.etl.formulary.aggregate import _hospital_lookup, _payer_lookup, _r
from app.etl.formulary.discover import BuildError
from app.etl.formulary.extract import cache_paths
from app.etl.spend import config as C

CATEGORY_SQL = ("CASE left(coalesce(x.material, ''), 1) WHEN '1' THEN 'Onco Drugs' "
                "WHEN '2' THEN 'Other Drugs' WHEN '3' THEN 'Consumables' ELSE 'Unclassified' END")


@dataclass
class MonthResult:
    period: str
    tables: dict                                   # table name -> DataFrame
    doctor_names: pd.DataFrame                     # plant, doctor_id, doctor_name, n
    checks: dict = field(default_factory=dict)


def _switch_frame(scheme) -> pd.DataFrame:
    rows = [{"material": m, "s_desc": s.material_desc, "hcg_category": s.hcg_category,
             "scheme_group": s.scheme_group, "suggested_material": s.suggested,
             "suggested_desc": s.suggested_desc, "attis": s.attis, "used_price": s.used_price,
             "suggested_price": s.suggested_price}
            for m, s in scheme.switches.items()]
    return pd.DataFrame(rows, columns=["material", "s_desc", "hcg_category", "scheme_group",
                                       "suggested_material", "suggested_desc", "attis",
                                       "used_price", "suggested_price"])


def _cash_frame(cash) -> pd.DataFrame:
    rows = [{"material": m, "s_desc": s.material_desc, "hcg_category": s.hcg_category,
             "suggested_material": s.suggested, "suggested_desc": s.suggested_desc,
             "used_price": s.used_price, "used_mrp": s.used_mrp, "used_gst": s.used_gst,
             "suggested_price": s.suggested_price, "suggested_mrp": s.suggested_mrp,
             "suggested_gst": s.suggested_gst, "sharing": s.sharing}
            for m, s in cash.switches.items()]
    return pd.DataFrame(rows, columns=["material", "s_desc", "hcg_category", "suggested_material",
                                       "suggested_desc", "used_price", "used_mrp", "used_gst",
                                       "suggested_price", "suggested_mrp", "suggested_gst",
                                       "sharing"])


def _values(df: pd.DataFrame) -> pd.DataFrame:
    """used_value / suggested_value at FY prices. Not rounded: totals are summed first."""
    df = df.copy()
    df["used_value"] = df["qty"].astype(float) * df["used_price"].astype(float)
    df["suggested_value"] = df["qty"].astype(float) * df["suggested_price"].astype(float)
    return df


def _cash_values(df: pd.DataFrame) -> pd.DataFrame:
    """HCG's CASH&TPA columns AC..AI for counted qty (not rounded)."""
    df = df.copy()
    q = df["qty"].astype(float)
    used_mrp_ex = df["used_mrp"].astype(float) / (1 + df["used_gst"].astype(float) / 100)
    sug_mrp_ex = df["suggested_mrp"].astype(float) / (1 + df["suggested_gst"].astype(float) / 100)
    df["used_cost_value"] = q * df["used_price"].astype(float)
    df["used_mrp_value"] = q * used_mrp_ex
    df["suggested_cost_value"] = q * df["suggested_price"].astype(float)
    df["suggested_mrp_value"] = q * sug_mrp_ex
    df["actual_margin"] = df["used_mrp_value"] - df["used_cost_value"]
    df["potential_margin"] = df["suggested_mrp_value"] - df["suggested_cost_value"]
    df["margin_loss"] = df["potential_margin"] - df["actual_margin"]
    return df


def totals(df: pd.DataFrame) -> dict:
    """used / suggested / saving missed (= used - suggested) of scheme rows."""
    used = float(df["used_value"].sum()) if len(df) else 0.0
    sug = float(df["suggested_value"].sum()) if len(df) else 0.0
    return {"used_value": _r(used, 2), "suggested_value": _r(sug, 2),
            "saving_missed": _r(used - sug, 2),
            "saving_pct": _r((used - sug) / used * 100, 4) if used else None,
            "qty": _r(df["qty"].sum(), 3) if len(df) else 0.0,
            "lines": int(df["lines"].sum()) if len(df) else 0,
            "materials": int(df["material"].nunique()) if len(df) else 0}


def cash_totals(df: pd.DataFrame) -> dict:
    """Actual / potential margin, margin loss and HCG's % (loss / potential) of cash rows."""
    s = {k: float(df[k].sum()) if len(df) else 0.0
         for k in ("actual_margin", "potential_margin", "margin_loss", "used_mrp_value",
                   "suggested_mrp_value", "qty", "cash_qty")}
    return {"actual_margin": _r(s["actual_margin"], 2), "potential_margin": _r(s["potential_margin"], 2),
            "margin_loss": _r(s["margin_loss"], 2),
            "loss_pct": _r(s["margin_loss"] / s["potential_margin"] * 100, 4) if s["potential_margin"] else None,
            "used_mrp_value": _r(s["used_mrp_value"], 2),
            "suggested_mrp_value": _r(s["suggested_mrp_value"], 2),
            "topline_increase": _r(s["suggested_mrp_value"] - s["used_mrp_value"], 2),
            "qty": _r(s["qty"], 3), "cash_qty": _r(s["cash_qty"], 3),
            "lines": int(df["lines"].sum()) if len(df) else 0,
            "materials": int(df["material"].nunique()) if len(df) else 0}


def healthium(df: pd.DataFrame) -> pd.DataFrame:
    """Scheme rows of HCG's 'J&J & Covidien vs Healthium' line (SUTURES + Endo groups)."""
    if "scheme_group" not in df.columns:
        return df.iloc[0:0]
    g = df["scheme_group"].fillna("").astype(str).str.strip().str.upper()
    return df[g.isin(C.HEALTHIUM_GROUPS)]


def _allowance(con, cash_rows: pd.DataFrame, cash_list: pd.DataFrame, bifurcation: dict,
               period: str) -> tuple:
    """(cash_allowance rows, {(plant, material): factor}) for the bifurcated brands.

    Per hospital: allowed = SHARING x the molecule group's all-payer qty; excess = cash/TPA
    qty - allowed, never below 0; factor = excess / cash/TPA qty (0 when that is <= 0)."""
    bif = {m: v for m, v in bifurcation.items() if m in set(cash_list["material"])}
    cols = ["period", "plant", "hospital", "hospital_name", "material", "material_desc", "flag",
            "sharing", "group_qty", "allowed_qty", "cash_qty", "excess_qty"]
    if not bif or cash_rows.empty:
        return pd.DataFrame(columns=cols), {}
    members = pd.DataFrame([{"brand": m, "member": g} for m, (_, grp) in bif.items() for g in grp])
    con.register("grp_df", members)
    group_qty = con.execute("""
        SELECT l.plant, g.brand AS material, sum(l.qty) AS group_qty
        FROM l JOIN grp_df g ON g.member = l.material GROUP BY ALL""").df()
    used = cash_rows[cash_rows["material"].isin(bif)]
    m = (used.groupby(["plant", "hospital", "hospital_name", "material"], as_index=False)
             .agg(cash_qty=("qty", "sum")))
    m = m.merge(group_qty, on=["plant", "material"], how="left")
    m = m.merge(cash_list[["material", "s_desc", "sharing"]], on="material", how="left")
    no_share = sorted(set(m.loc[m["sharing"].isna(), "material"]))
    if no_share:
        raise BuildError(f"{period}: bifurcated brands without a SHARING value in the CASH&TPA list: "
                         f"{', '.join(no_share)}")
    m["group_qty"] = m["group_qty"].fillna(0.0).astype(float)
    m["allowed_qty"] = m["group_qty"] * m["sharing"].astype(float)
    m["excess_qty"] = (m["cash_qty"].astype(float) - m["allowed_qty"]).clip(lower=0.0)
    m["flag"] = m["material"].map(lambda x: bif[x][0])
    m["period"] = period
    factor = {(p, mat): (ex / q if q > 0 else 0.0)
              for p, mat, ex, q in zip(m["plant"], m["material"], m["excess_qty"], m["cash_qty"])}
    return m.rename(columns={"s_desc": "material_desc"})[cols], factor


def aggregate_month(period: str, *, cache, scheme, cash, bifurcation: dict, payer_master,
                    overrides: dict, hospitals: dict, con) -> MonthResult:
    src_path, _ = cache_paths(cache, period)
    src = str(src_path).replace("'", "''")
    pay = _payer_lookup(con, period, payer_master, overrides, src)
    hosp, unmapped, _ = _hospital_lookup(con, hospitals, src)
    if unmapped:
        raise BuildError(
            f"{period}: hospital codes not in hospital_map.csv: "
            + ", ".join(f"{u['his_code']} ({u['location_name']}, {u['lines']:,} lines)"
                        for u in unmapped)
            + " — add them to app/data/reference/hospital_map.csv")
    con.register("pay_df", pay)
    con.register("hosp_df", hosp)
    con.register("sw_df", _switch_frame(scheme))
    cash_list = _cash_frame(cash)
    con.register("cash_df", cash_list)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE l AS
        SELECT x.period, h.plant, x.his_code AS hospital, h.hosp_name AS hospital_name,
               x.patient_type, {CATEGORY_SQL} AS category, p.payer_bucket,
               CASE WHEN x.doctor_id IS NULL THEN '{FC.UNKNOWN_DOCTOR_ID}' ELSE x.doctor_id END
                   AS doctor_id,
               CASE WHEN x.doctor_id IS NULL THEN NULL ELSE x.doctor_name END AS doctor_name,
               coalesce(x.material, '') AS material, x.material_desc, x.generic_name,
               x.manufacturer, coalesce(x.total_cost, 0) AS cost,
               coalesce(x.total_mrp, 0) AS mrp, coalesce(x.qty, 0) AS qty
        FROM read_parquet('{src}') x
        JOIN pay_df p ON p.pkey = coalesce(x.payer1, '')
        JOIN hosp_df h ON h.his_code = coalesce(x.his_code, '')""")

    switch = con.execute(f"""
        SELECT l.period, l.plant, l.hospital, l.hospital_name, l.patient_type, l.category,
               l.doctor_id, l.material, coalesce(s.s_desc, mode(l.material_desc)) AS material_desc,
               s.hcg_category, s.scheme_group, s.suggested_material, s.suggested_desc, s.attis,
               s.used_price, s.suggested_price,
               sum(l.qty) AS qty, round(sum(l.cost), 2) AS actual_cost, count(*) AS lines
        FROM l JOIN sw_df s ON s.material = l.material
        WHERE l.payer_bucket = '{C.SCHEME_PAYER}'
        GROUP BY l.period, l.plant, l.hospital, l.hospital_name, l.patient_type, l.category,
                 l.doctor_id, l.material, s.s_desc, s.hcg_category, s.scheme_group,
                 s.suggested_material, s.suggested_desc, s.attis, s.used_price,
                 s.suggested_price""").df()
    switch = _values(switch)

    payers = ", ".join(f"'{p}'" for p in C.CASH_PAYERS)
    cash_rows = con.execute(f"""
        SELECT l.period, l.plant, l.hospital, l.hospital_name, l.patient_type, l.category,
               l.payer_bucket, l.doctor_id, l.material,
               coalesce(s.s_desc, mode(l.material_desc)) AS material_desc, s.hcg_category,
               s.suggested_material, s.suggested_desc, s.used_price, s.used_mrp, s.used_gst,
               s.suggested_price, s.suggested_mrp, s.suggested_gst,
               sum(l.qty) AS qty, round(sum(l.cost), 2) AS actual_cost, count(*) AS lines
        FROM l JOIN cash_df s ON s.material = l.material
        WHERE l.payer_bucket IN ({payers})
        GROUP BY l.period, l.plant, l.hospital, l.hospital_name, l.patient_type, l.category,
                 l.payer_bucket, l.doctor_id, l.material, s.s_desc, s.hcg_category,
                 s.suggested_material, s.suggested_desc, s.used_price, s.used_mrp, s.used_gst,
                 s.suggested_price, s.suggested_mrp, s.suggested_gst""").df()
    allowance, factor = _allowance(con, cash_rows, cash_list, bifurcation, period)
    cash_rows = cash_rows.rename(columns={"qty": "cash_qty"})
    f = np.array([factor.get((p, m), 1.0) for p, m in zip(cash_rows["plant"], cash_rows["material"])],
                 dtype=float)
    cash_rows["bifurcated"] = [(p, m) in factor for p, m in zip(cash_rows["plant"], cash_rows["material"])]
    cash_rows["qty"] = cash_rows["cash_qty"].astype(float) * f
    cash_rows = _cash_values(cash_rows)

    attis_use = con.execute(f"""
        SELECT period, plant, hospital, hospital_name, patient_type, category, payer_bucket,
               material, mode(material_desc) AS material_desc, mode(generic_name) AS generic_name,
               round(sum(cost), 2) AS cost, round(sum(mrp), 2) AS mrp, sum(qty) AS qty,
               count(*) AS lines
        FROM l WHERE regexp_matches(upper(coalesce(manufacturer, '')), '{C.ATTIS_MANUFACTURER_RE}')
        GROUP BY period, plant, hospital, hospital_name, patient_type, category, payer_bucket,
                 material""").df()

    doctor_names = con.execute(f"""
        SELECT plant, doctor_id, doctor_name, count(*) AS n
        FROM l WHERE doctor_name IS NOT NULL AND doctor_id <> '{FC.UNKNOWN_DOCTOR_ID}'
        GROUP BY ALL""").df()

    gov = con.execute(f"""
        SELECT count(*), coalesce(sum(cost), 0), coalesce(sum(qty), 0)
        FROM l WHERE payer_bucket = '{C.SCHEME_PAYER}'""").fetchone()
    kr = switch[switch["plant"].astype(str) == FC.KR_PLANT]
    kr_cash = cash_rows[cash_rows["plant"].astype(str) == FC.KR_PLANT]
    checks = {
        "lines": int(con.execute("SELECT count(*) FROM l").fetchone()[0]),
        "gov": {"lines": int(gov[0]), "cost": _r(gov[1], 2), "qty": _r(gov[2], 3)},
        "scheme": totals(switch),
        "attis": totals(switch[switch["attis"]]),
        "healthium": totals(healthium(switch)),
        "cash": cash_totals(cash_rows),
        "bifurcation": {"brands": int(allowance["material"].nunique()),
                        "cash_qty": _r(allowance["cash_qty"].sum(), 3) if len(allowance) else 0.0,
                        "allowed_qty": _r(allowance["allowed_qty"].sum(), 3) if len(allowance) else 0.0,
                        "counted_qty": _r(allowance["excess_qty"].sum(), 3) if len(allowance) else 0.0},
        "kr": {"scheme": totals(kr), "attis": totals(kr[kr["attis"]]),
               "healthium": totals(healthium(kr)), "cash": cash_totals(kr_cash)},
        "attis_consumption": {"lines": int(attis_use["lines"].sum()) if len(attis_use) else 0,
                              "cost": _r(attis_use["cost"].sum(), 2) if len(attis_use) else 0.0,
                              "materials": int(attis_use["material"].nunique())},
    }
    return MonthResult(period, {C.SWITCH: switch, C.ATTIS_USE: attis_use, C.CASH_SWITCH: cash_rows,
                                C.CASH_ALLOWANCE: allowance}, doctor_names, checks)
