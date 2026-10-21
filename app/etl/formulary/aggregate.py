"""Classify one month of cached lines and aggregate them into the committed tables.

Task 4 adds two tables: formulary_rule_breach (the lines each per-doctor brand-rule measure of
brand_rules.csv counts, per doctor x material x patient type x payer) and formulary_rules (the
month's rule book: every FORMULARY item of that month's Item Master with its Category letter).

The classification rules run in Python on the month's DISTINCT keys only (~10k
materials, ~500 payer names, 23 hospitals); DuckDB then joins those small lookups onto
the ~900k cached lines and sums, under a memory limit. Measures: cost = sum
TOTALCOSTPRICE (empty -> 0), mrp = sum TOTALMRP, qty, lines = line count.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import duckdb
import pandas as pd

from app.etl.formulary import classify as K
from app.etl.formulary import config as C
from app.etl.formulary.discover import BuildError
from app.etl.formulary.extract import cache_paths

DRUGS_SQL = "category IN ('Onco Drugs', 'Other Drugs')"


@dataclass
class MonthResult:
    period: str
    tables: dict                                   # table name -> DataFrame
    doctor_names: pd.DataFrame                     # plant, doctor_id, doctor_name, n
    checks: dict = field(default_factory=dict)     # numbers for the validation report


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{C.DUCKDB_MEMORY_LIMIT}'")
    con.execute(f"SET threads={C.DUCKDB_THREADS}")
    con.execute("SET preserve_insertion_order=false")
    return con


def _material_lookup(con, master, src: str) -> pd.DataFrame:
    keys = con.execute(f"""
        SELECT material, mode(material_desc) AS d, mode(generic_name) AS g,
               mode(manufacturer) AS m
        FROM read_parquet('{src}') GROUP BY material""").df()
    rows = []
    for mat, d, g, m in keys.itertuples(index=False):
        it = master.items.get(mat) if mat else None
        cls = it.formulary_class if it else K.NOT_IN_MASTER
        rows.append({
            "material": mat or "",
            "category": K.category(mat),
            "formulary_class": cls,
            "formulary_group": K.formulary_group(cls),
            "m_desc": (it.material_desc if it and it.material_desc else d),
            "m_generic": (it.minor_group if it and it.minor_group else g),
            "m_manufacturer": (it.manufacturer if it and it.manufacturer else m),
        })
    return pd.DataFrame(rows, columns=["material", "category", "formulary_class",
                                       "formulary_group", "m_desc", "m_generic",
                                       "m_manufacturer"])


def _payer_lookup(con, period, payer_master, overrides, src: str) -> pd.DataFrame:
    keys = con.execute(f"SELECT DISTINCT coalesce(payer1, '') AS p FROM read_parquet('{src}')").df()
    rows = []
    for p in keys["p"]:
        bucket, how = K.resolve_payer(p, period, payer_master.names, overrides)
        rows.append({"pkey": p, "payer_bucket": bucket, "payer_how": how,
                     "payer_norm": "" if how == "blank" else K.payer_key(p)})
    return pd.DataFrame(rows, columns=["pkey", "payer_bucket", "payer_how", "payer_norm"])


def _hospital_lookup(con, hospitals: dict, src: str) -> tuple:
    keys = con.execute(f"""
        SELECT coalesce(his_code, '') AS his_code, mode(hospital_name) AS hospital_name,
               count(*) AS n
        FROM read_parquet('{src}') GROUP BY 1""").df()
    rows, unmapped, renamed = [], [], []
    for code, nm, n in keys.itertuples(index=False):
        h = hospitals.get(code)
        if h is None:
            unmapped.append({"his_code": code or None, "location_name": nm, "lines": int(n)})
            continue
        mapped_name = (h.get("his_location_name") or "").strip()
        if mapped_name and nm and mapped_name != nm:
            renamed.append({"his_code": code, "map_name": mapped_name, "data_name": nm})
        rows.append({"his_code": code, "plant": h["sap_plant"], "hosp_name": nm or mapped_name})
    return pd.DataFrame(rows, columns=["his_code", "plant", "hosp_name"]), unmapped, renamed


def _rule_frame(rules) -> pd.DataFrame:
    """One row per (measure, class, payer, patient type) a per-doctor rule counts; '*' = any."""
    rows = [{"measure": r.measure, "formulary_class": c, "payer": p, "ptype": t}
            for r in rules if r.by_doctor for c in sorted(r.classes)
            for p in (sorted(r.payers) if r.payers else ["*"])
            for t in (sorted(r.patient_types) if r.patient_types else ["*"])]
    return pd.DataFrame(rows, columns=["measure", "formulary_class", "payer", "ptype"])


def rule_book(period: str, item_master, sold: set) -> pd.DataFrame:
    """Every FORMULARY item of the month's Item Master: the brand-rule universe of that month."""
    rows = [{"period": period, "material": m, "material_desc": it.material_desc,
             "molecule": it.minor_group, "manufacturer": it.manufacturer,
             "category": K.category(m), "letter": K.rule_letter(it.formulary_class),
             "formulary_class": it.formulary_class, "sold": m in sold}
            for m, it in item_master.items.items()
            if K.formulary_group(it.formulary_class) == "A"]
    return pd.DataFrame(rows, columns=["period", "material", "material_desc", "molecule", "manufacturer",
                                       "category", "letter", "formulary_class", "sold"])


def _hcg_lookup(con, column: str, fn, src: str) -> pd.DataFrame:
    keys = con.execute(f"SELECT DISTINCT coalesce({column}, '') AS k FROM read_parquet('{src}')").df()
    return pd.DataFrame({"k": keys["k"], "v": [fn(k) for k in keys["k"]]})


def _agreement(con, ours: str, lookup: str, column: str, where: str = "TRUE") -> dict:
    """Lines and cost on which our value equals HCG's, overall and per hospital, plus the
    largest disagreeing (ours, HCG) pairs."""
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE agree AS
        SELECT l.hospital, l.cost, {ours} AS ours, h.v AS hcg,
               coalesce({ours} = h.v, FALSE) AS ok
        FROM l JOIN {lookup} h ON coalesce(l.{column}, '') = h.k
        WHERE {where}""")
    per = con.execute("""
        SELECT hospital, count(*) AS n, count(*) FILTER (WHERE ok) AS n_ok,
               sum(cost) AS c, coalesce(sum(cost) FILTER (WHERE ok), 0) AS c_ok
        FROM agree GROUP BY hospital ORDER BY hospital""").fetchall()
    diffs = con.execute("""
        SELECT ours, hcg, count(*) AS n, sum(cost) AS c FROM agree WHERE NOT ok
        GROUP BY ALL ORDER BY c DESC, n DESC LIMIT 10""").fetchall()

    def block(n, n_ok, c, c_ok):
        return {"lines": int(n), "lines_agree": int(n_ok),
                "lines_pct": _r(K.pct(n_ok, n), 4),
                "cost": _r(c, 2), "cost_agree": _r(c_ok, 2), "cost_pct": _r(K.pct(c_ok, c), 4)}

    tot = [sum(x[i] or 0 for x in per) for i in range(1, 5)]
    out = block(*tot)
    out["by_hospital"] = {h: block(n, n_ok, c, c_ok) for h, n, n_ok, c, c_ok in per}
    out["top_differences"] = [{"ours": o, "hcg": h, "lines": int(n), "cost": _r(c, 2)}
                              for o, h, n, c in diffs]
    return out


def _r(v, nd):
    return None if v is None else round(float(v), nd)


def aggregate_month(period: str, *, cache, item_master, payer_master, overrides: dict,
                    hospitals: dict, has_hcg: bool, con=None, rules=()) -> MonthResult:
    src_path, _ = cache_paths(cache, period)
    src = str(src_path).replace("'", "''")
    own = con is None
    con = con or connect()
    try:
        mat = _material_lookup(con, item_master, src)
        pay = _payer_lookup(con, period, payer_master, overrides, src)
        hosp, unmapped, renamed = _hospital_lookup(con, hospitals, src)
        if unmapped:
            raise BuildError(
                f"{period}: hospital codes not in hospital_map.csv: "
                + ", ".join(f"{u['his_code']} ({u['location_name']}, {u['lines']:,} lines)"
                            for u in unmapped)
                + " — add them to app/data/reference/hospital_map.csv")
        con.register("mat_df", mat)
        con.register("pay_df", pay)
        con.register("hosp_df", hosp)
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE l AS
            SELECT x.period, h.plant, x.his_code AS hospital, h.hosp_name AS hospital_name,
                   x.patient_type, m.category, m.formulary_class, m.formulary_group,
                   p.payer_bucket, p.payer_how, p.payer_norm,
                   CASE WHEN x.doctor_id IS NULL THEN '{C.UNKNOWN_DOCTOR_ID}' ELSE x.doctor_id END
                       AS doctor_id,
                   CASE WHEN x.doctor_id IS NULL THEN NULL ELSE x.doctor_name END AS doctor_name,
                   m.material, m.m_desc AS material_desc, m.m_generic AS generic_name,
                   m.m_manufacturer AS manufacturer,
                   coalesce(x.total_cost, 0) AS cost, coalesce(x.total_mrp, 0) AS mrp,
                   coalesce(x.qty, 0) AS qty,
                   x.hcg_category, x.hcg_payer, x.hcg_segment
            FROM read_parquet('{src}') x
            JOIN mat_df m ON m.material = coalesce(x.material, '')
            JOIN pay_df p ON p.pkey = coalesce(x.payer1, '')
            JOIN hosp_df h ON h.his_code = coalesce(x.his_code, '')""")

        summary = con.execute("""
            SELECT period, plant, hospital, hospital_name, patient_type, category,
                   payer_bucket, formulary_class, formulary_group,
                   round(sum(cost), 2) AS cost, round(sum(mrp), 2) AS mrp,
                   sum(qty) AS qty, count(*) AS lines
            FROM l GROUP BY ALL""").df()
        by_doctor = con.execute(f"""
            SELECT period, plant, hospital, hospital_name, patient_type, category,
                   payer_bucket, formulary_group, doctor_id,
                   round(sum(cost), 2) AS cost, sum(qty) AS qty, count(*) AS lines
            FROM l WHERE {DRUGS_SQL} GROUP BY ALL""").df()
        by_item = con.execute(f"""
            SELECT period, plant, hospital, hospital_name, patient_type, payer_bucket,
                   material, material_desc, generic_name, manufacturer, category,
                   formulary_class, formulary_group,
                   round(sum(cost), 2) AS cost, sum(qty) AS qty, count(*) AS lines
            FROM l WHERE {DRUGS_SQL} GROUP BY ALL""").df()
        doctor_item = con.execute(f"""
            SELECT period, plant, hospital, payer_bucket, doctor_id, material, material_desc,
                   formulary_class, formulary_group,
                   round(sum(cost), 2) AS cost, sum(qty) AS qty, count(*) AS lines
            FROM l WHERE {DRUGS_SQL} AND formulary_group IN ('B', 'C') GROUP BY ALL""").df()
        # Task 4: the per-doctor brand-rule measures. A line can count in more than one
        # measure (e.g. GOV R-OPD on an inpatient), so totals are per measure, never summed.
        con.register("rule_df", _rule_frame(rules))
        rule_breach = con.execute(f"""
            SELECT r.measure, l.period, l.plant, l.hospital, l.hospital_name, l.patient_type,
                   l.payer_bucket, l.category, l.doctor_id, l.material, l.material_desc,
                   l.generic_name, l.formulary_class,
                   round(sum(l.cost), 2) AS cost, sum(l.qty) AS qty, count(*) AS lines
            FROM l JOIN rule_df r ON r.formulary_class = l.formulary_class
                 AND (r.payer = '*' OR r.payer = l.payer_bucket)
                 AND (r.ptype = '*' OR r.ptype = l.patient_type)
            WHERE {DRUGS_SQL} GROUP BY ALL""").df()
        book = rule_book(period, item_master, set(mat["material"]))
        # spellings of each doctor's name on drug lines, for the most-frequent-name rule
        doctor_names = con.execute(f"""
            SELECT plant, doctor_id, doctor_name, count(*) AS n
            FROM l WHERE {DRUGS_SQL} AND doctor_name IS NOT NULL GROUP BY ALL""").df()

        checks = {
            "published_lines": int(con.execute("SELECT count(*) FROM l").fetchone()[0]),
            "drugs": dict(zip(
                ("lines", "not_in_master_lines", "not_in_master_cost"),
                con.execute(f"""
                    SELECT count(*), count(*) FILTER (WHERE formulary_class = '{K.NOT_IN_MASTER}'),
                           coalesce(sum(cost) FILTER (WHERE formulary_class = '{K.NOT_IN_MASTER}'), 0)
                    FROM l WHERE {DRUGS_SQL}""").fetchone())),
            "unclassified_lines": int(con.execute(
                f"SELECT count(*) FROM l WHERE category = '{K.UNCLASSIFIED}'").fetchone()[0]),
            "doctor_unknown_drug_lines": int(con.execute(
                f"SELECT count(*) FROM l WHERE {DRUGS_SQL} AND doctor_id = '{C.UNKNOWN_DOCTOR_ID}'"
            ).fetchone()[0]),
            "hospital_names_differ_from_map": renamed,
            "payers": {how: {"lines": int(n), "cost": _r(c, 2), "names": int(k)}
                       for how, n, c, k in con.execute("""
                           SELECT payer_how, count(*), sum(cost), count(DISTINCT payer_norm)
                           FROM l GROUP BY 1 ORDER BY 1""").fetchall()},
            "unmatched_payers": [{"name": nm, "lines": int(n), "cost": _r(c, 2)}
                                 for nm, n, c in con.execute("""
                                     SELECT payer_norm, count(*), sum(cost) FROM l
                                     WHERE payer_how = 'unmatched' GROUP BY 1
                                     ORDER BY 3 DESC, 2 DESC LIMIT 50""").fetchall()],
            "unmatched_payer_names": int(con.execute(
                "SELECT count(DISTINCT payer_norm) FROM l WHERE payer_how = 'unmatched'"
            ).fetchone()[0]),
            "used_conflicting_payers": [
                {"name": nm, "lines": int(n), "cost": _r(c, 2)}
                for nm, n, c in con.execute(
                    "SELECT payer_norm, count(*), sum(cost) FROM l WHERE payer_how = 'master' "
                    "AND payer_norm IN (SELECT unnest(?)) GROUP BY 1 ORDER BY 3 DESC",
                    [[c["name"] for c in payer_master.conflicts] or [""]]).fetchall()],
        }
        if has_hcg:
            con.register("hcg_cat", _hcg_lookup(con, "hcg_category", K.hcg_class, src))
            con.register("hcg_pay", _hcg_lookup(con, "hcg_payer", K.norm_bucket, src))
            con.register("hcg_seg", _hcg_lookup(con, "hcg_segment", K.hcg_segment, src))
            seg_case = ("CASE category WHEN 'Onco Drugs' THEN 'Onco' WHEN 'Other Drugs' "
                        "THEN 'Non-Onco' WHEN 'Consumables' THEN 'Consumables' END")
            checks["agreement"] = {
                "category": _agreement(con, "formulary_class", "hcg_cat", "hcg_category"),
                "payer": _agreement(con, "payer_bucket", "hcg_pay", "hcg_payer"),
                "segment": _agreement(con, seg_case, "hcg_seg", "hcg_segment"),
            }
        return MonthResult(period, {C.SUMMARY: summary, C.BY_DOCTOR: by_doctor,
                                    C.BY_ITEM: by_item, C.DOCTOR_ITEM: doctor_item,
                                    C.RULE_BREACH: rule_breach, C.RULE_BOOK: book},
                           doctor_names, checks)
    finally:
        if own:
            con.close()


def doctor_name_map(counts: pd.DataFrame) -> dict:
    """(plant, doctor_id) -> the most frequent spelling (ties: alphabetical)."""
    if counts.empty:
        return {}
    c = (counts.groupby(["plant", "doctor_id", "doctor_name"], as_index=False)["n"].sum()
               .sort_values(["plant", "doctor_id", "n", "doctor_name"],
                            ascending=[True, True, False, True]))
    c = c.drop_duplicates(["plant", "doctor_id"])
    return {(p, d): n for p, d, n in zip(c["plant"], c["doctor_id"], c["doctor_name"])}


def apply_doctor_names(df: pd.DataFrame, names: dict) -> pd.DataFrame:
    """Write doctor_name next to doctor_id. Blank doctor -> 'Not recorded'."""
    keys = list(zip(df["plant"], df["doctor_id"]))
    df = df.copy()
    pos = list(df.columns).index("doctor_id") + 1
    if "doctor_name" in df.columns:
        df = df.drop(columns="doctor_name")
    df.insert(pos, "doctor_name", [names.get(k) or C.UNKNOWN_DOCTOR_NAME for k in keys])
    return df
