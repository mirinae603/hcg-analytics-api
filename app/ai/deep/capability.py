"""What this warehouse can actually answer, derived from the warehouse itself.

The deep engine plans BEFORE it queries, and a plan is only as good as its picture of what
exists. Handed nothing, a model plans from the table names it can guess at — which is how
"sales trend of KEYTRUDA" became six UNION branches over a table with no month column, and
then company-wide monthly revenue with the material filter silently dropped.

So the first thing the engine does is ask the schema, not the model: which tables carry this
entity, which of those also carry time, what measures they hold. The planner is then choosing
among real options instead of imagining them, and a genuinely unanswerable question is known
to be unanswerable in phase one rather than after three failed attempts.

Everything here reads information_schema at runtime, so a new mart is visible the day it
lands. Nothing is hard-coded to a table.
"""
from __future__ import annotations

import re
from functools import lru_cache

from app.ai import warehouse

# Column-name families. Kept as patterns rather than a fixed list so a new mart naming its
# column `manufacturer_name` instead of `manufacturer_desc` is still recognised.
TIME_PAT = re.compile(r"(?:^|_)(month|month_num|year|posting_date|gr_date|po_date|snapshot_date|"
                      r"bill_date|sales_date|delivery_date|expiry_date|period)(?:$|_)", re.I)
MEASURE_PAT = re.compile(r"(?:^|_)(revenue|cost|margin|qty|quantity|value|price|units|lines|"
                         r"days|count|n|stock|amount|spend)(?:$|_)", re.I)
ENTITY_FAMILIES = {
    "material": re.compile(r"^(material|material_id|material_desc|generic_name)$", re.I),
    "manufacturer": re.compile(r"^(manufacturer|manufacturer_desc)$", re.I),
    "vendor": re.compile(r"^(vendor|vendor_name|vendor_code)$", re.I),
    "hospital": re.compile(r"^(hospital|plant)$", re.I),
    "category": re.compile(r"^(category|material_group|major_group_desc|minor_group_desc|material_type)$", re.I),
    "department": re.compile(r"^(department|costcenter|cost_center)$", re.I),
}


@lru_cache(maxsize=1)
def schema() -> dict[str, list[str]]:
    """{table: [columns]} for every non-internal table."""
    try:
        rows = warehouse.run_sql(
            "SELECT table_name, column_name FROM information_schema.columns "
            "ORDER BY table_name, ordinal_position", row_cap=4000)["rows"]
    except Exception:
        return {}
    out: dict[str, list[str]] = {}
    for r in rows:
        t = str(r.get("table_name") or "")
        if t.startswith("_"):
            continue
        out.setdefault(t, []).append(str(r.get("column_name") or ""))
    return out


def families(cols: list[str]) -> set[str]:
    found = set()
    for c in cols:
        for fam, pat in ENTITY_FAMILIES.items():
            if pat.match(c):
                found.add(fam)
    return found


def profile(entity_family: str | None = None) -> dict:
    """Which tables are usable, and at what grain.

    `by_entity` answers "where can I filter to this thing", `with_time` narrows that to
    "…and also break it down over time" — the exact distinction the model kept getting
    wrong by assuming any sales table could do both.
    """
    sch = schema()
    by_entity, with_time = [], []
    for t, cols in sch.items():
        fams = families(cols)
        has_time = any(TIME_PAT.match(c) or TIME_PAT.search(c) for c in cols)
        measures = [c for c in cols if MEASURE_PAT.search(c)]
        if entity_family and entity_family not in fams:
            continue
        entry = {"table": t, "entities": sorted(fams), "time": has_time,
                 "measures": measures[:8]}
        by_entity.append(entry)
        if has_time:
            with_time.append(entry)
    return {"tables": by_entity, "tables_with_time": with_time}


def brief(entity_family: str | None = None, max_tables: int = 200) -> str:
    """The plan-ready description of the answerable surface — ALL of it.

    This used to truncate at 26 tables of 47, alphabetically, "to keep the prompt short".
    That silently hid every `kpi_*`, `mart_*` and `sales_*` table from the planner. Asked
    for the top-selling drugs in Bangalore it replied that the data does not exist, and
    cited `forecast_sales` as the closest thing — because forecast_sales was the only
    sales-shaped table it could still see. The data was there; the model was blindfolded.

    One line per table is ~55 tokens; the whole schema is a few thousand, it is identical
    on every call, and this deployment has prompt caching on. There is no budget argument
    for hiding half the warehouse from the thing whose job is to navigate it.
    """
    p = profile(entity_family)
    sch = schema()
    lines = []
    for e in p["tables"][:max_tables]:
        # EVERY column, not just the measures. Listing measures alone meant `dim_plant`
        # appeared as "[hospital] measures: -" — so the planner never learned that
        # `plant_name` exists, and answered "the hospital dimension does not provide
        # location details" to a question about Bangalore, while four rows of that column
        # read "HCG KR, Bangalore". You cannot navigate a warehouse you cannot see.
        lines.append(f"{e['table']}: {', '.join(sch.get(e['table'], []))}")
    head = (f"Tables carrying a '{entity_family}': {len(p['tables'])}, "
            f"of which {len(p['tables_with_time'])} also carry time.\n"
            if entity_family else f"Tables: {len(p['tables'])}.\n")
    return head + "\n".join(lines) + "\n\n" + dimension_values()


@lru_cache(maxsize=1)
def dimension_values(per_col: int = 4) -> str:
    """A few real values from each small dimension column.

    Column names alone are not enough to plan with. "Bangalore" is not a column, it is a
    VALUE inside `dim_plant.plant_name` — and with only names visible the planner concluded
    the warehouse had no location data at all. Three sample values make the difference
    between guessing at the shape of the data and seeing it.

    Restricted to genuinely small dimensions (<= 400 distinct) so this stays a handful of
    lines and never leaks a long tail of material descriptions into every prompt.
    """
    out = []
    for table, cols in schema().items():
        if not (table.startswith("dim_") or table.startswith("sales_by_") or table == "kpi_vendor_lead_time"):
            continue
        for c in cols:
            if not any(k in c.lower() for k in ("name", "desc", "group", "type", "plant", "hospital", "category", "manufacturer", "patient", "scope")):
                continue
            try:
                n = warehouse.run_sql(f"SELECT COUNT(DISTINCT {c}) AS n FROM {table}", row_cap=1)["rows"][0]["n"]
                if not n or n > 400:
                    continue
                vals = warehouse.run_sql(
                    f"SELECT DISTINCT CAST({c} AS VARCHAR) AS v FROM {table} "
                    f"WHERE {c} IS NOT NULL ORDER BY 1 LIMIT {per_col}", row_cap=per_col)["rows"]
            except Exception:
                continue
            sample = ", ".join(str(r["v"])[:40] for r in vals)
            if sample:
                out.append(f"{table}.{c} ({n} distinct): {sample} …")
    return ("EXAMPLE VALUES (so you can see what is IN the columns, not just their names):\n"
            + "\n".join(out[:40])) if out else ""


@lru_cache(maxsize=1)
def entity_columns_for_search() -> tuple[tuple[str, str], ...]:
    """(table, column) pairs worth probing when hunting for where a value lives."""
    out = []
    for t, cols in schema().items():
        for c in cols:
            if any(pat.match(c) for pat in ENTITY_FAMILIES.values()) or c.lower().endswith(("_name", "_desc")):
                out.append((t, c))
    return tuple(out)
