"""Consumables · department consumption — the consumables (HCG's 3-series) each department (SAP
cost centre) was issued, per hospital. The consumables cut of the Consumption page's department view.

    GET /portfolio/consumption/consumables/overview     -> totals, month-by-month cost, the top
                                                           departments and the top consumables
    GET /portfolio/consumption/consumables/departments  -> {data,total,sums} one row per cost centre
    GET /portfolio/consumption/consumables/items        -> {data,total,sums} one row per consumable;
                                                           CostCentre= narrows it to one department

Plant= takes a plant code or a hospital name (the page's region selector sends the name), resolved
by da.resolve_plant like every other consumption endpoint; absent or 'All Plants' = every hospital.

Why not the page's existing department data: kpi_consumption_by_department is cut at plant x cost
centre x month with no material column, so it cannot be narrowed to consumables (see
`departments_category_scoped` in legacy_kpi._consumption_overview_cached). This reads
fact_consumption itself — SAP goods issues to a cost centre (movement 201; the ETL keeps only
those) — which does carry the material.

A consumable is a material whose code starts with 3: HCG's own series rule (1 = Onco, 2 =
Non-onco, 3 = Consumables), the same one the spend module uses (CATEGORY_SQL in
app/etl/spend/aggregate.py). Not dim_material.material_type: ZMC covers only part of the 3-series
and many 3-series materials carry no type at all.

Cost is amount_lc, the issue's value in rupees. Every share is a share of the cost total in the
same scope (the hospital, and for the items list the department too). HCG has not sent department
names yet, so dim_costcenter's name is the code itself; `names_pending` says so.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Request

from app.api import formulary as F
from app.core import data_access as da
from app.core import drill_sources as ds
from app.core import table_query as tq
from app.core.config import settings

router = APIRouter()

FACT, COSTCENTRES = "fact_consumption", "dim_costcenter"
CONSUMABLE_SERIES = "3"            # first digit of a consumable's material code
TOP_N = 8
UNASSIGNED = "(unassigned)"        # an issue with no cost centre (none today), kpi_generic's sentinel
NOT_BUILT = "Consumption data has not been built yet — run python -m app.etl.run_etl"
_READ_COLS = ["material", "material_desc", "plant", "cost_ctr", "qty", "amount_lc", "uom", "year", "month_num"]
_MON = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")

DEPT_COLS = ["cost_ctr", "department_name", "plant", "hospital", "cost", "qty", "items", "share_pct",
             "top_item_desc"]
ITEM_COLS = ["material", "material_desc", "uom", "cost", "qty", "share_pct", "departments", "months"]
SUMS = ["cost", "qty"]


def _cache(fn):
    return lru_cache(maxsize=64)(fn)


# ── Data ──────────────────────────────────────────────────────────────────────
# Read at CALL time from settings.CURATED_DIR and keyed on each file's path, mtime and size, like
# the formulary and spend tables: a rebuilt parquet, or a test's own fixture, is picked up without
# a restart, and every cached result below is keyed on the same version.
def _path(name: str) -> Path:
    return Path(settings.CURATED_DIR) / f"{name}.parquet"


def _version() -> tuple:
    fact, cc = _path(FACT), _path(COSTCENTRES)
    stamp = F._stamp(fact)
    if stamp is None:
        raise HTTPException(status_code=503, detail=NOT_BUILT)
    return str(fact), stamp, str(cc), F._stamp(cc)


@lru_cache(maxsize=1)
def _read(version: tuple) -> tuple:
    """(consumable rows, every (year, month) in the file, cost centre -> department name).

    Only the columns used are read and only the 3-series rows are kept (about a third of the
    table), so the frame held in memory stays small. The months come from the WHOLE file, before
    the cut: a hospital that issued no consumables in some month still shows that month, at zero,
    instead of a timeline that quietly skips it."""
    fact_path, _, cc_path, cc_stamp = version
    df = pd.read_parquet(fact_path, columns=_READ_COLS)
    ym = df[["year", "month_num"]].dropna().drop_duplicates().astype(int)
    periods = sorted(zip(ym["year"].tolist(), ym["month_num"].tolist()))

    df["material"] = df["material"].fillna("").astype(str).str.strip()
    df = df[df["material"].str[:1] == CONSUMABLE_SERIES].dropna(subset=["year", "month_num"])
    out = pd.DataFrame({
        "material": df["material"],
        "material_desc": df["material_desc"].fillna(df["material"]).astype(str),
        "plant": df["plant"].fillna("").astype(str).str.strip(),
        "cost_ctr": df["cost_ctr"].fillna(UNASSIGNED).astype(str).str.strip(),
        "uom": df["uom"].fillna("").astype(str),
        "qty": pd.to_numeric(df["qty"], errors="coerce").fillna(0.0).astype(float),
        "cost": pd.to_numeric(df["amount_lc"], errors="coerce").fillna(0.0).astype(float),
        "period": (df["year"].astype(int) * 100 + df["month_num"].astype(int)).astype(int),
    }).reset_index(drop=True)

    names: dict = {}
    if cc_stamp is not None:
        dc = pd.read_parquet(cc_path, columns=["cost_ctr", "department_name"])
        names = {str(c).strip(): str(n).strip() for c, n in zip(dc["cost_ctr"], dc["department_name"])
                 if pd.notna(n) and str(n).strip() not in ("", "nan")}
    return out, periods, names


def _scoped(pl: Optional[str], version: tuple) -> pd.DataFrame:
    df = _read(version)[0]
    return df if pl is None else df[df["plant"] == pl]


def _label(year: int, month: int) -> str:
    return f"{_MON[month - 1]}-{year % 100:02d}"


def _share(cost: pd.Series) -> pd.Series:
    total = float(cost.sum())
    return (cost / total * 100).round(4) if total else cost * 0.0


@_cache
def _departments_cached(pl: Optional[str], version: tuple) -> pd.DataFrame:
    """One row per department (plant x cost centre), most consumables cost first."""
    df = _scoped(pl, version)
    if df.empty:
        return pd.DataFrame(columns=DEPT_COLS + ["top_item", "top_item_cost"])
    names = _read(version)[2]
    g = df.groupby(["plant", "cost_ctr"], as_index=False).agg(
        cost=("cost", "sum"), qty=("qty", "sum"), items=("material", "nunique"))
    # The consumable each department spent most on; a tie goes to the lower code so the
    # answer never changes between calls.
    by_item = df.groupby(["plant", "cost_ctr", "material"], as_index=False).agg(
        top_item_cost=("cost", "sum"), top_item_desc=("material_desc", "first"))
    top = (by_item.sort_values(["top_item_cost", "material"], ascending=[False, True], kind="mergesort")
           .drop_duplicates(["plant", "cost_ctr"]).rename(columns={"material": "top_item"}))
    g = g.merge(top, on=["plant", "cost_ctr"], how="left")
    g["share_pct"] = _share(g["cost"])
    g["department_name"] = g["cost_ctr"].map(names).fillna(g["cost_ctr"])
    pnames = ds.plant_name_map()
    g["hospital"] = g["plant"].map(pnames).fillna(g["plant"])
    g["cost"] = g["cost"].round(2)
    g["top_item_cost"] = g["top_item_cost"].round(2)
    g["qty"] = g["qty"].round(3)
    g = g.sort_values(["cost", "cost_ctr"], ascending=[False, True], kind="mergesort")
    return g[DEPT_COLS + ["top_item", "top_item_cost"]].reset_index(drop=True)


@_cache
def _items_cached(pl: Optional[str], cc: Optional[str], version: tuple) -> pd.DataFrame:
    """One row per consumable, most cost first; `cc` narrows it to one department."""
    df = _scoped(pl, version)
    if cc is not None:
        df = df[df["cost_ctr"] == cc]
    if df.empty:
        return pd.DataFrame(columns=ITEM_COLS)
    g = df.groupby("material", as_index=False).agg(
        material_desc=("material_desc", "first"),
        # every unit the item was issued in ('NOS / BOX'), so a quantity in mixed units says so
        uom=("uom", lambda u: " / ".join(sorted({x for x in u if x})) or None),
        cost=("cost", "sum"), qty=("qty", "sum"), months=("period", "nunique"))
    depts = df.drop_duplicates(["material", "plant", "cost_ctr"]).groupby("material").size()
    g["departments"] = g["material"].map(depts).fillna(0).astype(int)
    g["share_pct"] = _share(g["cost"])
    g["cost"] = g["cost"].round(2)
    g["qty"] = g["qty"].round(3)
    g = g.sort_values(["cost", "material"], ascending=[False, True], kind="mergesort")
    return g[ITEM_COLS].reset_index(drop=True)


def _note(periods: list) -> str:
    if not periods:
        span = "no months of consumption are loaded yet"
    elif len(periods) == 1:
        span = _label(*periods[0])
    else:
        span = f"{_label(*periods[0])} to {_label(*periods[-1])}"
    return ("Consumables by HCG's series rule (material codes starting with 3) issued from store to each "
            f"department's cost centre, from SAP consumption (goods issues), {span}. Not the same set as "
            "the 'Consumables' category in the cards above, which follows the SAP material type (ZMC).")


@_cache
def _overview_cached(pl: Optional[str], version: tuple) -> dict:
    df = _scoped(pl, version)
    _, periods, names = _read(version)
    dept = _departments_cached(pl, version)
    items = _items_cached(pl, None, version)

    by_month = df.groupby("period").agg(cost=("cost", "sum"), qty=("qty", "sum"))
    months = []
    for y, m in periods:            # every month in the data, oldest first; 0 where none issued
        k = y * 100 + m
        months.append({"key": f"{y:04d}-{m:02d}", "label": _label(y, m),
                       "cost": round(float(by_month["cost"].get(k, 0.0)), 2),
                       "qty": round(float(by_month["qty"].get(k, 0.0)), 3)})

    departments = [{"code": str(r.cost_ctr), "name": str(r.department_name), "plant": str(r.plant),
                    "hospital": str(r.hospital), "cost": float(r.cost), "qty": float(r.qty),
                    "items": int(r.items), "share_pct": float(r.share_pct),
                    "top_item": {"material": str(r.top_item), "desc": str(r.top_item_desc),
                                 "cost": float(r.top_item_cost)}}
                   for r in dept.head(TOP_N).itertuples(index=False)]
    top_items = [{"material": str(r.material), "desc": str(r.material_desc), "cost": float(r.cost),
                  "qty": float(r.qty), "departments": int(r.departments)}
                 for r in items.head(TOP_N).itertuples(index=False)]
    pnames = ds.plant_name_map()
    return {
        "totals": {"cost": round(float(df["cost"].sum()), 2), "qty": round(float(df["qty"].sum()), 3),
                   "items": int(df["material"].nunique()), "departments": int(len(dept)),
                   "hospitals": int(df["plant"].nunique())},
        "months": months,
        "departments": departments,
        "items": top_items,
        "scope": {"plant": pl, "hospital": pnames.get(pl, pl) if pl else None},
        # True while no cost centre has a name of its own (HCG has not sent the names master).
        "names_pending": not any(n != c for c, n in names.items()),
        "note": _note(periods),
    }


def _page(request: Request, frame: pd.DataFrame, columns: list, *, facets: tuple = (),
          scope: tuple = ()) -> dict:
    # The shared {data,total} paging / sort / search and table filters, plus `sums` over every
    # filtered row (not just the page) so the table can print a Total row. The frame is already
    # cut to the plant (echoed in `scope` with facets=1); Hospital is a filter of its own rows.
    return da.paginate(FACT, None, request.query_params, col_map={}, frame=frame,
                       columns=columns, sum_cols=SUMS, facets=facets, scope=scope)


# ── Endpoints ─────────────────────────────────────────────────────────────────
@router.get("/portfolio/consumption/consumables/overview")
def consumables_overview(Plant: str = Query(None)):
    return _overview_cached(da.resolve_plant(Plant), _version())


@router.get("/portfolio/consumption/consumables/departments")
def consumables_departments(request: Request, Plant: str = Query(None)):
    # Only the shown columns, so a search never matches the hidden top-item code or cost.
    pl = da.resolve_plant(Plant)
    return _page(request, _departments_cached(pl, _version())[DEPT_COLS], DEPT_COLS, scope=(tq.plant_chip(pl),))


@router.get("/portfolio/consumption/consumables/items")
def consumables_items(request: Request, Plant: str = Query(None), CostCentre: str = Query(None)):
    cc = (CostCentre or "").strip() or None
    pl = da.resolve_plant(Plant)
    scope = (tq.plant_chip(pl), *([tq.chip("CostCentre", "Department", cc)] if cc else []))
    return _page(request, _items_cached(pl, cc, _version()), ITEM_COLS, facets=(tq.Facet("uom", "Unit of measure"),),
                 scope=scope)
