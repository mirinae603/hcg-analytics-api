"""Purchase insights: price variation and generic vs branded, from HCG's goods receipts (GRN).

Served from the two small aggregated tables `python -m app.etl.purchase` publishes into
settings.PURCHASE_DATA_DIR (no GR, PO or invoice number and no line-level row reaches the repo
or this process):

    GET /spend/prices/overview   -> price variation: headline, the products that cost most
                                    above their cheapest price (Molecule= picks one: its
                                    purchases and vendors), every hospital, the biggest
                                    vendors' price index, every month, what was not compared
    GET /spend/prices/molecules  -> {data,total,sums} per product (Contract A)
    GET /spend/generic/overview  -> generic vs branded: headline, every month, every hospital,
                                    onco / non-onco, each side by formulary class (months with
                                    their own Item Master), the branded products a generic of
                                    the same product undercut, the definition

Lines: the receipts of HCG's out-of-formulary workings (movement 101, drug series 1 and 2,
without S.Loc S125 and the BH*, HM03, HM04 plants) of EVERY GRN month, every formulary class.
Purchases carry no patient type, payer or doctor: an overview ignores Type= and Payer= and
says so.

Price variation. A product is a molecule_key — molecule | strength | form | kind, as
app/etl/purchase/molecule.py reads the Item Master's minor group and description. The unit
price is the price after discount (Total Amount without tax / GR Qty). Only rows with a
comparable unit price (price_status ok) and a product key are compared; `value` counts every
row. The cheapest is the product's USUAL cheapest price in the page's scope (hospital,
months, segment): the lowest unit price at which at least prices.USUAL_SHARE of its quantity
was bought (prices.usual_cheapest), so one odd purchase cannot set it; Vendor= and Material=
narrow whose purchases are set against it. Extra paid = Σ over purchases of what each paid
above the cheapest, max(0, value - cheapest x qty); price index = Σ value / Σ (cheapest x qty)
(1.00 = always at the cheapest; under 1 for a vendor that sold under it). A purchase under
prices.LOW_RATIO x the cheapest is unusually low: still compared, and listed (low_prices) for
HCG to check. A vendor
is a supplier as HCG names it ('Goods supplier name', spacing and case aside): one supplier
holds several SAP supplier codes (Vardhman Health Specialities has one per region).

Generic vs branded ('Estimated definition' until HCG confirms it): branded = an originator
company's product (app/data/reference/originator_manufacturers.csv, generic_overrides.csv);
generic = every other product. Extra vs generic = what the branded purchases cost above the
usual cheapest generic price of the same product that month (any hospital), worked out per
line by the build so it adds up over any cut; % = that / the branded value priced against a
generic.

Every figure is a ratio of SUMS taken after filtering — never an average of percentages — and
None when its denominator is 0.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Request

from app.api import formulary as F
from app.core import data_access as da
from app.core import table_query as tq
from app.core.config import settings
from app.etl.formulary import classify as K
from app.etl.purchase import config as C
from app.etl.purchase import generic as Gn
from app.etl.purchase import prices as P

router = APIRouter()

TABLES = (C.PRICES, C.GENERIC)
# The columns the endpoints read: a table without them was written by an older build.
REQUIRED = {C.PRICES: P.PRICES_COLUMNS, C.GENERIC: Gn.GENERIC_COLUMNS}
NOT_BUILT = ("Price variation and generic vs branded have not been built yet — run "
             "python -m app.etl.purchase")
NOT_CURRENT = ("Price variation and generic vs branded were built by an older version — run "
               "python -m app.etl.purchase again")

SEGMENT_LABELS = F.SEGMENT_LABELS
MOLECULE_LIMIT, MOLECULE_MAX = 100, 500      # products the overview sends for its chart
TOP_N = 10                                   # vendors and branded products listed (HCG's top 10)
PRICE_DECIMALS = P.UNIT_PRICE_DECIMALS

# Why a purchase's unit price is not compared: the build's price statuses, or no product key.
NO_PRODUCT = "no_product"
NOT_COMPARED = {NO_PRODUCT: "the description does not give the molecule, strength or form",
                **{s: m for s, m in P.PRICE_STATUSES.items() if s != P.OK}}

CLASS_ORDER = (Gn.GENERIC, Gn.BRANDED)
CLASS_LABELS = {Gn.GENERIC: "Generic", Gn.BRANDED: "Branded"}
FLAG_LABELS = {K.FORMULARY: "Formulary", K.NON_FORMULARY: "Non formulary",
               K.OUT_OF_FORMULARY: "Out of formulary", K.NOT_IN_MASTER: "Not in the Item Master"}
TIER_LABELS = {Gn.PREFERRED: "Preferred", Gn.SCHEME_ONLY: "Scheme only", Gn.RESTRICTED: "Restricted",
               Gn.OTHER_TIER: "Other formulary"}
ESTIMATED_LABEL = "Estimated definition"


# ── Tables ────────────────────────────────────────────────────────────────────
# Read at CALL time from settings.PURCHASE_DATA_DIR and keyed on each file's mtime/size, like
# the spend and formulary tables: a rebuild or a test's own fixtures are picked up without a
# restart.
def _data_dir() -> Path:
    return Path(settings.PURCHASE_DATA_DIR)


@lru_cache(maxsize=4)
def _read(path: str, stamp: tuple) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "vendor_name" in df.columns:
        df["vendor"] = P.vendor_keys(df["vendor_name"])
    return df


def _table(name: str) -> pd.DataFrame:
    p = _data_dir() / f"{name}.parquet"
    stamp = F._stamp(p)
    if stamp is None:
        raise HTTPException(status_code=503, detail=NOT_BUILT)
    df = _read(str(p), stamp)
    if not set(REQUIRED[name]) <= set(df.columns):
        raise HTTPException(status_code=503, detail=NOT_CURRENT)
    return df


def _version() -> tuple:
    d = _data_dir()
    return (str(d),) + tuple(F._stamp(d / f"{t}.parquet") for t in TABLES) + (F._stamp(d / C.VALIDATION_FILE),)


def _validation() -> Optional[dict]:
    try:
        return json.loads((_data_dir() / C.VALIDATION_FILE).read_text())
    except (FileNotFoundError, ValueError):
        return None


def _built_at() -> Optional[str]:
    v = _validation() or {}
    if v.get("built_at"):
        return str(v["built_at"])
    st = F._stamp(_data_dir() / f"{C.PRICES}.parquet")
    return datetime.fromtimestamp(st[0] / 1e9, tz=timezone.utc).isoformat(timespec="seconds") if st else None


_RESULT_CACHES: list = []


def _cache(fn):
    cached = lru_cache(maxsize=64)(fn)
    _RESULT_CACHES.append(cached)
    return cached


def clear_caches() -> None:
    _read.cache_clear()
    for c in _RESULT_CACHES:
        c.cache_clear()


# ── Arguments ─────────────────────────────────────────────────────────────────
class Scope(NamedTuple):
    plant: Optional[str]            # SAP plant code; None => all hospitals
    months: tuple                   # periods 'YYYY-MM'
    segment: Optional[str]          # 'onco' / 'non-onco'; None => both
    vendor: Optional[str] = None    # Vendor= (a supplier's name, or one SAP supplier code): its purchases only
    material: Optional[str] = None  # Material= (a brand): its purchases only


@_cache
def _periods_at(name: str, version: tuple) -> list:
    return sorted(_table(name)["period"].astype(str).unique())


def _periods(name: str) -> list:
    return _periods_at(name, _version())


def _months_arg(month, periods: list) -> tuple:
    s = F._str(month)
    if s is None:
        return tuple(periods[-1:])          # latest month, like HCG's monthly report
    if s.upper() == "ALL":
        return tuple(periods)
    out = []
    for m in (x.strip() for x in s.split(",")):
        if not m:
            continue
        if not F._PERIOD_RE.match(m):
            F._bad(f"Invalid Month '{m}'. Use YYYY-MM, a comma list of them, or ALL")
        if m not in periods:
            F._bad(f"No purchase data for {m}. Available months: {', '.join(periods)}")
        out.append(m)
    if not out:
        F._bad("Month is empty. Use YYYY-MM, a comma list of them, or ALL")
    return tuple(sorted(set(out)))


def _segment_arg(segment, category) -> Optional[str]:
    """Segment=onco|non-onco, or the app-wide Category= (Onco Drugs / Other Drugs): the purchase
    scope holds drugs only (material series 1 and 2)."""
    seg = F._choice(segment, F._SEGMENT_IN, "Segment")
    cat = da.resolve_category(F._str(category))
    if cat:
        by_cat = K.SEGMENT_KEY.get(cat)
        if by_cat is None:
            F._bad(f"Category '{category}' is not in the purchase analysis — it covers drugs only. "
                   f"Use Onco Drugs or Other Drugs (or Segment=onco / non-onco)")
        if seg and seg != by_cat:
            F._bad(f"Segment '{segment}' and Category '{category}' disagree")
        seg = by_cat
    return seg


def _scope(name: str, Plant, Month, Segment, Category, Vendor=None, Material=None) -> Scope:
    periods = _periods(name)                # 503 first: nothing below means anything without data
    return Scope(plant=F._plant_arg(Plant), months=_months_arg(Month, periods),
                 segment=_segment_arg(Segment, Category), vendor=F._str(Vendor), material=F._code(Material))


def _ignored(**filters) -> tuple:
    """The page filters sent that a purchase cannot apply (it has no patient type or payer)."""
    return tuple(k for k, v in filters.items() if (F._str(v) or "ALL").upper() != "ALL")


def _limit(limit) -> int:
    return max(1, min(int(limit) if isinstance(limit, int) else MOLECULE_LIMIT, MOLECULE_MAX))


# ── Filtering and echoes ──────────────────────────────────────────────────────
def _cut(df: pd.DataFrame, sc: Scope, *, plant=True, months=True, segment=True, drill=True) -> pd.DataFrame:
    """Rows of `df` in the page's scope (hospital, months, segment) and, with `drill`, of the
    purchases Vendor= / Material= narrow to. The flags drop a filter for the views that
    ignore it."""
    m = pd.Series(True, index=df.index)
    if plant and sc.plant:
        m &= df["plant"].astype(str) == sc.plant
    if months:
        m &= df["period"].astype(str).isin(sc.months)
    if segment and sc.segment:
        m &= df["segment"].astype(str) == sc.segment
    if drill and sc.material:
        m &= df["material"].astype(str) == sc.material
    if drill and sc.vendor:
        m &= (df["vendor"] == K.norm_text(sc.vendor)) | (df["vendor_code"].astype(str) == F._code(sc.vendor))
    return df[m]


def _names() -> dict:
    """SAP plant -> the name the app shows."""
    return tq._plant_names()


def _scope_echo(sc: Scope, *, molecule: Optional[str] = None, ignored: tuple = ()) -> dict:
    out = {"plant": sc.plant or "ALL", "plant_name": _names().get(sc.plant, sc.plant) if sc.plant else "All Hospitals",
           "months": list(sc.months), "category": sc.segment or "ALL"}
    if sc.vendor:
        out["vendor"] = sc.vendor
    if sc.material:
        out["material"] = sc.material
    if molecule:
        out["molecule"] = molecule
    if ignored:
        out["ignored"] = list(ignored)
    return out


def _months_available(name: str) -> list:
    """Every month the table holds; item_master: the month has its own Item Master (the
    formulary split of generic vs branded needs one)."""
    im = _item_master_periods()
    return [{"period": p, "label": tq.month_label(p), "item_master": p in im} for p in _periods(name)]


def months_meta() -> dict:
    """The purchase sections' months for /spend/meta, so the page knows them without fetching
    an overview: each section's months_available ([] while its table is not built, or was built
    by an older version) and the build time."""
    out = {}
    for key, name in (("prices_months", C.PRICES), ("generic_months", C.GENERIC)):
        try:
            out[key] = _months_available(name)
        except HTTPException:
            out[key] = []
    out["purchase_built_at"] = _built_at() if out["prices_months"] or out["generic_months"] else None
    return out


NOTE_SCOPE = ("Purchases: the goods receipts HCG's out-of-formulary workings count{steps} ({file}); every month "
              "with a GRN file, every formulary class.")
NOTE_PRICE_VALUES = ("Unit price = the price after discount (Total Amount without tax ÷ GR Qty). A product is one "
                     "molecule, strength, form and kind (e.g. CETUXIMAB 100MG injection): its brands are the same "
                     f"medicine. Cheapest = the usual cheapest price in this selection: the lowest unit price at "
                     f"which at least {P.USUAL_SHARE:.0%} of the product's quantity was bought, so one odd purchase "
                     f"cannot set it. Extra paid = what each purchase paid above it; price index = paid ÷ the same "
                     f"quantity at the cheapest (1.00 = always the cheapest). A purchase under "
                     f"{P.LOW_RATIO:.0%} of the cheapest is listed as an unusually low price to check.")
NOTE_PRICE_MONTHS = ("Several months are selected: the cheapest is the usual cheapest over all of them, so the "
                     "extra paid differs from the months' own figures added up.")
NOTE_PRICE_VIEWS = ("Every month ignores the Month filter and sets each month against its own cheapest; the "
                    "hospital ranking ignores the hospital selector and sets each hospital against the cheapest "
                    "price any hospital paid.")
NOTE_NOT_COMPARED = "Not compared (counted in the value only): " + "; ".join(NOT_COMPARED.values()) + "."
NOTE_DRILL = ("A vendor or brand is selected: only its purchases are counted, set against the cheapest price "
              "anyone paid in the selection.")
NOTE_IGNORED = "Purchases have no patient type or payer, so the {filters} filter does not apply to this section."
NOTE_MONTHS = ("Several months are selected: amounts are summed first and the % is taken once — never an "
               "average of monthly percentages.")
NOTE_GENERIC_VALUES = ("Value = Total Amount without tax (the price after discount). Extra vs generic: a branded "
                       "purchase of a product also bought as a generic that month (at any hospital) counts its unit "
                       f"price above the usual cheapest generic price (the lowest at which at least "
                       f"{P.USUAL_SHARE:.0%} of the generic quantity was bought) × the quantity; % = that ÷ the "
                       f"branded value set against a generic.")
NOTE_GENERIC_VIEWS = ("Every month ignores the Month filter, the hospital ranking ignores the hospital selector and "
                      "onco / non-onco ignores the segment filter.")
NOTE_COMPLIANCE = ("The formulary split uses each month's own Item Master, so it covers {months} only — "
                   "{missing} {verb} no Item Master.")
NOTE_NO_COMPLIANCE = "The formulary split needs each month's own Item Master, and {missing} {verb} none."


def _note_scope() -> str:
    """Which receipts count, in the words of the build's own scope funnel (latest month): the
    exclusions are HCG's reference file, so the note follows it."""
    v = _validation() or {}
    funnel = ((v.get("months") or {}).get(v.get("latest") or "") or {}).get("funnel") or []
    steps = [str(s["label"]) for s in funnel if s.get("kind") != "lines" and s.get("label")]
    return NOTE_SCOPE.format(steps=f" — {'; '.join(steps)}" if steps else "",
                             file=(v.get("scope") or {}).get("file") or C.SCOPE_FILE)


def _ignored_notes(ignored: tuple) -> list:
    return [NOTE_IGNORED.format(filters=" / ".join(ignored))] if ignored else []


# ── Price variation ───────────────────────────────────────────────────────────
PRODUCT_COLUMNS = ["molecule_key", "minor_group", "molecule", "strength", "form", "kind", "category",
                   "cheapest_price", "min_price", "max_price", "wavg_price", "spread_ratio", "price_index",
                   "varies", "qty", "value", "extra_paid", "extra_pct", "share_pct", "low_rows", "low_value",
                   "brands", "vendors", "plants", "lines", "cheapest_brand", "cheapest_vendor", "cheapest_plant"]
_PRODUCT_TEXT = ("molecule_key", "minor_group", "molecule", "strength", "form", "kind", "category",
                 "cheapest_brand", "cheapest_vendor", "cheapest_plant")
_PRODUCT_COUNTS = ("brands", "vendors", "plants", "lines", "low_rows")
# What the overview's chart gets per product (the list carries every column).
CHART_COLUMNS = ["molecule_key", "minor_group", "molecule", "strength", "form", "kind", "cheapest_price",
                 "min_price", "max_price", "wavg_price", "spread_ratio", "price_index", "varies", "qty", "value",
                 "extra_paid", "extra_pct", "share_pct", "low_rows", "brands", "vendors", "plants"]
VENDOR_COLUMNS = ["vendor_name", "vendor_codes", "value", "at_cheapest", "extra_paid", "price_index", "share_pct",
                  "molecules", "brands", "plants", "lines"]
POINT_COLUMNS = ["plant", "plant_name", "material", "material_desc", "vendor_name", "unit_price", "unit_price_min",
                 "unit_price_max", "qty", "value", "lines", "unusually_low"]
LOW_COLUMNS = ["molecule_key", "minor_group", "strength", "form", "plant", "plant_name", "material", "material_desc",
               "vendor_name", "unit_price", "cheapest_price", "qty", "value", "below_cheapest", "lines"]
_PRICE_FIELDS = ("cheapest_price", "min_price", "max_price", "wavg_price", "spread_ratio", "price_index",
                 "unit_price", "unit_price_min", "unit_price_max")


def _compared(df: pd.DataFrame) -> pd.DataFrame:
    """The rows whose unit price is compared: a comparable price (ok) and a product key."""
    return df[df["price_status"].eq(P.OK) & df["molecule_key"].notna()]


def _cheapest(base: pd.DataFrame) -> pd.DataFrame:
    """Per product of the compared rows `base` (index molecule_key): its usual cheapest unit
    price (prices.usual_cheapest over the rows' own prices) and the brand, vendor and hospital
    that paid it (of several at that price, the biggest purchase)."""
    best = P.usual_cheapest(base.assign(_price=P.row_prices(base)), "molecule_key", "_price",
                            carry=("material_desc", "vendor_name", "plant"))
    return best.rename(columns={"_price": "cheapest_price", "material_desc": "cheapest_brand",
                                "vendor_name": "cheapest_vendor", "plant": "cheapest_plant"})


def _row_cheapest(rows: pd.DataFrame, cheapest: pd.Series) -> pd.Series:
    """Each compared row's product's cheapest price (`cheapest`: per molecule_key)."""
    return rows["molecule_key"].map(cheapest).astype(float)


def _at_cheapest(rows: pd.DataFrame, cheapest: pd.Series) -> pd.Series:
    """What each compared row would have cost at its product's cheapest price."""
    return _row_cheapest(rows, cheapest) * rows["qty"].astype(float)


def _extra(rows: pd.DataFrame, cheapest: pd.Series) -> pd.Series:
    """What each compared row paid above its product's cheapest price (never below 0)."""
    return P.extra_paid(rows, _row_cheapest(rows, cheapest))


def _round_prices(df: pd.DataFrame, cols=_PRICE_FIELDS) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(float).round(PRICE_DECIMALS)
    return df


def _empty_products() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=object if c in _PRODUCT_TEXT else "int64" if c in _PRODUCT_COUNTS
                                      else bool if c == "varies" else float) for c in PRODUCT_COLUMNS})


def _products(rows: pd.DataFrame, cheapest: pd.DataFrame) -> pd.DataFrame:
    """One row per product of the compared `rows` (PRODUCT_COLUMNS): P.products' price range,
    value-weighted price and counts, set against `cheapest` (_cheapest of the scope) — spread =
    dearest / cheapest, price index, extra paid and its share of the whole; most extra paid
    first."""
    view = P.products(rows)
    if view.empty:
        return _empty_products()
    low = P.unusually_low(rows, _row_cheapest(rows, cheapest["cheapest_price"]))
    info = (rows.assign(_extra=_extra(rows, cheapest["cheapest_price"]), _low=low.astype(int),
                        _low_value=rows["value"].where(low, 0.0))
            .groupby("molecule_key").agg(molecule=("molecule", "first"), kind=("kind", "first"),
                                         category=("category", "first"), lines=("lines", "sum"),
                                         extra_paid=("_extra", "sum"), low_rows=("_low", "sum"),
                                         low_value=("_low_value", "sum")))
    out = (view.drop(columns=["cheapest_price", "spread_ratio", "extra_paid", "low_rows", "low_value"])
           .join(info, on="molecule_key").join(cheapest, on="molecule_key"))
    cheap = out["cheapest_price"].astype(float)
    at_cheapest = cheap * out["qty"].astype(float)
    out["price_index"] = out["value"] / at_cheapest.where(at_cheapest != 0)
    out["spread_ratio"] = out["max_price"] / cheap.where(cheap != 0)
    out["varies"] = (out["spread_ratio"] >= 1 + P.SPREAD_MIN).astype(bool)
    out["extra_pct"] = out["extra_paid"] / out["value"].where(out["value"] != 0) * 100
    tot = float(out["extra_paid"].sum())
    out["share_pct"] = out["extra_paid"] / tot * 100 if tot else np.nan
    out = _round_prices(F._round(out, money=("value", "extra_paid", "low_value"), pct=("extra_pct", "share_pct"),
                                 qty=("qty",)))
    out["low_rows"] = out["low_rows"].astype("int64")
    out = out.sort_values(["extra_paid", "value", "molecule_key"], ascending=[False, False, True], kind="mergesort")
    return out[PRODUCT_COLUMNS].reset_index(drop=True)


def _vendors(rows: pd.DataFrame, cheapest: pd.Series) -> pd.DataFrame:
    """One row per vendor of the compared `rows` (VENDOR_COLUMNS + qty): what it was paid, the
    same purchases at each product's cheapest price, the extra paid and the price index;
    biggest first."""
    if rows.empty:
        return pd.DataFrame(columns=VENDOR_COLUMNS + ["qty"])
    v = rows.assign(at_cheapest=_at_cheapest(rows, cheapest), extra=_extra(rows, cheapest),
                    qty=rows["qty"].astype(float))
    g = v.groupby("vendor").agg(vendor_name=("vendor_name", "last"),
                                vendor_codes=("vendor_code", lambda s: sorted(s.astype(str).unique())),
                                value=("value", "sum"), at_cheapest=("at_cheapest", "sum"), qty=("qty", "sum"),
                                extra_paid=("extra", "sum"), molecules=("molecule_key", "nunique"),
                                brands=("material", "nunique"), plants=("plant", "nunique"),
                                lines=("lines", "sum")).reset_index()
    g["price_index"] = g["value"] / g["at_cheapest"].where(g["at_cheapest"] != 0)
    tot = float(g["value"].sum())
    g["share_pct"] = g["value"] / tot * 100 if tot else np.nan
    g = _round_prices(F._round(g, money=("value", "at_cheapest", "extra_paid"), pct=("share_pct",), qty=("qty",)))
    g = g.sort_values(["value", "vendor"], ascending=[False, True], kind="mergesort")
    return g[VENDOR_COLUMNS + ["qty"]].reset_index(drop=True)


def _by_hospital(every: pd.DataFrame, network: pd.DataFrame, sc: Scope) -> list:
    """Every hospital's compared purchases set against the cheapest price ANY hospital paid in
    the same months (the ranking ignores the hospital selector): most extra paid first."""
    if every.empty:
        return []
    names = _names()
    v = every.assign(at_cheapest=_at_cheapest(every, network["cheapest_price"]),
                     extra=_extra(every, network["cheapest_price"]))
    g = v.groupby(v["plant"].astype(str)).agg(value=("value", "sum"), at_cheapest=("at_cheapest", "sum"),
                                              extra_paid=("extra", "sum"), molecules=("molecule_key", "nunique"),
                                              lines=("lines", "sum")).reset_index()
    g["price_index"] = g["value"] / g["at_cheapest"].where(g["at_cheapest"] != 0)
    g["extra_pct"] = g["extra_paid"] / g["value"].where(g["value"] != 0) * 100
    g = _round_prices(F._round(g, money=("value", "at_cheapest", "extra_paid"), pct=("extra_pct",)))
    g = g.sort_values(["extra_paid", "plant"], ascending=[False, True], kind="mergesort")
    return [{"plant": r["plant"], "name": names.get(r["plant"], r["plant"]),
             **{k: r[k] for k in ("value", "at_cheapest", "extra_paid", "extra_pct", "price_index", "molecules",
                                  "lines")},
             "rank": i + 1, "selected": r["plant"] == sc.plant} for i, r in enumerate(da._clean_records(g))]


def _by_month(rows: pd.DataFrame, base: pd.DataFrame, periods: list) -> list:
    """Every month of the selection's purchases (the Month filter aside), each set against its
    own month's cheapest: value, compared value, extra paid, products and how many varied."""
    comp, ref = _compared(rows), _compared(base)
    cheapest = P.usual_cheapest(ref.assign(_price=P.row_prices(ref)), ["period", "molecule_key"], "_price")["_price"]
    keys = pd.MultiIndex.from_arrays([comp["period"], comp["molecule_key"]])
    own = pd.Series(cheapest.reindex(keys).to_numpy(), index=comp.index, dtype=float)
    per = (comp.assign(qty=comp["qty"].astype(float), extra=P.extra_paid(comp, own), cheapest=own)
           .groupby(["period", "molecule_key"])
           .agg(value=("value", "sum"), qty=("qty", "sum"), max_price=("unit_price_max", "max"),
                extra=("extra", "sum"), cheapest=("cheapest", "first")))
    per["varies"] = per["max_price"] >= (1 + P.SPREAD_MIN) * per["cheapest"]
    m = per.groupby(level="period").agg(compared_value=("value", "sum"), extra_paid=("extra", "sum"),
                                        molecules=("value", "size"), molecules_with_spread=("varies", "sum"))
    tot = rows.groupby(rows["period"].astype(str))[["value", "lines"]].sum()
    out = []
    for p in periods:
        value = float(tot.loc[p, "value"]) if p in tot.index else 0.0
        cv = float(m.loc[p, "compared_value"]) if p in m.index else 0.0
        extra = float(m.loc[p, "extra_paid"]) if p in m.index else 0.0
        out.append({"period": p, "label": tq.month_label(p), "value": F._money(value),
                    "compared_value": F._money(cv), "extra_paid": F._money(extra), "extra_pct": F._share(extra, cv),
                    "molecules": int(m.loc[p, "molecules"]) if p in m.index else 0,
                    "molecules_with_spread": int(m.loc[p, "molecules_with_spread"]) if p in m.index else 0,
                    "lines": int(tot.loc[p, "lines"]) if p in tot.index else 0})
    return out


def _not_compared(rows: pd.DataFrame) -> list:
    """The purchases whose unit price is not compared, by reason (NOT_COMPARED)."""
    status = rows["price_status"].astype(str)
    reason = status.where(status.ne(P.OK), np.where(rows["molecule_key"].isna(), NO_PRODUCT, P.OK))
    g = rows.assign(reason=reason).groupby("reason")[["lines", "value"]].sum()
    return [{"reason": k, "label": label,
             "lines": int(g.loc[k, "lines"]) if k in g.index else 0,
             "value": F._money(g.loc[k, "value"]) if k in g.index else 0.0} for k, label in NOT_COMPARED.items()]


def _low_prices(comp: pd.DataFrame, cheapest: pd.DataFrame) -> dict:
    """The selection's purchases priced unusually low (prices.unusually_low: under LOW_RATIO x
    their product's usual cheapest) per hospital x brand x vendor: how many, what they cost,
    and the biggest TOP_N by how far under the cheapest they were bought. Still compared — a
    one-off deal or a keying slip for HCG to check, never the benchmark."""
    flag = P.unusually_low(comp, _row_cheapest(comp, cheapest["cheapest_price"]))
    rows = comp[flag]
    if rows.empty:
        return {"rows": 0, "value": 0.0, "products": 0, "items": []}
    g = (rows.assign(qty=rows["qty"].astype(float))
         .groupby(["molecule_key", "plant", "material", "vendor"])
         .agg(minor_group=("minor_group", "first"), strength=("strength", "first"), form=("form", "first"),
              material_desc=("material_desc", "last"), vendor_name=("vendor_name", "last"),
              qty=("qty", "sum"), value=("value", "sum"), lines=("lines", "sum")).reset_index())
    g["cheapest_price"] = g["molecule_key"].map(cheapest["cheapest_price"]).astype(float)
    g["unit_price"] = g["value"] / g["qty"].where(g["qty"] != 0)
    g["below_cheapest"] = g["cheapest_price"] * g["qty"] - g["value"]
    names = _names()
    g["plant_name"] = g["plant"].map(lambda p: names.get(p, p))
    g = _round_prices(F._round(g, money=("value", "below_cheapest"), qty=("qty",)))
    g = g.sort_values(["below_cheapest", "molecule_key", "plant"], ascending=[False, True, True], kind="mergesort")
    return {"rows": int(len(g)), "value": F._money(rows["value"].sum()), "products": int(g["molecule_key"].nunique()),
            "items": da._clean_records(g.head(TOP_N)[LOW_COLUMNS])}


def _molecule_detail(comp: pd.DataFrame, cheapest: pd.DataFrame, view: pd.DataFrame, key: str) -> Optional[dict]:
    """One product: its products-view row, its purchases per hospital x brand x supplier over the
    selected months (points, cheapest first) and its suppliers with their value-weighted price
    (by_vendor). None when the selection did not buy it."""
    hit = view[view["molecule_key"] == key]
    if hit.empty:
        return None
    rows = comp[comp["molecule_key"] == key].assign(qty=lambda d: d["qty"].astype(float))
    pts = rows.groupby(["plant", "material", "vendor"]).agg(
        material_desc=("material_desc", "last"), vendor_name=("vendor_name", "last"), lines=("lines", "sum"),
        qty=("qty", "sum"), value=("value", "sum"), unit_price_min=("unit_price_min", "min"),
        unit_price_max=("unit_price_max", "max")).reset_index()
    pts["unit_price"] = pts["value"] / pts["qty"].where(pts["qty"] != 0)
    pts["unusually_low"] = (pts["unit_price"] < P.LOW_RATIO * float(cheapest.loc[key, "cheapest_price"])
                            ).fillna(False).astype(bool)
    names = _names()
    pts["plant_name"] = pts["plant"].map(lambda p: names.get(p, p))
    pts = _round_prices(F._round(pts, money=("value",), qty=("qty",)))
    pts = pts.sort_values(["unit_price", "value", "plant", "material", "vendor"],
                          ascending=[True, False, True, True, True], kind="mergesort")
    vend = _vendors(rows, cheapest["cheapest_price"])
    vend["wavg_price"] = (vend["value"] / vend["qty"].where(vend["qty"] != 0)).astype(float).round(PRICE_DECIMALS)
    return {**da._clean_records(hit)[0], "points": da._clean_records(pts[POINT_COLUMNS]),
            "by_vendor": da._clean_records(vend[VENDOR_COLUMNS + ["qty", "wavg_price"]])}


@router.get("/spend/prices/overview")
def spend_prices_overview(Plant: str = Query(None), Month: str = Query(None), Segment: str = Query(None),
                          Category: str = Query(None), Molecule: str = Query(None), Vendor: str = Query(None),
                          Material: str = Query(None), Type: str = Query(None), Payer: str = Query(None),
                          limit: int = Query(MOLECULE_LIMIT)):
    """Price variation for the selection. Molecule= (a molecule_key) picks the product whose
    purchases and vendors are shown (default: the one with the most extra paid); `limit` caps
    the products sent for the chart (most extra paid first)."""
    sc = _scope(C.PRICES, Plant, Month, Segment, Category, Vendor, Material)
    molecule = (F._str(Molecule) or "").upper() or None
    return _prices_overview_cached(sc, molecule, _limit(limit), _ignored(Type=Type, Payer=Payer), _version())


@_cache
def _prices_overview_cached(sc: Scope, molecule: Optional[str], limit: int, ignored: tuple, version: tuple) -> dict:
    t = _table(C.PRICES)
    rows = _cut(t, sc)
    comp = _compared(rows)
    cheapest = _cheapest(_compared(_cut(t, sc, drill=False)))
    view = _products(comp, cheapest)

    value, cvalue = float(rows["value"].sum()), float(comp["value"].sum())
    extra = float(_extra(comp, cheapest["cheapest_price"]).sum())
    low = _low_prices(comp, cheapest)
    network = _cheapest(_compared(_cut(t, sc, plant=False, drill=False)))
    by_hospital = _by_hospital(_compared(_cut(t, sc, plant=False)), network, sc)
    headline = {
        "molecules": int(len(view)), "molecules_with_spread": int(view["varies"].sum()),
        "extra_paid_vs_cheapest": F._money(extra), "extra_pct": F._share(extra, cvalue),
        "low_rows": low["rows"], "low_value": low["value"],
        "value": F._money(value), "compared_value": F._money(cvalue), "compared_pct": F._share(cvalue, value),
        "lines": int(rows["lines"].sum()), "brands": int(rows["material"].nunique()),
        "vendors": int(rows["vendor"].nunique()), "hospitals": int(rows["plant"].nunique()),
        # one hospital: what it paid above the cheapest price ANY hospital paid (its ranking row)
        "extra_paid_vs_network_cheapest": (next((h["extra_paid"] for h in by_hospital if h["selected"]), 0.0)
                                           if sc.plant else None),
    }

    key = molecule or (str(view["molecule_key"].iloc[0]) if len(view) else None)
    chosen = _molecule_detail(comp, cheapest, view, key) if key else None
    vendors = _vendors(comp, cheapest["cheapest_price"])

    names = _names()
    notes = [_note_scope(), NOTE_PRICE_VALUES, NOTE_NOT_COMPARED, NOTE_PRICE_VIEWS, *_ignored_notes(ignored)]
    if len(sc.months) > 1:
        notes.append(NOTE_PRICE_MONTHS)
    if sc.vendor or sc.material:
        notes.append(NOTE_DRILL)
    if molecule and chosen is None:
        notes.append(f"{molecule} was not bought at a comparable price in this selection.")
    if sc.plant and rows.empty:
        notes.append(f"{names.get(sc.plant, sc.plant)} has no purchase in this selection.")
    return {
        "headline": headline,
        "scope": _scope_echo(sc, molecule=key, ignored=ignored),
        "molecules": da._clean_records(view.head(limit)[CHART_COLUMNS]),
        "molecules_total": int(len(view)),
        "molecule": chosen,
        "vendors": da._clean_records(vendors.head(TOP_N)[VENDOR_COLUMNS]),
        "vendors_total": int(len(vendors)),
        "by_hospital": by_hospital,
        "by_month": _by_month(_cut(t, sc, months=False), _cut(t, sc, months=False, drill=False),
                              _periods(C.PRICES)),
        "not_compared": _not_compared(rows),
        "low_prices": low,
        "months_available": _months_available(C.PRICES),
        "notes": notes,
        "data_as_of": _built_at(),
    }


LIST_SUMS = ["value", "extra_paid", "lines"]
# The list's own filters (Category comes from its column).
LIST_FACETS = (tq.Facet("form", "Form", order="label"),
               tq.Facet("varies", "Bought at more than one price", value_label=tq.yes_no))


def _vendor_name(vendor: str) -> str:
    """Vendor= in words: the supplier's name as the latest month spells it."""
    t = _table(C.PRICES)
    hit = t.loc[(t["vendor"] == K.norm_text(vendor)) | (t["vendor_code"].astype(str) == F._code(vendor)),
                "vendor_name"]
    return str(hit.iloc[-1]) if len(hit) else vendor


def _chips(sc: Scope) -> tuple:
    """The page filters a list was aggregated over, as table scope chips."""
    chips = [tq.plant_chip(sc.plant), tq.months_chip(sc.months, _periods(C.PRICES))]
    if sc.segment:
        chips.append(tq.chip("Segment", "Material type", sc.segment, SEGMENT_LABELS[sc.segment]))
    if sc.vendor:
        chips.append(tq.chip("Vendor", "Vendor", sc.vendor, _vendor_name(sc.vendor)))
    if sc.material:
        chips.append(tq.chip("Material", "Brand", sc.material))
    return tuple(chips)


@router.get("/spend/prices/molecules")
def spend_prices_molecules(request: Request, Plant: str = Query(None), Month: str = Query(None),
                           Segment: str = Query(None), Category: str = Query(None), Vendor: str = Query(None),
                           Material: str = Query(None)):
    """Every product of the selection, most extra paid first (Contract A: paging, sort, search,
    f.form / f.varies / f.category facets, sums for the Total row). Vendor= / Material= keep
    one vendor's or brand's purchases, set against the selection's cheapest."""
    sc = _scope(C.PRICES, Plant, Month, Segment, Category, Vendor, Material)
    return F._page(request, _molecules_cached(sc, _version()), LIST_SUMS, facets=LIST_FACETS, scope=_chips(sc))


@_cache
def _molecules_cached(sc: Scope, version: tuple) -> pd.DataFrame:
    t = _table(C.PRICES)
    return _products(_compared(_cut(t, sc)), _cheapest(_compared(_cut(t, sc, drill=False))))


# ── Generic vs branded ────────────────────────────────────────────────────────
TOP_MOLECULE_COLUMNS = ["molecule_key", "minor_group", "strength", "form", "brands", "companies", "branded_value",
                        "compared_value", "extra_vs_generic", "extra_vs_generic_pct", "share_pct", "generic_material",
                        "generic_desc", "generic_price", "generic_period", "plants"]


@_cache
def _item_master_periods_at(version: tuple) -> frozenset:
    """The months whose purchases carry their own month's Item Master class."""
    try:
        g = _table(C.GENERIC)
    except HTTPException:
        return frozenset()
    return frozenset(g.loc[g["formulary_flag"].notna(), "period"].astype(str).unique())


def _item_master_periods() -> frozenset:
    return _item_master_periods_at(_version())


def _generic_sums(df: pd.DataFrame, keys: list) -> pd.DataFrame:
    """Per `keys` ([] = one row): value split generic / branded, lines, and the cost impact."""
    branded = df["class"].astype(str).eq(Gn.BRANDED)
    value, lines = df["value"].astype(float), df["lines"].astype("int64")
    d = pd.DataFrame({"value": value, "generic_value": value.where(~branded, 0.0),
                      "branded_value": value.where(branded, 0.0), "lines": lines,
                      "branded_lines": lines.where(branded, 0),
                      "compared_value": df["compared_value"].astype(float),
                      "extra_vs_generic": df["extra_vs_generic"].astype(float)}, index=df.index)
    if keys:
        for k in keys:
            d[k] = df[k].astype(str)
        return d.groupby(keys, sort=True).sum().reset_index()
    return d.sum().to_frame().T


def _generic_block(r) -> dict:
    value, branded, compared = float(r["value"]), float(r["branded_value"]), float(r["compared_value"])
    return {"value": F._money(value), "generic_value": F._money(r["generic_value"]),
            "branded_value": F._money(branded),
            "generic_pct": F._share(r["generic_value"], value), "branded_pct": F._share(branded, value),
            "extra_vs_generic": F._money(r["extra_vs_generic"]), "compared_value": F._money(compared),
            "extra_vs_generic_pct": F._share(r["extra_vs_generic"], compared),
            "branded_not_compared": F._money(branded - compared),
            "lines": int(r["lines"]), "branded_lines": int(r["branded_lines"])}


GENERIC_EMPTY = _generic_block({"value": 0.0, "generic_value": 0.0, "branded_value": 0.0, "compared_value": 0.0,
                                "extra_vs_generic": 0.0, "lines": 0, "branded_lines": 0})


def _split(d: pd.DataFrame, col: str, labels: dict, total: float) -> list:
    """`d`'s value by `col`, in `labels` order (then any other value), each a share of `total`."""
    part = d[d[col].notna()]
    sums = part.groupby(part[col].astype(str))["value"].sum()
    return [{"key": k, "label": labels.get(k, k), "value": F._money(sums.get(k, 0.0)),
             "share_pct": F._share(sums.get(k, 0.0), total)}
            for k in [*labels, *(x for x in sums.index if x not in labels)]]


def _compliance(cur: pd.DataFrame, months: tuple) -> dict:
    """Each side's value by its month's own Item Master class — the formulary flag, and the
    tier group of a FORMULARY item — each a share of the side's value; months with an Item
    Master only (D7)."""
    im = _item_master_periods()
    with_im = [p for p in months if p in im]
    d = cur[cur["period"].astype(str).isin(with_im) & cur["formulary_flag"].notna()]
    by_class = []
    for cls in CLASS_ORDER:
        c = d[d["class"].astype(str) == cls]
        total = float(c["value"].sum())
        by_class.append({"class": cls, "label": CLASS_LABELS[cls], "value": F._money(total),
                         "flags": _split(c, "formulary_flag", FLAG_LABELS, total),
                         "tiers": _split(c, "tier_group", TIER_LABELS, total)})
    return {"months": with_im, "months_without_item_master": [p for p in months if p not in im],
            "available": bool(with_im), "by_class": by_class}


def _top_branded(cur: pd.DataFrame) -> list:
    """The branded products a generic of the same product undercut, most extra first: what the
    branded purchases cost above the cheapest generic, the value set against it, the brands
    (biggest first) and their companies, and the cheapest generic of the latest month."""
    b = cur[cur["class"].astype(str).eq(Gn.BRANDED) & cur["molecule_key"].notna()]
    hit = b[b["compared_value"] > 0]
    g = hit.groupby("molecule_key").agg(compared_value=("compared_value", "sum"),
                                        extra_vs_generic=("extra_vs_generic", "sum"), plants=("plant", "nunique"))
    g = g[g["extra_vs_generic"] > 0]
    if g.empty:
        return []
    g["branded_value"] = b.groupby("molecule_key")["value"].sum()
    brands = (b.groupby(["molecule_key", "material_desc"])["value"].sum().reset_index()
              .sort_values(["molecule_key", "value"], ascending=[True, False], kind="mergesort"))
    g["brands"] = brands.groupby("molecule_key")["material_desc"].agg(list)
    g["companies"] = b.groupby("molecule_key")["originator_company"].agg(
        lambda s: sorted(s.dropna().astype(str).unique()))
    latest = (hit[hit["generic_desc"].notna()].sort_values("period", kind="mergesort")
              .drop_duplicates("molecule_key", keep="last").set_index("molecule_key"))
    g = g.join(latest[["minor_group", "strength", "form", "generic_material", "generic_desc", "generic_price",
                       "period"]].rename(columns={"period": "generic_period"}))
    tot = float(cur["extra_vs_generic"].sum())
    g["share_pct"] = g["extra_vs_generic"] / tot * 100 if tot else np.nan
    g["extra_vs_generic_pct"] = g["extra_vs_generic"] / g["compared_value"] * 100
    g = _round_prices(F._round(g, money=("branded_value", "compared_value", "extra_vs_generic"),
                               pct=("share_pct", "extra_vs_generic_pct")), ("generic_price",))
    g = g.reset_index().sort_values(["extra_vs_generic", "molecule_key"], ascending=[False, True], kind="mergesort")
    return da._clean_records(g.head(TOP_N)[TOP_MOLECULE_COLUMNS])


def _definition() -> dict:
    """The generic vs branded definition the build used (its validation report), shown with
    the 'Estimated definition' badge."""
    g = (_validation() or {}).get("generic") or {}
    return {"text": g.get("definition") or Gn.DEFINITION, "estimated": bool(g.get("estimated", True)),
            "label": ESTIMATED_LABEL, "originator_companies": g.get("originator_companies"),
            "originator_patterns": g.get("originator_patterns"), "overrides": g.get("overrides") or [],
            "files": g.get("files") or {"originators": Gn.ORIGINATOR_FILE, "overrides": Gn.OVERRIDES_FILE},
            "class_sources": Gn.CLASS_SOURCES}


@router.get("/spend/generic/overview")
def spend_generic_overview(Plant: str = Query(None), Month: str = Query(None), Segment: str = Query(None),
                           Category: str = Query(None), Material: str = Query(None), Type: str = Query(None),
                           Payer: str = Query(None)):
    """Generic vs branded purchases for the selection; Material= narrows every view to one
    brand."""
    sc = _scope(C.GENERIC, Plant, Month, Segment, Category, Material=Material)
    return _generic_overview_cached(sc, _ignored(Type=Type, Payer=Payer), _version())


@_cache
def _generic_overview_cached(sc: Scope, ignored: tuple, version: tuple) -> dict:
    t = _table(C.GENERIC)
    cur = _cut(t, sc)
    names = _names()
    im = _item_master_periods()

    branded = cur["class"].astype(str).eq(Gn.BRANDED)
    headline = {**_generic_block(_generic_sums(cur, []).iloc[0]), "brands": int(cur["material"].nunique()),
                "generic_brands": int(cur.loc[~branded, "material"].nunique()),
                "branded_brands": int(cur.loc[branded, "material"].nunique()),
                "hospitals": int(cur["plant"].nunique())}

    mb = _generic_sums(_cut(t, sc, months=False), ["period"]).set_index("period")
    by_month = [{"period": p, "label": tq.month_label(p), "item_master": p in im,
                 **(_generic_block(mb.loc[p]) if p in mb.index else GENERIC_EMPTY)} for p in _periods(C.GENERIC)]

    hb = _generic_sums(_cut(t, sc, plant=False), ["plant"])
    by_hospital = [{"plant": r["plant"], "name": names.get(r["plant"], r["plant"]), **_generic_block(r),
                    "rank": None, "selected": r["plant"] == sc.plant} for r in hb.to_dict("records")]
    # most paid above the generics first
    by_hospital.sort(key=lambda x: (-x["extra_vs_generic"], -x["branded_value"], x["plant"]))
    for i, x in enumerate(by_hospital):
        x["rank"] = i + 1

    sb = _generic_sums(_cut(t, sc, segment=False), ["segment"]).set_index("segment")
    by_segment = [{"key": k, "label": label, **(_generic_block(sb.loc[k]) if k in sb.index else GENERIC_EMPTY)}
                  for k, label in SEGMENT_LABELS.items()]

    src = cur.groupby(cur["class_source"].astype(str)).agg(value=("value", "sum"), lines=("lines", "sum"))
    by_source = [{"key": k, "label": label, "value": F._money(src.loc[k, "value"]) if k in src.index else 0.0,
                  "lines": int(src.loc[k, "lines"]) if k in src.index else 0}
                 for k, label in Gn.CLASS_SOURCES.items()]

    compliance = _compliance(cur, sc.months)
    notes = [_note_scope(), NOTE_GENERIC_VALUES, NOTE_GENERIC_VIEWS, *_ignored_notes(ignored)]
    missing = compliance["months_without_item_master"]
    if missing:
        words = {"missing": ", ".join(tq.month_label(p) for p in missing), "verb": "has" if len(missing) == 1 else "have"}
        notes.append(NOTE_COMPLIANCE.format(months=", ".join(tq.month_label(p) for p in compliance["months"]), **words)
                     if compliance["months"] else NOTE_NO_COMPLIANCE.format(**words))
    if len(sc.months) > 1:
        notes.append(NOTE_MONTHS)
    if sc.plant and cur.empty:
        notes.append(f"{names.get(sc.plant, sc.plant)} has no purchase in this selection.")
    return {"headline": headline, "scope": _scope_echo(sc, ignored=ignored), "by_month": by_month,
            "by_hospital": by_hospital, "by_segment": by_segment, "by_source": by_source,
            "compliance": compliance, "top_molecules": _top_branded(cur), "definition": _definition(),
            "estimated": True, "months_available": _months_available(C.GENERIC), "notes": notes,
            "data_as_of": _built_at()}
