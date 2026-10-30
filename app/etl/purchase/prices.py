"""Price variation: what HCG paid per unit for one product, across hospitals, vendors and brands.

Lines: the purchase scope (config.Scope) of every GRN month, every formulary class. Per line:

  unit price    'Total Amount without tax' / GR Qty: the price after discount (decision D5;
                Net Price when the price unit is 1), to UNIT_PRICE_DECIMALS so equal prices
                compare equal
  product       the material's molecule (Item Master 'Minor Group'), strength, dosage form and
                kind as app/etl/purchase/molecule.py reads them. molecule_key
                'CETUXIMAB|100MG|INJ|' joins the brands of one product; a product whose
                molecule, strength or form is not written has no key and is not compared.
                Description and minor group: the month's Item Master, else the nearest other
                month's (so a product keeps one key in every month), else the GRN's own.
  price_status  whether the line's unit price can be compared (PRICE_STATUSES), first that
                applies:
                  no_price    no positive amount or quantity (free or bonus stock, a credit)
                  token       a unit price under TOKEN_SHARE of the line's own MRP without GST:
                              a placeholder, not a purchase price (OPDYTA booked at Rs 0.01)
                  other_unit  the line is priced per strip or pack while the brand is bought
                              per unit (or the reverse). With HCG's CASH&TPA MRP for the brand
                              (a per-unit MRP): the line's MRP is more than UNIT_RATIO times
                              away from it. Without it: the line's price AND MRP are both more
                              than UNIT_RATIO times away from the brand's month median (the
                              lower median, so of two lines a pack apart the per-unit one stays).
                  ok          a comparable unit price

The published table (PRICES_COLUMNS) sums the lines per period x plant x material x vendor x
price_status, with the unit price range. Per product (products()) over the rows with
price_status 'ok' and a molecule_key:
  cheapest      the USUAL cheapest price (usual_cheapest): the lowest unit price at which,
                counting the product's quantity up from its cheapest purchase, at least
                USUAL_SHARE of it was bought. The plain minimum let one odd purchase set the
                benchmark for everyone — a single bevacizumab vial at Rs 7,500 against Rs 39,000
                everywhere else turned Rs 98 L of ordinary purchases into 'extra paid'
  lowest        min unit_price_min: the lowest price anyone paid, shown beside it
  dearest       max unit_price_max; spread = dearest / cheapest
  extra paid    what each row paid above the cheapest: sum of max(0, value - cheapest x qty),
                a row at the row's own average price; a row under the cheapest adds nothing
  unusually low a row priced under LOW_RATIO x the cheapest: still compared, and listed for HCG
                to check (a one-off deal or a keying slip)
No GR, PO or invoice number and no line-level row is published.
"""
from __future__ import annotations

from typing import Mapping, Optional

import numpy as np
import pandas as pd

from app.etl.formulary import classify as K
from app.etl.purchase import molecule as Mo

# --- Vocabulary -------------------------------------------------------------------------------
OK, NO_PRICE, TOKEN, OTHER_UNIT = "ok", "no_price", "token", "other_unit"
PRICE_STATUSES = {
    OK: "a comparable unit price",
    NO_PRICE: "no positive amount or quantity (free or bonus stock, a credit)",
    TOKEN: "a unit price under 0.1% of the line's MRP without GST: a placeholder (Rs 0.01)",
    OTHER_UNIT: "priced per strip or pack while the brand is bought per unit (or the reverse)",
}
TOKEN_SHARE = 0.001              # a unit price under 0.1% of its MRP without GST is a token
UNIT_RATIO = 4.0                 # more than 4x away from the brand's unit: another unit
UNIT_PRICE_DECIMALS = 4
KEY_SEP = "|"
SPREAD_MIN = 0.01                # report: a product varies when its dearest is >= 1% above its cheapest
USUAL_SHARE = 0.10               # the usual cheapest: the lowest price at which >= 10% of the quantity was bought
LOW_RATIO = 0.5                  # a purchase under half the usual cheapest is unusually low (listed to check)
TOP_PRODUCTS = 10                # report: products listed per month

# --- The published table: purchase_prices ---------------------------------------------------
PRODUCT_COLUMNS = ["molecule_key", "molecule", "strength", "form", "kind"]
GROUP_KEYS = ["period", "plant", "material", "vendor_code", "price_status"]
ATTRIBUTES = ["category", "segment", "material_desc", "minor_group", "manufacturer",
              "vendor_name"] + PRODUCT_COLUMNS
TEXT_COLUMNS = ["period", "plant", "category", "segment", "material", "material_desc",
                "minor_group", "manufacturer", "vendor_code", "vendor_name", "price_status"
                ] + PRODUCT_COLUMNS
SUMS = ["lines", "qty", "value"]
PRICE_COLUMNS = ["unit_price_min", "unit_price_max"]
PRICES_COLUMNS = TEXT_COLUMNS + SUMS + PRICE_COLUMNS
PRICES_SORT = ["period", "material", "plant", "vendor_code", "price_status"]   # a brand's rows together


# --------------------------------------------------------------------------- #
# What each material is
# --------------------------------------------------------------------------- #
def _month_index(period: str) -> int:
    y, m = map(int, period.split("-"))
    return 12 * y + m


def nearest_first(period: str, periods) -> list:
    """`periods` in the order `period` reads them: itself, then the nearest (earlier on a tie)."""
    return sorted(periods, key=lambda p: (p != period, abs(_month_index(p) - _month_index(period)), p))


def _first_written(item_masters: Mapping[str, pd.DataFrame], order: list, column: str) -> dict:
    out: dict = {}
    for p in order:
        im = item_masters[p]
        for m, v in zip(im["material"], im[column]):
            if m not in out and isinstance(v, str) and v.strip():
                out[m] = v
    return out


def with_item_master_names(lines: pd.DataFrame, item_masters: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """The lines with material_desc and minor_group from the Item Master of their month,
    else of the nearest other month that writes them, else the GRN's own."""
    out = lines.copy()
    for period in out["period"].unique():
        rows = out["period"] == period
        order = nearest_first(period, item_masters)
        for col in ("material_desc", "minor_group"):
            names = _first_written(item_masters, order, col)
            out.loc[rows, col] = out.loc[rows, "material"].map(names).fillna(out.loc[rows, col])
    return out


def product_key(minor_group, desc) -> dict:
    """The product as price variation compares it (PRODUCT_COLUMNS): molecule, strength, form,
    kind, and molecule_key 'MOLECULE|STRENGTH|FORM|KIND' (None when the molecule, strength or
    form is not written)."""
    molecule = Mo.molecule_key(minor_group)
    strength, form, kind = Mo.strength_key(desc), Mo.form_key(desc), Mo.kind_key(desc, minor_group)
    key = KEY_SEP.join([molecule, strength, form, kind]) if (molecule and strength and form) else None
    return {"molecule_key": key, "molecule": molecule, "strength": strength or None,
            "form": form or None, "kind": kind or None}


def with_products(lines: pd.DataFrame) -> pd.DataFrame:
    """The lines with PRODUCT_COLUMNS (one parse per minor group x description)."""
    pairs = lines[["minor_group", "material_desc"]].drop_duplicates()
    keys = pd.DataFrame([product_key(g, d) for g, d in zip(pairs["minor_group"], pairs["material_desc"])],
                        columns=PRODUCT_COLUMNS, index=pairs.index)
    return lines.merge(pd.concat([pairs, keys], axis=1), on=["minor_group", "material_desc"],
                       how="left", validate="many_to_one")


# --------------------------------------------------------------------------- #
# Unit prices and which can be compared
# --------------------------------------------------------------------------- #
def unit_prices(lines: pd.DataFrame) -> pd.Series:
    """Total Amount without tax / GR Qty (decision D5); NaN without a positive quantity."""
    qty = lines["gr_qty"].where(lines["gr_qty"] > 0)
    return (lines["total_wo_tax"] / qty).round(UNIT_PRICE_DECIMALS)


def _far(a: pd.Series, b: pd.Series) -> pd.Series:
    """a is more than UNIT_RATIO times away from b (False when either is missing)."""
    return ((a > UNIT_RATIO * b) | (a * UNIT_RATIO < b)).fillna(False).astype(bool)


def price_statuses(lines: pd.DataFrame, list_mrp: Mapping[str, float]) -> pd.Series:
    """PRICE_STATUSES for each line (module docstring). list_mrp: material -> HCG's CASH&TPA
    MRP incl GST, per unit of the brand."""
    price = unit_prices(lines)
    mrp = lines["unit_mrp"].where(lines["unit_mrp"] > 0)
    no_price = ~((lines["total_wo_tax"] > 0) & (lines["gr_qty"] > 0))
    token = ~no_price & (price < TOKEN_SHARE * (mrp / (100 + lines["tax_pct"]) * 100))
    priced = ~no_price & ~token
    reference = lines["material"].map(list_mrp).astype("float64")
    by_list = reference.notna() & _far(mrp, reference)
    keys = ["period", "material"]
    base = lines.loc[priced, keys].assign(price=price[priced], mrp=mrp[priced])
    med = base.groupby(keys).quantile(0.5, interpolation="lower")
    med = lines[keys].merge(med, left_on=keys, right_index=True, how="left").set_index(lines.index)
    by_median = reference.isna() & _far(price, med["price"]) & _far(mrp, med["mrp"])
    other = priced & (by_list | by_median)
    status = np.select([no_price, token, other], [NO_PRICE, TOKEN, OTHER_UNIT], OK)
    return pd.Series(status, index=lines.index, dtype="str")


def priced_lines(lines: pd.DataFrame, item_masters: Mapping[str, pd.DataFrame],
                 list_mrp: Mapping[str, float]) -> pd.DataFrame:
    """The scope's lines with Item Master names, category and segment, the product
    (PRODUCT_COLUMNS), unit_price (NaN without a price) and price_status. In memory only."""
    out = with_products(with_item_master_names(lines, item_masters))
    out["category"] = out["material"].map(K.category)
    out["segment"] = out["category"].map(K.SEGMENT_KEY)
    out["price_status"] = price_statuses(out, list_mrp)
    out["unit_price"] = unit_prices(out).where(out["price_status"].ne(NO_PRICE))
    return out


def aggregate(priced: pd.DataFrame) -> pd.DataFrame:
    """priced_lines -> purchase_prices rows (PRICES_COLUMNS): one row per period x plant x
    material x vendor x price_status."""
    g = priced.assign(lines=1, qty=priced["gr_qty"], value=priced["total_wo_tax"]).groupby(
        GROUP_KEYS, dropna=False, sort=False)
    out = g.agg(**{c: (c, "first") for c in ATTRIBUTES}, **{c: (c, "sum") for c in SUMS},
                unit_price_min=("unit_price", "min"), unit_price_max=("unit_price", "max")).reset_index()
    out = out[PRICES_COLUMNS]
    out[TEXT_COLUMNS] = out[TEXT_COLUMNS].astype("str")
    out[["qty", "value"] + PRICE_COLUMNS] = out[["qty", "value"] + PRICE_COLUMNS].astype("float64")
    out["lines"] = out["lines"].astype("int64")
    return out.sort_values(PRICES_SORT, kind="stable", ignore_index=True)


# --------------------------------------------------------------------------- #
# Per product (the API's view; the report uses it too)
# --------------------------------------------------------------------------- #
PRODUCT_VIEW_COLUMNS = ["molecule_key", "minor_group", "strength", "form", "cheapest_price", "min_price",
                        "max_price", "wavg_price", "spread_ratio", "qty", "value", "extra_paid",
                        "low_rows", "low_value", "brands", "vendors", "plants"]


def usual_cheapest(df: pd.DataFrame, by, price: str, qty: str = "qty", share: float = USUAL_SHARE,
                   carry: tuple = ()) -> pd.DataFrame:
    """Per group `by` (index): the usual cheapest `price` — the lowest price at which, counting
    the group's `qty` up from its cheapest purchase, at least `share` of it was bought — and
    the `carry` columns of the purchase that sets it (of equal prices, the biggest by value
    when there is a value column). Purchases without a positive quantity or a price take no
    part. A group bought at a single price, or whose cheapest purchase alone holds `share` of
    the quantity, gets its plain minimum."""
    by = [by] if isinstance(by, str) else list(by)
    d = df[df[qty].astype(float).gt(0) & df[price].notna()]
    keys, asc = by + [price], [True] * (len(by) + 1)
    if "value" in d.columns:
        keys, asc = keys + ["value"], asc + [False]
    d = d.sort_values(keys, ascending=asc, kind="mergesort")
    q = d[qty].astype(float)
    reached = q.groupby([d[k] for k in by], sort=False).cumsum() >= share * q.groupby(
        [d[k] for k in by], sort=False).transform("sum") * (1 - 1e-12)
    return d[reached].drop_duplicates(by).set_index(by)[[price, *carry]]


def row_prices(rows: pd.DataFrame) -> pd.Series:
    """Each purchase_prices row's own unit price: its value / its quantity (NaN without one)."""
    qty = rows["qty"].astype(float)
    return rows["value"].astype(float) / qty.where(qty > 0)


def extra_paid(rows: pd.DataFrame, cheapest: pd.Series) -> pd.Series:
    """What each row paid above its cheapest price (`cheapest`: a price per row, aligned to
    `rows`): max(0, value - cheapest x qty); 0 for a row without one."""
    over = rows["value"].astype(float) - cheapest.astype(float) * rows["qty"].astype(float)
    return over.clip(lower=0.0).fillna(0.0)


def unusually_low(rows: pd.DataFrame, cheapest: pd.Series) -> pd.Series:
    """Whether each row's own price is under LOW_RATIO x its cheapest price (aligned to `rows`)."""
    return (row_prices(rows) < LOW_RATIO * cheapest.astype(float)).fillna(False).astype(bool)


def vendor_keys(names: pd.Series) -> pd.Series:
    """The supplier each row's 'Goods supplier name' names (spacing and case aside): one
    supplier holds several SAP supplier codes (Vardhman Health Specialities has one per
    region), so suppliers are counted by name."""
    names = names.astype(str)
    return names.map({n: K.norm_text(n) for n in names.unique()})


def products(table: pd.DataFrame) -> pd.DataFrame:
    """purchase_prices rows (any cut) -> one row per molecule_key of the comparable rows
    (PRODUCT_VIEW_COLUMNS): the usual cheapest, lowest and dearest unit price, the
    value-weighted price, the spread, the extra paid above the usual cheapest, the rows priced
    unusually low, and how many brands, vendors (suppliers by name, vendor_keys) and centres;
    most extra paid first. minor_group is the one on most rows (HCG's spelling)."""
    ok = table[table["price_status"].eq(OK) & table["molecule_key"].notna()]
    if ok.empty:
        return pd.DataFrame(columns=PRODUCT_VIEW_COLUMNS)
    ok = ok.assign(_vendor=vendor_keys(ok["vendor_name"]), _price=row_prices(ok))
    cheapest = usual_cheapest(ok, "molecule_key", "_price")["_price"]
    per_row = ok["molecule_key"].map(cheapest)
    low = unusually_low(ok, per_row)
    ok = ok.assign(_extra=extra_paid(ok, per_row), _low_rows=low.astype(int),
                   _low_value=ok["value"].where(low, 0.0))
    out = ok.groupby("molecule_key").agg(
        minor_group=("minor_group", lambda s: s.value_counts().index[0] if s.notna().any() else None),
        strength=("strength", "first"), form=("form", "first"),
        min_price=("unit_price_min", "min"), max_price=("unit_price_max", "max"),
        qty=("qty", "sum"), value=("value", "sum"), extra_paid=("_extra", "sum"),
        low_rows=("_low_rows", "sum"), low_value=("_low_value", "sum"), brands=("material", "nunique"),
        vendors=("_vendor", "nunique"), plants=("plant", "nunique"))
    out["cheapest_price"] = cheapest.reindex(out.index)
    out["wavg_price"] = out["value"] / out["qty"]
    out["spread_ratio"] = out["max_price"] / out["cheapest_price"]
    out = out.sort_values(["extra_paid", "value"], ascending=False, kind="stable").reset_index()
    return out[PRODUCT_VIEW_COLUMNS]


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #
def _r(v, nd=2) -> Optional[float]:
    return None if v is None or pd.isna(v) else round(float(v), nd)


def month_report(table: pd.DataFrame) -> dict:
    """The prices section of one month's validation report (counts and rupees only)."""
    by = table.groupby("price_status")[["lines", "value"]].sum()
    keyed = table[table["price_status"].eq(OK) & table["molecule_key"].notna()]
    per = products(table)
    return {
        "lines": int(table["lines"].sum()), "value": _r(table["value"].sum()),
        "by_status": {s: {"lines": int(by.loc[s, "lines"]), "value": _r(by.loc[s, "value"])}
                      for s in PRICE_STATUSES if s in by.index},
        "compared": {"lines": int(keyed["lines"].sum()), "value": _r(keyed["value"].sum()),
                     "materials": int(keyed["material"].nunique())},
        "products": int(len(per)),
        "products_with_spread": int((per["spread_ratio"] >= 1 + SPREAD_MIN).sum()),
        "extra_paid_vs_cheapest": _r(per["extra_paid"].sum()),
        "unusually_low": {"rows": int(per["low_rows"].sum()), "value": _r(per["low_value"].sum()),
                          "products": int((per["low_rows"] > 0).sum())},
        "top_products": [{"molecule_key": r.molecule_key, "minor_group": r.minor_group,
                          "cheapest_price": _r(r.cheapest_price), "min_price": _r(r.min_price),
                          "max_price": _r(r.max_price),
                          "brands": int(r.brands), "vendors": int(r.vendors), "value": _r(r.value),
                          "extra_paid": _r(r.extra_paid)}
                         for r in per.head(TOP_PRODUCTS).itertuples(index=False)],
    }


def summary_line(months: dict) -> str:
    """One log line: the months built and the latest month's headline."""
    if not months:
        return "[purchase] prices: no month built"
    p = max(months)
    rep = months[p]
    off = {s: rep["by_status"].get(s, {}).get("lines", 0) for s in (TOKEN, OTHER_UNIT, NO_PRICE)}
    return (f"[purchase] prices: {len(months)} months; {p}: {rep['products']:,} products, "
            f"{rep['products_with_spread']:,} bought at more than one price, "
            f"{rep['extra_paid_vs_cheapest']:,.2f} paid above the cheapest; not compared: "
            + ", ".join(f"{n:,} {s}" for s, n in off.items()) + " lines")


def report_section() -> dict:
    """The build-wide prices section of purchase_validation.json: the method."""
    return {
        "rules": {
            "lines": "every purchase-scope line of every GRN month, all formulary classes",
            "unit_price": "Total Amount without tax / GR Qty (the price after discount, decision D5)",
            "product": "molecule_key = molecule (Item Master minor group, salts and synonyms made "
                       "one) | strength | dosage form | kind; no key when the molecule, strength "
                       "or form is not written",
            "price_status": PRICE_STATUSES,
            "other_unit": f"with HCG's CASH&TPA MRP for the brand: the line's MRP more than "
                          f"{UNIT_RATIO:g}x away from it; else the line's price and MRP both more "
                          f"than {UNIT_RATIO:g}x away from the brand's month median (lower median)",
            "per_product": f"over rows with price_status ok and a molecule_key: cheapest = the usual "
                           f"cheapest, the lowest row price (value / qty) at which at least "
                           f"{USUAL_SHARE:.0%} of the product's quantity was bought, counting up from "
                           f"the cheapest; lowest = min unit_price_min; dearest = max unit_price_max; "
                           f"value-weighted price = sum value / sum qty; extra paid = sum over rows of "
                           f"max(0, value - cheapest x qty)",
            "unusually_low": f"a row priced under {LOW_RATIO:g} x its product's usual cheapest: "
                             f"still compared, listed for HCG to check",
            "report": f"a product varies when its dearest is at least {SPREAD_MIN:.0%} above its "
                      f"cheapest; extra paid per month over all centres",
        },
    }


# --------------------------------------------------------------------------- #
# The table of the build
# --------------------------------------------------------------------------- #
def build_table(lines: pd.DataFrame, item_masters: Mapping[str, pd.DataFrame],
                list_mrp: Mapping[str, float]) -> tuple:
    """purchase_prices for the lines' months -> (table, {period: month report})."""
    table = aggregate(priced_lines(lines, item_masters, list_mrp))
    months = {p: month_report(table[table["period"] == p]) for p in sorted(table["period"].unique())}
    return table, months
