# tests/test_purchase_prices_products.py — the name prices.products() gives each product: the
# minor group written on most of its rows, of equal counts the one written first.
#
# prices.most_common does this for every product at once. It replaced a value_counts() call per
# product (0.8 s of the prices overview's 1.5 s), so it is held here to that call's answer, on
# made-up rows and on random ones; the real tables were compared before and after in full when
# it was written (products() over 33 cuts and row orders, the month reports, 32 API answers).
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.etl.purchase import prices as P


def _per_group(df: pd.DataFrame) -> dict:
    """The call most_common replaced, group by group."""
    return {k: (s.value_counts().index[0] if s.notna().any() else None)
            for k, s in df.groupby("key", sort=False)["name"]}


def test_most_written_wins_and_a_tie_goes_to_the_first_written():
    df = pd.DataFrame({"key": ["A", "A", "A", "B", "B", "C", "C", "D"],
                       "name": ["X", "Y", "Y", "P", "Q", None, "R", None]})
    got = P.most_common(df, "key", "name")
    # A: Y twice; B: P and Q once each, P written first; C: the one written; D: none written
    assert got.to_dict() == {"A": "Y", "B": "P", "C": "R"}
    assert P.most_common(df.iloc[::-1], "key", "name").to_dict() == {"A": "Y", "B": "Q", "C": "R"}
    assert P.most_common(df.iloc[0:0], "key", "name").empty


@pytest.mark.parametrize("dtype", ["str", object])
@pytest.mark.parametrize("seed", range(5))
def test_the_same_answer_as_value_counts_on_random_rows(seed, dtype):
    rng = np.random.default_rng(seed)
    n = 2000
    names = rng.choice(["ONE", "TWO", "THREE", "FOUR", None], size=n, p=[0.3, 0.3, 0.2, 0.1, 0.1])
    df = pd.DataFrame({"key": rng.integers(0, 300, size=n).astype(str), "name": names}).astype({"name": dtype})
    want = {k: v for k, v in _per_group(df).items() if v is not None}
    assert P.most_common(df, "key", "name").to_dict() == want


def _rows(names: list, key: str = "PACLITAXEL|100MG|INJ|") -> pd.DataFrame:
    """Comparable purchase_prices rows of one product, one per minor-group spelling given."""
    n = len(names)
    return pd.DataFrame({
        "period": ["2026-08"] * n, "plant": [f"HC{i:02d}" for i in range(n)], "material": [f"10{i}" for i in range(n)],
        "vendor_name": ["VENDOR"] * n, "price_status": [P.OK] * n, "molecule_key": [key] * n,
        "minor_group": names, "strength": ["100MG"] * n, "form": ["INJ"] * n,
        "qty": [10.0] * n, "value": [1000.0] * n, "unit_price_min": [100.0] * n, "unit_price_max": [100.0] * n})


def test_products_names_each_product_by_its_most_written_spelling():
    rows = pd.concat([_rows(["PACLITAXEL", "PACLITAXEL ", "PACLITAXEL"]),
                      _rows(["CETUXIMAB", "CETUXIMAB."], key="CETUXIMAB|100MG|INJ|")], ignore_index=True)
    got = P.products(rows.astype({"minor_group": "str"})).set_index("molecule_key")["minor_group"]
    assert got.to_dict() == {"PACLITAXEL|100MG|INJ|": "PACLITAXEL", "CETUXIMAB|100MG|INJ|": "CETUXIMAB"}
    assert got.dtype == "str"                       # the column's own type, as before


def test_a_product_with_no_minor_group_written_gets_none():
    got = P.products(_rows([None, None]))["minor_group"].tolist()
    assert got == [None]
