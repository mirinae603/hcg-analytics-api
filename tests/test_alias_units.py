# tests/test_alias_units.py — an alias inherits the unit of what it aggregates.
#
# `SELECT SUM(line_value) AS total_procurement` yields a column named "total_procurement",
# which matches no money pattern. ₹174.02 Cr of Bangalore procurement was handed to the
# writer as 1740233400.36 and printed exactly that way — correct arithmetic, unreadable
# answer. The SQL already says what the number is.
from __future__ import annotations

from app.ai.deep.engine import _alias_units, _compact
from app.ai.orchestrator import _format_result


def test_a_unitless_alias_inherits_money_from_its_source():
    assert _alias_units("SELECT SUM(line_value) AS total_procurement FROM mart_procurement") \
        == {"total_procurement": "inr"}


def test_the_inherited_unit_reaches_the_text():
    out = _compact({"columns": ["total_procurement"],
                    "rows": [{"total_procurement": 1740233400.36}]},
                   sql="SELECT SUM(line_value) AS total_procurement FROM mart_procurement")
    assert "₹174.02 Cr" in out


def test_a_quantity_alias_is_not_turned_into_money():
    assert _alias_units("SELECT SUM(qty) AS moved FROM sales_by_material") == {}
    out = _compact({"columns": ["moved"], "rows": [{"moved": 1347643}]},
                   sql="SELECT SUM(qty) AS moved FROM sales_by_material")
    assert "₹" not in out


def test_days_and_percent_are_inherited_too():
    assert _alias_units("SELECT AVG(vendor_avg_lead_time_days) AS turnaround FROM t") \
        == {"turnaround": "days"}


def test_sql_keywords_are_never_mistaken_for_aliases():
    got = _alias_units("SELECT SUM(line_value) FROM t WHERE x > 1 GROUP BY plant ORDER BY 1")
    assert all(k not in got for k in ("from", "group", "order", "where"))


def test_a_column_that_already_names_its_unit_is_left_alone():
    # value_share_pct contains "value"; typing it as money printed "₹85" for 85%
    out = _compact({"columns": ["value_share_pct"], "rows": [{"value_share_pct": 85.0}]},
                   sql="SELECT SUM(revenue) AS value_share_pct FROM t")
    assert "85.0%" in out and "₹" not in out


def test_fast_mode_applies_the_same_rule():
    out = _format_result({"columns": ["total_procurement"],
                          "rows": [{"total_procurement": 1740233400.36}], "row_count": 1,
                          "sql": "SELECT SUM(line_value) AS total_procurement FROM mart_procurement"})
    assert out["rows"][0]["total_procurement"] == "₹174.02 Cr"


def test_missing_sql_is_not_an_error():
    assert _alias_units("") == {}
    assert "1,740,233,400" in _compact({"columns": ["total_procurement"],
                                        "rows": [{"total_procurement": 1740233400.36}]})


def test_a_rupee_sign_forces_house_format_whatever_the_sql_looked_like():
    # the alias rule needs to recognise the SQL shape; `SUM(line_value) OVER () AS total`
    # defeated it and left ₹1,743,233,400.48 in a brief quoting crores everywhere else
    from app.ai.deep.engine import _rupees_in_scale
    assert _rupees_in_scale("spend ₹1,743,233,400.48 on procurement") == "spend ₹174.32 Cr on procurement"
    assert _rupees_in_scale("exposure ₹39,97,000") == "exposure ₹39.97 L"


def test_figures_already_in_scale_are_untouched():
    from app.ai.deep.engine import _rupees_in_scale
    for t in ("₹3.03 Cr", "₹39.97 L", "₹1.15 crore", "₹27,012"):
        assert _rupees_in_scale(f"value {t} here") == f"value {t} here", t
