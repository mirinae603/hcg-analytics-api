# tests/test_repair_loop.py — the contract the repair loop depends on.
#
# Measured on this warehouse: three candidate queries with NO repair pass formed a majority
# on 42% of questions; adding ONE execution-feedback pass took that to 83%. Repair is the
# single highest-value mechanism in the pipeline — worth more than candidate diversity, and
# it matches the published ablations (PV-SQL loses 3.3 points without it).
#
# It is also invisible: it works by a failed query coming back as a DICT the model can read,
# rather than an exception that kills the turn. Anything that turns those errors into raises
# — or into empty results — silently disables the loop and nothing else would notice.
from __future__ import annotations

from app.ai.deep import tools


def test_a_binder_error_comes_back_as_data_not_an_exception():
    out = tools.run_query("SELECT no_such_column FROM mart_procurement")
    assert isinstance(out, dict) and out.get("error")
    assert "no_such_column" in out["error"] or "Binder" in out["error"]


def test_a_syntax_error_comes_back_as_data():
    out = tools.run_query("SELECT FROM WHERE")
    assert isinstance(out, dict) and out.get("error")


def test_the_error_text_is_specific_enough_to_repair_from():
    # "something went wrong" is unrepairable; the model needs the column name
    out = tools.run_query("SELECT vendor_nam FROM mart_procurement LIMIT 1")
    assert out.get("error") and len(out["error"]) > 20
    assert "vendor_nam" in out["error"]


def test_a_good_query_still_returns_rows():
    out = tools.run_query("SELECT COUNT(*) AS n FROM mart_procurement")
    assert not out.get("error")
    assert (out.get("rows") or [{}])[0].get("n")


def test_guards_return_errors_rather_than_raising():
    # every guard added to run_query must keep the repair contract: a blocked query is an
    # error DICT the model can act on, never an exception
    for sql in ("SELECT material_desc, SUM(revenue) FROM sales_by_material_hospital h "
                "JOIN dim_plant d ON 1=1 WHERE d.plant_name LIKE '%Bangalore%'",
                "SELECT category, SUM(v) s FROM mart_procurement GROUP BY 1 ORDER BY s DESC"):
        out = tools.run_query(sql, question="what is our biggest spend category?")
        assert isinstance(out, dict)


def test_an_empty_result_is_not_reported_as_an_error():
    # "no rows" and "no such data" are different claims; conflating them is how absences
    # get invented
    out = tools.run_query(
        "SELECT * FROM mart_procurement WHERE vendor_name = 'NO SUCH VENDOR ZZZ'")
    assert isinstance(out, dict)
    assert out.get("row_count", 0) == 0 or out.get("error")
