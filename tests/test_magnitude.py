"""A part cannot be larger than the whole."""
import pytest

from app.ai.deep import magnitude as m

FANOUT = ("SELECT p.category, SUM(p.line_value) AS spend FROM mart_procurement p "
          "JOIN consumption_all c ON p.material=c.material GROUP BY 1")


def _res(rows, truncated=False):
    return {"rows": rows, "truncated": truncated}


def test_the_shipped_error_is_caught():
    # The real answer: "procurement is dominated by Consumables, Rs 19,825.46 Cr", against a
    # warehouse holding about Rs 478 Cr of procurement in total. Well-formed SQL, correctly
    # scoped, no ontology violation, and the corroborator agreed because both derivations
    # shared the same inflated join. Only the arithmetic gives it away.
    hit = m.exceeds_the_whole(FANOUT, _res([{"category": "Consumables", "spend": 198_254_600_000.0}]))
    assert hit and "multiplied the rows" in hit


def test_a_plausible_figure_passes():
    assert m.exceeds_the_whole(FANOUT, _res([{"category": "Consumables", "spend": 1_800_000_000.0}])) is None


def test_negative_values_do_not_disable_the_check():
    # line_value carries credits and returns, so MIN() is negative. The first version bailed
    # out on any column with a negative value, which turned the check off for the exact
    # column whose fan-out shipped. The bound is the POSITIVE mass, which any subset sum
    # respects whether or not the column nets out against itself.
    from app.ai import warehouse
    lo = warehouse.run_sql('SELECT MIN(line_value) AS lo FROM mart_procurement')["rows"][0]["lo"]
    assert lo < 0, "fixture assumes this column has negatives; pick another if it stops being true"
    assert m._whole("mart_procurement", "line_value") > 0


def test_a_truncated_result_declines_rather_than_guessing():
    # Summing a truncated page understates the part, so the comparison could only produce a
    # false negative. Say nothing rather than pretend the check ran.
    assert m.exceeds_the_whole(FANOUT, _res([{"category": "x", "spend": 9e12}], truncated=True)) is None


def test_an_ambiguous_column_declines():
    # If two tables in the query both own the column, the bound could be taken from the
    # wrong one, and a wrong bound REJECTS CORRECT ANSWERS. Declining is the safe failure.
    assert m._owner("category", ["mart_procurement", "consumption_all"]) is None


def test_an_unknown_column_is_not_an_error():
    assert m.exceeds_the_whole("SELECT SUM(nope) AS n FROM mart_procurement", _res([{"n": 1e15}])) is None


def test_it_never_raises_on_odd_rows():
    # It runs on every successful query; it must never be the thing that breaks one.
    assert m.exceeds_the_whole(FANOUT, _res([{"spend": None}, {"spend": "n/a"}, {}])) is None


def test_the_guarded_path_applies_it():
    import inspect
    from app.ai.deep import tools
    assert "exceeds_the_whole" in inspect.getsource(tools.run_query)
