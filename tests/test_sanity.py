# tests/test_sanity.py — the checks that catch a well-formed wrong number.
from __future__ import annotations

from app.ai.deep import sanity


def test_a_subtotal_larger_than_the_total_is_reported():
    # the shape of the Bangalore failure: a JOIN fanned procurement rows out
    sql = ("SELECT SUM(p.line_value) v FROM mart_procurement p "
           "JOIN fact_inventory i ON p.material = i.material WHERE p.plant = 'HC05'")
    w = sanity.part_exceeds_whole(sql, {"rows": [{"v": 14_635_494_519.0}]})
    assert w and w.startswith("IMPOSSIBLE")
    assert "JOIN" in w                      # names the cause, not just the symptom


def test_a_legitimate_subtotal_passes():
    sql = "SELECT SUM(line_value) v FROM mart_procurement WHERE plant = 'HC05'"
    assert sanity.part_exceeds_whole(sql, {"rows": [{"v": 908_159_009.0}]}) is None


def test_an_unfiltered_total_is_never_flagged_against_itself():
    sql = "SELECT SUM(line_value) v FROM mart_procurement"
    assert sanity.part_exceeds_whole(sql, {"rows": [{"v": 4_782_690_674.0}]}) is None


def test_non_additive_aggregates_are_left_alone():
    # AVG/MAX over a filter can exceed the unfiltered AVG legitimately, so comparing proves
    # nothing and a warning would be noise
    for agg in ("AVG", "MAX", "MIN"):
        sql = f"SELECT {agg}(line_value) v FROM mart_procurement WHERE plant = 'HC05'"
        assert sanity.part_exceeds_whole(sql, {"rows": [{"v": 9e12}]}) is None


def test_a_measure_from_another_table_is_not_compared():
    sql = ("SELECT SUM(i.total_cost) v FROM mart_procurement p "
           "JOIN fact_inventory i ON p.material = i.material WHERE p.plant = 'HC05'")
    assert sanity.part_exceeds_whole(sql, {"rows": [{"v": 9e15}]}) is None


def test_empty_and_malformed_input_never_raises():
    assert sanity.part_exceeds_whole("", {}) is None
    assert sanity.part_exceeds_whole("SELECT 1", {"rows": []}) is None
    assert sanity.check("", {}) == []


def test_an_unnamed_bucket_may_not_lead_a_ranking():
    w = sanity.placeholder_leader({"rows": [{"category": "Uncategorized", "v": 173.31},
                                            {"category": "ANTINEOPLASTIC", "v": 84.45}]})
    assert w and "ANTINEOPLASTIC" in w


def test_a_named_leader_is_not_flagged():
    assert sanity.placeholder_leader({"rows": [{"category": "ANTINEOPLASTIC", "v": 84.45},
                                               {"category": "Uncategorized", "v": 20.0}]}) is None


def test_placeholder_spellings_are_all_caught():
    for label in ("Unknown", "N/A", "", "  ", "None", "Other", "Not Assigned", "-"):
        assert sanity.placeholder_leader(
            {"rows": [{"c": label, "v": 9.0}, {"c": "REAL", "v": 1.0}]}), label


def test_all_unnamed_is_a_data_fact_not_a_correction():
    # nothing to promote — stay silent rather than invent a leader
    assert sanity.placeholder_leader({"rows": [{"c": "Unknown", "v": 9.0},
                                               {"c": "N/A", "v": 1.0}]}) is None
