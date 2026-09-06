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


def test_a_city_filter_on_a_sales_table_is_refused():
    # the brief says plainly that no sales figure can be scoped to a city, and
    # "KEYTRUDA, ₹10.78 Cr in Bangalore hospitals" was written anyway, runs apart
    sql = ("SELECT material_desc, SUM(revenue) FROM sales_by_material_hospital h "
           "JOIN dim_plant d ON 1=1 WHERE d.plant_name LIKE '%Bangalore%'")
    w = sanity.city_on_unreachable_table(sql)
    assert w and w.startswith("IMPOSSIBLE FILTER")
    assert "Bangalore" in w


def test_procurement_by_city_is_allowed():
    assert sanity.city_on_unreachable_table(
        "SELECT SUM(line_value) FROM mart_procurement WHERE plant IN ('HC05')") is None


def test_sales_per_hospital_without_a_city_is_allowed():
    assert sanity.city_on_unreachable_table(
        "SELECT hospital, SUM(revenue) FROM sales_by_hospital GROUP BY 1") is None


def test_an_unnamed_bucket_is_moved_out_of_first_place():
    res = {"rows": [{"category": "Uncategorized", "v": 173.31},
                    {"category": "ANTINEOPLASTIC", "v": 84.45}]}
    assert sanity.sink_placeholders(res) is True
    assert res["rows"][0]["category"] == "ANTINEOPLASTIC"
    assert len(res["rows"]) == 2          # demoted, never dropped


def test_sinking_leaves_a_clean_result_alone():
    res = {"rows": [{"c": "A", "v": 2}, {"c": "B", "v": 1}]}
    assert sanity.sink_placeholders(res) is False
    assert res["rows"][0]["c"] == "A"


def test_a_ranking_that_returns_only_the_unnamed_bucket_is_sent_back():
    # ORDER BY ... LIMIT 1 returning "Uncategorized" cannot be fixed by reordering
    sql = "SELECT category, SUM(v) s FROM t GROUP BY 1 ORDER BY s DESC LIMIT 1"
    w = sanity.placeholder_won_a_ranking(sql, {"rows": [{"category": "Uncategorized", "s": 1}]},
                                         "What is our biggest spend category?")
    assert w and w.startswith("WRONG QUERY") and "NOT IN" in w


def test_a_question_about_the_gap_itself_is_allowed():
    sql = "SELECT category, SUM(v) s FROM t GROUP BY 1 ORDER BY s DESC LIMIT 1"
    assert sanity.placeholder_won_a_ranking(
        sql, {"rows": [{"category": "Uncategorized", "s": 1}]},
        "How much of our spend is uncategorized?") is None


def test_a_ranking_containing_a_real_category_is_left_to_ordering():
    sql = "SELECT category, SUM(v) s FROM t GROUP BY 1 ORDER BY s DESC"
    assert sanity.placeholder_won_a_ranking(
        sql, {"rows": [{"category": "Unknown", "s": 2}, {"category": "REAL", "s": 1}]},
        "biggest category?") is None


def test_an_unordered_query_is_not_a_ranking():
    assert sanity.placeholder_won_a_ranking(
        "SELECT category, SUM(v) FROM t GROUP BY 1",
        {"rows": [{"category": "Unknown", "s": 2}]}, "spend by category") is None


def test_gen_is_treated_as_a_placeholder_manufacturer():
    # "GEN" sits on 3,171 materials as a stand-in for "generic"; asked which manufacturer
    # supplies the most units, the engine answered "GEN, 10,581,027"
    res = {"rows": [{"manufacturer": "GEN", "u": 10_581_027},
                    {"manufacturer": "PENTAWIS INNOVATIONS PRIVATE LIMITED", "u": 1_679_406}]}
    assert sanity.placeholder_leader(res)
    assert sanity.sink_placeholders(res) is True
    assert res["rows"][0]["manufacturer"].startswith("PENTAWIS")


def test_a_real_name_containing_those_letters_is_untouched():
    # whole-value match only — GENERAL MEDICAL is a company, not a placeholder
    for name in ("GENERAL MEDICAL", "GENERIC HEALTH LTD", "MISCO PHARMA"):
        assert sanity.placeholder_leader(
            {"rows": [{"m": name, "v": 2}, {"m": "X", "v": 1}]}) is None, name
