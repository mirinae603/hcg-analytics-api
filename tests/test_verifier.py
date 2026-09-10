"""The content verifier — and specifically the trap that made its predecessor useless."""
from app.ai.deep import verifier as V


def test_a_shared_year_is_not_agreement():
    # THE BUG THIS MODULE EXISTS FOR. The old rule was `bool(a & b)`: any shared numeric
    # token counted as corroboration. Here the two derivations differ by two orders of
    # magnitude and share only the year, and the old rule called that agreement.
    a, b = ["2024", "8420000000"], ["2024", "31000000"]
    assert set(a) & set(b)                       # the old rule fired "agreed"
    assert V.compare(a, b) == V.DISAGREE         # this one reads the quantities


def test_rounding_is_not_disagreement():
    # The corroborator writes its own SELECT and rounds differently. 8.42 Cr vs 8.4 Cr is
    # one number presented twice, and calling it a contradiction would flag correct answers.
    assert V.compare(["842000000"], ["840000000"]) == V.AGREE


def test_parts_that_sum_to_the_total_agree():
    # The corroborate prompt asks for "a sum of parts instead of a stored total", so the
    # breakdown summing to the headline is the check succeeding, not two different answers.
    assert V.compare(["9000000"], ["4000000", "5000000"]) == V.AGREE


def test_a_stored_total_matching_a_breakdown_agrees_in_either_direction():
    assert V.compare(["4000000", "5000000"], ["9000000"]) == V.AGREE


def test_structural_numbers_alone_say_nothing():
    # Ranks, months and site counts are not headline figures. Comparing them across two
    # different queries is noise, so the verifier declines rather than guessing.
    assert V.compare(["1", "2", "3"], ["7", "8"]) == V.UNKNOWN


def test_unknown_is_not_a_soft_agree():
    # A verifier that returns "agree" when it has nothing to compare inflates its own
    # coverage and hides the cases it never checked.
    assert V.compare([], ["8420000000"]) == V.UNKNOWN
    assert V.compare(["8420000000"], []) == V.UNKNOWN


def test_one_disagreement_outranks_any_number_of_agreements():
    # A contradiction between two derivations is information; an agreement is only the
    # absence of one, so agreements must not be able to outvote it.
    pairs = [(["9000000"], ["9000000"]), (["9000000"], ["100000000"])]
    assert V.verdict(pairs) == V.DISAGREE


def test_no_corroboration_is_unknown_not_agree():
    assert V.verdict([]) == V.UNKNOWN


def test_malformed_tokens_do_not_raise():
    # Result rows carry nulls and text; the verifier runs on every answer and must never be
    # the thing that breaks one.
    assert V.compare(["n/a", None, "8420000000"], ["8420000000", ""]) == V.AGREE
