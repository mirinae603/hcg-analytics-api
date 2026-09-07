# tests/test_measure_family.py — which measure a question is ABOUT.
#
# _measure_disclosure prefixes an answer when the number came from a different measure than
# was asked for. That is right when it is true, and damaging when it is not: a correct
# procurement answer was prefixed "There is no sales figure available at this level for what
# you asked" because the question happened to contain the word "sold" — inside a filter.
from __future__ import annotations

import pytest

from app.ai.deep.engine import _family_asked


@pytest.mark.parametrize("q,want", [
    # the measure is the FIRST one named; a later one usually qualifies the population
    ("How much have we spent buying items that are never sold and never consumed?", "purchasing"),
    ("Which items do we consume most but never sell?", "consumption"),
    ("What is our second biggest selling product?", "sales"),
    ("Show me the sales trend for KEYTRUDA", "sales"),
    ("How much did Bangalore hospitals spend on procurement?", "purchasing"),
    ("Which hospital is holding the most inventory value?", "stock"),
    ("Which manufacturer supplies the most units we dispense to patients?", "consumption"),
])
def test_the_family_is_the_first_measure_named(q, want):
    assert _family_asked(q) == want


def test_spend_and_buying_are_in_the_vocabulary_at_all():
    # they were missing, so "how much have we SPENT" matched no purchasing word and the
    # question was classified by a word appearing thirty characters later
    assert _family_asked("how much have we spent") == "purchasing"
    assert _family_asked("what are we buying") == "purchasing"


def test_a_question_naming_no_measure_returns_none():
    assert _family_asked("how many hospitals do we have?") is None
