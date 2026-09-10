"""Does the second derivation of a figure AGREE with the first?

Phase 4 already recomputes the headline figure a different way — a different table, or a sum
of parts instead of a stored total. What it did with the result was `bool(a & b)`: do the two
result sets share ANY numeric token. On a result with eight rows that is nearly always true by
accident. A year like 2024, a rank column, a count of 1 — any of them satisfies it while the
totals differ by an order of magnitude. So the signal said "agreed" almost everywhere and
carried very little.

This module compares the quantities instead of the digit strings, and reports what it could
not compare rather than calling that agreement.

The reason it is written as a pure function over the two number sets, with no engine imports,
is that it has to be measurable offline against recorded runs. A verifier that can only be
evaluated by running the whole engine cannot have its precision established before it is
trusted, and trusting an unmeasured signal is exactly what cost three points last time.
"""
from __future__ import annotations

AGREE, DISAGREE, UNKNOWN = "agree", "disagree", "unknown"

# Below this, two figures are different presentations of one number rather than two answers:
# 8.42 vs 8.4 Cr is rounding in the corroborator's SELECT, not a contradiction.
TOLERANCE = 0.02

# A "headline" figure is the largest magnitude in the result — a total dwarfs the ranks, years
# and row counts around it. Numbers below this are structural (a rank, a month index, a count
# of sites) and comparing them across two different queries means nothing.
FLOOR = 1000.0


def _nums(tokens) -> list[float]:
    out = []
    for t in tokens or []:
        try:
            out.append(abs(float(str(t).replace(",", ""))))
        except (TypeError, ValueError):
            continue
    return out


def _close(x: float, y: float, tol: float = TOLERANCE) -> bool:
    if x == y:
        return True
    scale = max(abs(x), abs(y))
    return scale > 0 and abs(x - y) / scale <= tol


def compare(primary_tokens, alt_tokens, tol: float = TOLERANCE) -> str:
    """AGREE / DISAGREE / UNKNOWN for two independently derived result sets.

    UNKNOWN is a real answer and not a soft agree. If neither side produced a figure big
    enough to be a headline, this has nothing to say, and saying nothing is what keeps its
    precision meaningful.
    """
    a = [n for n in _nums(primary_tokens) if n >= FLOOR]
    b = [n for n in _nums(alt_tokens) if n >= FLOOR]
    if not a or not b:
        return UNKNOWN
    head = max(a)
    # The corroborator often returns ONLY the total while the primary returns a breakdown
    # whose parts sum to it, so match the primary's headline against anything on the other
    # side — including that side's sum, which is the "sum of parts vs stored total" check
    # the corroborate prompt actually asks for.
    # Symmetric on purpose. Either side may be the breakdown and either may be the stored
    # total — which one the corroborator chose is not something this can rely on, and a
    # first draft that only summed ONE side called a correct total-vs-parts check a
    # contradiction.
    if any(_close(head, x, tol) for x in b) or _close(head, sum(b), tol):
        return AGREE
    if any(_close(max(b), x, tol) for x in a) or _close(max(b), sum(a), tol):
        return AGREE
    return DISAGREE


def verdict(pairs, tol: float = TOLERANCE) -> str:
    """Fold every corroboration attempt for one answer into a single verdict.

    One DISAGREE outranks any number of AGREEs. A contradiction between two derivations of
    the same quantity is information; an agreement is only the absence of it.
    """
    seen = [compare(p[0], p[1], tol) for p in pairs if len(p) == 2]
    if DISAGREE in seen:
        return DISAGREE
    if AGREE in seen:
        return AGREE
    return UNKNOWN
