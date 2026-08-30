"""Structured shapes the swarm passes between phases.

Every agent returns JSON so the engine can branch on it in code. A phase whose output has
to be parsed out of prose is a phase that degrades silently — and silent degradation is the
exact failure mode this whole engine exists to remove.
"""

FRAME = """Return: {"shape": str, "kpi_key": str, "intent": str, "entity": str|null, "entity_family":
"material"|"manufacturer"|"vendor"|"hospital"|"category"|null, "needs_time_series": bool,
"answerable": bool, "blocked_reason": str}. `shape` is the KIND of answer this question
needs, chosen from the catalogue given — it decides what a complete answer owes the reader,
so pick the one the question really is rather than the one the words resemble.

`kpi_key` is the canonical dashboard metric that already answers this, from the list given,
or "" if none does. These are the SAME calculations the dashboard cards use, so when one
covers the question its figure is authoritative and re-deriving it in SQL can only produce
a number that disagrees with what the user sees on their screen. Set answerable=false ONLY if the schema
provided genuinely cannot support the question (e.g. the measure and the requested
breakdown never appear on the same table); put the specific reason in blocked_reason."""

PLAN = """Return: {"sub_questions": [{"id": str, "slot": str, "question": str, "why": str,
"table": str, "needs": str}]}. One per SLOT you are given, in that order, plus at most two
extras if something genuinely load-bearing is missing. Set `slot` to the slot id.

A slot is not a table to fetch, it is a thing the answer OWES the reader. "series" means
one row per period with every other dimension summed away — not a grid of hospital by
month, which is not a trend and cannot be described as one. Each must be answerable by ONE SQL
query against ONE table named in `table`, and each must earn its place: no sub-question
whose answer cannot change the conclusion.

`needs` is what must be established BEFORE this one can run — "" if nothing. Put the
question that establishes an entity ("which vendor has the highest spend?") first with
needs="", and set needs="the top vendor" on the ones that depend on knowing it. A
sub-question that says "the top vendor" with needs="" cannot be written as SQL by anyone,
because nothing in it says WHICH vendor."""

SQL = """Return: {"sql": str, "purpose": str}. One SELECT. No semicolons. Use only the
columns listed for that table. If the table cannot answer the sub-question, return
{"sql": "", "purpose": "<why not>"} rather than an approximation."""

CORROBORATE = """Return: {"sql": str, "measures": str}. Compute the SAME figure by a
DIFFERENT route — another table, or a sum of parts rather than a stored total. If no
independent route exists return {"sql": "", "measures": "no independent route"}."""

CRITIQUE = """Return: {"refuted": bool, "problem": str, "severity": "high"|"medium"|"low"}.
Set refuted=true only for a defect that would change the conclusion — a wrong scope, a
double count, a wrong dimension, a figure that does not follow from the evidence. Style,
tone and completeness are NOT refutations."""

GAPS = """Return: {"gaps": [{"question": str, "table": str, "why_it_matters": str}]}. At
most three, and only ones that could change the conclusion. Return {"gaps": []} freely —
an investigation that is genuinely complete is the normal case, not a failure."""
