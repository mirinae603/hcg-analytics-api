"""A golden set for deep mode — the thing that turns "it broke again" into a number.

Every accuracy fix in this codebase so far was verified by running ONE question ONCE. That
proves nothing about a system with four LLM decisions in series: the same KEYTRUDA question
answered correctly at 10:20 and collapsed at 10:57 with no code change in between. Judging a
stochastic system from single runs is how you end up unable to say whether three hours of
fixes helped.

Each case asserts on the ANSWER, not on wording — a figure that must appear, a claim that
must not, a table that must have been used. Run each case N times and report a pass RATE.

    .venv/bin/python -m evals.golden            # 1 run each
    .venv/bin/python -m evals.golden 3          # 3 runs each, for variance
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.getcwd())
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"))
except Exception:
    pass


from evals.cases import CASES  # the 90-question bank + its ground truth


def run(case, mode="deep"):
    if mode == "deep":
        from app.ai.deep import engine as eng
    else:
        from app.ai import orchestrator as eng
    text, queries, kind, t0 = "", 0, "answer", time.time()
    try:
        for ev in eng.answer(case["q"]):
            t = ev.get("type")
            if t == "answer_delta":
                text += ev.get("text") or ""
            elif t == "answer":
                text = ev.get("text") or text
            elif t in ("clarify", "error"):
                # Fast mode can end a turn by asking a question back or by erroring, and
                # neither is an "answer" event. Ignoring them scored those turns as EMPTY
                # and blamed the engine for a gap in the measurement — the exact mistake
                # this harness exists to stop. A clarification is a real outcome: record
                # it, mark it, and let the case decide whether it counts.
                text = ev.get("text") or text
                kind = t
            elif t == "sql":
                queries += 1
    except Exception as e:
        return {"ok": False, "err": str(e)[:200], "secs": time.time() - t0,
                "queries": queries, "text": "", "kind": "exception"}
    ok = bool(case["check"](text))
    if case.get("must_answer") and kind != "answer":
        ok = False
    return {"ok": ok, "secs": time.time() - t0, "queries": queries, "text": text, "kind": kind}


def main():
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    mode = sys.argv[2] if len(sys.argv) > 2 else "deep"
    only = sys.argv[3] if len(sys.argv) > 3 else ""
    # a third arg filters by case id OR by level ("L1"/"L2"/"L3")
    cases = [c for c in CASES
             if not only or only in c["id"] or c.get("level", "") == only.upper()]
    print(f"golden set · mode={mode} · {runs} run(s) each · {len(cases)} cases\n", flush=True)

    # 30 cases x N runs is an hour of wall clock run one at a time, which is long enough
    # that nobody runs it — and an eval nobody runs is not a safety net.
    from concurrent.futures import ThreadPoolExecutor
    jobs = [(c, i) for c in cases for i in range(runs)]
    # 3, not 6: each deep run fans out to 4 workers of its own, so 6 concurrent cases put
    # ~24 requests on Azure at once and the eval started failing itself with 429s —
    # measuring the harness, not the engine.
    # Report each case AS IT LANDS. Printing only the final table made a 90-case run opaque
    # for half an hour, and when one stalled there was no way to tell a slow run from a hung
    # one without attaching a profiler to the process.
    from threading import Lock
    seen, tick = [0], Lock()

    def _one(j):
        cid, r = j[0]["id"], run(j[0], mode)
        with tick:
            seen[0] += 1
            print(f"  · {seen[0]:>3}/{len(jobs)} {'ok ' if r['ok'] else 'FAIL'} "
                  f"{cid[:34]:36s} {r['secs']:5.1f}s", file=sys.stderr, flush=True)
        return cid, r

    with ThreadPoolExecutor(max_workers=2) as pool:
        done = list(pool.map(_one, jobs))
    by_id: dict = {}
    for cid, r in done:
        by_id.setdefault(cid, []).append(r)

    total = passed = 0
    for c in cases:
        results = by_id.get(c["id"], [])
        ok = sum(1 for r in results if r["ok"])
        total += runs
        passed += ok
        if not results:
            continue
        secs = sum(r["secs"] for r in results) / len(results)
        q = sum(r["queries"] for r in results) / len(results)
        mark = "PASS" if ok == runs else ("FLAKY" if ok else "FAIL")
        kinds = {r.get("kind") for r in results} - {"answer"}
        tag = f"  [{'/'.join(sorted(kinds))}]" if kinds else ""
        print(f"  [{mark:5s}] {c['id']:26s} {ok}/{len(results)}   {secs:5.1f}s  {q:.1f}q{tag}", flush=True)
        if ok < runs:
            bad = next(r for r in results if not r["ok"])
            print(f"           why it should pass: {c['why']}")
            print(f"           got: {(bad.get('err') or bad['text'])[:200].strip()}", flush=True)
    # per level, because an average hides which KIND of question is failing
    print()
    for lv in ("L1", "L2", "L3", "L4"):
        rows = [(c, by_id.get(c["id"], [])) for c in cases if c.get("level") == lv]
        n = sum(len(r) for _, r in rows)
        ok = sum(1 for _, rs in rows for r in rs if r["ok"])
        if n:
            print(f"  {lv}: {ok}/{n} = {100 * ok / n:.0f}%")
    print(f"\n  OVERALL {passed}/{total} = {100 * passed / max(total, 1):.0f}%")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
