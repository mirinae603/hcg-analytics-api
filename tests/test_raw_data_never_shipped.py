# tests/test_raw_data_never_shipped.py — HCG's raw extracts must never reach git or the image.
#
# The formulary build reads HCG's monthly pharmacy sales workbooks from master/ in the
# checkout. Those are line-level: patient names, UHIDs, IP/OP numbers and invoice numbers.
# Only the small aggregated tables built from them (app/data/formulary/) are meant to ship.
#
# Before these rules master/ (2.3 GB) was untracked but NOT ignored, so one `git add .`
# would have committed it, and `COPY . .` in the Dockerfile would have baked it into every
# image. .dockerignore patterns are also anchored at the build-context root, so a rule
# like `master/` misses the copies inside .claude/worktrees/<wt>/ — each of which carries
# its own .venv and, potentially, a line-level .cache/. Hence the **/ forms.
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

MUST_NOT_SHIP = [
    "master/9f2c/Formulary Compliance and Spend Analysis Dashboard data/"
    "IP and OP sales/IP & OP Consolidated File August 26.xlsx",
    "master/9f2c/Formulary Compliance and Spend Analysis Dashboard data/"
    "Copy of Copy of Payer Master Final Sheet.xlsb",
    ".cache/formulary/lines_2026-08.parquet",
    ".claude/worktrees/p1/master/9f2c/sales.xlsb",
    ".claude/worktrees/p1/.cache/formulary/lines_2026-08.parquet",
    ".claude/worktrees/p1/.venv/lib/python3.14/site-packages/x.py",
    "HCG_Bidezy_Meeting_Requirements_updated.docx",
    "app/data/scratch/some export.xlsb",
]

MUST_SHIP = [
    "app/data/formulary/formulary_summary.parquet",
    "app/data/formulary/formulary_validation.json",
    "app/data/spend/scheme_switch.parquet",
    "app/data/spend/spend_validation.json",
    "app/data/spend/cash_switch.parquet",
    "app/data/reference/cash_tpa_bifurcation.csv",
    "app/data/reference/cash_tpa_excluded.csv",
    "app/data/reference/brand_rules.csv",
    "app/data/formulary/formulary_rule_breach.parquet",
    "app/data/formulary/formulary_rules.parquet",
    "app/data/spend/brand_blocks.parquet",
    "app/data/reference/hospital_map.csv",
    "app/data/reference/payer_overrides.csv",
    "app/data/kpi/kpi_stock_value.parquet",
    "app/main.py",
]


def _lines(name: str) -> list[str]:
    text = (ROOT / name).read_text()
    return [ln.strip() for ln in text.splitlines()
            if ln.strip() and not ln.strip().startswith("#")]


def _glob_to_regex(pattern: str) -> re.Pattern:
    out, i = "", 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out, i = out + "(?:.*/)?", i + 3
        elif pattern.startswith("**", i):
            out, i = out + ".*", i + 2
        elif pattern[i] == "*":
            out, i = out + "[^/]*", i + 1
        elif pattern[i] == "?":
            out, i = out + "[^/]", i + 1
        else:
            out, i = out + re.escape(pattern[i]), i + 1
    return re.compile(out)


def _docker_excludes(path: str) -> bool:
    # Docker's rules: patterns are relative to the context root, `*` stays within one
    # directory, `**` spans any number of them, a rule that matches a parent directory
    # excludes everything under it, and the last matching rule (incl. `!`) wins.
    parts = path.split("/")
    prefixes = ["/".join(parts[:i]) for i in range(1, len(parts) + 1)]
    excluded = False
    for raw in _lines(".dockerignore"):
        negate = raw.startswith("!")
        rx = _glob_to_regex(raw.lstrip("!").strip("/"))
        if any(rx.fullmatch(p) for p in prefixes):
            excluded = not negate
    return excluded


@pytest.mark.parametrize("path", MUST_NOT_SHIP)
def test_raw_client_data_is_excluded_from_the_image(path):
    assert _docker_excludes(path), (
        f"{path!r} is not matched by .dockerignore — `COPY . .` would bake it into the image")


@pytest.mark.parametrize("path", MUST_SHIP)
def test_the_aggregated_tables_still_ship(path):
    # the ignore rules must not take the deployed tables down with the raw files
    assert not _docker_excludes(path), f"{path!r} is excluded from the image by .dockerignore"


def test_dockerignore_lists_every_raw_data_rule():
    patterns = set(_lines(".dockerignore"))
    for p in ("master/", "**/master/", ".cache/", "**/.cache/", ".claude/",
              "*.xlsx", "*.xlsb", "*.docx", "**/*.xlsx", "**/*.xlsb", "**/*.docx"):
        assert p in patterns, f"{p} is missing from .dockerignore"


def test_gitignore_lists_every_raw_data_rule():
    patterns = set(_lines(".gitignore"))
    for p in ("/master/", "**/master/", ".cache/", "*.xlsx", "*.xlsb", "*.docx"):
        assert p in patterns, f"{p} is missing from .gitignore"


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_git_agrees_about_what_is_ignored():
    # ask git itself, so a pattern that reads right but matches wrong still fails here
    def ignored(path: str) -> bool:
        r = subprocess.run(["git", "check-ignore", "--no-index", "-q", path],
                           cwd=ROOT, capture_output=True)
        if r.returncode not in (0, 1):
            pytest.skip(f"git check-ignore unavailable here: {r.stderr.decode().strip()}")
        return r.returncode == 0

    for path in MUST_NOT_SHIP:
        if "/.venv/" in path:
            continue  # .venv/ has its own gitignore rule; not this test's concern
        assert ignored(path), f"git would commit {path!r}"
    for path in MUST_SHIP:
        assert not ignored(path), f"git ignores {path!r}, so it would never be committed"
