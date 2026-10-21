"""Hospital names as people say them, mapped to SAP plant codes.

HCG's own reports call its Bangalore hospitals "KR" and "DR", and everyone says "Nagpur" or
"Cuttack" rather than "HCG NCHRI Oncology LLP, Nagpur". The report tools accepted only a SAP
code (HC05), a HIS code (KABHK) or the full HIS name, so "How much did KR spend on restricted
brands…" failed with "Unknown hospital 'KR'". The model then called again with no hospital
and answered for the WHOLE network without saying so (golden cases fs-kr-*, 10-10-2026: KR's
₹43.47 L came back as the network's ₹3.17 Cr).

This module is that missing vocabulary, and nothing else. It is used three ways:
- resolve(): a plant argument the exact lookups did not recognise ("KR", "North Bangalore").
- mentioned(): hospitals a question names, so the scope guard can refuse a call that left
  the hospital out, and the brief can say which codes the tables use.
- prompt_line(): the short list given to the model in the get_kpi plant description.

The list is written out here, not guessed from hospital_map.csv: words such as "City",
"North", "Sun" or "New Town" occur in hospital names and in ordinary questions alike. A test
checks every code below against hospital_map.csv.
"""
from __future__ import annotations

import csv
import os
import re
from functools import lru_cache

# alias -> SAP plant code(s). Several codes = a city with more than one HCG hospital; the
# Jun–Aug report tools pool them, the Dec–May KPIs ask for one call per hospital.
_ALIASES: dict[str, tuple[str, ...]] = {
    "KR": ("HC05",), "HCG KR": ("HC05",), "KR BANGALORE": ("HC05",),
    "DR": ("HC06",), "HCG DR": ("HC06",), "DR BANGALORE": ("HC06",),
    "NORTH BANGALORE": ("HC40",), "NORTH BENGALURU": ("HC40",),
    "BANGALORE": ("HC05", "HC06", "HC40"), "BENGALURU": ("HC05", "HC06", "HC40"),
    "ONGOLE": ("HC26",),
    "VIJAYAWADA": ("HC27",), "CURIE CITY": ("HC27",),
    "VIZAG": ("HC32",), "VISAKHAPATNAM": ("HC32",),
    "AASTHA": ("HM01",),
    "HMS": ("HC16",), "HMS AHMEDABAD": ("HC16",), "HOSPITALS AHMEDABAD": ("HC16",),
    "AHMEDABAD": ("HM01", "HC16"),
    "BHAVNAGAR": ("HC22",),
    "RAJKOT": ("HS01",),
    "VADODARA": ("HO01",), "BARODA": ("HO01",),
    "RANCHI": ("HC17",),
    "GULBARGA": ("HC25",), "KALABURAGI": ("HC25",),
    "SUCHIRAYU": ("SH01",),
    "HUBLI": ("HC24",), "HUBBALLI": ("HC24",),
    "BORIVALI": ("AH01",), "BORIVILI": ("AH01",),
    "KHUBCHANDANI": ("HC33",),
    "MUMBAI": ("AH01", "HC33"),
    "NAGPUR": ("HC38",), "NCHRI": ("HC38",),
    "INDORE": ("HC35",),
    "RATLAM": ("HO02",), "RATHLAM": ("HO02",),
    "CUTTACK": ("HC13",), "PANDA": ("HC13",),
    "JAIPUR": ("HC31",),
    "KOLKATA": ("HE01",),
    "NASHIK UNIT 1": ("HM03",), "NASHIK 1": ("HM03",), "NASIK UNIT 1": ("HM03",),
    "NASHIK UNIT 2": ("HM04",), "NASHIK 2": ("HM04",), "NASIK UNIT 2": ("HM04",),
    "NASHIK": ("HM03", "HM04"), "NASIK": ("HM03", "HM04"),
}
# Two-letter aliases are matched in a question only in capitals: "KR" is the hospital, while
# "Dr" before a name is a doctor.
_CASE_SENSITIVE = {"KR", "DR"}


def _key(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", " ", str(s or "")).strip().upper()
    s = re.sub(r"^HCG\s+(?=\S)", "", s) if s not in ("HCG KR", "HCG DR") else s
    return " ".join(s.split())


@lru_cache(maxsize=1)
def _rows() -> dict[str, tuple[str, str]]:
    """SAP code -> (HIS location name, HIS code), from the reference map (empty if missing)."""
    from app.core.config import settings
    path = os.path.join(settings.REFERENCE_DIR, "hospital_map.csv")
    out: dict[str, tuple[str, str]] = {}
    try:
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                code = (r.get("sap_plant") or "").strip()
                if code and code not in out:
                    out[code] = (" ".join((r.get("his_location_name") or code).split()),
                                 (r.get("his_code") or "").strip())
    except OSError:
        pass
    return out


def name(code: str) -> str:
    return _rows().get(code, (code, ""))[0]


def his_code(code: str) -> str:
    """The sales tables' hospital code (KABHK) for a SAP plant (HC05), or ''."""
    return _rows().get(code, ("", ""))[1]


def describe(codes) -> str:
    return ", ".join(f"{c} ({name(c)})" for c in codes)


def resolve(text) -> tuple[str, ...] | None:
    """SAP code(s) for a hospital alias, or None when it is not one. "KR" -> ("HC05",);
    "Bangalore" -> its three hospitals."""
    k = _key(text)
    if not k:
        return None
    if k in _ALIASES:
        return _ALIASES[k]
    k2 = _key("HCG " + str(text))       # "hcg kr" written in lower case
    return _ALIASES.get(k2)


@lru_cache(maxsize=1)
def _patterns() -> list[tuple[re.Pattern, str]]:
    pats = []
    # longest first, so "North Bangalore" wins over "Bangalore" and "HCG KR" over "KR"
    for alias in sorted(_ALIASES, key=len, reverse=True):
        words = r"[\s,\-]+".join(map(re.escape, alias.split()))
        flags = 0 if alias in _CASE_SENSITIVE else re.I
        # "DR." is a doctor's title; "at KR." is the end of a sentence
        tail = r"(?![A-Za-z0-9.])" if alias == "DR" else r"(?![A-Za-z0-9])"
        pats.append((re.compile(rf"(?<![A-Za-z0-9]){words}{tail}", flags), alias))
    return pats


def mentioned(question: str) -> list[tuple[str, tuple[str, ...]]]:
    """(alias as written, codes) for each hospital the question names, without overlaps."""
    q = question or ""
    taken: list[tuple[int, int]] = []
    found: list[tuple[int, str]] = []
    for pat, alias in _patterns():
        for m in pat.finditer(q):
            if any(m.start() < e and s < m.end() for s, e in taken):
                continue
            taken.append((m.start(), m.end()))
            if not any(a == alias for _, a in found):
                found.append((m.start(), alias))
    # in the order the question names them
    return [(alias, _ALIASES[alias]) for _, alias in sorted(found)]


def prompt_line() -> str:
    """The short list for the model: one name per hospital."""
    seen: dict[str, str] = {}
    for alias, codes in _ALIASES.items():
        if len(codes) == 1 and codes[0] not in seen:
            seen[codes[0]] = alias if alias in _CASE_SENSITIVE or alias in ("HMS", "NCHRI") else alias.title()
    return "; ".join(f"{a}={c}" for c, a in seen.items())
