"""What a drug's description and Item Master 'Minor Group' say about the product, as keys
the out-of-formulary fallback rule (oof.py) compares across makers.

Builds on the shared parsers in app/etl/spend/oof.py and fixes what HCG's descriptions get
wrong for this purpose (each fix found on the Jun-Aug 26 purchases):

  molecule_key      'ACECLOFENAC+TRAMADOL HCL (SCH H1)' -> 'ACECLOFENAC+TRAMADOL': schedule
                    tags and bracketed text dropped, combinations split on + / & AND ',',
                    salt words dropped (HCL, SODIUM, MALEATE, ...), each part written without
                    spaces, synonyms made one (LIGNOCAINE = LIDOCAINE, ...), parts sorted
  clean_desc        a unit glued to the next word split ('25MGINJ' -> '25MG INJ'), POW and
                    GRANULES read as POWDER, '10/75MG' read as '10MG/75MG'
  strength_key      spend.oof.strength_key on the cleaned text, except that a dose per mL
                    with one pack volume is the pack's total dose ('50MG/ML INJ 2ML' -> '100MG')
  dose_key          the doses alone, without volumes; 50 GM or more is a pack weight, not a dose
  pack_size         the largest volume (mL) or pack weight, to keep 'pack within 4x'
  form_key          spend.oof.form_key on the cleaned text; form_group: tablet = capsule (oral
                    solid), oral liquid = drops
  kind_key          spend.oof.kind_key on the description and minor group (MR, LIPO, NAB), else
                    MD for mouth-dissolving tablets (MD, DT, ODT)
  descriptor_words  the words after the brand name that name the variant (XT, D, PLUS, ...)
  describe          all of the above for one product -> Product
  close_molecules   molecule keys spelt nearly the same (NICRONDIL / NICORANDIL)

strength_key, form_key and kind_key repeat spend.oof's names on purpose: these are the rule's
versions of the same keys.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import Iterable, Optional

from app.etl.formulary.classify import norm_text
from app.etl.spend import oof as S

# Words naming the salt or grade, not the molecule ('MONTELUKAST SODIUM' is montelukast). A
# part made only of such words keeps them ('CALCIUM CARBONATE').
SALT_WORDS = frozenset({
    "HCL", "HYDROCHLORIDE", "SODIUM", "POTASSIUM", "SULPHATE", "SULFATE", "MALEATE", "OXALATE",
    "SUCCINATE", "TARTRATE", "PHOSPHATE", "BESYLATE", "BESILATE", "MESYLATE", "CITRATE", "ACETATE",
    "BROMIDE", "HYDROBROMIDE", "MAGNESIUM", "CALCIUM", "DISODIUM", "FUMARATE", "MONOHYDRATE",
    "DIHYDRATE", "TRIHYDRATE", "HYDRATE", "BITARTRATE", "DIPROPIONATE", "PROPIONATE", "FUROATE",
    "VALERATE", "DECANOATE", "MOFETIL", "AXETIL", "ESTOLATE", "STEARATE", "LACTATE", "GLUCONATE",
    "CARBONATE", "ANHYDROUS", "BASE", "IP", "USP", "BP"})
# One molecule, two names (keys as molecule_key writes them: no spaces, salts dropped).
SYNONYMS = {
    "VITAMIND3": "CHOLECALCIFEROL", "NORMALSALINE": "CHLORIDE", "LIGNOCAINE": "LIDOCAINE",
    "ACETAMINOPHEN": "PARACETAMOL", "AMOXYCILLIN": "AMOXICILLIN",
    "CHLORPHENERAMINE": "CHLORPHENIRAMINE", "EPOETINALFA": "ERYTHROPOIETIN",
    "EPOETIN": "ERYTHROPOIETIN", "OXETHAZINE": "OXETACAINE", "GLYCOPYRONIUM": "GLYCOPYRROLATE",
    "GLYCOPYRRONIUM": "GLYCOPYRROLATE", "NACETYLCYSTEINE": "ACETYLCYSTEINE",
    "SULBACTUM": "SULBACTAM", "DEXTROMETHORPHANHYDROBROMIDE": "DEXTROMETHORPHAN",
    "ALUMINIUMHYDROXIDEGEL": "ALUMINIUMHYDROXIDE", "ALBUTEROL": "SALBUTAMOL",
    "FRUSEMIDE": "FUROSEMIDE"}
# Dosage forms one can stand in for the other.
FORM_GROUPS = {"TAB": "ORAL SOLID", "CAP": "ORAL SOLID", "ORAL LIQUID": "LIQUID", "DROPS": "LIQUID"}
# Mouth-dissolving tablets: a kind of their own (the MR / LIPO / NAB kinds come first).
MD_KIND = "MD"
# Words that say nothing about the variant (forms, packs, units).
PACK_WORDS = frozenset({
    "TAB", "TABS", "TABLET", "CAP", "CAPS", "CAPSULE", "INJ", "VIAL", "VIALS", "AMP", "AMPS",
    "BOTTLE", "CONTAINER", "POW", "POWDER", "SYR", "SYRUP", "SUSP", "SOL", "ML", "MG", "GM", "MCG",
    "IU", "S", "PB", "GB", "PFS", "POUCH", "SACHET", "TUBE", "CONT", "BOTTELE", "RESP", "IV", "OINT",
    "GEL", "CREAM", "DROPS", "SPRAY", "ORAL"})
# A pack weight, not a dose: powders and creams sold by the 50 GM and more.
PACK_WEIGHT_MIN_GM = 50.0
GRAMS = ("GM", "G")

_SCHEDULE = re.compile(r"\(?\bSCH\s*-?\s*[HX]1?\)?")          # '(SCH H1)', 'SCH-X'
_BRACKETS = re.compile(r"\([^)]*\)")
_PARTS = re.compile(r"\+|/|&|\bAND\b|,")
_NON_WORD = re.compile(r"[^A-Z0-9 ]")
_GLUE = re.compile(r"(\d(?:\.\d+)?\s*(?:MCG|MG|GM|IU|ML|%))(?=[A-Z])")
_POWDER = re.compile(r"\bPOW\b|\bGRANULES?\b")
_RATIO = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*(MCG|MG|GM|IU)(?![A-Z])")
_PER_ML = re.compile(r"(\d+(?:\.\d+)?)\s*(MCG|MG|IU)\s*/\s*(?:1\s*)?ML\b")
_ML = re.compile(r"(\d+(?:\.\d+)?)\s*ML\b")
_MD = re.compile(r"\bMD\b|\bODT\b|\bDT\b|ORALLY DISINTEGRATING|MOUTH DISSOLV")
_WORDS = re.compile(r"[A-Z]+")


def molecule_key(minor_group) -> Optional[str]:
    """The Item Master 'Minor Group' as a molecule key: 'ACECLOFENAC+TRAMADOL HCL (SCH H1)' ->
    'ACECLOFENAC+TRAMADOL', 'LIGNOCAINE HCL' -> 'LIDOCAINE'. None when nothing is written."""
    if not isinstance(minor_group, str) or not minor_group.strip():
        return None
    s = _BRACKETS.sub(" ", _SCHEDULE.sub("", norm_text(minor_group)))
    parts = set()
    for part in _PARTS.split(s):
        words = _NON_WORD.sub(" ", part).split()
        key = "".join([w for w in words if w not in SALT_WORDS] or words)
        key = SYNONYMS.get(key, key)
        if key:
            parts.add(key)
    return "+".join(sorted(parts)) or None


def molecule_parts(key: Optional[str]) -> frozenset:
    """'A+B' -> {'A', 'B'}; empty for None."""
    return frozenset(key.split("+")) if key else frozenset()


def close_molecules(key: Optional[str], known: Iterable[str], n: int = 3,
                    cutoff: float = 0.8) -> list:
    """Up to n keys of `known` spelt nearly like `key` (difflib ratio >= cutoff) with the same
    number of parts, best first: 'NICRONDIL' -> ['NICORANDIL']."""
    if not key:
        return []
    size = key.count("+")
    return difflib.get_close_matches(key, sorted({k for k in known if k.count("+") == size}),
                                     n=n, cutoff=cutoff)


def clean_desc(desc) -> str:
    """The description, normalised for the keys: 'DIRIPLEVO 25MGINJ VIAL' -> 'DIRIPLEVO 25MG INJ
    VIAL', 'KABIPRO POW 200GM' -> 'KABIPRO POWDER 200GM', 'ATOCOR CV 10/75MG CAP' -> 'ATOCOR
    CV 10MG/75MG CAP'."""
    s = _GLUE.sub(r"\1 ", norm_text(desc))
    s = _POWDER.sub("POWDER", s)
    return _RATIO.sub(lambda m: f"{m.group(1)}{m.group(3)}/{m.group(2)}{m.group(3)}", s)


def total_dose(text: str) -> Optional[str]:
    """A dose per mL times the one pack volume written: 'COPTRADOL 50MG/ML INJ 2ML AMP' ->
    '100MG'. None when there is no per-mL dose or not exactly one other volume."""
    m = _PER_ML.search(text)
    if not m:
        return None
    volumes = _ML.findall(_PER_ML.sub(" ", text))
    if len(volumes) != 1:
        return None
    unit, factor = S._DOSE[m.group(2)]
    return S._num(float(m.group(1)) * factor * float(volumes[0])) + unit


def strength_key(desc) -> str:
    """The product's strength: the pack's total dose for a per-mL product, else
    spend.oof.strength_key of the cleaned description."""
    s = clean_desc(desc)
    return total_dose(s) or S.strength_key(s)


def dose_key(desc) -> str:
    """The doses alone ('0.5MG|2000MG'), without volumes or pack weights; '' when none is
    written. A per-mL product's total dose as in strength_key."""
    s = clean_desc(desc)
    total = total_dose(s)
    if total:
        return total
    out = set()
    for n, unit in S._AMOUNT.findall(s):
        if unit not in S._DOSE or (unit in GRAMS and float(n) >= PACK_WEIGHT_MIN_GM):
            continue
        canon, factor = S._DOSE[unit]
        out.add(S._num(float(n) * factor) + canon)
    return "|".join(sorted(out))


def pack_size(desc) -> Optional[float]:
    """The largest volume (in mL) or pack weight (50 GM and more, in mg) written; None when
    there is neither. Compared only between products of one molecule and form."""
    sizes = []
    for n, unit in S._AMOUNT.findall(clean_desc(desc)):
        if unit in S._VOLUME:
            sizes.append(float(n) * S._VOLUME[unit][1])
        elif unit in GRAMS and float(n) >= PACK_WEIGHT_MIN_GM:
            sizes.append(float(n) * S._DOSE[unit][1])
    return max(sizes) if sizes else None


def form_key(desc) -> str:
    """spend.oof.form_key of the cleaned description ('KABIPRO POW 200GM' is a POWDER)."""
    return S.form_key(clean_desc(desc))


def form_group(form: str) -> str:
    """TAB and CAP -> 'ORAL SOLID', ORAL LIQUID and DROPS -> 'LIQUID', any other form itself."""
    return FORM_GROUPS.get(form, form)


def kind_key(desc, minor_group=None) -> str:
    """spend.oof.kind_key of the description and minor group ('PACLITAXEL PROTEIN BOUND' is
    NAB), else 'MD' for a mouth-dissolving tablet ('ACIVIR-DT 400MG TAB'), else ''."""
    s = clean_desc(desc)
    return S.kind_key(f"{s} {norm_text(minor_group)}") or (MD_KIND if _MD.search(s) else "")


def descriptor_words(desc) -> frozenset:
    """The words after the brand that name its variant: 'FERTIRO XT TAB' -> {'XT'}; the
    brand (first word), short words and pack words are left out."""
    words = _WORDS.findall(clean_desc(desc))
    if not words:
        return frozenset()
    return frozenset(w for w in words if w != words[0] and w not in PACK_WORDS and len(w) >= 2)


@dataclass(frozen=True)
class Product:
    """One product as the fallback rule compares it."""
    molecule: Optional[str]
    strength: str
    dose: str
    form: str
    form_group: str
    kind: str
    words: frozenset
    pack: Optional[float]


def describe(minor_group, desc) -> Product:
    """Every key of one product from its Item Master minor group and its description."""
    form = form_key(desc)
    return Product(molecule=molecule_key(minor_group), strength=strength_key(desc),
                   dose=dose_key(desc), form=form, form_group=form_group(form),
                   kind=kind_key(desc, minor_group), words=descriptor_words(desc),
                   pack=pack_size(desc))
