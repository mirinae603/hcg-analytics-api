"""HCG's Item Master and Payer Master, and our two committed reference files
(hospital_map.csv, payer_overrides.csv)."""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from app.etl.formulary import classify as K
from app.etl.formulary import config as C
from app.etl.formulary import excel as X
from app.etl.formulary.discover import BuildError


@dataclass
class Item:
    formulary_class: str
    material_desc: str | None
    minor_group: str | None
    manufacturer: str | None


@dataclass
class ItemMaster:
    path: Path
    flag_column: str
    items: dict = field(default_factory=dict)      # material code -> Item

    def __len__(self):
        return len(self.items)


def load_item_master(path) -> ItemMaster:
    """Sheet1, header on row 1: Material, Material Description, Minor Group, Manufacturer,
    'Formulary' or 'Formulary flagging', Category, Material type. Codes are Excel floats."""
    path = Path(path)
    with X.Workbook(path) as wb:
        sheet = C.ITEM_MASTER_SHEET if C.ITEM_MASTER_SHEET in wb.sheet_names else wb.sheet_names[0]
        rows = wb.rows(sheet)
        _, hdr = X.find_header(rows, ["MATERIAL", "CATEGORY"])
        if hdr is None:
            raise BuildError(f"{path.name}: no header row with Material and Category")
        flag_cols = [i for i, h in enumerate(hdr) if h.startswith("FORMULARY")]
        if not flag_cols:
            raise BuildError(f"{path.name}: no column whose header starts with 'Formulary'")
        i_flag = flag_cols[0]

        def col(*names):
            for n in names:
                if n in hdr:
                    return hdr.index(n)
            return None

        i_code, i_cat = hdr.index("MATERIAL"), hdr.index("CATEGORY")
        i_desc = col("MATERIAL DESCRIPTION")
        i_group = col("MINOR GROUP")
        i_mfr = col("MANUFACTURER")
        m = ItemMaster(path=path, flag_column=hdr[i_flag])

        def get(r, i):
            return r[i] if i is not None and i < len(r) else None

        for r in rows:
            if not r:
                continue
            code = X.code(get(r, i_code))
            if not code:
                continue
            try:
                cls = K.formulary_class(X.text(get(r, i_flag)) or "", get(r, i_cat))
            except ValueError as e:
                raise BuildError(f"{path.name}: material {code}: {e}") from None
            old = m.items.get(code)
            if old is not None and old.formulary_class != cls:
                raise BuildError(f"{path.name}: material {code} listed twice with different "
                                 f"classes ({old.formulary_class}, {cls})")
            m.items[code] = Item(cls, X.text(get(r, i_desc)), X.name(get(r, i_group)),
                                 X.name(get(r, i_mfr)))
    if not m.items:
        raise BuildError(f"{path.name}: the Item Master read 0 rows")
    return m


@dataclass(frozen=True)
class BrandRule:
    """One measure of brand_rules.csv: sales of these classes, to these payers (None = any)
    and patient types (None = both), are a breach / a watched figure / information."""
    measure: str
    label: str
    classes: frozenset
    payers: frozenset | None
    patient_types: frozenset | None
    verdict: str
    by_doctor: bool                 # published per doctor x material (formulary_rule_breach)
    alternatives: frozenset         # letters that count as an acceptable alternative brand
    rule_ref: str
    note: str

    def as_dict(self) -> dict:
        return {"measure": self.measure, "label": self.label, "classes": sorted(self.classes),
                "payers": sorted(self.payers) if self.payers else ["*"],
                "patient_types": sorted(self.patient_types) if self.patient_types else ["*"],
                "verdict": self.verdict, "by_doctor": self.by_doctor,
                "alternatives": sorted(self.alternatives), "rule_ref": self.rule_ref,
                "note": self.note}


def _list(v) -> list:
    return [x.strip() for x in str(v or "").split(";") if x.strip()]


def load_brand_rules(ref_dir=None) -> list:
    """brand_rules.csv -> [BrandRule]. HCG's guidelines say which brands are wrong for which
    patients; the file holds them as data so HCG's answers change one file, not the code.
    A missing file means no brand-rule measures."""
    path = C.reference_dir(ref_dir) / C.BRAND_RULES_FILE
    if not path.exists():
        return []
    out, seen = [], set()
    with open(path, newline="", encoding="utf-8") as f:
        for n, row in enumerate(csv.DictReader(f), start=2):
            key = (row.get("measure") or "").strip()
            where = f"{path.name} line {n}"
            if not key or key in seen:
                raise BuildError(f"{where}: needs a unique measure name")
            seen.add(key)
            classes = []
            for c in _list(row.get("classes")):
                cls = K.rule_class(c)
                if cls is None:
                    raise BuildError(f"{where}: {c!r} is not a formulary class (FORMULARY + one of "
                                     f"{', '.join(K.LETTERS)}, OUT OF FORMULARY or NON FORMULARY)")
                classes.append(cls)
            if not classes:
                raise BuildError(f"{where}: needs at least one class")
            payers = [K.norm_bucket(p) or p for p in _list(row.get("payers"))]
            if payers != ["*"] and (not payers or any(p not in K.PAYER_BUCKETS for p in payers)):
                raise BuildError(f"{where}: payers must be * or ; separated {', '.join(K.PAYER_BUCKETS)}")
            types = [t.upper() for t in _list(row.get("patient_types"))]
            if types != ["*"] and (not types or any(t not in ("IP", "OP") for t in types)):
                raise BuildError(f"{where}: patient_types must be *, IP, OP or IP;OP")
            verdict = (row.get("verdict") or "").strip().lower()
            if verdict not in K.VERDICTS:
                raise BuildError(f"{where}: verdict must be one of {', '.join(K.VERDICTS)}")
            by_doctor = (row.get("by_doctor") or "").strip().lower()
            if by_doctor not in ("yes", "no"):
                raise BuildError(f"{where}: by_doctor must be yes or no")
            alts = [K.canonical_category(a) for a in _list(row.get("alternatives"))]
            bad = [a for a in alts if a not in K.LETTERS]
            if bad:
                raise BuildError(f"{where}: unknown alternative letters {bad}")
            out.append(BrandRule(
                measure=key, label=(row.get("label") or key).strip(), classes=frozenset(classes),
                payers=None if payers == ["*"] else frozenset(payers),
                patient_types=None if types == ["*"] else frozenset(types), verdict=verdict,
                by_doctor=by_doctor == "yes", alternatives=frozenset(alts),
                rule_ref=(row.get("rule_ref") or "").strip(), note=(row.get("note") or "").strip()))
    return out


@dataclass
class PayerMaster:
    path: Path
    rows: int
    bucket_column: str
    names: dict                     # normalised CUSTOMERNAME -> bucket of its first row
    conflicts: list                 # names whose rows map to more than one bucket


def load_payer_master(path) -> PayerMaster:
    """Sheet1; row 1 holds marker cells, the header is the row containing CUSTOMERNAME.
    Bucket = the one column whose header starts with 'Final after Discussion' (its date
    suffix will change with the next master)."""
    path = Path(path)
    with X.Workbook(path) as wb:
        sheet = C.PAYER_MASTER_SHEET if C.PAYER_MASTER_SHEET in wb.sheet_names else wb.sheet_names[0]
        rows = wb.rows(sheet)
        _, hdr = X.find_header(rows, ["CUSTOMERNAME"])
        if hdr is None:
            raise BuildError(f"{path.name}: no header row with CUSTOMERNAME")
        cols = [i for i, h in enumerate(hdr) if h.startswith(C.PAYER_BUCKET_PREFIX)]
        if len(cols) != 1:
            raise BuildError(f"{path.name}: expected exactly one column starting with "
                             f"'Final after Discussion', found {len(cols)}")
        i_name, i_b = hdr.index("CUSTOMERNAME"), cols[0]
        pairs, n = [], 0
        for r in rows:
            nm = r[i_name] if r and i_name < len(r) else None
            if X.text(nm) is None:
                continue
            raw_b = r[i_b] if i_b < len(r) else None
            b = K.norm_bucket(raw_b)
            if b is None and X.text(raw_b) is not None:
                raise BuildError(f"{path.name}: unknown bucket {raw_b!r} for payer {nm!r}")
            pairs.append((nm, b))
            n += 1
    names, conflicts = K.first_row_buckets(pairs)
    if not names:
        raise BuildError(f"{path.name}: the Payer Master read 0 payer rows")
    return PayerMaster(path=path, rows=n, bucket_column=hdr[i_b], names=names,
                       conflicts=sorted(conflicts, key=lambda c: c["name"]))


def load_payer_overrides(ref_dir=None) -> dict:
    """payer_overrides.csv -> {normalised name: [(from, to, bucket), ...]}. Blank bounds
    are open. Two rows for one name must not cover the same month."""
    path = C.reference_dir(ref_dir) / C.PAYER_OVERRIDES_FILE
    if not path.exists():
        return {}
    out: dict = {}
    with open(path, newline="", encoding="utf-8") as f:
        for n, row in enumerate(csv.DictReader(f), start=2):
            k = K.payer_key(row.get("payer_name"))
            b = K.norm_bucket(row.get("bucket"))
            lo = (row.get("from_period") or "").strip() or None
            hi = (row.get("to_period") or "").strip() or None
            if not k or b is None:
                raise BuildError(f"{path.name} line {n}: needs payer_name and a bucket in "
                                 f"{', '.join(K.PAYER_BUCKETS)}")
            for p in (lo, hi):
                if p is not None and not K.is_period(p):
                    raise BuildError(f"{path.name} line {n}: period {p!r} is not YYYY-MM")
            if lo and hi and lo > hi:
                raise BuildError(f"{path.name} line {n}: from_period is after to_period")
            for olo, ohi, _ in out.get(k, ()):
                if (lo is None or ohi is None or lo <= ohi) and (hi is None or olo is None or olo <= hi):
                    raise BuildError(f"{path.name} line {n}: overlaps another row for {k!r}")
            out.setdefault(k, []).append((lo, hi, b))
    return out


def load_hospital_map(ref_dir=None) -> dict:
    """hospital_map.csv -> {his_code: row dict}. his_code is the invoice prefix (KABHK)."""
    path = C.reference_dir(ref_dir) / C.HOSPITAL_MAP_FILE
    if not path.exists():
        raise BuildError(f"hospital map not found: {path}")
    out: dict = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            k = (row.get("his_code") or "").strip().upper()
            plant = (row.get("sap_plant") or "").strip().upper()
            if not k:
                continue
            if not plant:
                raise BuildError(f"{path.name}: {k} has no sap_plant")
            if k in out:
                raise BuildError(f"{path.name}: {k} listed twice")
            out[k] = {**row, "his_code": k, "sap_plant": plant}
    plants = [r["sap_plant"] for r in out.values()]
    dup = sorted({p for p in plants if plants.count(p) > 1})
    if dup:
        raise BuildError(f"{path.name}: SAP plant used by more than one hospital: {dup}")
    return out
