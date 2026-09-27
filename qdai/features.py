"""Feature registry: how each dataset column is presented in the prediction form.

The registry only *describes* features. Ranges, defaults and code lists are
always derived from the loaded dataset, so the form adapts to any variant of
the dataset (e.g. with or without 'Admission grade'). Any column the registry
does not know is still exposed, under "Additional signals", so the model always
receives a real value for every feature it was trained on.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

SECTIONS = [
    ("academic", "Academic profile", "How the student entered the programme."),
    ("profile", "Student profile", "Who the student is and how they enrolled."),
    ("socio", "Socioeconomic context", "Family, finances and the wider economy at enrollment."),
    ("progress", "Academic progress", "What has happened in the first two semesters."),
]


@dataclass(frozen=True)
class Spec:
    name: str
    section: str
    label: str
    kind: str                     # binary | code | int | float
    options: tuple | None = None  # binary: ((value, label), ...)
    help: str | None = None


def _bin(a, b):  # (value0_label, value1_label)
    return ((0, a), (1, b))


_S = Spec
SPECS: list[Spec] = [
    # -- academic profile
    _S("Admission grade", "academic", "Admission grade", "float", help="Score at admission (0–200)."),
    _S("Previous qualification (grade)", "academic", "Previous qualification grade", "float"),
    _S("Previous qualification", "academic", "Previous qualification", "code"),
    _S("Application mode", "academic", "Application mode", "code"),
    _S("Application order", "academic", "Application order", "int",
       help="0 means the programme was the student's first choice."),
    _S("Course", "academic", "Course", "code"),
    _S("Daytime/evening attendance", "academic", "Attendance schedule", "binary",
       options=((0, "Evening"), (1, "Daytime"))),
    # -- student profile
    _S("Age at enrollment", "profile", "Age at enrollment", "int"),
    _S("Gender", "profile", "Gender", "binary", options=((0, "Female"), (1, "Male"))),
    _S("Marital status", "profile", "Marital status", "code"),
    _S("Nacionality", "profile", "Nationality", "code"),
    _S("International", "profile", "International student", "binary", options=_bin("No", "Yes")),
    _S("Displaced", "profile", "Displaced from home", "binary", options=_bin("No", "Yes")),
    _S("Educational special needs", "profile", "Educational special needs", "binary", options=_bin("No", "Yes")),
    # -- socioeconomic context
    _S("Tuition fees up to date", "socio", "Tuition fees up to date", "binary", options=_bin("No", "Yes")),
    _S("Debtor", "socio", "Has outstanding debt", "binary", options=_bin("No", "Yes")),
    _S("Scholarship holder", "socio", "Scholarship holder", "binary", options=_bin("No", "Yes")),
    _S("Mother's qualification", "socio", "Mother's qualification", "code"),
    _S("Father's qualification", "socio", "Father's qualification", "code"),
    _S("Mother's occupation", "socio", "Mother's occupation", "code"),
    _S("Father's occupation", "socio", "Father's occupation", "code"),
    _S("Unemployment rate", "socio", "Unemployment rate (%)", "float", help="National rate at enrollment."),
    _S("Inflation rate", "socio", "Inflation rate (%)", "float", help="National rate at enrollment."),
    _S("GDP", "socio", "GDP growth (%)", "float", help="National figure at enrollment."),
]

for _sem, _tag in (("1st", "Semester 1"), ("2nd", "Semester 2")):
    SPECS += [
        _S(f"Curricular units {_sem} sem (enrolled)", "progress", f"{_tag}: units enrolled", "int"),
        _S(f"Curricular units {_sem} sem (evaluations)", "progress", f"{_tag}: evaluations taken", "int"),
        _S(f"Curricular units {_sem} sem (approved)", "progress", f"{_tag}: units approved", "int"),
        _S(f"Curricular units {_sem} sem (grade)", "progress", f"{_tag}: average grade", "float",
           help="Average grade out of 20."),
        _S(f"Curricular units {_sem} sem (credited)", "progress", f"{_tag}: units credited", "int"),
        _S(f"Curricular units {_sem} sem (without evaluations)", "progress",
           f"{_tag}: units without evaluation", "int"),
    ]

BY_NAME = {s.name: s for s in SPECS}

# Value labels that follow the public UCI dataset documentation. Any code that is
# not listed falls back to "Code N" instead of guessing a meaning.
CODE_LABELS = {
    "Marital status": {1: "Single", 2: "Married", 3: "Widower", 4: "Divorced",
                       5: "Facto union", 6: "Legally separated"},
    "Course": {33: "Biofuel Production Technologies", 171: "Animation and Multimedia Design",
               8014: "Social Service (evening)", 9003: "Agronomy", 9070: "Communication Design",
               9085: "Veterinary Nursing", 9119: "Informatics Engineering", 9130: "Equinculture",
               9147: "Management", 9238: "Social Service", 9254: "Tourism", 9500: "Nursing",
               9556: "Oral Hygiene", 9670: "Advertising and Marketing Management",
               9773: "Journalism and Communication", 9853: "Basic Education",
               9991: "Management (evening)"},
}


def code_label(feature: str, value) -> str:
    v = int(value) if float(value).is_integer() else value
    name = CODE_LABELS.get(feature, {}).get(v)
    return f"{name} ({v})" if name else f"Code {v}"


# --------------------------------------------------------------- domains
# Used by the intervention engine to group model signals into themes.
def domain_of(feature: str) -> str:
    f = feature.lower()
    if f.startswith("curricular units"):
        return "progress"
    if feature in ("Tuition fees up to date", "Debtor", "Scholarship holder"):
        return "financial"
    if feature in ("Unemployment rate", "Inflation rate", "GDP"):
        return "economy"
    if feature in ("Mother's qualification", "Father's qualification",
                   "Mother's occupation", "Father's occupation"):
        return "family"
    if feature in ("Age at enrollment", "Gender", "Marital status", "Nacionality",
                   "International", "Displaced", "Educational special needs"):
        return "profile"
    return "pathway"   # admission grade, previous qualification, application, course, schedule


def infer_kind(series: pd.Series) -> str:
    vals = set(series.dropna().unique().tolist())
    if vals <= {0, 1}:
        return "binary"
    if pd.api.types.is_integer_dtype(series) or all(float(v).is_integer() for v in vals):
        return "int"
    return "float"


def spec_for(name: str, series: pd.Series) -> Spec:
    """Registry entry if known, otherwise a sensible generic one inferred from the data."""
    if name in BY_NAME:
        return BY_NAME[name]
    kind = infer_kind(series)
    opts = _bin("No", "Yes") if kind == "binary" else None
    return Spec(name, "extra", name, kind, options=opts)
