"""Preventative-action engine.

Every recommendation is produced from two real inputs:
  1. the SHAP contributions of THIS student towards the Dropout outcome
  2. the values that were entered for the student
plus explicit rules defined in this file. Nothing is generated freely and
nothing is presented as clinical or psychological advice: these are suggested
institutional follow-ups for an advisor to weigh.

A theme is only suggested when the features in that theme push the dropout
probability up by at least INTERVENTION_MIN_PUSH (2 percentage points), or when
a direct financial flag is present and the dropout probability is above the
Low band.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .config import INTERVENTION_MIN_PUSH, RISK_LOW_MAX
from .features import domain_of


@dataclass
class Intervention:
    key: str
    title: str
    push: float                       # summed upward contribution to P(dropout), 0-1
    evidence: list[str]
    action: str
    drivers: list[tuple[str, float]]  # (feature, contribution in probability points)
    priority: int = 0


@dataclass
class Plan:
    items: list[Intervention] = field(default_factory=list)
    headline: str = ""
    context_note: str | None = None
    protective: list[tuple[str, float, float]] = field(default_factory=list)


def _f(values: dict, name: str, default=None):
    v = values.get(name, default)
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _drivers(contrib: pd.DataFrame, domain: str, n: int = 3) -> list[tuple[str, float]]:
    sub = contrib[(contrib["domain"] == domain) & (contrib["shap"] > 0.0005)].sort_values("shap", ascending=False)
    return [(r.feature, float(r.shap) * 100) for r in sub.head(n).itertuples()]


def build_plan(contrib: pd.DataFrame, values: dict, p_dropout: float,
               graduate_median: pd.Series | None = None) -> Plan:
    """contrib: columns feature, value, shap (towards Dropout). values: the student's inputs."""
    contrib = contrib.copy()
    contrib["domain"] = contrib["feature"].map(domain_of)
    push = contrib[contrib["shap"] > 0].groupby("domain")["shap"].sum().to_dict()
    gm = graduate_median if graduate_median is not None else pd.Series(dtype=float)
    plan = Plan()
    items: list[Intervention] = []

    # ---- academic progress -------------------------------------------------
    if push.get("progress", 0) >= INTERVENTION_MIN_PUSH:
        ev = []
        for tag, n in (("1st", 1), ("2nd", 2)):
            enrolled = _f(values, f"Curricular units {tag} sem (enrolled)")
            approved = _f(values, f"Curricular units {tag} sem (approved)")
            grade = _f(values, f"Curricular units {tag} sem (grade)")
            unassessed = _f(values, f"Curricular units {tag} sem (without evaluations)")
            if enrolled and approved is not None and approved / enrolled < 0.5:
                ev.append(f"Semester {n}: {approved:.0f} of {enrolled:.0f} enrolled units approved.")
            gcol = f"Curricular units {tag} sem (grade)"
            if grade is not None and gcol in gm.index and grade < float(gm[gcol]) - 1:
                ev.append(f"Semester {n} average grade is {grade:.1f}, against a median of "
                          f"{float(gm[gcol]):.1f} among graduates in this dataset.")
            if unassessed and unassessed > 0:
                ev.append(f"Semester {n}: {unassessed:.0f} unit(s) without an evaluation.")
        items.append(Intervention(
            "academic", "Academic support", push["progress"],
            ev or ["Semester results contribute more to the dropout estimate than any other theme."],
            "Arrange advisor follow-up on the units not yet passed, and consider tutoring or "
            "structured study support before the next assessment period.",
            _drivers(contrib, "progress")))

    # ---- financial ---------------------------------------------------------
    tuition = _f(values, "Tuition fees up to date")
    debtor = _f(values, "Debtor")
    scholar = _f(values, "Scholarship holder")
    fin_flag = (tuition == 0) or (debtor == 1)
    if push.get("financial", 0) >= INTERVENTION_MIN_PUSH or (fin_flag and p_dropout >= RISK_LOW_MAX):
        ev = []
        if tuition == 0:
            ev.append("Tuition fees are not up to date.")
        if debtor == 1:
            ev.append("An outstanding debt is recorded.")
        if scholar == 0 and (tuition == 0 or debtor == 1 or push.get("financial", 0) >= INTERVENTION_MIN_PUSH):
            ev.append("The student does not currently hold a scholarship.")
        items.append(Intervention(
            "financial", "Financial support",
            max(push.get("financial", 0), 0.0) + (0.001 if fin_flag else 0),
            ev or ["Financial indicators contribute to the dropout estimate."],
            "Review scholarship eligibility, fee-payment plans or other financial assistance "
            "options with the student.",
            _drivers(contrib, "financial")))

    # ---- mentoring / engagement -------------------------------------------
    if push.get("profile", 0) >= INTERVENTION_MIN_PUSH:
        ev = []
        age = _f(values, "Age at enrollment")
        if age is not None and age >= 25:
            ev.append(f"Enrolled at age {age:.0f}; age at enrollment raises the model's dropout estimate for this student.")
        if _f(values, "Displaced") == 1:
            ev.append("The student is displaced from their home area.")
        if _f(values, "Educational special needs") == 1:
            ev.append("Educational special needs are recorded.")
        items.append(Intervention(
            "mentoring", "Mentoring and engagement", push["profile"],
            ev or ["Student-profile factors contribute to the dropout estimate."],
            "Offer an advisor or peer mentor and schedule an early check-in on workload and how "
            "the student is settling in.",
            _drivers(contrib, "profile")))

    # ---- pathway / programme fit ------------------------------------------
    if push.get("pathway", 0) >= INTERVENTION_MIN_PUSH:
        ev = []
        order = _f(values, "Application order")
        if order is not None and order > 0:
            ev.append(f"The programme was choice number {order:.0f} on the application, not the first.")
        adm = _f(values, "Admission grade")
        if adm is not None and "Admission grade" in gm.index and adm < float(gm["Admission grade"]) - 5:
            ev.append(f"Admission grade of {adm:.0f} is below the graduate median of "
                      f"{float(gm['Admission grade']):.0f} in this dataset.")
        if _f(values, "Daytime/evening attendance") == 0:
            ev.append("The student attends in the evening schedule.")
        items.append(Intervention(
            "pathway", "Programme fit review", push["pathway"],
            ev or ["Admission and pathway factors contribute to the dropout estimate."],
            "Discuss programme fit, workload and alternative pathways with an academic advisor.",
            _drivers(contrib, "pathway")))

    items.sort(key=lambda i: i.push, reverse=True)
    for rank, it in enumerate(items, 1):
        it.priority = rank
    plan.items = items

    # ---- what is deliberately NOT turned into an action -------------------
    outside = push.get("family", 0) + push.get("economy", 0)
    if outside >= INTERVENTION_MIN_PUSH:
        plan.context_note = (
            f"Family background and national economic indicators add about {outside * 100:.0f} "
            "percentage points to the dropout estimate. They are context the institution cannot "
            "change, so no action is suggested for them.")

    plan.protective = [(r.feature, float(r.shap) * 100, float(r.value))
                       for r in contrib[contrib["shap"] < 0].sort_values("shap").head(3).itertuples()]

    if items and p_dropout < RISK_LOW_MAX:
        plan.headline = ("Estimated dropout risk is low, so no intervention is needed. The themes below nudge the "
                         "estimate up slightly and are worth keeping an eye on.")
    elif items:
        plan.headline = ("Recommended institutional interventions, ordered by how much each theme "
                         "raises the estimated dropout probability.")
    elif p_dropout < RISK_LOW_MAX:
        plan.headline = ("No intervention is flagged. Continue routine monitoring and keep the "
                         "supports that are already working.")
    else:
        plan.headline = ("No single theme stands out. A general advisor check-in is a reasonable "
                         "next step.")
    return plan
