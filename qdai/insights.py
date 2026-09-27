"""Data-derived findings.

Rules that keep this module honest:
  * every figure is computed from the loaded dataframe
  * a comparison is only stated when BOTH groups have at least MIN_GROUP_N students
  * wording is descriptive ("in this dataset"), never causal
  * if a needed column is missing the function returns None / an empty result
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import MIN_GROUP_N
from .features import BY_NAME


@dataclass
class Finding:
    figure: str      # big editorial number, e.g. "3.4×"
    headline: str    # short statement
    detail: str      # the numbers behind it


def has(df: pd.DataFrame, *cols: str) -> bool:
    return "Target" in df.columns and all(c in df.columns for c in cols)


def is_dropout(df: pd.DataFrame) -> pd.Series:
    return (df["Target"].astype(str) == "Dropout")


# ------------------------------------------------------------ rate tables
def rate_by(df: pd.DataFrame, by: pd.Series, min_n: int = MIN_GROUP_N) -> pd.DataFrame:
    """Dropout rate (%) per group, with group size and a reliability flag."""
    d = is_dropout(df)
    g = pd.DataFrame({"group": by.values, "drop": d.values}).dropna(subset=["group"])
    out = g.groupby("group", observed=True)["drop"].agg(n="size", dropouts="sum").reset_index()
    out["rate"] = out["dropouts"] / out["n"] * 100
    out["reliable"] = out["n"] >= min_n
    return out


def age_bands(df: pd.DataFrame) -> pd.Series | None:
    if not has(df, "Age at enrollment"):
        return None
    return pd.cut(df["Age at enrollment"], bins=[0, 19, 24, 29, 200],
                  labels=["Under 20", "20–24", "25–29", "30 and over"])


def unit_bands(df: pd.DataFrame, col: str) -> pd.Series | None:
    if not has(df, col):
        return None
    return pd.cut(df[col], bins=[-1, 0, 2, 4, 6, 1000],
                  labels=["0", "1–2", "3–4", "5–6", "7 or more"])


def numeric_bins(series: pd.Series, q: int = 5) -> pd.Series:
    """Equal-width bins (as the original feature explorer used) with readable labels."""
    b = pd.cut(series, bins=q)
    b = b.cat.rename_categories(lambda iv: f"{iv.left:.3g}–{iv.right:.3g}")
    return b


def extremes_sentence(table: pd.DataFrame, subject: str) -> str | None:
    t = table[table["reliable"]].sort_values("rate")
    if len(t) < 2:
        return None
    lo, hi = t.iloc[0], t.iloc[-1]
    return (f"{subject}: the dropout rate is highest for {hi['group']} "
            f"({hi['rate']:.0f}%, {int(hi['n']):,} students) and lowest for {lo['group']} "
            f"({lo['rate']:.0f}%, {int(lo['n']):,} students).")


def mean_gap_sentence(df: pd.DataFrame, col: str, label: str, fmt: str = "{:.1f}") -> str | None:
    if not has(df, col):
        return None
    grp = df.groupby(df["Target"].astype(str))[col]
    means, ns = grp.mean(), grp.size()
    if "Graduate" not in means or "Dropout" not in means:
        return None
    if ns["Graduate"] < MIN_GROUP_N or ns["Dropout"] < MIN_GROUP_N:
        return None
    g, d = means["Graduate"], means["Dropout"]
    return (f"{label}: graduates average {fmt.format(g)}, students who dropped out average "
            f"{fmt.format(d)} (difference {fmt.format(abs(g - d))}).")


# -------------------------------------------------------- correlations
def dropout_correlates(df: pd.DataFrame) -> pd.DataFrame:
    """Correlation of each numeric, non-code column with a 0/1 dropout indicator.

    Code-type columns (course, application mode, occupations...) are excluded
    because a linear correlation with an arbitrary numeric code is meaningless.
    """
    if "Target" not in df.columns:
        return pd.DataFrame(columns=["feature", "corr"])
    y = is_dropout(df).astype(float)
    rows = []
    for c in df.select_dtypes(include="number").columns:
        if c in ("id", "Target"):
            continue
        spec = BY_NAME.get(c)
        if spec is not None and spec.kind == "code":
            continue
        if df[c].nunique() < 2:
            continue
        r = df[c].corr(y)
        if pd.notna(r):
            rows.append((c, float(r)))
    return pd.DataFrame(rows, columns=["feature", "corr"]).sort_values("corr", key=np.abs, ascending=False)


# -------------------------------------------------------- headline findings
def headline_findings(df: pd.DataFrame) -> list[Finding]:
    out: list[Finding] = []
    if has(df, "Tuition fees up to date"):
        t = rate_by(df, df["Tuition fees up to date"].map({1: "Yes", 0: "No"})).set_index("group")
        if {"Yes", "No"} <= set(t.index) and t["reliable"].all() and t.loc["Yes", "rate"] > 0:
            r = t.loc["No", "rate"] / t.loc["Yes", "rate"]
            out.append(Finding(
                f"{r:.1f}×", "Unpaid tuition and dropout",
                f"{t.loc['No', 'rate']:.0f}% of students whose fees were not up to date dropped out "
                f"({int(t.loc['No', 'n']):,} students), against {t.loc['Yes', 'rate']:.0f}% of those up to date "
                f"({int(t.loc['Yes', 'n']):,})."))
    if has(df, "Scholarship holder"):
        t = rate_by(df, df["Scholarship holder"].map({1: "Holders", 0: "Non-holders"})).set_index("group")
        if {"Holders", "Non-holders"} <= set(t.index) and t["reliable"].all() and t.loc["Holders", "rate"] > 0:
            r = t.loc["Non-holders", "rate"] / t.loc["Holders", "rate"]
            out.append(Finding(
                f"{r:.1f}×", "Scholarship status and dropout",
                f"{t.loc['Non-holders', 'rate']:.0f}% of students without a scholarship dropped out "
                f"({int(t.loc['Non-holders', 'n']):,}), against {t.loc['Holders', 'rate']:.0f}% of "
                f"scholarship holders ({int(t.loc['Holders', 'n']):,})."))
    ub = unit_bands(df, "Curricular units 2nd sem (approved)")
    if ub is not None:
        t = rate_by(df, ub).set_index("group")
        if "0" in t.index and "5–6" in t.index and t.loc["0", "reliable"] and t.loc["5–6", "reliable"]:
            out.append(Finding(
                f"{t.loc['0', 'rate']:.0f}%", "Dropout with no second-semester units approved",
                f"{int(t.loc['0', 'dropouts'])} of {int(t.loc['0', 'n']):,} students who passed no second-semester "
                f"units dropped out, against {t.loc['5–6', 'rate']:.0f}% of those who passed 5–6 units."))
    ab = age_bands(df)
    if ab is not None:
        t = rate_by(df, ab).set_index("group")
        if "Under 20" in t.index and "30 and over" in t.index and t.loc["Under 20", "reliable"] \
                and t.loc["30 and over", "reliable"]:
            out.append(Finding(
                f"{t.loc['30 and over', 'rate']:.0f}%", "Dropout among students enrolling at 30 or older",
                f"{t.loc['30 and over', 'rate']:.0f}% of students aged 30 and over dropped out "
                f"({int(t.loc['30 and over', 'n']):,}), against {t.loc['Under 20', 'rate']:.0f}% of "
                f"students under 20 ({int(t.loc['Under 20', 'n']):,})."))
    return out


def graduate_median(df: pd.DataFrame) -> pd.Series:
    if "Target" not in df.columns:
        return pd.Series(dtype=float)
    g = df[df["Target"].astype(str) == "Graduate"]
    return g.select_dtypes(include="number").median()


def risk_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Prevalence and dropout rate for simple, transparent risk flags (extends the original risk-factor chart)."""
    flags = []
    def add(name, mask):
        if mask is None or mask.sum() == 0:
            return
        sub = df[mask]
        flags.append((name, int(mask.sum()),
                      float(is_dropout(sub).mean() * 100) if len(sub) else np.nan))
    if has(df, "Curricular units 1st sem (approved)"):
        add("Two or fewer first-semester units approved", df["Curricular units 1st sem (approved)"] <= 2)
    if has(df, "Age at enrollment"):
        add("Enrolled aged over 25", df["Age at enrollment"] > 25)
    if has(df, "Scholarship holder"):
        add("No scholarship", df["Scholarship holder"] == 0)
    if has(df, "Displaced"):
        add("Displaced from home", df["Displaced"] == 1)
    if has(df, "Tuition fees up to date"):
        add("Tuition not up to date", df["Tuition fees up to date"] == 0)
    if has(df, "Debtor"):
        add("Has outstanding debt", df["Debtor"] == 1)
    return pd.DataFrame(flags, columns=["flag", "students", "dropout_rate"])
