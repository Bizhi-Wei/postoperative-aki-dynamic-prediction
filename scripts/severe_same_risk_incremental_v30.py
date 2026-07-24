"""Same-risk-set incremental-value analysis for severe postoperative SCr-AKI.

This analysis reuses the audited v25 paired-comparison machinery but replaces
the outcome, datasets, predictor sets, and split with the v27 severe-AKI
specification.  Within each target risk set, only the information horizon is
changed; patients, outcome, model family, and held-out split remain identical.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import same_risk_set_incremental_analysis_v25 as engine  # noqa: E402
from final_sensitivity_and_actionable_analysis_v14 import simplified_predictors  # noqa: E402


OUT = ROOT / "outputs" / "modeling_v30_severe_same_risk_incremental"
DATA_DIR = ROOT / "outputs" / "modeling_v27_severity_recovery"
OUTCOME = "outcome_severe_scr_after_landmark_to_7d"
SPLIT_FILE = DATA_DIR / "audit_v27_subject_split_assignment.csv"
STATUS_FEATURES = [
    "prior_stage1_aki_by_landmark",
    "hours_since_first_aki_at_landmark",
    "current_scr_stage_at_landmark",
    "current_scr_at_landmark",
    "current_scr_ratio_at_landmark",
    "scr_measurement_n_by_landmark",
]


def load_severe_data(landmark: int) -> pd.DataFrame:
    path = DATA_DIR / f"dataset_v27_severe_{landmark}h.csv"
    data = pd.read_csv(path, low_memory=False)
    if OUTCOME not in data or data["stay_id"].duplicated().any():
        raise AssertionError(f"Invalid severe-AKI dataset at {landmark} h")
    return data


def severe_predictors(info_hours: int) -> list[str]:
    data = load_severe_data(info_hours)
    predictors = [p for p in simplified_predictors(info_hours) if p in data.columns]
    if info_hours > 0:
        predictors.extend(p for p in STATUS_FEATURES if p in data.columns and p not in predictors)
    forbidden = {OUTCOME, *engine.KEYS, "landmark_hours"}
    if forbidden.intersection(predictors):
        raise AssertionError("Metadata or outcome leakage in severe predictor set")
    if info_hours == 0 and any("_0_6h_" in p or "_0_24h_" in p for p in predictors):
        raise AssertionError("Post-index feature in admission information set")
    if info_hours == 6 and any("_0_24h_" in p for p in predictors):
        raise AssertionError("0-24 h feature in 0-6 h information set")
    return predictors


def inherited_split(data: pd.DataFrame) -> tuple[set[int], set[int], dict[str, float]]:
    split = pd.read_csv(SPLIT_FILE)
    subject_labels = split[["subject_id", "split"]].drop_duplicates()
    conflicting = subject_labels.groupby("subject_id")["split"].nunique().gt(1)
    if conflicting.any():
        raise AssertionError("A subject has conflicting v27 split assignments")
    train_subjects = set(subject_labels.loc[subject_labels["split"].eq("train"), "subject_id"].astype(int))
    test_subjects = set(subject_labels.loc[subject_labels["split"].eq("test"), "subject_id"].astype(int))
    if train_subjects & test_subjects:
        raise AssertionError("Subject overlap in inherited v27 split")
    group = data["subject_id"].astype(int)
    train = group.isin(train_subjects)
    test = group.isin(test_subjects)
    if (~(train | test)).any() or (train & test).any():
        raise AssertionError("Incomplete inherited split coverage")
    y = data[OUTCOME].astype(int)
    return train_subjects, test_subjects, {
        "overall_n": len(data),
        "train_n": int(train.sum()),
        "test_n": int(test.sum()),
        "overall_event_rate": float(y.mean()),
        "train_event_rate": float(y.loc[train].mean()),
        "test_event_rate": float(y.loc[test].mean()),
        "train_subjects": len(train_subjects),
        "test_subjects": len(test_subjects),
    }


def configure_engine() -> None:
    engine.OUT = OUT
    engine.OUTCOME = OUTCOME
    engine.load_data = load_severe_data
    engine.choose_grouped_split = inherited_split
    engine.information_predictors = severe_predictors
    engine.PRIMARY_MODEL = {6: "XGBoost", 24: "Logistic Regression"}
    original_save = engine.save_figure

    def save_v30(fig, stem: str) -> None:
        original_save(fig, stem.replace("v25", "v30"))

    engine.save_figure = save_v30


def fmt_ci(row: pd.Series, metric: str) -> str:
    return (
        f"{row[metric]:.3f} ({row[f'{metric}_ci_lower']:.3f}–"
        f"{row[f'{metric}_ci_upper']:.3f})"
    )


def write_audit(performance: pd.DataFrame, deltas: pd.DataFrame, audit: pd.DataFrame) -> None:
    primary = performance.loc[performance["primary_model_for_risk_set"].astype(bool)].copy()
    pdeltas = deltas.loc[deltas["primary_model_for_risk_set"].astype(bool)].copy()
    d6 = pdeltas.loc[
        pdeltas["risk_set_hours"].eq(6)
        & pdeltas["new_information_hours"].eq(6)
        & pdeltas["reference_information_hours"].eq(0)
    ].iloc[0]
    d24 = pdeltas.loc[
        pdeltas["risk_set_hours"].eq(24)
        & pdeltas["new_information_hours"].eq(24)
        & pdeltas["reference_information_hours"].eq(6)
    ].iloc[0]
    p6 = primary.loc[primary["risk_set_hours"].eq(6)].sort_values("information_hours")
    p24 = primary.loc[primary["risk_set_hours"].eq(24)].sort_values("information_hours")
    lines = [
        "# v30 severe-AKI same-risk-set incremental-value analysis",
        "",
        "## Design",
        "",
        "The target is new KDIGO serum-creatinine stage 2/3 AKI after the landmark through ICU day 7. "
        "Patients already at stage 2/3 were excluded before forming the 6-h and 24-h risk sets. Within "
        "each target risk set, all information variants used identical patients, outcomes, v27 subject-grouped "
        "train/test assignments, and model family. Confidence intervals and paired differences used 1,000 "
        "subject-cluster bootstrap resamples.",
        "",
        "## Primary results",
        "",
        f"- 6-h risk set: held-out n={int(p6.iloc[0].test_n):,}, events={int(p6.iloc[0].test_event_n):,}. "
        f"XGBoost AUROC was {p6.iloc[0].auroc:.3f} with admission information and {p6.iloc[-1].auroc:.3f} "
        f"with 0–6-h information; paired ΔAUROC {d6.delta_auroc:+.3f} "
        f"(95% CI {d6.delta_auroc_ci_lower:+.3f} to {d6.delta_auroc_ci_upper:+.3f}).",
        f"- 24-h risk set: held-out n={int(p24.iloc[0].test_n):,}, events={int(p24.iloc[0].test_event_n):,}. "
        f"Logistic-regression AUROC was {p24.iloc[1].auroc:.3f} with 0–6-h information and "
        f"{p24.iloc[-1].auroc:.3f} with 0–24-h information; paired ΔAUROC {d24.delta_auroc:+.3f} "
        f"(95% CI {d24.delta_auroc_ci_lower:+.3f} to {d24.delta_auroc_ci_upper:+.3f}).",
        "",
        "## Integrity checks",
        "",
        f"- Information-set source coverage: {audit.source_coverage_percent.min():.1f}%–{audit.source_coverage_percent.max():.1f}%.",
        "- Subject overlap between training and held-out test partitions: 0.",
        "- Same-risk comparisons use row-identical targets and paired held-out predictions.",
        "- The outcome and post-landmark measurements are excluded from predictor sets.",
    ]
    (OUT / "audit_v30_results_brief.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    checks = {
        "outcome": OUTCOME,
        "bootstrap_resamples": engine.BOOTSTRAPS,
        "information_source_coverage_min_percent": float(audit.source_coverage_percent.min()),
        "information_source_coverage_max_percent": float(audit.source_coverage_percent.max()),
        "subject_overlap_n": 0,
        "performance_rows": len(performance),
        "paired_delta_rows": len(deltas),
    }
    (OUT / "audit_v30_validation.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")


def plot_incremental_deltas(deltas: pd.DataFrame) -> None:
    """Publication panel focused explicitly on severe SCr-AKI."""
    engine.setup_style()
    primary = deltas.loc[deltas["primary_model_for_risk_set"].astype(bool)].copy()
    primary["comparison"] = primary.apply(
        lambda r: (
            f"{int(r.risk_set_hours)} h risk set: "
            f"{engine.INFORMATION_SHORT[int(r.new_information_hours)]} vs "
            f"{engine.INFORMATION_SHORT[int(r.reference_information_hours)]}"
        ), axis=1,
    )
    order = [
        "6 h risk set: 0-6 h vs 0 h", "24 h risk set: 0-6 h vs 0 h",
        "24 h risk set: 0-24 h vs 0-6 h", "24 h risk set: 0-24 h vs 0 h",
    ]
    primary["comparison"] = pd.Categorical(primary["comparison"], categories=order, ordered=True)
    primary = primary.sort_values("comparison")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.2), sharey=True)
    specs = [
        ("auroc", "Delta AUROC", "Positive favours later information"),
        ("auprc", "Delta AUPRC", "Positive favours later information"),
        ("brier_score", "Delta Brier score", "Negative favours later information"),
    ]
    y = np.arange(len(primary))
    colors = ["#4C78A8" if int(x) == 6 else "#D97706" for x in primary["risk_set_hours"]]
    for panel, (ax, (metric, xlabel, subtitle)) in enumerate(zip(axes, specs)):
        value = primary[f"delta_{metric}"].to_numpy(float)
        lower = primary[f"delta_{metric}_ci_lower"].to_numpy(float)
        upper = primary[f"delta_{metric}_ci_upper"].to_numpy(float)
        for i, color in enumerate(colors):
            ax.errorbar(value[i], y[i], xerr=np.array([[value[i] - lower[i]], [upper[i] - value[i]]]),
                        fmt="none", ecolor=color, elinewidth=1.2, capsize=2.5)
        ax.scatter(value, y, c=colors, s=24, zorder=3)
        ax.axvline(0, color="#98A2B3", linestyle=":", linewidth=0.9)
        ax.set_xlabel(xlabel); ax.set_title(subtitle, fontsize=7.2)
        ax.grid(axis="x", color="#E6E8F0", linewidth=0.6); ax.invert_yaxis()
        engine.panel_label(ax, chr(97 + panel))
    axes[0].set_yticks(y, [str(x) for x in primary["comparison"]])
    fig.suptitle("Incremental value of later information for severe SCr-AKI",
                 x=0.08, y=1.02, ha="left", fontsize=9.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96), w_pad=1.5)
    engine.save_figure(fig, "figure_v30_paired_incremental_value")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    configure_engine()
    performance, deltas, predictions, audit = engine.fit_models()
    calibration, dca = engine.source_tables(performance, predictions)

    for frame in [performance, deltas, calibration, dca]:
        numeric = frame.select_dtypes(include=[np.number]).columns
        frame[numeric] = frame[numeric].round(8)
    performance.to_csv(OUT / "model_v30_severe_same_risk_performance.csv", index=False)
    deltas.to_csv(OUT / "model_v30_severe_paired_incremental_deltas.csv", index=False)
    predictions.to_csv(OUT / "model_v30_severe_same_risk_test_predictions.csv", index=False)
    audit.to_csv(OUT / "audit_v30_severe_information_set_coverage.csv", index=False)
    calibration.to_csv(OUT / "figure_v30_severe_calibration_source_data.csv", index=False)
    dca.to_csv(OUT / "figure_v30_severe_dca_source_data.csv", index=False)

    engine.plot_discrimination(performance, predictions)
    engine.plot_calibration_dca(calibration, dca)
    plot_incremental_deltas(deltas)
    write_audit(performance, deltas, audit)

    old_manifest = OUT / "audit_v25_split_manifest.json"
    if old_manifest.exists():
        old_manifest.replace(OUT / "audit_v30_split_manifest.json")
    print(performance.loc[performance["primary_model_for_risk_set"].astype(bool), [
        "risk_set_hours", "information_set", "model", "test_n", "test_event_n", "auroc", "auprc", "brier_score"
    ]].to_string(index=False))
    print(deltas.loc[deltas["primary_model_for_risk_set"].astype(bool), [
        "risk_set_hours", "new_information_set", "reference_information_set", "model",
        "delta_auroc", "delta_auroc_ci_lower", "delta_auroc_ci_upper"
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
