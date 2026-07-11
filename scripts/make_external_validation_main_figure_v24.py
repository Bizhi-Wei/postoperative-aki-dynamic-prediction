"""Build the main-text eICU external-validation figure.

Figure contract
---------------
Core conclusion: eICU external discrimination is moderate at all three
landmarks, while hospital-held-out logistic recalibration improves absolute
risk calibration.

Panels a-c show full-cohort eICU ROC curves. Panels d-f show calibration in
held-out hospitals before and after a logistic recalibration update learned in
separate calibration hospitals.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve


ROOT = Path(__file__).resolve().parents[1]
V16 = ROOT / "outputs" / "modeling_v16_eicu_external_validation"
V17 = ROOT / "outputs" / "modeling_v17_eicu_recalibration_heterogeneity"
OUT = ROOT / "outputs" / "manuscript_figure_v24_external_validation"
OUTCOME = "outcome_aki_after_landmark_to_7d"

BLUE = "#3D6FA6"
ORANGE = "#D57A2A"
GRID = "#D9DEE7"
TEXT = "#202633"
MUTED = "#697386"


def calibration_bins(data: pd.DataFrame, probability: str) -> pd.DataFrame:
    work = data[[OUTCOME, probability]].dropna().copy()
    work["bin"] = pd.qcut(work[probability], q=min(10, work[probability].nunique()), duplicates="drop")
    return (
        work.groupby("bin", observed=False)
        .agg(mean_predicted=(probability, "mean"), observed=(OUTCOME, "mean"), n=(OUTCOME, "size"))
        .reset_index(drop=True)
    )


def export_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(V16 / "model_v16_portable_external_predictions.csv", low_memory=False)
    performance = pd.read_csv(V16 / "model_v16_portable_external_validation_performance.csv")
    performance = performance.loc[performance["evaluation_dataset"].eq("eICU external validation")].set_index("landmark_hours")
    heldout = pd.read_csv(V17 / "model_v17_heldout_hospital_recalibrated_predictions.csv", low_memory=False)
    recalibration = pd.read_csv(V17 / "model_v17_external_recalibration_performance.csv")

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 8,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.25), dpi=180)
    panel_letters = iter("abcdef")
    roc_source: list[pd.DataFrame] = []
    cal_source: list[pd.DataFrame] = []

    for col, landmark in enumerate((0, 6, 24)):
        ax = axes[0, col]
        data = predictions.loc[predictions["landmark_hours"].eq(landmark)]
        y = data[OUTCOME].astype(int).to_numpy()
        probability = data["predicted_risk_portable_model"].to_numpy()
        fpr, tpr, thresholds = roc_curve(y, probability)
        row = performance.loc[landmark]
        ax.plot(fpr, tpr, color=BLUE, linewidth=1.7)
        ax.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=0.8)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(color=GRID, linewidth=0.5, alpha=0.75)
        ax.set_title(f"{landmark} h | {row.model_family}", pad=5, color=TEXT)
        ax.text(
            0.97, 0.05,
            f"n={int(row.n):,}\nAUROC {row.auroc:.3f}\n95% CI {row.auroc_ci_lower:.3f}-{row.auroc_ci_upper:.3f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.2, color=TEXT,
        )
        ax.set_xlabel("1 - specificity")
        if col == 0:
            ax.set_ylabel("Sensitivity")
        ax.text(-0.18, 1.07, next(panel_letters), transform=ax.transAxes, fontweight="bold", fontsize=8.5)
        roc_source.append(pd.DataFrame({
            "landmark_hours": landmark, "false_positive_rate": fpr,
            "true_positive_rate": tpr, "threshold": thresholds,
        }))

    calibration_cache: dict[tuple[int, str], pd.DataFrame] = {}
    max_value = 0.0
    for landmark in (0, 6, 24):
        data = heldout.loc[heldout["landmark_hours"].eq(landmark)]
        for label, column in (("Frozen", "predicted_risk_frozen"), ("Logistic recalibration", "predicted_risk_logistic_recalibrated")):
            bins = calibration_bins(data, column)
            calibration_cache[(landmark, label)] = bins
            max_value = max(max_value, float(bins[["mean_predicted", "observed"]].to_numpy().max()))
            export = bins.copy()
            export.insert(0, "method", label)
            export.insert(0, "landmark_hours", landmark)
            cal_source.append(export)
    calibration_limit = min(1.0, max(0.4, np.ceil((max_value + 0.02) * 10) / 10))

    for col, landmark in enumerate((0, 6, 24)):
        ax = axes[1, col]
        for label, color in (("Frozen", BLUE), ("Logistic recalibration", ORANGE)):
            bins = calibration_cache[(landmark, label)]
            ax.plot(bins["mean_predicted"], bins["observed"], color=color, marker="o", markersize=3.2, linewidth=1.35, label=label)
        ax.plot([0, calibration_limit], [0, calibration_limit], color=MUTED, linestyle="--", linewidth=0.8)
        ax.set_xlim(0, calibration_limit); ax.set_ylim(0, calibration_limit)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(color=GRID, linewidth=0.5, alpha=0.75)
        ax.set_title(f"{landmark} h | held-out hospitals", pad=5, color=TEXT)
        raw = recalibration.loc[(recalibration["landmark_hours"].eq(landmark)) & recalibration["method"].eq("frozen_unrecalibrated")].iloc[0]
        updated = recalibration.loc[(recalibration["landmark_hours"].eq(landmark)) & recalibration["method"].eq("logistic_recalibration_update")].iloc[0]
        ax.text(
            0.04, 0.96,
            f"Frozen: Brier {raw.brier_score:.3f}; slope {raw.calibration_slope:.2f}\n"
            f"Updated: Brier {updated.brier_score:.3f}; slope {updated.calibration_slope:.2f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=5.8, color=TEXT,
        )
        ax.set_xlabel("Mean predicted risk")
        if col == 0:
            ax.set_ylabel("Observed AKI risk")
        ax.text(-0.18, 1.07, next(panel_letters), transform=ax.transAxes, fontweight="bold", fontsize=8.5)

    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.012), fontsize=7)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.955, bottom=0.155, wspace=0.30, hspace=0.48)

    stem = OUT / "Figure_4_eicu_external_validation"
    export_figure(fig, stem)
    plt.close(fig)
    pd.concat(roc_source, ignore_index=True).to_csv(OUT / "Figure_4_ROC_source_data.csv", index=False)
    pd.concat(cal_source, ignore_index=True).to_csv(OUT / "Figure_4_calibration_source_data.csv", index=False)


if __name__ == "__main__":
    main()
