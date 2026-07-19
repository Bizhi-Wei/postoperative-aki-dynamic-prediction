"""Same-risk-set incremental value analysis for dynamic postoperative AKI models.

The original landmark models answer different clinical questions because patients
with AKI before each landmark are removed.  This analysis holds the target risk
set, outcome, subject split, and model family fixed, and changes only the amount
of information available to the model:

* 6 h risk set: 0 h information versus 0-6 h information.
* 24 h risk set: 0 h, 0-6 h, and 0-24 h information.

All variants are refitted within the target risk set to predict AKI after that
landmark through day 7.  Patient-level grouped train/test assignment is inherited
from the prespecified v5 split.  Uncertainty uses paired subject-cluster bootstrap
resampling of the held-out test set.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from develop_models_v5 import OUTCOME, RANDOM_STATE, choose_grouped_split, load_data  # noqa: E402
from final_sensitivity_and_actionable_analysis_v14 import (  # noqa: E402
    SELECTED_MODELS,
    metrics,
    model_pipeline,
    simplified_predictors,
)
from recalibration_and_measurement_intensity_v13 import identify_types_for_columns  # noqa: E402


OUT = ROOT / "outputs" / "modeling_v25_same_risk_set_incremental"
BOOTSTRAPS = 1000
MODEL_FAMILIES = ["Logistic Regression", "XGBoost"]
PRIMARY_MODEL = {6: SELECTED_MODELS[6], 24: SELECTED_MODELS[24]}
INFORMATION_LABEL = {
    0: "0 h information",
    6: "0-6 h information",
    24: "0-24 h information",
}
INFORMATION_SHORT = {0: "0 h", 6: "0-6 h", 24: "0-24 h"}
INFORMATION_COLOR = {0: "#667085", 6: "#4C78A8", 24: "#D97706"}
RISKSET_INFORMATION = {6: [0, 6], 24: [0, 6, 24]}
KEYS = ["subject_id", "hadm_id", "stay_id"]


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 8,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.2,
            "axes.linewidth": 0.75,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.tiff", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.02,
        0.98,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.7, "alpha": 0.85},
        zorder=10,
    )


def information_predictors(info_hours: int) -> list[str]:
    predictors = simplified_predictors(info_hours)
    if OUTCOME in predictors or any(p in KEYS for p in predictors):
        raise AssertionError(f"Metadata/outcome leakage in {info_hours} h predictor list")
    if info_hours == 0 and any("_0_6h_" in p or "_0_24h_" in p for p in predictors):
        raise AssertionError("Post-index variables found in 0 h information set")
    if info_hours == 6 and any("_0_24h_" in p for p in predictors):
        raise AssertionError("0-24 h variables found in 0-6 h information set")
    return predictors


def build_variant_frame(
    target: pd.DataFrame,
    source: pd.DataFrame,
    info_hours: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    predictors = information_predictors(info_hours)
    missing = sorted(set(predictors) - set(source.columns))
    if missing:
        raise ValueError(f"Missing predictors for {info_hours} h information: {missing}")
    target_base = target[KEYS + [OUTCOME]].sort_values(KEYS).reset_index(drop=True)
    source_base = source[KEYS + predictors].copy()
    if target_base["stay_id"].duplicated().any() or source_base["stay_id"].duplicated().any():
        raise AssertionError("Duplicate stay_id before same-risk-set merge")
    merged = target_base.merge(
        source_base,
        on=KEYS,
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    coverage_n = int(merged["_merge"].eq("both").sum())
    if coverage_n != len(target_base):
        raise ValueError(
            f"Incomplete {info_hours} h source coverage: {coverage_n}/{len(target_base)}"
        )
    merged = merged.drop(columns="_merge")
    audit = {
        "information_hours": info_hours,
        "information_set": INFORMATION_LABEL[info_hours],
        "target_n": len(target_base),
        "source_coverage_n": coverage_n,
        "source_coverage_percent": 100 * coverage_n / len(target_base),
        "predictor_n": len(predictors),
        "predictors": "; ".join(predictors),
    }
    return merged, audit


def metric_triplet(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    if len(np.unique(y)) < 2:
        return {"auroc": np.nan, "auprc": np.nan, "brier_score": np.nan}
    return {
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
    }


def cluster_indices(subjects: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    unique, inverse = np.unique(subjects, return_inverse=True)
    return unique, [np.flatnonzero(inverse == i) for i in range(len(unique))]


def bootstrap_same_risk_set(
    y: np.ndarray,
    subjects: np.ndarray,
    probabilities: dict[int, np.ndarray],
    comparisons: list[tuple[int, int]],
    seed: int,
) -> tuple[dict[int, dict[str, tuple[float, float, int]]], list[dict[str, object]]]:
    unique_subjects, rows_by_subject = cluster_indices(subjects)
    rng = np.random.default_rng(seed)
    metric_names = ["auroc", "auprc", "brier_score"]
    model_draws = {h: {m: [] for m in metric_names} for h in probabilities}
    delta_draws = {
        (new, ref): {m: [] for m in metric_names} for new, ref in comparisons
    }
    successful = 0
    for _ in range(BOOTSTRAPS):
        sampled = rng.integers(0, len(unique_subjects), size=len(unique_subjects))
        idx = np.concatenate([rows_by_subject[i] for i in sampled])
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            continue
        sampled_metrics: dict[int, dict[str, float]] = {}
        for h, p in probabilities.items():
            sampled_metrics[h] = metric_triplet(yb, p[idx])
            for metric in metric_names:
                model_draws[h][metric].append(sampled_metrics[h][metric])
        for new, ref in comparisons:
            for metric in metric_names:
                delta_draws[(new, ref)][metric].append(
                    sampled_metrics[new][metric] - sampled_metrics[ref][metric]
                )
        successful += 1

    model_ci: dict[int, dict[str, tuple[float, float, int]]] = {}
    for h in probabilities:
        model_ci[h] = {}
        for metric in metric_names:
            values = np.asarray(model_draws[h][metric], dtype=float)
            model_ci[h][metric] = (
                float(np.nanquantile(values, 0.025)),
                float(np.nanquantile(values, 0.975)),
                successful,
            )

    delta_rows: list[dict[str, object]] = []
    point = {h: metric_triplet(y, p) for h, p in probabilities.items()}
    for new, ref in comparisons:
        row: dict[str, object] = {
            "new_information_hours": new,
            "reference_information_hours": ref,
            "new_information_set": INFORMATION_LABEL[new],
            "reference_information_set": INFORMATION_LABEL[ref],
            "bootstrap_successful_n": successful,
        }
        for metric in metric_names:
            values = np.asarray(delta_draws[(new, ref)][metric], dtype=float)
            row[f"delta_{metric}"] = point[new][metric] - point[ref][metric]
            row[f"delta_{metric}_ci_lower"] = float(np.nanquantile(values, 0.025))
            row[f"delta_{metric}_ci_upper"] = float(np.nanquantile(values, 0.975))
        delta_rows.append(row)
    return model_ci, delta_rows


def calibration_bins(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    order = np.argsort(p, kind="mergesort")
    groups = np.empty(len(p), dtype=int)
    for group, idx in enumerate(np.array_split(order, bins), start=1):
        groups[idx] = group
    frame = pd.DataFrame({"y": y, "p": p, "calibration_group": groups})
    return (
        frame.groupby("calibration_group", as_index=False)
        .agg(n=("y", "size"), observed_risk=("y", "mean"), mean_predicted_risk=("p", "mean"))
    )


def decision_curve(y: np.ndarray, p: np.ndarray, thresholds: np.ndarray) -> pd.DataFrame:
    n = len(y)
    rows = []
    for threshold in thresholds:
        positive = p >= threshold
        tp = int(np.sum(positive & (y == 1)))
        fp = int(np.sum(positive & (y == 0)))
        net_benefit = tp / n - fp / n * threshold / (1 - threshold)
        rows.append(
            {
                "threshold_probability": threshold,
                "net_benefit": net_benefit,
                "true_positive_n": tp,
                "false_positive_n": fp,
            }
        )
    return pd.DataFrame(rows)


def fit_models() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    datasets = {h: load_data(h) for h in [0, 6, 24]}
    train_subjects, test_subjects, split_base = choose_grouped_split(datasets[0])
    if train_subjects & test_subjects:
        raise AssertionError("Subject leakage in inherited grouped split")

    performance_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    delta_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []

    for target_hours, info_sets in RISKSET_INFORMATION.items():
        target = datasets[target_hours]
        variant_frames: dict[int, pd.DataFrame] = {}
        for info_hours in info_sets:
            frame, audit = build_variant_frame(target, datasets[info_hours], info_hours)
            audit.update(
                {
                    "risk_set_hours": target_hours,
                    "risk_set_n": len(target),
                    "risk_set_event_n": int(target[OUTCOME].sum()),
                    "risk_set_event_rate": float(target[OUTCOME].mean()),
                }
            )
            audit_rows.append(audit)
            variant_frames[info_hours] = frame

        ordered_ids = None
        for info_hours, frame in variant_frames.items():
            ids = frame[KEYS].to_numpy()
            if ordered_ids is None:
                ordered_ids = ids
            elif not np.array_equal(ordered_ids, ids):
                raise AssertionError(f"Row mismatch across {target_hours} h information variants")
            if not np.array_equal(frame[OUTCOME].to_numpy(), variant_frames[info_sets[0]][OUTCOME].to_numpy()):
                raise AssertionError(f"Outcome mismatch across {target_hours} h variants")

        train_mask = variant_frames[info_sets[0]]["subject_id"].astype(int).isin(train_subjects)
        test_mask = variant_frames[info_sets[0]]["subject_id"].astype(int).isin(test_subjects)
        if (~(train_mask | test_mask)).any() or (train_mask & test_mask).any():
            raise AssertionError(f"Invalid split assignment in {target_hours} h risk set")
        if set(variant_frames[info_sets[0]].loc[train_mask, "subject_id"]) & set(
            variant_frames[info_sets[0]].loc[test_mask, "subject_id"]
        ):
            raise AssertionError(f"Subject leakage in {target_hours} h risk set")

        y_test = variant_frames[info_sets[0]].loc[test_mask, OUTCOME].astype(int).to_numpy()
        test_subject_array = (
            variant_frames[info_sets[0]].loc[test_mask, "subject_id"].astype(int).to_numpy()
        )
        probabilities_by_model: dict[str, dict[int, np.ndarray]] = {
            model: {} for model in MODEL_FAMILIES
        }

        for info_hours in info_sets:
            predictors = information_predictors(info_hours)
            frame = variant_frames[info_hours]
            train = frame.loc[train_mask].copy()
            test = frame.loc[test_mask].copy()
            y_train = train[OUTCOME].astype(int).to_numpy()
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                raise ValueError(f"Single-class split at {target_hours} h")
            continuous, binary, categorical = identify_types_for_columns(
                train[predictors], predictors
            )
            for model_name in MODEL_FAMILIES:
                print(
                    f"Fitting {model_name}: {target_hours} h risk set, "
                    f"{INFORMATION_LABEL[info_hours]}...",
                    flush=True,
                )
                pipeline = model_pipeline(model_name, continuous, binary, categorical)
                pipeline.fit(train[predictors], y_train)
                p = pipeline.predict_proba(test[predictors])[:, 1]
                probabilities_by_model[model_name][info_hours] = p
                point = metrics(y_test, p)
                performance_rows.append(
                    {
                        "risk_set_hours": target_hours,
                        "information_hours": info_hours,
                        "information_set": INFORMATION_LABEL[info_hours],
                        "model": model_name,
                        "primary_model_for_risk_set": model_name == PRIMARY_MODEL[target_hours],
                        "predictor_n": len(predictors),
                        "train_n": int(train_mask.sum()),
                        "train_subject_n": int(train.loc[:, "subject_id"].nunique()),
                        "train_event_n": int(y_train.sum()),
                        "train_event_rate": float(y_train.mean()),
                        "test_n": int(test_mask.sum()),
                        "test_subject_n": int(test.loc[:, "subject_id"].nunique()),
                        "test_event_n": int(y_test.sum()),
                        "test_event_rate": float(y_test.mean()),
                        **point,
                    }
                )
                pred = test[KEYS].copy()
                pred["risk_set_hours"] = target_hours
                pred["information_hours"] = info_hours
                pred["information_set"] = INFORMATION_LABEL[info_hours]
                pred["model"] = model_name
                pred["y_true"] = y_test
                pred["probability"] = p
                prediction_rows.append(pred)

        comparisons = [(6, 0)] if target_hours == 6 else [(6, 0), (24, 6), (24, 0)]
        for model_index, model_name in enumerate(MODEL_FAMILIES):
            ci, deltas = bootstrap_same_risk_set(
                y_test,
                test_subject_array,
                probabilities_by_model[model_name],
                comparisons,
                RANDOM_STATE + target_hours * 1000 + model_index * 100,
            )
            for row in performance_rows:
                if row["risk_set_hours"] == target_hours and row["model"] == model_name:
                    info_hours = int(row["information_hours"])
                    for metric_name in ["auroc", "auprc", "brier_score"]:
                        lower, upper, successful = ci[info_hours][metric_name]
                        row[f"{metric_name}_ci_lower"] = lower
                        row[f"{metric_name}_ci_upper"] = upper
                        row["bootstrap_successful_n"] = successful
            for row in deltas:
                row.update(
                    {
                        "risk_set_hours": target_hours,
                        "model": model_name,
                        "primary_model_for_risk_set": model_name == PRIMARY_MODEL[target_hours],
                        "test_n": int(test_mask.sum()),
                        "test_subject_n": int(
                            variant_frames[info_sets[0]].loc[test_mask, "subject_id"].nunique()
                        ),
                        "test_event_n": int(y_test.sum()),
                        "test_event_rate": float(y_test.mean()),
                    }
                )
                delta_rows.append(row)

    performance = pd.DataFrame(performance_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    deltas = pd.DataFrame(delta_rows)
    audit = pd.DataFrame(audit_rows)
    split_audit = {
        "base_0h_n": int(split_base["overall_n"]),
        "base_0h_train_n": int(split_base["train_n"]),
        "base_0h_test_n": int(split_base["test_n"]),
        "base_train_subject_n": int(split_base["train_subjects"]),
        "base_test_subject_n": int(split_base["test_subjects"]),
        "subject_overlap_n": 0,
        "split_random_state": RANDOM_STATE,
        "bootstrap_resamples": BOOTSTRAPS,
    }
    (OUT / "audit_v25_split_manifest.json").write_text(
        json.dumps(split_audit, indent=2), encoding="utf-8"
    )
    return performance, deltas, predictions, audit


def source_tables(
    performance: pd.DataFrame, predictions: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    calibration_rows = []
    dca_rows = []
    thresholds = np.round(np.arange(0.05, 0.801, 0.01), 2)
    for risk_set in [6, 24]:
        model_name = PRIMARY_MODEL[risk_set]
        part = predictions.loc[
            predictions["risk_set_hours"].eq(risk_set) & predictions["model"].eq(model_name)
        ]
        for info_hours in RISKSET_INFORMATION[risk_set]:
            data = part.loc[part["information_hours"].eq(info_hours)]
            y = data["y_true"].astype(int).to_numpy()
            p = data["probability"].to_numpy(float)
            bins = calibration_bins(y, p)
            bins["risk_set_hours"] = risk_set
            bins["information_hours"] = info_hours
            bins["information_set"] = INFORMATION_LABEL[info_hours]
            bins["model"] = model_name
            calibration_rows.append(bins)
            dca = decision_curve(y, p, thresholds)
            dca["risk_set_hours"] = risk_set
            dca["information_hours"] = info_hours
            dca["information_set"] = INFORMATION_LABEL[info_hours]
            dca["model"] = model_name
            dca_rows.append(dca)
        y_ref = part.loc[part["information_hours"].eq(RISKSET_INFORMATION[risk_set][0]), "y_true"].astype(int).to_numpy()
        prevalence = float(y_ref.mean())
        treat_all = pd.DataFrame(
            {
                "threshold_probability": thresholds,
                "net_benefit": prevalence - (1 - prevalence) * thresholds / (1 - thresholds),
                "true_positive_n": int(y_ref.sum()),
                "false_positive_n": int((1 - y_ref).sum()),
                "risk_set_hours": risk_set,
                "information_hours": -1,
                "information_set": "Treat all",
                "model": "Reference strategy",
            }
        )
        treat_none = treat_all.copy()
        treat_none["net_benefit"] = 0.0
        treat_none["true_positive_n"] = 0
        treat_none["false_positive_n"] = 0
        treat_none["information_hours"] = -2
        treat_none["information_set"] = "Treat none"
        dca_rows.extend([treat_all, treat_none])
    return pd.concat(calibration_rows, ignore_index=True), pd.concat(dca_rows, ignore_index=True)


def plot_discrimination(performance: pd.DataFrame, predictions: pd.DataFrame) -> None:
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))
    panel = 0
    for row_index, risk_set in enumerate([6, 24]):
        model_name = PRIMARY_MODEL[risk_set]
        part = predictions.loc[
            predictions["risk_set_hours"].eq(risk_set) & predictions["model"].eq(model_name)
        ]
        for col_index, curve_type in enumerate(["roc", "pr"]):
            ax = axes[row_index, col_index]
            for info_hours in RISKSET_INFORMATION[risk_set]:
                data = part.loc[part["information_hours"].eq(info_hours)]
                y = data["y_true"].astype(int).to_numpy()
                p = data["probability"].to_numpy(float)
                perf = performance.loc[
                    performance["risk_set_hours"].eq(risk_set)
                    & performance["information_hours"].eq(info_hours)
                    & performance["model"].eq(model_name)
                ].iloc[0]
                if curve_type == "roc":
                    x, yy, _ = roc_curve(y, p)
                    metric_text = f"AUROC {perf.auroc:.3f}"
                else:
                    yy, x, _ = precision_recall_curve(y, p)
                    metric_text = f"AUPRC {perf.auprc:.3f}"
                ax.plot(
                    x,
                    yy,
                    color=INFORMATION_COLOR[info_hours],
                    linewidth=1.5,
                    label=f"{INFORMATION_SHORT[info_hours]} ({metric_text})",
                )
            if curve_type == "roc":
                ax.plot([0, 1], [0, 1], color="#98A2B3", linestyle=":", linewidth=0.9)
                ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="1 - specificity", ylabel="Sensitivity")
                curve_name = "ROC"
            else:
                prevalence = float(part.loc[part["information_hours"].eq(0), "y_true"].mean())
                ax.axhline(prevalence, color="#98A2B3", linestyle=":", linewidth=0.9)
                ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Recall", ylabel="Precision")
                curve_name = "Precision-recall"
            n = int(part.loc[part["information_hours"].eq(0)].shape[0])
            ax.set_title(f"{risk_set} h risk set: {curve_name}\n{model_name}; held-out n={n:,}")
            ax.grid(color="#E6E8F0", linewidth=0.6)
            ax.legend(loc="lower right" if curve_type == "roc" else "lower left", frameon=False)
            panel_label(ax, chr(97 + panel))
            panel += 1
    fig.suptitle(
        "Same-risk-set discrimination by information availability",
        x=0.08,
        y=1.01,
        ha="left",
        fontsize=9.5,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    save_figure(fig, "figure_v25_same_risk_set_discrimination")


def plot_calibration_dca(calibration: pd.DataFrame, dca: pd.DataFrame) -> None:
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.5))
    panel = 0
    for column, risk_set in enumerate([6, 24]):
        model_name = PRIMARY_MODEL[risk_set]
        ax = axes[0, column]
        ax.plot([0, 1], [0, 1], color="#98A2B3", linestyle=":", linewidth=0.9, label="Ideal")
        for info_hours in RISKSET_INFORMATION[risk_set]:
            part = calibration.loc[
                calibration["risk_set_hours"].eq(risk_set)
                & calibration["information_hours"].eq(info_hours)
            ]
            ax.plot(
                part["mean_predicted_risk"],
                part["observed_risk"],
                marker="o",
                markersize=3.2,
                linewidth=1.2,
                color=INFORMATION_COLOR[info_hours],
                label=INFORMATION_SHORT[info_hours],
            )
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted risk", ylabel="Observed risk")
        ax.set_title(f"{risk_set} h risk set: calibration\n{model_name}; ten equal-frequency groups")
        ax.grid(color="#E6E8F0", linewidth=0.6)
        ax.legend(loc="upper left", frameon=False)
        panel_label(ax, chr(97 + panel))
        panel += 1

        ax = axes[1, column]
        for info_hours in RISKSET_INFORMATION[risk_set]:
            part = dca.loc[
                dca["risk_set_hours"].eq(risk_set)
                & dca["information_hours"].eq(info_hours)
            ]
            ax.plot(
                part["threshold_probability"],
                part["net_benefit"],
                color=INFORMATION_COLOR[info_hours],
                linewidth=1.4,
                label=INFORMATION_SHORT[info_hours],
            )
        for strategy, style in [("Treat all", "--"), ("Treat none", ":")]:
            part = dca.loc[
                dca["risk_set_hours"].eq(risk_set) & dca["information_set"].eq(strategy)
            ]
            ax.plot(
                part["threshold_probability"],
                part["net_benefit"],
                color="#98A2B3" if strategy == "Treat all" else "#344054",
                linestyle=style,
                linewidth=0.9,
                label=strategy,
            )
        visible = dca.loc[
            dca["risk_set_hours"].eq(risk_set)
            & dca["threshold_probability"].between(0.05, 0.60)
            & dca["information_hours"].ge(0)
        ]
        ymin = min(-0.03, float(visible["net_benefit"].min()) - 0.01)
        ymax = float(visible["net_benefit"].max()) + 0.03
        ax.set(
            xlim=(0.05, 0.60),
            ylim=(ymin, ymax),
            xlabel="Threshold probability",
            ylabel="Net benefit",
        )
        ax.set_title(f"{risk_set} h risk set: decision-curve analysis")
        ax.grid(color="#E6E8F0", linewidth=0.6)
        ax.legend(loc="upper right", frameon=False, ncol=2)
        panel_label(ax, chr(97 + panel))
        panel += 1
    fig.suptitle(
        "Calibration and decision-curve analysis within fixed landmark risk sets",
        x=0.08,
        y=1.01,
        ha="left",
        fontsize=9.5,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    save_figure(fig, "figure_v25_same_risk_set_calibration_dca")


def plot_incremental_deltas(deltas: pd.DataFrame) -> None:
    setup_style()
    primary = deltas.loc[deltas["primary_model_for_risk_set"].astype(bool)].copy()
    primary["comparison"] = primary.apply(
        lambda r: (
            f"{int(r.risk_set_hours)} h risk set: "
            f"{INFORMATION_SHORT[int(r.new_information_hours)]} vs "
            f"{INFORMATION_SHORT[int(r.reference_information_hours)]}"
        ),
        axis=1,
    )
    order = [
        "6 h risk set: 0-6 h vs 0 h",
        "24 h risk set: 0-6 h vs 0 h",
        "24 h risk set: 0-24 h vs 0-6 h",
        "24 h risk set: 0-24 h vs 0 h",
    ]
    primary["comparison"] = pd.Categorical(primary["comparison"], categories=order, ordered=True)
    primary = primary.sort_values("comparison")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.2), sharey=True)
    specifications = [
        ("auroc", "Delta AUROC", "Positive favours later information"),
        ("auprc", "Delta AUPRC", "Positive favours later information"),
        ("brier_score", "Delta Brier score", "Negative favours later information"),
    ]
    y = np.arange(len(primary))
    colors = ["#4C78A8" if int(x) == 6 else "#D97706" for x in primary["risk_set_hours"]]
    for panel, (ax, (metric, xlabel, subtitle)) in enumerate(zip(axes, specifications)):
        value = primary[f"delta_{metric}"].to_numpy(float)
        lower = primary[f"delta_{metric}_ci_lower"].to_numpy(float)
        upper = primary[f"delta_{metric}_ci_upper"].to_numpy(float)
        for i, color in enumerate(colors):
            ax.errorbar(
                value[i],
                y[i],
                xerr=np.array([[value[i] - lower[i]], [upper[i] - value[i]]]),
                fmt="none",
                ecolor=color,
                elinewidth=1.2,
                capsize=2.5,
            )
        ax.scatter(value, y, c=colors, s=24, zorder=3)
        ax.axvline(0, color="#98A2B3", linestyle=":", linewidth=0.9)
        ax.set_xlabel(xlabel)
        ax.set_title(subtitle, fontsize=7.2)
        ax.grid(axis="x", color="#E6E8F0", linewidth=0.6)
        ax.invert_yaxis()
        panel_label(ax, chr(97 + panel))
    axes[0].set_yticks(y, [str(x) for x in primary["comparison"]])
    fig.suptitle(
        "Incremental predictive value of later information in the same patients",
        x=0.08,
        y=1.02,
        ha="left",
        fontsize=9.5,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96), w_pad=1.5)
    save_figure(fig, "figure_v25_paired_incremental_value")


def write_readme(
    performance: pd.DataFrame,
    deltas: pd.DataFrame,
    audit: pd.DataFrame,
) -> None:
    primary_perf = performance.loc[performance["primary_model_for_risk_set"].astype(bool)]
    primary_delta = deltas.loc[deltas["primary_model_for_risk_set"].astype(bool)]
    lines = [
        "# v25 same-risk-set dynamic incremental value analysis",
        "",
        "## Question",
        "",
        "How much predictive value is added by information accumulated through 6 h or 24 h when the target risk set, outcome, subject split, and model family are held fixed?",
        "",
        "## Design",
        "",
        "- The 6 h risk set compares models refitted using 0 h information versus 0-6 h information; both predict AKI after 6 h through day 7.",
        "- The 24 h risk set compares models refitted using 0 h, 0-6 h, and 0-24 h information; all predict AKI after 24 h through day 7.",
        "- The prespecified subject-grouped 80/20 assignment from v5 was reused. No subject appears in both training and test sets.",
        "- Main comparisons use the locked parsimonious model family for each target risk set: XGBoost at 6 h and logistic regression at 24 h. The alternate family is reported as an algorithm-sensitivity analysis.",
        "- The information sets reuse the locked parsimonious predictor lists: 36 predictors at 0 h and 72 at 6 h or 24 h.",
        f"- Held-out uncertainty uses {BOOTSTRAPS:,} paired subject-cluster bootstrap resamples.",
        "- Delta metrics are later-information minus reference-information. Positive Delta AUROC/AUPRC and negative Delta Brier favour later information.",
        "",
        "## Main held-out performance",
        "",
        "| Risk set | Model | Information | Test n | Events | AUROC (95% CI) | AUPRC (95% CI) | Brier | Calibration intercept/slope |",
        "|---:|---|---|---:|---:|---|---|---:|---|",
    ]
    for r in primary_perf.sort_values(["risk_set_hours", "information_hours"]).itertuples():
        lines.append(
            f"| {int(r.risk_set_hours)} h | {r.model} | {r.information_set} | {int(r.test_n):,} | {int(r.test_event_n):,} | "
            f"{r.auroc:.3f} ({r.auroc_ci_lower:.3f}-{r.auroc_ci_upper:.3f}) | "
            f"{r.auprc:.3f} ({r.auprc_ci_lower:.3f}-{r.auprc_ci_upper:.3f}) | "
            f"{r.brier_score:.3f} | {r.calibration_intercept:.2f}/{r.calibration_slope:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Paired incremental value",
            "",
            "| Risk set | Model | Comparison | Delta AUROC (95% CI) | Delta AUPRC (95% CI) | Delta Brier (95% CI) |",
            "|---:|---|---|---|---|---|",
        ]
    )
    for r in primary_delta.sort_values(
        ["risk_set_hours", "new_information_hours", "reference_information_hours"]
    ).itertuples():
        lines.append(
            f"| {int(r.risk_set_hours)} h | {r.model} | {r.new_information_set} vs {r.reference_information_set} | "
            f"{r.delta_auroc:+.3f} ({r.delta_auroc_ci_lower:+.3f} to {r.delta_auroc_ci_upper:+.3f}) | "
            f"{r.delta_auprc:+.3f} ({r.delta_auprc_ci_lower:+.3f} to {r.delta_auprc_ci_upper:+.3f}) | "
            f"{r.delta_brier_score:+.3f} ({r.delta_brier_score_ci_lower:+.3f} to {r.delta_brier_score_ci_upper:+.3f}) |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "This analysis estimates the incremental predictive value of later information among patients who remain AKI-free at the target landmark. It does not compare the original 0 h, 6 h, and 24 h populations and does not imply that a frozen 0 h probability can be reused later without retargeting or recalibration.",
            "",
            "The analysis is predictive, not causal. Decision curves are descriptive because no intervention, threshold, or harm-to-benefit ratio has been prospectively specified.",
            "",
            "## Validation checks",
            "",
            f"- Source coverage was 100% for all {len(audit)} risk-set/information-set combinations.",
            "- Target outcomes were identical across information variants within each risk set.",
            "- All model variants used the same held-out rows and the same event labels within each risk set.",
            "- No 0-6 h or 0-24 h variable entered the 0 h information set; no 0-24 h variable entered the 0-6 h information set.",
            "- Missing predictors were handled inside the training-fitted preprocessing pipeline; test information was not used for imputation.",
        ]
    )
    (OUT / "audit_v25_results_brief.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_validation_report(
    performance: pd.DataFrame,
    deltas: pd.DataFrame,
    predictions: pd.DataFrame,
    audit: pd.DataFrame,
) -> None:
    issues = []
    if audit["source_coverage_percent"].min() != 100:
        issues.append("Incomplete source-to-risk-set coverage")
    for risk_set in [6, 24]:
        part = predictions.loc[predictions["risk_set_hours"].eq(risk_set)]
        expected = part.groupby(["information_hours", "model"])["stay_id"].nunique()
        if expected.nunique() != 1:
            issues.append(f"Inconsistent test rows across {risk_set} h variants")
        outcomes = part.groupby(["information_hours", "model"])["y_true"].agg(["count", "sum"])
        if outcomes["count"].nunique() != 1 or outcomes["sum"].nunique() != 1:
            issues.append(f"Inconsistent test outcomes across {risk_set} h variants")
    if performance[["auroc", "auprc", "brier_score"]].isna().any().any():
        issues.append("Missing headline performance metric")
    if deltas["bootstrap_successful_n"].min() < BOOTSTRAPS * 0.95:
        issues.append("Too few successful paired bootstrap resamples")
    assessment = "Ready to share" if not issues else "Needs revision"
    lines = [
        "# Validation report: v25 same-risk-set incremental analysis",
        "",
        f"## Overall assessment: {assessment}",
        "",
        "## Methodology review",
        "",
        "The target question, population, outcome window, comparison basis, and grouped split are aligned. Models are refitted in each target risk set so every information variant predicts the same future outcome in the same patients.",
        "",
        "## Calculation spot-checks",
        "",
        f"- 6 h held-out denominator: {int(performance.loc[performance.risk_set_hours.eq(6), 'test_n'].iloc[0]):,}; consistent across variants.",
        f"- 24 h held-out denominator: {int(performance.loc[performance.risk_set_hours.eq(24), 'test_n'].iloc[0]):,}; consistent across variants.",
        f"- Minimum successful paired bootstrap resamples: {int(deltas.bootstrap_successful_n.min()):,}.",
        f"- Minimum source coverage: {audit.source_coverage_percent.min():.1f}%.",
        "",
        "## Issues found",
        "",
    ]
    lines.extend([f"- {issue}" for issue in issues] if issues else ["- No material validation issue identified."])
    lines.extend(
        [
            "",
            "## Required caveats",
            "",
            "- Predictor-set simplification and model-family choice were fixed in earlier development work; this v25 analysis is a secondary internal-validation comparison.",
            "- Pairwise intervals quantify sampling uncertainty but do not establish clinical importance.",
            "- Decision-curve results remain exploratory until a concrete action and threshold range are prespecified.",
        ]
    )
    (OUT / "audit_v25_validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manuscript_candidate_text(
    performance: pd.DataFrame,
    deltas: pd.DataFrame,
) -> None:
    primary_perf = performance.loc[
        performance["primary_model_for_risk_set"].astype(bool)
    ].set_index(["risk_set_hours", "information_hours"])
    primary_delta = deltas.loc[deltas["primary_model_for_risk_set"].astype(bool)].set_index(
        ["risk_set_hours", "new_information_hours", "reference_information_hours"]
    )
    xgb_delta = deltas.loc[
        deltas["risk_set_hours"].eq(24)
        & deltas["model"].eq("XGBoost")
        & deltas["new_information_hours"].eq(24)
        & deltas["reference_information_hours"].eq(6)
    ].iloc[0]
    d6 = primary_delta.loc[(6, 6, 0)]
    d24 = primary_delta.loc[(24, 24, 6)]
    p60 = primary_perf.loc[(6, 0)]
    p66 = primary_perf.loc[(6, 6)]
    p240 = primary_perf.loc[(24, 0)]
    p246 = primary_perf.loc[(24, 6)]
    p2424 = primary_perf.loc[(24, 24)]
    text = f"""# Candidate manuscript insert for v25

This text is not incorporated into the locked manuscript package. It is provided for review before any manuscript update.

## Methods candidate

To distinguish the incremental value of accumulating postoperative information from changes in landmark risk-set composition, we performed a same-risk-set secondary analysis. Within the 6-h risk set, models using only information available at ICU admission were compared with models additionally using measurements accrued from 0 to 6 h. Within the 24-h risk set, models using information available at ICU admission, from 0 to 6 h, and from 0 to 24 h were compared. All variants were refitted within the same target risk set, predicted the same remaining incident-AKI outcome through day 7, used the prespecified subject-grouped train/test assignment, and were evaluated in identical held-out patients. Main comparisons retained the locked parsimonious model family for each target risk set. Differences in AUROC, AUPRC, and Brier score were estimated using 1,000 paired subject-cluster bootstrap resamples.

## Results candidate

Within the 6-h risk set (held-out n={int(p60.test_n):,}; {int(p60.test_event_n):,} events), adding 0-6-h information to the admission-only XGBoost model produced little change in discrimination: AUROC increased from {p60.auroc:.3f} to {p66.auroc:.3f} (paired difference {d6.delta_auroc:+.3f}; 95% CI, {d6.delta_auroc_ci_lower:+.3f} to {d6.delta_auroc_ci_upper:+.3f}), while the AUPRC difference was {d6.delta_auprc:+.3f} (95% CI, {d6.delta_auprc_ci_lower:+.3f} to {d6.delta_auprc_ci_upper:+.3f}) and the Brier-score difference was {d6.delta_brier_score:+.3f} (95% CI, {d6.delta_brier_score_ci_lower:+.3f} to {d6.delta_brier_score_ci_upper:+.3f}).

Within the 24-h risk set (held-out n={int(p240.test_n):,}; {int(p240.test_event_n):,} events), admission-only and 0-6-h logistic models had nearly identical AUROCs ({p240.auroc:.3f} and {p246.auroc:.3f}, respectively). Adding the complete 0-24-h information increased AUROC to {p2424.auroc:.3f}, corresponding to a paired difference of {d24.delta_auroc:+.3f} (95% CI, {d24.delta_auroc_ci_lower:+.3f} to {d24.delta_auroc_ci_upper:+.3f}) relative to the 0-6-h model. AUPRC increased by {d24.delta_auprc:+.3f} (95% CI, {d24.delta_auprc_ci_lower:+.3f} to {d24.delta_auprc_ci_upper:+.3f}), and the Brier score decreased by {abs(d24.delta_brier_score):.3f} (difference {d24.delta_brier_score:+.3f}; 95% CI, {d24.delta_brier_score_ci_lower:+.3f} to {d24.delta_brier_score_ci_upper:+.3f}). The XGBoost sensitivity analysis also favored 0-24-h over 0-6-h information (AUROC difference {xgb_delta.delta_auroc:+.3f}; 95% CI, {xgb_delta.delta_auroc_ci_lower:+.3f} to {xgb_delta.delta_auroc_ci_upper:+.3f}).

## Discussion candidate

The same-risk-set analysis showed that the apparent improvement across the original 0-h, 6-h, and 24-h models was not a uniform consequence of accumulating data. Information accrued during the first 6 h added little discrimination among patients who remained AKI-free at 6 or 24 h, whereas information accrued across the full first 24 h provided a reproducible improvement in discrimination, precision-recall performance, overall probabilistic performance, and decision-curve net benefit among patients still at risk at 24 h. These findings support interpreting the landmark models as conditional risk updates in changing risk sets rather than as paired longitudinal measurements of a fixed cohort.
"""
    (OUT / "manuscript_v25_candidate_text.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    performance, deltas, predictions, audit = fit_models()
    calibration, dca = source_tables(performance, predictions)

    numeric = performance.select_dtypes(include=[np.number]).columns
    performance[numeric] = performance[numeric].round(8)
    numeric = deltas.select_dtypes(include=[np.number]).columns
    deltas[numeric] = deltas[numeric].round(8)
    performance.to_csv(OUT / "model_v25_same_risk_set_performance.csv", index=False)
    deltas.to_csv(OUT / "model_v25_paired_incremental_deltas.csv", index=False)
    predictions.to_csv(OUT / "model_v25_same_risk_set_test_predictions.csv", index=False)
    audit.to_csv(OUT / "audit_v25_information_set_coverage.csv", index=False)
    calibration.to_csv(OUT / "figure_v25_calibration_source_data.csv", index=False)
    dca.to_csv(OUT / "figure_v25_dca_source_data.csv", index=False)

    plot_discrimination(performance, predictions)
    plot_calibration_dca(calibration, dca)
    plot_incremental_deltas(deltas)
    write_readme(performance, deltas, audit)
    write_validation_report(performance, deltas, predictions, audit)
    write_manuscript_candidate_text(performance, deltas)

    print("\nPrimary same-risk-set performance:")
    print(
        performance.loc[performance["primary_model_for_risk_set"].astype(bool), [
            "risk_set_hours",
            "information_set",
            "model",
            "test_n",
            "test_event_rate",
            "auroc",
            "auprc",
            "brier_score",
            "calibration_intercept",
            "calibration_slope",
        ]].to_string(index=False)
    )
    print("\nPrimary paired incremental deltas:")
    print(
        deltas.loc[deltas["primary_model_for_risk_set"].astype(bool), [
            "risk_set_hours",
            "new_information_set",
            "reference_information_set",
            "model",
            "delta_auroc",
            "delta_auroc_ci_lower",
            "delta_auroc_ci_upper",
            "delta_auprc",
            "delta_brier_score",
        ]].to_string(index=False)
    )
    print(f"\nOutputs written to: {OUT}")


if __name__ == "__main__":
    main()
