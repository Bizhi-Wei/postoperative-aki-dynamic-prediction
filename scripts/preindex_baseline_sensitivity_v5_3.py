"""Models restricted to patients with a pre-ICU 7-day creatinine baseline."""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from develop_models_v5 import METADATA, OUTCOME, RANDOM_STATE, choose_grouped_split, evaluate, identify_types, load_data  # noqa: E402
from extend_models_v5_1 import bootstrap_ci, metric_vector, model_definitions  # noqa: E402
from no_creatinine_sensitivity_v5_2 import paired_bootstrap_difference  # noqa: E402


PROJECT_ROOT = SCRIPT_DIR.parent
FULL_RESULT_DIR = PROJECT_ROOT / "outputs" / "modeling_v5_1"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "modeling_v5_3_preindex_baseline"
LANDMARKS = [0, 6, 24]
BOOTSTRAPS = 1000
BASELINE_SOURCE = "lowest_scr_7d_pre_icu"

MODEL_SAFE = {
    "Logistic Regression": "logistic_regression", "Random Forest": "random_forest",
    "XGBoost": "xgboost", "LightGBM": "lightgbm",
}
MODEL_COLORS = {
    "Logistic Regression": "#A3BEFA", "Random Forest": "#F0986E",
    "XGBoost": "#A3D576", "LightGBM": "#F390CA",
}
TOKENS = {"surface": "#FCFCFD", "ink": "#1F2430", "muted": "#6F768A", "grid": "#E6E8F0", "axis": "#D7DBE7"}


def use_theme() -> None:
    sns.set_theme(style="whitegrid", rc={
        "figure.facecolor": TOKENS["surface"], "axes.facecolor": "#FFFFFF",
        "axes.edgecolor": TOKENS["axis"], "grid.color": TOKENS["grid"],
        "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "axes.spines.top": False, "axes.spines.right": False,
    })


def add_header(fig, ax, title: str, subtitle: str, top: float = 0.82) -> None:
    fig.subplots_adjust(top=top, left=0.10, right=0.97, bottom=0.11, hspace=0.30, wspace=0.24)
    left = ax.get_position().x0
    fig.text(left, 0.98, textwrap.fill(title, 84), ha="left", va="top", fontsize=14, fontweight="semibold", color=TOKENS["ink"])
    fig.text(left, 0.94, textwrap.fill(subtitle, 118), ha="left", va="top", fontsize=9, color=TOKENS["muted"])


def plot_24h_roc(y: np.ndarray, full: dict[str, np.ndarray], restricted: dict[str, np.ndarray]) -> None:
    use_theme()
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.3), dpi=180, sharex=True, sharey=True)
    for ax, model in zip(axes.flat, MODEL_SAFE):
        color = MODEL_COLORS[model]
        xf, yf, _ = roc_curve(y, full[model]); xr, yr, _ = roc_curve(y, restricted[model])
        full_auc = roc_auc_score(y, full[model]); restricted_auc = roc_auc_score(y, restricted[model])
        sns.lineplot(x=xf, y=yf, ax=ax, color=color, linewidth=1.35, label=f"Full-cohort model ({full_auc:.3f})")
        sns.lineplot(x=xr, y=yr, ax=ax, color=color, linestyle="--", linewidth=1.35, label=f"Pre-index-only model ({restricted_auc:.3f})")
        ax.plot([0, 1], [0, 1], color=TOKENS["ink"], linestyle=":", linewidth=0.8)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="1 − Specificity", ylabel="Sensitivity")
        ax.set_title(model, fontsize=10); ax.legend(frameon=False, fontsize=7.8, loc="lower right")
        sns.despine(ax=ax)
    add_header(fig, axes.flat[0], "Full-cohort versus pre-index-baseline-only 24 h ROC curves", f"Same restricted held-out patients; n={len(y):,}. Dashed models were retrained only in patients with a 7-day pre-ICU baseline.", top=0.86)
    fig.savefig(OUTPUT_DIR / "figure_v5_3_24h_roc_full_vs_preindex.png", bbox_inches="tight", facecolor=TOKENS["surface"])
    plt.close(fig)


def plot_24h_delta(comparison: pd.DataFrame) -> None:
    use_theme()
    plot = comparison.loc[comparison.landmark_hours.eq(24)].sort_values("delta_auroc")
    fig, ax = plt.subplots(figsize=(9, 5.4), dpi=180)
    positions = np.arange(len(plot))
    error = np.vstack([plot.delta_auroc - plot.delta_auroc_ci_lower, plot.delta_auroc_ci_upper - plot.delta_auroc])
    ax.errorbar(plot.delta_auroc, positions, xerr=error, fmt="o", color="#A3D576", ecolor="#386411", capsize=4, linewidth=1.1)
    ax.axvline(0, color=TOKENS["ink"], linestyle=":", linewidth=1)
    ax.set_yticks(positions, plot.model)
    ax.set(xlabel="ΔAUROC: pre-index-only retrained − full-cohort model", ylabel="")
    add_header(fig, ax, "Restricting baseline definition has limited 24 h discrimination impact", "Paired patient-level bootstrap 95% CIs on the same restricted test patients; positive values favor retraining.")
    sns.despine(ax=ax)
    fig.savefig(OUTPUT_DIR / "figure_v5_3_24h_auroc_delta.png", bbox_inches="tight", facecolor=TOKENS["surface"])
    plt.close(fig)


def write_readme(performance: pd.DataFrame, comparison: pd.DataFrame, sample_audit: list[dict[str, object]], constants: dict[int, list[str]]) -> None:
    audit = pd.DataFrame(sample_audit).set_index("landmark_hours")
    c24 = comparison.loc[comparison.landmark_hours.eq(24)].sort_values("model")
    lines = "\n".join(
        f"- {row.model}: full model {row.full_model_auroc:.3f}, restricted retrained {row.preindex_model_auroc:.3f}, Δ {row.delta_auroc:+.3f} (95% CI {row.delta_auroc_ci_lower:+.3f} to {row.delta_auroc_ci_upper:+.3f})."
        for row in c24.itertuples()
    )
    content = f"""# Pre-index baseline-only sensitivity models v5.3

## Cohort

Rows were restricted at each landmark to `baseline_scr_source_at_landmark == {BASELINE_SOURCE}`. The original v5 patient assignment was retained.

| Landmark | Total restricted n | Train n | Test n | Events |
|---:|---:|---:|---:|---:|
| 0 h | {int(audit.loc[0,'restricted_n']):,} | {int(audit.loc[0,'train_n']):,} | {int(audit.loc[0,'test_n']):,} | {int(audit.loc[0,'event_n']):,} |
| 6 h | {int(audit.loc[6,'restricted_n']):,} | {int(audit.loc[6,'train_n']):,} | {int(audit.loc[6,'test_n']):,} | {int(audit.loc[6,'event_n']):,} |
| 24 h | {int(audit.loc[24,'restricted_n']):,} | {int(audit.loc[24,'train_n']):,} | {int(audit.loc[24,'test_n']):,} | {int(audit.loc[24,'event_n']):,} |

Constant predictors were removed before fitting: {constants}.

## Primary 24 h paired comparison

{lines}

The comparison uses identical restricted test patients. “Full model” means the original v5.1 model trained in the full evaluable cohort, evaluated only in restricted patients. “Restricted retrained” means the same model family and hyperparameters retrained only in pre-index-baseline patients.

Overall performance and paired differences use {BOOTSTRAPS:,} patient-level bootstrap resamples. This is a baseline-definition sensitivity analysis, not external validation.
"""
    (OUTPUT_DIR / "audit_v5_3_preindex_baseline_readme.md").write_text(content, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    datasets = {landmark: load_data(landmark) for landmark in LANDMARKS}
    train_subjects, test_subjects, _ = choose_grouped_split(datasets[0])
    performance_rows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []
    sample_audit: list[dict[str, object]] = []
    constants: dict[int, list[str]] = {}
    full_24: dict[str, np.ndarray] = {}; restricted_24: dict[str, np.ndarray] = {}; y_24: np.ndarray | None = None

    for landmark, data in datasets.items():
        restricted = data.loc[data.baseline_scr_source_at_landmark.eq(BASELINE_SOURCE)].copy()
        train = restricted.loc[restricted.subject_id.astype(int).isin(train_subjects)].copy()
        test = restricted.loc[restricted.subject_id.astype(int).isin(test_subjects)].copy()
        all_predictors = [column for column in restricted.columns if column not in {*METADATA, OUTCOME}]
        constant = [column for column in all_predictors if train[column].nunique(dropna=False) <= 1]
        constants[landmark] = constant
        predictors = [column for column in all_predictors if column not in constant]
        model_data = restricted[[*METADATA, *predictors, OUTCOME]]
        continuous, binary, categorical = identify_types(model_data)
        x_train, y_train = train[predictors], train[OUTCOME].to_numpy(dtype=int)
        x_test, y_test = test[predictors], test[OUTCOME].to_numpy(dtype=int)
        sample_audit.append({"landmark_hours": landmark, "restricted_n": len(restricted), "train_n": len(train), "test_n": len(test), "event_n": int(restricted[OUTCOME].sum()), "test_event_n": int(y_test.sum())})

        full_file = pd.read_csv(FULL_RESULT_DIR / f"model_v5_1_{landmark}h_test_predictions.csv")
        aligned = test[["stay_id"]].merge(full_file, on="stay_id", how="left", validate="one_to_one")
        if aligned.y_true.isna().any() or not np.array_equal(aligned.y_true.to_numpy(int), y_test):
            raise AssertionError("Full-model predictions do not align")
        output = test[["subject_id", "hadm_id", "stay_id", "landmark_hours"]].copy(); output["y_true"] = y_test

        for model_index, (model_name, pipeline) in enumerate(model_definitions(continuous, binary, categorical).items()):
            print(f"Training pre-index-only {model_name} at {landmark} h...", flush=True)
            pipeline.fit(x_train, y_train)
            train_probability = pipeline.predict_proba(x_train)[:, 1]
            test_probability = pipeline.predict_proba(x_test)[:, 1]
            row, youden = evaluate(landmark, model_name, y_train, train_probability, y_test, test_probability, len(train), len(test), train.subject_id.nunique(), test.subject_id.nunique())
            ci = bootstrap_ci(y_test, test_probability, test.subject_id.to_numpy(int), BOOTSTRAPS, RANDOM_STATE + 9000 + landmark * 100 + model_index)
            performance_rows.append({"sensitivity_analysis": "preindex_baseline_only", "restricted_total_n": len(restricted), "constant_predictor_n": len(constant), **row, **ci})

            safe = MODEL_SAFE[model_name]
            full_probability = aligned[f"prob_{safe}"].to_numpy(float)
            full_metrics = metric_vector(y_test, full_probability); restricted_metrics = metric_vector(y_test, test_probability)
            paired = paired_bootstrap_difference(y_test, full_probability, test_probability, test.subject_id.to_numpy(int), BOOTSTRAPS, RANDOM_STATE + 11000 + landmark * 100 + model_index)
            comparison_rows.append({
                "landmark_hours": landmark, "model": model_name, "restricted_test_n": len(test),
                "full_model_auroc": full_metrics["auroc"], "preindex_model_auroc": restricted_metrics["auroc"], "delta_auroc": restricted_metrics["auroc"] - full_metrics["auroc"],
                "full_model_auprc": full_metrics["auprc"], "preindex_model_auprc": restricted_metrics["auprc"], "delta_auprc": restricted_metrics["auprc"] - full_metrics["auprc"],
                "full_model_brier": full_metrics["brier_score"], "preindex_model_brier": restricted_metrics["brier_score"], "delta_brier_score": restricted_metrics["brier_score"] - full_metrics["brier_score"],
                **paired,
            })
            output[f"prob_preindex_{safe}"] = test_probability
            output[f"pred_0_5_preindex_{safe}"] = (test_probability >= 0.5).astype(int)
            output[f"youden_threshold_preindex_{safe}"] = youden
            output[f"pred_youden_preindex_{safe}"] = (test_probability >= youden).astype(int)
            if landmark == 24:
                full_24[model_name] = full_probability; restricted_24[model_name] = test_probability
        output.to_csv(OUTPUT_DIR / f"model_v5_3_preindex_{landmark}h_test_predictions.csv", index=False)
        if landmark == 24: y_24 = y_test

    performance = pd.DataFrame(performance_rows); comparison = pd.DataFrame(comparison_rows)
    for table in [performance, comparison]:
        numeric = table.select_dtypes(include=[np.number]).columns; table[numeric] = table[numeric].round(6)
    performance.to_csv(OUTPUT_DIR / "model_v5_3_preindex_performance.csv", index=False)
    comparison.to_csv(OUTPUT_DIR / "model_v5_3_full_vs_preindex_paired_comparison.csv", index=False)
    pd.DataFrame(sample_audit).to_csv(OUTPUT_DIR / "audit_v5_3_preindex_sample_summary.csv", index=False)
    assert y_24 is not None
    plot_24h_roc(y_24, full_24, restricted_24); plot_24h_delta(comparison)
    write_readme(performance, comparison, sample_audit, constants)
    print("\n24 h paired comparison:")
    print(comparison.loc[comparison.landmark_hours.eq(24), ["model", "full_model_auroc", "preindex_model_auroc", "delta_auroc", "delta_auroc_ci_lower", "delta_auroc_ci_upper", "full_model_auprc", "preindex_model_auprc"]].to_string(index=False))
    print(f"\nOutputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

