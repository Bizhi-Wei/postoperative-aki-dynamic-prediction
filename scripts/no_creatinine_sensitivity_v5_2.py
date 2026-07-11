"""No-creatinine sensitivity models with emphasis on the 24 h landmark."""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score, roc_curve

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from develop_models_v5 import METADATA, OUTCOME, RANDOM_STATE, choose_grouped_split, evaluate, identify_types, load_data  # noqa: E402
from extend_models_v5_1 import bootstrap_ci, metric_vector, model_definitions, subject_bootstrap_indices  # noqa: E402


PROJECT_ROOT = SCRIPT_DIR.parent
FULL_RESULT_DIR = PROJECT_ROOT / "outputs" / "modeling_v5_1"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "modeling_v5_2_no_creatinine"
LANDMARKS = [0, 6, 24]
BOOTSTRAPS = 1000

MODEL_SAFE = {
    "Logistic Regression": "logistic_regression",
    "Random Forest": "random_forest",
    "XGBoost": "xgboost",
    "LightGBM": "lightgbm",
}
MODEL_COLORS = {
    "Logistic Regression": "#A3BEFA", "Random Forest": "#F0986E",
    "XGBoost": "#A3D576", "LightGBM": "#F390CA",
}
TOKENS = {"surface": "#FCFCFD", "ink": "#1F2430", "muted": "#6F768A", "grid": "#E6E8F0", "axis": "#D7DBE7"}


def is_creatinine_predictor(column: str) -> bool:
    lowered = column.lower()
    return any(token in lowered for token in ["creatinine", "baseline_scr", "baseline_to_icu"])


def paired_bootstrap_difference(
    y: np.ndarray,
    full_probability: np.ndarray,
    sensitivity_probability: np.ndarray,
    subjects: np.ndarray,
    n_boot: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    distributions = {"auroc": [], "auprc": [], "brier_score": []}
    attempts = 0
    while len(distributions["auroc"]) < n_boot and attempts < n_boot * 3:
        attempts += 1
        idx = subject_bootstrap_indices(subjects, rng)
        if len(np.unique(y[idx])) < 2:
            continue
        full = metric_vector(y[idx], full_probability[idx])
        sensitivity = metric_vector(y[idx], sensitivity_probability[idx])
        for metric in distributions:
            distributions[metric].append(sensitivity[metric] - full[metric])
    result = {"paired_bootstrap_successful_n": len(distributions["auroc"])}
    for metric, values in distributions.items():
        result[f"delta_{metric}_ci_lower"] = float(np.quantile(values, 0.025))
        result[f"delta_{metric}_ci_upper"] = float(np.quantile(values, 0.975))
    return result


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


def plot_24h_curve_comparison(y: np.ndarray, full: dict[str, np.ndarray], sensitivity: dict[str, np.ndarray], kind: str) -> None:
    use_theme()
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.3), dpi=180, sharex=True, sharey=True)
    for ax, model in zip(axes.flat, MODEL_SAFE):
        color = MODEL_COLORS[model]
        if kind == "roc":
            xf, yf, _ = roc_curve(y, full[model]); xs, ys, _ = roc_curve(y, sensitivity[model])
            full_metric = roc_auc_score(y, full[model]); sensitivity_metric = roc_auc_score(y, sensitivity[model])
            ax.plot([0, 1], [0, 1], color=TOKENS["ink"], linestyle=":", linewidth=0.8)
            ax.set_xlabel("1 − Specificity"); ax.set_ylabel("Sensitivity")
            metric_name = "AUROC"
        else:
            yf, xf, _ = precision_recall_curve(y, full[model]); ys, xs, _ = precision_recall_curve(y, sensitivity[model])
            full_metric = average_precision_score(y, full[model]); sensitivity_metric = average_precision_score(y, sensitivity[model])
            ax.axhline(y.mean(), color=TOKENS["ink"], linestyle=":", linewidth=0.8)
            ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
            metric_name = "AUPRC"
        sns.lineplot(x=xf, y=yf, ax=ax, color=color, linewidth=1.35, label=f"Full ({full_metric:.3f})")
        sns.lineplot(x=xs, y=ys, ax=ax, color=color, linestyle="--", linewidth=1.35, label=f"No creatinine ({sensitivity_metric:.3f})")
        ax.set_title(model, fontsize=10, color=TOKENS["ink"])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(frameon=False, fontsize=8, loc="lower right" if kind == "roc" else "upper right")
        sns.despine(ax=ax)
    add_header(
        fig, axes.flat[0],
        f"Full versus no-creatinine 24 h {metric_name} curves",
        f"Paired held-out patients; n={len(y):,}. Solid lines are full models; dashed lines remove all creatinine and baseline-SCr predictors.",
        top=0.86,
    )
    fig.savefig(OUTPUT_DIR / f"figure_v5_2_24h_{kind}_full_vs_no_creatinine.png", bbox_inches="tight", facecolor=TOKENS["surface"])
    plt.close(fig)


def plot_24h_delta(comparison: pd.DataFrame) -> None:
    use_theme()
    plot = comparison.loc[comparison.landmark_hours.eq(24)].sort_values("delta_auroc").copy()
    fig, ax = plt.subplots(figsize=(9, 5.4), dpi=180)
    y_pos = np.arange(len(plot))
    errors = np.vstack([
        plot.delta_auroc - plot.delta_auroc_ci_lower,
        plot.delta_auroc_ci_upper - plot.delta_auroc,
    ])
    ax.errorbar(plot.delta_auroc, y_pos, xerr=errors, fmt="o", color="#F0986E", ecolor="#804126", capsize=4, linewidth=1.1)
    ax.axvline(0, color=TOKENS["ink"], linestyle=":", linewidth=1)
    ax.set_yticks(y_pos, plot.model)
    ax.set_xlabel("ΔAUROC: no-creatinine − full model")
    ax.set_ylabel("")
    add_header(fig, ax, "Removing creatinine predictors reduces 24 h discrimination", "Point estimates and paired patient-level bootstrap 95% CIs; negative values favor the full model.")
    sns.despine(ax=ax)
    fig.savefig(OUTPUT_DIR / "figure_v5_2_24h_auroc_delta.png", bbox_inches="tight", facecolor=TOKENS["surface"])
    plt.close(fig)


def write_readme(performance: pd.DataFrame, comparison: pd.DataFrame, removed: dict[int, list[str]]) -> None:
    c24 = comparison.loc[comparison.landmark_hours.eq(24)].sort_values("model")
    lines = "\n".join(
        f"- {row.model}: full AUROC {row.full_auroc:.3f}, no-creatinine {row.no_creatinine_auroc:.3f}, Δ {row.delta_auroc:.3f} (95% CI {row.delta_auroc_ci_lower:.3f} to {row.delta_auroc_ci_upper:.3f})."
        for row in c24.itertuples()
    )
    content = f"""# No-creatinine sensitivity models v5.2

## Definition

All predictors containing `creatinine`, `baseline_scr`, or `baseline_to_icu` were removed. This excludes measured creatinine values, baseline SCr, baseline availability/source, and baseline timing. CKD, BUN, urine-independent vitals/labs, demographics, and surgery variables were retained. The outcome remains serum-creatinine-defined incident AKI.

The v5.1 patient split, preprocessing, hyperparameters, and four model families were reproduced without test-set tuning.

## Removed predictor counts

- 0 h: {len(removed[0])} — `{', '.join(removed[0])}`
- 6 h: {len(removed[6])} — `{', '.join(removed[6])}`
- 24 h: {len(removed[24])} — `{', '.join(removed[24])}`

## Primary 24 h comparison

{lines}

Overall no-creatinine model CIs use {BOOTSTRAPS:,} patient-level bootstrap resamples. Full-versus-sensitivity differences use paired patient-level bootstrap resampling, so each replicate contains the same held-out patients for both models.

## Interpretation

This analysis tests dependence on creatinine-derived predictors and label proximity. It does not make the outcome independent of creatinine, because the AKI endpoint itself is defined using serum creatinine. Performance reductions at 24 h therefore quantify predictive reliance, not causal importance.
"""
    (OUTPUT_DIR / "audit_v5_2_no_creatinine_readme.md").write_text(content, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    datasets = {landmark: load_data(landmark) for landmark in LANDMARKS}
    train_subjects, test_subjects, _ = choose_grouped_split(datasets[0])

    performance_rows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []
    removed_by_landmark: dict[int, list[str]] = {}
    full_24: dict[str, np.ndarray] = {}
    sensitivity_24: dict[str, np.ndarray] = {}
    y_24: np.ndarray | None = None

    for landmark, data in datasets.items():
        all_predictors = [column for column in data.columns if column not in {*METADATA, OUTCOME}]
        removed = sorted(column for column in all_predictors if is_creatinine_predictor(column))
        removed_by_landmark[landmark] = removed
        retained = [column for column in all_predictors if column not in removed]
        sensitivity_data = data[[*METADATA, *retained, OUTCOME]].copy()
        continuous, binary, categorical = identify_types(sensitivity_data)

        train = sensitivity_data.loc[sensitivity_data.subject_id.astype(int).isin(train_subjects)].copy()
        test = sensitivity_data.loc[sensitivity_data.subject_id.astype(int).isin(test_subjects)].copy()
        x_train, y_train = train[retained], train[OUTCOME].to_numpy(dtype=int)
        x_test, y_test = test[retained], test[OUTCOME].to_numpy(dtype=int)

        full_predictions_file = pd.read_csv(FULL_RESULT_DIR / f"model_v5_1_{landmark}h_test_predictions.csv")
        aligned_full = test[["stay_id"]].merge(full_predictions_file, on="stay_id", how="left", validate="one_to_one")
        if aligned_full.y_true.isna().any() or not np.array_equal(aligned_full.y_true.to_numpy(dtype=int), y_test):
            raise AssertionError("Full-model predictions do not align with sensitivity test rows")

        output = test[["subject_id", "hadm_id", "stay_id", "landmark_hours"]].copy()
        output["y_true"] = y_test
        for model_index, (model_name, pipeline) in enumerate(model_definitions(continuous, binary, categorical).items()):
            print(f"Training no-creatinine {model_name} at {landmark} h...", flush=True)
            pipeline.fit(x_train, y_train)
            train_probability = pipeline.predict_proba(x_train)[:, 1]
            test_probability = pipeline.predict_proba(x_test)[:, 1]
            row, youden = evaluate(
                landmark, model_name, y_train, train_probability, y_test, test_probability,
                len(train), len(test), train.subject_id.nunique(), test.subject_id.nunique(),
            )
            ci = bootstrap_ci(y_test, test_probability, test.subject_id.to_numpy(dtype=int), BOOTSTRAPS, RANDOM_STATE + 5000 + landmark * 100 + model_index)
            performance_rows.append({"sensitivity_analysis": "no_creatinine", "removed_predictor_n": len(removed), **row, **ci})

            safe = MODEL_SAFE[model_name]
            full_probability = aligned_full[f"prob_{safe}"].to_numpy(float)
            full_metrics = metric_vector(y_test, full_probability)
            sensitivity_metrics = metric_vector(y_test, test_probability)
            paired = paired_bootstrap_difference(
                y_test, full_probability, test_probability, test.subject_id.to_numpy(dtype=int),
                BOOTSTRAPS, RANDOM_STATE + 7000 + landmark * 100 + model_index,
            )
            comparison_rows.append({
                "landmark_hours": landmark, "model": model_name,
                "full_auroc": full_metrics["auroc"], "no_creatinine_auroc": sensitivity_metrics["auroc"],
                "delta_auroc": sensitivity_metrics["auroc"] - full_metrics["auroc"],
                "full_auprc": full_metrics["auprc"], "no_creatinine_auprc": sensitivity_metrics["auprc"],
                "delta_auprc": sensitivity_metrics["auprc"] - full_metrics["auprc"],
                "full_brier_score": full_metrics["brier_score"], "no_creatinine_brier_score": sensitivity_metrics["brier_score"],
                "delta_brier_score": sensitivity_metrics["brier_score"] - full_metrics["brier_score"],
                **paired,
            })
            output[f"prob_no_creatinine_{safe}"] = test_probability
            output[f"pred_0_5_no_creatinine_{safe}"] = (test_probability >= 0.5).astype(int)
            output[f"youden_threshold_no_creatinine_{safe}"] = youden
            output[f"pred_youden_no_creatinine_{safe}"] = (test_probability >= youden).astype(int)
            if landmark == 24:
                full_24[model_name] = full_probability
                sensitivity_24[model_name] = test_probability
        output.to_csv(OUTPUT_DIR / f"model_v5_2_no_creatinine_{landmark}h_test_predictions.csv", index=False)
        if landmark == 24:
            y_24 = y_test

    performance = pd.DataFrame(performance_rows)
    comparison = pd.DataFrame(comparison_rows)
    for table in [performance, comparison]:
        numeric = table.select_dtypes(include=[np.number]).columns
        table[numeric] = table[numeric].round(6)
    performance.to_csv(OUTPUT_DIR / "model_v5_2_no_creatinine_performance.csv", index=False)
    comparison.to_csv(OUTPUT_DIR / "model_v5_2_full_vs_no_creatinine_paired_comparison.csv", index=False)
    assert y_24 is not None
    plot_24h_curve_comparison(y_24, full_24, sensitivity_24, "roc")
    plot_24h_curve_comparison(y_24, full_24, sensitivity_24, "pr")
    plot_24h_delta(comparison)
    write_readme(performance, comparison, removed_by_landmark)

    print("\n24 h full versus no-creatinine comparison:")
    print(comparison.loc[comparison.landmark_hours.eq(24), ["model", "full_auroc", "no_creatinine_auroc", "delta_auroc", "delta_auroc_ci_lower", "delta_auroc_ci_upper", "full_auprc", "no_creatinine_auprc"]].to_string(index=False))
    print(f"\nOutputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

