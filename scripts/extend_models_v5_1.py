"""Extend v5 with boosting, SHAP, DCA, subgroup analysis, and bootstrap CIs."""

from __future__ import annotations

import sys
import textwrap
import zlib
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import xgboost as xgb
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from develop_models_v5 import (  # noqa: E402
    METADATA,
    OUTCOME,
    RANDOM_STATE,
    calibration_intercept_slope,
    choose_grouped_split,
    evaluate,
    identify_types,
    load_data,
    make_preprocessor,
)


PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "modeling_v5_1"
LANDMARKS = [0, 6, 24]
OVERALL_BOOTSTRAPS = 1000
SUBGROUP_BOOTSTRAPS = 300

MODEL_STYLES = {
    "Logistic Regression": ("#A3BEFA", "-"),
    "Random Forest": ("#F0986E", "--"),
    "XGBoost": ("#A3D576", "-."),
    "LightGBM": ("#F390CA", ":"),
}
TOKENS = {
    "surface": "#FCFCFD", "panel": "#FFFFFF", "ink": "#1F2430",
    "muted": "#6F768A", "grid": "#E6E8F0", "axis": "#D7DBE7",
}


def model_definitions(continuous: list[str], binary: list[str], categorical: list[str]) -> dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline(
            [
                ("preprocess", make_preprocessor(continuous, binary, categorical, scale=True)),
                ("model", LogisticRegression(max_iter=3000, solver="lbfgs", random_state=RANDOM_STATE)),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("preprocess", make_preprocessor(continuous, binary, categorical, scale=False)),
                ("model", RandomForestClassifier(n_estimators=500, min_samples_leaf=5, max_features="sqrt", n_jobs=-1, random_state=RANDOM_STATE)),
            ]
        ),
        "XGBoost": Pipeline(
            [
                ("preprocess", make_preprocessor(continuous, binary, categorical, scale=False)),
                ("model", xgb.XGBClassifier(
                    n_estimators=500, learning_rate=0.03, max_depth=4,
                    min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                    reg_lambda=1.0, objective="binary:logistic", eval_metric="logloss",
                    n_jobs=-1, random_state=RANDOM_STATE,
                )),
            ]
        ),
        "LightGBM": Pipeline(
            [
                ("preprocess", make_preprocessor(continuous, binary, categorical, scale=False)),
                ("model", lgb.LGBMClassifier(
                    n_estimators=500, learning_rate=0.03, num_leaves=31,
                    min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                    reg_lambda=1.0, n_jobs=-1, random_state=RANDOM_STATE, verbosity=-1,
                )),
            ]
        ),
    }


def subject_bootstrap_indices(subjects: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    unique = np.unique(subjects)
    sampled = rng.choice(unique, size=len(unique), replace=True)
    lookup = {subject: np.flatnonzero(subjects == subject) for subject in unique}
    return np.concatenate([lookup[subject] for subject in sampled])


def metric_vector(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    if len(np.unique(y)) < 2:
        raise ValueError("Both outcome classes are required")
    predicted = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "auroc": roc_auc_score(y, p),
        "auprc": average_precision_score(y, p),
        "brier_score": brier_score_loss(y, p),
        "sensitivity_0_5": tp / (tp + fn) if tp + fn else np.nan,
        "specificity_0_5": tn / (tn + fp) if tn + fp else np.nan,
    }


def bootstrap_ci(
    y: np.ndarray,
    p: np.ndarray,
    subjects: np.ndarray,
    n_boot: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    distributions: dict[str, list[float]] = {
        "auroc": [], "auprc": [], "brier_score": [],
        "sensitivity_0_5": [], "specificity_0_5": [],
    }
    attempts = 0
    while len(distributions["auroc"]) < n_boot and attempts < n_boot * 3:
        attempts += 1
        idx = subject_bootstrap_indices(subjects, rng)
        if len(np.unique(y[idx])) < 2:
            continue
        values = metric_vector(y[idx], p[idx])
        for metric, value in values.items():
            distributions[metric].append(value)
    result: dict[str, float] = {"bootstrap_successful_n": len(distributions["auroc"])}
    for metric, values in distributions.items():
        result[f"{metric}_ci_lower"] = float(np.quantile(values, 0.025))
        result[f"{metric}_ci_upper"] = float(np.quantile(values, 0.975))
    return result


def subgroup_definitions(data: pd.DataFrame) -> list[tuple[str, str, pd.Series]]:
    definitions: list[tuple[str, str, pd.Series]] = []
    for value in sorted(data["gender"].dropna().astype(str).unique()):
        definitions.append(("gender", value, data["gender"].astype(str).eq(value)))
    definitions.extend([
        ("age_group", "<65", pd.to_numeric(data["anchor_age"], errors="coerce").lt(65)),
        ("age_group", ">=65", pd.to_numeric(data["anchor_age"], errors="coerce").ge(65)),
    ])
    binary_groups = [
        "cardiac_surgery", "non_cardiac_surgery", "vascular_surgery",
        "general_gi_hepatobiliary_surgery", "orthopedic_major_surgery",
        "neurosurgery", "thoracic_respiratory_surgery", "ckd",
    ]
    for column in binary_groups:
        values = pd.to_numeric(data[column], errors="coerce")
        definitions.append((column, "yes", values.eq(1)))
        if column in {"cardiac_surgery", "ckd"}:
            definitions.append((column, "no", values.eq(0)))
    for value in data["baseline_scr_source_at_landmark"].fillna("<missing>").value_counts().index:
        definitions.append(("baseline_scr_source", str(value), data["baseline_scr_source_at_landmark"].fillna("<missing>").eq(value)))
    for value, n in data["first_careunit"].value_counts().items():
        if n >= 100:
            definitions.append(("first_careunit", str(value), data["first_careunit"].eq(value)))
    return definitions


def subgroup_performance_rows(
    landmark: int,
    test: pd.DataFrame,
    predictions: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    y_all = test[OUTCOME].to_numpy(dtype=int)
    subjects_all = test["subject_id"].to_numpy(dtype=int)
    for dimension, level, mask in subgroup_definitions(test):
        selected = mask.to_numpy(dtype=bool)
        y = y_all[selected]
        if len(y) < 50 or len(np.unique(y)) < 2:
            continue
        for model_name, all_prob in predictions.items():
            p = all_prob[selected]
            point = metric_vector(y, p)
            seed = RANDOM_STATE + landmark * 10000 + zlib.crc32(f"{model_name}|{dimension}|{level}".encode()) % 10000
            ci = bootstrap_ci(y, p, subjects_all[selected], SUBGROUP_BOOTSTRAPS, seed)
            rows.append({
                "landmark_hours": landmark, "model": model_name,
                "subgroup_dimension": dimension, "subgroup_level": level,
                "n": len(y), "subject_n": len(np.unique(subjects_all[selected])),
                "event_n": int(y.sum()), "event_rate": float(y.mean()),
                **point, **ci,
            })
    return rows


def dca_rows(landmark: int, y: np.ndarray, predictions: dict[str, np.ndarray]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    n = len(y)
    prevalence = float(y.mean())
    for threshold in np.arange(0.05, 0.801, 0.01):
        odds = threshold / (1 - threshold)
        rows.append({"landmark_hours": landmark, "strategy": "Treat none", "threshold": threshold, "net_benefit": 0.0})
        rows.append({"landmark_hours": landmark, "strategy": "Treat all", "threshold": threshold, "net_benefit": prevalence - (1 - prevalence) * odds})
        for model_name, probability in predictions.items():
            predicted = probability >= threshold
            tp = np.sum(predicted & (y == 1))
            fp = np.sum(predicted & (y == 0))
            net_benefit = tp / n - fp / n * odds
            rows.append({"landmark_hours": landmark, "strategy": model_name, "threshold": threshold, "net_benefit": net_benefit})
    return rows


def use_theme() -> None:
    sns.set_theme(style="whitegrid", rc={
        "figure.facecolor": TOKENS["surface"], "axes.facecolor": TOKENS["panel"],
        "axes.edgecolor": TOKENS["axis"], "axes.labelcolor": TOKENS["ink"],
        "grid.color": TOKENS["grid"], "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "axes.spines.top": False, "axes.spines.right": False,
    })


def add_header(fig, ax, title: str, subtitle: str, top: float = 0.80) -> None:
    fig.subplots_adjust(top=top, left=0.11, right=0.97, bottom=0.12)
    left = ax.get_position().x0
    fig.text(left, 0.97, textwrap.fill(title, 80), ha="left", va="top", fontsize=14, fontweight="semibold", color=TOKENS["ink"])
    fig.text(left, 0.91, textwrap.fill(subtitle, 115), ha="left", va="top", fontsize=9, color=TOKENS["muted"])
    sns.despine(ax=ax)


def plot_discrimination(landmark: int, y: np.ndarray, predictions: dict[str, np.ndarray], kind: str) -> None:
    use_theme()
    fig, ax = plt.subplots(figsize=(9, 6.3), dpi=180)
    for model, p in predictions.items():
        color, linestyle = MODEL_STYLES[model]
        if kind == "roc":
            x_axis, y_axis, _ = roc_curve(y, p)
            label = f"{model} ({roc_auc_score(y, p):.3f})"
        else:
            y_axis, x_axis, _ = precision_recall_curve(y, p)
            label = f"{model} ({average_precision_score(y, p):.3f})"
        sns.lineplot(x=x_axis, y=y_axis, ax=ax, color=color, linestyle=linestyle, linewidth=1.35, label=label)
    if kind == "roc":
        ax.plot([0, 1], [0, 1], color=TOKENS["ink"], linestyle=":", linewidth=1, label="Chance")
        ax.set(xlabel="1 − Specificity", ylabel="Sensitivity", xlim=(0, 1), ylim=(0, 1))
        title = f"ROC curves for {landmark} h AKI prediction"
        subtitle = f"Subject-grouped held-out test set; n={len(y):,}. Legend values are AUROC."
        location = "lower right"
    else:
        ax.axhline(y.mean(), color=TOKENS["ink"], linestyle=":", linewidth=1, label=f"Prevalence ({y.mean():.3f})")
        ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1))
        title = f"Precision–recall curves for {landmark} h AKI prediction"
        subtitle = f"Subject-grouped held-out test set; n={len(y):,}. Legend values are AUPRC."
        location = "upper right"
    ax.legend(loc=location, frameon=False, fontsize=8.5)
    add_header(fig, ax, title, subtitle)
    fig.savefig(OUTPUT_DIR / f"figure_v5_1_{kind}_{landmark}h.png", bbox_inches="tight", facecolor=TOKENS["surface"])
    plt.close(fig)


def plot_calibration(landmark: int, y: np.ndarray, predictions: dict[str, np.ndarray]) -> None:
    use_theme()
    fig, ax = plt.subplots(figsize=(9, 6.3), dpi=180)
    for model, p in predictions.items():
        observed, predicted = calibration_curve(y, p, n_bins=10, strategy="quantile")
        color, linestyle = MODEL_STYLES[model]
        sns.lineplot(x=predicted, y=observed, ax=ax, color=color, linestyle=linestyle, marker="o", markersize=4, linewidth=1.2, label=model)
    ax.plot([0, 1], [0, 1], color=TOKENS["ink"], linestyle=":", linewidth=1, label="Ideal")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed event proportion", xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="upper left", frameon=False, fontsize=8.5)
    add_header(fig, ax, f"Calibration curves for {landmark} h AKI prediction", f"Ten equal-frequency bins; subject-grouped held-out test set; n={len(y):,}.")
    fig.savefig(OUTPUT_DIR / f"figure_v5_1_calibration_{landmark}h.png", bbox_inches="tight", facecolor=TOKENS["surface"])
    plt.close(fig)


def plot_dca(landmark: int, dca: pd.DataFrame) -> None:
    use_theme()
    fig, ax = plt.subplots(figsize=(9, 6.3), dpi=180)
    for strategy, part in dca.groupby("strategy", sort=False):
        if strategy == "Treat none":
            color, linestyle, width = TOKENS["muted"], ":", 1.0
        elif strategy == "Treat all":
            color, linestyle, width = TOKENS["ink"], "--", 1.0
        else:
            color, linestyle = MODEL_STYLES[strategy]
            width = 1.3
        sns.lineplot(data=part, x="threshold", y="net_benefit", ax=ax, color=color, linestyle=linestyle, linewidth=width, label=strategy)
    ax.axhline(0, color=TOKENS["axis"], linewidth=0.8)
    ax.set(xlabel="Threshold probability", ylabel="Net benefit", xlim=(0.05, 0.80))
    ax.set_ylim(max(-0.08, dca.net_benefit.quantile(0.01)), dca.net_benefit.quantile(0.99) + 0.03)
    ax.legend(loc="upper right", frameon=False, fontsize=8.2, ncol=2)
    add_header(fig, ax, f"Decision curve analysis for {landmark} h AKI prediction", "Held-out test predictions; net benefit compared with treat-all and treat-none strategies.")
    fig.savefig(OUTPUT_DIR / f"figure_v5_1_dca_{landmark}h.png", bbox_inches="tight", facecolor=TOKENS["surface"])
    plt.close(fig)


def clean_feature_name(name: str) -> str:
    return name.split("__", 1)[-1].replace("missingindicator_", "missing: ")


def shap_analysis(
    landmark: int,
    model_name: str,
    pipeline: Pipeline,
    x_test: pd.DataFrame,
) -> list[dict[str, object]]:
    rng = np.random.default_rng(RANDOM_STATE + landmark)
    if len(x_test) > 1000:
        sample_idx = np.sort(rng.choice(len(x_test), size=1000, replace=False))
        x_sample = x_test.iloc[sample_idx]
    else:
        x_sample = x_test
    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    transformed = preprocessor.transform(x_sample)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    feature_names = [clean_feature_name(name) for name in preprocessor.get_feature_names_out()]
    explanation = shap.TreeExplainer(model)(transformed)
    values = explanation.values
    if values.ndim == 3:
        values = values[:, :, 1]
    mean_abs = np.mean(np.abs(values), axis=0)
    mean_signed = np.mean(values, axis=0)
    order = np.argsort(mean_abs)[::-1]
    rows = [
        {
            "landmark_hours": landmark, "explained_model": model_name,
            "feature": feature_names[index], "mean_abs_shap": mean_abs[index],
            "mean_signed_shap": mean_signed[index], "rank": rank,
            "shap_sample_n": len(x_sample),
        }
        for rank, index in enumerate(order, start=1)
    ]

    top = pd.DataFrame(rows).head(20).sort_values("mean_abs_shap")
    use_theme()
    fig, ax = plt.subplots(figsize=(9.5, 7.2), dpi=180)
    sns.barplot(data=top, x="mean_abs_shap", y="feature", color="#FFE15B", edgecolor="#736422", linewidth=0.8, ax=ax)
    ax.set(xlabel="Mean |SHAP value|", ylabel="")
    add_header(fig, ax, f"Leading SHAP features for {landmark} h AKI prediction", f"{model_name}; held-out sample n={len(x_sample):,}; global magnitude, not causal effect.", top=0.84)
    fig.savefig(OUTPUT_DIR / f"figure_v5_1_shap_bar_{landmark}h.png", bbox_inches="tight", facecolor=TOKENS["surface"])
    plt.close(fig)
    return rows


def write_readme(performance: pd.DataFrame, shap_models: dict[int, str]) -> None:
    best = performance.loc[performance.groupby("landmark_hours")["auroc"].idxmax()]
    best_lines = "\n".join(
        f"- {int(row.landmark_hours)} h: {row.model}, AUROC {row.auroc:.3f} (95% CI {row.auroc_ci_lower:.3f}–{row.auroc_ci_upper:.3f}), AUPRC {row.auprc:.3f}."
        for row in best.itertuples()
    )
    shap_lines = "\n".join(f"- {landmark} h: {model}." for landmark, model in shap_models.items())
    content = f"""# Model extension v5.1: boosting, SHAP, DCA, subgroups, and bootstrap CIs

## Design

- The v5 subject-level 80/20 split was reproduced exactly and reused across all landmarks.
- Logistic regression, random forest, XGBoost 3.3.0, and LightGBM 4.6.0 used fixed development hyperparameters; there was no test-set tuning.
- Overall 95% CIs use {OVERALL_BOOTSTRAPS:,} patient-level bootstrap resamples of the held-out test set.
- Subgroup 95% CIs use {SUBGROUP_BOOTSTRAPS:,} patient-level bootstrap resamples; groups with fewer than 50 rows or one outcome class are omitted.
- Youden thresholds were selected from training predictions and applied to test predictions.

## Best discrimination

{best_lines}

## SHAP

SHAP was calculated on up to 1,000 held-out rows for the highest-AUROC tree model at each landmark:

{shap_lines}

SHAP magnitudes describe model behavior, not causal or modifiable effects. One-hot encoded levels and imputation indicators may appear as separate features.

## Decision curves

DCA reports net benefit from threshold probabilities 0.05–0.80, with treat-all and treat-none references. Curves are exploratory internal-validation evidence and do not establish clinical utility without external validation and a specified intervention pathway.

## Subgroups

Performance is reported by sex, age, selected surgery flags, CKD, baseline source, and sufficiently large ICU types. Overlapping surgery flags are not mutually exclusive. Subgroup comparisons are descriptive and were not multiplicity-adjusted.

## Scope limitations

This remains single-center internal validation. No SHAP-based feature selection, causal interpretation, manuscript drafting, or external validation was performed.
"""
    (OUTPUT_DIR / "audit_v5_1_readme.md").write_text(content, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    datasets = {landmark: load_data(landmark) for landmark in LANDMARKS}
    train_subjects, test_subjects, _ = choose_grouped_split(datasets[0])

    performance_rows: list[dict[str, object]] = []
    subgroup_rows: list[dict[str, object]] = []
    all_dca_rows: list[dict[str, object]] = []
    all_shap_rows: list[dict[str, object]] = []
    shap_models: dict[int, str] = {}

    for landmark, data in datasets.items():
        train = data.loc[data.subject_id.astype(int).isin(train_subjects)].copy()
        test = data.loc[data.subject_id.astype(int).isin(test_subjects)].copy()
        if set(train.subject_id) & set(test.subject_id):
            raise AssertionError("Subject overlap")
        predictors = [column for column in data.columns if column not in {*METADATA, OUTCOME}]
        continuous, binary, categorical = identify_types(data)
        x_train, y_train = train[predictors], train[OUTCOME].to_numpy(dtype=int)
        x_test, y_test = test[predictors], test[OUTCOME].to_numpy(dtype=int)
        predictions: dict[str, np.ndarray] = {}
        fitted: dict[str, Pipeline] = {}
        prediction_file = test[["subject_id", "hadm_id", "stay_id", "landmark_hours"]].copy()
        prediction_file["y_true"] = y_test

        for model_index, (model_name, pipeline) in enumerate(model_definitions(continuous, binary, categorical).items()):
            print(f"Training {model_name} at {landmark} h...", flush=True)
            pipeline.fit(x_train, y_train)
            train_prob = pipeline.predict_proba(x_train)[:, 1]
            test_prob = pipeline.predict_proba(x_test)[:, 1]
            point_row, youden = evaluate(
                landmark, model_name, y_train, train_prob, y_test, test_prob,
                len(train), len(test), train.subject_id.nunique(), test.subject_id.nunique(),
            )
            ci = bootstrap_ci(
                y_test, test_prob, test.subject_id.to_numpy(dtype=int),
                OVERALL_BOOTSTRAPS, RANDOM_STATE + landmark * 100 + model_index,
            )
            performance_rows.append({**point_row, **ci})
            safe = model_name.lower().replace(" ", "_")
            prediction_file[f"prob_{safe}"] = test_prob
            prediction_file[f"pred_0_5_{safe}"] = (test_prob >= 0.5).astype(int)
            prediction_file[f"youden_threshold_{safe}"] = youden
            prediction_file[f"pred_youden_{safe}"] = (test_prob >= youden).astype(int)
            predictions[model_name] = test_prob
            fitted[model_name] = pipeline

        prediction_file.to_csv(OUTPUT_DIR / f"model_v5_1_{landmark}h_test_predictions.csv", index=False)
        subgroup_rows.extend(subgroup_performance_rows(landmark, test, predictions))
        landmark_dca = pd.DataFrame(dca_rows(landmark, y_test, predictions))
        all_dca_rows.extend(landmark_dca.to_dict("records"))
        plot_discrimination(landmark, y_test, predictions, "roc")
        plot_discrimination(landmark, y_test, predictions, "pr")
        plot_calibration(landmark, y_test, predictions)
        plot_dca(landmark, landmark_dca)

        tree_models = ["Random Forest", "XGBoost", "LightGBM"]
        best_tree = max(tree_models, key=lambda model: roc_auc_score(y_test, predictions[model]))
        shap_models[landmark] = best_tree
        all_shap_rows.extend(shap_analysis(landmark, best_tree, fitted[best_tree], x_test))

    performance = pd.DataFrame(performance_rows)
    numeric = performance.select_dtypes(include=[np.number]).columns
    performance[numeric] = performance[numeric].round(6)
    performance.to_csv(OUTPUT_DIR / "model_v5_1_performance_bootstrap_ci.csv", index=False)
    pd.DataFrame(subgroup_rows).to_csv(OUTPUT_DIR / "model_v5_1_subgroup_performance.csv", index=False)
    pd.DataFrame(all_dca_rows).to_csv(OUTPUT_DIR / "model_v5_1_dca.csv", index=False)
    pd.DataFrame(all_shap_rows).to_csv(OUTPUT_DIR / "model_v5_1_shap_importance.csv", index=False)
    write_readme(performance, shap_models)

    print("\n" + performance[["landmark_hours", "model", "auroc", "auroc_ci_lower", "auroc_ci_upper", "auprc", "auprc_ci_lower", "auprc_ci_upper", "brier_score"]].to_string(index=False))
    print(f"\nOutputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

