"""Develop and internally validate dynamic postoperative AKI models.

Models are fit separately at 0 h, 6 h, and 24 h. A single deterministic
subject-level 80/20 assignment is selected on the 0 h cohort and reused at all
landmarks. SHAP and manuscript generation are out of scope.
"""

from __future__ import annotations

import math
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "outputs" / "modeling_v4_1"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "modeling_v5"
OUTCOME = "outcome_aki_after_landmark_to_7d"
METADATA = ["subject_id", "hadm_id", "stay_id", "landmark_hours"]
LANDMARKS = [0, 6, 24]
RANDOM_STATE = 20260704

CATEGORICAL_COLUMNS = [
    "first_careunit", "gender", "race", "admission_type", "insurance",
    "marital_status", "baseline_scr_source_at_landmark",
]

TOKENS = {
    "surface": "#FCFCFD", "panel": "#FFFFFF", "ink": "#1F2430",
    "muted": "#6F768A", "grid": "#E6E8F0", "axis": "#D7DBE7",
}
MODEL_STYLES = {
    "Logistic Regression": {"color": "#A3BEFA", "edge": "#2E4780", "linestyle": "-"},
    "Random Forest": {"color": "#F0986E", "edge": "#804126", "linestyle": "--"},
}


def bool_mask(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype("string").str.strip().str.lower().isin(["true", "1", "yes"])


def load_data(landmark: int) -> pd.DataFrame:
    path = INPUT_DIR / f"modeling_v4_1_{landmark}h.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    data = pd.read_csv(path, low_memory=False)
    required = {*METADATA, OUTCOME}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"{landmark} h file missing columns: {missing}")
    if data["stay_id"].duplicated().any():
        raise ValueError(f"{landmark} h file contains duplicate stay_id")
    data[OUTCOME] = bool_mask(data[OUTCOME]).astype(int)
    return data


def choose_grouped_split(data: pd.DataFrame) -> tuple[set[int], set[int], dict[str, float]]:
    groups = data["subject_id"].astype(int)
    y = data[OUTCOME].astype(int)
    splitter = GroupShuffleSplit(n_splits=500, test_size=0.20, random_state=RANDOM_STATE)
    overall_rate = float(y.mean())
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    for train_idx, test_idx in splitter.split(data, y, groups):
        size_penalty = abs(len(test_idx) / len(data) - 0.20)
        prevalence_penalty = abs(y.iloc[test_idx].mean() - overall_rate) + abs(
            y.iloc[train_idx].mean() - overall_rate
        )
        score = 2 * size_penalty + prevalence_penalty
        if best is None or score < best[0]:
            best = (score, train_idx, test_idx)
    assert best is not None
    _, train_idx, test_idx = best
    train_subjects = set(groups.iloc[train_idx])
    test_subjects = set(groups.iloc[test_idx])
    if train_subjects & test_subjects:
        raise AssertionError("Subject overlap in grouped split")
    audit = {
        "overall_n": len(data),
        "train_n": len(train_idx),
        "test_n": len(test_idx),
        "overall_event_rate": overall_rate,
        "train_event_rate": float(y.iloc[train_idx].mean()),
        "test_event_rate": float(y.iloc[test_idx].mean()),
        "train_subjects": len(train_subjects),
        "test_subjects": len(test_subjects),
    }
    return train_subjects, test_subjects, audit


def identify_types(data: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    predictors = [column for column in data.columns if column not in {*METADATA, OUTCOME}]
    categorical = [column for column in CATEGORICAL_COLUMNS if column in predictors]
    binary: list[str] = []
    continuous: list[str] = []
    for column in predictors:
        if column in categorical:
            continue
        series = data[column]
        observed = pd.to_numeric(series, errors="coerce").dropna()
        unique = set(observed.unique())
        if pd.api.types.is_bool_dtype(series) or (unique and unique.issubset({0, 1})):
            binary.append(column)
        elif pd.api.types.is_numeric_dtype(series):
            continuous.append(column)
        else:
            raise ValueError(f"Unclassified nonnumeric predictor: {column}")
    return continuous, binary, categorical


def make_preprocessor(
    continuous: list[str], binary: list[str], categorical: list[str], *, scale: bool
) -> ColumnTransformer:
    continuous_steps = [
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
    ]
    if scale:
        continuous_steps.append(("scaler", StandardScaler()))
    continuous_pipe = Pipeline(continuous_steps)
    binary_pipe = Pipeline(
        [("imputer", SimpleImputer(strategy="most_frequent", add_indicator=True))]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )
    return ColumnTransformer(
        [
            ("continuous", continuous_pipe, continuous),
            ("binary", binary_pipe, binary),
            ("categorical", categorical_pipe, categorical),
        ],
        remainder="drop",
    )


def model_definitions(
    continuous: list[str], binary: list[str], categorical: list[str]
) -> dict[str, Pipeline]:
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
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=500,
                        min_samples_leaf=5,
                        max_features="sqrt",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def youden_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    fpr, tpr, thresholds = roc_curve(y_true, probabilities)
    finite = np.isfinite(thresholds)
    j = tpr - fpr
    eligible = np.where(finite)[0]
    best = eligible[np.argmax(j[eligible])]
    return float(thresholds[best]), float(j[best])


def threshold_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float | int]:
    predicted = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if tn + fp else np.nan
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, predicted),
        "sensitivity": recall_score(y_true, predicted, zero_division=0),
        "specificity": specificity,
        "precision": precision_score(y_true, predicted, zero_division=0),
        "f1": f1_score(y_true, predicted, zero_division=0),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def calibration_intercept_slope(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    calibration_model = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=2000)
    calibration_model.fit(logits, y_true)
    return float(calibration_model.intercept_[0]), float(calibration_model.coef_[0, 0])


def evaluate(
    landmark: int,
    model_name: str,
    y_train: np.ndarray,
    train_prob: np.ndarray,
    y_test: np.ndarray,
    test_prob: np.ndarray,
    train_n: int,
    test_n: int,
    train_subject_n: int,
    test_subject_n: int,
) -> tuple[dict[str, object], float]:
    optimized_threshold, train_youden_j = youden_threshold(y_train, train_prob)
    fixed = threshold_metrics(y_test, test_prob, 0.5)
    optimized = threshold_metrics(y_test, test_prob, optimized_threshold)
    calibration_intercept, calibration_slope = calibration_intercept_slope(y_test, test_prob)
    row: dict[str, object] = {
        "landmark_hours": landmark,
        "model": model_name,
        "train_n": train_n,
        "test_n": test_n,
        "train_subject_n": train_subject_n,
        "test_subject_n": test_subject_n,
        "train_event_rate": float(np.mean(y_train)),
        "test_event_rate": float(np.mean(y_test)),
        "auroc": roc_auc_score(y_test, test_prob),
        "auprc": average_precision_score(y_test, test_prob),
        "brier_score": brier_score_loss(y_test, test_prob),
        "calibration_intercept": calibration_intercept,
        "calibration_slope": calibration_slope,
        "youden_threshold_source": "training_set",
        "youden_threshold": optimized_threshold,
        "train_youden_j": train_youden_j,
    }
    for name, metrics in [("threshold_0_5", fixed), ("threshold_youden", optimized)]:
        for metric, value in metrics.items():
            row[f"{name}_{metric}"] = value
    return row, optimized_threshold


def use_chart_theme() -> None:
    sns.set_theme(
        style="whitegrid",
        rc={
            "figure.facecolor": TOKENS["surface"], "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": TOKENS["axis"], "axes.labelcolor": TOKENS["ink"],
            "grid.color": TOKENS["grid"], "grid.linewidth": 0.8,
            "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
            "axes.spines.top": False, "axes.spines.right": False,
        },
    )


def add_chart_header(fig, ax, title: str, subtitle: str) -> None:
    title = textwrap.fill(title, 78, break_long_words=False)
    subtitle = textwrap.fill(subtitle, 112, break_long_words=False)
    fig.subplots_adjust(top=0.80, left=0.11, right=0.97, bottom=0.12)
    left = ax.get_position().x0
    fig.text(left, 0.97, title, ha="left", va="top", fontsize=14, fontweight="semibold", color=TOKENS["ink"])
    fig.text(left, 0.91, subtitle, ha="left", va="top", fontsize=9, color=TOKENS["muted"])
    sns.despine(ax=ax)


def plot_curves(
    landmark: int,
    y_test: np.ndarray,
    probabilities: dict[str, np.ndarray],
    kind: str,
) -> None:
    use_chart_theme()
    fig, ax = plt.subplots(figsize=(8.8, 6.2), dpi=180)
    for model_name, probability in probabilities.items():
        style = MODEL_STYLES[model_name]
        if kind == "roc":
            x, y, _ = roc_curve(y_test, probability)
            label = f"{model_name} (AUROC {roc_auc_score(y_test, probability):.3f})"
            x_label, y_label = "1 − Specificity", "Sensitivity"
        else:
            y, x, _ = precision_recall_curve(y_test, probability)
            label = f"{model_name} (AUPRC {average_precision_score(y_test, probability):.3f})"
            x_label, y_label = "Recall", "Precision"
        sns.lineplot(x=x, y=y, ax=ax, color=style["color"], linestyle=style["linestyle"], linewidth=1.5, label=label)
    if kind == "roc":
        ax.plot([0, 1], [0, 1], color=TOKENS["ink"], linestyle=":", linewidth=1.0, label="Chance")
        title = f"ROC curves for {landmark} h postoperative AKI prediction"
        subtitle = f"Held-out subject-grouped test set; n={len(y_test):,}; outcome after {landmark} h through day 7."
    else:
        prevalence = float(np.mean(y_test))
        ax.axhline(prevalence, color=TOKENS["ink"], linestyle=":", linewidth=1.0, label=f"Prevalence ({prevalence:.3f})")
        title = f"Precision–recall curves for {landmark} h postoperative AKI prediction"
        subtitle = f"Held-out subject-grouped test set; n={len(y_test):,}; event prevalence={prevalence:.3f}."
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel=x_label, ylabel=y_label)
    ax.legend(loc="lower right" if kind == "roc" else "upper right", frameon=False)
    add_chart_header(fig, ax, title, subtitle)
    fig.savefig(OUTPUT_DIR / f"figure_v5_{kind}_{landmark}h.png", bbox_inches="tight", facecolor=TOKENS["surface"])
    plt.close(fig)


def plot_calibration(landmark: int, y_test: np.ndarray, probabilities: dict[str, np.ndarray]) -> None:
    use_chart_theme()
    fig, ax = plt.subplots(figsize=(8.8, 6.2), dpi=180)
    for model_name, probability in probabilities.items():
        observed, predicted = calibration_curve(y_test, probability, n_bins=10, strategy="quantile")
        style = MODEL_STYLES[model_name]
        sns.lineplot(
            x=predicted, y=observed, ax=ax, color=style["color"], linestyle=style["linestyle"],
            marker="o", markersize=5, linewidth=1.3, label=model_name,
        )
    ax.plot([0, 1], [0, 1], color=TOKENS["ink"], linestyle=":", linewidth=1.0, label="Ideal calibration")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability", ylabel="Observed event proportion")
    ax.legend(loc="upper left", frameon=False)
    add_chart_header(
        fig, ax,
        f"Calibration curves for {landmark} h postoperative AKI prediction",
        f"Ten equal-frequency bins on the held-out subject-grouped test set; n={len(y_test):,}.",
    )
    fig.savefig(OUTPUT_DIR / f"figure_v5_calibration_{landmark}h.png", bbox_inches="tight", facecolor=TOKENS["surface"])
    plt.close(fig)


def write_readme(performance: pd.DataFrame, split_audit: dict[int, dict[str, float]]) -> None:
    best = performance.loc[performance.groupby("landmark_hours")["auroc"].idxmax()]
    best_lines = "\n".join(
        f"- {int(row.landmark_hours)} h: {row.model}, AUROC {row.auroc:.3f}, AUPRC {row.auprc:.3f}, Brier {row.brier_score:.3f}."
        for row in best.itertuples()
    )
    split_lines = "\n".join(
        f"- {landmark} h: train n={int(a['train_n']):,} ({a['train_event_rate']:.3f} events), test n={int(a['test_n']):,} ({a['test_event_rate']:.3f} events), subject overlap=0."
        for landmark, a in split_audit.items()
    )
    content = f"""# Dynamic postoperative AKI model development v5

## Scope

- Separate models were developed at 0 h, 6 h, and 24 h.
- Outcome: `{OUTCOME}`.
- Identifiers and `landmark_hours` were not used as predictors.
- A deterministic subject-level 80/20 split was selected from 500 grouped split candidates to approximate overall event prevalence and was reused at all landmarks.
- No SHAP analysis and no manuscript drafting were performed.

## Models

- Logistic regression: median imputation plus missingness indicators for continuous variables, most-frequent categorical/binary imputation, one-hot categorical encoding, and continuous standardization.
- Random forest: the same variable roles, median/most-frequent imputation with continuous missingness indicators, and one-hot categorical encoding; 500 trees, `min_samples_leaf=5`.
- XGBoost and LightGBM were not installed in the execution environment and were therefore not trained.
- Fixed hyperparameters were used; there was no test-set-driven model tuning.

## Split audit

{split_lines}

## Best held-out discrimination by landmark

{best_lines}

## Thresholds and calibration

- The 0.5 threshold metrics and confusion matrix are reported directly on the held-out test set.
- Youden thresholds were selected on the training predictions, then applied unchanged to the test set. They were not optimized on the test outcomes.
- Calibration intercept and slope are diagnostic regressions fitted on held-out predictions; ideal values are 0 and 1.
- Calibration curves use ten equal-frequency bins.

## Internal-validation caveats

This is a single-center, single grouped holdout validation. The random forest and logistic regression use fixed development settings without nested tuning. Test-set calibration diagnostics and model comparisons are exploratory internal-validation results, not external validation.
"""
    (OUTPUT_DIR / "audit_v5_modeling_readme.md").write_text(content, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    datasets = {landmark: load_data(landmark) for landmark in LANDMARKS}
    train_subjects, test_subjects, base_audit = choose_grouped_split(datasets[0])

    performance_rows: list[dict[str, object]] = []
    split_audits: dict[int, dict[str, float]] = {}

    for landmark, data in datasets.items():
        train_mask = data["subject_id"].astype(int).isin(train_subjects)
        test_mask = data["subject_id"].astype(int).isin(test_subjects)
        if (~(train_mask | test_mask)).any() or (train_mask & test_mask).any():
            raise AssertionError(f"Invalid split assignment at {landmark} h")
        train = data.loc[train_mask].copy()
        test = data.loc[test_mask].copy()
        if set(train["subject_id"]) & set(test["subject_id"]):
            raise AssertionError(f"Subject leakage at {landmark} h")

        predictors = [column for column in data.columns if column not in {*METADATA, OUTCOME}]
        continuous, binary, categorical = identify_types(data)
        x_train, y_train = train[predictors], train[OUTCOME].to_numpy(dtype=int)
        x_test, y_test = test[predictors], test[OUTCOME].to_numpy(dtype=int)

        split_audits[landmark] = {
            "train_n": len(train), "test_n": len(test),
            "train_event_rate": float(np.mean(y_train)), "test_event_rate": float(np.mean(y_test)),
            "train_subjects": train["subject_id"].nunique(), "test_subjects": test["subject_id"].nunique(),
        }
        prediction_output = test[["subject_id", "hadm_id", "stay_id", "landmark_hours"]].copy()
        prediction_output["y_true"] = y_test
        probabilities: dict[str, np.ndarray] = {}

        for model_name, pipeline in model_definitions(continuous, binary, categorical).items():
            print(f"Training {model_name} at {landmark} h...", flush=True)
            pipeline.fit(x_train, y_train)
            train_prob = pipeline.predict_proba(x_train)[:, 1]
            test_prob = pipeline.predict_proba(x_test)[:, 1]
            row, threshold = evaluate(
                landmark, model_name, y_train, train_prob, y_test, test_prob,
                len(train), len(test), train["subject_id"].nunique(), test["subject_id"].nunique(),
            )
            performance_rows.append(row)
            safe_name = model_name.lower().replace(" ", "_")
            prediction_output[f"prob_{safe_name}"] = test_prob
            prediction_output[f"pred_0_5_{safe_name}"] = (test_prob >= 0.5).astype(int)
            prediction_output[f"youden_threshold_{safe_name}"] = threshold
            prediction_output[f"pred_youden_{safe_name}"] = (test_prob >= threshold).astype(int)
            probabilities[model_name] = test_prob

        prediction_output.to_csv(OUTPUT_DIR / f"model_v5_{landmark}h_test_predictions.csv", index=False)
        plot_curves(landmark, y_test, probabilities, "roc")
        plot_curves(landmark, y_test, probabilities, "pr")
        plot_calibration(landmark, y_test, probabilities)

    performance = pd.DataFrame(performance_rows)
    numeric_columns = performance.select_dtypes(include=[np.number]).columns
    performance[numeric_columns] = performance[numeric_columns].round(6)
    performance.to_csv(OUTPUT_DIR / "model_v5_performance_summary.csv", index=False)
    write_readme(performance, split_audits)
    print("\n" + performance[["landmark_hours", "model", "auroc", "auprc", "brier_score", "calibration_intercept", "calibration_slope"]].to_string(index=False))
    print(f"\nOutputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
