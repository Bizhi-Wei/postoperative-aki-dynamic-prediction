"""v7 secondary modeling for SCr-or-urine KDIGO AKI.

This is a secondary/sensitivity modeling analysis. It preserves the v4.1
time-restricted predictor tables and replaces the outcome with the v6 combined
SCr-or-urine KDIGO outcome.
"""

from __future__ import annotations

import sys
import textwrap
import warnings
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.pipeline import Pipeline

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from develop_models_v5 import (  # noqa: E402
    METADATA,
    RANDOM_STATE,
    calibration_intercept_slope,
    choose_grouped_split,
    identify_types,
    make_preprocessor,
)
from extend_models_v5_1 import bootstrap_ci  # noqa: E402


warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
)

PROJECT_ROOT = SCRIPT_DIR.parent
INPUT_DIR = PROJECT_ROOT / "outputs" / "modeling_v4_1"
V6_COHORT = PROJECT_ROOT / "outputs" / "v6_urine_output_or_pacu" / "cohort_v6_strict_primary_aki_scr_uo.csv"
V5_1_DIR = PROJECT_ROOT / "outputs" / "modeling_v5_1"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "modeling_v7_scr_or_uo_secondary"

LANDMARKS = [0, 6, 24]
OUTCOME = "outcome_aki_after_landmark_to_7d"
ONSET = "outcome_aki_onset_hours_from_icu"
SELECTED_MODELS = {0: "XGBoost", 6: "XGBoost", 24: "Logistic Regression"}
BOOTSTRAPS = 1000

TOKENS = {
    "surface": "#FCFCFD", "panel": "#FFFFFF", "ink": "#1F2430",
    "muted": "#6F768A", "grid": "#E6E8F0", "axis": "#D7DBE7",
}
MODEL_STYLES = {
    "Logistic Regression": ("#A3BEFA", "-"),
    "Random Forest": ("#F0986E", "--"),
    "XGBoost": ("#A3D576", "-."),
    "LightGBM": ("#F390CA", ":"),
}


def bool_mask(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype("string").str.strip().str.lower().isin(["true", "1", "yes"])


def model_definitions(continuous: list[str], binary: list[str], categorical: list[str]) -> dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline([
            ("preprocess", make_preprocessor(continuous, binary, categorical, scale=True)),
            ("model", LogisticRegression(max_iter=3000, solver="lbfgs", random_state=RANDOM_STATE)),
        ]),
        "Random Forest": Pipeline([
            ("preprocess", make_preprocessor(continuous, binary, categorical, scale=False)),
            ("model", RandomForestClassifier(n_estimators=500, min_samples_leaf=5, max_features="sqrt", n_jobs=-1, random_state=RANDOM_STATE)),
        ]),
        "XGBoost": Pipeline([
            ("preprocess", make_preprocessor(continuous, binary, categorical, scale=False)),
            ("model", xgb.XGBClassifier(
                n_estimators=500, learning_rate=0.03, max_depth=4,
                min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                reg_lambda=1.0, objective="binary:logistic", eval_metric="logloss",
                n_jobs=-1, random_state=RANDOM_STATE,
            )),
        ]),
        "LightGBM": Pipeline([
            ("preprocess", make_preprocessor(continuous, binary, categorical, scale=False)),
            ("model", lgb.LGBMClassifier(
                n_estimators=500, learning_rate=0.03, num_leaves=31,
                min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                reg_lambda=1.0, n_jobs=-1, random_state=RANDOM_STATE, verbosity=-1,
            )),
        ]),
    }


def load_v6_outcomes() -> pd.DataFrame:
    usecols = [
        "subject_id", "hadm_id", "stay_id", "aki_final", "aki_onset_hours_final",
        "uo_aki", "uo_onset_hours", "aki_scr_or_uo_final",
        "aki_onset_hours_scr_or_uo_final", "aki_stage_scr_or_uo_final",
    ]
    v6 = pd.read_csv(V6_COHORT, usecols=usecols, low_memory=False)
    v6["aki_scr_or_uo_final"] = bool_mask(v6["aki_scr_or_uo_final"])
    return v6


def load_secondary_dataset(landmark: int, v6: pd.DataFrame) -> pd.DataFrame:
    path = INPUT_DIR / f"modeling_v4_1_{landmark}h.csv"
    data = pd.read_csv(path, low_memory=False)
    original_n = len(data)
    data = data.drop(columns=[OUTCOME], errors="ignore")
    data = data.merge(v6, on=["subject_id", "hadm_id", "stay_id"], how="inner", validate="one_to_one")
    if len(data) != original_n:
        raise ValueError(f"{landmark} h merge changed row count: {original_n} -> {len(data)}")
    onset = pd.to_numeric(data["aki_onset_hours_scr_or_uo_final"], errors="coerce")
    event = bool_mask(data["aki_scr_or_uo_final"])
    # Landmark risk set: exclude events already occurred by the landmark.
    keep = ~(event & onset.notna() & onset.le(landmark))
    data = data.loc[keep].copy()
    data[OUTCOME] = (event.loc[keep] & onset.loc[keep].gt(landmark)).astype(int)
    data[ONSET] = onset.loc[keep]
    # Remove v6 outcome columns from predictors.
    leakage_cols = [
        "aki_final", "aki_onset_hours_final", "uo_aki", "uo_onset_hours",
        "aki_scr_or_uo_final", "aki_onset_hours_scr_or_uo_final",
        "aki_stage_scr_or_uo_final", ONSET,
    ]
    data = data.drop(columns=[c for c in leakage_cols if c in data.columns and c != ONSET])
    return data


def count_excluded_by_landmark(landmark: int, v6: pd.DataFrame) -> int:
    path = INPUT_DIR / f"modeling_v4_1_{landmark}h.csv"
    base = pd.read_csv(path, usecols=["subject_id", "hadm_id", "stay_id"], low_memory=False)
    merged = base.merge(
        v6[["subject_id", "hadm_id", "stay_id", "aki_scr_or_uo_final", "aki_onset_hours_scr_or_uo_final"]],
        on=["subject_id", "hadm_id", "stay_id"],
        how="inner",
        validate="one_to_one",
    )
    event = bool_mask(merged["aki_scr_or_uo_final"])
    onset = pd.to_numeric(merged["aki_onset_hours_scr_or_uo_final"], errors="coerce")
    return int((event & onset.notna() & onset.le(landmark)).sum())


def youden_threshold(y_true: np.ndarray, p: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, p)
    finite = np.isfinite(thresholds)
    j = tpr - fpr
    eligible = np.where(finite)[0]
    return float(thresholds[eligible[np.argmax(j[eligible])]])


def threshold_metrics(y_true: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float | int]:
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, pred),
        "sensitivity": recall_score(y_true, pred, zero_division=0),
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "precision": precision_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def evaluate_model(landmark: int, model: str, y_train: np.ndarray, p_train: np.ndarray, y_test: np.ndarray, p_test: np.ndarray, test_subjects: np.ndarray) -> dict[str, object]:
    threshold = youden_threshold(y_train, p_train)
    fixed = threshold_metrics(y_test, p_test, 0.5)
    youden = threshold_metrics(y_test, p_test, threshold)
    intercept, slope = calibration_intercept_slope(y_test, p_test)
    row: dict[str, object] = {
        "landmark_hours": landmark,
        "model": model,
        "selected_model_for_landmark": model == SELECTED_MODELS[landmark],
        "test_n": len(y_test),
        "test_event_n": int(y_test.sum()),
        "test_event_rate": float(y_test.mean()),
        "auroc": roc_auc_score(y_test, p_test),
        "auprc": average_precision_score(y_test, p_test),
        "brier_score": brier_score_loss(y_test, p_test),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "youden_threshold": threshold,
    }
    row.update({f"threshold_0_5_{k}": v for k, v in fixed.items()})
    row.update({f"threshold_youden_{k}": v for k, v in youden.items()})
    row.update(bootstrap_ci(y_test, p_test, test_subjects, BOOTSTRAPS, RANDOM_STATE + 7000 + landmark * 100 + len(model)))
    return row


def use_theme() -> None:
    sns.set_theme(style="whitegrid", rc={
        "figure.facecolor": TOKENS["surface"], "axes.facecolor": TOKENS["panel"],
        "axes.edgecolor": TOKENS["axis"], "axes.labelcolor": TOKENS["ink"],
        "grid.color": TOKENS["grid"], "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "axes.spines.top": False, "axes.spines.right": False,
    })


def add_header(fig, ax, title: str, subtitle: str) -> None:
    fig.subplots_adjust(top=0.80, left=0.11, right=0.97, bottom=0.12)
    left = ax.get_position().x0
    fig.text(left, 0.97, textwrap.fill(title, 80), ha="left", va="top", fontsize=14, fontweight="semibold", color=TOKENS["ink"])
    fig.text(left, 0.91, textwrap.fill(subtitle, 115), ha="left", va="top", fontsize=9, color=TOKENS["muted"])
    sns.despine(ax=ax)


def plot_curves(landmark: int, y: np.ndarray, preds: dict[str, np.ndarray], kind: str) -> None:
    use_theme()
    fig, ax = plt.subplots(figsize=(9, 6.2), dpi=180)
    for model, p in preds.items():
        color, linestyle = MODEL_STYLES[model]
        if kind == "roc":
            x, yy, _ = roc_curve(y, p)
            label = f"{model} ({roc_auc_score(y, p):.3f})"
        else:
            yy, x, _ = precision_recall_curve(y, p)
            label = f"{model} ({average_precision_score(y, p):.3f})"
        sns.lineplot(x=x, y=yy, ax=ax, color=color, linestyle=linestyle, linewidth=1.35, label=label)
    if kind == "roc":
        ax.plot([0, 1], [0, 1], color=TOKENS["ink"], linestyle=":", linewidth=1)
        ax.set(xlabel="1 - Specificity", ylabel="Sensitivity", xlim=(0, 1), ylim=(0, 1))
        title = f"Secondary SCr-or-urine AKI ROC curves at {landmark} h"
        subtitle = f"Subject-grouped held-out test set; n={len(y):,}. Legend values are AUROC."
        loc = "lower right"
    else:
        ax.axhline(y.mean(), color=TOKENS["ink"], linestyle=":", linewidth=1, label=f"Prevalence ({y.mean():.3f})")
        ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1))
        title = f"Secondary SCr-or-urine AKI precision-recall curves at {landmark} h"
        subtitle = f"Subject-grouped held-out test set; event prevalence={y.mean():.3f}."
        loc = "upper right"
    ax.legend(loc=loc, frameon=False)
    add_header(fig, ax, title, subtitle)
    fig.savefig(OUTPUT_DIR / f"figure_v7_{kind}_{landmark}h.png", bbox_inches="tight", facecolor=TOKENS["surface"])
    plt.close(fig)


def plot_calibration(landmark: int, y: np.ndarray, preds: dict[str, np.ndarray]) -> None:
    use_theme()
    fig, ax = plt.subplots(figsize=(9, 6.2), dpi=180)
    for model, p in preds.items():
        obs, pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
        color, linestyle = MODEL_STYLES[model]
        sns.lineplot(x=pred, y=obs, ax=ax, color=color, linestyle=linestyle, marker="o", markersize=4, label=model)
    ax.plot([0, 1], [0, 1], color=TOKENS["ink"], linestyle=":", linewidth=1, label="Ideal")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed event proportion", xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="upper left", frameon=False)
    add_header(fig, ax, f"Secondary SCr-or-urine AKI calibration at {landmark} h", "Ten equal-frequency calibration groups in the held-out test set.")
    fig.savefig(OUTPUT_DIR / f"figure_v7_calibration_{landmark}h.png", bbox_inches="tight", facecolor=TOKENS["surface"])
    plt.close(fig)


def dca_rows(landmark: int, y: np.ndarray, preds: dict[str, np.ndarray]) -> list[dict[str, object]]:
    rows = []
    n = len(y)
    prevalence = float(y.mean())
    for threshold in np.arange(0.05, 0.801, 0.01):
        odds = threshold / (1 - threshold)
        rows.append({"landmark_hours": landmark, "strategy": "Treat none", "threshold": threshold, "net_benefit": 0.0})
        rows.append({"landmark_hours": landmark, "strategy": "Treat all", "threshold": threshold, "net_benefit": prevalence - (1 - prevalence) * odds})
        for model, p in preds.items():
            pred = p >= threshold
            tp = np.sum(pred & (y == 1))
            fp = np.sum(pred & (y == 0))
            nb = tp / n - fp / n * odds
            rows.append({"landmark_hours": landmark, "strategy": model, "threshold": threshold, "net_benefit": nb})
    return rows


def build_comparison(perf: pd.DataFrame) -> pd.DataFrame:
    old = pd.read_csv(V5_1_DIR / "model_v5_1_performance_bootstrap_ci.csv")
    rows = []
    for landmark, model in SELECTED_MODELS.items():
        new_row = perf[(perf.landmark_hours.eq(landmark)) & (perf.model.eq(model))].iloc[0]
        old_row = old[(old.landmark_hours.eq(landmark)) & (old.model.eq(model))].iloc[0]
        rows.append({
            "landmark_hours": landmark,
            "selected_model": model,
            "scr_only_test_event_rate": old_row["test_event_rate"],
            "scr_or_uo_test_event_rate": new_row["test_event_rate"],
            "scr_only_auroc": old_row["auroc"],
            "scr_or_uo_auroc": new_row["auroc"],
            "auroc_difference_scr_or_uo_minus_scr_only": new_row["auroc"] - old_row["auroc"],
            "scr_only_auprc": old_row["auprc"],
            "scr_or_uo_auprc": new_row["auprc"],
            "auprc_difference_scr_or_uo_minus_scr_only": new_row["auprc"] - old_row["auprc"],
            "scr_only_brier": old_row["brier_score"],
            "scr_or_uo_brier": new_row["brier_score"],
        })
    return pd.DataFrame(rows)


def write_readme(summary: pd.DataFrame, comparison: pd.DataFrame, risksets: pd.DataFrame) -> None:
    def simple_markdown_table(df: pd.DataFrame) -> str:
        text_df = df.copy()
        for col in text_df.columns:
            if pd.api.types.is_float_dtype(text_df[col]):
                text_df[col] = text_df[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
            else:
                text_df[col] = text_df[col].astype(str)
        rows = []
        cols = list(text_df.columns)
        rows.append("| " + " | ".join(cols) + " |")
        rows.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for _, row in text_df.iterrows():
            rows.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
        return "\n".join(rows)

    lines = ["# v7 SCr-or-urine KDIGO secondary modeling", ""]
    lines.append("This analysis uses the same time-restricted v4.1 predictors and replaces the outcome with the v6 combined SCr-or-urine KDIGO AKI endpoint.")
    lines.append("")
    lines.append("## Dynamic risk sets")
    lines.append("")
    lines.append(simple_markdown_table(risksets))
    lines.append("")
    lines.append("## Selected-model comparison")
    lines.append("")
    lines.append(simple_markdown_table(comparison))
    lines.append("")
    lines.append("Important: this is a secondary outcome analysis. It should not silently replace the SCr-only primary model because the event rate and clinical interpretation are materially different.")
    (OUTPUT_DIR / "audit_v7_readme.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    v6 = load_v6_outcomes()
    datasets = {landmark: load_secondary_dataset(landmark, v6) for landmark in LANDMARKS}
    train_subjects, test_subjects, _ = choose_grouped_split(datasets[0])

    performance_rows = []
    dca_all = []
    riskset_rows = []

    for landmark, data in datasets.items():
        train = data[data["subject_id"].astype(int).isin(train_subjects)].copy()
        test = data[data["subject_id"].astype(int).isin(test_subjects)].copy()
        if set(train["subject_id"]).intersection(set(test["subject_id"])):
            raise AssertionError("Subject leakage")
        predictors = [c for c in data.columns if c not in {*METADATA, OUTCOME, ONSET}]
        continuous, binary, categorical = identify_types(data.drop(columns=[ONSET], errors="ignore"))
        x_train, y_train = train[predictors], train[OUTCOME].to_numpy(dtype=int)
        x_test, y_test = test[predictors], test[OUTCOME].to_numpy(dtype=int)
        test_subject_array = test["subject_id"].to_numpy(dtype=int)
        riskset_rows.append({
            "landmark_hours": landmark,
            "risk_set_n": len(data),
            "event_n": int(data[OUTCOME].sum()),
            "event_rate": float(data[OUTCOME].mean()),
            "excluded_events_at_or_before_landmark": count_excluded_by_landmark(landmark, v6),
            "train_n": len(train),
            "test_n": len(test),
            "test_event_rate": float(y_test.mean()),
        })

        pred_out = test[["subject_id", "hadm_id", "stay_id", "landmark_hours"]].copy()
        pred_out["y_true"] = y_test
        preds = {}
        for model_name, pipe in model_definitions(continuous, binary, categorical).items():
            print(f"Training v7 {model_name} at {landmark} h...", flush=True)
            pipe.fit(x_train, y_train)
            p_train = pipe.predict_proba(x_train)[:, 1]
            p_test = pipe.predict_proba(x_test)[:, 1]
            performance_rows.append(evaluate_model(landmark, model_name, y_train, p_train, y_test, p_test, test_subject_array))
            safe = model_name.lower().replace(" ", "_")
            pred_out[f"prob_{safe}"] = p_test
            pred_out[f"pred_0_5_{safe}"] = (p_test >= 0.5).astype(int)
            pred_out[f"youden_threshold_{safe}"] = youden_threshold(y_train, p_train)
            pred_out[f"pred_youden_{safe}"] = (p_test >= pred_out[f"youden_threshold_{safe}"].iloc[0]).astype(int)
            preds[model_name] = p_test
        pred_out.to_csv(OUTPUT_DIR / f"model_v7_{landmark}h_test_predictions.csv", index=False)
        plot_curves(landmark, y_test, preds, "roc")
        plot_curves(landmark, y_test, preds, "pr")
        plot_calibration(landmark, y_test, preds)
        dca_all.extend(dca_rows(landmark, y_test, preds))

    perf = pd.DataFrame(performance_rows)
    perf.to_csv(OUTPUT_DIR / "model_v7_performance_bootstrap_ci.csv", index=False)
    pd.DataFrame(dca_all).to_csv(OUTPUT_DIR / "model_v7_dca.csv", index=False)
    risksets = pd.DataFrame(riskset_rows)
    risksets.to_csv(OUTPUT_DIR / "audit_v7_dynamic_risk_sets.csv", index=False)
    comparison = build_comparison(perf)
    comparison.to_csv(OUTPUT_DIR / "model_v7_selected_model_scr_only_vs_scr_uo_comparison.csv", index=False)
    write_readme(perf, comparison, risksets)
    print(perf[["landmark_hours", "model", "test_n", "test_event_rate", "auroc", "auprc", "brier_score"]].to_string(index=False))
    print(f"Wrote v7 outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
