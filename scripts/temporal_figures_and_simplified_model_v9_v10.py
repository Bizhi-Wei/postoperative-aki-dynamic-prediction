"""Formal temporal-validation figures and simplified clinical model analysis.

v9:
- Build publication-ready temporal-validation figures/tables from v5.4.

v10:
- Compare full v4.1 predictors against a prespecified simplified clinical
  feature set at 0 h, 6 h, and 24 h.
"""

from __future__ import annotations

import sys
import textwrap
import warnings
import zlib
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from develop_models_v5 import (  # noqa: E402
    METADATA,
    OUTCOME,
    RANDOM_STATE,
    choose_grouped_split,
    identify_types,
    load_data,
    make_preprocessor,
)
from extend_models_v5_1 import bootstrap_ci  # noqa: E402


warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
)

PROJECT_ROOT = SCRIPT_DIR.parent
V54_DIR = PROJECT_ROOT / "outputs" / "modeling_v5_4_small_optimizations"
V51_DIR = PROJECT_ROOT / "outputs" / "modeling_v5_1"
V9_DIR = PROJECT_ROOT / "outputs" / "modeling_v9_temporal_validation_figures"
V10_DIR = PROJECT_ROOT / "outputs" / "modeling_v10_simplified_model"
LANDMARKS = [0, 6, 24]
BOOTSTRAPS = 1000

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
SELECTED_MODELS = {0: "XGBoost", 6: "XGBoost", 24: "Logistic Regression"}


BASE_SIMPLIFIED = [
    "first_careunit", "gender", "anchor_age", "race", "admission_type",
    "chf", "hypertension", "dm", "ckd", "copd", "liver", "cancer",
    "pvd", "stroke", "mi", "obesity", "anemia", "charlson_score",
    "cardiac_surgery", "non_cardiac_surgery", "vascular_surgery",
    "general_gi_hepatobiliary_surgery", "orthopedic_major_surgery",
    "neurosurgery", "thoracic_respiratory_surgery",
    "baseline_scr_at_landmark", "baseline_scr_source_at_landmark",
    "baseline_to_icu_hours_at_landmark",
    "lab_pre24h_bun_last", "lab_pre24h_creatinine_last",
    "lab_pre24h_hemoglobin_last", "lab_pre24h_lactate_last",
    "lab_pre24h_wbc_last", "lab_pre24h_platelet_last",
    "lab_pre24h_potassium_last", "lab_pre24h_sodium_last",
]
EARLY_LABS = ["bun", "creatinine", "hemoglobin", "lactate", "wbc", "platelet", "potassium", "sodium"]
EARLY_VITALS = ["map", "heart_rate", "sbp", "spo2"]


def use_theme() -> None:
    sns.set_theme(style="whitegrid", rc={
        "figure.facecolor": TOKENS["surface"], "axes.facecolor": TOKENS["panel"],
        "axes.edgecolor": TOKENS["axis"], "axes.labelcolor": TOKENS["ink"],
        "grid.color": TOKENS["grid"], "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "axes.spines.top": False, "axes.spines.right": False,
    })


def add_header(fig, ax, title: str, subtitle: str, top: float = 0.80) -> None:
    fig.subplots_adjust(top=top, left=0.12, right=0.96, bottom=0.14)
    left = ax.get_position().x0
    fig.text(left, 0.97, textwrap.fill(title, 80), ha="left", va="top", fontsize=14, fontweight="semibold", color=TOKENS["ink"])
    fig.text(left, 0.91, textwrap.fill(subtitle, 115), ha="left", va="top", fontsize=9, color=TOKENS["muted"])
    sns.despine(ax=ax)


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


def build_v9_temporal_figures() -> None:
    V9_DIR.mkdir(parents=True, exist_ok=True)
    temporal = pd.read_csv(V54_DIR / "model_v5_4_temporal_validation_performance.csv")
    split = pd.read_csv(V54_DIR / "model_v5_4_temporal_split_audit.csv")
    random_perf = pd.read_csv(V51_DIR / "model_v5_1_performance_bootstrap_ci.csv")
    selected_temporal = temporal[temporal["selected_model_for_landmark"].eq(True)].copy()
    comparison_rows = []
    for _, row in selected_temporal.iterrows():
        old = random_perf[
            random_perf["landmark_hours"].eq(row["landmark_hours"])
            & random_perf["model"].eq(row["model"])
        ].iloc[0]
        comparison_rows.append({
            "landmark_hours": int(row["landmark_hours"]),
            "selected_model": row["model"],
            "random_split_auroc": old["auroc"],
            "temporal_auroc": row["auroc"],
            "delta_temporal_minus_random_auroc": row["auroc"] - old["auroc"],
            "random_split_auprc": old["auprc"],
            "temporal_auprc": row["auprc"],
            "delta_temporal_minus_random_auprc": row["auprc"] - old["auprc"],
            "random_split_brier": old["brier_score"],
            "temporal_brier": row["brier_score"],
            "temporal_auroc_ci_lower": row["auroc_ci_lower"],
            "temporal_auroc_ci_upper": row["auroc_ci_upper"],
            "temporal_auprc_ci_lower": row["auprc_ci_lower"],
            "temporal_auprc_ci_upper": row["auprc_ci_upper"],
        })
    comp = pd.DataFrame(comparison_rows)
    comp.to_csv(V9_DIR / "model_v9_temporal_vs_random_selected_model_comparison.csv", index=False)
    split.to_csv(V9_DIR / "audit_v9_temporal_split_audit.csv", index=False)

    use_theme()
    fig, ax = plt.subplots(figsize=(9, 5.6), dpi=180)
    x = np.arange(len(comp))
    width = 0.34
    ax.bar(x - width / 2, comp["random_split_auroc"], width, label="Random held-out", color="#A3BEFA")
    ax.bar(x + width / 2, comp["temporal_auroc"], width, label="Temporal validation", color="#A3D576")
    ax.errorbar(
        x + width / 2, comp["temporal_auroc"],
        yerr=[comp["temporal_auroc"] - comp["temporal_auroc_ci_lower"], comp["temporal_auroc_ci_upper"] - comp["temporal_auroc"]],
        fmt="none", ecolor="#4F7A39", capsize=3, linewidth=1,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(v)} h" for v in comp["landmark_hours"]])
    ax.set_ylim(0.60, 0.82)
    ax.set_ylabel("AUROC")
    ax.legend(frameon=False, loc="upper left")
    add_header(fig, ax, "Temporal validation versus random held-out discrimination", "Selected model at each landmark; temporal validation uses later first-ICU years.")
    fig.savefig(V9_DIR / "figure_v9_temporal_vs_random_auroc.png", bbox_inches="tight", facecolor=TOKENS["surface"])
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.6), dpi=180)
    for model, group in temporal.groupby("model"):
        color, linestyle = MODEL_STYLES[model]
        ax.plot(group["landmark_hours"], group["auroc"], marker="o", color=color, linestyle=linestyle, label=model)
    ax.set_xticks(LANDMARKS)
    ax.set_xlabel("Prediction landmark")
    ax.set_ylabel("Temporal validation AUROC")
    ax.set_ylim(0.65, 0.80)
    ax.legend(frameon=False, loc="lower right")
    add_header(fig, ax, "Temporal validation AUROC across all model families", "Later-year held-out validation; lines connect model families across landmarks.")
    fig.savefig(V9_DIR / "figure_v9_temporal_auroc_all_models.png", bbox_inches="tight", facecolor=TOKENS["surface"])
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.6), dpi=180)
    ax.axhline(1, color=TOKENS["ink"], linestyle=":", linewidth=1)
    for model, group in temporal.groupby("model"):
        color, linestyle = MODEL_STYLES[model]
        ax.plot(group["landmark_hours"], group["calibration_slope"], marker="o", color=color, linestyle=linestyle, label=model)
    ax.set_xticks(LANDMARKS)
    ax.set_xlabel("Prediction landmark")
    ax.set_ylabel("Calibration slope")
    ax.set_ylim(0.65, 1.40)
    ax.legend(frameon=False, loc="upper right")
    add_header(fig, ax, "Temporal validation calibration slopes", "Ideal slope is 1; slopes below 1 indicate over-extreme predictions.")
    fig.savefig(V9_DIR / "figure_v9_temporal_calibration_slope.png", bbox_inches="tight", facecolor=TOKENS["surface"])
    plt.close(fig)

    brief = ["# v9 temporal validation figure package", ""]
    brief.append("Temporal split: subjects with first ICU year <2176 were used for development; subjects with first ICU year >=2176 were used for validation.")
    brief.append("")
    brief.append("## Selected-model temporal validation")
    brief.append("")
    brief.append("| Landmark | Model | Random AUROC | Temporal AUROC | Difference | Temporal AUPRC | Temporal Brier |")
    brief.append("|---:|---|---:|---:|---:|---:|---:|")
    for _, r in comp.iterrows():
        brief.append(f"| {int(r.landmark_hours)} h | {r.selected_model} | {r.random_split_auroc:.3f} | {r.temporal_auroc:.3f} | {r.delta_temporal_minus_random_auroc:+.3f} | {r.temporal_auprc:.3f} | {r.temporal_brier:.3f} |")
    brief.append("")
    brief.append("Interpretation: temporal validation was broadly similar to random held-out validation, supporting internal robustness but not replacing external validation.")
    (V9_DIR / "audit_v9_results_brief.md").write_text("\n".join(brief), encoding="utf-8")


def simplified_columns(data: pd.DataFrame, landmark: int) -> list[str]:
    cols = [c for c in BASE_SIMPLIFIED if c in data.columns]
    if landmark in {6, 24}:
        for lab in EARLY_LABS:
            for suffix in ["last", "min", "max"]:
                c = f"lab_0_{landmark}h_{lab}_{suffix}"
                if c in data.columns:
                    cols.append(c)
        for vital in EARLY_VITALS:
            for suffix in ["last", "min", "max"]:
                c = f"vital_0_{landmark}h_{vital}_{suffix}"
                if c in data.columns:
                    cols.append(c)
    return list(dict.fromkeys(cols))


def evaluate(y: np.ndarray, p: np.ndarray, subjects: np.ndarray, seed: int) -> dict[str, float]:
    out = {
        "auroc": roc_auc_score(y, p),
        "auprc": average_precision_score(y, p),
        "brier_score": brier_score_loss(y, p),
    }
    out.update(bootstrap_ci(y, p, subjects, BOOTSTRAPS, seed))
    return out


def paired_delta(y: np.ndarray, p_full: np.ndarray, p_simple: np.ndarray, subjects: np.ndarray, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    unique = np.unique(subjects)
    lookup = {s: np.flatnonzero(subjects == s) for s in unique}
    dist = {"delta_auroc": [], "delta_auprc": [], "delta_brier_score": []}
    attempts = 0
    while len(dist["delta_auroc"]) < BOOTSTRAPS and attempts < BOOTSTRAPS * 3:
        attempts += 1
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([lookup[s] for s in sampled])
        if len(np.unique(y[idx])) < 2:
            continue
        dist["delta_auroc"].append(roc_auc_score(y[idx], p_simple[idx]) - roc_auc_score(y[idx], p_full[idx]))
        dist["delta_auprc"].append(average_precision_score(y[idx], p_simple[idx]) - average_precision_score(y[idx], p_full[idx]))
        dist["delta_brier_score"].append(brier_score_loss(y[idx], p_simple[idx]) - brier_score_loss(y[idx], p_full[idx]))
    out = {"paired_bootstrap_successful_n": len(dist["delta_auroc"])}
    for k, values in dist.items():
        out[k] = float(np.mean(values))
        out[f"{k}_ci_lower"] = float(np.quantile(values, 0.025))
        out[f"{k}_ci_upper"] = float(np.quantile(values, 0.975))
    return out


def build_v10_simplified_model() -> None:
    V10_DIR.mkdir(parents=True, exist_ok=True)
    datasets = {lm: load_data(lm) for lm in LANDMARKS}
    train_subjects, test_subjects, _ = choose_grouped_split(datasets[0])
    perf_rows = []
    delta_rows = []
    dict_rows = []

    for lm, data in datasets.items():
        simple_cols = simplified_columns(data, lm)
        dict_rows.append({"landmark_hours": lm, "simplified_predictor_n": len(simple_cols), "predictors": "; ".join(simple_cols)})
        train_mask = data["subject_id"].astype(int).isin(train_subjects)
        test_mask = data["subject_id"].astype(int).isin(test_subjects)
        y_train = data.loc[train_mask, OUTCOME].to_numpy(dtype=int)
        y_test = data.loc[test_mask, OUTCOME].to_numpy(dtype=int)
        subjects_test = data.loc[test_mask, "subject_id"].to_numpy(dtype=int)
        predictions = {}
        for variant, cols in [
            ("full", [c for c in data.columns if c not in {*METADATA, OUTCOME}]),
            ("simplified", simple_cols),
        ]:
            model_data = data[[*METADATA, OUTCOME, *cols]].copy()
            continuous, binary, categorical = identify_types(model_data)
            x_train = model_data.loc[train_mask, cols]
            x_test = model_data.loc[test_mask, cols]
            for model_name, pipe in model_definitions(continuous, binary, categorical).items():
                print(f"Training {variant} {model_name} at {lm} h...", flush=True)
                pipe.fit(x_train, y_train)
                p = pipe.predict_proba(x_test)[:, 1]
                predictions[(variant, model_name)] = p
                perf_rows.append({
                    "landmark_hours": lm,
                    "variant": variant,
                    "model": model_name,
                    "predictor_n": len(cols),
                    "test_n": len(y_test),
                    "test_event_rate": float(y_test.mean()),
                    **evaluate(y_test, p, subjects_test, RANDOM_STATE + 10000 + lm * 100 + zlib.crc32(f"{variant}|{model_name}".encode()) % 100),
                })
        pred_out = data.loc[test_mask, ["subject_id", "hadm_id", "stay_id", "landmark_hours"]].copy()
        pred_out["y_true"] = y_test
        for model_name in MODEL_STYLES:
            full = predictions[("full", model_name)]
            simple = predictions[("simplified", model_name)]
            pred_out[f"prob_full_{model_name.lower().replace(' ', '_')}"] = full
            pred_out[f"prob_simplified_{model_name.lower().replace(' ', '_')}"] = simple
            delta_rows.append({
                "landmark_hours": lm,
                "model": model_name,
                **paired_delta(y_test, full, simple, subjects_test, RANDOM_STATE + 11000 + lm * 100 + zlib.crc32(model_name.encode()) % 100),
            })
        pred_out.to_csv(V10_DIR / f"model_v10_{lm}h_full_vs_simplified_predictions.csv", index=False)

    perf = pd.DataFrame(perf_rows)
    delta = pd.DataFrame(delta_rows)
    dictionary = pd.DataFrame(dict_rows)
    perf.to_csv(V10_DIR / "model_v10_full_vs_simplified_performance.csv", index=False)
    delta.to_csv(V10_DIR / "model_v10_full_vs_simplified_paired_delta.csv", index=False)
    dictionary.to_csv(V10_DIR / "audit_v10_simplified_predictor_set.csv", index=False)

    use_theme()
    fig, ax = plt.subplots(figsize=(9.5, 6.0), dpi=180)
    selected = delta[delta["model"].isin(["Logistic Regression", "XGBoost"])].copy()
    selected["label"] = selected["landmark_hours"].astype(str) + "h " + selected["model"]
    y = np.arange(len(selected))
    ax.axvline(0, color=TOKENS["ink"], linestyle=":", linewidth=1)
    ax.errorbar(
        selected["delta_auroc"], y,
        xerr=[selected["delta_auroc"] - selected["delta_auroc_ci_lower"], selected["delta_auroc_ci_upper"] - selected["delta_auroc"]],
        fmt="o", color="#2E74B5", ecolor="#8AAED6", capsize=3,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(selected["label"])
    ax.set_xlabel("AUROC difference (simplified - full)")
    add_header(fig, ax, "Simplified clinical model versus full model", "Positive values favor simplified predictors; same subject-grouped split and model settings.")
    fig.savefig(V10_DIR / "figure_v10_simplified_minus_full_auroc_delta.png", bbox_inches="tight", facecolor=TOKENS["surface"])
    plt.close(fig)

    lines = ["# v10 simplified clinical model results", ""]
    lines.append("Simplified predictors were prespecified clinical variables, baseline kidney-function variables, selected pre-index labs, and landmark-appropriate early labs/vitals.")
    lines.append("")
    lines.append("## Paired differences")
    lines.append("")
    lines.append("| Landmark | Model | Delta AUROC | 95% CI | Delta AUPRC | Delta Brier |")
    lines.append("|---:|---|---:|---:|---:|---:|")
    for _, r in delta.iterrows():
        lines.append(f"| {int(r.landmark_hours)} h | {r.model} | {r.delta_auroc:+.4f} | {r.delta_auroc_ci_lower:+.4f} to {r.delta_auroc_ci_upper:+.4f} | {r.delta_auprc:+.4f} | {r.delta_brier_score:+.4f} |")
    lines.append("")
    lines.append("Interpretation: simplified models should be judged by whether AUROC/AUPRC loss is small enough to justify easier clinical implementation.")
    (V10_DIR / "audit_v10_results_brief.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    build_v9_temporal_figures()
    build_v10_simplified_model()
    print(f"Wrote v9 outputs to {V9_DIR}")
    print(f"Wrote v10 outputs to {V10_DIR}")


if __name__ == "__main__":
    main()
