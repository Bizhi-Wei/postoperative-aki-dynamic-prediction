"""Perioperative predictor sensitivity analysis.

Question: do charted OR/PACU proxy variables improve 0 h or 6 h prediction of
the primary SCr-only incident AKI outcome?

This analysis keeps the original v4.1 SCr-only modeling outcome and compares:
- base: original v4.1 predictors;
- augmented: original predictors + landmark-appropriate OR/PACU proxy and
  early-exposure variables from v6.

Timing:
- 0 h augmented features: *_preicu_or_at_icu only.
- 6 h augmented features: *_preicu_or_at_icu plus *_0_6h only.
- No *_0_24h features are used here.
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
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
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
FEATURES = PROJECT_ROOT / "outputs" / "v6_urine_output_or_pacu" / "features_v6_or_pacu_early_exposures.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "modeling_v8_perioperative_predictor_sensitivity"
LANDMARKS = [0, 6]
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


def load_features(landmark: int) -> pd.DataFrame:
    feats = pd.read_csv(FEATURES, low_memory=False)
    keep = ["subject_id", "hadm_id", "stay_id"]
    if landmark == 0:
        keep += [c for c in feats.columns if c.endswith("_preicu_or_at_icu")]
    elif landmark == 6:
        keep += [
            c for c in feats.columns
            if c.endswith("_preicu_or_at_icu") or c.endswith("_0_6h")
        ]
    else:
        raise ValueError(landmark)
    out = feats[keep].copy()
    for c in out.columns:
        if c not in {"subject_id", "hadm_id", "stay_id"} and pd.api.types.is_object_dtype(out[c]):
            out[c] = out[c].astype("string").str.lower().isin(["true", "1", "yes"])
    return out


def augmented_dataset(landmark: int) -> tuple[pd.DataFrame, list[str]]:
    base = load_data(landmark)
    feats = load_features(landmark)
    data = base.merge(feats, on=["subject_id", "hadm_id", "stay_id"], how="left", validate="one_to_one")
    added = [c for c in feats.columns if c not in {"subject_id", "hadm_id", "stay_id"}]
    for c in added:
        if pd.api.types.is_bool_dtype(data[c]):
            data[c] = data[c].fillna(False)
        else:
            data[c] = pd.to_numeric(data[c], errors="coerce").fillna(0)
    return data, added


def evaluate(y: np.ndarray, p: np.ndarray, subjects: np.ndarray, seed: int) -> dict[str, float]:
    row = {
        "auroc": roc_auc_score(y, p),
        "auprc": average_precision_score(y, p),
        "brier_score": brier_score_loss(y, p),
    }
    ci = bootstrap_ci(y, p, subjects, BOOTSTRAPS, seed)
    row.update(ci)
    return row


def paired_delta_ci(y: np.ndarray, p_base: np.ndarray, p_aug: np.ndarray, subjects: np.ndarray, seed: int) -> dict[str, float]:
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
        dist["delta_auroc"].append(roc_auc_score(y[idx], p_aug[idx]) - roc_auc_score(y[idx], p_base[idx]))
        dist["delta_auprc"].append(average_precision_score(y[idx], p_aug[idx]) - average_precision_score(y[idx], p_base[idx]))
        dist["delta_brier_score"].append(brier_score_loss(y[idx], p_aug[idx]) - brier_score_loss(y[idx], p_base[idx]))
    out = {"paired_bootstrap_successful_n": len(dist["delta_auroc"])}
    for k, values in dist.items():
        out[k] = float(np.mean(values))
        out[f"{k}_ci_lower"] = float(np.quantile(values, 0.025))
        out[f"{k}_ci_upper"] = float(np.quantile(values, 0.975))
    return out


def use_theme() -> None:
    sns.set_theme(style="whitegrid", rc={
        "figure.facecolor": TOKENS["surface"], "axes.facecolor": TOKENS["panel"],
        "axes.edgecolor": TOKENS["axis"], "axes.labelcolor": TOKENS["ink"],
        "grid.color": TOKENS["grid"], "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "axes.spines.top": False, "axes.spines.right": False,
    })


def add_header(fig, ax, title: str, subtitle: str) -> None:
    fig.subplots_adjust(top=0.78, left=0.12, right=0.96, bottom=0.15)
    left = ax.get_position().x0
    fig.text(left, 0.97, textwrap.fill(title, 80), ha="left", va="top", fontsize=14, fontweight="semibold", color=TOKENS["ink"])
    fig.text(left, 0.90, textwrap.fill(subtitle, 110), ha="left", va="top", fontsize=9, color=TOKENS["muted"])
    sns.despine(ax=ax)


def plot_delta(delta: pd.DataFrame) -> None:
    use_theme()
    fig, ax = plt.subplots(figsize=(9.5, 5.8), dpi=180)
    data = delta.copy()
    data["label"] = data["landmark_hours"].astype(str) + "h " + data["model"]
    y = np.arange(len(data))
    ax.axvline(0, color=TOKENS["ink"], linestyle=":", linewidth=1)
    ax.errorbar(
        data["delta_auroc"], y,
        xerr=[data["delta_auroc"] - data["delta_auroc_ci_lower"], data["delta_auroc_ci_upper"] - data["delta_auroc"]],
        fmt="o", color="#2E74B5", ecolor="#8AAED6", capsize=3,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(data["label"])
    ax.set_xlabel("AUROC difference (augmented - base)")
    add_header(
        fig, ax,
        "Incremental AUROC from OR/PACU proxy variables",
        "Positive values favor models augmented with landmark-appropriate OR/PACU proxy and early-exposure variables.",
    )
    fig.savefig(OUTPUT_DIR / "figure_v8_perioperative_proxy_auroc_delta.png", bbox_inches="tight", facecolor=TOKENS["surface"])
    plt.close(fig)


def plot_selected_curves(predictions: dict[tuple[int, str], pd.DataFrame]) -> None:
    use_theme()
    for landmark in LANDMARKS:
        fig, ax = plt.subplots(figsize=(8.8, 6.2), dpi=180)
        for model in ["XGBoost", "Logistic Regression"]:
            for variant, style_suffix in [("base", "-"), ("augmented", "--")]:
                key = (landmark, model)
                if key not in predictions:
                    continue
                df = predictions[key]
                y = df["y_true"].to_numpy(dtype=int)
                p = df[f"prob_{variant}"].to_numpy(float)
                fpr, tpr, _ = roc_curve(y, p)
                color = MODEL_STYLES[model][0]
                ax.plot(fpr, tpr, linestyle=style_suffix, color=color, linewidth=1.25, label=f"{model} {variant} ({roc_auc_score(y,p):.3f})")
        ax.plot([0, 1], [0, 1], color=TOKENS["ink"], linestyle=":", linewidth=1)
        ax.set(xlabel="1 - Specificity", ylabel="Sensitivity", xlim=(0, 1), ylim=(0, 1))
        ax.legend(loc="lower right", frameon=False, fontsize=8)
        add_header(fig, ax, f"Base versus OR/PACU-augmented ROC at {landmark} h", "Primary SCr-only AKI outcome; same subject-grouped split and model settings.")
        fig.savefig(OUTPUT_DIR / f"figure_v8_base_vs_augmented_roc_{landmark}h.png", bbox_inches="tight", facecolor=TOKENS["surface"])
        plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base0 = load_data(0)
    train_subjects, test_subjects, split_audit = choose_grouped_split(base0)

    performance_rows = []
    delta_rows = []
    feature_rows = []
    selected_prediction_tables: dict[tuple[int, str], pd.DataFrame] = {}

    for landmark in LANDMARKS:
        base = load_data(landmark)
        aug, added = augmented_dataset(landmark)
        if len(base) != len(aug):
            raise ValueError("Augmented merge changed row count")
        feature_rows.append({
            "landmark_hours": landmark,
            "added_feature_n": len(added),
            "added_features": "; ".join(added),
        })
        train_mask = base["subject_id"].astype(int).isin(train_subjects)
        test_mask = base["subject_id"].astype(int).isin(test_subjects)
        y_train = base.loc[train_mask, OUTCOME].astype(int).to_numpy()
        y_test = base.loc[test_mask, OUTCOME].astype(int).to_numpy()
        subjects_test = base.loc[test_mask, "subject_id"].astype(int).to_numpy()

        variant_predictions: dict[tuple[str, str], np.ndarray] = {}
        for variant, data in [("base", base), ("augmented", aug)]:
            continuous, binary, categorical = identify_types(data)
            predictors = [c for c in data.columns if c not in {*METADATA, OUTCOME}]
            x_train = data.loc[train_mask, predictors]
            x_test = data.loc[test_mask, predictors]
            for model_name, pipe in model_definitions(continuous, binary, categorical).items():
                print(f"Training {variant} {model_name} at {landmark} h...", flush=True)
                pipe.fit(x_train, y_train)
                p_test = pipe.predict_proba(x_test)[:, 1]
                variant_predictions[(variant, model_name)] = p_test
                seed = RANDOM_STATE + 8000 + landmark * 100 + zlib.crc32(f"{variant}|{model_name}".encode()) % 100
                row = {
                    "landmark_hours": landmark,
                    "variant": variant,
                    "model": model_name,
                    "test_n": len(y_test),
                    "test_event_n": int(y_test.sum()),
                    "test_event_rate": float(y_test.mean()),
                    **evaluate(y_test, p_test, subjects_test, seed),
                }
                performance_rows.append(row)

        pred_out = base.loc[test_mask, ["subject_id", "hadm_id", "stay_id", "landmark_hours"]].copy()
        pred_out["y_true"] = y_test
        for model_name in ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM"]:
            p_base = variant_predictions[("base", model_name)]
            p_aug = variant_predictions[("augmented", model_name)]
            pred_out[f"prob_base_{model_name.lower().replace(' ', '_')}"] = p_base
            pred_out[f"prob_augmented_{model_name.lower().replace(' ', '_')}"] = p_aug
            delta = paired_delta_ci(
                y_test, p_base, p_aug, subjects_test,
                seed=RANDOM_STATE + 8100 + landmark * 100 + zlib.crc32(model_name.encode()) % 100,
            )
            delta_rows.append({
                "landmark_hours": landmark,
                "model": model_name,
                **delta,
            })
            if model_name in {"XGBoost", "Logistic Regression"}:
                selected_prediction_tables[(landmark, model_name)] = pd.DataFrame({
                    "y_true": y_test,
                    "prob_base": p_base,
                    "prob_augmented": p_aug,
                })
        pred_out.to_csv(OUTPUT_DIR / f"model_v8_{landmark}h_base_vs_augmented_predictions.csv", index=False)

    performance = pd.DataFrame(performance_rows)
    delta = pd.DataFrame(delta_rows)
    features = pd.DataFrame(feature_rows)
    performance.to_csv(OUTPUT_DIR / "model_v8_perioperative_proxy_performance.csv", index=False)
    delta.to_csv(OUTPUT_DIR / "model_v8_perioperative_proxy_paired_delta.csv", index=False)
    features.to_csv(OUTPUT_DIR / "audit_v8_added_features.csv", index=False)
    plot_delta(delta)
    plot_selected_curves(selected_prediction_tables)

    brief_lines = ["# v8 perioperative predictor sensitivity results", ""]
    brief_lines.append("Primary outcome: original SCr-only incident AKI. Variants compare original v4.1 predictors versus landmark-appropriate OR/PACU proxy augmentation.")
    brief_lines.append("")
    brief_lines.append("## Paired AUROC differences")
    brief_lines.append("")
    brief_lines.append("| Landmark | Model | Delta AUROC | 95% CI | Delta AUPRC | Delta Brier |")
    brief_lines.append("|---:|---|---:|---:|---:|---:|")
    for _, r in delta.iterrows():
        brief_lines.append(f"| {int(r.landmark_hours)} h | {r.model} | {r.delta_auroc:+.4f} | {r.delta_auroc_ci_lower:+.4f} to {r.delta_auroc_ci_upper:+.4f} | {r.delta_auprc:+.4f} | {r.delta_brier_score:+.4f} |")
    brief_lines.append("")
    brief_lines.append("Interpretation guide: positive AUROC/AUPRC differences favor OR/PACU augmentation; negative Brier differences favor OR/PACU augmentation. Very small differences should not be interpreted as clinically meaningful.")
    brief_lines.append("")
    brief_lines.append("Important wording: OR/PACU variables are charted proxy variables, not complete anesthesia records. Early vasoactive variables after ICU admission should be called early ICU exposure, not intraoperative exposure.")
    (OUTPUT_DIR / "audit_v8_results_brief.md").write_text("\n".join(brief_lines), encoding="utf-8")

    readme = f"""# v8 perioperative predictor sensitivity

Question: do charted OR/PACU proxy variables materially improve 0 h or 6 h SCr-only AKI prediction?

Split: same deterministic subject-level split as v5/v5.1, selected on the 0 h cohort.

Timing:

- 0 h augmentation: only `*_preicu_or_at_icu` features.
- 6 h augmentation: `*_preicu_or_at_icu` plus `*_0_6h` features.
- No `*_0_24h` or outcome-derived variables were used.

Outputs:

- `model_v8_perioperative_proxy_performance.csv`
- `model_v8_perioperative_proxy_paired_delta.csv`
- `model_v8_0h_base_vs_augmented_predictions.csv`
- `model_v8_6h_base_vs_augmented_predictions.csv`
- `figure_v8_perioperative_proxy_auroc_delta.png`
- `figure_v8_base_vs_augmented_roc_0h.png`
- `figure_v8_base_vs_augmented_roc_6h.png`
- `audit_v8_added_features.csv`
- `audit_v8_results_brief.md`
"""
    (OUTPUT_DIR / "audit_v8_readme.md").write_text(readme, encoding="utf-8")
    print(delta.to_string(index=False))
    print(f"Wrote v8 outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
