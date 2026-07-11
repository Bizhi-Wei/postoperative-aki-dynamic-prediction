"""Rigorous missingness and measurement-intensity audit for dynamic AKI models.

This script does not change the clinical prediction models. It audits:

1. predictor-level missingness in the modeling-ready landmark tables;
2. feature-level measurement coverage from the underlying time-aligned lab/vital caches;
3. measurement intensity by landmark and outcome status;
4. whether missingness/measurement intensity appears outcome-informative.

The goal is to document EHR measurement-process bias for manuscript review.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELING_DIR = PROJECT_ROOT / "outputs" / "modeling_v4_1"
CACHE_DIR = PROJECT_ROOT / "outputs" / "cache_v4"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "modeling_v11_missingness_measurement_intensity"

LANDMARKS = [0, 6, 24]
OUTCOME = "outcome_aki_after_landmark_to_7d"
ID_COLUMNS = {"subject_id", "hadm_id", "stay_id", "landmark_hours"}

LAB_CACHE = CACHE_DIR / "aligned_dynamic_labs_pre24_to_post24.csv.gz"
VITAL_CACHE = CACHE_DIR / "aligned_dynamic_vitals_post24.csv.gz"


@dataclass(frozen=True)
class Window:
    landmark: int
    source: str
    label: str
    lower: float
    upper: float


WINDOWS = [
    Window(0, "lab", "lab_pre24h", -24.0, 0.0),
    Window(6, "lab", "lab_0_6h", 0.0, 6.0),
    Window(6, "vital", "vital_0_6h", 0.0, 6.0),
    Window(24, "lab", "lab_0_24h", 0.0, 24.0),
    Window(24, "vital", "vital_0_24h", 0.0, 24.0),
]


def bool_mask(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).astype(int).astype(bool)
    return series.astype("string").str.strip().str.lower().isin(["true", "1", "yes"])


def iqr_text(values: pd.Series) -> str:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return ""
    return f"{numeric.median():.2f} ({numeric.quantile(0.25):.2f}-{numeric.quantile(0.75):.2f})"


def safe_mannwhitney(event_values: pd.Series, nonevent_values: pd.Series) -> float:
    try:
        from scipy.stats import mannwhitneyu

        x = pd.to_numeric(event_values, errors="coerce").dropna()
        y = pd.to_numeric(nonevent_values, errors="coerce").dropna()
        if len(x) == 0 or len(y) == 0:
            return np.nan
        return float(mannwhitneyu(x, y, alternative="two-sided").pvalue)
    except Exception:
        return np.nan


def standardized_mean_difference(x: pd.Series, y: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    y = pd.to_numeric(y, errors="coerce").dropna()
    if len(x) < 2 or len(y) < 2:
        return np.nan
    pooled = np.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2)
    if pooled == 0 or np.isnan(pooled):
        return 0.0
    return float((x.mean() - y.mean()) / pooled)


def load_modeling(landmark: int) -> pd.DataFrame:
    path = MODELING_DIR / f"modeling_v4_1_{landmark}h.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    data = pd.read_csv(path, low_memory=False)
    required = {"subject_id", "stay_id", OUTCOME}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"{path} is missing {sorted(missing)}")
    data[OUTCOME] = bool_mask(data[OUTCOME])
    return data


def predictor_columns(data: pd.DataFrame) -> list[str]:
    return [c for c in data.columns if c not in ID_COLUMNS and c != OUTCOME]


def variable_family(column: str) -> str:
    if column.startswith("lab_pre24h_"):
        return "pre-index laboratory"
    if re.match(r"lab_0_\d+h_", column):
        return "post-index laboratory"
    if re.match(r"vital_0_\d+h_", column):
        return "post-index vital sign"
    if column.startswith("baseline_scr") or column.startswith("baseline_to_icu"):
        return "baseline kidney function"
    if column.endswith("_surgery") or column in {"n_qualifying_codes", "days_from_procedure_to_icu"}:
        return "surgical phenotype"
    if column in {"first_careunit", "gender", "anchor_age", "race", "admission_type", "insurance", "marital_status"}:
        return "demographics/admission"
    if column in {
        "chf", "hypertension", "dm", "dm_comp", "ckd", "copd", "liver", "cancer",
        "pvd", "stroke", "mi", "obesity", "anemia", "charlson_score",
    }:
        return "comorbidity"
    return "other"


def feature_from_column(column: str) -> str:
    prefixes = ["lab_pre24h_", "lab_0_6h_", "lab_0_24h_", "vital_0_6h_", "vital_0_24h_"]
    suffixes = ["_min", "_max", "_last", "_count"]
    value = column
    for prefix in prefixes:
        if value.startswith(prefix):
            value = value[len(prefix) :]
            break
    for suffix in suffixes:
        if value.endswith(suffix):
            value = value[: -len(suffix)]
            break
    return value


def predictor_missingness_tables(modeling: dict[int, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overall_rows = []
    by_outcome_rows = []
    family_rows = []

    for landmark, data in modeling.items():
        y = data[OUTCOME].astype(bool)
        for col in predictor_columns(data):
            missing = data[col].isna()
            observed = ~missing
            event_obs = observed[y]
            nonevent_obs = observed[~y]
            event_observed_pct = float(event_obs.mean() * 100) if len(event_obs) else np.nan
            nonevent_observed_pct = float(nonevent_obs.mean() * 100) if len(nonevent_obs) else np.nan
            overall_rows.append(
                {
                    "landmark_hours": landmark,
                    "variable": col,
                    "family": variable_family(col),
                    "feature": feature_from_column(col),
                    "n": len(data),
                    "n_missing": int(missing.sum()),
                    "missing_percent": round(float(missing.mean() * 100), 3),
                    "n_observed": int(observed.sum()),
                    "observed_percent": round(float(observed.mean() * 100), 3),
                }
            )
            by_outcome_rows.append(
                {
                    "landmark_hours": landmark,
                    "variable": col,
                    "family": variable_family(col),
                    "feature": feature_from_column(col),
                    "event_n": int(y.sum()),
                    "nonevent_n": int((~y).sum()),
                    "event_observed_percent": round(event_observed_pct, 3),
                    "nonevent_observed_percent": round(nonevent_observed_pct, 3),
                    "event_minus_nonevent_observed_percent": round(event_observed_pct - nonevent_observed_pct, 3),
                    "absolute_difference_percent": round(abs(event_observed_pct - nonevent_observed_pct), 3),
                }
            )

        family_summary = (
            pd.DataFrame(overall_rows)
            .query("landmark_hours == @landmark")
            .groupby(["landmark_hours", "family"], as_index=False)
            .agg(
                variables=("variable", "count"),
                median_missing_percent=("missing_percent", "median"),
                max_missing_percent=("missing_percent", "max"),
                variables_missing_gt40pct=("missing_percent", lambda s: int((s > 40).sum())),
            )
        )
        family_rows.extend(family_summary.to_dict("records"))

    return pd.DataFrame(overall_rows), pd.DataFrame(by_outcome_rows), pd.DataFrame(family_rows)


def feature_level_coverage(overall: pd.DataFrame) -> pd.DataFrame:
    dynamic = overall.loc[
        overall["family"].isin(["pre-index laboratory", "post-index laboratory", "post-index vital sign"])
    ].copy()
    if dynamic.empty:
        return pd.DataFrame()
    return (
        dynamic.groupby(["landmark_hours", "family", "feature"], as_index=False)
        .agg(
            derived_variables=("variable", "count"),
            min_missing_percent=("missing_percent", "min"),
            median_missing_percent=("missing_percent", "median"),
            max_missing_percent=("missing_percent", "max"),
            observed_percent=("observed_percent", "median"),
        )
        .sort_values(["landmark_hours", "family", "median_missing_percent", "feature"])
    )


def load_measurement_cache() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not LAB_CACHE.exists():
        raise FileNotFoundError(LAB_CACHE)
    if not VITAL_CACHE.exists():
        raise FileNotFoundError(VITAL_CACHE)
    labs = pd.read_csv(LAB_CACHE, usecols=["stay_id", "feature", "hours_from_icu"], low_memory=False)
    vitals = pd.read_csv(VITAL_CACHE, usecols=["stay_id", "feature", "hours_from_icu"], low_memory=False)
    labs["source"] = "lab"
    vitals["source"] = "vital"
    return labs, vitals


def measurement_counts_for_window(records: pd.DataFrame, window: Window, stay_ids: pd.Series) -> pd.DataFrame:
    selected = records.loc[
        records["hours_from_icu"].between(window.lower, window.upper, inclusive="both")
    ].copy()
    index = pd.DataFrame({"stay_id": stay_ids.astype(int).unique()})
    if selected.empty:
        result = index.copy()
        result[f"{window.label}_total_count"] = 0
        result[f"{window.label}_distinct_features"] = 0
        return result

    total = selected.groupby("stay_id").size().rename(f"{window.label}_total_count")
    distinct = selected.groupby("stay_id")["feature"].nunique().rename(f"{window.label}_distinct_features")
    result = index.merge(total, on="stay_id", how="left").merge(distinct, on="stay_id", how="left")
    count_cols = [f"{window.label}_total_count", f"{window.label}_distinct_features"]
    result[count_cols] = result[count_cols].fillna(0).astype(int)
    return result


def feature_count_for_window(records: pd.DataFrame, window: Window, stay_ids: pd.Series) -> pd.DataFrame:
    selected = records.loc[
        records["hours_from_icu"].between(window.lower, window.upper, inclusive="both")
    ].copy()
    index = pd.DataFrame({"stay_id": stay_ids.astype(int).unique()})
    if selected.empty:
        return pd.DataFrame()
    counts = selected.groupby(["stay_id", "feature"]).size().rename("measurement_count").reset_index()
    counts = index.merge(counts, on="stay_id", how="left")
    counts["measurement_count"] = counts["measurement_count"].fillna(0).astype(int)
    return counts


def measurement_intensity_tables(
    modeling: dict[int, pd.DataFrame], labs: pd.DataFrame, vitals: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, pd.DataFrame]]:
    summary_rows = []
    by_feature_rows = []
    landmark_count_tables: dict[int, pd.DataFrame] = {}

    for landmark, data in modeling.items():
        merged = data[["stay_id", OUTCOME]].copy()
        relevant = [w for w in WINDOWS if w.landmark == landmark]
        for window in relevant:
            records = labs if window.source == "lab" else vitals
            window_counts = measurement_counts_for_window(records, window, merged["stay_id"])
            merged = merged.merge(window_counts, on="stay_id", how="left")

            feature_counts = feature_count_for_window(records, window, merged["stay_id"])
            if not feature_counts.empty:
                feature_counts[OUTCOME] = feature_counts["stay_id"].map(
                    merged.set_index("stay_id")[OUTCOME].to_dict()
                )
                for feature, feature_data in feature_counts.groupby("feature"):
                    y = feature_data[OUTCOME].astype(bool)
                    event = feature_data.loc[y, "measurement_count"]
                    nonevent = feature_data.loc[~y, "measurement_count"]
                    by_feature_rows.append(
                        {
                            "landmark_hours": landmark,
                            "window": window.label,
                            "source": window.source,
                            "feature": feature,
                            "event_median_iqr": iqr_text(event),
                            "nonevent_median_iqr": iqr_text(nonevent),
                            "event_mean_count": round(float(event.mean()), 3) if len(event) else np.nan,
                            "nonevent_mean_count": round(float(nonevent.mean()), 3) if len(nonevent) else np.nan,
                            "event_minus_nonevent_mean_count": round(float(event.mean() - nonevent.mean()), 3)
                            if len(event) and len(nonevent)
                            else np.nan,
                            "smd_event_vs_nonevent": round(standardized_mean_difference(event, nonevent), 4),
                            "mannwhitney_p": safe_mannwhitney(event, nonevent),
                        }
                    )

        count_cols = [c for c in merged.columns if c.endswith("_total_count") or c.endswith("_distinct_features")]
        merged[count_cols] = merged[count_cols].fillna(0).astype(int)
        merged["all_measurement_total_count"] = merged[[c for c in count_cols if c.endswith("_total_count")]].sum(axis=1)
        merged["all_measurement_distinct_features"] = merged[[c for c in count_cols if c.endswith("_distinct_features")]].sum(axis=1)
        landmark_count_tables[landmark] = merged

        y = merged[OUTCOME].astype(bool)
        for col in [*count_cols, "all_measurement_total_count", "all_measurement_distinct_features"]:
            event = merged.loc[y, col]
            nonevent = merged.loc[~y, col]
            summary_rows.append(
                {
                    "landmark_hours": landmark,
                    "measurement_metric": col,
                    "overall_median_iqr": iqr_text(merged[col]),
                    "event_median_iqr": iqr_text(event),
                    "nonevent_median_iqr": iqr_text(nonevent),
                    "overall_mean": round(float(merged[col].mean()), 3),
                    "event_mean": round(float(event.mean()), 3) if len(event) else np.nan,
                    "nonevent_mean": round(float(nonevent.mean()), 3) if len(nonevent) else np.nan,
                    "event_minus_nonevent_mean": round(float(event.mean() - nonevent.mean()), 3)
                    if len(event) and len(nonevent)
                    else np.nan,
                    "smd_event_vs_nonevent": round(standardized_mean_difference(event, nonevent), 4),
                    "mannwhitney_p": safe_mannwhitney(event, nonevent),
                }
            )

    return pd.DataFrame(summary_rows), pd.DataFrame(by_feature_rows), landmark_count_tables


def informative_missingness_signal(
    modeling: dict[int, pd.DataFrame], count_tables: dict[int, pd.DataFrame]
) -> pd.DataFrame:
    rows = []
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import average_precision_score, roc_auc_score
        from sklearn.model_selection import GroupKFold
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:
        return pd.DataFrame(
            [
                {
                    "landmark_hours": landmark,
                    "signal_set": "not_run",
                    "reason": f"sklearn unavailable: {exc}",
                }
                for landmark in modeling
            ]
        )

    for landmark, data in modeling.items():
        y = data[OUTCOME].astype(int).to_numpy()
        groups = data["subject_id"].to_numpy()
        predictors = predictor_columns(data)
        missing_indicators = data[predictors].isna().astype(int)
        missing_indicators.columns = [f"missing__{c}" for c in missing_indicators.columns]

        count_data = count_tables[landmark].set_index("stay_id")
        counts = data[["stay_id"]].merge(
            count_data.drop(columns=[OUTCOME], errors="ignore"), on="stay_id", how="left"
        )
        counts = counts.drop(columns=["stay_id"])
        counts = counts.fillna(0)

        candidate_sets = {
            "missingness_indicators_only": missing_indicators,
            "measurement_counts_only": counts,
            "missingness_plus_measurement_counts": pd.concat([missing_indicators, counts], axis=1),
        }

        for label, x in candidate_sets.items():
            if x.shape[1] == 0 or len(np.unique(y)) < 2:
                rows.append(
                    {
                        "landmark_hours": landmark,
                        "signal_set": label,
                        "n_features": x.shape[1],
                        "cv_auroc": np.nan,
                        "cv_auprc": np.nan,
                        "note": "not enough features or outcome classes",
                    }
                )
                continue

            unique_groups = np.unique(groups)
            n_splits = min(5, len(unique_groups))
            if n_splits < 2:
                rows.append(
                    {
                        "landmark_hours": landmark,
                        "signal_set": label,
                        "n_features": x.shape[1],
                        "cv_auroc": np.nan,
                        "cv_auprc": np.nan,
                        "note": "not enough groups",
                    }
                )
                continue

            preds = np.full(len(y), np.nan)
            splitter = GroupKFold(n_splits=n_splits)
            for train_idx, test_idx in splitter.split(x, y, groups):
                if len(np.unique(y[train_idx])) < 2:
                    continue
                pipe = Pipeline(
                    steps=[
                        (
                            "prep",
                            ColumnTransformer(
                                transformers=[
                                    (
                                        "numeric",
                                        Pipeline(
                                            steps=[
                                                ("imputer", SimpleImputer(strategy="median")),
                                                ("scaler", StandardScaler(with_mean=False)),
                                            ]
                                        ),
                                        list(x.columns),
                                    )
                                ],
                                remainder="drop",
                            ),
                        ),
                        (
                            "model",
                            LogisticRegression(max_iter=1000, solver="liblinear", class_weight="balanced"),
                        ),
                    ]
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pipe.fit(x.iloc[train_idx], y[train_idx])
                preds[test_idx] = pipe.predict_proba(x.iloc[test_idx])[:, 1]
            valid = ~np.isnan(preds)
            rows.append(
                {
                    "landmark_hours": landmark,
                    "signal_set": label,
                    "n_features": x.shape[1],
                    "cv_auroc": round(float(roc_auc_score(y[valid], preds[valid])), 4)
                    if valid.any() and len(np.unique(y[valid])) > 1
                    else np.nan,
                    "cv_auprc": round(float(average_precision_score(y[valid], preds[valid])), 4)
                    if valid.any() and len(np.unique(y[valid])) > 1
                    else np.nan,
                    "note": "grouped cross-validated diagnostic audit; not a clinical prediction model",
                }
            )
    return pd.DataFrame(rows)


def make_figures(
    overall: pd.DataFrame,
    by_outcome: pd.DataFrame,
    intensity: pd.DataFrame,
    signal: pd.DataFrame,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    family_heat = (
        overall.groupby(["landmark_hours", "family"], as_index=False)["missing_percent"]
        .median()
        .pivot(index="family", columns="landmark_hours", values="missing_percent")
        .fillna(0)
    )
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    im = ax.imshow(family_heat.values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=max(1, family_heat.values.max()))
    ax.set_xticks(range(len(family_heat.columns)), labels=[f"{c} h" for c in family_heat.columns])
    ax.set_yticks(range(len(family_heat.index)), labels=family_heat.index)
    ax.set_title("Median predictor missingness by variable family and landmark")
    for i in range(family_heat.shape[0]):
        for j in range(family_heat.shape[1]):
            ax.text(j, i, f"{family_heat.iloc[i, j]:.1f}%", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Median missingness (%)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "figure_v11_missingness_family_heatmap.png", dpi=300)
    plt.close(fig)

    top = by_outcome.sort_values("absolute_difference_percent", ascending=False).head(25).copy()
    top["label"] = top["landmark_hours"].astype(str) + "h: " + top["variable"]
    fig, ax = plt.subplots(figsize=(8.5, max(5, len(top) * 0.22)))
    ax.barh(top["label"], top["event_minus_nonevent_observed_percent"], color="#4c78a8")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.invert_yaxis()
    ax.set_xlabel("Observed % in AKI events minus non-events")
    ax.set_title("Largest outcome-associated differences in predictor observation")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "figure_v11_top_outcome_differential_missingness.png", dpi=300)
    plt.close(fig)

    total = intensity.loc[intensity["measurement_metric"].eq("all_measurement_total_count")].copy()
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    x = np.arange(len(total))
    width = 0.35
    ax.bar(x - width / 2, total["nonevent_mean"], width=width, label="No AKI after landmark", color="#72b7b2")
    ax.bar(x + width / 2, total["event_mean"], width=width, label="AKI after landmark", color="#e45756")
    ax.set_xticks(x, labels=[f"{int(v)} h" for v in total["landmark_hours"]])
    ax.set_ylabel("Mean lab/vital measurement count")
    ax.set_title("Measurement intensity by future AKI status")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "figure_v11_measurement_intensity_by_outcome.png", dpi=300)
    plt.close(fig)

    if {"signal_set", "cv_auroc"}.issubset(signal.columns) and signal["cv_auroc"].notna().any():
        plot_signal = signal.loc[signal["cv_auroc"].notna()].copy()
        fig, ax = plt.subplots(figsize=(7.8, 4.8))
        for label, group in plot_signal.groupby("signal_set"):
            group = group.sort_values("landmark_hours")
            ax.plot(group["landmark_hours"], group["cv_auroc"], marker="o", label=label.replace("_", " "))
        ax.axhline(0.5, color="black", linewidth=0.8, linestyle="--")
        ax.set_xticks(LANDMARKS)
        ax.set_xlabel("Landmark")
        ax.set_ylabel("Grouped CV AUROC")
        ax.set_title("Outcome signal carried by missingness/measurement process only")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "figure_v11_informative_missingness_signal.png", dpi=300)
        plt.close(fig)


def write_readme(
    modeling: dict[int, pd.DataFrame],
    family: pd.DataFrame,
    by_outcome: pd.DataFrame,
    intensity: pd.DataFrame,
    signal: pd.DataFrame,
) -> None:
    sample_lines = []
    for landmark, data in modeling.items():
        y = data[OUTCOME].astype(bool)
        sample_lines.append(
            f"- {landmark} h: n={len(data):,}, events={int(y.sum()):,} ({y.mean() * 100:.1f}%), "
            f"candidate predictors={len(predictor_columns(data)):,}."
        )

    high_missing = family.sort_values(["landmark_hours", "median_missing_percent"], ascending=[True, False])
    high_missing_text = "\n".join(
        f"- {int(row.landmark_hours)} h / {row.family}: median missing {row.median_missing_percent:.1f}%, "
        f"max {row.max_missing_percent:.1f}%, variables >40% missing={int(row.variables_missing_gt40pct)}"
        for row in high_missing.itertuples()
    )

    differential = by_outcome.sort_values("absolute_difference_percent", ascending=False).head(8)
    differential_text = "\n".join(
        f"- {int(row.landmark_hours)} h `{row.variable}`: observed in events {row.event_observed_percent:.1f}% vs "
        f"non-events {row.nonevent_observed_percent:.1f}% "
        f"(difference {row.event_minus_nonevent_observed_percent:+.1f} percentage points)."
        for row in differential.itertuples()
    )

    total_intensity = intensity.loc[intensity["measurement_metric"].eq("all_measurement_total_count")]
    intensity_text = "\n".join(
        f"- {int(row.landmark_hours)} h: total measurements median {row.overall_median_iqr}; "
        f"events {row.event_median_iqr} vs non-events {row.nonevent_median_iqr}; "
        f"SMD={row.smd_event_vs_nonevent:.3f}."
        for row in total_intensity.itertuples()
    )

    signal_text = "Not run."
    if {"signal_set", "cv_auroc"}.issubset(signal.columns):
        rows = []
        for row in signal.sort_values(["landmark_hours", "signal_set"]).itertuples():
            if pd.notna(getattr(row, "cv_auroc", np.nan)):
                rows.append(
                    f"- {int(row.landmark_hours)} h / {row.signal_set}: "
                    f"AUROC={row.cv_auroc:.3f}, AUPRC={row.cv_auprc:.3f}."
                )
        if rows:
            signal_text = "\n".join(rows)

    content = f"""# v11 Missingness and measurement-intensity audit

## Scope

This audit evaluates missingness and measurement intensity in the dynamic postoperative AKI prediction datasets. It does not modify the v5-v10 model results and does not select new clinical predictors.

## Analytic samples

{chr(10).join(sample_lines)}

## Main missingness patterns

{high_missing_text}

## Largest outcome-associated observation differences

Positive values mean the variable was observed more often among patients who subsequently developed AKI after the landmark.

{differential_text}

## Measurement intensity

Underlying time-aligned laboratory and vital-sign caches were used to count actual measurements, not merely derived feature availability.

{intensity_text}

## Diagnostic informative-missingness signal

The following grouped cross-validated logistic regressions used only missingness indicators and/or measurement counts. These are diagnostic measurement-process audits, not clinical prediction models.

{signal_text}

## Recommended manuscript wording

Missingness was not completely random. Laboratory and vital-sign observation patterns varied by landmark and by subsequent AKI status, consistent with intensity-of-care and clinician-ordering effects in routinely collected ICU data. The primary modeling pipeline therefore used explicit imputation within the training data and internally validated performance by subject-level random and temporal splits. Measurement-process analyses should be presented as an EHR bias audit rather than as causal evidence.

## Output files

- `audit_v11_predictor_missingness_overall.csv`
- `audit_v11_missingness_by_outcome.csv`
- `audit_v11_missingness_by_family.csv`
- `audit_v11_feature_level_coverage.csv`
- `audit_v11_measurement_intensity_by_outcome.csv`
- `audit_v11_measurement_intensity_by_feature.csv`
- `audit_v11_informative_missingness_signal.csv`
- `figure_v11_missingness_family_heatmap.png`
- `figure_v11_top_outcome_differential_missingness.png`
- `figure_v11_measurement_intensity_by_outcome.png`
- `figure_v11_informative_missingness_signal.png`
"""
    (OUTPUT_DIR / "audit_v11_results_brief.md").write_text(content, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    modeling = {landmark: load_modeling(landmark) for landmark in LANDMARKS}

    overall, by_outcome, family = predictor_missingness_tables(modeling)
    coverage = feature_level_coverage(overall)
    labs, vitals = load_measurement_cache()
    intensity, intensity_by_feature, count_tables = measurement_intensity_tables(modeling, labs, vitals)
    signal = informative_missingness_signal(modeling, count_tables)

    overall.to_csv(OUTPUT_DIR / "audit_v11_predictor_missingness_overall.csv", index=False)
    by_outcome.to_csv(OUTPUT_DIR / "audit_v11_missingness_by_outcome.csv", index=False)
    family.to_csv(OUTPUT_DIR / "audit_v11_missingness_by_family.csv", index=False)
    coverage.to_csv(OUTPUT_DIR / "audit_v11_feature_level_coverage.csv", index=False)
    intensity.to_csv(OUTPUT_DIR / "audit_v11_measurement_intensity_by_outcome.csv", index=False)
    intensity_by_feature.to_csv(OUTPUT_DIR / "audit_v11_measurement_intensity_by_feature.csv", index=False)
    signal.to_csv(OUTPUT_DIR / "audit_v11_informative_missingness_signal.csv", index=False)

    for landmark, table in count_tables.items():
        table.to_csv(OUTPUT_DIR / f"audit_v11_measurement_counts_{landmark}h.csv", index=False)

    make_figures(overall, by_outcome, intensity, signal)
    write_readme(modeling, family, by_outcome, intensity, signal)
    print(f"v11 missingness / measurement-intensity audit complete: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
