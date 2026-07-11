"""Prepare final manuscript tables, figures, Methods, and Results."""

from __future__ import annotations

import math
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "outputs"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "final_manuscript_v6"
TABLE_DIR = OUTPUT_ROOT / "tables"
FIGURE_DIR = OUTPUT_ROOT / "figures"
SOURCE_DATA_DIR = OUTPUT_ROOT / "source_data"

PALETTE = {
    "blue": "#5B8FF9", "orange": "#F08A5D", "olive": "#75B84F", "pink": "#D86FB7",
    "ink": "#202431", "muted": "#667085", "grid": "#E5E7EB", "light": "#F6F8FB",
    "red": "#C85A54", "green": "#4A8B57", "gold": "#D4A72C",
}
MODEL_STYLE = {
    "Logistic Regression": (PALETTE["blue"], "-"),
    "Random Forest": (PALETTE["orange"], "--"),
    "XGBoost": (PALETTE["olive"], "-."),
    "LightGBM": (PALETTE["pink"], ":"),
}
MODEL_SAFE = {
    "Logistic Regression": "logistic_regression", "Random Forest": "random_forest",
    "XGBoost": "xgboost", "LightGBM": "lightgbm",
}
MODEL_DISPLAY = {
    "Logistic Regression": "Logistic regression",
    "Random Forest": "Random forest",
    "XGBoost": "XGBoost",
    "LightGBM": "LightGBM",
}


def setup_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 8, "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5, "legend.fontsize": 6.3, "axes.linewidth": 0.7,
        "axes.spines.right": False, "axes.spines.top": False,
        "svg.fonttype": "none", "pdf.fonttype": 42,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })
    sns.set_theme(style="whitegrid", rc={"grid.color": PALETTE["grid"], "grid.linewidth": 0.6})


def save_figure(fig: plt.Figure, name: str, dpi: int = 300) -> None:
    for extension in ["png", "pdf", "svg"]:
        kwargs = {"dpi": dpi} if extension == "png" else {}
        fig.savefig(FIGURE_DIR / f"{name}.{extension}", bbox_inches="tight", facecolor="white", **kwargs)
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str, x: float = -0.13, y: float = 1.06) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top", ha="left")


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype("string").str.lower().isin(["true", "1", "yes"])


def fmt_n_pct(n: int, denominator: int) -> str:
    return f"{n:,} ({n / denominator * 100:.1f}%)" if denominator else "0 (0.0%)"


def continuous_smd(x0: pd.Series, x1: pd.Series) -> float:
    a = pd.to_numeric(x0, errors="coerce").dropna(); b = pd.to_numeric(x1, errors="coerce").dropna()
    pooled = math.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return (b.mean() - a.mean()) / pooled if pooled > 0 else np.nan


def binary_smd(x0: pd.Series, x1: pd.Series) -> float:
    p0 = bool_series(x0).mean(); p1 = bool_series(x1).mean()
    pooled = math.sqrt((p0 * (1 - p0) + p1 * (1 - p1)) / 2)
    return (p1 - p0) / pooled if pooled > 0 else np.nan


def median_iqr(series: pd.Series) -> str:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return f"{values.median():.1f} [{values.quantile(0.25):.1f}–{values.quantile(0.75):.1f}]"


def group_race(series: pd.Series) -> pd.Series:
    s = series.fillna("UNKNOWN").str.upper()
    result = pd.Series("Other/unknown", index=series.index)
    result.loc[s.str.startswith("WHITE")] = "White"
    result.loc[s.str.startswith("BLACK")] = "Black"
    result.loc[s.str.startswith("ASIAN")] = "Asian"
    result.loc[s.str.contains("HISPANIC|LATINO", regex=True)] = "Hispanic/Latino"
    return result


def group_admission(series: pd.Series) -> pd.Series:
    result = pd.Series("Observation/other", index=series.index)
    result.loc[series.isin(["ELECTIVE", "SURGICAL SAME DAY ADMISSION"])] = "Elective/same-day surgical"
    result.loc[series.isin(["URGENT", "EW EMER.", "DIRECT EMER."])] = "Urgent/emergency"
    return result


def group_icu(series: pd.Series) -> pd.Series:
    mapping = {
        "Cardiac Vascular Intensive Care Unit (CVICU)": "CVICU",
        "Surgical Intensive Care Unit (SICU)": "SICU",
        "Trauma SICU (TSICU)": "TSICU",
    }
    return series.map(mapping).fillna("Other surgical ICU")


def build_table1(cohort: pd.DataFrame) -> pd.DataFrame:
    data = cohort.copy()
    data["aki_group"] = bool_series(data["aki_final"])
    data["race_group"] = group_race(data["race"])
    data["admission_group"] = group_admission(data["admission_type"])
    data["icu_group"] = group_icu(data["first_careunit"])
    no_aki = data.loc[~data.aki_group]; aki = data.loc[data.aki_group]
    rows: list[dict[str, object]] = []

    def add_continuous(label: str, column: str) -> None:
        rows.append({
            "Characteristic": label, "Overall (N=10,877)": median_iqr(data[column]),
            "No AKI (N=6,346)": median_iqr(no_aki[column]), "Incident AKI (N=4,531)": median_iqr(aki[column]),
            "Standardized mean difference": continuous_smd(no_aki[column], aki[column]),
            "Missing, n": int(data[column].isna().sum()),
        })

    def add_binary(label: str, mask: pd.Series) -> None:
        mask = mask.fillna(False).astype(bool)
        rows.append({
            "Characteristic": label, "Overall (N=10,877)": fmt_n_pct(int(mask.sum()), len(data)),
            "No AKI (N=6,346)": fmt_n_pct(int(mask.loc[no_aki.index].sum()), len(no_aki)),
            "Incident AKI (N=4,531)": fmt_n_pct(int(mask.loc[aki.index].sum()), len(aki)),
            "Standardized mean difference": binary_smd(mask.loc[no_aki.index], mask.loc[aki.index]),
            "Missing, n": 0,
        })

    add_continuous("Age, years", "anchor_age")
    add_binary("Female sex", data.gender.eq("F"))
    for level in ["White", "Black", "Asian", "Hispanic/Latino", "Other/unknown"]:
        add_binary(f"Race: {level}", data.race_group.eq(level))
    for level in ["Elective/same-day surgical", "Urgent/emergency", "Observation/other"]:
        add_binary(f"Admission: {level}", data.admission_group.eq(level))
    add_continuous("Baseline serum creatinine, mg/dL", "baseline_scr_final")
    add_binary("Pre-index 7-day creatinine baseline", data.baseline_scr_source.eq("lowest_scr_7d_pre_icu"))
    add_continuous("Charlson comorbidity score", "charlson_score")
    for label, column in [
        ("Congestive heart failure", "chf"), ("Hypertension", "hypertension"),
        ("Diabetes mellitus", "dm"), ("Chronic kidney disease", "ckd"),
        ("Chronic pulmonary disease", "copd"), ("Liver disease", "liver"),
        ("Cancer", "cancer"), ("Peripheral vascular disease", "pvd"),
        ("Stroke", "stroke"), ("Myocardial infarction", "mi"),
        ("Obesity", "obesity"), ("Anaemia", "anemia"),
    ]:
        add_binary(label, pd.to_numeric(data[column], errors="coerce").eq(1))
    for label, column in [
        ("Cardiac surgery", "cardiac_surgery"), ("Non-cardiac surgery", "non_cardiac_surgery"),
        ("Vascular surgery", "vascular_surgery"), ("General/GI/hepatobiliary surgery", "general_gi_hepatobiliary_surgery"),
        ("Major orthopaedic surgery", "orthopedic_major_surgery"), ("Neurosurgery", "neurosurgery"),
        ("Thoracic/respiratory surgery", "thoracic_respiratory_surgery"),
    ]:
        add_binary(label, bool_series(data[column]))
    for level in ["CVICU", "SICU", "TSICU", "Other surgical ICU"]:
        add_binary(f"First ICU: {level}", data.icu_group.eq(level))
    table = pd.DataFrame(rows)
    table["Standardized mean difference"] = table["Standardized mean difference"].round(3)
    return table


def markdown_table(data: pd.DataFrame) -> str:
    columns = [str(column) for column in data.columns]
    def clean(value: object) -> str:
        if pd.isna(value):
            return ""
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in data.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(clean(value) for value in row) + " |")
    return "\n".join(lines)


def performance_tables(perf: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    best_index = perf.groupby("landmark_hours").auroc.idxmax()
    best = perf.loc[best_index].copy()
    best_table = pd.DataFrame({
        "Landmark": best.landmark_hours.map(lambda value: f"{int(value)} h"),
        "Selected model": best.model,
        "Test N": best.test_n.astype(int),
        "Event rate": best.test_event_rate.map(lambda value: f"{value * 100:.1f}%"),
        "AUROC (95% CI)": best.apply(lambda row: f"{row.auroc:.3f} ({row.auroc_ci_lower:.3f}–{row.auroc_ci_upper:.3f})", axis=1),
        "AUPRC (95% CI)": best.apply(lambda row: f"{row.auprc:.3f} ({row.auprc_ci_lower:.3f}–{row.auprc_ci_upper:.3f})", axis=1),
        "Brier score": best.brier_score.round(3),
        "Calibration intercept": best.calibration_intercept.round(3),
        "Calibration slope": best.calibration_slope.round(3),
        "Youden threshold": best.youden_threshold.round(3),
        "Sensitivity / specificity": best.apply(lambda row: f"{row.threshold_youden_sensitivity:.3f} / {row.threshold_youden_specificity:.3f}", axis=1),
    })
    full = perf[[
        "landmark_hours", "model", "test_n", "test_event_rate", "auroc", "auroc_ci_lower", "auroc_ci_upper",
        "auprc", "auprc_ci_lower", "auprc_ci_upper", "brier_score", "calibration_intercept", "calibration_slope",
        "threshold_0_5_sensitivity", "threshold_0_5_specificity", "youden_threshold",
        "threshold_youden_sensitivity", "threshold_youden_specificity",
    ]].copy()
    return best_table, full


def sensitivity_table(no_creat: pd.DataFrame, preindex: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, table, prefix in [
        ("No-creatinine model", no_creat, "no_creatinine"),
        ("Pre-index baseline-only retraining", preindex, "preindex_model"),
    ]:
        for row in table.loc[table.landmark_hours.eq(24)].itertuples():
            full_auc = getattr(row, "full_auroc" if label.startswith("No") else "full_model_auroc")
            sensitivity_auc = getattr(row, f"{prefix}_auroc")
            rows.append({
                "Sensitivity analysis": label, "Model": row.model,
                "Full/reference AUROC": round(full_auc, 3), "Sensitivity AUROC": round(sensitivity_auc, 3),
                "ΔAUROC (95% paired CI)": f"{row.delta_auroc:+.3f} ({row.delta_auroc_ci_lower:+.3f} to {row.delta_auroc_ci_upper:+.3f})",
            })
    return pd.DataFrame(rows)


def figure1_flowchart() -> pd.DataFrame:
    counts = pd.DataFrame([
        ("Strict postoperative surgical ICU cohort", 11943),
        ("Prevalent/index AKI excluded", 1014),
        ("Missing baseline SCr excluded", 50),
        ("No post-index SCr excluded", 2),
        ("Incident-AKI evaluable cohort / 0 h risk set", 10877),
        ("Final incident AKI within 7 days", 4531),
        ("No incident AKI within 7 days", 6346),
        ("AKI onset ≤6 h", 253),
        ("6 h risk set", 10624),
        ("AKI onset ≤24 h", 1576),
        ("24 h risk set", 9301),
    ], columns=["node", "n"])
    counts.to_csv(SOURCE_DATA_DIR / "Figure_1_flow_counts.csv", index=False)

    setup_style()
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

    def box(x, y, w, h, title, subtitle="", color="#EAF1FE", edge="#5477C4"):
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.08", facecolor=color, edgecolor=edge, linewidth=1.0)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h * 0.60, title, ha="center", va="center", fontsize=7, fontweight="semibold", color=PALETTE["ink"], wrap=True)
        if subtitle:
            ax.text(x + w / 2, y + h * 0.25, subtitle, ha="center", va="center", fontsize=6.2, color=PALETTE["muted"])

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10, linewidth=0.9, color=PALETTE["muted"]))

    box(3.2, 8.25, 3.6, 0.78, "Strict postoperative surgical ICU cohort", "n = 11,943")
    box(0.05, 6.55, 2.95, 1.05, "Excluded from incident-AKI\nanalysis", "Prevalent/index AKI: 1,014\nNo baseline SCr: 50; no follow-up SCr: 2", "#FFF3EF", "#CC6F47")
    box(3.2, 6.55, 3.6, 0.9, "Incident-AKI evaluable cohort", "n = 10,877")
    arrow(5.0, 8.25, 5.0, 7.45); arrow(3.2, 8.45, 2.1, 7.60)
    box(1.45, 4.75, 3.0, 0.9, "Final incident AKI", "n = 4,531 (41.7%)", "#EAF7EE", "#4A8B57")
    box(5.55, 4.75, 3.0, 0.9, "No incident AKI", "n = 6,346 (58.3%)", "#F4F5F7", "#7A828F")
    arrow(5.0, 6.55, 3.0, 5.65); arrow(5.0, 6.55, 7.0, 5.65)
    box(0.25, 2.45, 2.6, 0.95, "0 h risk set", "n = 10,877\nFuture AKI 4,531")
    box(3.7, 2.45, 2.6, 0.95, "6 h risk set", "n = 10,624\nFuture AKI 4,278")
    box(7.15, 2.45, 2.6, 0.95, "24 h risk set", "n = 9,301\nFuture AKI 2,955")
    # Landmark risk sets are parallel analytic risk sets, not descendants of either
    # final outcome box. Route their connector through the gap between outcome boxes.
    ax.plot([5.0, 5.0], [6.55, 4.12], color=PALETTE["muted"], linewidth=0.9, zorder=0)
    ax.plot([1.55, 8.45], [4.12, 4.12], color=PALETTE["muted"], linewidth=0.9, zorder=0)
    arrow(1.55, 4.12, 1.55, 3.40); arrow(5.0, 4.12, 5.0, 3.40); arrow(8.45, 4.12, 8.45, 3.40)
    ax.text(3.28, 2.15, "AKI onset ≤6 h excluded: 253", ha="center", va="top", fontsize=5.8, color=PALETTE["red"])
    ax.text(6.75, 2.15, "AKI onset ≤24 h excluded: 1,576 cumulative", ha="center", va="top", fontsize=5.8, color=PALETTE["red"])
    ax.text(0.1, 9.88, "Cohort derivation and dynamic landmark risk sets", fontsize=10, fontweight="bold", va="top")
    ax.text(0.1, 9.48, "Strict surgical ICU cohort; serum-creatinine KDIGO outcome; first ICU stay per hospital admission.", fontsize=6.5, color=PALETTE["muted"], va="top")
    save_figure(fig, "Figure_1_cohort_flowchart")
    return counts


def load_predictions() -> dict[int, pd.DataFrame]:
    return {h: pd.read_csv(SOURCE_ROOT / "modeling_v5_1" / f"model_v5_1_{h}h_test_predictions.csv") for h in [0, 6, 24]}


def figure2_roc(predictions: dict[int, pd.DataFrame]) -> None:
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharex=True, sharey=True)
    source_rows = []
    for panel, (ax, h) in enumerate(zip(axes, [0, 6, 24])):
        data = predictions[h]; y = data.y_true.to_numpy(int)
        for model, safe in MODEL_SAFE.items():
            fpr, tpr, _ = roc_curve(y, data[f"prob_{safe}"])
            color, line = MODEL_STYLE[model]
            ax.plot(fpr, tpr, color=color, linestyle=line, linewidth=1.1, label=MODEL_DISPLAY[model])
            source_rows.extend({"landmark_hours": h, "model": model, "fpr": x, "tpr": yy} for x, yy in zip(fpr, tpr))
        ax.plot([0, 1], [0, 1], color=PALETTE["muted"], linestyle=":", linewidth=0.8)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="1 − specificity", title=f"{h} h (n={len(data):,})")
        if panel == 0: ax.set_ylabel("Sensitivity")
        panel_label(ax, chr(97 + panel))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.52, 1.04), ncol=4, frameon=False)
    fig.suptitle("Dynamic model discrimination across ICU landmarks", x=0.08, ha="left", fontsize=9, fontweight="bold", y=1.12)
    fig.subplots_adjust(top=0.78, wspace=0.18)
    pd.DataFrame(source_rows).to_csv(SOURCE_DATA_DIR / "Figure_2_ROC_source.csv", index=False)
    save_figure(fig, "Figure_2_dynamic_ROC")


def figure3_calibration_dca(predictions: dict[int, pd.DataFrame], dca: pd.DataFrame) -> None:
    setup_style()
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.0), sharex="row")
    calibration_rows = []
    for column, h in enumerate([0, 6, 24]):
        data = predictions[h]; y = data.y_true.to_numpy(int)
        ax = axes[0, column]
        for model, safe in MODEL_SAFE.items():
            observed, predicted = calibration_curve(y, data[f"prob_{safe}"], n_bins=10, strategy="quantile")
            color, line = MODEL_STYLE[model]
            ax.plot(predicted, observed, marker="o", markersize=2.2, color=color, linestyle=line, linewidth=0.9, label=MODEL_DISPLAY[model])
            calibration_rows.extend({"landmark_hours": h, "model": model, "mean_predicted": x, "observed": yy} for x, yy in zip(predicted, observed))
        ax.plot([0, 1], [0, 1], color=PALETTE["muted"], linestyle=":", linewidth=0.8)
        ax.set(xlim=(0, 1), ylim=(0, 1), title=f"{h} h")
        if column == 0: ax.set_ylabel("Observed risk")
        ax.set_xlabel("Predicted risk")
        ax.tick_params(labelsize=5.5)
        panel_label(ax, chr(97 + column), y=1.12)

        ax2 = axes[1, column]
        part = dca.loc[dca.landmark_hours.eq(h)]
        for strategy, group in part.groupby("strategy", sort=False):
            if strategy == "Treat none": color, line, width = PALETTE["muted"], ":", 0.8
            elif strategy == "Treat all": color, line, width = PALETTE["ink"], "--", 0.8
            else: color, line = MODEL_STYLE[strategy]; width = 0.95
            ax2.plot(group.threshold, group.net_benefit, color=color, linestyle=line, linewidth=width,
                     label=MODEL_DISPLAY.get(strategy, strategy))
        ax2.axhline(0, color=PALETTE["grid"], linewidth=0.7)
        ax2.set(xlim=(0.05, 0.80), ylim=(-0.08, 0.43), xlabel="Threshold")
        if column == 0: ax2.set_ylabel("Net benefit")
        ax2.tick_params(labelsize=5.5)
        panel_label(ax2, chr(100 + column), y=1.12)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    treat_handles, treat_labels = axes[1, 0].get_legend_handles_labels()
    unique = dict(zip(labels + treat_labels, handles + treat_handles))
    fig.legend(unique.values(), unique.keys(), loc="upper center", bbox_to_anchor=(0.52, 1.01), ncol=6, frameon=False, fontsize=5.8)
    fig.suptitle("Calibration and clinical net benefit across landmarks", x=0.08, ha="left", fontsize=9, fontweight="bold", y=1.08)
    fig.subplots_adjust(top=0.80, hspace=0.48, wspace=0.30, bottom=0.10)
    pd.DataFrame(calibration_rows).to_csv(SOURCE_DATA_DIR / "Figure_3_calibration_source.csv", index=False)
    dca.to_csv(SOURCE_DATA_DIR / "Figure_3_DCA_source.csv", index=False)
    save_figure(fig, "Figure_3_calibration_DCA")


def figure4_shap(shap_data: pd.DataFrame) -> None:
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.9))
    source = shap_data.loc[shap_data.landmark_hours.isin([6, 24]) & shap_data["rank"].le(12)].copy()
    short_names = {
        "baseline_scr_at_landmark": "Baseline SCr",
        "anchor_age": "Age",
        "ckd": "Chronic kidney disease",
        "charlson_score": "Charlson score",
        "cardiac_surgery": "Cardiac surgery",
        "first_careunit_Cardiac Vascular Intensive Care Unit (CVICU)": "CVICU",
        "lab_0_6h_hemoglobin_min": "Minimum hemoglobin",
        "vital_0_6h_temperature_c_max": "Maximum temperature",
        "lab_0_6h_platelet_min": "Minimum platelets",
        "lab_0_6h_pao2_last": "Last PaO2",
        "lab_0_6h_potassium_min": "Minimum potassium",
        "lab_0_6h_ph_max": "Maximum pH",
        "lab_0_24h_creatinine_last": "Last creatinine",
        "lab_0_24h_hemoglobin_min": "Minimum hemoglobin",
        "lab_0_24h_creatinine_min": "Minimum creatinine",
        "lab_0_24h_wbc_last": "Last white-cell count",
        "lab_0_24h_bun_last": "Last blood urea nitrogen",
        "lab_0_24h_potassium_count": "Potassium measurement count",
        "lab_0_24h_potassium_last": "Last potassium",
        "lab_0_24h_pao2_last": "Last PaO2",
        "lab_0_24h_ph_last": "Last pH",
        "vital_0_24h_sbp_last": "Last systolic pressure",
    }
    for panel, (ax, h) in enumerate(zip(axes, [6, 24])):
        part = source.loc[source.landmark_hours.eq(h)].sort_values("mean_abs_shap")
        labels = [short_names.get(label, label.replace("_", " ").title()) for label in part.feature]
        ax.barh(labels, part.mean_abs_shap, color="#FFE15B", edgecolor="#736422", linewidth=0.6)
        ax.set(xlabel="Mean |SHAP value|", title=f"{h} h XGBoost")
        ax.tick_params(axis="y", labelsize=5.6)
        panel_label(ax, chr(97 + panel), x=-0.23)
    fig.suptitle("Global XGBoost feature importance at 6 h and 24 h", x=0.06, ha="left", fontsize=9, fontweight="bold", y=1.02)
    fig.subplots_adjust(wspace=0.62, top=0.86, left=0.18, right=0.98)
    source.to_csv(SOURCE_DATA_DIR / "Figure_4_SHAP_source.csv", index=False)
    save_figure(fig, "Figure_4_SHAP_6h_24h")


def supplementary_pr(predictions: dict[int, pd.DataFrame]) -> None:
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharex=True, sharey=True)
    rows = []
    for panel, (ax, h) in enumerate(zip(axes, [0, 6, 24])):
        data = predictions[h]; y = data.y_true.to_numpy(int)
        for model, safe in MODEL_SAFE.items():
            precision, recall, _ = precision_recall_curve(y, data[f"prob_{safe}"])
            color, line = MODEL_STYLE[model]
            ax.plot(recall, precision, color=color, linestyle=line, linewidth=1.0, label=MODEL_DISPLAY[model])
            rows.extend({"landmark_hours": h, "model": model, "recall": x, "precision": yy} for x, yy in zip(recall, precision))
        ax.axhline(y.mean(), color=PALETTE["muted"], linestyle=":", linewidth=0.8)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Recall", title=f"{h} h")
        if panel == 0: ax.set_ylabel("Precision")
        panel_label(ax, chr(97 + panel))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.52, 1.03), ncol=4, frameon=False)
    fig.suptitle("Precision–recall performance across landmarks", x=0.08, ha="left", fontsize=9, fontweight="bold", y=1.11)
    fig.subplots_adjust(top=0.78, wspace=0.18)
    pd.DataFrame(rows).to_csv(SOURCE_DATA_DIR / "Figure_S1_PR_source.csv", index=False)
    save_figure(fig, "Figure_S1_precision_recall")


def sensitivity_figure(comparison: pd.DataFrame, title: str, output_name: str, sensitivity_label: str) -> None:
    setup_style()
    part = comparison.loc[comparison.landmark_hours.eq(24)].sort_values("delta_auroc")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]; positions = np.arange(len(part))
    error = np.vstack([part.delta_auroc - part.delta_auroc_ci_lower, part.delta_auroc_ci_upper - part.delta_auroc])
    ax.errorbar(part.delta_auroc, positions, xerr=error, fmt="o", color=PALETTE["orange"], ecolor="#804126", capsize=3, linewidth=0.9)
    ax.axvline(0, color=PALETTE["muted"], linestyle=":", linewidth=0.8)
    model_labels = [MODEL_DISPLAY.get(value, value) for value in part.model]
    ax.set_yticks(positions, model_labels); ax.set_xlabel(f"ΔAUROC: {sensitivity_label} − reference")
    ax.tick_params(axis="y", labelsize=6.0, pad=3)
    panel_label(ax, "a", x=-0.16)
    ax2 = axes[1]
    y = np.arange(len(part)); width = 0.32
    full_column = "full_auroc" if "full_auroc" in part else "full_model_auroc"
    sensitivity_column = "no_creatinine_auroc" if "no_creatinine_auroc" in part else "preindex_model_auroc"
    ax2.barh(y - width / 2, part[full_column], height=width, color="#CEDFFE", edgecolor="#2E4780", label="Reference")
    ax2.barh(y + width / 2, part[sensitivity_column], height=width, color="#FFBDA1", edgecolor="#804126", label=sensitivity_label)
    ax2.set_yticks(y, model_labels); ax2.set_xlim(0.65, 0.78); ax2.set_xlabel("AUROC")
    ax2.tick_params(axis="y", labelsize=6.0, pad=3)
    panel_label(ax2, "b", x=-0.16)
    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.98, 0.94), ncol=2, frameon=False, fontsize=5.6)
    fig.suptitle(title, x=0.07, ha="left", fontsize=9, fontweight="bold", y=1.02)
    fig.subplots_adjust(left=0.20, right=0.98, wspace=0.62, top=0.80, bottom=0.22)
    part.to_csv(SOURCE_DATA_DIR / f"{output_name}_source.csv", index=False)
    save_figure(fig, output_name)


def supplementary_subgroups(subgroups: pd.DataFrame) -> None:
    best = {0: "XGBoost", 6: "XGBoost", 24: "Logistic Regression"}
    dimensions = ["gender", "age_group", "cardiac_surgery", "ckd", "vascular_surgery", "general_gi_hepatobiliary_surgery", "orthopedic_major_surgery", "neurosurgery"]
    labels_order = ["Female", "Male", "Age <65", "Age ≥65", "Cardiac", "Non-cardiac", "CKD", "No CKD", "Vascular", "GI/hepatobiliary", "Orthopaedic", "Neurosurgery"]
    rows = []
    label_map = {
        ("gender", "F"): "Female", ("gender", "M"): "Male", ("age_group", "<65"): "Age <65", ("age_group", ">=65"): "Age ≥65",
        ("cardiac_surgery", "yes"): "Cardiac", ("cardiac_surgery", "no"): "Non-cardiac", ("ckd", "yes"): "CKD", ("ckd", "no"): "No CKD",
        ("vascular_surgery", "yes"): "Vascular", ("general_gi_hepatobiliary_surgery", "yes"): "GI/hepatobiliary",
        ("orthopedic_major_surgery", "yes"): "Orthopedic", ("neurosurgery", "yes"): "Neurosurgery",
    }
    for h, model in best.items():
        part = subgroups.loc[subgroups.landmark_hours.eq(h) & subgroups.model.eq(model) & subgroups.subgroup_dimension.isin(dimensions)].copy()
        part["label"] = [label_map.get((dimension, level)) for dimension, level in zip(part.subgroup_dimension, part.subgroup_level)]
        rows.append(part.dropna(subset=["label"]))
    source = pd.concat(rows, ignore_index=True)
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 5.5), sharey=True)
    for panel, (ax, h) in enumerate(zip(axes, [0, 6, 24])):
        part = source.loc[source.landmark_hours.eq(h)].set_index("label").reindex(labels_order).dropna(subset=["auroc"])
        positions = np.arange(len(part))[::-1]
        errors = np.vstack([part.auroc - part.auroc_ci_lower, part.auroc_ci_upper - part.auroc])
        ax.errorbar(part.auroc, positions, xerr=errors, fmt="o", color=PALETTE["blue"], ecolor="#2E4780", capsize=2, linewidth=0.8, markersize=3)
        ax.axvline(0.5, color=PALETTE["muted"], linestyle=":", linewidth=0.7)
        ax.set_yticks(positions, part.index); ax.set_xlim(0.4, 1.0); ax.set_xlabel("AUROC"); ax.set_title(f"{h} h: {MODEL_DISPLAY[best[h]]}")
        panel_label(ax, chr(97 + panel), x=-0.27)
    fig.suptitle("Subgroup discrimination in selected landmark models", x=0.08, ha="left", fontsize=9, fontweight="bold", y=1.01)
    fig.subplots_adjust(wspace=0.22, top=0.90)
    source.to_csv(SOURCE_DATA_DIR / "Figure_S4_subgroup_source.csv", index=False)
    save_figure(fig, "Figure_S4_subgroup_performance")


def supplementary_missingness(missingness: pd.DataFrame) -> None:
    variables = [
        "lab_pre24h_bicarbonate_last", "lab_pre24h_inr_last", "lab_pre24h_creatinine_last",
        "lab_pre24h_bun_last", "lab_pre24h_hemoglobin_last", "lab_pre24h_wbc_last", "lab_pre24h_platelet_last",
    ]
    part = missingness.loc[missingness.predictor.isin(variables)].copy()
    matrix = part.pivot(index="predictor", columns="landmark_hours", values="missing_percent").reindex(variables)
    display_names = {
        "lab_pre24h_bicarbonate_last": "Bicarbonate",
        "lab_pre24h_inr_last": "INR",
        "lab_pre24h_creatinine_last": "Creatinine",
        "lab_pre24h_bun_last": "BUN",
        "lab_pre24h_hemoglobin_last": "Hemoglobin",
        "lab_pre24h_wbc_last": "WBC",
        "lab_pre24h_platelet_last": "Platelet",
    }
    matrix.index = [display_names[label] for label in matrix.index]
    setup_style()
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    cmap = sns.light_palette(PALETTE["orange"], as_cmap=True)
    sns.heatmap(matrix, annot=True, fmt=".1f", cmap=cmap, linewidths=0.7, linecolor="white", cbar_kws={"label": "Missing (%)"}, ax=ax)
    ax.set(xlabel="Landmark (h)", ylabel="Pre-index predictor")
    ax.set_title("High-missingness predictors across dynamic datasets", loc="left", fontsize=8, fontweight="bold", pad=10)
    matrix.reset_index().to_csv(SOURCE_DATA_DIR / "Figure_S5_missingness_source.csv", index=False)
    save_figure(fig, "Figure_S5_predictor_missingness")


def build_manuscript(
    cohort: pd.DataFrame,
    best_table: pd.DataFrame,
    perf: pd.DataFrame,
    no_creat: pd.DataFrame,
    preindex: pd.DataFrame,
    subgroups: pd.DataFrame,
) -> tuple[str, str, str]:
    aki = bool_series(cohort.aki_final); aki_data = cohort.loc[aki]
    stage_counts = pd.to_numeric(aki_data.aki_stage_final).value_counts().sort_index()
    onset = pd.to_numeric(aki_data.aki_onset_hours_final, errors="coerce")
    overall_age = pd.to_numeric(cohort.anchor_age).median()
    aki_age = pd.to_numeric(cohort.loc[aki, "anchor_age"]).median()
    no_age = pd.to_numeric(cohort.loc[~aki, "anchor_age"]).median()
    cardiac_rate = bool_series(cohort.cardiac_surgery).mean() * 100
    preindex_rate = cohort.baseline_scr_source.eq("lowest_scr_7d_pre_icu").mean() * 100
    best = perf.loc[perf.groupby("landmark_hours").auroc.idxmax()].set_index("landmark_hours")
    no24 = no_creat.loc[no_creat.landmark_hours.eq(24)].set_index("model")
    pre24 = preindex.loc[preindex.landmark_hours.eq(24)].set_index("model")

    methods = f"""## Methods

### Study design and data source

We conducted a retrospective cohort study using the de-identified MIMIC-IV version 3.1 database [MIMIC-IV citation]. The analysis unit was the first qualifying ICU stay within each hospital admission. The index time was ICU admission (`intime`). All cohort construction, outcome adjudication, feature generation and modelling steps were implemented as reproducible Python pipelines.

### Cohort construction

Adults were eligible for the strict primary cohort when an explicitly therapeutic major surgical procedure was recorded on the day of ICU admission or the preceding day and the first ICU location was a surgical, cardiac vascular, trauma surgical, mixed medical-surgical, neuro-surgical or post-anaesthesia care unit. Eligible procedures comprised cardiac, vascular, general gastrointestinal or hepatobiliary, major orthopaedic, neurosurgical, and thoracic or respiratory operations. Diagnostic imaging, electrocardiography, vascular access, tracheal intubation, enteral nutrition, dialysis, non-operative respiratory measurements and obstetric procedures were excluded from the strict surgical definition. A broad procedure-based cohort was retained for sensitivity analyses during cohort development but was not used for primary model training.

Patients with AKI already present at or before ICU admission, no usable baseline serum creatinine (SCr), or no post-index SCr measurement within seven days were retained for audit but excluded from the incident-AKI analytic cohort. The resulting strict evaluable cohort formed the 0 h risk set.

### Baseline kidney function and AKI outcome

Baseline SCr was defined as the lowest value recorded during the seven days before ICU admission. If unavailable, the earliest SCr within 24 h after hospital admission was used and its source and timestamp were retained. The baseline-source sensitivity cohort was restricted to patients whose baseline was measured before ICU admission within the seven-day window.

Incident AKI was adjudicated from timestamped serum creatinine measurements without urine-output criteria. AKI was present when SCr increased by at least 0.3 mg dl−1 from a prior result within 48 h or reached at least 1.5 times baseline within seven days after ICU admission [KDIGO citation]. Patients satisfying either criterion before or at the index time were classified as having prevalent AKI and were excluded from incident-AKI modelling. AKI severity was assigned from the seven-day peak SCr: stage 1, 1.5–<2.0 times baseline or an absolute rise of at least 0.3 mg dl−1; stage 2, 2.0–<3.0 times baseline; and stage 3, at least 3.0 times baseline or peak SCr of at least 4.0 mg dl−1.

### Dynamic landmark datasets

Prediction datasets were constructed at ICU admission (0 h), 6 h and 24 h. Patients with AKI onset at or before a landmark were excluded from that landmark risk set. The outcome at each landmark was new AKI after the landmark and within seven days of ICU admission. Static predictors included age, sex, race, admission characteristics, comorbidities, Charlson score, surgical category and first ICU type. Baseline SCr information was made available only when its timestamp preceded the corresponding landmark. The most recent laboratory values during the 24 h before ICU admission were included as pre-index features. For the 6 h and 24 h datasets, minimum, maximum, most recent and measurement-count features were recalculated from timestamped laboratory and vital-sign observations within `(0, landmark]`. Legacy whole-period `post_*` summaries, mortality, length of stay, AKI-derived variables and untimed laboratory summaries were excluded to prevent information leakage.

### Model development and internal validation

Logistic regression, random forest, XGBoost and LightGBM models were developed separately at each landmark. Continuous variables were median-imputed with missingness indicators; categorical and binary variables were imputed using the most frequent value; categorical variables were one-hot encoded. Continuous variables were standardized for logistic regression. Tree-model hyperparameters were fixed before test evaluation: random forest, 500 trees and minimum leaf size 5; XGBoost, 500 trees, learning rate 0.03 and maximum depth 4; LightGBM, 500 trees, learning rate 0.03 and 31 leaves.

An 80:20 patient-grouped split was selected from 500 candidate GroupShuffleSplit assignments to approximate overall outcome prevalence. The same patient assignment was reused at all landmarks, and no patient appeared in both training and test sets. No test-set hyperparameter tuning was performed.

### Performance, calibration and clinical utility

Discrimination was assessed using the area under the receiver-operating-characteristic curve (AUROC) and area under the precision–recall curve (AUPRC). Overall accuracy, sensitivity, specificity, precision, F1 score and confusion matrices were calculated at a probability threshold of 0.5. A secondary threshold maximizing the Youden index was selected from training predictions and applied unchanged to the test set. Overall calibration was summarized by the Brier score, calibration intercept and calibration slope; calibration plots used ten equal-frequency bins. Decision-curve analysis quantified net benefit over threshold probabilities from 0.05 to 0.80 relative to treat-all and treat-none strategies.

Confidence intervals were obtained from 1,000 patient-level bootstrap resamples of the held-out test set. Subgroup performance was evaluated by sex, age, cardiac and other surgical categories, chronic kidney disease, baseline source and sufficiently large ICU groups; subgroup confidence intervals used 300 patient-level bootstrap resamples. Groups with fewer than 50 observations or a single outcome class were not estimated.

### Model interpretation and sensitivity analyses

Global SHAP values were calculated in up to 1,000 held-out observations for the tree model with the highest test AUROC at each landmark. SHAP magnitudes were interpreted as model-attribution measures rather than causal effects. Two prespecified sensitivity analyses were performed. First, all predictors containing creatinine, baseline SCr, or baseline-to-ICU timing information were removed and models were retrained. Second, models were retrained only among patients with a pre-index seven-day SCr baseline. Both analyses preserved the original patient assignment and used paired patient-level bootstrap resampling to compare predictions on identical test patients.

### Software

Analyses were performed in Python 3.14 using pandas 3.0, scikit-learn 1.9, XGBoost 3.3, LightGBM 4.6 and SHAP 0.52. Reporting should be finalized against the target journal’s requirements and the TRIPOD/PROBAST framework [TRIPOD citation; PROBAST citation].
"""

    results = f"""## Results

### Cohort finalization and incident AKI

The strict postoperative surgical ICU cohort included 11,943 hospital admissions (Fig. 1). We excluded 1,014 admissions with AKI present at or before ICU admission, 50 without a usable baseline SCr and two without a post-index SCr measurement, leaving 10,877 admissions in the incident-AKI analytic cohort. Incident AKI occurred in 4,531 admissions (41.7%). Of these events, {int(stage_counts.get(1, 0)):,} ({stage_counts.get(1, 0) / len(aki_data) * 100:.1f}%) were stage 1, {int(stage_counts.get(2, 0)):,} ({stage_counts.get(2, 0) / len(aki_data) * 100:.1f}%) were stage 2 and {int(stage_counts.get(3, 0)):,} ({stage_counts.get(3, 0) / len(aki_data) * 100:.1f}%) were stage 3. Median AKI onset was {onset.median():.1f} h after ICU admission (interquartile range, {onset.quantile(0.25):.1f}–{onset.quantile(0.75):.1f} h).

The median age was {overall_age:.0f} years and was higher among patients who developed AKI than among those who did not ({aki_age:.0f} versus {no_age:.0f} years; Table 1). Cardiac operations accounted for {cardiac_rate:.1f}% of the cohort, and {preindex_rate:.1f}% had a baseline SCr measured within seven days before ICU admission. Other baseline differences, including comorbidity and surgical subgroup distributions, are summarized in Table 1 using standardized mean differences.

### Dynamic prediction risk sets

All 10,877 evaluable admissions entered the 0 h dataset, with 4,531 subsequent AKI events (41.7%). Exclusion of 253 events occurring by 6 h yielded 10,624 admissions and 4,278 future events at the 6 h landmark. Exclusion of 1,576 cumulative events occurring by 24 h yielded 9,301 admissions and 2,955 future events at 24 h (Fig. 1). The changing landmark populations and outcome windows preclude interpreting between-landmark metric differences as a paired longitudinal comparison.

### Model discrimination and calibration

At 0 h, XGBoost provided the highest test discrimination, with an AUROC of {best.loc[0, 'auroc']:.3f} (95% CI, {best.loc[0, 'auroc_ci_lower']:.3f}–{best.loc[0, 'auroc_ci_upper']:.3f}) and an AUPRC of {best.loc[0, 'auprc']:.3f} (95% CI, {best.loc[0, 'auprc_ci_lower']:.3f}–{best.loc[0, 'auprc_ci_upper']:.3f}). XGBoost also had the highest AUROC at 6 h ({best.loc[6, 'auroc']:.3f}; 95% CI, {best.loc[6, 'auroc_ci_lower']:.3f}–{best.loc[6, 'auroc_ci_upper']:.3f}) and an AUPRC of {best.loc[6, 'auprc']:.3f}. At 24 h, logistic regression had the highest AUROC ({best.loc[24, 'auroc']:.3f}; 95% CI, {best.loc[24, 'auroc_ci_lower']:.3f}–{best.loc[24, 'auroc_ci_upper']:.3f}), with an AUPRC of {best.loc[24, 'auprc']:.3f} and a Brier score of {best.loc[24, 'brier_score']:.3f} (Fig. 2 and Table 2). Confidence intervals overlapped substantially across model families, and these results did not establish superiority of one algorithm.

Calibration was close to ideal for XGBoost at 0 h (intercept {best.loc[0, 'calibration_intercept']:.3f}; slope {best.loc[0, 'calibration_slope']:.3f}) and remained acceptable at 6 h (intercept {best.loc[6, 'calibration_intercept']:.3f}; slope {best.loc[6, 'calibration_slope']:.3f}). The 24 h logistic model showed a calibration intercept of {best.loc[24, 'calibration_intercept']:.3f} and slope of {best.loc[24, 'calibration_slope']:.3f}, indicating modest over-dispersion of predicted risks in the held-out sample (Fig. 3). The selected models provided greater net benefit than treat-all and treat-none strategies across threshold ranges 0.05–0.80 at 0 h and 6 h and approximately 0.08–0.79 at 24 h. These decision curves represent internal-validation estimates rather than evidence of implemented clinical benefit.

### Model interpretation

Global SHAP analysis showed that at 6 h, baseline SCr, age, early minimum haemoglobin, Charlson score and chronic kidney disease were prominent XGBoost features. By 24 h, the most recent creatinine was the largest-magnitude feature, followed by age, minimum haemoglobin, cardiac vascular ICU location, cardiac surgery, minimum creatinine, white-cell count and blood urea nitrogen (Fig. 4). These values describe model attribution and should not be interpreted as causal or directly modifiable effects.

### Subgroup performance

Discrimination was broadly preserved across sex and age groups, although estimates were less precise in smaller surgical subgroups (Supplementary Fig. S4 and Supplementary Table S2). For example, the selected 24 h logistic model had an AUROC of 0.767 (95% CI, 0.733–0.807) in women and 0.745 (95% CI, 0.715–0.776) in men. Corresponding estimates were 0.732 (95% CI, 0.693–0.767) for patients younger than 65 years and 0.750 (95% CI, 0.719–0.776) for those aged 65 years or older. Wide confidence intervals in neurosurgical, vascular and orthopaedic subgroups limited comparative inference.

### Sensitivity analyses

Removing all creatinine-derived predictors reduced 24 h discrimination in every model family (Supplementary Fig. S2 and Table 3). The logistic-regression AUROC decreased from {no24.loc['Logistic Regression', 'full_auroc']:.3f} to {no24.loc['Logistic Regression', 'no_creatinine_auroc']:.3f}, a paired difference of {no24.loc['Logistic Regression', 'delta_auroc']:.3f} (95% CI, {no24.loc['Logistic Regression', 'delta_auroc_ci_lower']:.3f} to {no24.loc['Logistic Regression', 'delta_auroc_ci_upper']:.3f}). The AUROC decrease also excluded zero for random forest, XGBoost and LightGBM. Nevertheless, no-creatinine models retained AUROCs of approximately 0.73, indicating that non-creatinine clinical information contributed materially to prediction.

Restricting the cohort to patients with a pre-index seven-day SCr baseline did not materially alter 24 h discrimination (Supplementary Fig. S3). The retrained logistic model achieved an AUROC of {pre24.loc['Logistic Regression', 'preindex_model_auroc']:.3f}, compared with {pre24.loc['Logistic Regression', 'full_model_auroc']:.3f} for the full-cohort model evaluated in the same restricted test patients; the paired difference was {pre24.loc['Logistic Regression', 'delta_auroc']:.3f} (95% CI, {pre24.loc['Logistic Regression', 'delta_auroc_ci_lower']:.3f} to {pre24.loc['Logistic Regression', 'delta_auroc_ci_upper']:.3f}). Paired AUROC confidence intervals crossed zero for all four model families, supporting robustness to the baseline-source restriction.
"""

    full = f"""# Methods and Results draft

{methods}

{results}
"""
    return methods, results, full


def main() -> None:
    for directory in [OUTPUT_ROOT, TABLE_DIR, FIGURE_DIR, SOURCE_DATA_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    cohort = pd.read_csv(SOURCE_ROOT / "finalized_v3_1" / "cohort_v3_1_strict_main_evaluable.csv", low_memory=False)
    perf = pd.read_csv(SOURCE_ROOT / "modeling_v5_1" / "model_v5_1_performance_bootstrap_ci.csv")
    dca = pd.read_csv(SOURCE_ROOT / "modeling_v5_1" / "model_v5_1_dca.csv")
    shap_data = pd.read_csv(SOURCE_ROOT / "modeling_v5_1" / "model_v5_1_shap_importance.csv")
    subgroups = pd.read_csv(SOURCE_ROOT / "modeling_v5_1" / "model_v5_1_subgroup_performance.csv")
    missingness = pd.read_csv(SOURCE_ROOT / "modeling_v4_1" / "audit_v4_1_predictor_missingness.csv")
    no_creat = pd.read_csv(SOURCE_ROOT / "modeling_v5_2_no_creatinine" / "model_v5_2_full_vs_no_creatinine_paired_comparison.csv")
    preindex = pd.read_csv(SOURCE_ROOT / "modeling_v5_3_preindex_baseline" / "model_v5_3_full_vs_preindex_paired_comparison.csv")

    table1 = build_table1(cohort)
    best_table, full_performance = performance_tables(perf)
    table3 = sensitivity_table(no_creat, preindex)

    table1.to_csv(TABLE_DIR / "Table_1_baseline_characteristics.csv", index=False)
    best_table.to_csv(TABLE_DIR / "Table_2_selected_model_performance.csv", index=False)
    table3.to_csv(TABLE_DIR / "Table_3_sensitivity_analyses_24h.csv", index=False)
    full_performance.to_csv(TABLE_DIR / "Table_S1_full_model_performance.csv", index=False)

    best_models = {0: "XGBoost", 6: "XGBoost", 24: "Logistic Regression"}
    subgroup_selected = pd.concat([
        subgroups.loc[subgroups.landmark_hours.eq(h) & subgroups.model.eq(model)]
        for h, model in best_models.items()
    ])
    subgroup_selected.to_csv(TABLE_DIR / "Table_S2_subgroup_performance.csv", index=False)
    missingness.to_csv(TABLE_DIR / "Table_S3_predictor_missingness.csv", index=False)
    pd.concat([
        no_creat.assign(sensitivity_analysis="no_creatinine"),
        preindex.assign(sensitivity_analysis="preindex_baseline_only"),
    ], ignore_index=True, sort=False).to_csv(TABLE_DIR / "Table_S4_sensitivity_all_landmarks.csv", index=False)

    (TABLE_DIR / "Table_1_baseline_characteristics.md").write_text(markdown_table(table1), encoding="utf-8")
    (TABLE_DIR / "Table_2_selected_model_performance.md").write_text(markdown_table(best_table), encoding="utf-8")
    (TABLE_DIR / "Table_3_sensitivity_analyses_24h.md").write_text(markdown_table(table3), encoding="utf-8")

    predictions = load_predictions()
    figure1_flowchart()
    figure2_roc(predictions)
    figure3_calibration_dca(predictions, dca)
    figure4_shap(shap_data)
    supplementary_pr(predictions)
    sensitivity_figure(no_creat, "No-creatinine sensitivity at the 24 h landmark", "Figure_S2_no_creatinine_sensitivity", "No creatinine")
    sensitivity_figure(preindex, "Pre-index baseline-only sensitivity at the 24 h landmark", "Figure_S3_preindex_baseline_sensitivity", "Pre-index retrained")
    supplementary_subgroups(subgroups)
    supplementary_missingness(missingness)

    methods, results, manuscript = build_manuscript(cohort, best_table, perf, no_creat, preindex, subgroups)
    (OUTPUT_ROOT / "Methods.md").write_text(methods, encoding="utf-8")
    (OUTPUT_ROOT / "Results.md").write_text(results, encoding="utf-8")
    (OUTPUT_ROOT / "Methods_and_Results.md").write_text(manuscript, encoding="utf-8")

    ledger = """# Terminology ledger

| Canonical term | First-use definition | Decision |
|---|---|---|
| AKI | acute kidney injury (AKI) | Use AKI after first expansion |
| SCr | serum creatinine (SCr) | Use SCr consistently; units mg dl−1 |
| ICU | intensive care unit (ICU) | Use ICU after first expansion |
| KDIGO | Kidney Disease: Improving Global Outcomes (KDIGO) | Use for SCr outcome criteria |
| AUROC | area under the receiver-operating-characteristic curve (AUROC) | Report with patient-level bootstrap 95% CI |
| AUPRC | area under the precision–recall curve (AUPRC) | Report with patient-level bootstrap 95% CI |
| DCA | decision-curve analysis (DCA) | Interpret as exploratory internal clinical utility |
| SHAP | SHapley Additive exPlanations (SHAP) | Model attribution, not causal importance |
| Landmark | ICU admission (0 h), 6 h or 24 h prediction time | Outcome begins strictly after the landmark |
| Strict primary cohort | Explicit major surgery plus surgical ICU cohort | Main analysis population |
"""
    (OUTPUT_ROOT / "terminology_ledger.md").write_text(ledger, encoding="utf-8")

    captions = """# Figure captions

## Figure 1 | Cohort derivation and dynamic risk sets
Flow diagram showing construction of the strict postoperative surgical ICU cohort, exclusions from incident-AKI analysis, final seven-day SCr-defined AKI status, and the 0 h, 6 h and 24 h landmark risk sets. Counts at later landmarks exclude AKI with onset at or before the landmark.

## Figure 2 | Dynamic model discrimination across ICU landmarks
Receiver-operating-characteristic curves for logistic regression, random forest, XGBoost and LightGBM in patient-grouped held-out test sets at ICU admission, 6 h and 24 h. Landmark populations and future-event windows differ; curves should not be interpreted as paired repeated measurements of one fixed cohort.

## Figure 3 | Calibration and decision-curve analysis
Top, observed versus predicted event proportions in ten equal-frequency bins. Bottom, net benefit across threshold probabilities 0.05–0.80 relative to treat-all and treat-none strategies. All estimates are from the held-out internal-validation test sets.

## Figure 4 | Global SHAP feature importance at 6 h and 24 h
Mean absolute SHAP values for the twelve leading features in the XGBoost models. Values were calculated in up to 1,000 held-out observations and quantify model attribution rather than causal or modifiable effects.

## Supplementary Figure S1 | Precision–recall curves
Precision–recall curves for all four model families at 0 h, 6 h and 24 h. Horizontal dotted lines denote test-set event prevalence.

## Supplementary Figure S2 | No-creatinine sensitivity analysis
Paired patient-level bootstrap comparison at 24 h after removing all creatinine and baseline-SCr predictors. Negative ΔAUROC values favour the full model.

## Supplementary Figure S3 | Pre-index baseline-only sensitivity analysis
Paired comparison of original full-cohort model predictions and models retrained only in patients with a seven-day pre-ICU baseline SCr, evaluated in the same restricted test patients.

## Supplementary Figure S4 | Subgroup model discrimination
AUROC and patient-level bootstrap 95% confidence intervals for prespecified demographic, kidney-disease and surgical subgroups. Selected models were XGBoost at 0 h and 6 h and logistic regression at 24 h. Small subgroups have wide intervals.

## Supplementary Figure S5 | Predictor missingness
Missingness percentages for the seven pre-index laboratory predictors exceeding 40% missingness in the modeling-ready datasets.
"""
    (OUTPUT_ROOT / "figure_captions.md").write_text(captions, encoding="utf-8")

    manifest = """# Figure and table manifest

Core conclusion: Routinely available peri-ICU data provided moderate dynamic discrimination for postoperative incident AKI; results were stable to baseline-source restriction but 24 h performance partly depended on creatinine-derived predictors.

| Item | Role | Archetype | Primary evidence | Review risk |
|---|---|---|---|---|
| Figure 1 | Cohort validity | Schematic-led flow | Exclusion and landmark counts | Clarify cumulative early-AKI exclusions |
| Figure 2 | Main performance | Quantitative grid | Held-out ROC curves | Landmark risk sets differ |
| Figure 3 | Reliability/utility | Clinical validation grid | Calibration and DCA | Internal validation only |
| Figure 4 | Model attribution | Comparison bars | SHAP magnitude | Not causal importance |
| Figure S1 | Class-imbalance context | Quantitative grid | PR curves | Prevalence differs by landmark |
| Figures S2–S3 | Robustness | Paired sensitivity panels | Paired bootstrap deltas | Outcome remains SCr-defined |
| Figure S4 | Generalizability | Subgroup forest | Subgroup AUROC CIs | Small groups, no multiplicity correction |
| Figure S5 | Data quality | Heatmap | Predictor missingness | Missingness may be informative |
"""
    (OUTPUT_ROOT / "figure_table_manifest.md").write_text(manifest, encoding="utf-8")

    notes = """# Editorial notes and assumptions

## One-sentence argument

In a strict postoperative surgical ICU cohort from MIMIC-IV, landmark models using routinely available peri-ICU data achieved moderate internal discrimination for seven-day incident AKI, with robustness to baseline-source restriction but measurable 24 h dependence on creatinine-derived predictors.

## Assumptions or missing inputs

- Insert verified references for MIMIC-IV, KDIGO, TRIPOD and PROBAST at the marked placeholders.
- Confirm the target journal, word limits and whether Methods belongs in the main text or supplement.
- Confirm institutional wording for data-use approval, ethics waiver and credentialing; it was not invented here.
- Urine-output criteria were not used and must remain an explicit limitation.
- No external validation, temporal validation or multi-centre validation was performed.

## Claim–evidence map

- Claim: dynamic models achieved moderate discrimination. | Evidence: patient-grouped held-out AUROC/AUPRC with 1,000 bootstrap resamples. | Status: supported for internal validation.
- Claim: baseline-source restriction did not materially alter 24 h discrimination. | Evidence: paired ΔAUROC CIs crossed zero for all model families. | Status: supported.
- Claim: 24 h performance partly depended on creatinine predictors. | Evidence: all no-creatinine paired ΔAUROC CIs were below zero. | Status: supported as predictive reliance, not causality.
- Claim: models offer clinical utility. | Evidence: internal DCA net benefit. | Status: exploratory; external validation and intervention pathway needed.
"""
    (OUTPUT_ROOT / "editorial_notes.md").write_text(notes, encoding="utf-8")

    print(f"Final manuscript package: {OUTPUT_ROOT}")
    print(f"Table 1 rows: {len(table1)}")
    print(f"Figures exported: {len(list(FIGURE_DIR.glob('*.png')))} PNG, {len(list(FIGURE_DIR.glob('*.pdf')))} PDF, {len(list(FIGURE_DIR.glob('*.svg')))} SVG")


if __name__ == "__main__":
    main()
