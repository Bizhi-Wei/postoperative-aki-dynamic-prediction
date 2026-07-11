"""v20 external clinical utility and competing-risk sensitivity analyses.

Part A: Decision-curve and threshold-policy evaluation on the v17 held-out
eICU hospital partition, comparing frozen and recalibrated probabilities.

Part B: Competing-risk audit for AKI versus death/discharge before the 7-day
follow-up horizon. MIMIC uses hospital exit (discharge/death) because labs can
continue after ICU transfer; eICU uses ICU-unit exit because eICU laboratory
data are unit-stay scoped. This is an outcome-ascertainment sensitivity, not a
replacement of the prespecified binary AKI outcome.
"""

from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
V17_DIR = PROJECT_ROOT / "outputs" / "modeling_v17_eicu_recalibration_heterogeneity"
V16_DIR = PROJECT_ROOT / "outputs" / "modeling_v16_eicu_external_validation"
V3_1_FILE = PROJECT_ROOT / "outputs" / "finalized_v3_1" / "cohort_v3_1_strict_main_evaluable.csv"
EICU_ROOT = Path(os.environ.get("EICU_ROOT", str(PROJECT_ROOT.parents[1] / "eicu-collaborative-research-database-2.0")))
EICU_PATIENT_FILE = EICU_ROOT / "patient.csv" / "patient.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "modeling_v20_external_utility_competing_risk"

OUTCOME = "outcome_aki_after_landmark_to_7d"
HORIZON_HOURS = 168.0
LANDMARKS = (0, 6, 24)


def decision_curve(y: np.ndarray, p: np.ndarray, thresholds: np.ndarray) -> pd.DataFrame:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    rows = []
    prevalence = y.mean()
    for threshold in thresholds:
        alert = p >= threshold
        tp = int(np.sum(alert & (y == 1)))
        fp = int(np.sum(alert & (y == 0)))
        weight = threshold / (1 - threshold)
        rows.append({
            "threshold": float(threshold), "net_benefit": float(tp / len(y) - fp / len(y) * weight),
            "treat_all_net_benefit": float(prevalence - (1 - prevalence) * weight),
            "treat_none_net_benefit": 0.0, "alert_rate": float(alert.mean()), "true_positive_n": tp, "false_positive_n": fp,
        })
    return pd.DataFrame(rows)


def threshold_policy(y: np.ndarray, p: np.ndarray, thresholds: list[float]) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        alert = p >= threshold
        tp = int(np.sum(alert & (y == 1))); fp = int(np.sum(alert & (y == 0)))
        fn = int(np.sum((~alert) & (y == 1))); tn = int(np.sum((~alert) & (y == 0)))
        rows.append({
            "threshold": threshold, "n": int(len(y)), "event_n": int(y.sum()), "event_rate": float(y.mean()),
            "alert_n": int(alert.sum()), "alert_rate": float(alert.mean()), "true_positive_n": tp,
            "false_positive_n": fp, "false_negative_n": fn, "true_negative_n": tn,
            "sensitivity": float(tp / (tp + fn)) if tp + fn else np.nan,
            "specificity": float(tn / (tn + fp)) if tn + fp else np.nan,
            "positive_predictive_value": float(tp / (tp + fp)) if tp + fp else np.nan,
            "false_alerts_per_100_patients": float(100 * fp / len(y)),
            "alerts_per_100_patients": float(100 * alert.mean()),
            "events_captured_per_100_patients": float(100 * tp / len(y)),
            "net_benefit": float(tp / len(y) - fp / len(y) * threshold / (1 - threshold)),
        })
    return pd.DataFrame(rows)


def external_clinical_utility() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pred = pd.read_csv(V17_DIR / "model_v17_heldout_hospital_recalibrated_predictions.csv", low_memory=False)
    # A competing-risk utility sensitivity confines each landmark analysis to
    # patients still in the eICU unit and defines benefit for AKI occurring
    # before unit exit. This prevents an early ICU exit from being treated as a
    # continuing 24-h ICU prediction opportunity.
    eicu_exit = eicu_competing_records().rename(columns={"hospital_exit_hours": "unit_exit_hours", "hospital_death": "icu_death"})
    pred = pred.merge(eicu_exit[["stay_id", "aki_final", "aki_onset_hours_final", "unit_exit_hours", "icu_death"]], on="stay_id", how="left", validate="many_to_one")
    method_columns = {
        "frozen_unrecalibrated": "predicted_risk_frozen",
        "intercept_only_update": "predicted_risk_intercept_updated",
        "logistic_recalibration_update": "predicted_risk_logistic_recalibrated",
    }
    thresholds = np.round(np.arange(.05, .51, .01), 2)
    policy_thresholds = [.10, .15, .20, .25, .30, .40, .50]
    dca_rows, policy_rows, perf_rows = [], [], []
    for landmark in LANDMARKS:
        original = pred.loc[pred["landmark_hours"].eq(landmark)].copy()
        active = original.loc[pd.to_numeric(original["unit_exit_hours"], errors="coerce").gt(landmark)].copy()
        onset = pd.to_numeric(active["aki_onset_hours_final"], errors="coerce")
        exit_time = pd.to_numeric(active["unit_exit_hours"], errors="coerce")
        active["competing_aki_before_unit_exit"] = (
            active["aki_final"].fillna(0).astype(float).eq(1)
            & onset.gt(landmark)
            & onset.le(exit_time)
            & onset.le(HORIZON_HOURS)
        ).astype(int)
        for risk_definition, d, outcome_col in [
            ("v17_original_risk_set", original, OUTCOME),
            ("active_icu_competing_risk_sensitivity", active, "competing_aki_before_unit_exit"),
        ]:
            y = d[outcome_col].astype(int).to_numpy()
            for method, column in method_columns.items():
                p = d[column].to_numpy()
                dc = decision_curve(y, p, thresholds)
                dc["landmark_hours"] = landmark; dc["method"] = method; dc["risk_definition"] = risk_definition
                dca_rows.append(dc)
                if method in {"frozen_unrecalibrated", "logistic_recalibration_update"}:
                    po = threshold_policy(y, p, policy_thresholds)
                    po["landmark_hours"] = landmark; po["method"] = method; po["risk_definition"] = risk_definition
                    policy_rows.append(po)
                perf_rows.append({
                    "landmark_hours": landmark, "method": method, "risk_definition": risk_definition, "n": len(y), "event_n": int(y.sum()), "event_rate": float(y.mean()),
                    "auroc": float(roc_auc_score(y, p)), "auprc": float(average_precision_score(y, p)), "brier_score": float(brier_score_loss(y, p)),
                })
    return pd.concat(dca_rows, ignore_index=True), pd.concat(policy_rows, ignore_index=True), pd.DataFrame(perf_rows)


def bool_series(x: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(x):
        return x.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(x):
        return pd.to_numeric(x, errors="coerce").fillna(0).astype(int).astype(bool)
    return x.astype("string").str.strip().str.lower().isin(["true", "1", "1.0", "yes", "expired", "death"])


def aalen_johansen(time: np.ndarray, event: np.ndarray, horizon: float) -> pd.DataFrame:
    """CIFs for event=1 AKI, 2 death, 3 discharge; event=0 censored."""
    time = np.minimum(np.asarray(time, dtype=float), horizon)
    event = np.asarray(event, dtype=int)
    unique = np.sort(np.unique(time[(event > 0) & (time <= horizon)]))
    survival, cif_aki, cif_death, cif_discharge = 1.0, 0.0, 0.0, 0.0
    rows = [{"time_hours": 0.0, "survival_free_of_any_event": survival, "cif_aki": 0.0, "cif_death": 0.0, "cif_discharge": 0.0}]
    for t in unique:
        at_risk = int(np.sum(time >= t))
        if at_risk == 0:
            continue
        d1 = int(np.sum((time == t) & (event == 1)))
        d2 = int(np.sum((time == t) & (event == 2)))
        d3 = int(np.sum((time == t) & (event == 3)))
        cif_aki += survival * d1 / at_risk
        cif_death += survival * d2 / at_risk
        cif_discharge += survival * d3 / at_risk
        survival *= 1 - (d1 + d2 + d3) / at_risk
        rows.append({"time_hours": float(t), "survival_free_of_any_event": survival, "cif_aki": cif_aki, "cif_death": cif_death, "cif_discharge": cif_discharge})
    return pd.DataFrame(rows)


def classify_competing(event_time: float, exit_time: float, death: bool, landmark: int) -> tuple[float, int]:
    """Return time since landmark and type: AKI 1, death 2, discharge 3, censor 0."""
    event_time = float(event_time) if pd.notna(event_time) else np.inf
    exit_time = float(exit_time) if pd.notna(exit_time) else np.inf
    # A patient must still be event-free and observable at the landmark.
    if exit_time <= landmark:
        return 0.0, 0
    if event_time <= landmark:
        return 0.0, 0
    end = landmark + HORIZON_HOURS
    if event_time <= min(exit_time, end):
        return event_time - landmark, 1
    if exit_time < min(event_time, end):
        return exit_time - landmark, 2 if death else 3
    return HORIZON_HOURS, 0


def mimic_competing_records() -> pd.DataFrame:
    data = pd.read_csv(V3_1_FILE, low_memory=False).copy()
    data["intime"] = pd.to_datetime(data["intime"], errors="coerce")
    data["dischtime"] = pd.to_datetime(data["dischtime"], errors="coerce")
    data["aki_onset_hours_final"] = pd.to_numeric(data["aki_onset_hours_final"], errors="coerce")
    data["hospital_exit_hours"] = (data["dischtime"] - data["intime"]).dt.total_seconds() / 3600
    data["hospital_death"] = bool_series(data["hosp_death"] if "hosp_death" in data else data["hospital_expire_flag"])
    data["database"] = "MIMIC-IV strict primary"
    return data[["stay_id", "aki_final", "aki_onset_hours_final", "hospital_exit_hours", "hospital_death", "database"]]


def eicu_competing_records() -> pd.DataFrame:
    cohort = pd.read_csv(V16_DIR / "cohort_v16_eicu_external_validation.csv.gz", low_memory=False, usecols=["stay_id", "incident_aki_evaluable", "aki_final", "aki_onset_hours_final"])
    patient = pd.read_csv(EICU_PATIENT_FILE, usecols=["patientunitstayid", "unitdischargeoffset", "unitdischargestatus"], low_memory=False)
    patient = patient.rename(columns={"patientunitstayid": "stay_id"})
    patient["stay_id"] = pd.to_numeric(patient["stay_id"], errors="coerce").astype(int)
    patient["unit_exit_hours"] = pd.to_numeric(patient["unitdischargeoffset"], errors="coerce") / 60
    patient["icu_death"] = patient["unitdischargestatus"].astype("string").str.lower().eq("expired")
    data = cohort.merge(patient[["stay_id", "unit_exit_hours", "icu_death"]], on="stay_id", how="left", validate="one_to_one")
    data = data.loc[data["incident_aki_evaluable"].fillna(False).astype(bool)].copy()
    data["database"] = "eICU external"
    return data.rename(columns={"unit_exit_hours": "hospital_exit_hours", "icu_death": "hospital_death"})[["stay_id", "aki_final", "aki_onset_hours_final", "hospital_exit_hours", "hospital_death", "database"]]


def competing_risk_analysis() -> tuple[pd.DataFrame, pd.DataFrame]:
    records = pd.concat([mimic_competing_records(), eicu_competing_records()], ignore_index=True)
    summary_rows, curves = [], []
    for database, group in records.groupby("database", sort=False):
        group["aki_final"] = bool_series(group["aki_final"])
        for landmark in LANDMARKS:
            risk = group.loc[~(group["aki_final"] & pd.to_numeric(group["aki_onset_hours_final"], errors="coerce").le(landmark))].copy()
            classified = risk.apply(lambda r: classify_competing(r["aki_onset_hours_final"] if bool(r["aki_final"]) else np.nan, r["hospital_exit_hours"], bool(r["hospital_death"]), landmark), axis=1)
            risk[["followup_hours", "competing_event_type"]] = pd.DataFrame(classified.tolist(), index=risk.index)
            risk = risk.loc[risk["followup_hours"].gt(0)].copy()
            curve = aalen_johansen(risk["followup_hours"].to_numpy(), risk["competing_event_type"].to_numpy(), HORIZON_HOURS)
            curve["database"] = database; curve["landmark_hours"] = landmark
            curves.append(curve)
            final = curve.iloc[-1]
            original_binary = float(np.mean(risk["competing_event_type"].eq(1) | ((risk["aki_final"]) & pd.to_numeric(risk["aki_onset_hours_final"], errors="coerce").gt(landmark))))
            # Original binary outcome after landmark counts AKI even if it occurred after an exit event; by construction this is the risk-set rate.
            binary_after_landmark = float(((risk["aki_final"]) & pd.to_numeric(risk["aki_onset_hours_final"], errors="coerce").gt(landmark)).mean())
            summary_rows.append({
                "database": database, "landmark_hours": landmark, "risk_set_n": len(risk),
                "binary_aki_rate_after_landmark": binary_after_landmark,
                "aki_first_n": int(risk["competing_event_type"].eq(1).sum()), "death_first_n": int(risk["competing_event_type"].eq(2).sum()),
                "discharge_first_n": int(risk["competing_event_type"].eq(3).sum()), "censored_at_7d_n": int(risk["competing_event_type"].eq(0).sum()),
                "cif_aki_7d": float(final["cif_aki"]), "cif_death_7d": float(final["cif_death"]), "cif_discharge_7d": float(final["cif_discharge"]),
                "competing_event_definition": "hospital discharge/death" if database.startswith("MIMIC") else "ICU-unit discharge/death",
            })
    return pd.DataFrame(summary_rows), pd.concat(curves, ignore_index=True)


def make_figures(dca: pd.DataFrame, policy: pd.DataFrame, curves: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    colors = {"frozen_unrecalibrated": "#4c78a8", "intercept_only_update": "#59a14f", "logistic_recalibration_update": "#e17c05"}
    labels = {"frozen_unrecalibrated": "Frozen", "intercept_only_update": "Intercept update", "logistic_recalibration_update": "Logistic recalibration"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), dpi=180, sharey=True)
    for ax, landmark in zip(axes, LANDMARKS):
        d = dca.loc[(dca["landmark_hours"].eq(landmark)) & (dca["risk_definition"].eq("active_icu_competing_risk_sensitivity"))]
        for method, group in d.groupby("method"):
            ax.plot(group["threshold"], group["net_benefit"], color=colors[method], linewidth=1.8, label=labels[method])
        first = d.iloc[0:len(d.loc[d["method"].eq("frozen_unrecalibrated")])]
        ax.plot(first["threshold"], first["treat_all_net_benefit"], color="#687080", linestyle="--", label="Treat all")
        ax.axhline(0, color="#30343b", linewidth=.8, label="Treat none")
        ax.set_title(f"{landmark} h"); ax.set_xlabel("Risk threshold")
        ax.set_ylim(-.05, .20)
    axes[0].set_ylabel("Net benefit")
    axes[-1].legend(frameon=False, fontsize=7, loc="upper right")
    fig.suptitle("Clinical utility in held-out eICU hospitals", y=1.03)
    fig.text(.5, .965, "Active ICU risk set; AKI before unit exit; focused net-benefit scale (−0.05 to 0.20)", ha="center", fontsize=9, color="#5f6675")
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(OUTPUT_DIR / "figure_v20_external_decision_curve.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), dpi=180, sharey=True)
    for ax, landmark in zip(axes, LANDMARKS):
        d = policy.loc[(policy["landmark_hours"] == landmark) & (policy["method"] == "logistic_recalibration_update") & (policy["risk_definition"] == "active_icu_competing_risk_sensitivity")].sort_values("threshold")
        ax.plot(d["alerts_per_100_patients"], d["events_captured_per_100_patients"], marker="o", color="#e17c05")
        for row in d.itertuples():
            ax.annotate(f"{row.threshold:.2f}", (row.alerts_per_100_patients, row.events_captured_per_100_patients), xytext=(3, 3), textcoords="offset points", fontsize=7)
        ax.set_title(f"{landmark} h"); ax.set_xlabel("Alerts per 100 patients")
    axes[0].set_ylabel("AKI events captured per 100 patients")
    fig.suptitle("Threshold-policy trade-off after logistic recalibration", y=1.03)
    fig.text(.5, .965, "Held-out eICU hospitals; active ICU risk set; AKI before unit exit; labels are absolute-risk alert thresholds", ha="center", fontsize=8.5, color="#5f6675")
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(OUTPUT_DIR / "figure_v20_external_threshold_policy.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), dpi=180, sharey=True)
    colors2 = {"cif_aki": "#e17c05", "cif_death": "#af7aa1", "cif_discharge": "#4c78a8"}
    labels2 = {"cif_aki": "AKI first", "cif_death": "Death first", "cif_discharge": "Discharge first"}
    for ax, database in zip(axes, ["MIMIC-IV strict primary", "eICU external"]):
        d = curves.loc[(curves["database"] == database) & (curves["landmark_hours"] == 0)]
        for col in ("cif_aki", "cif_death", "cif_discharge"):
            ax.step(d["time_hours"], d[col] * 100, where="post", color=colors2[col], label=labels2[col])
        ax.set_title(database); ax.set_xlabel("Hours after ICU admission"); ax.set_xlim(0, 168)
    axes[0].set_ylabel("Cumulative incidence (%)")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle("Competing-risk audit through day 7", y=1.03)
    fig.text(.5, .965, "MIMIC: hospital exit; eICU: ICU-unit exit because laboratory follow-up is unit-stay scoped", ha="center", fontsize=8.5, color="#5f6675")
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(OUTPUT_DIR / "figure_v20_competing_risk_cumulative_incidence.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_readme(dca: pd.DataFrame, policy: pd.DataFrame, perf: pd.DataFrame, competing: pd.DataFrame) -> None:
    lines = ["# v20 external clinical utility and competing-risk sensitivity", "", "## External clinical utility", ""]
    lines.append("Decision-curve and threshold-policy analyses use the v17 held-out hospital partition. The external logistic recalibration update is evaluated separately from frozen predictions; recalibration changes absolute risk but not ranking discrimination. In addition to the original v17 risk set, a competing-risk utility sensitivity retains only patients still in the eICU unit at the landmark and defines an event as AKI before unit exit.")
    for landmark in LANDMARKS:
        d = perf.loc[(perf["landmark_hours"] == landmark) & (perf["method"] == "logistic_recalibration_update") & (perf["risk_definition"] == "active_icu_competing_risk_sensitivity")].iloc[0]
        lines.append(f"- {landmark} h logistic recalibration, active-ICU competing-risk sensitivity: n={int(d.n):,}, AKI-before-unit-exit events={int(d.event_n):,} ({d.event_rate*100:.1f}%), AUROC {d.auroc:.3f}, Brier {d.brier_score:.3f}.")
    lines.extend(["", "## Competing-risk sensitivity", ""])
    for r in competing.itertuples():
        lines.append(f"- {r.database}, {int(r.landmark_hours)} h: risk set n={int(r.risk_set_n):,}; 7-day CIF AKI first {r.cif_aki_7d*100:.1f}%, death first {r.cif_death_7d*100:.1f}%, discharge first {r.cif_discharge_7d*100:.1f}%; binary AKI rate {r.binary_aki_rate_after_landmark*100:.1f}%.")
    lines.extend(["", "## Interpretation", "", "The original binary KDIGO outcome remains primary. Competing-risk estimates quantify the degree to which death or exit precludes observed AKI within the available follow-up system. MIMIC hospital exit and eICU ICU-unit exit are deliberately different because their laboratory follow-up scopes differ; estimates should not be compared as identical censoring processes."])
    (OUTPUT_DIR / "audit_v20_readme.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dca, policy, perf = external_clinical_utility()
    competing, curves = competing_risk_analysis()
    dca.to_csv(OUTPUT_DIR / "analysis_v20_external_decision_curve.csv", index=False)
    policy.to_csv(OUTPUT_DIR / "analysis_v20_external_threshold_policy.csv", index=False)
    perf.to_csv(OUTPUT_DIR / "analysis_v20_external_utility_performance.csv", index=False)
    competing.to_csv(OUTPUT_DIR / "analysis_v20_competing_risk_summary.csv", index=False)
    curves.to_csv(OUTPUT_DIR / "analysis_v20_competing_risk_cumulative_incidence.csv", index=False)
    make_figures(dca, policy, curves)
    write_readme(dca, policy, perf, competing)
    print(f"Wrote v20 outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
