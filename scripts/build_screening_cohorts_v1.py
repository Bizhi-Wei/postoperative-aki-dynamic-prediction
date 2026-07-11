"""Build two auditable postoperative-AKI screening cohorts from MIMIC-IV v3.1.

Version A is a strict surgical postoperative ICU screen. Version B preserves the
existing broad procedure cohort for sensitivity analysis. AKI labels inherited
from the old extract are explicitly marked provisional and must be recomputed
after the index time is finalized.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Set MIMIC_IV_ROOT when the raw MIMIC-IV release is stored elsewhere.
DATA_ROOT = Path(os.environ.get("MIMIC_IV_ROOT", str(PROJECT_ROOT.parent)))
LEGACY_PROJECT = DATA_ROOT / "MIMIC-IV_AKI_Project"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "screening"

SURGICAL_ICUS = {
    "Cardiac Vascular Intensive Care Unit (CVICU)",
    "Surgical Intensive Care Unit (SICU)",
    "Trauma SICU (TSICU)",
    "Neuro Surgical Intensive Care Unit (Neuro SICU)",
    "Medical/Surgical Intensive Care Unit (MICU/SICU)",
    "PACU",
}

# Exclusions take precedence over inclusion rules. These are deliberately
# conservative because Version A is intended to have high surgical specificity.
EXCLUSION_RULES = {
    "obstetric": r"obstetric|cesarean|caesarean|delivery|abortion|fetal|placenta|puerper|episiotomy",
    "imaging_or_diagnostic": r"radiograph|x-ray|fluoroscop|angiograph|arteriograph|tomograph|ultrasonograph|imaging|scan\b|electrocardio|ecg\b|measurement of|sampling and pressure|diagnostic",
    "airway_or_ventilation": r"respiratory ventilation|mechanical ventilation|endotracheal|intubat|extubat|insertion of airway|respiratory measurement|spirometr|oxygenation",
    "vascular_access_or_monitoring": r"venous catheter|central venous|infusion device|monitoring device|arterial catheter|cardiac output|swan.ganz|pressure monitoring|vascular access",
    "nutrition_or_infusion": r"nutritional substance|enteral infusion|parenteral nutrition|transfusion|infusion of|injection of",
    "dialysis_or_rrt": r"hemodialysis|haemodialysis|hemofiltration|haemofiltration|renal replacement|peritoneal dialysis|arteriovenous fistula",
    "routine_nonoperative": r"spinal tap|lumbar puncture|biopsy\b|endoscop|bronchoscop|colonoscopy|gastroscopy|cardiopulmonary resuscitation|chest tube|thoracentesis|paracentesis|without internal fixation",
}

INCLUSION_RULES = {
    "cardiac": r"coronary artery bypass|bypass coronary artery|coronary bypass|aorto.?coronary bypass|internal mammary.coronary|cabg|replacement of .* valve|repair of .* valve|valvuloplasty|open heart|heart transplant|transplantation of heart|ventricular assist device|repair of .* aorta|replacement of .* aorta|resection of .* aorta",
    "vascular": r"endarterectomy|embolectomy|thrombectomy|vascular reconstruction|revascularization|bypass .* artery|arterial bypass|repair of .* artery|replacement of .* artery|aneurysm repair|repair of aneurysm|ligation of .* artery|embolization|occlusion of .* artery",
    "general_gi_hepatobiliary": r"laparotomy|exploration of abdomen|abdominal exploration|gastrectomy|colectomy|hemicolectomy|proctectomy|enterectomy|resection of .* (stomach|intestin|colon|rectum|liver|hepatic|pancrea)|excision of .* (stomach|intestin|colon|rectum|liver|hepatic|pancrea)|appendectomy|cholecystectomy|pancreatectomy|hepatectomy|splenectomy|anastomosis of .* (stomach|intestin|colon)|bypass .* (stomach|intestin)|repair of .* (stomach|intestin|colon|rectum)|drainage of .* (abdominal|peritoneal|retroperitoneal|hepatic|liver|pancrea|biliary)|abdominal drainage",
    "orthopedic_major": r"open reduction|internal fixation|orif|joint replacement|replacement of .* (hip|knee|shoulder|elbow)|arthroplasty|spinal fusion|fusion of .* vertebral|laminectomy|decompression of spinal|fixation of .* (femur|tibia|fibula|humerus|pelvis|vertebra)|repair of fracture|reposition of .* (femur|tibia|fibula|humerus|pelvis|vertebra).*(internal fixation|external fixation)",
    "neurosurgery": r"craniotomy|craniectomy|resection of .* brain|excision of .* brain|drainage of .* (intracranial|subdural|epidural|cerebral)|repair of .* (brain|mening|dura)|ventriculostomy|ventricular shunt|neurostimulator|lobectomy of brain",
    "thoracic_respiratory": r"thoracotomy|lobectomy of lung|pneumonectomy|resection of .* lung|excision of .* lung|segmentectomy|wedge resection of lung|lung transplant|transplantation of lung|pleurectomy|decortication of lung|repair of .* bronch|resection of .* bronch|mediastinotomy|mediastinal exploration",
}


def normalize_title(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def first_matching_rule(title: str, rules: dict[str, str]) -> str | None:
    for name, pattern in rules.items():
        if re.search(pattern, title, flags=re.IGNORECASE):
            return name
    return None


def classify_title(title: object) -> tuple[str, str]:
    normalized = normalize_title(title)
    excluded_by = first_matching_rule(normalized, EXCLUSION_RULES)
    if excluded_by:
        return "exclude", excluded_by
    included_by = first_matching_rule(normalized, INCLUSION_RULES)
    if included_by:
        return "include", included_by
    return "review", "no_rule_match"


def join_unique(values: pd.Series) -> str:
    return " | ".join(sorted({str(value) for value in values if pd.notna(value)}))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    procedures = pd.read_csv(
        DATA_ROOT / "hosp" / "procedures_icd.csv",
        dtype={"icd_code": "string"},
        parse_dates=["chartdate"],
    )
    procedure_dictionary = pd.read_csv(
        DATA_ROOT / "hosp" / "d_icd_procedures.csv",
        dtype={"icd_code": "string"},
    )
    procedures = procedures.merge(
        procedure_dictionary, on=["icd_code", "icd_version"], how="left", validate="many_to_one"
    )

    unique_codes = procedures[["icd_code", "icd_version", "long_title"]].drop_duplicates().copy()
    classifications = unique_codes["long_title"].apply(classify_title)
    unique_codes[["version_a_status", "version_a_category_or_reason"]] = pd.DataFrame(
        classifications.tolist(), index=unique_codes.index
    )
    procedures = procedures.merge(
        unique_codes,
        on=["icd_code", "icd_version", "long_title"],
        how="left",
        validate="many_to_one",
    )

    icu = pd.read_csv(DATA_ROOT / "icu" / "icustays.csv", parse_dates=["intime", "outtime"])
    icu["icu_date"] = icu["intime"].dt.normalize()
    procedure_candidates = icu.merge(
        procedures,
        on=["subject_id", "hadm_id"],
        how="inner",
        validate="many_to_many",
    )
    procedure_candidates["days_from_procedure_to_icu"] = (
        procedure_candidates["icu_date"] - procedure_candidates["chartdate"]
    ).dt.days

    a_candidates = procedure_candidates.loc[
        procedure_candidates["first_careunit"].isin(SURGICAL_ICUS)
        & procedure_candidates["days_from_procedure_to_icu"].isin([0, 1])
        & procedure_candidates["version_a_status"].eq("include")
    ].copy()

    # Prefer an operation on the ICU admission date; otherwise use the preceding date.
    a_candidates["best_day_gap"] = a_candidates.groupby("stay_id")["days_from_procedure_to_icu"].transform("min")
    a_candidates = a_candidates.loc[
        a_candidates["days_from_procedure_to_icu"].eq(a_candidates["best_day_gap"])
    ]
    code_usage = (
        a_candidates.groupby(["icd_code", "icd_version"], as_index=False)
        .agg(version_a_qualifying_stays=("stay_id", "nunique"))
    )
    code_review = unique_codes.merge(code_usage, on=["icd_code", "icd_version"], how="left")
    code_review["version_a_qualifying_stays"] = (
        code_review["version_a_qualifying_stays"].fillna(0).astype(int)
    )
    code_review.sort_values(
        ["version_a_status", "version_a_qualifying_stays", "version_a_category_or_reason"],
        ascending=[True, False, True],
    ).to_csv(OUTPUT_DIR / "procedure_code_review.csv", index=False)
    a_index = (
        a_candidates.groupby("stay_id", as_index=False)
        .agg(
            index_surgery_date=("chartdate", "first"),
            surgery_categories=("version_a_category_or_reason", join_unique),
            qualifying_icd_codes=("icd_code", join_unique),
            qualifying_procedure_titles=("long_title", join_unique),
            n_qualifying_codes=("icd_code", "nunique"),
            days_from_procedure_to_icu=("days_from_procedure_to_icu", "first"),
        )
    )

    legacy_analysis = pd.read_csv(LEGACY_PROJECT / "analysis_dataset.csv", low_memory=False)
    cohort_a_all = legacy_analysis.merge(a_index, on="stay_id", how="inner", validate="one_to_one")
    cohort_a_all.insert(0, "cohort_version", "A_strict_surgical_postop_icu")
    cohort_a_all.insert(1, "aki_label_status", "provisional_recompute_after_index_finalization")
    cohort_a_all = cohort_a_all.sort_values(["subject_id", "intime", "stay_id"])
    cohort_a_all.to_csv(OUTPUT_DIR / "cohort_version_a_all_stays_audit.csv", index=False)
    cohort_a = cohort_a_all.drop_duplicates("subject_id", keep="first").copy()
    cohort_a.to_csv(OUTPUT_DIR / "cohort_version_a_screen.csv", index=False)

    cohort_b_all = legacy_analysis.copy()
    cohort_b_all.insert(0, "cohort_version", "B_broad_procedure_sensitivity")
    cohort_b_all.insert(1, "aki_label_status", "provisional_legacy_definition")
    cohort_b_all = cohort_b_all.sort_values(["subject_id", "intime", "stay_id"])
    cohort_b_all.to_csv(OUTPUT_DIR / "cohort_version_b_all_stays_audit.csv", index=False)
    cohort_b = cohort_b_all.drop_duplicates("subject_id", keep="first").copy()
    cohort_b.to_csv(OUTPUT_DIR / "cohort_version_b_screen.csv", index=False)

    a_candidates[
        [
            "subject_id", "hadm_id", "stay_id", "first_careunit", "intime", "outtime",
            "chartdate", "days_from_procedure_to_icu", "icd_code", "icd_version", "long_title",
            "version_a_category_or_reason",
        ]
    ].sort_values(["stay_id", "chartdate", "icd_version", "icd_code"]).to_csv(
        OUTPUT_DIR / "cohort_a_qualifying_procedures.csv", index=False
    )

    category_counts = (
        a_candidates[["stay_id", "version_a_category_or_reason"]]
        .drop_duplicates()
        .groupby("version_a_category_or_reason")
        .size()
        .rename("stays")
        .sort_values(ascending=False)
        .reset_index()
    )
    category_counts.to_csv(OUTPUT_DIR / "cohort_a_category_counts.csv", index=False)

    summary = pd.DataFrame(
        [
            {"metric": "version_b_all_stays", "value": len(cohort_b_all)},
            {"metric": "version_b_main_first_stay", "value": len(cohort_b)},
            {"metric": "version_a_all_stays", "value": len(cohort_a_all)},
            {"metric": "version_a_main_first_stay", "value": len(cohort_a)},
            {"metric": "version_a_aki_provisional_n", "value": int(cohort_a.aki.sum())},
            {"metric": "version_a_aki_provisional_percent", "value": round(cohort_a.aki.mean() * 100, 2)},
            {"metric": "procedure_codes_include", "value": int((unique_codes.version_a_status == "include").sum())},
            {"metric": "procedure_codes_exclude", "value": int((unique_codes.version_a_status == "exclude").sum())},
            {"metric": "procedure_codes_review", "value": int((unique_codes.version_a_status == "review").sum())},
        ]
    )
    summary.to_csv(OUTPUT_DIR / "screening_summary.csv", index=False)

    print(summary.to_string(index=False))
    print("\nVersion A categories (a stay may have more than one category):")
    print(category_counts.to_string(index=False))
    print(f"\nOutputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
