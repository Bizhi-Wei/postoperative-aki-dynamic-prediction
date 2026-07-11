"""Add urine-output KDIGO AKI outcome and OR/PACU proxy variables.

Scope:
- Primary strict cohort only: cohort_v3_1_strict_main_evaluable.csv.
- Adds a conservative urine-output AKI assessment using ICU outputevents.
- Adds auditable OR/PACU charted proxy variables and early ICU vasoactive/PRBC
  exposure variables. These are *not* claimed to be complete anesthesia records.

Urine-output KDIGO implementation:
- Use urine-like outputevents itemids.
- Use patient weight from ICU inputevents/procedureevents patientweight,
  with chartevents admission/daily weight fallback.
- Rolling 6/12/24 h windows are evaluated hourly after ICU admission.
- A window is evaluable only if at least 75% of expected hourly bins have a
  urine-output observation, reducing false oliguria from missing charting.
- Stage 1: <0.5 mL/kg/h for 6 h.
- Stage 2: <0.5 mL/kg/h for 12 h.
- Stage 3: <0.3 mL/kg/h for 24 h. Anuria for 12 h is not used because
  absence of output charting cannot be safely treated as zero in this dataset.
"""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MIMIC_ROOT = PROJECT_ROOT.parent
INPUT_COHORT = PROJECT_ROOT / "outputs" / "finalized_v3_1" / "cohort_v3_1_strict_main_evaluable.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "v6_urine_output_or_pacu"
CACHE_DIR = OUTPUT_DIR / "cache"

OUTPUTEVENTS = MIMIC_ROOT / "icu" / "outputevents.csv"
INPUTEVENTS = MIMIC_ROOT / "icu" / "inputevents.csv"
PROCEDUREEVENTS = MIMIC_ROOT / "icu" / "procedureevents.csv"
CHARTEVENTS = MIMIC_ROOT / "icu" / "chartevents.csv"

CHUNK_SIZE = 1_000_000
FOLLOWUP_HOURS = 7 * 24
COVERAGE_FRACTION = 0.75

warnings.filterwarnings("ignore", category=PerformanceWarning)

# Urine-like outputs. Irrigant-mixed itemids are excluded from the primary urine
# outcome and counted separately for audit.
URINE_ITEMIDS_PRIMARY = {
    226557,  # R Ureteral Stent
    226558,  # L Ureteral Stent
    226559,  # Foley
    226560,  # Void
    226561,  # Condom Cath
    226563,  # Suprapubic
    226564,  # R Nephrostomy
    226565,  # L Nephrostomy
    226567,  # Straight Cath
    226627,  # OR Urine
    226631,  # PACU Urine
    226713,  # Incontinent/voids estimate
}
URINE_ITEMIDS_MIXED_IRRIGANT = {226566, 227489}

OR_PACU_OUTPUT_ITEMIDS = {
    226626: "or_ebl_ml",
    226627: "or_urine_ml",
    226629: "pacu_ebl_ml",
    226631: "pacu_urine_ml",
}
OR_PACU_INPUT_ITEMIDS = {
    226364: "or_crystalloid_ml",
    226375: "pacu_crystalloid_ml",
}
PRBC_ITEMIDS = {225168}
VASOACTIVE_ITEMIDS = {
    221906: "norepinephrine",
    221749: "phenylephrine",
    229630: "phenylephrine",
    229632: "phenylephrine",
    221289: "epinephrine",
    229617: "epinephrine",
    222315: "vasopressin",
    221653: "dobutamine",
    221662: "dopamine",
}
WEIGHT_ITEMIDS = {224639, 226512}  # Daily Weight, Admission Weight (kg)


def bool_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    return s.astype("string").str.lower().isin(["true", "1", "yes"])


def load_cohort() -> pd.DataFrame:
    data = pd.read_csv(INPUT_COHORT, low_memory=False)
    for col in ["intime", "outtime", "admittime", "index_surgery_date", "aki_onset_time_final"]:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col], errors="coerce")
    required = {"subject_id", "hadm_id", "stay_id", "intime", "aki_final", "aki_stage_final", "aki_onset_hours_final"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing required cohort columns: {sorted(missing)}")
    if data["stay_id"].duplicated().any():
        raise ValueError("Cohort should be unique by stay_id")
    return data


def read_filtered_csv(path: Path, usecols: list[str], stay_ids: set[int], itemids: set[int] | None, time_cols: list[str], value_col: str | None = None) -> pd.DataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_name = path.stem + "_" + ("allitems" if itemids is None else str(abs(hash(tuple(sorted(itemids)))))) + ".csv.gz"
    cache_path = CACHE_DIR / cache_name
    if cache_path.exists():
        df = pd.read_csv(cache_path, low_memory=False)
        for col in time_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        return df
    frames: list[pd.DataFrame] = []
    for i, chunk in enumerate(pd.read_csv(path, usecols=usecols, chunksize=CHUNK_SIZE, low_memory=False), start=1):
        selected = chunk[chunk["stay_id"].isin(stay_ids)].copy()
        if itemids is not None:
            selected = selected[selected["itemid"].isin(itemids)].copy()
        if value_col is not None:
            selected[value_col] = pd.to_numeric(selected[value_col], errors="coerce")
            selected = selected[selected[value_col].notna()]
        if not selected.empty:
            for col in time_cols:
                selected[col] = pd.to_datetime(selected[col], errors="coerce")
            frames.append(selected)
        if i % 10 == 0:
            print(f"Scanned {i * CHUNK_SIZE:,} rows from {path.name}", flush=True)
    if frames:
        df = pd.concat(frames, ignore_index=True)
    else:
        df = pd.DataFrame(columns=usecols)
    df.to_csv(cache_path, index=False, compression="gzip")
    return df


def extract_outputevents(cohort: pd.DataFrame) -> pd.DataFrame:
    itemids = URINE_ITEMIDS_PRIMARY | URINE_ITEMIDS_MIXED_IRRIGANT | set(OR_PACU_OUTPUT_ITEMIDS)
    df = read_filtered_csv(
        OUTPUTEVENTS,
        ["subject_id", "hadm_id", "stay_id", "charttime", "itemid", "value", "valueuom"],
        set(cohort["stay_id"].astype(int)),
        itemids,
        ["charttime"],
        "value",
    )
    df = df[df["charttime"].notna() & df["value"].ge(0) & df["value"].le(100000)].copy()
    return df


def extract_inputevents(cohort: pd.DataFrame) -> pd.DataFrame:
    itemids = set(OR_PACU_INPUT_ITEMIDS) | PRBC_ITEMIDS | set(VASOACTIVE_ITEMIDS)
    df = read_filtered_csv(
        INPUTEVENTS,
        [
            "subject_id", "hadm_id", "stay_id", "starttime", "endtime", "itemid",
            "amount", "amountuom", "rate", "rateuom", "patientweight", "statusdescription",
        ],
        set(cohort["stay_id"].astype(int)),
        itemids,
        ["starttime", "endtime"],
        "amount",
    )
    df["patientweight"] = pd.to_numeric(df.get("patientweight"), errors="coerce")
    return df


def extract_weights_from_input_proc(cohort: pd.DataFrame) -> pd.DataFrame:
    stay_ids = set(cohort["stay_id"].astype(int))
    frames = []
    for path, tcols in [(INPUTEVENTS, ["starttime", "endtime"]), (PROCEDUREEVENTS, ["starttime", "endtime"])]:
        cache = CACHE_DIR / f"{path.stem}_patientweight.csv.gz"
        if cache.exists():
            df = pd.read_csv(cache, low_memory=False)
            for c in tcols:
                if c in df.columns:
                    df[c] = pd.to_datetime(df[c], errors="coerce")
        else:
            pieces = []
            usecols = ["subject_id", "hadm_id", "stay_id", *tcols, "patientweight"]
            for i, chunk in enumerate(pd.read_csv(path, usecols=usecols, chunksize=CHUNK_SIZE, low_memory=False), start=1):
                selected = chunk[chunk["stay_id"].isin(stay_ids)].copy()
                selected["patientweight"] = pd.to_numeric(selected["patientweight"], errors="coerce")
                selected = selected[selected["patientweight"].between(20, 300)]
                if not selected.empty:
                    for c in tcols:
                        selected[c] = pd.to_datetime(selected[c], errors="coerce")
                    pieces.append(selected)
                if i % 20 == 0:
                    print(f"Scanned {i * CHUNK_SIZE:,} rows for weights from {path.name}", flush=True)
            df = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=usecols)
            df.to_csv(cache, index=False, compression="gzip")
        frames.append(df.rename(columns={"patientweight": "weight_kg"}))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def extract_weights_from_chartevents(cohort: pd.DataFrame) -> pd.DataFrame:
    df = read_filtered_csv(
        CHARTEVENTS,
        ["subject_id", "hadm_id", "stay_id", "charttime", "itemid", "valuenum", "valueuom"],
        set(cohort["stay_id"].astype(int)),
        WEIGHT_ITEMIDS,
        ["charttime"],
        "valuenum",
    )
    df = df[df["valuenum"].between(20, 300)].copy()
    return df.rename(columns={"charttime": "starttime", "valuenum": "weight_kg"})


def derive_weight(cohort: pd.DataFrame) -> pd.DataFrame:
    weights = pd.concat([
        extract_weights_from_input_proc(cohort),
        extract_weights_from_chartevents(cohort),
    ], ignore_index=True, sort=False)
    if weights.empty:
        return cohort[["stay_id"]].assign(weight_kg=np.nan, weight_source="missing", weight_time=pd.NaT)
    weights["weight_time"] = pd.to_datetime(weights["starttime"], errors="coerce")
    merged = weights.merge(cohort[["stay_id", "intime"]], on="stay_id", how="inner")
    merged["abs_hours_from_icu"] = (merged["weight_time"] - merged["intime"]).abs().dt.total_seconds() / 3600
    merged = merged.sort_values(["stay_id", "abs_hours_from_icu"])
    chosen = merged.groupby("stay_id", as_index=False).first()
    chosen["weight_source"] = "icu_events_patientweight_or_chartweight"
    return chosen[["stay_id", "weight_kg", "weight_source", "weight_time"]]


def urine_aki_for_stay(row: pd.Series, urine: pd.DataFrame) -> dict[str, object]:
    stay_id = int(row["stay_id"])
    index_time = row["intime"]
    weight = row["weight_kg"]
    if pd.isna(weight) or weight <= 0:
        return {
            "uo_evaluable": False,
            "uo_ineligibility_reason": "missing_weight",
            "uo_aki": False,
            "uo_stage": 0,
            "uo_onset_time": pd.NaT,
            "uo_onset_hours": np.nan,
            "uo_min_6h_mlkg_hr": np.nan,
            "uo_min_12h_mlkg_hr": np.nan,
            "uo_min_24h_mlkg_hr": np.nan,
            "uo_observed_hour_fraction_7d": np.nan,
            "uo_total_ml_7d": np.nan,
        }
    followup_end = index_time + pd.Timedelta(hours=FOLLOWUP_HOURS)
    stay = urine[
        (urine["stay_id"].eq(stay_id))
        & (urine["charttime"].gt(index_time))
        & (urine["charttime"].le(followup_end))
        & (urine["itemid"].isin(URINE_ITEMIDS_PRIMARY))
    ].copy()
    if stay.empty:
        return {
            "uo_evaluable": False,
            "uo_ineligibility_reason": "no_urine_output_records_7d",
            "uo_aki": False,
            "uo_stage": 0,
            "uo_onset_time": pd.NaT,
            "uo_onset_hours": np.nan,
            "uo_min_6h_mlkg_hr": np.nan,
            "uo_min_12h_mlkg_hr": np.nan,
            "uo_min_24h_mlkg_hr": np.nan,
            "uo_observed_hour_fraction_7d": 0.0,
            "uo_total_ml_7d": 0.0,
        }

    stay["hour"] = stay["charttime"].dt.floor("h")
    hourly = stay.groupby("hour")["value"].sum().sort_index()
    hours = pd.date_range(index_time.ceil("h"), followup_end.floor("h"), freq="h")
    series = hourly.reindex(hours)
    observed_fraction = float(series.notna().mean()) if len(series) else np.nan
    total_ml = float(series.sum(skipna=True))

    min_rates: dict[int, float] = {}
    first_hits: list[tuple[int, pd.Timestamp]] = []
    for window, threshold in [(6, 0.5), (12, 0.5), (24, 0.3)]:
        min_periods = int(np.ceil(window * COVERAGE_FRACTION))
        roll_sum = series.rolling(window=window, min_periods=min_periods).sum()
        roll_count = series.rolling(window=window, min_periods=min_periods).count()
        rate = roll_sum / (float(weight) * window)
        eligible = roll_count.ge(min_periods)
        rate = rate.where(eligible)
        min_rates[window] = float(rate.min(skipna=True)) if rate.notna().any() else np.nan
        hits = rate[rate.lt(threshold)]
        if not hits.empty:
            first_hits.append((window, hits.index[0]))

    if not first_hits:
        return {
            "uo_evaluable": bool(observed_fraction >= 0.10),
            "uo_ineligibility_reason": "eligible_no_oliguria" if observed_fraction >= 0.10 else "insufficient_urine_output_coverage",
            "uo_aki": False,
            "uo_stage": 0,
            "uo_onset_time": pd.NaT,
            "uo_onset_hours": np.nan,
            "uo_min_6h_mlkg_hr": min_rates[6],
            "uo_min_12h_mlkg_hr": min_rates[12],
            "uo_min_24h_mlkg_hr": min_rates[24],
            "uo_observed_hour_fraction_7d": observed_fraction,
            "uo_total_ml_7d": total_ml,
        }

    stage = max(3 if w == 24 else 2 if w == 12 else 1 for w, _ in first_hits)
    onset_time = min(t for _, t in first_hits)
    return {
        "uo_evaluable": True,
        "uo_ineligibility_reason": "eligible",
        "uo_aki": True,
        "uo_stage": stage,
        "uo_onset_time": onset_time,
        "uo_onset_hours": (onset_time - index_time).total_seconds() / 3600,
        "uo_min_6h_mlkg_hr": min_rates[6],
        "uo_min_12h_mlkg_hr": min_rates[12],
        "uo_min_24h_mlkg_hr": min_rates[24],
        "uo_observed_hour_fraction_7d": observed_fraction,
        "uo_total_ml_7d": total_ml,
    }


def derive_urine_outcomes(cohort: pd.DataFrame, outputevents: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    data = cohort.merge(weights, on="stay_id", how="left")
    rows = []
    for i, row in data.iterrows():
        rows.append(urine_aki_for_stay(row, outputevents))
        if (i + 1) % 1000 == 0:
            print(f"Derived urine-output AKI for {i + 1:,} stays", flush=True)
    return pd.concat([data.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def aggregate_window_events(cohort: pd.DataFrame, outputevents: pd.DataFrame, inputevents: pd.DataFrame) -> pd.DataFrame:
    base = cohort[["subject_id", "hadm_id", "stay_id", "intime", "index_surgery_date"]].copy()
    windows = {
        "preicu_or_at_icu": (pd.Timedelta(hours=-24), pd.Timedelta(hours=0)),
        "0_6h": (pd.Timedelta(hours=0), pd.Timedelta(hours=6)),
        "0_24h": (pd.Timedelta(hours=0), pd.Timedelta(hours=24)),
    }
    feature_rows = []
    for _, row in base.iterrows():
        stay_id = row["stay_id"]
        intime = row["intime"]
        out_stay = outputevents[outputevents["stay_id"].eq(stay_id)]
        in_stay = inputevents[inputevents["stay_id"].eq(stay_id)]
        record = {"subject_id": row["subject_id"], "hadm_id": row["hadm_id"], "stay_id": stay_id}
        for name, (start_delta, end_delta) in windows.items():
            start = intime + start_delta
            end = intime + end_delta
            out_win = out_stay[out_stay["charttime"].gt(start) & out_stay["charttime"].le(end)]
            in_win = in_stay[in_stay["starttime"].lt(end) & in_stay["endtime"].gt(start)]
            for itemid, label in OR_PACU_OUTPUT_ITEMIDS.items():
                vals = out_win[out_win["itemid"].eq(itemid)]["value"]
                record[f"{label}_{name}"] = float(vals.sum()) if not vals.empty else 0.0
                record[f"{label}_{name}_recorded"] = bool(not vals.empty)
            for itemid, label in OR_PACU_INPUT_ITEMIDS.items():
                vals = in_win[in_win["itemid"].eq(itemid)]["amount"]
                record[f"{label}_{name}"] = float(vals.sum()) if not vals.empty else 0.0
                record[f"{label}_{name}_recorded"] = bool(not vals.empty)
            prbc = in_win[in_win["itemid"].isin(PRBC_ITEMIDS)]["amount"]
            record[f"prbc_ml_{name}"] = float(prbc.sum()) if not prbc.empty else 0.0
            record[f"prbc_recorded_{name}"] = bool(not prbc.empty)
            vaso = in_win[in_win["itemid"].isin(VASOACTIVE_ITEMIDS)]
            record[f"vasoactive_any_{name}"] = bool(not vaso.empty)
            record[f"vasoactive_amount_sum_raw_{name}"] = float(vaso["amount"].sum()) if not vaso.empty else 0.0
            for drug in sorted(set(VASOACTIVE_ITEMIDS.values())):
                ids = [k for k, v in VASOACTIVE_ITEMIDS.items() if v == drug]
                record[f"{drug}_any_{name}"] = bool(in_win["itemid"].isin(ids).any())
        feature_rows.append(record)
    return pd.DataFrame(feature_rows)


def combine_scr_uo_outcome(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    scr_aki = bool_series(data["aki_final"])
    uo_aki = bool_series(data["uo_aki"])
    combined_stage = np.maximum(
        pd.to_numeric(data["aki_stage_final"], errors="coerce").fillna(0).astype(int).to_numpy(),
        pd.to_numeric(data["uo_stage"], errors="coerce").fillna(0).astype(int).to_numpy(),
    )
    onset_hours = []
    for _, r in data.iterrows():
        candidates = []
        if bool(r["aki_final"]) and pd.notna(r["aki_onset_hours_final"]):
            candidates.append(float(r["aki_onset_hours_final"]))
        if bool(r["uo_aki"]) and pd.notna(r["uo_onset_hours"]):
            candidates.append(float(r["uo_onset_hours"]))
        onset_hours.append(min(candidates) if candidates else np.nan)
    extra = pd.DataFrame({
        "aki_scr_or_uo_final": scr_aki | uo_aki,
        "aki_uo_only": (~scr_aki) & uo_aki,
        "aki_scr_and_uo": scr_aki & uo_aki,
        "aki_scr_only": scr_aki & (~uo_aki),
        "aki_stage_scr_or_uo_final": combined_stage,
        "aki_onset_hours_scr_or_uo_final": onset_hours,
    }, index=data.index)
    data = pd.concat([data, extra], axis=1)
    return data


def write_audits(data: pd.DataFrame, features: pd.DataFrame, outputevents: pd.DataFrame) -> None:
    overall = []
    n = len(data)
    for label, mask in [
        ("SCr-only KDIGO outcome", bool_series(data["aki_final"])),
        ("Urine-output KDIGO outcome", bool_series(data["uo_aki"])),
        ("SCr-or-urine KDIGO outcome", bool_series(data["aki_scr_or_uo_final"])),
        ("Urine-output-only AKI added to SCr-negative", bool_series(data["aki_uo_only"])),
    ]:
        overall.append({"outcome": label, "n": n, "event_n": int(mask.sum()), "event_rate": float(mask.mean())})
    pd.DataFrame(overall).to_csv(OUTPUT_DIR / "audit_v6_urine_output_aki_summary.csv", index=False)

    coverage = data.groupby("uo_ineligibility_reason", dropna=False).agg(
        n=("stay_id", "size"),
        urine_aki_n=("uo_aki", lambda x: int(bool_series(x).sum())),
        median_observed_hour_fraction=("uo_observed_hour_fraction_7d", "median"),
        median_total_urine_ml_7d=("uo_total_ml_7d", "median"),
    ).reset_index()
    coverage.to_csv(OUTPUT_DIR / "audit_v6_urine_output_coverage.csv", index=False)

    stage = data.groupby("aki_stage_scr_or_uo_final", dropna=False).size().reset_index(name="n")
    stage.to_csv(OUTPUT_DIR / "audit_v6_combined_aki_stage_distribution.csv", index=False)

    item_audit = outputevents.groupby("itemid").agg(n=("value", "size"), total_ml=("value", "sum")).reset_index()
    item_audit.to_csv(OUTPUT_DIR / "audit_v6_outputevents_itemid_counts.csv", index=False)

    feature_cols = [c for c in features.columns if c not in {"subject_id", "hadm_id", "stay_id"}]
    summary = []
    for c in feature_cols:
        s = features[c]
        if pd.api.types.is_bool_dtype(s):
            summary.append({"feature": c, "type": "bool", "n_true_or_sum": int(s.sum()), "mean": float(s.mean()), "median": np.nan, "p75": np.nan})
        else:
            summary.append({"feature": c, "type": "numeric", "n_true_or_sum": float(s.sum()), "mean": float(s.mean()), "median": float(s.median()), "p75": float(s.quantile(0.75))})
    pd.DataFrame(summary).to_csv(OUTPUT_DIR / "audit_v6_intraop_proxy_summary.csv", index=False)

    readme = f"""# v6 urine-output AKI and OR/PACU proxy variables

Primary cohort: `cohort_v3_1_strict_main_evaluable.csv` ({n:,} stays).

Important limitations:

- Urine-output AKI uses ICU outputevents only and is conservative: rolling
  6/12/24 h windows require at least {COVERAGE_FRACTION:.0%} hourly charting
  coverage before oliguria is called.
- Anuria for 12 h is not used because missing output charting cannot safely be
  interpreted as zero urine output.
- OR/PACU variables are charted proxy variables available in MIMIC-IV ICU
  tables. They are not a complete anesthesia record and should not be described
  as comprehensive intraoperative physiology.
- Vasoactive medication variables are early ICU exposure variables when
  occurring after ICU admission; they should not be called intraoperative.

Generated files:

- `cohort_v6_strict_primary_aki_scr_uo.csv`
- `features_v6_or_pacu_early_exposures.csv`
- `audit_v6_urine_output_aki_summary.csv`
- `audit_v6_urine_output_coverage.csv`
- `audit_v6_combined_aki_stage_distribution.csv`
- `audit_v6_intraop_proxy_summary.csv`
- `audit_v6_outputevents_itemid_counts.csv`
"""
    (OUTPUT_DIR / "audit_v6_readme.md").write_text(readme, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cohort = load_cohort()
    print(f"Loaded cohort: {len(cohort):,}", flush=True)
    outputevents = extract_outputevents(cohort)
    print(f"Extracted outputevents rows: {len(outputevents):,}", flush=True)
    inputevents = extract_inputevents(cohort)
    print(f"Extracted inputevents rows: {len(inputevents):,}", flush=True)
    weights = derive_weight(cohort)
    print(f"Derived weights for {weights['weight_kg'].notna().sum():,} stays", flush=True)
    urine_data = derive_urine_outcomes(cohort, outputevents, weights)
    urine_data = combine_scr_uo_outcome(urine_data)
    features = aggregate_window_events(cohort, outputevents, inputevents)
    urine_data.to_csv(OUTPUT_DIR / "cohort_v6_strict_primary_aki_scr_uo.csv", index=False)
    features.to_csv(OUTPUT_DIR / "features_v6_or_pacu_early_exposures.csv", index=False)
    write_audits(urine_data, features, outputevents)
    print(f"Wrote v6 outputs to {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
