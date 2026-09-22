"""
CHIMERA Task 3 - Masking Ablation Study with cohort separation + parallel masking jobs

C-index comparison: baseline vs. all masking conditions
Statistical test: paired bootstrap + paired sign-flip permutation

Added features
--------------
1. Cohort separation from case-folder / case_id prefix:
       3B_xxx -> cohort B
       3U_xxx -> cohort U
2. Run cohort-specific analyses with --cohorts, e.g. --cohorts B U or --cohorts all B U.
3. Parallelize non-baseline masking-condition computations using process workers via --n_jobs.
   This is process-level parallelism, which is appropriate for CPU-bound bootstrap/permutation loops.

Examples
--------
# Run B and U separately using 8 worker processes per model
python ablation_cindex_analysis_cohort_parallel.py \
    --results_dir ./Results_Masking \
    --gt_csv ./task3_test.csv \
    --output_dir ./ablation_results_task3_by_cohort \
    --cohorts B U \
    --n_jobs 8

# Run overall + B + U
python ablation_cindex_analysis_cohort_parallel.py \
    --results_dir ./Results_Masking \
    --gt_csv ./task3_test.csv \
    --output_dir ./ablation_results_task3_by_cohort \
    --cohorts all B U \
    --n_jobs 8

# Run only one model, merge with previous results for each requested cohort
python ablation_cindex_analysis_cohort_parallel.py \
    --results_dir ./Results_Masking \
    --gt_csv ./task3_test.csv \
    --output_dir ./ablation_results_task3_by_cohort \
    --models task3_wl \
    --cohorts B U \
    --merge_existing \
    --n_jobs 8

# Rebuild summary tables/plots only from existing cohort CSVs
python ablation_cindex_analysis_cohort_parallel.py \
    --output_dir ./ablation_results_task3_by_cohort \
    --cohorts B U \
    --replot_only
"""

from __future__ import annotations

# Keep each worker process from internally spawning many BLAS threads.
# This prevents CPU oversubscription when --n_jobs > 1.
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import hashlib
import json
import re
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from lifelines.utils import concordance_index as _lifelines_concordance_index
except Exception:  
    _lifelines_concordance_index = None

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
MODELS = [
    "task3_caggen-biit",
    "task3_hkkh",
    "task3_nmil",
    "task3_smile",
    "task3_tia-pegasus",
    "task3_wl",
]

CONDITION_ORDER = [
    "baseline",
    "drop_age", "drop_sex", "drop_smoking",
    "drop_grade", "drop_stage", "drop_substage",
    "drop_BRS", "drop_EORTC", "drop_LVI",
    "drop_no_instillations", "drop_reTUR",
    "drop_tumor", "drop_variant",
    "random_mask_25", "random_mask_50", "random_mask_75",
]

JSON_FILENAME = "likelihood-of-bladder-cancer-recurrence.json"
COMBINED_RESULTS_FILENAME = "all_models_ablation.csv"
VALID_COHORTS = {"all", "B", "U"}


# ─────────────────────────────────────────────
# Generic helpers
# ─────────────────────────────────────────────
def stable_seed(*parts: Any, base_seed: int = 42) -> int:
    """Generate a deterministic 32-bit seed independent of Python's randomized hash()."""
    text = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) + int(base_seed)) % (2**32 - 1)


def resolve_n_jobs(n_jobs: int | None) -> int:
    """Normalize --n_jobs. Use -1 for all available CPUs. Use 0/None as 1."""
    if n_jobs is None or int(n_jobs) == 0:
        return 1
    n_jobs = int(n_jobs)
    if n_jobs < 0:
        return max(1, os.cpu_count() or 1)
    return max(1, n_jobs)


def normalize_requested_cohorts(raw_cohorts: list[str]) -> list[str]:
    """Normalize --cohorts values while preserving user-specified order."""
    out: list[str] = []
    for cohort in raw_cohorts:
        c = str(cohort).strip()
        c_norm = "all" if c.lower() == "all" else c.upper()
        if c_norm not in VALID_COHORTS:
            raise ValueError(f"Unsupported cohort '{cohort}'. Use one or more of: all, B, U")
        if c_norm not in out:
            out.append(c_norm)
    return out or ["all"]


def cohort_output_subdir(output_dir: Path, cohort: str, n_cohorts: int) -> Path:
    """
    If only one cohort=all is requested, keep backward-compatible output_dir.
    Otherwise, split outputs into cohort_all/, cohort_B/, cohort_U/.
    """
    if n_cohorts == 1 and cohort == "all":
        return output_dir
    return output_dir / f"cohort_{cohort}"


def infer_cohort_from_case_id(case_id: Any) -> str | None:
    """
    Infer cohort from folder/case names such as:
        3B_205, 3B-205, 3B205 -> B
        3U_205, 3U-205, 3U205 -> U
    Returns None if the pattern is not recognized.
    """
    s = str(case_id).strip()
    m = re.match(r"^3([BUbu])(?:[_\-].*|\d.*|$)", s)
    if m:
        return m.group(1).upper()
    return None


def filter_df_by_cohort(df: pd.DataFrame, cohort: str, case_col: str = "case_id") -> pd.DataFrame:
    """Filter a DataFrame by inferred cohort from case_id. cohort='all' returns all rows."""
    if df.empty or cohort == "all":
        return df.copy()
    if case_col not in df.columns:
        return df.copy()
    out = df.copy()
    if "cohort" not in out.columns:
        out["cohort"] = out[case_col].map(infer_cohort_from_case_id)
    return out[out["cohort"] == cohort].copy()


def safe_percentile(values: np.ndarray, q: list[float] | tuple[float, float]) -> tuple[float, float]:
    """Percentiles after removing NaN/Inf values."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan, np.nan
    lo, hi = np.percentile(arr, q)
    return float(lo), float(hi)


def safe_std(values: np.ndarray, ddof: int = 1) -> float:
    """Standard deviation after removing NaN/Inf values."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan
    if len(arr) == 1:
        return 0.0
    return float(np.std(arr, ddof=ddof))


# ─────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────
def load_predictions(
    model_dir: Path,
    condition: str,
    json_filename: str = JSON_FILENAME,
) -> pd.DataFrame:
    """
    Load risk scores from all case subfolders for a given condition.
    Returns DataFrame with columns: case_id, predicted_time, cohort.
    """
    cond_dir = model_dir / condition
    if not cond_dir.exists():
        return pd.DataFrame(columns=["case_id", "predicted_time", "cohort"])

    records: list[dict[str, Any]] = []
    for case_dir in sorted(cond_dir.iterdir()):
        if not case_dir.is_dir():
            continue
        json_path = case_dir / json_filename
        if not json_path.exists():
            continue
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                score = data.get("risk_score")
                if score is None:
                    score = data.get("score")
                if score is None:
                    score = data.get("predicted_time")
                if score is None:
                    score = data.get("prediction")
                if score is None:
                    score = data.get("output")
                if score is None:
                    # Fall back to the first scalar value in the dict.
                    scalar_values = [v for v in data.values() if isinstance(v, (int, float, np.integer, np.floating))]
                    if scalar_values:
                        score = scalar_values[0]
                    else:
                        score = list(data.values())[0]
            else:
                score = float(data)

            records.append({
                "case_id": case_dir.name,
                "predicted_time": float(score),
                "cohort": infer_cohort_from_case_id(case_dir.name),
            })
        except Exception as e:
            print(f"  [WARN] Could not parse {json_path}: {e}")

    return pd.DataFrame(records)


def load_ground_truth(gt_csv: str) -> pd.DataFrame:
    df = pd.read_csv(gt_csv)
    orig_cols = list(df.columns)
    df.columns = [c.strip().lower() for c in df.columns]

    if "case_id" not in df.columns:
        raise ValueError(f"Ground-truth CSV must contain case_id. Found columns: {orig_cols}")

    time_candidates = [c for c in df.columns if "time" in c]
    if not time_candidates:
        raise ValueError(f"Could not infer survival time column. Found columns: {orig_cols}")

    time_col = time_candidates[0]

    # Prefer explicit event columns. Avoid accidentally selecting columns such as
    # time_to_progression as the event column just because they contain "progress".
    event_candidates = [c for c in df.columns if c != time_col and ("event" in c or "progress" in c)]
    if not event_candidates:
        raise ValueError(f"Could not infer event/progression column. Found columns: {orig_cols}")
    event_col = event_candidates[0]
    out = df.rename(columns={time_col: "time", event_col: "event"})[["case_id", "time", "event"]].copy()
    out["case_id"] = out["case_id"].astype(str)
    out["time"] = pd.to_numeric(out["time"], errors="coerce")
    out["event"] = pd.to_numeric(out["event"], errors="coerce").fillna(0).astype(int)
    out["cohort"] = out["case_id"].map(infer_cohort_from_case_id)
    out = out.dropna(subset=["time"])
    return out


def load_existing_results(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    expected = {
        "model", "condition", "n_cases", "c_index", "ci_lower", "ci_upper",
        "delta_vs_baseline", "bootstrap_se", "p_bootstrap", "p_signflip",
        "significance_bootstrap", "significance_signflip",
    }
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"Existing results file is missing columns: {sorted(missing)}")
    return df


def merge_results(existing: pd.DataFrame, new_results: pd.DataFrame) -> pd.DataFrame:
    """
    Replace existing rows for re-run models/cohorts with newly computed rows.
    Keep other rows untouched.
    """
    if existing.empty:
        combined = new_results.copy()
    elif new_results.empty:
        combined = existing.copy()
    else:
        new_keys = new_results[["model", "cohort"]].drop_duplicates() if "cohort" in new_results.columns else new_results[["model"]].drop_duplicates()
        existing_kept = existing.copy()
        for _, key in new_keys.iterrows():
            mask = existing_kept["model"] == key["model"]
            if "cohort" in existing_kept.columns and "cohort" in key.index:
                mask &= existing_kept["cohort"] == key["cohort"]
            existing_kept = existing_kept[~mask].copy()
        combined = pd.concat([existing_kept, new_results], ignore_index=True)

    if combined.empty:
        return combined

    if "cohort" not in combined.columns:
        combined["cohort"] = "all"

    condition_rank = {cond: i for i, cond in enumerate(CONDITION_ORDER)}
    combined["_condition_rank"] = combined["condition"].map(lambda x: condition_rank.get(x, 10_000))
    combined = combined.sort_values(["cohort", "model", "_condition_rank", "condition"]).drop(columns="_condition_rank")
    combined = combined.reset_index(drop=True)
    return combined


# ─────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────
def _fallback_concordance_index(event_times: np.ndarray, predicted_event_times: np.ndarray, event_observed: np.ndarray) -> float:
    """
    Small fallback C-index implementation used only when lifelines is unavailable.
    Matches the usual right-censored pair logic sufficiently for this script, but lifelines is preferred.
    """
    t = np.asarray(event_times, dtype=float)
    p = np.asarray(predicted_event_times, dtype=float)
    e = np.asarray(event_observed, dtype=int)
    n = len(t)
    permissible = 0.0
    concordant = 0.0

    for i in range(n):
        for j in range(i + 1, n):
            if not (np.isfinite(t[i]) and np.isfinite(t[j]) and np.isfinite(p[i]) and np.isfinite(p[j])):
                continue

            if t[i] == t[j]:
                # If tied observed times, count tied predictions as half credit when at least one event occurred.
                if e[i] == 1 or e[j] == 1:
                    permissible += 1.0
                    if p[i] == p[j]:
                        concordant += 0.5
                continue

            if t[i] < t[j] and e[i] == 1:
                permissible += 1.0
                if p[i] < p[j]:
                    concordant += 1.0
                elif p[i] == p[j]:
                    concordant += 0.5
            elif t[j] < t[i] and e[j] == 1:
                permissible += 1.0
                if p[j] < p[i]:
                    concordant += 1.0
                elif p[i] == p[j]:
                    concordant += 0.5

    if permissible == 0:
        return np.nan
    return float(concordant / permissible)


def compute_cindex(predicted_times: np.ndarray, times: np.ndarray, events: np.ndarray) -> float:
    """
    Predicted value is a time (longer = lower risk).
    lifelines.concordance_index(event_times, predicted_event_times, events)
    checks whether higher predicted time corresponds to longer actual time.
    """
    predicted_times = np.asarray(predicted_times, dtype=float)
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    valid = np.isfinite(predicted_times) & np.isfinite(times) & np.isfinite(events)
    if int(valid.sum()) < 2:
        return np.nan

    try:
        if _lifelines_concordance_index is not None:
            return float(_lifelines_concordance_index(times[valid], predicted_times[valid], events[valid]))
        return _fallback_concordance_index(times[valid], predicted_times[valid], events[valid])
    except Exception:
        return np.nan


def bootstrap_cindex(
    risk_scores: np.ndarray,
    times: np.ndarray,
    events: np.ndarray,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> np.ndarray:
    """Return array of bootstrap C-index values."""
    rng = np.random.default_rng(seed)
    n = len(risk_scores)
    boot_cindices = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_cindices[i] = compute_cindex(risk_scores[idx], times[idx], events[idx])
    return boot_cindices


def bootstrap_paired_pvalue(
    risk_base: np.ndarray,
    risk_cond: np.ndarray,
    times: np.ndarray,
    events: np.ndarray,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    Paired bootstrap test: H0 = delta C-index (condition - baseline) == 0.
    Returns (delta, p_value, bootstrap_se).
    """
    rng = np.random.default_rng(seed)
    n = len(risk_base)
    delta_obs = compute_cindex(risk_cond, times, events) - compute_cindex(risk_base, times, events)
    deltas: list[float] = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        c_b = compute_cindex(risk_base[idx], times[idx], events[idx])
        c_c = compute_cindex(risk_cond[idx], times[idx], events[idx])
        delta = c_c - c_b
        if np.isfinite(delta):
            deltas.append(float(delta))

    deltas_arr = np.asarray(deltas, dtype=float)
    if len(deltas_arr) == 0 or not np.isfinite(delta_obs):
        return float(delta_obs), np.nan, np.nan

    se = safe_std(deltas_arr)
    null_deltas = deltas_arr - delta_obs
    p_value = (np.sum(np.abs(null_deltas) >= np.abs(delta_obs)) + 1) / (len(null_deltas) + 1)
    return float(delta_obs), float(p_value), float(se)


def signflip_permutation_pvalue(
    pred_base: np.ndarray,
    pred_cond: np.ndarray,
    times: np.ndarray,
    events: np.ndarray,
    n_permutations: int = 10000,
    seed: int = 99,
) -> float:
    """
    Paired sign-flip permutation test for ΔC-index.
    Returns two-sided p-value.
    """
    rng = np.random.default_rng(seed)
    n = len(pred_base)
    delta_obs = compute_cindex(pred_cond, times, events) - compute_cindex(pred_base, times, events)
    if not np.isfinite(delta_obs):
        return np.nan

    perm_deltas: list[float] = []
    for _ in range(n_permutations):
        flip = rng.integers(0, 2, size=n).astype(bool)
        perm_base = np.where(flip, pred_cond, pred_base)
        perm_cond = np.where(flip, pred_base, pred_cond)
        delta = compute_cindex(perm_cond, times, events) - compute_cindex(perm_base, times, events)
        if np.isfinite(delta):
            perm_deltas.append(float(delta))

    if len(perm_deltas) == 0:
        return np.nan
    perm_arr = np.asarray(perm_deltas, dtype=float)
    p_value = float(np.mean(np.abs(perm_arr) >= np.abs(delta_obs)))
    return max(p_value, 1 / n_permutations)


def add_significance_stars(p: float) -> str:
    if pd.isna(p):
        return "na"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.1:
        return "."
    return "ns"


# ─────────────────────────────────────────────
# Parallel condition worker
# ─────────────────────────────────────────────
def _analyse_condition_worker(payload: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    """
    Worker for one non-baseline masking condition. Returns (row, log_message).
    Top-level function so it can be pickled by ProcessPoolExecutor.
    """
    model_name = payload["model_name"]
    model_dir = Path(payload["model_dir"])
    cond = payload["condition"]
    gt = payload["gt"]
    base_merged = payload["base_merged"]
    n_bootstrap = int(payload["n_bootstrap"])
    cohort = payload["cohort"]
    json_filename = payload["json_filename"]

    preds = load_predictions(model_dir, cond, json_filename=json_filename)
    preds = filter_df_by_cohort(preds, cohort)
    if preds.empty:
        return None, f"  [SKIP] {cond}: no predictions found"

    merged = preds.merge(gt[["case_id", "time", "event"]], on="case_id")
    if len(merged) < 5:
        return None, f"  [SKIP] {cond}: too few cases ({len(merged)})"

    common = base_merged[["case_id", "time", "event", "predicted_time"]].merge(
        merged[["case_id", "predicted_time"]],
        on="case_id",
        suffixes=("_base", "_cond"),
    )
    if len(common) < 5:
        return None, f"  [SKIP] {cond}: too few paired cases ({len(common)})"

    rb = common["predicted_time_base"].to_numpy(dtype=float)
    rc = common["predicted_time_cond"].to_numpy(dtype=float)
    t = common["time"].to_numpy(dtype=float)
    ev = common["event"].to_numpy(dtype=int)

    seed_base = stable_seed(model_name, cohort, cond, "base", base_seed=42)
    seed_pair = stable_seed(model_name, cohort, cond, "paired", base_seed=42)
    seed_perm = stable_seed(model_name, cohort, cond, "perm", base_seed=99)

    delta, p_boot, se = bootstrap_paired_pvalue(rb, rc, t, ev, n_bootstrap=n_bootstrap, seed=seed_pair)
    p_perm = signflip_permutation_pvalue(rb, rc, t, ev, n_permutations=n_bootstrap, seed=seed_perm)

    boot_cond = bootstrap_cindex(rc, t, ev, n_bootstrap=n_bootstrap, seed=seed_base)
    ci_lo_c, ci_hi_c = safe_percentile(boot_cond, [2.5, 97.5])
    cindex_cond = compute_cindex(rc, t, ev)

    sig_boot = add_significance_stars(p_boot)
    sig_perm = add_significance_stars(p_perm)

    row = {
        "cohort": cohort,
        "model": model_name,
        "condition": cond,
        "n_cases": len(common),
        "n_events": int(ev.sum()),
        "n_censored": int((ev == 0).sum()),
        "c_index": cindex_cond,
        "ci_lower": ci_lo_c,
        "ci_upper": ci_hi_c,
        "delta_vs_baseline": delta,
        "bootstrap_se": se,
        "p_bootstrap": p_boot,
        "p_signflip": p_perm,
        "significance_bootstrap": sig_boot,
        "significance_signflip": sig_perm,
    }

    log = (
        f"  {cond:<25} C={cindex_cond:.4f}  Δ={delta:+.4f}  "
        f"p_boot={p_boot:.4f}{sig_boot}  p_perm={p_perm:.4f}{sig_perm}"
    )
    return row, log


def run_condition_jobs(
    model_name: str,
    model_dir: Path,
    conditions: list[str],
    gt: pd.DataFrame,
    base_merged: pd.DataFrame,
    n_bootstrap: int,
    cohort: str,
    json_filename: str,
    n_jobs: int = 1,
) -> list[dict[str, Any]]:
    """Run all non-baseline condition analyses, sequentially or in parallel."""
    non_baseline = [c for c in conditions if c != "baseline"]
    if not non_baseline:
        return []

    payloads = [
        {
            "model_name": model_name,
            "model_dir": str(model_dir),
            "condition": cond,
            "gt": gt,
            "base_merged": base_merged,
            "n_bootstrap": n_bootstrap,
            "cohort": cohort,
            "json_filename": json_filename,
        }
        for cond in non_baseline
    ]

    # Sequential mode is easier to debug and avoids process overhead for small jobs.
    n_jobs = min(resolve_n_jobs(n_jobs), len(payloads))
    if n_jobs <= 1:
        rows: list[dict[str, Any]] = []
        for payload in payloads:
            row, log = _analyse_condition_worker(payload)
            print(log)
            if row is not None:
                rows.append(row)
        return rows

    print(f"  Parallel masking jobs: {n_jobs} worker processes for {len(payloads)} conditions")
    rows_by_condition: dict[str, dict[str, Any]] = {}
    logs_by_condition: dict[str, str] = {}

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = {executor.submit(_analyse_condition_worker, payload): payload["condition"] for payload in payloads}
        for future in as_completed(futures):
            cond = futures[future]
            try:
                row, log = future.result()
            except Exception as e:
                row, log = None, f"  [ERROR] {cond}: {e}"
            logs_by_condition[cond] = log
            if row is not None:
                rows_by_condition[cond] = row

    # Print and return in requested condition order for reproducible CSVs/logs.
    rows: list[dict[str, Any]] = []
    for cond in non_baseline:
        if cond in logs_by_condition:
            print(logs_by_condition[cond])
        if cond in rows_by_condition:
            rows.append(rows_by_condition[cond])
    return rows


# ─────────────────────────────────────────────
# Main analysis
# ─────────────────────────────────────────────
def analyse_model(
    model_name: str,
    results_root: Path,
    gt: pd.DataFrame,
    n_bootstrap: int,
    output_dir: Path,
    cohort: str = "all",
    json_filename: str = JSON_FILENAME,
    n_jobs: int = 1,
) -> pd.DataFrame:
    model_dir = results_root / model_name
    print(f"\n{'=' * 60}")
    print(f"Model: {model_name} | cohort: {cohort}")
    print(f"{'=' * 60}")

    if not model_dir.exists():
        print(f"  [SKIP] {model_name} not found")
        return pd.DataFrame()

    conditions = sorted([d.name for d in model_dir.iterdir() if d.is_dir()])
    ordered = [c for c in CONDITION_ORDER if c in conditions]
    extras = [c for c in conditions if c not in ordered]
    conditions = ordered + extras

    if "baseline" not in conditions:
        print(f"  [SKIP] No baseline found for {model_name}")
        return pd.DataFrame()

    gt_cohort = filter_df_by_cohort(gt, cohort)
    if len(gt_cohort) < 5:
        print(f"  [SKIP] cohort {cohort}: too few ground-truth cases ({len(gt_cohort)})")
        return pd.DataFrame()

    base_preds = load_predictions(model_dir, "baseline", json_filename=json_filename)
    base_preds = filter_df_by_cohort(base_preds, cohort)
    base_merged = base_preds.merge(gt_cohort[["case_id", "time", "event"]], on="case_id")
    if len(base_merged) < 5:
        print(f"  [SKIP] baseline has too few cases ({len(base_merged)}) for cohort {cohort}")
        return pd.DataFrame()

    risk_base = base_merged["predicted_time"].to_numpy(dtype=float)
    times_base = base_merged["time"].to_numpy(dtype=float)
    events_base = base_merged["event"].to_numpy(dtype=int)
    cindex_base = compute_cindex(risk_base, times_base, events_base)

    boot_base = bootstrap_cindex(
        risk_base,
        times_base,
        events_base,
        n_bootstrap=n_bootstrap,
        seed=stable_seed(model_name, cohort, "baseline", base_seed=42),
    )
    ci_lo, ci_hi = safe_percentile(boot_base, [2.5, 97.5])

    print(
        f"  Baseline C-index: {cindex_base:.4f} [{ci_lo:.4f}, {ci_hi:.4f}] "
        f"(n={len(base_merged)}, events={int(events_base.sum())}, censored={int((events_base == 0).sum())})"
    )

    rows: list[dict[str, Any]] = [{
        "cohort": cohort,
        "model": model_name,
        "condition": "baseline",
        "n_cases": len(base_merged),
        "n_events": int(events_base.sum()),
        "n_censored": int((events_base == 0).sum()),
        "c_index": cindex_base,
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
        "delta_vs_baseline": 0.0,
        "bootstrap_se": safe_std(boot_base),
        "p_bootstrap": np.nan,
        "p_signflip": np.nan,
        "significance_bootstrap": "—",
        "significance_signflip": "—",
    }]

    rows.extend(
        run_condition_jobs(
            model_name=model_name,
            model_dir=model_dir,
            conditions=conditions,
            gt=gt_cohort,
            base_merged=base_merged,
            n_bootstrap=n_bootstrap,
            cohort=cohort,
            json_filename=json_filename,
            n_jobs=n_jobs,
        )
    )

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / f"{model_name}_ablation.csv", index=False)
    return df


def save_summary_tables(combined: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Save long and wide summaries from a combined result table."""
    if combined.empty:
        return combined

    if "cohort" not in combined.columns:
        combined = combined.copy()
        combined["cohort"] = "all"

    condition_rank = {cond: i for i, cond in enumerate(CONDITION_ORDER)}
    combined["_condition_rank"] = combined["condition"].map(lambda x: condition_rank.get(x, 10_000))
    combined = combined.sort_values(["cohort", "model", "_condition_rank", "condition"]).drop(columns="_condition_rank")
    combined.to_csv(output_dir / COMBINED_RESULTS_FILENAME, index=False)

    pivot = combined.pivot_table(index="condition", columns="model", values="c_index", aggfunc="first")
    pivot["mean_cindex"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("mean_cindex", ascending=False)

    delta_pivot = combined.pivot_table(index="condition", columns="model", values="delta_vs_baseline", aggfunc="first")
    delta_pivot["mean_delta"] = delta_pivot.mean(axis=1)

    sig_boot_pivot = combined.pivot_table(
        index="condition", columns="model", values="significance_bootstrap", aggfunc="first"
    )
    sig_perm_pivot = combined.pivot_table(
        index="condition", columns="model", values="significance_signflip", aggfunc="first"
    )

    pivot.to_csv(output_dir / "cindex_summary_wide.csv")
    delta_pivot.to_csv(output_dir / "delta_summary_wide.csv")
    sig_boot_pivot.to_csv(output_dir / "significance_bootstrap_wide.csv")
    sig_perm_pivot.to_csv(output_dir / "significance_signflip_wide.csv")

    cohort_label = str(combined["cohort"].iloc[0]) if "cohort" in combined.columns and not combined.empty else "all"
    print("\n" + "=" * 70)
    print(f"SUMMARY: Mean C-index per condition (across models) | cohort: {cohort_label}")
    print("=" * 70)
    summary = delta_pivot[["mean_delta"]].join(pivot[["mean_cindex"]])
    summary.index.name = "condition"
    print(summary.to_string(float_format="{:.4f}".format))
    return combined


# ─────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────
def _get_model_order_by_baseline(combined: pd.DataFrame) -> list[str]:
    baseline_df = (
        combined[combined["condition"] == "baseline"][["model", "c_index"]]
        .dropna()
        .drop_duplicates(subset=["model"])
        .sort_values("c_index", ascending=False)
    )
    return baseline_df["model"].tolist()


def plot_results(combined: pd.DataFrame, output_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[INFO] matplotlib not available, skipping plots")
        return

    if combined.empty:
        print("[INFO] No combined results available, skipping plots")
        return

    conditions = [c for c in CONDITION_ORDER if c in combined["condition"].unique()]
    conditions += [c for c in combined["condition"].unique() if c not in conditions]
    non_baseline = [c for c in conditions if c != "baseline"]

    model_order = _get_model_order_by_baseline(combined)

    # Heatmap: color = delta, text = absolute c-index, order = baseline c-index high -> low
    delta_data = combined[combined["condition"] != "baseline"].pivot_table(
        index="condition", columns="model", values="delta_vs_baseline", aggfunc="first"
    )
    delta_data = delta_data.reindex([c for c in non_baseline if c in delta_data.index])

    value_data = combined[combined["condition"] != "baseline"].pivot_table(
        index="condition", columns="model", values="c_index", aggfunc="first"
    ).reindex(index=delta_data.index, columns=delta_data.columns)

    sig_data = combined[combined["condition"] != "baseline"].pivot_table(
        index="condition", columns="model", values="significance_signflip", aggfunc="first"
    ).reindex(index=delta_data.index, columns=delta_data.columns)

    cohort_label = str(combined["cohort"].iloc[0]) if "cohort" in combined.columns and not combined.empty else "all"

    if len(delta_data) > 0 and len(delta_data.columns) > 0:
        extra_models = [m for m in delta_data.columns if m not in model_order]
        model_order = [m for m in model_order if m in delta_data.columns] + extra_models

        delta_data = delta_data.reindex(columns=model_order)
        value_data = value_data.reindex(columns=model_order)
        sig_data = sig_data.reindex(columns=model_order)

        fig, ax = plt.subplots(figsize=(len(model_order) * 1.8 + 2, len(non_baseline) * 0.60 + 2))
        vmax = max(0.05, float(np.nanmax(np.abs(delta_data.values))))
        im = ax.imshow(delta_data.values, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")

        ax.set_xticks(range(len(delta_data.columns)))
        ax.set_xticklabels(
            [m.replace("task3_", "") for m in delta_data.columns],
            rotation=30,
            ha="right",
            fontsize=9,
        )
        ax.set_yticks(range(len(delta_data.index)))
        ax.set_yticklabels(delta_data.index, fontsize=8)

        for i in range(len(delta_data.index)):
            for j in range(len(delta_data.columns)):
                delta_val = delta_data.values[i, j]
                cindex_val = value_data.values[i, j]
                sig = sig_data.values[i, j] if pd.notna(sig_data.values[i, j]) else ""
                if not np.isnan(delta_val) and not np.isnan(cindex_val):
                    ax.text(
                        j,
                        i,
                        f"{cindex_val:.3f}\n{sig}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="black",
                    )

        plt.colorbar(im, ax=ax, label="ΔC-index vs. baseline")
        ax.set_title(
            f"Ablation Study: C-index | cohort: {cohort_label}\n"
            "(cell text = absolute C-index, color = Δ vs baseline; columns ordered by baseline best → worst)",
            fontsize=11,
            pad=12,
        )
        plt.tight_layout()
        plt.savefig(output_dir / "ablation_heatmap.pdf", bbox_inches="tight", dpi=150)
        plt.savefig(output_dir / "ablation_heatmap.png", bbox_inches="tight", dpi=150)
        plt.close()
        print(f"\nSaved heatmap → {output_dir / 'ablation_heatmap.png'}")

    # Per-model CI plots
    for model in list(combined["model"].dropna().unique()):
        mdf = combined[combined["model"] == model].copy()
        mdf = (
            mdf[mdf["condition"].isin(conditions)]
            .set_index("condition")
            .reindex(conditions)
            .dropna(subset=["c_index"])
        )
        if mdf.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, max(4, len(mdf) * 0.45 + 1.5)))
        y = np.arange(len(mdf))
        colors = [
            "steelblue" if c == "baseline" else ("tomato" if mdf.loc[c, "delta_vs_baseline"] < 0 else "seagreen")
            for c in mdf.index
        ]

        ax.barh(
            y,
            mdf["c_index"],
            xerr=[mdf["c_index"] - mdf["ci_lower"], mdf["ci_upper"] - mdf["c_index"]],
            color=colors,
            ecolor="gray",
            capsize=3,
            height=0.6,
            alpha=0.85,
        )

        if "baseline" in mdf.index:
            base_val = mdf.loc["baseline", "c_index"]
            ax.axvline(
                base_val,
                color="navy",
                linestyle="--",
                linewidth=1.2,
                alpha=0.7,
                label=f"Baseline ({base_val:.3f})",
            )

        ax.set_yticks(y)
        ax.set_yticklabels(mdf.index, fontsize=8)
        ax.set_xlabel("C-index (95% bootstrap CI)")
        ax.set_title(f"{model.replace('task3_', '').upper()} — Ablation C-index | cohort: {cohort_label}", fontsize=11)
        ax.legend(fontsize=8)
        xmin = max(0.3, float(np.nanmin(mdf["ci_lower"])) - 0.02)
        xmax = min(1.0, float(np.nanmax(mdf["ci_upper"])) + 0.05)
        ax.set_xlim(xmin, xmax)
        plt.tight_layout()
        plt.savefig(output_dir / f"{model}_cindex_ci.png", bbox_inches="tight", dpi=150)
        plt.close()

    print(f"Saved per-model CI plots → {output_dir}/")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="CHIMERA Task3 masking ablation — C-index analysis with cohort separation and parallel masking jobs")
    parser.add_argument("--results_dir", default=None, help="Path to Results_Masking/")
    parser.add_argument("--gt_csv", default=None, help="Path to task3_test.csv")
    parser.add_argument("--n_bootstrap", type=int, default=10000, help="Bootstrap/permutation iterations (default: 10000)")
    parser.add_argument("--output_dir", default="./ablation_results", help="Output directory")
    parser.add_argument("--models", nargs="+", default=MODELS, help="Subset of models to analyse")
    parser.add_argument("--json_filename", default=JSON_FILENAME, help=f"JSON filename inside each case folder (default: {JSON_FILENAME})")
    parser.add_argument("--n_jobs", type=int, default=1, help="Parallel worker processes for masking conditions per model. Use -1 for all CPUs. Default: 1")
    parser.add_argument(
        "--cohorts",
        nargs="+",
        default=["all"],
        help="Cohorts to analyse: all, B, U. Examples: --cohorts B U or --cohorts all B U. Default: all",
    )
    parser.add_argument(
        "--existing_results_csv",
        default=None,
        help="Optional existing all_models_ablation.csv to merge with or replot from. "
             "Default: <cohort_output_dir>/all_models_ablation.csv",
    )
    parser.add_argument(
        "--merge_existing",
        action="store_true",
        help="Merge new model results with existing combined CSV instead of replacing everything.",
    )
    parser.add_argument(
        "--replot_only",
        action="store_true",
        help="Only regenerate summary tables and plots from existing combined CSV(s).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    requested_cohorts = normalize_requested_cohorts(args.cohorts)
    n_jobs = resolve_n_jobs(args.n_jobs)
    print(f"Requested cohort analyses: {', '.join(requested_cohorts)}")
    print(f"Parallel workers per model: {n_jobs}")

    gt: pd.DataFrame | None = None
    results_root: Path | None = None

    if not args.replot_only:
        if not args.results_dir or not args.gt_csv:
            raise ValueError("--results_dir and --gt_csv are required unless --replot_only is used")
        results_root = Path(args.results_dir)
        print(f"Loading ground truth from: {args.gt_csv}")
        gt = load_ground_truth(args.gt_csv)
        print(
            f"  {len(gt)} cases loaded | events: {int(gt['event'].sum())} | "
            f"censored: {int((gt['event'] == 0).sum())} | "
            f"B: {int((gt['cohort'] == 'B').sum())} | U: {int((gt['cohort'] == 'U').sum())} | unknown: {int(gt['cohort'].isna().sum())}"
        )

    for cohort in requested_cohorts:
        cohort_dir = cohort_output_subdir(output_dir, cohort, n_cohorts=len(requested_cohorts))
        cohort_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'#' * 72}")
        print(f"Cohort analysis: {cohort} → {cohort_dir}")
        print(f"{'#' * 72}")

        existing_csv = Path(args.existing_results_csv) if args.existing_results_csv else cohort_dir / COMBINED_RESULTS_FILENAME

        if args.replot_only:
            combined = load_existing_results(existing_csv)
            if combined.empty:
                raise FileNotFoundError(
                    f"--replot_only requested, but no existing results found at: {existing_csv}"
                )
            if "cohort" not in combined.columns:
                combined["cohort"] = cohort
            combined = save_summary_tables(combined, cohort_dir)
            plot_results(combined, cohort_dir)
            print(f"\n✓ Replotted cohort {cohort} from existing results: {existing_csv}")
            continue

        assert gt is not None
        assert results_root is not None

        new_results = []
        for model in args.models:
            if not (results_root / model).exists():
                print(f"[SKIP] {model} not found in {results_root}")
                continue
            df = analyse_model(
                model_name=model,
                results_root=results_root,
                gt=gt,
                n_bootstrap=args.n_bootstrap,
                output_dir=cohort_dir,
                cohort=cohort,
                json_filename=args.json_filename,
                n_jobs=n_jobs,
            )
            if not df.empty:
                new_results.append(df)

        if not new_results and not args.merge_existing:
            print(f"[WARN] No model results were analysed for cohort {cohort}. Check results_dir / model names / JSON parsing.")
            continue

        new_combined = pd.concat(new_results, ignore_index=True) if new_results else pd.DataFrame()

        if args.merge_existing:
            existing = load_existing_results(existing_csv)
            if not existing.empty and "cohort" not in existing.columns:
                existing["cohort"] = cohort
            combined = merge_results(existing, new_combined)
        else:
            combined = new_combined

        if combined.empty:
            print(f"[WARN] Combined result table is empty for cohort {cohort}. Nothing to save or plot.")
            continue

        combined = save_summary_tables(combined, cohort_dir)
        plot_results(combined, cohort_dir)

        print(f"\n✓ Results saved to: {cohort_dir}/")
        print("  • all_models_ablation.csv   — full per-model-condition results")
        print("  • cindex_summary_wide.csv   — C-index summary table")
        print("  • delta_summary_wide.csv    — ΔC-index table vs baseline")
        print("  • ablation_heatmap.png/pdf  — heatmap (color=Δ, text=absolute C-index)")
        print("  • <model>_cindex_ci.png     — per-model CI plots")

    print("\nExamples:")
    print("  • Run B and U cohorts with 8 workers:")
    print("      python ablation_cindex_analysis_cohort_parallel.py --results_dir ... --gt_csv ... --output_dir ... --cohorts B U --n_jobs 8")
    print("  • Run overall + B + U:")
    print("      python ablation_cindex_analysis_cohort_parallel.py --results_dir ... --gt_csv ... --output_dir ... --cohorts all B U --n_jobs 8")
    print("  • Replot existing cohort outputs only:")
    print("      python ablation_cindex_analysis_cohort_parallel.py --output_dir ... --cohorts B U --replot_only")


if __name__ == "__main__":
    main()
