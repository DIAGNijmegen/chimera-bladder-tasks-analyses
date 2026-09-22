"""
CHIMERA Task 2 - Masking Ablation Study
AUROC + weighted F1 comparison: baseline vs. all masking conditions
Statistical test: paired bootstrap test + paired sign-flip permutation test

Task 2 assumption
-----------------
Binary classification where target_class = 1 corresponds to BRS3 and
predictions are BRS probabilities. The primary score used for evaluation is
P(BRS3), extracted from each case JSON.

This script mirrors the I/O pattern of the Task 3 c-index ablation script:
    results_dir/
      model_name/
        baseline/
          case_id/
            <prediction.json>
        drop_xxx/
          case_id/
            <prediction.json>

Usage:
    python task2_ablation_auc_weightedf1_analysis.py \
        --results_dir ./Results_Masking \
        --gt_csv ./task2_test.csv \
        --n_bootstrap 10000 \
        --decision_threshold 0.5 \
        --output_dir ./ablation_results_task2

Run overall + cohort-specific analyses where cohort is inferred from case folder
names such as 2B_205 or 2U_205:
    python task2_ablation_auc_weightedf1_analysis.py \
        --results_dir ./Results_Masking \
        --gt_csv ./task2_test.csv \
        --n_bootstrap 10000 \
        --decision_threshold 0.5 \
        --output_dir ./ablation_results_task2_by_cohort \
        --cohorts all B U
        --n_jobs 8

Or only add model and replot the heatmap:
python ./task2_ablation_auc_weightedf1_analysis.py  \
    --results_dir ./Results_Masking         \
    --gt_csv ./task2_test.csv         \
    --n_bootstrap 10000         --decision_threshold 0.5         \
    --output_dir ./ablation_results_task --models task2_gris \
    --reuse_existing_csvs
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# Avoid CPU oversubscription when using multiple worker processes.
# Users can override these before running the script if desired.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from sklearn.metrics import f1_score, roc_auc_score
except Exception as e:  
    raise ImportError("scikit-learn is required for AUROC and weighted F1 computation") from e


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
MODELS = [
    "task2_aillis",
    "task2_hkkh",
    "task2_biototem",
    "task2_wl",
    "task2_gris",
    "task2_caggen",
    "task2_mitel-uniud",
    "task2_buaa_remex"
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

DEFAULT_JSON_FILENAMES = [
    "molecular-subtype-of-bladder-cancer.json",
    "bladder-cancer-molecular-subtype.json",
    "bladder-cancer-subtype.json",
    "subtype-of-bladder-cancer.json",
    "prediction.json",
    "predictions.json",
    "result.json",
    "output.json",
]

METRICS = ("auroc", "weighted_f1")


# ─────────────────────────────────────────────
# Generic parsing helpers
# ─────────────────────────────────────────────
def _normalize_key(s: str) -> str:
    return "".join(ch.lower() for ch in str(s) if ch.isalnum())


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool)


def _find_json_file(case_dir: Path, explicit_name: str | None = None) -> Path | None:
    if explicit_name:
        p = case_dir / explicit_name
        if p.exists():
            return p

    for name in DEFAULT_JSON_FILENAMES:
        p = case_dir / name
        if p.exists():
            return p

    json_files = sorted(case_dir.glob("*.json"))
    if len(json_files) == 1:
        return json_files[0]

    priority = []
    for p in json_files:
        stem = _normalize_key(p.stem)
        if any(tok in stem for tok in ["prediction", "predict", "result", "output", "subtype", "brs"]):
            priority.append(p)
    if len(priority) == 1:
        return priority[0]
    if priority:
        return priority[0]
    if json_files:
        return json_files[0]
    return None


def infer_cohort_from_case_id(case_id: Any) -> str | None:
    """
    Infer Task 2 cohort from the case folder / case_id naming convention.

    Examples:
      2B_205 -> B
      2U_205 -> U
    """
    s = str(case_id).strip().upper()
    if s.startswith("2B_"):
        return "B"
    if s.startswith("2U_"):
        return "U"
    return None


def _extract_positive_probability(
    obj: Any,
    positive_label: str = "BRS3",
    positive_index: int = 2,
) -> float:
    """
    Robustly extract the positive-class score from a JSON object.

    Supported patterns include:
      1) scalar score
      2) {"BRS1": ..., "BRS2": ..., "BRS3": ...}
      3) {"probabilities": {"BRS1": ..., "BRS2": ..., "BRS3": ...}}
      4) {"probs": [...]} with BRS1/BRS2/BRS3 order (positive_index=2)
      5) deeply nested dict/list combinations
    """
    positive_norm = _normalize_key(positive_label)
    candidate_keys = {
        positive_norm,
        "positive",
        "positiveclass",
        "positiveprob",
        "positiveprobability",
        "p1",
        "prob1",
        "score1",
        "brs3",
        "pbrs3",
        "probbrs3",
        "probabilitybrs3",
    }

    def recurse(x: Any) -> float | None:
        if _is_number(x):
            return float(x)

        if isinstance(x, (list, tuple, np.ndarray)):
            seq = list(x)
            if len(seq) > positive_index and _is_number(seq[positive_index]):
                return float(seq[positive_index])
            for item in seq:
                out = recurse(item)
                if out is not None:
                    return out
            return None

        if isinstance(x, dict):
            norm_map = {_normalize_key(k): v for k, v in x.items()}

            for key in candidate_keys:
                if key in norm_map and _is_number(norm_map[key]):
                    return float(norm_map[key])

            class_triplets = [
                ("brs1", "brs2", "brs3"),
                ("class0", "class1", "class2"),
                ("p0", "p1", "p2"),
                ("prob0", "prob1", "prob2"),
                ("score0", "score1", "score2"),
            ]
            for triplet in class_triplets:
                if all(k in norm_map and _is_number(norm_map[k]) for k in triplet):
                    if positive_norm == "brs3":
                        return float(norm_map[triplet[2]])

            nested_keys = [
                "probabilities", "probability", "probs", "scores", "predictions",
                "prediction", "output", "result", "results", "logits", "softmax",
            ]
            for key in nested_keys:
                if key in norm_map:
                    out = recurse(norm_map[key])
                    if out is not None:
                        return out

            for _, value in x.items():
                out = recurse(value)
                if out is not None:
                    return out

        return None

    result = recurse(obj)
    if result is None:
        raise ValueError("Could not extract positive-class probability from JSON")
    return float(result)


# ─────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────
def load_predictions(
    model_dir: Path,
    condition: str,
    json_filename: str | None = None,
    positive_label: str = "BRS3",
    positive_index: int = 2,
) -> pd.DataFrame:
    """
    Load positive-class probabilities from all case subfolders for a condition.
    Returns DataFrame with columns: case_id, cohort, pos_prob, json_file
    """
    cond_dir = model_dir / condition
    if not cond_dir.exists():
        return pd.DataFrame(columns=["case_id", "cohort", "pos_prob", "json_file"])

    records = []
    for case_dir in sorted(cond_dir.iterdir()):
        if not case_dir.is_dir():
            continue
        json_path = _find_json_file(case_dir, explicit_name=json_filename)
        if json_path is None or not json_path.exists():
            continue
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            score = _extract_positive_probability(
                data,
                positive_label=positive_label,
                positive_index=positive_index,
            )
            records.append({
                "case_id": case_dir.name,
                "cohort": infer_cohort_from_case_id(case_dir.name),
                "pos_prob": float(score),
                "json_file": json_path.name,
            })
        except Exception as e:
            print(f"  [WARN] Could not parse {json_path}: {e}")

    return pd.DataFrame(records)


def load_ground_truth(gt_csv: str, positive_label: str = "BRS3") -> pd.DataFrame:
    df = pd.read_csv(gt_csv)
    orig_cols = list(df.columns)
    df.columns = [c.strip().lower() for c in df.columns]

    if "case_id" not in df.columns:
        raise ValueError(f"Ground-truth CSV must contain case_id. Found columns: {orig_cols}")

    if "target_class" in df.columns:
        y = df["target_class"].astype(int)
    elif "label" in df.columns:
        y = df["label"].astype(int)
    elif "target" in df.columns:
        y = (df["target"].astype(str).str.upper() == positive_label.upper()).astype(int)
    elif "brs" in df.columns:
        y = (df["brs"].astype(str).str.upper() == positive_label.upper()).astype(int)
    else:
        raise ValueError(
            "Could not infer binary target from GT CSV. Expected one of: target_class, label, target, brs"
        )

    out = pd.DataFrame({
        "case_id": df["case_id"].astype(str),
        "target_class": y.astype(int),
    })

    n_pos = int(out["target_class"].sum())
    n_neg = int((out["target_class"] == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            f"Ground truth must contain both classes. Got positives={n_pos}, negatives={n_neg}."
        )
    return out


# ─────────────────────────────────────────────
# Metrics / statistics
# ─────────────────────────────────────────────
def prob_to_pred(y_score: np.ndarray, decision_threshold: float = 0.5) -> np.ndarray:
    y_score = np.asarray(y_score, dtype=float)
    return (y_score >= decision_threshold).astype(int)


def compute_metric(
    metric_name: str,
    y_true: np.ndarray,
    y_score: np.ndarray,
    decision_threshold: float = 0.5,
) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    if metric_name == "auroc":
        if len(np.unique(y_true)) < 2:
            return np.nan
        try:
            return float(roc_auc_score(y_true, y_score))
        except Exception:
            return np.nan

    if metric_name == "weighted_f1":
        y_pred = prob_to_pred(y_score, decision_threshold=decision_threshold)
        try:
            return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        except Exception:
            return np.nan

    raise ValueError(f"Unsupported metric: {metric_name}")


def bootstrap_metric(
    metric_name: str,
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bootstrap: int = 10000,
    seed: int = 42,
    decision_threshold: float = 0.5,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        val = compute_metric(metric_name, y_true[idx], y_score[idx], decision_threshold=decision_threshold)
        if np.isfinite(val):
            vals.append(val)
    return np.asarray(vals, dtype=float)


def bootstrap_paired_pvalue(
    metric_name: str,
    y_true: np.ndarray,
    score_base: np.ndarray,
    score_cond: np.ndarray,
    n_bootstrap: int = 10000,
    seed: int = 42,
    decision_threshold: float = 0.5,
) -> tuple[float, float, float]:
    """
    Paired bootstrap test for Δmetric = metric(condition) - metric(baseline).
    Returns (delta_obs, p_value, bootstrap_se).
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    delta_obs = (
        compute_metric(metric_name, y_true, score_cond, decision_threshold=decision_threshold)
        - compute_metric(metric_name, y_true, score_base, decision_threshold=decision_threshold)
    )
    deltas = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        m_base = compute_metric(metric_name, y_true[idx], score_base[idx], decision_threshold=decision_threshold)
        m_cond = compute_metric(metric_name, y_true[idx], score_cond[idx], decision_threshold=decision_threshold)
        if np.isfinite(m_base) and np.isfinite(m_cond):
            deltas.append(m_cond - m_base)
    deltas = np.asarray(deltas, dtype=float)
    if len(deltas) == 0:
        return float(delta_obs), np.nan, np.nan
    se = float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0
    null_deltas = deltas - delta_obs
    p_value = (np.sum(np.abs(null_deltas) >= np.abs(delta_obs)) + 1) / (len(null_deltas) + 1)
    return float(delta_obs), float(p_value), float(se)


def signflip_permutation_pvalue(
    metric_name: str,
    y_true: np.ndarray,
    score_base: np.ndarray,
    score_cond: np.ndarray,
    n_permutations: int = 10000,
    seed: int = 99,
    decision_threshold: float = 0.5,
) -> float:
    """
    Paired sign-flip permutation test for Δmetric.

    For each patient, under H0 we can swap baseline/condition scores at random,
    then recompute Δmetric.
    """
    rng = np.random.default_rng(seed)
    delta_obs = (
        compute_metric(metric_name, y_true, score_cond, decision_threshold=decision_threshold)
        - compute_metric(metric_name, y_true, score_base, decision_threshold=decision_threshold)
    )
    n = len(y_true)
    perm_deltas = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        flip = rng.integers(0, 2, size=n).astype(bool)
        perm_base = np.where(flip, score_cond, score_base)
        perm_cond = np.where(flip, score_base, score_cond)
        perm_deltas[i] = (
            compute_metric(metric_name, y_true, perm_cond, decision_threshold=decision_threshold)
            - compute_metric(metric_name, y_true, perm_base, decision_threshold=decision_threshold)
        )
    p_value = float(np.mean(np.abs(perm_deltas) >= np.abs(delta_obs)))
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


def resolve_n_jobs(n_jobs: int | None) -> int:
    """Normalize --n_jobs. Use -1 for all available CPUs."""
    if n_jobs is None or int(n_jobs) == 0:
        return 1
    n_jobs = int(n_jobs)
    if n_jobs < 0:
        return max(1, os.cpu_count() or 1)
    return max(1, n_jobs)


def _condition_worker(payload: dict[str, Any]) -> dict[str, Any]:
    """Compute one non-baseline masking condition.

    This function is intentionally top-level so it can be pickled by
    ProcessPoolExecutor. It does not write files; the parent process keeps
    output ordering stable and writes the per-model CSV.
    """
    cond = payload["condition"]
    model_dir = Path(payload["model_dir"])
    gt = payload["gt"]
    base_merged = payload["base_merged"]
    json_filename = payload["json_filename"]
    positive_label = payload["positive_label"]
    positive_index = payload["positive_index"]
    decision_threshold = payload["decision_threshold"]
    n_bootstrap = payload["n_bootstrap"]
    cohort_label = payload["cohort_label"]

    preds = load_predictions(
        model_dir,
        cond,
        json_filename=json_filename,
        positive_label=positive_label,
        positive_index=positive_index,
    )
    if preds.empty:
        return {"condition": cond, "row": None, "log": f"  [SKIP] {cond}: no predictions found"}

    merged = preds.merge(gt, on="case_id")
    if "cohort" not in merged.columns:
        merged["cohort"] = merged["case_id"].map(infer_cohort_from_case_id)
    if cohort_label != "all":
        merged = merged[merged["cohort"] == cohort_label].copy()

    if len(merged) < 5:
        return {
            "condition": cond,
            "row": None,
            "log": f"  [SKIP] {cond}/{cohort_label}: too few cases ({len(merged)})",
        }

    common = base_merged[["case_id", "cohort", "target_class", "pos_prob"]].merge(
        merged[["case_id", "pos_prob"]],
        on="case_id",
        suffixes=("_base", "_cond"),
    )
    if len(common) < 5:
        return {
            "condition": cond,
            "row": None,
            "log": f"  [SKIP] {cond}: too few paired cases ({len(common)})",
        }

    y = common["target_class"].to_numpy(dtype=int)
    sb = common["pos_prob_base"].to_numpy(dtype=float)
    sc = common["pos_prob_cond"].to_numpy(dtype=float)

    row = {
        "cohort": cohort_label,
        "model": payload["model_name"],
        "condition": cond,
        "n_cases": len(common),
        "n_positive": int(y.sum()),
        "n_negative": int((y == 0).sum()),
        "decision_threshold": decision_threshold,
    }

    log_parts = [f"  {cond:<25}"]
    for metric_name in METRICS:
        metric_val = compute_metric(metric_name, y, sc, decision_threshold=decision_threshold)
        delta, p_boot, se = bootstrap_paired_pvalue(
            metric_name,
            y,
            sb,
            sc,
            n_bootstrap=n_bootstrap,
            decision_threshold=decision_threshold,
        )
        p_perm = signflip_permutation_pvalue(
            metric_name,
            y,
            sb,
            sc,
            n_permutations=n_bootstrap,
            decision_threshold=decision_threshold,
        )
        boot_vals = bootstrap_metric(
            metric_name,
            y,
            sc,
            n_bootstrap=n_bootstrap,
            decision_threshold=decision_threshold,
        )
        if len(boot_vals) == 0:
            ci_lo, ci_hi = np.nan, np.nan
        else:
            ci_lo, ci_hi = np.nanpercentile(boot_vals, [2.5, 97.5])

        sig_boot = add_significance_stars(p_boot)
        sig_perm = add_significance_stars(p_perm)

        row[f"{metric_name}"] = metric_val
        row[f"{metric_name}_ci_lower"] = ci_lo
        row[f"{metric_name}_ci_upper"] = ci_hi
        row[f"{metric_name}_delta_vs_baseline"] = delta
        row[f"{metric_name}_bootstrap_se"] = se
        row[f"{metric_name}_p_bootstrap"] = p_boot
        row[f"{metric_name}_p_signflip"] = p_perm
        row[f"{metric_name}_significance_bootstrap"] = sig_boot
        row[f"{metric_name}_significance_signflip"] = sig_perm

        disp_name = "AUROC" if metric_name == "auroc" else "wF1"
        log_parts.append(
            f"{disp_name}={metric_val:.4f} Δ={delta:+.4f} pB={p_boot:.4f}{sig_boot} pP={p_perm:.4f}{sig_perm}"
        )

    return {"condition": cond, "row": row, "log": "  | ".join(log_parts)}


def run_condition_jobs(
    *,
    conditions: list[str],
    model_name: str,
    model_dir: Path,
    gt: pd.DataFrame,
    base_merged: pd.DataFrame,
    n_bootstrap: int,
    json_filename: str | None,
    positive_label: str,
    positive_index: int,
    decision_threshold: float,
    cohort_label: str,
    n_jobs: int = 1,
) -> list[dict[str, Any]]:
    """Run non-baseline masking condition analyses, optionally in parallel."""
    non_baseline = [c for c in conditions if c != "baseline"]
    payloads = [
        {
            "condition": cond,
            "model_name": model_name,
            "model_dir": str(model_dir),
            "gt": gt,
            "base_merged": base_merged,
            "n_bootstrap": n_bootstrap,
            "json_filename": json_filename,
            "positive_label": positive_label,
            "positive_index": positive_index,
            "decision_threshold": decision_threshold,
            "cohort_label": cohort_label,
        }
        for cond in non_baseline
    ]

    if not payloads:
        return []

    n_jobs = min(resolve_n_jobs(n_jobs), len(payloads))
    if n_jobs <= 1:
        return [_condition_worker(payload) for payload in payloads]

    print(f"  Parallel masking jobs: {n_jobs} worker processes for {len(payloads)} conditions")
    results_by_condition: dict[str, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        future_to_cond = {executor.submit(_condition_worker, payload): payload["condition"] for payload in payloads}
        for future in as_completed(future_to_cond):
            cond = future_to_cond[future]
            try:
                results_by_condition[cond] = future.result()
            except Exception as e:
                results_by_condition[cond] = {
                    "condition": cond,
                    "row": None,
                    "log": f"  [ERROR] {cond}: worker failed: {e}",
                }

    # Preserve the original condition order in logs and output rows.
    return [results_by_condition[cond] for cond in non_baseline]


# ─────────────────────────────────────────────
# Main analysis
# ─────────────────────────────────────────────
def analyse_model(
    model_name: str,
    results_root: Path,
    gt: pd.DataFrame,
    n_bootstrap: int,
    output_dir: Path,
    json_filename: str | None = None,
    positive_label: str = "BRS3",
    positive_index: int = 2,
    decision_threshold: float = 0.5,
    cohort: str = "all",
    n_jobs: int = 1,
) -> pd.DataFrame:
    model_dir = results_root / model_name
    cohort_label = "all" if str(cohort).lower() == "all" else str(cohort).upper()
    print(f"\n{'=' * 60}")
    print(f"Model: {model_name} | Cohort: {cohort_label}")
    print(f"{'=' * 60}")

    if not model_dir.exists():
        print(f"  [SKIP] Missing model dir: {model_dir}")
        return pd.DataFrame()

    conditions = sorted([d.name for d in model_dir.iterdir() if d.is_dir()])
    ordered = [c for c in CONDITION_ORDER if c in conditions]
    extras = [c for c in conditions if c not in ordered]
    conditions = ordered + extras

    if "baseline" not in conditions:
        print(f"  [SKIP] No baseline found for {model_name}")
        return pd.DataFrame()

    base_preds = load_predictions(
        model_dir,
        "baseline",
        json_filename=json_filename,
        positive_label=positive_label,
        positive_index=positive_index,
    )
    base_merged = base_preds.merge(gt, on="case_id")
    if "cohort" not in base_merged.columns:
        base_merged["cohort"] = base_merged["case_id"].map(infer_cohort_from_case_id)
    if cohort_label != "all":
        base_merged = base_merged[base_merged["cohort"] == cohort_label].copy()

    if len(base_merged) < 5:
        print(f"  [SKIP] baseline/{cohort_label}: too few cases ({len(base_merged)})")
        return pd.DataFrame()

    y_base = base_merged["target_class"].to_numpy(dtype=int)
    s_base = base_merged["pos_prob"].to_numpy(dtype=float)

    base_stats = {}
    for metric_name in METRICS:
        metric_val = compute_metric(metric_name, y_base, s_base, decision_threshold=decision_threshold)
        boot_vals = bootstrap_metric(
            metric_name,
            y_base,
            s_base,
            n_bootstrap=n_bootstrap,
            decision_threshold=decision_threshold,
        )
        if len(boot_vals) == 0:
            ci_lo, ci_hi = np.nan, np.nan
            se = np.nan
        else:
            ci_lo, ci_hi = np.nanpercentile(boot_vals, [2.5, 97.5])
            se = float(np.nanstd(boot_vals, ddof=1)) if len(boot_vals) > 1 else 0.0
        base_stats[metric_name] = {
            "value": metric_val,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "bootstrap_se": se,
        }

    print(
        f"  Baseline AUROC: {base_stats['auroc']['value']:.4f} "
        f"[{base_stats['auroc']['ci_lower']:.4f}, {base_stats['auroc']['ci_upper']:.4f}] "
        f"(n={len(base_merged)}, cohort={cohort_label})"
    )
    print(
        f"  Baseline weighted F1: {base_stats['weighted_f1']['value']:.4f} "
        f"[{base_stats['weighted_f1']['ci_lower']:.4f}, {base_stats['weighted_f1']['ci_upper']:.4f}] "
        f"@ threshold={decision_threshold:.3f}"
    )

    rows = [{
        "cohort": cohort_label,
        "model": model_name,
        "condition": "baseline",
        "n_cases": len(base_merged),
        "n_positive": int(y_base.sum()),
        "n_negative": int((y_base == 0).sum()),
        "decision_threshold": decision_threshold,
        "auroc": base_stats["auroc"]["value"],
        "auroc_ci_lower": base_stats["auroc"]["ci_lower"],
        "auroc_ci_upper": base_stats["auroc"]["ci_upper"],
        "auroc_delta_vs_baseline": 0.0,
        "auroc_bootstrap_se": base_stats["auroc"]["bootstrap_se"],
        "auroc_p_bootstrap": np.nan,
        "auroc_p_signflip": np.nan,
        "auroc_significance_bootstrap": "—",
        "auroc_significance_signflip": "—",
        "weighted_f1": base_stats["weighted_f1"]["value"],
        "weighted_f1_ci_lower": base_stats["weighted_f1"]["ci_lower"],
        "weighted_f1_ci_upper": base_stats["weighted_f1"]["ci_upper"],
        "weighted_f1_delta_vs_baseline": 0.0,
        "weighted_f1_bootstrap_se": base_stats["weighted_f1"]["bootstrap_se"],
        "weighted_f1_p_bootstrap": np.nan,
        "weighted_f1_p_signflip": np.nan,
        "weighted_f1_significance_bootstrap": "—",
        "weighted_f1_significance_signflip": "—",
    }]

    condition_results = run_condition_jobs(
        conditions=conditions,
        model_name=model_name,
        model_dir=model_dir,
        gt=gt,
        base_merged=base_merged,
        n_bootstrap=n_bootstrap,
        json_filename=json_filename,
        positive_label=positive_label,
        positive_index=positive_index,
        decision_threshold=decision_threshold,
        cohort_label=cohort_label,
        n_jobs=n_jobs,
    )
    for result in condition_results:
        print(result["log"])
        if result["row"] is not None:
            rows.append(result["row"])

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / f"{model_name}_ablation.csv", index=False)
    return df


def load_existing_model_results(
    output_dir: Path,
    exclude_models: set[str] | None = None,
) -> list[pd.DataFrame]:
    """
    Reuse previously computed per-model CSVs from output_dir.

    This allows rerunning only failed/missing models while rebuilding the
    combined summary tables and heatmaps from old + new model results.
    """
    exclude_models = exclude_models or set()
    reused = []

    for csv_path in sorted(output_dir.glob("*_ablation.csv")):
        if csv_path.name == "all_models_ablation.csv":
            continue
        model_name = csv_path.stem.replace("_ablation", "")
        if model_name in exclude_models:
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[WARN] Could not read existing CSV {csv_path}: {e}")
            continue
        if df.empty or "model" not in df.columns:
            print(f"[WARN] Existing CSV invalid/empty, skipping: {csv_path}")
            continue
        reused.append(df)
        print(f"[REUSE] Loaded existing results for {model_name} from {csv_path.name}")

    return reused



def make_summary_table(all_results: list[pd.DataFrame], output_dir: Path) -> pd.DataFrame:
    combined = pd.concat(all_results, ignore_index=True)
    dedup_cols = ["model", "condition"]
    if "cohort" in combined.columns:
        dedup_cols = ["cohort"] + dedup_cols
    combined = combined.drop_duplicates(subset=dedup_cols, keep="last")
    combined.to_csv(output_dir / "all_models_ablation.csv", index=False)

    print("\n" + "=" * 70)
    print("SUMMARY: Mean metric per condition (across models)")
    print("=" * 70)

    for metric_name in METRICS:
        metric_pivot = combined.pivot_table(index="condition", columns="model", values=metric_name, aggfunc="first")
        metric_pivot[f"mean_{metric_name}"] = metric_pivot.mean(axis=1)
        metric_pivot = metric_pivot.sort_values(f"mean_{metric_name}", ascending=False)

        delta_pivot = combined.pivot_table(
            index="condition", columns="model", values=f"{metric_name}_delta_vs_baseline", aggfunc="first"
        )
        delta_pivot[f"mean_delta_{metric_name}"] = delta_pivot.mean(axis=1)

        sig_boot_pivot = combined.pivot_table(
            index="condition", columns="model", values=f"{metric_name}_significance_bootstrap", aggfunc="first"
        )
        sig_perm_pivot = combined.pivot_table(
            index="condition", columns="model", values=f"{metric_name}_significance_signflip", aggfunc="first"
        )

        metric_pivot.to_csv(output_dir / f"{metric_name}_summary_wide.csv")
        delta_pivot.to_csv(output_dir / f"{metric_name}_delta_summary_wide.csv")
        sig_boot_pivot.to_csv(output_dir / f"{metric_name}_significance_bootstrap_wide.csv")
        sig_perm_pivot.to_csv(output_dir / f"{metric_name}_significance_signflip_wide.csv")

        summary = delta_pivot[[f"mean_delta_{metric_name}"]].join(metric_pivot[[f"mean_{metric_name}"]])
        summary.index.name = "condition"
        print(f"\n[{metric_name}]")
        print(summary.to_string(float_format="{:.4f}".format))

    return combined



def _plot_metric_heatmap(combined: pd.DataFrame, metric_name: str, output_dir: Path):
    import matplotlib.pyplot as plt

    value_col = metric_name
    delta_col = f"{metric_name}_delta_vs_baseline"
    sig_col = f"{metric_name}_significance_signflip"
    disp_name = "ΔAUROC vs. Baseline" if metric_name == "auroc" else "Δweighted F1 vs. Baseline"
    title_name = "AUROC" if metric_name == "auroc" else "weighted F1"

    conditions = [c for c in CONDITION_ORDER if c in combined["condition"].unique()]
    conditions += [c for c in combined["condition"].unique() if c not in conditions]
    non_baseline = [c for c in conditions if c != "baseline"]

    baseline_df = (
        combined[combined["condition"] == "baseline"][["model", value_col]]
        .dropna()
        .drop_duplicates(subset=["model"])
        .sort_values(value_col, ascending=False)
    )
    model_order = baseline_df["model"].tolist()

    delta_data = combined[combined["condition"] != "baseline"].pivot_table(
        index="condition", columns="model", values=delta_col, aggfunc="first"
    )
    delta_data = delta_data.reindex([c for c in non_baseline if c in delta_data.index])

    if len(delta_data) == 0 or len(delta_data.columns) == 0:
        return

    value_data = combined[combined["condition"] != "baseline"].pivot_table(
        index="condition", columns="model", values=value_col, aggfunc="first"
    ).reindex(index=delta_data.index, columns=delta_data.columns)

    sig_data = combined[combined["condition"] != "baseline"].pivot_table(
        index="condition", columns="model", values=sig_col, aggfunc="first"
    ).reindex(index=delta_data.index, columns=delta_data.columns)

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
        [m.replace("task2_", "") for m in delta_data.columns],
        rotation=30,
        ha="right",
        fontsize=9,
    )
    ax.set_yticks(range(len(delta_data.index)))
    ax.set_yticklabels(delta_data.index, fontsize=8)

    for i in range(len(delta_data.index)):
        for j in range(len(delta_data.columns)):
            delta_val = delta_data.values[i, j]
            metric_val = value_data.values[i, j]
            sig = sig_data.values[i, j] if pd.notna(sig_data.values[i, j]) else ""
            if not np.isnan(delta_val) and not np.isnan(metric_val):
                ax.text(j, i, f"{metric_val:.3f}\n{sig}", ha="center", va="center", fontsize=7, color="black")

    plt.colorbar(im, ax=ax, label=disp_name)
    ax.set_title(
        f"Ablation Study: {title_name}\n"
        f"(cell text = {title_name}, color = Δ vs baseline; columns ordered by baseline best → worst)",
        fontsize=11,
        pad=12,
    )
    plt.tight_layout()
    plt.savefig(output_dir / f"ablation_heatmap_{metric_name}.pdf", bbox_inches="tight", dpi=150)
    plt.savefig(output_dir / f"ablation_heatmap_{metric_name}.png", bbox_inches="tight", dpi=150)
    plt.close()



def _plot_metric_ci_per_model(combined: pd.DataFrame, metric_name: str, output_dir: Path):
    import matplotlib.pyplot as plt

    metric_label = "AUROC" if metric_name == "auroc" else "weighted F1"
    value_col = metric_name
    lower_col = f"{metric_name}_ci_lower"
    upper_col = f"{metric_name}_ci_upper"
    delta_col = f"{metric_name}_delta_vs_baseline"

    conditions = [c for c in CONDITION_ORDER if c in combined["condition"].unique()]
    conditions += [c for c in combined["condition"].unique() if c not in conditions]

    for model in list(combined["model"].dropna().unique()):
        mdf = combined[combined["model"] == model].copy()
        mdf = mdf[mdf["condition"].isin(conditions)].set_index("condition").reindex(conditions).dropna(subset=[value_col])
        if mdf.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, max(4, len(mdf) * 0.45 + 1.5)))
        y = np.arange(len(mdf))
        colors = [
            "steelblue" if c == "baseline" else ("tomato" if mdf.loc[c, delta_col] < 0 else "seagreen")
            for c in mdf.index
        ]

        ax.barh(
            y,
            mdf[value_col],
            xerr=[mdf[value_col] - mdf[lower_col], mdf[upper_col] - mdf[value_col]],
            color=colors,
            ecolor="gray",
            capsize=3,
            height=0.6,
            alpha=0.85,
        )

        if "baseline" in mdf.index:
            base_val = mdf.loc["baseline", value_col]
            ax.axvline(base_val, color="navy", linestyle="--", linewidth=1.2, alpha=0.7, label=f"Baseline ({base_val:.3f})")

        ax.set_yticks(y)
        ax.set_yticklabels(mdf.index, fontsize=8)
        ax.set_xlabel(f"{metric_label} (95% bootstrap CI)")
        ax.set_title(f"{model.replace('task2_', '').upper()} — Ablation {metric_label}", fontsize=11)
        ax.legend(fontsize=8)
        xmin = max(0.0, float(np.nanmin(mdf[lower_col])) - 0.02)
        xmax = min(1.0, float(np.nanmax(mdf[upper_col])) + 0.05)
        ax.set_xlim(xmin, xmax)
        plt.tight_layout()
        plt.savefig(output_dir / f"{model}_{metric_name}_ci.png", bbox_inches="tight", dpi=150)
        plt.close()



def plot_results(combined: pd.DataFrame, output_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError:
        print("[INFO] matplotlib not available, skipping plots")
        return

    for metric_name in METRICS:
        _plot_metric_heatmap(combined, metric_name, output_dir)
        _plot_metric_ci_per_model(combined, metric_name, output_dir)

    print(f"Saved plots → {output_dir}/")


# ─────────────────────────────────────────────
# Cohort CLI helper
# ─────────────────────────────────────────────
def normalize_requested_cohorts(raw_cohorts: list[str]) -> list[str]:
    """Normalize --cohorts values while preserving user-specified order."""
    out = []
    for c in raw_cohorts:
        c_norm = str(c).strip().upper()
        if c_norm in {"ALL", "A"}:
            c_norm = "all"
        elif c_norm not in {"B", "U"}:
            raise ValueError(f"Unsupported cohort '{c}'. Use one or more of: all B U")
        if c_norm not in out:
            out.append(c_norm)
    return out or ["all"]


def cohort_output_subdir(output_dir: Path, cohort: str, n_cohorts: int) -> Path:
    """
    Preserve the old output layout for a single overall run, but create clean
    subfolders when running multiple strata.
    """
    if n_cohorts == 1 and cohort == "all":
        return output_dir
    if cohort == "all":
        return output_dir / "cohort_all"
    return output_dir / f"cohort_{cohort}"


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="CHIMERA Task2 masking ablation — AUROC + weighted F1 analysis")
    parser.add_argument("--results_dir", required=True, help="Path to Results_Masking/")
    parser.add_argument("--gt_csv", required=True, help="Path to task2_test.csv")
    parser.add_argument("--n_bootstrap", type=int, default=10000, help="Bootstrap/permutation iterations (default: 10000)")
    parser.add_argument("--n_jobs", type=int, default=1, help="Parallel worker processes for masking conditions per model. Use -1 for all CPUs. Default: 1")
    parser.add_argument("--output_dir", default="./ablation_results_task2", help="Output directory")
    parser.add_argument("--models", nargs="+", default=MODELS, help="Subset of models to analyse")
    parser.add_argument("--reuse_existing_csvs", action="store_true", help="Reuse existing <model>_ablation.csv files in output_dir for models not listed in --models")
    parser.add_argument("--plots_only", action="store_true", help="Skip fresh analysis and rebuild summary tables/plots only from existing per-model CSVs in output_dir")
    parser.add_argument("--json_filename", default=None, help="Optional exact JSON filename inside each case folder")
    parser.add_argument("--positive_label", default="BRS3", help="Positive class label (default: BRS3)")
    parser.add_argument("--positive_index", type=int, default=2, help="If probabilities are stored as a list, index of the positive class (default: 2 for BRS3)")
    parser.add_argument("--decision_threshold", type=float, default=0.5, help="Threshold for converting positive probability to binary prediction for weighted F1")
    parser.add_argument(
        "--cohorts",
        nargs="+",
        default=["all"],
        help="Cohort strata to analyse. Use any combination of: all B U. Cohort is inferred from case folders like 2B_xxx or 2U_xxx.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_root = Path(args.results_dir)

    print(f"Loading ground truth from: {args.gt_csv}")
    gt = load_ground_truth(args.gt_csv, positive_label=args.positive_label)
    print(
        f"  {len(gt)} cases loaded | positives: {int(gt['target_class'].sum())} | "
        f"negatives: {int((gt['target_class'] == 0).sum())}"
    )

    requested_cohorts = normalize_requested_cohorts(args.cohorts)
    n_jobs = resolve_n_jobs(args.n_jobs)
    print(f"Requested cohort analyses: {', '.join(requested_cohorts)}")
    print(f"Parallel workers per model: {n_jobs}")

    for cohort in requested_cohorts:
        cohort_dir = cohort_output_subdir(output_dir, cohort, n_cohorts=len(requested_cohorts))
        cohort_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'#' * 70}")
        print(f"Running cohort stratum: {cohort} → {cohort_dir}")
        print(f"{'#' * 70}")

        all_results = []

        if args.reuse_existing_csvs or args.plots_only:
            all_results.extend(
                load_existing_model_results(
                    cohort_dir,
                    exclude_models=set() if args.plots_only else set(args.models),
                )
            )

        if not args.plots_only:
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
                    json_filename=args.json_filename,
                    positive_label=args.positive_label,
                    positive_index=args.positive_index,
                    decision_threshold=args.decision_threshold,
                    cohort=cohort,
                    n_jobs=n_jobs,
                )
                if not df.empty:
                    all_results.append(df)

        if all_results:
            combined = make_summary_table(all_results, cohort_dir)
            plot_results(combined, cohort_dir)
        else:
            print(
                "[WARN] No model results available for this cohort. "
                "Use --reuse_existing_csvs or check results_dir / model names / JSON parsing / cohort labels."
            )

    print(f"\n✓ Results saved under: {output_dir}/")
    print("  When multiple cohorts are requested, outputs are placed in cohort_all/, cohort_B/, and/or cohort_U/.")
    print("  • all_models_ablation.csv                      — full per-model-condition results for that cohort stratum")
    print("  • auroc_summary_wide.csv                       — AUROC summary table")
    print("  • auroc_delta_summary_wide.csv                 — ΔAUROC vs baseline")
    print("  • weighted_f1_summary_wide.csv                 — weighted F1 summary table")
    print("  • weighted_f1_delta_summary_wide.csv           — Δweighted F1 vs baseline")
    print("  • ablation_heatmap_auroc.png/pdf               — AUROC heatmap")
    print("  • ablation_heatmap_weighted_f1.png/pdf         — weighted F1 heatmap")
    print("  • <model>_auroc_ci.png                         — per-model AUROC CI plots")
    print("  • <model>_weighted_f1_ci.png                   — per-model weighted F1 CI plots")
    print("\nExamples:")
    print("  • Rerun only one repaired model, but reuse old CSVs + remake heatmaps:")
    print("      python task2_ablation_auc_weightedf1_analysis_updated.py --results_dir ... --gt_csv ... --output_dir ... --models task2_wl --reuse_existing_csvs")
    print("  • Run overall + B/U cohort-specific analyses:")
    print("      python task2_ablation_auc_weightedf1_analysis_updated.py --results_dir ... --gt_csv ... --output_dir ... --cohorts all B U")
    print("  • Run only B and U cohort-specific analyses with parallel masking jobs:")
    print("      python task2_ablation_auc_weightedf1_analysis_updated.py --results_dir ... --gt_csv ... --output_dir ... --cohorts B U --n_jobs 8")
    print("  • Rebuild summary tables/plots only from existing per-model CSVs:")
    print("      python task2_ablation_auc_weightedf1_analysis_updated.py --results_dir ... --gt_csv ... --output_dir ... --plots_only --cohorts all B U")


if __name__ == "__main__":
    main()
