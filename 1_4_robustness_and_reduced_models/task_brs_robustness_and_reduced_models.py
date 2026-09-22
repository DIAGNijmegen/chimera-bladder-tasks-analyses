#!/usr/bin/env python3
"""
Unified Task BRS (Task 2) robustness + reduced-model (non-image-derived-features, nif) analysis, producing
two LaTeX tables (one for F1, one for AUC), each with, per team x {Combined, Cohort B, Cohort U}:
  - Metric (Full model), 95% bootstrap CI                              [all teams]
  - Metric (Reduced / non-image-derived-features model), 95% CI        [nif teams only]
  - Delta = Full - Reduced, 95% CI, FDR-adjusted p-value                [nif teams only]
  - A dagger marker on the B/U cohort label if Full-model B vs U differs significantly.
"""

from glob import glob
import json
import os

import numpy as np
import pandas as pd
import statsmodels.stats.multitest
from sklearn.metrics import f1_score, roc_auc_score

#############################################################
# Paths and directory
#############################################################
path_task2 = '/path/to/task2_results.csv'

dir_output = '/output/path'
os.makedirs(dir_output, exist_ok=True)

# Reduced ("nif") model per-case prediction directories -- only these 3 teams submitted a reduced (non-image-derived-features) model.
l_teams_nif = ['BioToTem', 'BUAA-REMEX', 'MITEL-UNIUD']

dir_biototem = "path/to/reduced_model/biototem/pred"
dir_buaa_remex = "path/to/reduced_model/buaa_remex/pred"
dir_mitel_uniud = "path/to/reduced_model/mitel_uniud/pred"

path_output_biototem = os.path.join(dir_biototem, "nif_task2_biototem_v2_summary_brs_probability.json")
path_output_buaa_remex = os.path.join(dir_buaa_remex, "nif_task2_BUAA_REMEX_summary_brs_probability.json")
path_output_mitel_uniud = os.path.join(dir_mitel_uniud, "nif_task2_MITEL_UNIUD_summary_brs_probability.json")

N_BOOTSTRAP = 10000
RANDOM_SEED = 1


#############################################################
# 0. Prepare the full-model dataframe, all leaderboard teams.
#############################################################
df_task2 = pd.read_csv(path_task2)

# Normalize team names to match the nif dataframe's naming (ref email 20151217).
df_task2['team_name'] = df_task2['team_name'].replace({
    'BUAA-REMAX': 'BUAA-REMEX',
    'Biototem': 'BioToTem',
})

# Dictionary of leaderboard F1 (for ordering the table's rows).
dict_teams_f1 = {
    'BioToTem':     0.7261,
    'GRIS':         0.6921,
    'BUAA-REMEX':   0.6726,
    'CADGEN-BIIT':  0.6593,
    'WL':           0.6518,
    'MITEL-UNIUD':  0.6344,
    'HKKH':         0.6105,
    'Aillis':       0.5981,
    }

l_teams = df_task2['team_name'].unique().tolist()
l_teams = sorted(l_teams, key=lambda x: dict_teams_f1.get(x, 0), reverse=True)


#############################################################
# Functions
#############################################################
def _f1_metric(df_subset: pd.DataFrame) -> float:
    """F1 at a fixed 0.5 probability threshold."""
    if len(df_subset) < 2:
        return np.nan
    y_true = df_subset['case_id_gt'].values
    if len(np.unique(y_true)) < 2:
        return np.nan
    y_pred = (df_subset['case_id_pred'].values >= 0.5).astype(int)
    try:
        return f1_score(y_true, y_pred, average='weighted')
    except ValueError:
        return np.nan


def _auc_metric(df_subset: pd.DataFrame) -> float:
    """ROC-AUC on the raw predicted probabilities (no thresholding)."""
    if len(df_subset) < 2:
        return np.nan
    y_true = df_subset['case_id_gt'].values
    if len(np.unique(y_true)) < 2:
        return np.nan
    try:
        return roc_auc_score(y_true, df_subset['case_id_pred'].values)
    except ValueError:
        return np.nan


def _bootstrap_metric_single(df_group: pd.DataFrame, metric_fn, n_bootstrap: int):
    """Independent percentile bootstrap of metric_fn over one group (no pairing)."""
    n = len(df_group)
    if n < 2:
        return np.nan, np.nan, np.nan
    observed = metric_fn(df_group)
    boot = np.full(n_bootstrap, np.nan)
    for i in range(n_bootstrap):
        idx = np.random.randint(0, n, size=n)
        boot[i] = metric_fn(df_group.iloc[idx])
    valid = boot[np.isfinite(boot)]
    if len(valid) == 0:
        return observed, np.nan, np.nan
    ci_lower, ci_upper = np.quantile(valid, [0.025, 0.975])
    return observed, float(ci_lower), float(ci_upper)

def compute_full_model_summary(df_task2: pd.DataFrame, l_teams: list, metric_fn, n_bootstrap: int) -> pd.DataFrame:
    # Full-model metric, per team x {Combined, B, U} -- all leaderboard teams.
    rows = []
    for team in l_teams:
        print(f"[full model] Processing team: {team}")
        df_team = df_task2[df_task2['team_name'] == team]
        strata = {
            'Combined': df_team,
            'B': df_team[df_team['cohort'] == 'B'],
            'U': df_team[df_team['cohort'] == 'U'],
        }
        for stratum_label, df_stratum in strata.items():
            observed, ci_lower, ci_upper = _bootstrap_metric_single(df_stratum, metric_fn, n_bootstrap)
            rows.append({
                'team_name': team,
                'stratum': stratum_label,
                'n': len(df_stratum),
                'metric_full': observed,
                'metric_full_ci_lower': ci_lower,
                'metric_full_ci_upper': ci_upper,
            })
    return pd.DataFrame(rows)


#############################################################
# Full-model B vs U robustness comparison, per team - unpaired bootstrap since cohort B and U are disjoint patient populations.
#############################################################
def _unpaired_bootstrap_BU(df_b: pd.DataFrame, df_u: pd.DataFrame, metric_fn, n_bootstrap: int):
    n_b, n_u = len(df_b), len(df_u)
    nan_arr = np.full(n_bootstrap, np.nan)
    if n_b < 2 or n_u < 2:
        return nan_arr.copy(), nan_arr.copy(), nan_arr.copy()
    out_b = np.full(n_bootstrap, np.nan)
    out_u = np.full(n_bootstrap, np.nan)
    out_diff = np.full(n_bootstrap, np.nan)
    for i in range(n_bootstrap):
        idx_b = np.random.randint(0, n_b, size=n_b)
        idx_u = np.random.randint(0, n_u, size=n_u)
        m_b = metric_fn(df_b.iloc[idx_b])
        m_u = metric_fn(df_u.iloc[idx_u])
        if np.isfinite(m_b) and np.isfinite(m_u):
            out_b[i] = m_b
            out_u[i] = m_u
            out_diff[i] = m_b - m_u
    return out_diff, out_b, out_u


def bootstrap_test_metric_BU(df_task2: pd.DataFrame, team: str, metric_fn, n_bootstrap: int) -> dict:
    df_b = df_task2[(df_task2['team_name'] == team) & (df_task2['cohort'] == 'B')]
    df_u = df_task2[(df_task2['team_name'] == team) & (df_task2['cohort'] == 'U')]
    boot_diff, _, _ = _unpaired_bootstrap_BU(df_b, df_u, metric_fn, n_bootstrap)
    m_b_observed = metric_fn(df_b)
    m_u_observed = metric_fn(df_u)
    valid_diff = boot_diff[np.isfinite(boot_diff)]
    ci_diff = tuple(np.quantile(valid_diff, [0.025, 0.975])) if len(valid_diff) else (np.nan, np.nan)
    if len(valid_diff) == 0:
        p_value = np.nan
    else:
        p_low = np.mean(valid_diff <= 0)
        p_high = np.mean(valid_diff >= 0)
        p_value = min(1.0, 2 * min(p_low, p_high))
    return {
        'team_name': team,
        'metric_full_cohort_B': m_b_observed,
        'metric_full_cohort_U': m_u_observed,
        'difference_B_minus_U': m_b_observed - m_u_observed,
        'difference_ci_lower': ci_diff[0],
        'difference_ci_upper': ci_diff[1],
        'p_value': p_value,
    }


def run_BU_robustness_tests(df_task2: pd.DataFrame, l_teams: list, metric_fn, n_bootstrap: int, path_out: str) -> pd.DataFrame:
    rows = [bootstrap_test_metric_BU(df_task2, team, metric_fn, n_bootstrap) for team in l_teams]
    df_all = pd.DataFrame(rows)
    mask = df_all['p_value'].notna()
    df_all['p_adj'] = np.nan
    if mask.any():
        df_all.loc[mask, 'p_adj'] = statsmodels.stats.multitest.multipletests(
            df_all.loc[mask, 'p_value'], method='fdr_bh'
        )[1]
    df_all.to_csv(path_out, index=False)
    print(f"Saved supplementary B-vs-U robustness test to: {path_out}\n")
    return df_all


#############################################################
# Reduced ("nif") model data loading -- 3 teams only.
#############################################################
def load_json_file(*, location):
    with open(location, "r") as f:
        return json.loads(f.read())


def write_json_file(*, location, content):
    with open(location, "w") as f:
        f.write(json.dumps(content, indent=4))


def summarize_json_files(l_json_files, path_summary_output):
    summary_list = []
    for path_json_output in l_json_files:
        case_id = os.path.basename(path_json_output).split("_brs_probability")[0]
        raw_pred = load_json_file(location=path_json_output)
        case_id_pred = raw_pred['case_id_pred'] if isinstance(raw_pred, dict) else (
            raw_pred[0] if isinstance(raw_pred, list) else raw_pred
        )
        summary_list.append({"case_id": case_id, "case_id_pred": case_id_pred})
    write_json_file(location=path_summary_output, content=summary_list)


def build_nif_full_and_reduced_dataframe(df_task2_all: pd.DataFrame) -> pd.DataFrame:
    """
    Full (from the shared leaderboard dataframe, already cohort-tagged) vs reduced (non-image-derived-features) model predictions, 
    for the 3 teams that submitted a reduced model. The export to CSV is commented because it contains sensitive information (ground truth and predictions) and should not be shared publicly.
    """
    l_json_biototem = glob(os.path.join(dir_biototem, "*_brs_probability.json"))
    l_json_buaa_remex = glob(os.path.join(dir_buaa_remex, "*_brs_probability.json"))
    l_json_mitel_uniud = glob(os.path.join(dir_mitel_uniud, "*_brs_probability.json"))
    #
    l_json_biototem = [f for f in l_json_biototem if 'summary' not in os.path.basename(f)]
    l_json_buaa_remex = [f for f in l_json_buaa_remex if 'summary' not in os.path.basename(f)]
    l_json_mitel_uniud = [f for f in l_json_mitel_uniud if 'summary' not in os.path.basename(f)]
    #
    summarize_json_files(l_json_biototem, path_output_biototem)
    summarize_json_files(l_json_buaa_remex, path_output_buaa_remex)
    summarize_json_files(l_json_mitel_uniud, path_output_mitel_uniud)
    #
    df_orig = df_task2_all[df_task2_all['team_name'].isin(l_teams_nif)].copy()
    df_orig['model_variant'] = 'full'
    #
    def _load_nif_team(path_summary, team_name):
        df_nif_team = pd.read_json(path_summary, orient="records", lines=False)
        df_nif_team['team_name'] = team_name
        df_orig_team = df_orig[df_orig['team_name'] == team_name]
        return df_nif_team.merge(df_orig_team[['case_id', 'case_id_gt']], on='case_id', how='left')
    #
    df_nif_biototem = _load_nif_team(path_output_biototem, 'BioToTem')
    df_nif_buaa_remex = _load_nif_team(path_output_buaa_remex, 'BUAA-REMEX')
    df_nif_mitel_uniud = _load_nif_team(path_output_mitel_uniud, 'MITEL-UNIUD')
    #
    df_nif = pd.concat([df_nif_biototem, df_nif_buaa_remex, df_nif_mitel_uniud], ignore_index=True)
    df_nif['model_variant'] = 'reduced'
    # Reduced-model JSONs carry no 'cohort' column -- derive it from case_id, equivalent to the leaderboard's own 'cohort' 
    df_nif['cohort'] = np.where(
        df_nif['case_id'].str.contains("2B_"), "B",
        np.where(df_nif['case_id'].str.contains("2U_"), "U", "Unknown"),
    )
    #
    df_combined = pd.concat([df_orig, df_nif], ignore_index=True)
    # path_out = os.path.join(dir_output, 'task2_nif_combined_full_and_reduced.csv')
    # df_combined.to_csv(path_out, index=False)
    return df_combined


#############################################################
# Paired bootstrap (matched by case_id): full vs reduced model metric, per team x {Combined, B, U} -- 3 nif teams only.
#############################################################
def _paired_bootstrap_metric(df_full: pd.DataFrame, df_reduced: pd.DataFrame, metric_fn, n_bootstrap: int):
    """
    Paired bootstrap over patients. Each iteration resamples patients (case_id) with replacement once, then uses that SAME set of patients to
    recompute the metric for both the full and reduced model, so the two model variants are always compared on identical bootstrap samples.
    """
    case_ids = np.intersect1d(df_full['case_id'].values, df_reduced['case_id'].values)
    n = len(case_ids)
    nan_arr = np.full(n_bootstrap, np.nan)
    if n < 2:
        return nan_arr.copy(), nan_arr.copy(), case_ids
    #
    df_full_idx = df_full.set_index('case_id').loc[case_ids]
    df_reduced_idx = df_reduced.set_index('case_id').loc[case_ids]
    #
    out_diff = np.full(n_bootstrap, np.nan)
    out_reduced = np.full(n_bootstrap, np.nan)
    for i in range(n_bootstrap):
        sample_pos = np.random.randint(0, n, size=n)
        m_full = metric_fn(df_full_idx.iloc[sample_pos])
        m_reduced = metric_fn(df_reduced_idx.iloc[sample_pos])
        if np.isfinite(m_full) and np.isfinite(m_reduced):
            out_reduced[i] = m_reduced
            out_diff[i] = m_full - m_reduced
    return out_diff, out_reduced, case_ids


def bootstrap_test_full_vs_reduced(df_stratum: pd.DataFrame, team: str, metric_fn, n_bootstrap: int) -> dict:
    df_team = df_stratum[(df_stratum['team_name'] == team) & (df_stratum['model_variant'].isin(['full', 'reduced']))]
    df_full = df_team[df_team['model_variant'] == 'full']
    df_reduced = df_team[df_team['model_variant'] == 'reduced']
    #
    boot_diff, boot_reduced, paired_case_ids = _paired_bootstrap_metric(df_full, df_reduced, metric_fn, n_bootstrap)
    #
    metric_full_observed = metric_fn(df_full[df_full['case_id'].isin(paired_case_ids)])
    metric_reduced_observed = metric_fn(df_reduced[df_reduced['case_id'].isin(paired_case_ids)])
    observed_diff = metric_full_observed - metric_reduced_observed
    #
    valid_diff = boot_diff[np.isfinite(boot_diff)]
    valid_reduced = boot_reduced[np.isfinite(boot_reduced)]
    #
    ci_reduced = tuple(np.quantile(valid_reduced, [0.025, 0.975])) if len(valid_reduced) else (np.nan, np.nan)
    ci_diff = tuple(np.quantile(valid_diff, [0.025, 0.975])) if len(valid_diff) else (np.nan, np.nan)
    #
    if len(valid_diff) == 0:
        p_value = np.nan
    else:
        # Two-sided bootstrap p-value: twice the smaller tail of the
        # difference distribution on either side of zero.
        p_low = np.mean(valid_diff <= 0)
        p_high = np.mean(valid_diff >= 0)
        p_value = min(1.0, 2 * min(p_low, p_high))
    #
    return {
        'team_name': team,
        'n_paired_patients': len(paired_case_ids),
        'metric_reduced': metric_reduced_observed,
        'metric_reduced_ci_lower': ci_reduced[0],
        'metric_reduced_ci_upper': ci_reduced[1],
        'delta_metric_full_minus_reduced': observed_diff,
        'delta_metric_ci_lower': ci_diff[0],
        'delta_metric_ci_upper': ci_diff[1],
        'p_value': p_value,
    }


def run_nif_bootstrap_tests(df_nif_combined: pd.DataFrame, l_teams_nif: list, metric_fn, n_bootstrap: int, path_out: str) -> pd.DataFrame:
    strata = {
        'Combined': df_nif_combined,
        'B': df_nif_combined[df_nif_combined['cohort'] == 'B'],
        'U': df_nif_combined[df_nif_combined['cohort'] == 'U'],
    }
    rows = []
    for stratum_label, df_stratum in strata.items():
        for team in l_teams_nif:
            print(f"[nif full vs reduced] Processing team: {team}, stratum: {stratum_label}")
            result = bootstrap_test_full_vs_reduced(df_stratum, team, metric_fn, n_bootstrap)
            result['stratum'] = stratum_label
            rows.append(result)
    df_all = pd.DataFrame(rows)
    # Single FDR correction across the whole family of Delta-F1 tests shown in the table.
    mask = df_all['p_value'].notna()
    df_all['p_adj'] = np.nan
    if mask.any():
        df_all.loc[mask, 'p_adj'] = statsmodels.stats.multitest.multipletests(
            df_all.loc[mask, 'p_value'], method='fdr_bh'
        )[1]
    df_all.to_csv(path_out, index=False)
    print(f"Saved nif full-vs-reduced bootstrap test to: {path_out}\n")
    return df_all


#############################################################
# 6. Merge full-model + nif summaries, then render the unified LaTeX table. Row-per-cohort layout (Combined/B/U as three rows per team, via \multirow on the team name), matching the paper's existing table style. Requires \usepackage{multirow} in the LaTeX preamble.
#############################################################
def fmt_val_ci(val, lo, hi) -> str:
    if pd.isna(val):
        return "--"
    return f"{val:.3f} [{lo:.3f}, {hi:.3f}]"


def fmt_delta_ci(val, lo, hi) -> str:
    if pd.isna(val):
        return "--"
    return f"{val:+.3f} [{lo:.3f}, {hi:.3f}]"


def fmt_p_adj(p_adj) -> str:
    if pd.isna(p_adj):
        return "--"
    return f"{p_adj:.3f}"


def build_latex_table(df_merged: pd.DataFrame, df_BU_robustness: pd.DataFrame, team_order: list,
                       caption: str, label: str, metric_label: str = "F1", bu_alpha: float = 0.05) -> str:
    strata_order = ['Combined', 'B', 'U']
    cohort_label = {'Combined': 'Combined', 'B': 'B', 'U': 'U'}
    col_spec = "llcccc"
    #
    lines = []
    lines.append("\\begin{table}[H]")
    lines.append("  \\centering")
    lines.append(f"  \\caption{{{caption}}}")
    lines.append("  \\begin{adjustbox}{width=1\\textwidth}")
    lines.append(f"  \\begin{{tabular}}{{{col_spec}}}")
    metric_header = ("\\textbf{Team} & \\textbf{Cohort} & "
                      f"\\textbf{{{metric_label} (Full) [95\\% CI]}} & "
                      f"\\textbf{{{metric_label} (Reduced) [95\\% CI]}} & "
                      f"\\textbf{{$\\Delta${metric_label} [95\\% CI]}} & \\textbf{{p\\_adj}}")
    lines.append(f"  {metric_header} \\\\")
    lines.append("  \\hline")
    #
    df_indexed = df_merged.set_index(['team_name', 'stratum'])
    df_BU_indexed = df_BU_robustness.set_index('team_name')
    for team in team_order:
        bu_significant = False
        if team in df_BU_indexed.index:
            p_bu = df_BU_indexed.loc[team].get('p_adj')
            bu_significant = pd.notna(p_bu) and p_bu < bu_alpha
        #
        team_label = team.replace('_', '\\_')
        for i, stratum in enumerate(strata_order):
            first_col = f"\\multirow{{3}}{{*}}{{{team_label}}}" if i == 0 else ""
            cohort_cell = cohort_label[stratum]
            if stratum in ('B', 'U') and bu_significant:
                cohort_cell += "$^{\\dagger}$"
            #
            if (team, stratum) in df_indexed.index:
                r = df_indexed.loc[(team, stratum)]
                metric_full_cell = fmt_val_ci(r['metric_full'], r['metric_full_ci_lower'], r['metric_full_ci_upper'])
                metric_reduced_cell = fmt_val_ci(r.get('metric_reduced'), r.get('metric_reduced_ci_lower'), r.get('metric_reduced_ci_upper'))
                delta_cell = fmt_delta_ci(r.get('delta_metric_full_minus_reduced'), r.get('delta_metric_ci_lower'), r.get('delta_metric_ci_upper'))
                p_adj_cell = fmt_p_adj(r.get('p_adj'))
            else:
                metric_full_cell, metric_reduced_cell, delta_cell, p_adj_cell = "--", "--", "--", "--"
            #
            lines.append(f"  {first_col} & {cohort_cell} & {metric_full_cell} & {metric_reduced_cell} & {delta_cell} & {p_adj_cell} \\\\")
        lines.append("  \\hline")
    lines.append("  \\end{tabular}")
    lines.append("  \\end{adjustbox}")
    lines.append(f"  \\label{{{label}}}")
    lines.append("\\end{table}")
    return "\n".join(lines)


#############################################################
# 7. Per-metric pipeline: run everything above for one metric_fn and render  its own LaTeX table. Called once for F1, once for AUC.
#############################################################
def run_metric_pipeline(df_task2: pd.DataFrame, df_nif_combined: pd.DataFrame, l_teams: list, l_teams_nif: list,metric_fn, metric_label: str, metric_slug: str, n_bootstrap: int, dir_output: str):
    df_full_summary = compute_full_model_summary(df_task2, l_teams, metric_fn, n_bootstrap)
    df_BU_robustness = run_BU_robustness_tests(
        df_task2, l_teams, metric_fn, n_bootstrap,
        os.path.join(dir_output, f'task2_bootstrap_test_{metric_slug}_full_B_vs_U.csv'),
    )
    df_nif_summary = run_nif_bootstrap_tests(
        df_nif_combined, l_teams_nif, metric_fn, n_bootstrap,
        os.path.join(dir_output, f'task2_bootstrap_test_{metric_slug}_full_vs_reduced.csv'),
    )
    df_merged = df_full_summary.merge(
        df_nif_summary[[
            'team_name', 'stratum', 'metric_reduced', 'metric_reduced_ci_lower', 'metric_reduced_ci_upper',
            'delta_metric_full_minus_reduced', 'delta_metric_ci_lower', 'delta_metric_ci_upper', 'p_adj',
        ]],
        on=['team_name', 'stratum'], how='left',
    )
    path_merged_csv = os.path.join(dir_output, f'task2_robustness_nif_unified_summary_{metric_slug}.csv')
    df_merged.to_csv(path_merged_csv, index=False)
    print(f"Saved unified {metric_label} summary to: {path_merged_csv}\n")
    # n per stratum, for the caption -- reported as a range if it varies across teams
    # (e.g. a team missing a prediction for one patient).
    n_by_stratum = df_full_summary.groupby('stratum')['n'].agg(['min', 'max'])
    def _n_str(stratum: str) -> str:
        lo, hi = n_by_stratum.loc[stratum, 'min'], n_by_stratum.loc[stratum, 'max']
        return f"n={hi}" if lo == hi else f"n={lo}-{hi}"
    caption = (
        f"Task BRS: full vs. reduced structured-data models, evaluated on {metric_label} and stratified "
        f"by cohort (Combined = full test set, {_n_str('Combined')}; B = Erasmus Cohort B, {_n_str('B')}; "
        f"U = Urolife Cohort U, {_n_str('U')}). "
        f"\\textit{{Legend: $\\Delta${metric_label} = {metric_label}(Full) $-$ {metric_label}(Reduced), "
        "positive values indicate the full model outperformed the reduced model; shown only for the "
        "three teams that submitted a reduced (non-image-derived-features) model, -- otherwise. All "
        "CIs are 95\\% percentile intervals from a paired patient-level bootstrap (10,000 resamples; "
        "full and reduced models evaluated on the identical resampled patients in each replicate); "
        "p\\_adj = two-sided bootstrap p-value for the difference, Benjamini-Hochberg corrected across "
        f"all full-vs-reduced comparisons in this table. $^{{\\dagger}}$ on the B/U cohort label indicates "
        f"{metric_label}(B) differs significantly from {metric_label}(U) for the full model (independent "
        "bootstrap, since cohort B and U are disjoint patient populations; FDR-adjusted $p<0.05$, "
        "corrected across all teams).}"
    )
    tex = build_latex_table(
        df_merged, df_BU_robustness, l_teams, caption,
        f"tab:task2_robustness_nif_unified_{metric_slug}", metric_label=metric_label,
    )
    path_tex = os.path.join(dir_output, f'tab_task2_robustness_nif_unified_{metric_slug}.tex')
    with open(path_tex, 'w') as f:
        f.write(tex + "\n")
    print(f"Saved unified {metric_label} LaTeX table to: {path_tex}")
    return df_merged, df_BU_robustness


if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)

    df_nif_combined = build_nif_full_and_reduced_dataframe(df_task2)

    df_merged_f1, df_BU_robustness_f1 = run_metric_pipeline(df_task2, df_nif_combined, l_teams, l_teams_nif, _f1_metric, "F1", "f1", N_BOOTSTRAP, dir_output)

    df_merged_auc, df_BU_robustness_auc = run_metric_pipeline(df_task2, df_nif_combined, l_teams, l_teams_nif, _auc_metric, "AUC", "auc", N_BOOTSTRAP, dir_output)
