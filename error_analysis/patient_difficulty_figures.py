"""
Patient-level difficulty figures for the BRS and Progression tasks.

Inputs   : Task2_test.csv, Task3_test.csv, clinical_with_criteria_all.csv (hidden for challenge purpose)
Outputs  : task2_brs_difficulty.{png,pdf}, task3_progression_difficulty.{png,pdf}
           task2_easy_hard_clinical_tests.csv, task3_easy_hard_clinical_tests.csv
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FixedLocator
import matplotlib.patheffects as pe
from scipy import stats

# ---- edit these two paths ----
IN = Path(".")             # folder holding the three input CSVs
OUT = Path(".")            # folder for figures and statistics tables
OUT.mkdir(parents=True, exist_ok=True)
TASK2_CSV = IN / "Task2_test.csv"
TASK3_CSV = IN / "Task3_test.csv"
CLINICAL_CSV = IN / "clinical_with_criteria_all.csv"


# ----------------------------------------------------------------------
# Style
# ----------------------------------------------------------------------
def use_paper_style() -> None:
    preferred = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
    available = {f.name for f in mpl.font_manager.fontManager.ttflist}
    family = next((f for f in preferred if f in available), "DejaVu Sans")
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [family],
        "font.size": 9,
        "axes.linewidth": 0.6,
        "axes.edgecolor": "#3F3F46",
        "text.color": "#18181B",
        "axes.labelcolor": "#18181B",
        "xtick.color": "#3F3F46",
        "ytick.color": "#3F3F46",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,   # editable text in Illustrator / Inkscape
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


GREY_MISSING = "#D4D4D8"
INK = "#18181B"
MUTED = "#71717A"
RULE = "#3F3F46"

# Diverging map centred on the 0.5 decision threshold.
CMAP_DIVERGING = LinearSegmentedColormap.from_list(
    "threshold_div",
    ["#08306B", "#4292C6", "#C6DBEF", "#F7F7F7", "#FDD0A2", "#F16913", "#7F2704"],
)
CMAP_DIVERGING.set_bad("white")

# Sequential map for the consensus failure row (0 = nobody fails).
CMAP_CONSENSUS = LinearSegmentedColormap.from_list(
    "consensus_fail",
    ["#FFF5F0", "#FCBBA1", "#FB6A4A", "#CB181D", "#67000D"],
)
CMAP_CONSENSUS.set_bad("white")

# Sequential map for numeric annotation tracks.
CMAP_NUMERIC = plt.get_cmap("YlGnBu").copy()
CMAP_NUMERIC.set_bad(GREY_MISSING)

# Colour-blind-safe qualitative palette (Okabe-Ito, reordered).
QUALITATIVE = [
    "#0072B2", "#E69F00", "#009E73", "#CC79A7",
    "#56B4E9", "#D55E00", "#F0E442", "#8C6BB1",
]

# Fixed palettes so a level always keeps the same colour across both figures.
FIXED_PALETTES: dict[str, dict[str, str]] = {
    "group":        {"hard": "#B2182B", "easy": "#2166AC"},
    "truth":        {"BRS1/2": "#08306B", "BRS3": "#7F2704"},   # = ends of CMAP_DIVERGING
    "event_status": {"Non-progressor": "#A6BDDB", "Progressed": "#B2182B"},
    "sex":          {"Male": "#0072B2", "Female": "#CC79A7"},
    "smoking":      {"No": "#BFD3E6", "Yes": "#4A1486"},
    "reTUR":        {"No": "#BFD3E6", "Yes": "#238B45"},
    "LVI":          {"No": "#BFD3E6", "Yes": "#D55E00"},
    "variant":      {"UCC": "#BFD3E6", "UCC + Variant": "#D55E00"},
    "stage":        {"TaHG": "#A1D99B", "T1HG": "#00441B"},
    "substage":     {"T1e": "#FDD0A2", "T1m": "#8C2D04"},
    "grade":        {"G2": "#C7E9C0", "G3": "#006D2C"},
    "EORTC":        {"Intermediate risk": "#C6DBEF",
                     "High risk": "#F16913",
                     "Highest risk": "#7F2704"},
    "BRS":          {"BRS1": "#C6DBEF", "BRS2": "#6BAED6", "BRS3": "#08519C"},
}

# Preferred level order (missing levels are appended alphabetically).
LEVEL_ORDER: dict[str, list[str]] = {
    "group": ["hard", "easy"],
    "truth": ["BRS1/2", "BRS3"],
    "event_status": ["Progressed", "Non-progressor"],
    "sex": ["Male", "Female"],
    "smoking": ["No", "Yes"],
    "reTUR": ["No", "Yes"],
    "LVI": ["No", "Yes"],
    "variant": ["UCC", "UCC + Variant"],
    "stage": ["TaHG", "T1HG"],
    "substage": ["T1e", "T1m"],
    "grade": ["G2", "G3"],
    "EORTC": ["Intermediate risk", "High risk", "Highest risk"],
    "BRS": ["BRS1", "BRS2", "BRS3"],
}

PRETTY = {
    "group": "Difficulty group",
    "truth": "Reference BRS class",
    "event_status": "Observed outcome",
    "age": "Age (years)",
    "sex": "Sex",
    "smoking": "Smoking",
    "tumor": "Tumour presentation",
    "stage": "Stage",
    "substage": "T1 substage",
    "grade": "Grade",
    "reTUR": "re-TUR",
    "LVI": "LVI",
    "variant": "Histology",
    "EORTC": "EORTC risk group",
    "no_instillations": "BCG instillations",
    "BRS": "BRS class",
}

MISSING_LABEL = "Missing"
NA_TOKENS = {"-1", "-1.0", "na", "n/a", "nan", "none", "unknown", "unk", "", "?"}


def pretty(feature: str) -> str:
    return PRETTY.get(feature, feature.replace("_", " ").capitalize())


# ----------------------------------------------------------------------
# Data helpers
# ----------------------------------------------------------------------
def clean_missing(series: pd.Series) -> pd.Series:
    """Collapse -1 / 'NA' / 'unknown' / blanks to a real NaN."""
    if pd.api.types.is_numeric_dtype(series):
        return series.mask(series < 0)
    cleaned = series.astype("object").where(series.notna())
    cleaned = cleaned.map(
        lambda v: np.nan
        if (v is None or (isinstance(v, float) and np.isnan(v))
            or str(v).strip().lower() in NA_TOKENS)
        else str(v).strip()
    )
    return cleaned


def parse_grand_challenge(csv_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(csv_path)
    parsed = []
    for _, row in raw.iterrows():
        results = json.loads(row["outputs"])[0]["value"]["results"]
        frame = pd.DataFrame(results)
        frame["participant"] = row["title"].split(" ", 1)[1]
        frame["rank"] = int(row["rank"])
        parsed.append(frame)
    return pd.concat(parsed, ignore_index=True)


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    ranked = p_values[order]
    n = len(ranked)
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty(n, dtype=float)
    result[order] = np.minimum(adjusted, 1.0)
    return result


METADATA_COLUMNS = {
    "case_id", "key", "group", "gt", "gt_time", "gt_event",
    "n_models", "n_wrong", "n_fail", "failure_fraction",
    "median_correct_conf", "mean_correct_conf", "min_correct_conf",
    "max_disc", "median_disc", "mean_disc", "min_pairs",
    "truth", "event_status", "criteria",
}


def compare_clinical_features(
    frame: pd.DataFrame,
    excluded: set[str] | None = None,
) -> pd.DataFrame:
    excluded = excluded or set()
    results = []

    for feature in frame.columns:
        if feature in METADATA_COLUMNS or feature in excluded:
            continue

        subset = frame.loc[frame["group"].isin(["hard", "easy"]),
                           ["group", feature]].copy()
        subset[feature] = clean_missing(subset[feature])
        subset = subset.dropna()

        if pd.api.types.is_numeric_dtype(frame[feature]):
            hard = subset.loc[subset["group"] == "hard", feature].to_numpy(float)
            easy = subset.loc[subset["group"] == "easy", feature].to_numpy(float)
            if len(hard) >= 2 and len(easy) >= 2 and len(np.unique(np.r_[hard, easy])) > 1:
                test = stats.mannwhitneyu(hard, easy, alternative="two-sided")
                p_value = float(test.pvalue)
                effect = 2 * float(test.statistic) / (len(hard) * len(easy)) - 1
            else:
                p_value = effect = np.nan
            results.append({"feature": feature, "type": "continuous",
                            "p_value": p_value, "effect_size": effect,
                            "n_hard": len(hard), "n_easy": len(easy)})
            continue

        table = pd.crosstab(subset["group"], subset[feature])
        if "hard" not in table.index or "easy" not in table.index:
            p_value = effect = np.nan
        else:
            table = table.loc[["hard", "easy"]]
            table = table.loc[:, table.sum(axis=0) > 0]
            if table.shape[1] < 2:
                p_value = effect = np.nan
            elif table.shape[1] == 2:
                odds_ratio, p_value = stats.fisher_exact(table.to_numpy())
                effect = (np.log2(odds_ratio)
                          if np.isfinite(odds_ratio) and odds_ratio > 0 else np.nan)
            else:
                try:
                    chi2, p_value, _, expected = stats.chi2_contingency(table.to_numpy())
                    effect = np.sqrt(chi2 / (table.to_numpy().sum()
                                             * min(table.shape[0] - 1, table.shape[1] - 1)))
                    if np.any(expected == 0):
                        p_value = np.nan
                except ValueError:
                    p_value = effect = np.nan

        results.append({"feature": feature, "type": "categorical",
                        "p_value": p_value, "effect_size": effect,
                        "n_hard": int((subset["group"] == "hard").sum()),
                        "n_easy": int((subset["group"] == "easy").sum())})

    result = pd.DataFrame(results)
    valid = result["p_value"].notna()
    result["q_value"] = np.nan
    result.loc[valid, "q_value"] = bh_adjust(result.loc[valid, "p_value"].to_numpy())
    return result.sort_values("p_value", na_position="last").reset_index(drop=True)


def level_palette(feature: str, levels: list[str]) -> dict[str, str]:
    fixed = FIXED_PALETTES.get(feature, {})
    palette, spare = {}, iter(QUALITATIVE)
    for level in levels:
        if level == MISSING_LABEL:
            palette[level] = GREY_MISSING
        elif level in fixed:
            palette[level] = fixed[level]
        else:
            palette[level] = next(spare)
    return palette


def ordered_levels(feature: str, values: pd.Series) -> list[str]:
    present = set(values.dropna().unique())
    preferred = [lv for lv in LEVEL_ORDER.get(feature, []) if lv in present]
    extra = sorted(present - set(preferred))
    levels = preferred + extra
    if values.isna().any():
        levels.append(MISSING_LABEL)
    return levels


def short_id(case_id: str) -> str:
    return pd.Series([case_id]).str.replace(r"^[23]", "", regex=True).iloc[0]


def _fmt_p(value: float) -> str:
    if pd.isna(value):
        return "n.a."
    return "< 0.001" if value < 0.001 else f"= {value:.3f}"


def stat_annotation(row: pd.Series) -> str:
    if pd.isna(row["p_value"]):
        return ""
    q = f"   q {_fmt_p(row['q_value'])}" if pd.notna(row["q_value"]) else ""
    return f"p {_fmt_p(row['p_value'])}{q}"


# ----------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------
GAP_COLS = 1.4          # width of the hard/easy gap, in cell units
CELL_W = 0.155          # inches per patient column
ROW_H = 0.30            # inches per matrix row


def _insert_gap(matrix: np.ndarray, split: int) -> np.ndarray:
    """Insert a NaN spacer column at `split` so hard/easy blocks separate."""
    pad = np.full((matrix.shape[0], 1), np.nan)
    return np.hstack([matrix[:, :split], pad, matrix[:, split:]])


def _column_centres(n: int, split: int) -> np.ndarray:
    left = np.arange(split) + 0.5
    right = np.arange(split, n) + 0.5 + GAP_COLS
    return np.concatenate([left, right])


def _edges(n: int, split: int) -> np.ndarray:
    """n + 2 edges: n patient cells plus one spacer cell at the split."""
    left = np.arange(split + 1, dtype=float)
    right = split + GAP_COLS + np.arange(n - split + 1, dtype=float)
    return np.concatenate([left, right])


def draw_matrix_figure(
    *,
    values: pd.DataFrame,
    patient_data: pd.DataFrame,
    clinical_features: list[str],
    clinical_stats: pd.DataFrame,
    continuous_label: str,
    continuous_ticklabels: tuple[str, str, str],
    failure_mask: pd.DataFrame,
    strict_mask: pd.Series,
    truth_feature: str,
    output_stem: str,
    title: str,
    subtitle: str,
) -> None:
    patients = patient_data["case_id"].tolist()
    models = values.index.tolist()
    n_patients = len(patients)
    split = int((patient_data["group"] == "hard").sum())
    n_easy = n_patients - split

    info = patient_data.set_index("case_id")
    track_features = [truth_feature] + clinical_features
    n_tracks = len(track_features)

    # ---------------- layout (all sizes in inches) ----------------
    left = 1.95
    stat_gutter = 1.55
    legend_w = 2.35
    top = 1.15
    bottom = 1.30
    main_w = (n_patients + GAP_COLS) * CELL_W

    h_header = 0.34
    h_star = 0.20
    h_models = len(models) * ROW_H
    h_cons = ROW_H
    h_tracks = n_tracks * ROW_H
    gap_a, gap_b, gap_c = 0.10, 0.16, 0.16

    body_h = h_header + gap_a + h_star + h_models + gap_b + h_cons + gap_c + h_tracks
    fig_w = left + main_w + stat_gutter + legend_w + 0.25
    fig_h = top + body_h + bottom

    fig = plt.figure(figsize=(fig_w, fig_h))

    def add_axes(y_top_in: float, height_in: float, x_in=left, w_in=main_w):
        """y measured downwards from the top of the figure."""
        return fig.add_axes([x_in / fig_w,
                             1 - (top + y_top_in + height_in) / fig_h,
                             w_in / fig_w,
                             height_in / fig_h])

    y = 0.0
    ax_header = add_axes(y, h_header); y += h_header + gap_a
    ax_star = add_axes(y, h_star); y += h_star
    ax_models = add_axes(y, h_models); y += h_models + gap_b
    ax_cons = add_axes(y, h_cons); y += h_cons + gap_c
    ax_tracks = add_axes(y, h_tracks)

    x_edges = _edges(n_patients, split)
    x_centres = _column_centres(n_patients, split)
    x_span = (0, n_patients + GAP_COLS)

    for ax in (ax_header, ax_star, ax_models, ax_cons, ax_tracks):
        ax.set_xlim(*x_span)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)
        ax.set_xticks([])
        ax.set_yticks([])

    # ---------------- group header ----------------
    ax_header.set_ylim(0, 1)
    ax_header.set_axis_off()
    for x0, width, label, colour in (
        (0, split, f"Hard  (n = {split})", FIXED_PALETTES["group"]["hard"]),
        (split + GAP_COLS, n_easy, f"Easy  (n = {n_easy})", FIXED_PALETTES["group"]["easy"]),
    ):
        ax_header.add_patch(Rectangle((x0, 0.05), width, 0.34,
                                      facecolor=colour, alpha=0.85, linewidth=0))
        ax_header.text(x0 + width / 2, 0.62, label, ha="center", va="center",
                       fontsize=10, fontweight="semibold", color=colour)

    # ---------------- all-model-failure strip ----------------
    ax_star.set_ylim(0, 1)
    ax_star.set_axis_off()
    strict = strict_mask.reindex(patients).fillna(False)
    for centre, patient in zip(x_centres, patients):
        if bool(strict.loc[patient]):
            ax_star.plot(centre, 0.45, marker="v", markersize=3.4,
                         color=INK, clip_on=False)

    # ---------------- model block ----------------
    matrix = _insert_gap(values.loc[models, patients].to_numpy(float), split)
    ax_models.pcolormesh(x_edges, np.arange(len(models) + 1),
                         np.ma.masked_invalid(matrix),
                         cmap=CMAP_DIVERGING, norm=Normalize(0, 1),
                         edgecolors="white", linewidth=0.35)
    ax_models.set_ylim(len(models), 0)
    ax_models.set_yticks(np.arange(len(models)) + 0.5)
    ax_models.set_yticklabels(models, fontsize=8.5)
    ax_models.yaxis.set_major_locator(FixedLocator(np.arange(len(models)) + 0.5))

    fails = failure_mask.loc[models, patients].to_numpy(bool)
    rows, cols = np.nonzero(fails)
    if len(rows):
        ax_models.scatter(x_centres[cols], rows + 0.5, s=3.6, marker="o",
                          facecolor=INK, edgecolor="white", linewidth=0.25,
                          alpha=0.9, zorder=3)

    # ---------------- consensus row ----------------
    consensus = info.loc[patients, "failure_fraction"].to_numpy(float)[None, :]
    ax_cons.pcolormesh(x_edges, np.arange(2),
                       np.ma.masked_invalid(_insert_gap(consensus, split)),
                       cmap=CMAP_CONSENSUS, norm=Normalize(0, 1),
                       edgecolors="white", linewidth=0.35)
    ax_cons.set_ylim(1, 0)
    ax_cons.set_yticks([0.5])
    ax_cons.set_yticklabels(["Fraction of models failing"], fontsize=8.5)

    # ---------------- annotation tracks ----------------
    ax_tracks.set_ylim(n_tracks, 0)
    legend_blocks: list[tuple[str, list[Patch]]] = []
    numeric_bars: list[tuple[str, Normalize]] = []
    track_labels = []

    for row_idx, feature in enumerate(track_features):
        raw = clean_missing(info.loc[patients, feature])
        numeric = (pd.api.types.is_numeric_dtype(patient_data[feature])
                   and feature not in FIXED_PALETTES)

        if numeric:
            data = pd.to_numeric(raw, errors="coerce").to_numpy(float)[None, :]
            finite = data[np.isfinite(data)]
            norm = Normalize(finite.min(), finite.max()) if finite.size else Normalize(0, 1)
            ax_tracks.pcolormesh(x_edges, [row_idx, row_idx + 1],
                                 np.ma.masked_invalid(_insert_gap(data, split)),
                                 cmap=CMAP_NUMERIC, norm=norm,
                                 edgecolors="white", linewidth=0.35)
            numeric_bars.append((feature, norm))
        else:
            values_str = raw.astype("object")
            levels = ordered_levels(feature, values_str)
            palette = level_palette(feature, levels)
            index = {lv: i for i, lv in enumerate(levels)}
            codes = np.array([index[MISSING_LABEL] if pd.isna(v) else index[str(v)]
                              for v in values_str], dtype=float)[None, :]
            cmap = ListedColormap([palette[lv] for lv in levels])
            ax_tracks.pcolormesh(x_edges, [row_idx, row_idx + 1],
                                 np.ma.masked_invalid(_insert_gap(codes, split)),
                                 cmap=cmap, norm=Normalize(-0.5, len(levels) - 0.5),
                                 edgecolors="white", linewidth=0.35)
            legend_blocks.append((
                pretty(feature),
                [Patch(facecolor=palette[lv], edgecolor="none", label=lv) for lv in levels],
            ))

        track_labels.append(pretty(feature))

        if feature in clinical_features:
            row = clinical_stats.loc[clinical_stats["feature"] == feature].iloc[0]
            text = stat_annotation(row)
            if text:
                ax_tracks.text(1.012, 1 - (row_idx + 0.5) / n_tracks, text,
                               transform=ax_tracks.transAxes, ha="left", va="center",
                               fontsize=7.5, color=MUTED, style="italic")

    ax_tracks.set_yticks(np.arange(n_tracks) + 0.5)
    ax_tracks.set_yticklabels(track_labels, fontsize=8.5)

    # patient labels
    ax_tracks.set_xticks(x_centres)
    ax_tracks.set_xticklabels([short_id(p) for p in patients],
                              rotation=90, fontsize=5.6, color=MUTED)
    ax_tracks.tick_params(axis="x", length=0, pad=2)
    ax_tracks.set_xlabel("Patients, ordered from hardest to easiest",
                         fontsize=9, labelpad=8)

    # ---------------- block outlines and hard/easy separator ----------------
    for ax, n_rows in ((ax_models, len(models)), (ax_cons, 1), (ax_tracks, n_tracks)):
        for x0, width in ((0, split), (split + GAP_COLS, n_easy)):
            if width:
                ax.add_patch(Rectangle((x0, 0), width, n_rows, fill=False,
                                       edgecolor="#A1A1AA", linewidth=0.55, zorder=5))
        ax.axvline(split + GAP_COLS / 2, color=RULE, linewidth=0.8,
                   linestyle=(0, (3, 2)), zorder=4)

    # ---------------- title ----------------
    fig.text(left / fig_w, 1 - 0.34 / fig_h, title,
             ha="left", va="top", fontsize=12.5, fontweight="semibold")
    fig.text(left / fig_w, 1 - 0.66 / fig_h, subtitle,
             ha="left", va="top", fontsize=8.8, color=MUTED)

    # ---------------- legend column ----------------
    lx = (left + main_w + stat_gutter) / fig_w
    cursor = top + 0.02        # inches from top of figure

    def put(text: str, size: float, weight: str = "normal", colour: str = INK,
            dy: float = 0.22) -> None:
        nonlocal cursor
        fig.text(lx, 1 - cursor / fig_h, text, ha="left", va="top",
                 fontsize=size, fontweight=weight, color=colour)
        cursor += dy

    # continuous colour bar
    put(continuous_label, 9, "semibold", dy=0.26)
    bar_h, bar_w = 0.15, 1.55
    cax = fig.add_axes([lx, 1 - (cursor + bar_h) / fig_h, bar_w / fig_w, bar_h / fig_h])
    cax.imshow(np.linspace(0, 1, 256)[None, :], aspect="auto",
               cmap=CMAP_DIVERGING, extent=(0, 1, 0, 1))
    cax.set_yticks([])
    cax.set_xticks([0, 0.5, 1])
    cax.set_xticklabels(continuous_ticklabels, fontsize=7)
    cax.tick_params(length=2, pad=1.5)
    for spine in cax.spines.values():
        spine.set_edgecolor(RULE)
        spine.set_linewidth(0.5)
    cursor += bar_h + 0.34

    fig.text(lx, 1 - cursor / fig_h, "0.5 = decision threshold",
             ha="left", va="top", fontsize=7, color=MUTED, style="italic")
    cursor += 0.30

    marker_ax = fig.add_axes([lx, 1 - (cursor + 0.30) / fig_h,
                              bar_w / fig_w, 0.30 / fig_h])
    marker_ax.set_axis_off()
    marker_ax.set_xlim(0, 1); marker_ax.set_ylim(0, 2)
    marker_ax.scatter([0.035], [1.45], s=3.6, facecolor=INK,
                      edgecolor="white", linewidth=0.25)
    marker_ax.text(0.11, 1.45, "model-level failure", va="center", fontsize=7.6)
    marker_ax.plot([0.035], [0.45], marker="v", markersize=3.4, color=INK)
    marker_ax.text(0.11, 0.45, "failed by every model", va="center", fontsize=7.6)
    cursor += 0.30 + 0.22

    put("Fraction of models failing", 8.6, "semibold", dy=0.22)
    ccax = fig.add_axes([lx, 1 - (cursor + 0.13) / fig_h, bar_w / fig_w, 0.13 / fig_h])
    ccax.imshow(np.linspace(0, 1, 256)[None, :], aspect="auto",
                cmap=CMAP_CONSENSUS, extent=(0, 1, 0, 1))
    ccax.set_yticks([])
    ccax.set_xticks([0, 0.5, 1])
    ccax.set_xticklabels(["0", "0.5", "1"], fontsize=7)
    ccax.tick_params(length=2, pad=1.5)
    for spine in ccax.spines.values():
        spine.set_edgecolor(RULE)
        spine.set_linewidth(0.5)
    cursor += 0.13 + 0.40

    for feature, norm in numeric_bars:
        put(pretty(feature), 8.6, "semibold", dy=0.22)
        nax = fig.add_axes([lx, 1 - (cursor + 0.13) / fig_h, bar_w / fig_w, 0.13 / fig_h])
        nax.imshow(np.linspace(0, 1, 256)[None, :], aspect="auto",
                   cmap=CMAP_NUMERIC, extent=(0, 1, 0, 1))
        nax.set_yticks([])
        nax.set_xticks([0, 1])
        nax.set_xticklabels([f"{norm.vmin:g}", f"{norm.vmax:g}"], fontsize=7)
        nax.tick_params(length=2, pad=1.5)
        for spine in nax.spines.values():
            spine.set_edgecolor(RULE)
            spine.set_linewidth(0.5)
        cursor += 0.13 + 0.34

    for label, handles in legend_blocks:
        put(label, 8.6, "semibold", dy=0.20)
        block_h = 0.185 * len(handles) + 0.05
        lax = fig.add_axes([lx, 1 - (cursor + block_h) / fig_h,
                            legend_w / fig_w, block_h / fig_h])
        lax.set_axis_off()
        lax.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.012, 1.06),
                   frameon=False, fontsize=7.6, handlelength=0.95,
                   handleheight=0.95, handletextpad=0.5,
                   labelspacing=0.32, borderpad=0)
        cursor += block_h + 0.14

    fig.savefig(OUT / f"{output_stem}.png", dpi=600)
    fig.savefig(OUT / f"{output_stem}.pdf")
    plt.close(fig)


# ----------------------------------------------------------------------
# Clinical table
# ----------------------------------------------------------------------
use_paper_style()

clinical = pd.read_csv(CLINICAL_CSV)
clinical["key"] = clinical["case_id"].str.replace(r"^[23]", "", regex=True)
clinical_base = clinical.drop(columns=["case_id", "criteria"])
for column in clinical_base.columns:
    if column != "key":
        clinical_base[column] = clean_missing(clinical_base[column])


# ----------------------------------------------------------------------
# TASK BRS
# ----------------------------------------------------------------------
task2 = parse_grand_challenge(TASK2_CSV)
task2 = task2.loc[task2["rank"] <= 8].copy()
task2["case_id"] = task2["case_id"].str.replace("_HE", "", regex=False)
task2["gt"] = task2["case_id_gt"].astype(int)
task2["prob"] = pd.to_numeric(task2["case_id_pred"])
task2["wrong"] = (task2["prob"] >= 0.5).astype(int) != task2["gt"]
task2["correct_conf"] = np.where(task2["gt"] == 1, task2["prob"], 1 - task2["prob"])

task2_summary = (
    task2.groupby(["case_id", "gt"])
    .agg(n_models=("participant", "nunique"),
         n_wrong=("wrong", "sum"),
         min_correct_conf=("correct_conf", "min"),
         median_correct_conf=("correct_conf", "median"),
         mean_correct_conf=("correct_conf", "mean"))
    .reset_index()
)
task2_summary["failure_fraction"] = task2_summary["n_wrong"] / task2_summary["n_models"]
task2_summary["group"] = np.select(
    [task2_summary["n_wrong"] >= 5,
     (task2_summary["n_wrong"] == 0) & (task2_summary["median_correct_conf"] >= 0.75)],
    ["hard", "easy"], default="other")
task2_summary["truth"] = np.where(task2_summary["gt"] == 1, "BRS3", "BRS1/2")
task2_summary["key"] = task2_summary["case_id"].str.replace(r"^[23]", "", regex=True)

task2_merged = task2_summary.merge(clinical_base, on="key", how="left")
task2_stats = compare_clinical_features(task2_merged, excluded={"BRS", "tumor"})
task2_stats.to_csv(OUT / "task2_easy_hard_clinical_tests.csv", index=False)

task2_features = task2_stats.loc[task2_stats["q_value"] < 0.05, "feature"].tolist()
if not task2_features:
    task2_features = task2_stats.loc[task2_stats["p_value"] < 0.05, "feature"].tolist()

selected = task2_merged.loc[task2_merged["group"].isin(["hard", "easy"])].copy()
task2_selected = pd.concat([
    selected.loc[selected["group"] == "hard"]
            .sort_values(["n_wrong", "median_correct_conf"], ascending=[False, True]),
    selected.loc[selected["group"] == "easy"]
            .sort_values("median_correct_conf", ascending=False),
], ignore_index=True)

task2_team_map = {1: "BioToTem", 2: "GRIS", 3: "BUAA-REMEX", 4: "CAGGEN-BIIT",
                  5: "WL", 6: "MITEL-UNIUD", 7: "HKKH", 8: "Aillis"}
task2["team"] = task2["rank"].map(task2_team_map)
task2_patients = task2_selected["case_id"].tolist()
teams2 = list(task2_team_map.values())

task2_values = (task2.pivot(index="team", columns="case_id", values="prob")
                .reindex(index=teams2, columns=task2_patients))
task2_failures = (task2.pivot(index="team", columns="case_id", values="wrong")
                  .reindex(index=teams2, columns=task2_patients).fillna(False).astype(bool))

draw_matrix_figure(
    values=task2_values,
    patient_data=task2_selected,
    clinical_features=task2_features,
    clinical_stats=task2_stats,
    continuous_label="Predicted P(BRS3)",
    continuous_ticklabels=("0\nBRS1/2", "0.5", "1\nBRS3"),
    failure_mask=task2_failures,
    strict_mask=task2_summary.set_index("case_id")["n_wrong"].eq(8),
    truth_feature="truth",
    output_stem="task2_brs_difficulty",
    title="Task BRS — per-patient model predictions and consensus difficulty",
    subtitle=("Hard: \u22655/8 models incorrect.   "
              "Easy: all models correct and median correct-class probability \u22650.75.   "
              "Clinical tracks: Fisher / \u03c7\u00b2 or Mann\u2013Whitney, Benjamini\u2013Hochberg q."),
)


# ----------------------------------------------------------------------
# TASK PROGRESSION
# ----------------------------------------------------------------------
task3 = parse_grand_challenge(TASK3_CSV).rename(columns={
    "case_id_gt_time": "gt_time",
    "case_id_gt_event": "gt_event",
    "case_id_prediction_years_to_recurrence": "pred_time",
})
task3["case_id"] = task3["case_id"].str.replace("_HE", "", regex=False)

truth = task3[["case_id", "gt_time", "gt_event"]].drop_duplicates()
records = truth.to_dict("records")
pairs = pd.DataFrame(
    [(a["case_id"], b["case_id"])
     for a in records if int(a["gt_event"]) == 1
     for b in records
     if a["case_id"] != b["case_id"] and float(a["gt_time"]) < float(b["gt_time"])],
    columns=["early_case", "other_case"])

pred_wide = task3.pivot(index="case_id", columns="participant", values="pred_time")
rows = []
for participant in pred_wide.columns:
    predictions = pred_wide[participant]
    current = pairs.copy()
    current["pred_early"] = current["early_case"].map(predictions)
    current["pred_other"] = current["other_case"].map(predictions)
    current = current.dropna()
    current["wrong"] = current["pred_early"] > current["pred_other"]

    early = current.groupby("early_case")["wrong"].agg(["size", "sum"])
    other = current.groupby("other_case")["wrong"].agg(["size", "sum"])
    for case_id in truth["case_id"]:
        n_pairs = ((early.loc[case_id, "size"] if case_id in early.index else 0)
                   + (other.loc[case_id, "size"] if case_id in other.index else 0))
        n_wrong = ((early.loc[case_id, "sum"] if case_id in early.index else 0)
                   + (other.loc[case_id, "sum"] if case_id in other.index else 0))
        rows.append({"case_id": case_id, "participant": participant,
                     "discordance": n_wrong / n_pairs if n_pairs else np.nan,
                     "n_pairs": n_pairs})

task3_patient_model = pd.DataFrame(rows)
task3_patient_model["failed"] = task3_patient_model["discordance"] > 0.5

task3_summary = (
    task3_patient_model.groupby("case_id")
    .agg(n_fail=("failed", "sum"), max_disc=("discordance", "max"),
         median_disc=("discordance", "median"), mean_disc=("discordance", "mean"),
         min_pairs=("n_pairs", "min"))
    .reset_index()
    .merge(truth, on="case_id", how="left")
)
n_models3 = task3_patient_model["participant"].nunique()
task3_summary["failure_fraction"] = task3_summary["n_fail"] / n_models3
task3_summary["group"] = np.select(
    [task3_summary["n_fail"] >= 3,
     (task3_summary["n_fail"] == 0) & (task3_summary["mean_disc"] <= 0.25)],
    ["hard", "easy"], default="other")
task3_summary["event_status"] = np.where(task3_summary["gt_event"] == 1,
                                         "Progressed", "Non-progressor")
task3_summary["key"] = task3_summary["case_id"].str.replace(r"^[23]", "", regex=True)

task3_merged = task3_summary.merge(clinical_base, on="key", how="left")
task3_stats = compare_clinical_features(task3_merged, excluded={"tumor"})
task3_stats.to_csv(OUT / "task3_easy_hard_clinical_tests.csv", index=False)
task3_features = task3_stats.loc[task3_stats["p_value"] < 0.05, "feature"].tolist()

selected3 = task3_merged.loc[task3_merged["group"].isin(["hard", "easy"])].copy()
task3_selected = pd.concat([
    selected3.loc[selected3["group"] == "hard"]
             .sort_values(["n_fail", "median_disc"], ascending=[False, False]),
    selected3.loc[selected3["group"] == "easy"]
             .sort_values("mean_disc", ascending=True),
], ignore_index=True)

task3_team_map = {1: "SMILE", 2: "HKKH", 3: "NMIL", 4: "WL", 5: "TIA-Pegasus"}
participant_rank = (task3[["participant", "rank"]].drop_duplicates()
                    .set_index("participant")["rank"])
task3_patient_model["team"] = task3_patient_model["participant"].map(
    participant_rank.map(task3_team_map))
task3_patients = task3_selected["case_id"].tolist()
teams3 = list(task3_team_map.values())

task3_values = (task3_patient_model.pivot(index="team", columns="case_id",
                                          values="discordance")
                .reindex(index=teams3, columns=task3_patients))
task3_failures = (task3_patient_model.pivot(index="team", columns="case_id",
                                            values="failed")
                  .reindex(index=teams3, columns=task3_patients)
                  .fillna(False).astype(bool))

draw_matrix_figure(
    values=task3_values,
    patient_data=task3_selected,
    clinical_features=task3_features,
    clinical_stats=task3_stats,
    continuous_label="Patient-level discordance rate",
    continuous_ticklabels=("0\nconcordant", "0.5", "1\ndiscordant"),
    failure_mask=task3_failures,
    strict_mask=task3_summary.set_index("case_id")["n_fail"].eq(n_models3),
    truth_feature="event_status",
    output_stem="task3_progression_difficulty",
    title="Task Progression — per-patient ranking discordance and consensus difficulty",
    subtitle=("Hard: \u22653/5 models fail (discordance >0.5).   "
              "Easy: no model fails and mean discordance \u22640.25.   "
              "Clinical tracks: Fisher / \u03c7\u00b2 or Mann\u2013Whitney, Benjamini\u2013Hochberg q."),
)


# ----------------------------------------------------------------------
# OVERLAP OF HARD CASES BETWEEN THE TWO TASKS
# ----------------------------------------------------------------------
overlap = (
    task2_summary[["key", "case_id", "group", "n_wrong", "median_correct_conf", "truth"]]
    .merge(task3_summary[["key", "case_id", "group", "n_fail", "mean_disc", "event_status"]],
           on="key", suffixes=("_brs", "_prog"))
)
group_order = ["hard", "other", "easy"]
difficulty_crosstab = (pd.crosstab(overlap["group_brs"], overlap["group_prog"])
                       .reindex(index=group_order, columns=group_order, fill_value=0))
difficulty_crosstab.index.name = "BRS \\ Progression"
difficulty_crosstab.to_csv(OUT / "difficulty_group_crosstab.csv")

hard_brs = overlap["group_brs"] == "hard"
hard_prog = overlap["group_prog"] == "hard"
contingency = [[int((hard_brs & hard_prog).sum()), int((hard_brs & ~hard_prog).sum())],
               [int((~hard_brs & hard_prog).sum()), int((~hard_brs & ~hard_prog).sum())]]
overlap_or, overlap_p = stats.fisher_exact(contingency, alternative="greater")
expected_overlap = hard_brs.sum() * hard_prog.sum() / len(overlap)

both_hard = (
    overlap.loc[hard_brs & hard_prog]
    .merge(clinical_base, on="key", how="left")
    .assign(severity=lambda d: d["n_wrong"] / 8 + d["n_fail"] / n_models3)
    .sort_values(["severity", "median_correct_conf"], ascending=[False, True])
    .drop(columns="severity")
    .reset_index(drop=True)
)
both_hard.to_csv(OUT / "hard_in_both_tasks.csv", index=False)

print(f"\nPatients scored in both tasks: {len(overlap)}")
print(f"Hard in BRS: {int(hard_brs.sum())}   hard in Progression: {int(hard_prog.sum())}   "
      f"hard in both: {contingency[0][0]}  (expected by chance {expected_overlap:.1f}; "
      f"one-sided Fisher p = {overlap_p:.3f}, OR = {overlap_or:.2f})")
print(difficulty_crosstab, "\n")
print(both_hard[["key", "truth", "n_wrong", "median_correct_conf",
                 "event_status", "n_fail", "mean_disc",
                 "substage", "variant", "stage", "EORTC", "age"]].to_string(index=False))


def draw_overlap_figure(
    *,
    keys: list[str],
    brs_values: pd.DataFrame, brs_fail: pd.DataFrame,
    prog_values: pd.DataFrame, prog_fail: pd.DataFrame,
    patient_info: pd.DataFrame,
    track_features: list[str],
    output_stem: str,
    title: str,
    subtitle: str,
) -> None:
    """Compact side-by-side view of patients that are hard in both tasks."""
    n = len(keys)
    info = patient_info.set_index("key")
    cell_w, row_h = 0.42, 0.30
    left, gutter, legend_w, top, bottom = 1.95, 0.35, 2.35, 1.05, 0.95
    main_w = n * cell_w
    blocks = [("Task BRS — predicted P(BRS3)", brs_values, brs_fail, CMAP_DIVERGING),
              ("Task Progression — discordance rate", prog_values, prog_fail, CMAP_DIVERGING)]
    h_caption, gap = 0.26, 0.18
    body_h = sum(h_caption + len(v.index) * row_h + gap for _, v, _, _ in blocks) \
        + h_caption + len(track_features) * row_h
    fig_w = left + main_w + gutter + legend_w + 0.25
    fig_h = top + body_h + bottom
    fig = plt.figure(figsize=(fig_w, fig_h))
    edges = np.arange(n + 1, dtype=float)
    centres = edges[:-1] + 0.5

    def add_axes(y_top: float, h: float):
        return fig.add_axes([left / fig_w, 1 - (top + y_top + h) / fig_h,
                             main_w / fig_w, h / fig_h])

    def tidy(ax):
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)
        ax.set_xticks([])
        ax.set_xlim(0, n)

    y = 0.0
    for caption, vals, fails, cmap in blocks:
        fig.text(left / fig_w, 1 - (top + y) / fig_h, caption, ha="left", va="top",
                 fontsize=8.6, fontweight="semibold")
        y += h_caption
        rows_ = vals.index.tolist()
        ax = add_axes(y, len(rows_) * row_h)
        tidy(ax)
        ax.pcolormesh(edges, np.arange(len(rows_) + 1),
                      np.ma.masked_invalid(vals.loc[rows_, keys].to_numpy(float)),
                      cmap=cmap, norm=Normalize(0, 1), edgecolors="white", linewidth=0.5)
        ax.set_ylim(len(rows_), 0)
        ax.set_yticks(np.arange(len(rows_)) + 0.5)
        ax.set_yticklabels(rows_, fontsize=8.5)
        r, c = np.nonzero(fails.loc[rows_, keys].to_numpy(bool))
        if len(r):
            ax.scatter(centres[c], r + 0.5, s=6, facecolor=INK, edgecolor="white",
                       linewidth=0.3, zorder=3)
        ax.add_patch(Rectangle((0, 0), n, len(rows_), fill=False,
                               edgecolor="#A1A1AA", linewidth=0.55, zorder=5))
        y += len(rows_) * row_h + gap

    fig.text(left / fig_w, 1 - (top + y) / fig_h, "Clinical annotation", ha="left",
             va="top", fontsize=8.6, fontweight="semibold")
    y += h_caption
    ax = add_axes(y, len(track_features) * row_h)
    tidy(ax)
    ax.set_ylim(len(track_features), 0)
    legend_blocks, numeric_bars = [], []
    for i, feature in enumerate(track_features):
        raw = clean_missing(info.loc[keys, feature])
        if pd.api.types.is_numeric_dtype(patient_info[feature]) and feature not in FIXED_PALETTES:
            data = pd.to_numeric(raw, errors="coerce").to_numpy(float)[None, :]
            finite = data[np.isfinite(data)]
            norm = Normalize(finite.min(), finite.max()) if finite.size else Normalize(0, 1)
            ax.pcolormesh(edges, [i, i + 1], np.ma.masked_invalid(data), cmap=CMAP_NUMERIC,
                          norm=norm, edgecolors="white", linewidth=0.5)
            for centre, v in zip(centres, data[0]):
                if np.isfinite(v):
                    ax.text(centre, i + 0.5, f"{v:g}", ha="center", va="center", fontsize=6.4,
                            color="white" if norm(v) > 0.55 else INK)
            numeric_bars.append((feature, norm))
        else:
            values_str = raw.astype("object")
            levels = ordered_levels(feature, values_str)
            palette = level_palette(feature, levels)
            index = {lv: k for k, lv in enumerate(levels)}
            codes = np.array([index[MISSING_LABEL] if pd.isna(v) else index[str(v)]
                              for v in values_str], dtype=float)[None, :]
            ax.pcolormesh(edges, [i, i + 1], np.ma.masked_invalid(codes),
                          cmap=ListedColormap([palette[lv] for lv in levels]),
                          norm=Normalize(-0.5, len(levels) - 0.5),
                          edgecolors="white", linewidth=0.5)
            legend_blocks.append((pretty(feature),
                                  [Patch(facecolor=palette[lv], edgecolor="none", label=lv)
                                   for lv in levels]))
    ax.add_patch(Rectangle((0, 0), n, len(track_features), fill=False,
                           edgecolor="#A1A1AA", linewidth=0.55, zorder=5))
    ax.set_yticks(np.arange(len(track_features)) + 0.5)
    ax.set_yticklabels([pretty(f) for f in track_features], fontsize=8.5)
    ax.set_xticks(centres)
    ax.set_xticklabels(keys, rotation=90, fontsize=7, color=MUTED)
    ax.tick_params(axis="x", length=0, pad=2)

    fig.text(left / fig_w, 1 - 0.30 / fig_h, title, ha="left", va="top",
             fontsize=12, fontweight="semibold")
    fig.text(left / fig_w, 1 - 0.60 / fig_h, subtitle, ha="left", va="top",
             fontsize=8.4, color=MUTED)

    # legend column
    lx = (left + main_w + gutter) / fig_w
    cursor = top + 0.02
    bar_w = 1.55

    def put(text, size, weight="normal", dy=0.22):
        nonlocal cursor
        fig.text(lx, 1 - cursor / fig_h, text, ha="left", va="top",
                 fontsize=size, fontweight=weight)
        cursor += dy

    def bar(cmap, ticks, labels, h=0.13):
        nonlocal cursor
        cax = fig.add_axes([lx, 1 - (cursor + h) / fig_h, bar_w / fig_w, h / fig_h])
        cax.imshow(np.linspace(0, 1, 256)[None, :], aspect="auto", cmap=cmap,
                   extent=(0, 1, 0, 1))
        cax.set_yticks([]); cax.set_xticks(ticks); cax.set_xticklabels(labels, fontsize=7)
        cax.tick_params(length=2, pad=1.5)
        for spine in cax.spines.values():
            spine.set_edgecolor(RULE); spine.set_linewidth(0.5)
        cursor += h + 0.36

    put("Predicted P(BRS3)  /  discordance rate", 8.6, "semibold", dy=0.24)
    bar(CMAP_DIVERGING, [0, 0.5, 1], ["0", "0.5", "1"])
    fig.text(lx, 1 - (cursor - 0.14) / fig_h, "\u25cf  model-level failure",
             ha="left", va="top", fontsize=7.6)
    cursor += 0.28
    for feature, norm in numeric_bars:
        put(pretty(feature), 8.6, "semibold")
        bar(CMAP_NUMERIC, [0, 1], [f"{norm.vmin:g}", f"{norm.vmax:g}"])
    for label, handles in legend_blocks:
        put(label, 8.6, "semibold", dy=0.20)
        block_h = 0.185 * len(handles) + 0.05
        lax = fig.add_axes([lx, 1 - (cursor + block_h) / fig_h, legend_w / fig_w, block_h / fig_h])
        lax.set_axis_off()
        lax.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.012, 1.06),
                   frameon=False, fontsize=7.6, handlelength=0.95, handleheight=0.95,
                   handletextpad=0.5, labelspacing=0.32, borderpad=0)
        cursor += block_h + 0.14

    fig.savefig(OUT / f"{output_stem}.png", dpi=600)
    fig.savefig(OUT / f"{output_stem}.pdf")
    plt.close(fig)


if len(both_hard):
    overlap_keys = both_hard["key"].tolist()
    key_of = lambda ids: [pd.Series([i]).str.replace(r"^[23]", "", regex=True).iloc[0] for i in ids]

    brs_vals = task2.pivot(index="team", columns="case_id", values="prob").reindex(index=teams2)
    brs_vals.columns = key_of(brs_vals.columns)
    brs_fail = task2.pivot(index="team", columns="case_id", values="wrong").reindex(index=teams2)
    brs_fail.columns = key_of(brs_fail.columns)
    prog_vals = (task3_patient_model.pivot(index="team", columns="case_id", values="discordance")
                 .reindex(index=teams3))
    prog_vals.columns = key_of(prog_vals.columns)
    prog_fail = (task3_patient_model.pivot(index="team", columns="case_id", values="failed")
                 .reindex(index=teams3))
    prog_fail.columns = key_of(prog_fail.columns)

    draw_overlap_figure(
        keys=overlap_keys,
        brs_values=brs_vals[overlap_keys],
        brs_fail=brs_fail[overlap_keys].fillna(False).astype(bool),
        prog_values=prog_vals[overlap_keys],
        prog_fail=prog_fail[overlap_keys].fillna(False).astype(bool),
        patient_info=both_hard,
        track_features=["truth", "event_status", "substage", "variant", "EORTC",
                        "smoking", "age"],
        output_stem="hard_in_both_tasks",
        title=f"Patients that are hard in both tasks (n = {len(overlap_keys)})",
        subtitle=(f"{int(hard_brs.sum())} hard in BRS, {int(hard_prog.sum())} hard in Progression, "
                  f"{len(overlap)} patients scored in both.\n"
                  f"Expected overlap by chance {expected_overlap:.1f}; "
                  f"Fisher one-sided p = {overlap_p:.3f}, OR = {overlap_or:.2f}."),
    )

print("\ndone")
for name in ["task2_brs_difficulty.png", "task3_progression_difficulty.png",
             "hard_in_both_tasks.png", "hard_in_both_tasks.csv",
             "difficulty_group_crosstab.csv"]:
    print(" ", (OUT / name).resolve())
