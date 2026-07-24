"""Create the v30 cohort and analysis flow diagram with Python/matplotlib.

Figure contract
---------------
Core conclusion: the paper contains two prespecified MIMIC-IV analysis paths
(severe-AKI landmark prediction and post-AKI trajectories) and one explicitly
selected eICU external-validation path.
Evidence chain: panel a traces all internal denominators; panel b exposes the
external outcome-observability denominator before landmark risk sets.
Archetype: schematic-led two-panel composite.
Export: 183-mm double-column vector PDF/SVG plus 600-dpi TIFF and PNG preview.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "manuscript_figure_v30_secondary_flow"

BLUE = "#3F6C8E"
BLUE_LIGHT = "#EAF1F6"
ORANGE = "#B86832"
ORANGE_LIGHT = "#F8EEE6"
TEAL = "#3F7F78"
TEAL_LIGHT = "#E9F3F1"
INK = "#23303A"
MUTED = "#5B6770"
LINE = "#9AA8B2"


def style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
        }
    )


def box(ax, x, y, w, h, title, detail, face, edge, *, fs=7.0, z=3):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        linewidth=0.9, edgecolor=edge, facecolor=face, zorder=z,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * 0.61, title, ha="center", va="center",
            fontsize=fs, fontweight="bold", color=INK, zorder=z + 1)
    ax.text(x + w / 2, y + h * 0.28, detail, ha="center", va="center",
            fontsize=fs - 0.5, color=MUTED, zorder=z + 1)


def arrow(ax, start, end, *, color=LINE, lw=1.0):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=8,
                                 linewidth=lw, color=color, shrinkA=2, shrinkB=2, zorder=2))


def elbow_arrow(ax, points, *, color=LINE, lw=1.0):
    """Draw an orthogonal connector whose final segment carries the arrowhead."""
    for start, end in zip(points[:-2], points[1:-1]):
        ax.plot([start[0], end[0]], [start[1], end[1]], color=color, lw=lw, zorder=2)
    arrow(ax, points[-2], points[-1], color=color, lw=lw)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 5.05), gridspec_kw={"width_ratios": [1.52, 1]})
    ax, bx = axes
    for panel in axes:
        panel.set_xlim(0, 1); panel.set_ylim(0, 1); panel.axis("off")

    ax.text(0.0, 1.01, "a", fontsize=9, fontweight="bold", va="bottom")
    ax.text(0.06, 1.01, "MIMIC-IV development and internal validation", fontsize=8.2,
            fontweight="bold", color=INK, va="bottom")
    box(ax, 0.11, 0.86, 0.78, 0.105, "Strict postoperative ICU cohort",
        "10,877 outcome-evaluable admissions", BLUE_LIGHT, BLUE, fs=7.4)

    ax.text(0.04, 0.79, "Severe-AKI landmark prediction", color=BLUE,
            fontweight="bold", fontsize=7.2)
    y = 0.66
    xs = [0.03, 0.35, 0.67]
    titles = ["0 h risk set", "6 h risk set", "24 h risk set"]
    details = ["n=10,877 | 679 events", "n=10,856 | 658 events", "n=10,736 | 538 events"]
    for x, title, detail in zip(xs, titles, details):
        box(ax, x, y, 0.28, 0.105, title, detail, BLUE_LIGHT, BLUE, fs=6.8)
    arrow(ax, (0.31, y + 0.052), (0.35, y + 0.052), color=BLUE)
    arrow(ax, (0.63, y + 0.052), (0.67, y + 0.052), color=BLUE)
    ax.text(0.335, y - 0.024, "21 already severe", ha="center", fontsize=5.6, color=MUTED)
    ax.text(0.655, y - 0.024, "+120 already severe", ha="center", fontsize=5.6, color=MUTED)

    ax.text(0.04, 0.56, "Post-AKI trajectory analysis", color=ORANGE,
            fontweight="bold", fontsize=7.2)
    trajectory = [
        (0.04, 0.425, "Incident SCr-AKI", "4,531 admissions"),
        (0.53, 0.425, "Trajectory eligible", "4,519 admissions"),
        (0.04, 0.245, "Observed recovery", "3,936 admissions"),
        (0.53, 0.245, "Recurrent AKI", "641 after recovery"),
    ]
    for x, y0, title, detail in trajectory:
        box(ax, x, y0, 0.42, 0.105, title, detail, ORANGE_LIGHT, ORANGE, fs=6.9)
    arrow(ax, (0.46, 0.477), (0.53, 0.477), color=ORANGE)
    ax.text(0.495, 0.410, "Excluded: 12 post-disposition onsets", ha="center", va="top",
            fontsize=5.3, color=MUTED,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.7})
    elbow_arrow(ax, [(0.74, 0.425), (0.74, 0.385), (0.25, 0.385), (0.25, 0.35)], color=ORANGE)
    arrow(ax, (0.46, 0.297), (0.53, 0.297), color=ORANGE)
    ax.text(0.50, 0.105, "Competing events: live discharge and in-hospital death\nFollow-up truncated at ICU day 7",
            ha="center", va="center", fontsize=6.0, color=MUTED,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "#F6F7F8", "edgecolor": "#C9D0D5"})

    bx.text(0.0, 1.01, "b", fontsize=9, fontweight="bold", va="bottom")
    bx.text(0.09, 1.01, "eICU external validation", fontsize=8.2,
            fontweight="bold", color=INK, va="bottom")
    box(bx, 0.08, 0.85, 0.84, 0.11, "Strict surgical first ICU stays",
        "30,365 stays across 197 hospitals", TEAL_LIGHT, TEAL, fs=7.2)
    arrow(bx, (0.50, 0.85), (0.50, 0.735), color=TEAL)
    box(bx, 0.08, 0.62, 0.84, 0.115, "SCr outcome evaluable",
        "14,229 (46.9%); not evaluable 16,136", TEAL_LIGHT, TEAL, fs=7.1)
    bx.text(0.50, 0.575, "Frozen feature mapping and models", ha="center",
            fontsize=6.0, color=MUTED)
    arrow(bx, (0.50, 0.62), (0.50, 0.52), color=TEAL)
    ext = [
        (0.08, 0.405, "0 h", "n=14,229 | 910 events"),
        (0.08, 0.255, "6 h", "n=14,155 | 836 events"),
        (0.08, 0.105, "24 h", "n=13,857 | 538 events"),
    ]
    for x, y0, title, detail in ext:
        box(bx, x, y0, 0.84, 0.10, f"{title} external risk set", detail,
            TEAL_LIGHT, TEAL, fs=6.9)
    arrow(bx, (0.50, 0.52), (0.50, 0.505), color=TEAL)
    arrow(bx, (0.50, 0.405), (0.50, 0.355), color=TEAL)
    arrow(bx, (0.50, 0.255), (0.50, 0.205), color=TEAL)

    fig.subplots_adjust(left=0.025, right=0.99, top=0.94, bottom=0.03, wspace=0.08)
    stem = OUT / "figure_v30_cohort_analysis_flow"
    fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, facecolor="white")
    plt.close(fig)

    rows = [
        ("MIMIC-IV", "Strict postoperative ICU cohort", 10877, "Outcome-evaluable admissions"),
        ("MIMIC-IV", "Severe-AKI 0 h risk set", 10877, "679 events"),
        ("MIMIC-IV", "Severe-AKI 6 h risk set", 10856, "658 events; 21 already severe"),
        ("MIMIC-IV", "Severe-AKI 24 h risk set", 10736, "538 events; 141 cumulatively already severe"),
        ("MIMIC-IV", "Incident SCr-AKI", 4531, "Any incident SCr-AKI"),
        ("MIMIC-IV", "Trajectory eligible", 4519, "12 post-disposition onsets excluded"),
        ("MIMIC-IV", "Observed recovery", 3936, "After AKI onset"),
        ("MIMIC-IV", "Recurrent AKI", 641, "After observed recovery"),
        ("eICU", "Strict surgical first ICU stays", 30365, "197 hospitals"),
        ("eICU", "SCr outcome evaluable", 14229, "46.9%; 16,136 not evaluable"),
        ("eICU", "Severe-AKI 0 h risk set", 14229, "910 events"),
        ("eICU", "Severe-AKI 6 h risk set", 14155, "836 events"),
        ("eICU", "Severe-AKI 24 h risk set", 13857, "538 events"),
    ]
    pd.DataFrame(rows, columns=["database", "node", "n", "detail"]).to_csv(
        OUT / "figure_v30_flow_source_data.csv", index=False
    )
    (OUT / "audit_v30_figure_qa.md").write_text(
        "# Figure QA\n\n"
        "- Core conclusion: internal prediction, trajectory, and external validation denominators are explicit.\n"
        "- Backend: Python/matplotlib only.\n"
        "- Final size: 7.2 x 5.05 inches (double-column).\n"
        "- Exports: editable PDF/SVG, 600-dpi TIFF, 300-dpi PNG.\n"
        "- External observability denominator: 14,229/30,365 (46.9%).\n"
        "- Source data: figure_v30_flow_source_data.csv.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
