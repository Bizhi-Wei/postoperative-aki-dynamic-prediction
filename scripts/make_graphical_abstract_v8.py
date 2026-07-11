from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "_v8_figure_stage"


def box(ax, x, y, w, h, title, lines, face, edge):
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.1, facecolor=face, edgecolor=edge,
    )
    ax.add_patch(patch)
    ax.text(x + 0.025 * w, y + h * 0.72, title, fontsize=10, fontweight="bold", color="#202431")
    ax.text(x + 0.025 * w, y + h * 0.46, lines, fontsize=7.3, color="#445066", va="top", linespacing=1.3)


def main():
    # Figure contract: show the dynamic risk-set design, internally validated
    # discrimination, and the bounded creatinine-sensitivity conclusion without
    # implying causality, external validity, or demonstrated clinical benefit.
    OUT.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(9.2, 3.0), dpi=100)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.text(0.02, 0.93, "Dynamic postoperative AKI prediction in surgical intensive care",
            fontsize=14, fontweight="bold", color="#17233C", va="top")
    ax.text(0.02, 0.835, "Retrospective MIMIC-IV cohort | serum creatinine KDIGO outcome | patient-grouped internal validation",
            fontsize=7.5, color="#5C6980", va="top")

    box(ax, 0.02, 0.23, 0.25, 0.47, "Incident-AKI cohort",
        "10,877 evaluable admissions\n4,531 events within 7 days (41.7%)\nPrevalent AKI excluded",
        "#EAF1FF", "#5276C8")
    box(ax, 0.375, 0.23, 0.25, 0.47, "Dynamic landmarks",
        "0 h: n=10,877 | AUROC 0.728\n6 h: n=10,624 | AUROC 0.740\n24 h: n=9,301 | AUROC 0.754",
        "#EDF8F0", "#4A8B57")
    box(ax, 0.73, 0.23, 0.25, 0.47, "Sensitivity findings",
        "24-h no-creatinine AUROC: 0.729\nPre-index baseline restriction: stable\nExternal validation remains required",
        "#FFF4E9", "#C8753F")
    for x1, x2 in [(0.27, 0.375), (0.625, 0.73)]:
        ax.add_patch(FancyArrowPatch((x1 + 0.012, 0.465), (x2 - 0.012, 0.465),
                                     arrowstyle="-|>", mutation_scale=14,
                                     linewidth=1.2, color="#6C778C"))
    ax.text(0.02, 0.08,
            "Moderate internal discrimination was maintained across clinically defined risk sets; 24-h performance partly depended on creatinine trajectories.",
            fontsize=8.1, color="#202431", fontweight="bold")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(OUT / "graphical_abstract.png", dpi=100, bbox_inches=None, pad_inches=0)
    fig.savefig(OUT / "graphical_abstract.svg", bbox_inches=None, pad_inches=0)
    fig.savefig(OUT / "graphical_abstract.pdf", bbox_inches=None, pad_inches=0)
    plt.close(fig)


if __name__ == "__main__":
    main()
