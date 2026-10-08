"""
plotting.py  -  Shared figure style for the whole project.
All notebooks and scripts should call apply_style() once at the top.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl


# ── colour palette ──────────────────────────────────────────────────────────
COLORS = {
    "Wind":   "#4C9BE8",   # sky blue
    "Solar":  "#F5A623",   # amber
    "Mixed":  "#7ED321",   # green
    "neutral": "#8E8E93",
}

SOURCE_PALETTE = [COLORS["Wind"], COLORS["Solar"], COLORS["Mixed"]]


def apply_style():
    """Call once per script/notebook to apply the project-wide style."""
    plt.style.use("seaborn-v0_8-darkgrid")
    mpl.rcParams.update({
        # figure
        "figure.dpi":        120,
        "figure.facecolor":  "white",
        "axes.facecolor":    "#F8F9FA",
        # fonts
        "font.family":       "sans-serif",
        "font.size":         11,
        "axes.titlesize":    13,
        "axes.titleweight":  "bold",
        "axes.labelsize":    11,
        # lines
        "lines.linewidth":   1.8,
        "lines.markersize":  5,
        # grid
        "grid.alpha":        0.4,
        "grid.linestyle":    "--",
        # legend
        "legend.framealpha": 0.85,
        "legend.fontsize":   10,
        # ticks
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
    })


def savefig(path: str, **kwargs):
    """Wrapper around plt.savefig with consistent defaults."""
    plt.savefig(path, bbox_inches="tight", dpi=150, **kwargs)
    print(f"Saved -> {path}")
