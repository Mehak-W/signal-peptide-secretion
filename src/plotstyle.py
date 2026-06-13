"""Shared Tufte-style matplotlib settings for all figures.

Embodies Tufte's principles (*The Visual Display of
Quantitative Information*): maximize the data-ink ratio, erase non-data ink and
redundant ink, no chartjunk (gridlines, heavy frames, decorative color), and let
the caption carry context rather than titles/labels on the plot.

Usage in a plotting script:
    from src.plotstyle import apply_tufte, tufte_ax, GRAY, PALETTE
    apply_tufte()                 # set global rcParams (call once)
    fig, ax = plt.subplots(...)
    ...
    tufte_ax(ax)                  # despine + clean each axis before saving
"""
import matplotlib as mpl

# Muted, print-friendly palette (use gray by default; color only to encode).
GRAY = "#4d4d4d"
LIGHT_GRAY = "#b0b0b0"
PALETTE = ["#4d4d4d", "#4878CF", "#D65F5F", "#6ACC65", "#B47CC7", "#C4AD66"]


def apply_tufte():
    """Set global rcParams for a clean, single-column, Tufte-style look."""
    mpl.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "axes.linewidth": 0.6,
        "axes.edgecolor": GRAY,
        "axes.labelcolor": "black",
        "axes.titlepad": 4.0,
        "axes.spines.top": False,      # remove non-data frame ink
        "axes.spines.right": False,
        "axes.grid": False,            # no gridline chartjunk
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
        "xtick.color": GRAY,
        "ytick.color": GRAY,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,       # no legend box
        "legend.fontsize": 8,
        "lines.linewidth": 1.3,
        "patch.linewidth": 0.0,        # bars: no edge ink
        "figure.figsize": (3.4, 2.6),  # single-column default
    })


def tufte_ax(ax):
    """Final per-axis cleanup: ensure top/right spines gone, ticks outward,
    no grid. Returns the axis for chaining."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRAY)
        ax.spines[side].set_linewidth(0.6)
    ax.grid(False)
    ax.tick_params(direction="out", length=3, width=0.6, colors=GRAY)
    return ax
