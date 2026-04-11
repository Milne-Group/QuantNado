"""Peak overlap analysis and Venn diagram visualisation.

Typical workflow::

    # Call peaks for each assay
    atac_beds  = qn.subset(assay="ATAC").call_peaks("peaks/atac/")
    chip_beds  = qn.subset(assay="ChIP").call_peaks("peaks/chip/")
    cat_beds   = qn.subset(assay="CUT&TAG").call_peaks("peaks/cat/")

    # Build a labelled dict (one BED per set) for overlap analysis
    peak_sets = {
        "ATAC":    atac_beds["ATAC-SEM-1"],
        "ChIP":    chip_beds["CM-RCHACV-1_H3K27ac"],
        "CUT&TAG": cat_beds["CAT-HSC_H3K27ac"],
    }

    counts = qn.peak_overlap(peak_sets)
    ax     = qn.venn_peaks(peak_sets, title="H3K27ac peak overlaps")
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pyranges1 as pr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from loguru import logger


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_peaks(path: str | Path) -> pr.PyRanges:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Peak BED file not found: {path}")
    gr = pr.read_bed(str(path))
    return gr if len(gr) > 0 else pr.PyRanges()


def _safe_overlap(a: pr.PyRanges, b: pr.PyRanges) -> pr.PyRanges:
    """Intervals from `a` that overlap at least one interval in `b`."""
    if len(a) == 0 or len(b) == 0:
        return pr.PyRanges()
    return a.overlap(b, strand_behavior="ignore")


def _safe_subtract(a: pr.PyRanges, b: pr.PyRanges) -> pr.PyRanges:
    """Remove from `a` any interval that has ANY overlap with `b` (bedtools intersect -v).

    pyranges1 ``subtract_overlaps`` clips intervals rather than removing them,
    so we explicitly drop any row from ``a`` that appears in ``a.overlap(b)``.
    """
    if len(a) == 0 or len(b) == 0:
        return a
    overlapping = a.overlap(b, strand_behavior="ignore")
    if len(overlapping) == 0:
        return a

    a_df = pd.DataFrame(a)
    ov_df = pd.DataFrame(overlapping)
    ov_keys = set(zip(ov_df["Chromosome"], ov_df["Start"], ov_df["End"]))
    keys = list(zip(a_df["Chromosome"], a_df["Start"], a_df["End"]))
    kept = a_df[[k not in ov_keys for k in keys]]
    return pr.PyRanges(kept) if len(kept) > 0 else pr.PyRanges()


def _venn_counts(
    merged: list[pr.PyRanges],
    labels: list[str],
) -> dict[frozenset[str], int]:
    """Count peaks per exclusive Venn region.

    For each region (combination of labels), counts peaks from the first set
    in that combination that overlap all others in the combination and don't
    overlap any set outside it.  This is the standard genomics convention.
    """
    n = len(merged)
    counts: dict[frozenset[str], int] = {}

    for r in range(1, n + 1):
        for combo in combinations(range(n), r):
            in_idx = set(combo)
            out_idx = set(range(n)) - in_idx

            result = merged[combo[0]]
            for j in combo[1:]:
                result = _safe_overlap(result, merged[j])
            for j in out_idx:
                result = _safe_subtract(result, merged[j])

            key = frozenset(labels[i] for i in in_idx)
            counts[key] = len(result)

    return counts


# ── public API ───────────────────────────────────────────────────────────────

def overlap_peaks(peak_sets: dict[str, str | Path]) -> pd.DataFrame:
    """Compute peak overlap statistics across 2–4 sets.

    Parameters
    ----------
    peak_sets : dict[str, str | Path]
        Mapping of label → BED file path.

    Returns
    -------
    pd.DataFrame
        One row per non-empty combination with columns:

        ``combination``
            Tuple of set labels in the region.
        ``n_peaks``
            Number of merged peaks in that exclusive region (counted from the
            first set in the combination).
        ``pct_of_first``
            ``n_peaks`` as a percentage of the total peaks in the first set of
            the combination.
    """
    labels = list(peak_sets.keys())
    n = len(labels)
    if not (2 <= n <= 4):
        raise ValueError(f"overlap_peaks requires 2–4 sets, got {n}")

    prs = [_load_peaks(p) for p in peak_sets.values()]
    merged = [gr.merge_overlaps() if len(gr) > 0 else gr for gr in prs]
    totals = {lbl: len(m) for lbl, m in zip(labels, merged)}

    counts = _venn_counts(merged, labels)

    rows = []
    for combo_set, n_peaks in counts.items():
        combo = tuple(sorted(combo_set))
        first = combo[0]
        rows.append({
            "combination": combo,
            "n_peaks": n_peaks,
            "pct_of_first": round(100 * n_peaks / totals[first], 1) if totals[first] else 0.0,
        })

    df = (
        pd.DataFrame(rows)
        .sort_values("combination", key=lambda col: col.map(lambda x: (len(x), x)))
        .reset_index(drop=True)
    )
    return df


# ── Venn diagram ─────────────────────────────────────────────────────────────

# Circle centres and text positions for n=2 and n=3
_V2_CENTRES = [(-0.35, 0.0), (0.35, 0.0)]
_V3_CENTRES = [(-0.40, 0.23), (0.40, 0.23), (0.00, -0.35)]
_RADIUS = 0.65

# Approximate centres of each exclusive region (used for count labels)
_V2_TEXT = {
    frozenset(["A"]):      (-0.62, 0.00),
    frozenset(["B"]):      ( 0.62, 0.00),
    frozenset(["A", "B"]): ( 0.00, 0.00),
}
_V3_TEXT = {
    frozenset(["A"]):           (-0.72,  0.42),
    frozenset(["B"]):           ( 0.72,  0.42),
    frozenset(["C"]):           ( 0.00, -0.72),
    frozenset(["A", "B"]):      ( 0.00,  0.48),
    frozenset(["A", "C"]):      (-0.50, -0.22),
    frozenset(["B", "C"]):      ( 0.50, -0.22),
    frozenset(["A", "B", "C"]): ( 0.00,  0.08),
}
# Set label positions (just outside each circle)
_V2_SET_LABELS = [(-0.88, 0.72), (0.88, 0.72)]
_V3_SET_LABELS = [(-0.88, 0.75), (0.88, 0.75), (0.00, -0.95)]

_DEFAULT_COLORS = ["#4472C4", "#ED7D31", "#70AD47", "#FFC000"]


def venn_plot(
    peak_sets: dict[str, str | Path],
    ax: plt.Axes | None = None,
    title: str | None = None,
    colors: list[str] | None = None,
    alpha: float = 0.50,
    figsize: tuple[float, float] = (5.5, 5.5),
    count_fontsize: float = 12,
    label_fontsize: float = 13,
) -> plt.Axes:
    """Venn diagram for 2 or 3 peak sets.

    Parameters
    ----------
    peak_sets : dict[str, str | Path]
        Label → BED file path.  Must have exactly 2 or 3 entries.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into.  A new figure is created if omitted.
    title : str, optional
        Plot title.
    colors : list of str, optional
        Fill colour for each circle (default: blue / orange / green).
    alpha : float
        Circle opacity (default 0.50).
    figsize : tuple
        Figure size when ``ax`` is None.
    count_fontsize : float
        Font size for region count labels.
    label_fontsize : float
        Font size for set name labels.

    Returns
    -------
    matplotlib.axes.Axes
    """
    n = len(peak_sets)
    if n not in (2, 3):
        raise ValueError(f"venn_plot supports exactly 2 or 3 peak sets, got {n}")

    real_labels = list(peak_sets.keys())
    prs = [_load_peaks(p) for p in peak_sets.values()]
    merged = [gr.merge_overlaps() if len(gr) > 0 else gr for gr in prs]

    # Map real labels → single-char placeholders for lookup, then substitute back
    placeholders = ["A", "B", "C"][:n]
    ph_to_real = dict(zip(placeholders, real_labels))

    raw_counts = _venn_counts(merged, placeholders)
    # Remap to real labels
    counts: dict[frozenset[str], int] = {
        frozenset(ph_to_real[p] for p in k): v for k, v in raw_counts.items()
    }

    if colors is None:
        colors = _DEFAULT_COLORS[:n]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    centres = _V2_CENTRES if n == 2 else _V3_CENTRES
    text_pos = _V2_TEXT if n == 2 else _V3_TEXT
    set_label_pos = _V2_SET_LABELS if n == 2 else _V3_SET_LABELS

    # Draw circles
    for i, (cx, cy) in enumerate(centres):
        circle = mpatches.Circle(
            (cx, cy), _RADIUS,
            facecolor=colors[i], alpha=alpha,
            edgecolor="white", linewidth=2,
            zorder=2,
        )
        ax.add_patch(circle)

    # Set name labels (bold, outside circles)
    for i, (lbl, (tx, ty)) in enumerate(zip(real_labels, set_label_pos)):
        ax.text(
            tx, ty, lbl,
            ha="center", va="center",
            fontsize=label_fontsize, fontweight="bold",
            color=colors[i], zorder=5,
        )

    # Region count labels
    ph_text = _V2_TEXT if n == 2 else _V3_TEXT
    for ph_key, (tx, ty) in ph_text.items():
        real_key = frozenset(ph_to_real[p] for p in ph_key)
        n_peaks = counts.get(real_key, 0)
        ax.text(
            tx, ty, str(n_peaks),
            ha="center", va="center",
            fontsize=count_fontsize, fontweight="bold",
            color="black", zorder=6,
        )

    if title:
        ax.set_title(title, fontsize=label_fontsize + 1, pad=8)

    margin = _RADIUS + 0.5
    ax.set_xlim(-margin, margin)
    ax.set_ylim(-margin, margin)
    ax.set_aspect("equal")
    ax.axis("off")
    return ax
