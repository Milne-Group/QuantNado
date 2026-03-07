from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    import matplotlib.pyplot as plt

__all__ = ["metaplot", "tornadoplot"]

# Sensible per-modality defaults that metaplot / tornadoplot apply when
# ``modality`` is provided.  Explicit kwargs passed by the caller always win.
_MODALITY_DEFAULTS: "dict[str, dict]" = {
    "coverage": {
        "ylabel": "Coverage (CPM)",
        "cmap": "RdYlBu_r",
    },
    "methylation": {
        "ylabel": "Methylation (%)",
        "cmap": "RdPu",
        "vmin": 0.0,
        "vmax": 100.0,
    },
    "variant": {
        "ylabel": "Allele frequency",
        "cmap": "OrRd",
        "vmin": 0.0,
        "vmax": 1.0,
    },
}


def _resolve_palette(palette, n: int, labels: "list | None" = None) -> list:
    """Return a list of n colors from a palette name, list, dict, or None (default cycle)."""
    import matplotlib.pyplot as plt

    if palette is None:
        prop_cycle = plt.rcParams["axes.prop_cycle"]
        cycle = [c["color"] for c in prop_cycle]
        return [cycle[i % len(cycle)] for i in range(n)]
    if isinstance(palette, dict):
        # Map label → color; fall back to default cycle for missing labels
        prop_cycle = plt.rcParams["axes.prop_cycle"]
        cycle = [c["color"] for c in prop_cycle]
        return [palette.get(lab, cycle[i % len(cycle)]) for i, lab in enumerate(labels or [])] if labels else list(palette.values())[:n]
    if isinstance(palette, list):
        return [palette[i % len(palette)] for i in range(n)]
    # string: matplotlib colormap name
    cmap = plt.get_cmap(palette)
    return [cmap(i / max(n - 1, 1)) for i in range(n)]


def metaplot(
    data: xr.DataArray,
    *,
    modality: "str | None" = None,
    samples: list[str] | None = None,
    groups: "dict[str, list[str]] | None" = None,
    flip_minus_strand: bool = True,
    error_stat: "str | None" = "sem",
    palette: "str | list | dict | None" = None,
    reference_point: "float | None" = 0,
    reference_label: str = "TSS",
    xlabel: str = "Relative position",
    ylabel: "str | None" = None,
    title: str = "Metagene profile",
    figsize: "tuple[float, float]" = (8, 4),
    ax: "plt.Axes | None" = None,
    filepath: "str | Path | None" = None,
) -> "plt.Axes":
    """
    Plot a metagene profile from the output of ``qn.extract()``.

    Averages signal across all intervals to produce a per-sample (or per-group)
    line plot over relative position, optionally with a confidence band.

    ``qn.extract()`` correctly anchors extraction at the strand-aware TSS
    (5' end) when ``anchor="start"``, but the signal arrays are always stored
    in genomic (left-to-right) order. Minus-strand features therefore run
    3'→5', so their profiles must be reversed before averaging. This is done
    automatically when ``flip_minus_strand=True`` and a ``strand`` coordinate
    is present on the ``interval`` dimension.

    Parameters
    ----------
    data : DataArray
        Output of ``qn.extract()``, with dimensions
        ``(interval, relative_position, sample)`` or ``(interval, bin, sample)``.
        May be dask-backed; ``.compute()`` is called automatically if needed.
    modality : {"coverage", "methylation", "variant"}, optional
        Data modality. Sets sensible defaults for ``ylabel`` and visual style.
        Explicit kwargs always override modality defaults.
    samples : list of str, optional
        Subset of samples to plot. Defaults to all samples. Ignored when
        ``groups`` is provided.
    groups : dict of {str: list[str]}, optional
        Sample grouping for averaging. Keys are group labels, values are lists
        of sample names. Each group is plotted as one line (mean across samples
        and intervals). The error band reflects variability between samples
        within the group. Example::

            groups={"control": ["s1", "s2"], "treated": ["s3", "s4"]}

    flip_minus_strand : bool, default True
        If True and a ``strand`` coordinate exists on the ``interval`` dimension,
        reverse the position axis for minus-strand intervals before averaging.
        Set to False if ``qn.extract()`` was called with ``anchor="midpoint"``
        (not strand-aware) or if you have already handled strand orientation.
    error_stat : {"sem", "std", None}, default "sem"
        Error band to draw around each line.

        - ``"sem"`` — standard error of the mean across intervals (or across
          samples when ``groups`` is provided); matches deeptools default.
        - ``"std"`` — standard deviation.
        - ``None`` — no error band.

    reference_point : float or None, default 0
        X-axis value at which to draw a vertical reference line.
        Set to None to omit the line.
    reference_label : str, default "TSS"
        Label for the reference line in the legend.
    xlabel : str, default "Relative position"
        X-axis label.
    ylabel : str, default "Mean signal"
        Y-axis label.
    title : str, default "Metagene profile"
        Plot title.
    ax : matplotlib Axes, optional
        Axes to draw on. Creates a new figure if None.
    filepath : str or Path, optional
        If provided, save the figure to this path.

    Returns
    -------
    ax : matplotlib.axes.Axes

    Examples
    --------
    >>> binned = qn.extract(
    ...     feature_type="promoter",
    ...     gtf_path="genes.gtf",
    ...     fixed_width=2000,
    ...     anchor="start",
    ...     bin_size=50,
    ... )
    >>> ax = metaplot(binned, title="Promoter metagene")
    >>> ax = metaplot(
    ...     binned,
    ...     groups={"control": ["s1", "s2"], "treated": ["s3", "s4"]},
    ...     error_stat="sem",
    ... )
    """
    import matplotlib.pyplot as plt

    # Apply modality defaults (explicit kwargs take priority)
    if modality is not None:
        if modality not in _MODALITY_DEFAULTS:
            raise ValueError(f"modality must be one of {list(_MODALITY_DEFAULTS)}, got {modality!r}")
        defaults = _MODALITY_DEFAULTS[modality]
        if ylabel is None:
            ylabel = defaults.get("ylabel", "Mean signal")
    if ylabel is None:
        ylabel = "Mean signal"

    # Accept either "relative_position" (no binning) or "bin" (binned output)
    position_dim = next(
        (d for d in data.dims if d in ("relative_position", "bin")), None
    )
    if position_dim is None or "interval" not in data.dims or "sample" not in data.dims:
        raise ValueError(
            f"data must have dims (interval, relative_position|bin, sample), got {data.dims}"
        )
    if data.dims != ("interval", position_dim, "sample"):
        data = data.transpose("interval", position_dim, "sample")

    if hasattr(data, "chunks"):
        data = data.compute()

    # Flip minus-strand intervals so all profiles run 5'→3'
    if flip_minus_strand and "strand" in data.coords:
        strands = data.coords["strand"].values
        minus_mask = strands == "-"
        if minus_mask.any():
            arr = data.values.copy()  # (interval, position, sample)
            arr[minus_mask] = arr[minus_mask, ::-1, :]
            data = xr.DataArray(arr, dims=data.dims, coords=data.coords)

    all_sample_labels = list(data.coords["sample"].values)
    x = data.coords[position_dim].values

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    if groups is not None:
        # One line per group: mean across (intervals × samples in group).
        # Error band = SEM across per-sample means within the group.
        colors = _resolve_palette(palette, len(groups), labels=list(groups.keys()))
        for (group_label, group_samples), color in zip(groups.items(), colors):
            missing = set(group_samples) - set(all_sample_labels)
            if missing:
                raise ValueError(
                    f"Samples not found in data for group '{group_label}': {missing}"
                )
            # per_sample_means: (n_samples, n_positions)
            per_sample_means = np.stack(
                [data.sel(sample=s).values.mean(axis=0) for s in group_samples], axis=0
            )
            group_mean = per_sample_means.mean(axis=0)
            (line,) = ax.plot(x, group_mean, alpha=0.9, label=group_label, color=color)
            if error_stat is not None and len(group_samples) > 1:
                group_std = per_sample_means.std(axis=0)
                err = group_std / np.sqrt(len(group_samples)) if error_stat == "sem" else group_std
                ax.fill_between(
                    x, group_mean - err, group_mean + err,
                    alpha=0.2, color=line.get_color()
                )
    else:
        # One line per sample: mean across intervals.
        # Error band = SEM (or STD) across intervals.
        if samples is not None:
            missing = set(samples) - set(all_sample_labels)
            if missing:
                raise ValueError(f"Samples not found in data: {missing}")
            sample_labels = samples
        else:
            sample_labels = all_sample_labels

        colors = _resolve_palette(palette, len(sample_labels), labels=sample_labels)
        n_intervals = data.sizes["interval"]
        mean_profile = data.mean(dim="interval")  # (position, sample)
        std_profile = data.std(dim="interval")    # (position, sample)

        for sample, color in zip(sample_labels, colors):
            y = mean_profile.sel(sample=sample).values
            (line,) = ax.plot(x, y, alpha=0.9, label=sample, color=color)
            if error_stat is not None:
                err = std_profile.sel(sample=sample).values
                if error_stat == "sem":
                    err = err / np.sqrt(n_intervals)
                ax.fill_between(
                    x, y - err, y + err,
                    alpha=0.15, color=line.get_color()
                )

    if reference_point is not None:
        ax.axvline(
            reference_point,
            color="black",
            linestyle="--",
            linewidth=0.8,
            label=reference_label,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    ax.figure.tight_layout()

    if filepath is not None:
        ax.figure.savefig(filepath, bbox_inches="tight")

    return ax


def _prep_extract(data: xr.DataArray, flip_minus_strand: bool) -> "tuple[xr.DataArray, np.ndarray, str]":
    """Shared pre-processing: validate dims, compute, strand-flip. Returns (data, x, position_dim)."""
    position_dim = next(
        (d for d in data.dims if d in ("relative_position", "bin")), None
    )
    if position_dim is None or "interval" not in data.dims or "sample" not in data.dims:
        raise ValueError(
            f"data must have dims (interval, relative_position|bin, sample), got {data.dims}"
        )
    if data.dims != ("interval", position_dim, "sample"):
        data = data.transpose("interval", position_dim, "sample")
    if hasattr(data, "chunks"):
        data = data.compute()
    if flip_minus_strand and "strand" in data.coords:
        strands = data.coords["strand"].values
        minus_mask = strands == "-"
        if minus_mask.any():
            arr = data.values.copy()
            arr[minus_mask] = arr[minus_mask, ::-1, :]
            data = xr.DataArray(arr, dims=data.dims, coords=data.coords)
    x = data.coords[position_dim].values
    return data, x, position_dim


def tornadoplot(
    data: xr.DataArray,
    *,
    modality: "str | None" = None,
    samples: "list[str] | None" = None,
    sample_names: "list[str] | None" = None,
    groups: "dict[str, list[str]] | None" = None,
    flip_minus_strand: bool = True,
    sort_by: "str | None" = "mean",
    vmin: "float | None" = None,
    vmax: "float | None" = None,
    cmap: "str | None" = None,
    reference_point: "float | None" = 0,
    reference_label: str = "TSS",
    xlabel: str = "Relative position",
    ylabel: "str | None" = None,
    title: str = "Signal heatmap",
    figsize: "tuple[float, float] | None" = None,
    filepath: "str | Path | None" = None,
) -> "list":
    """
    Plot a tornado / heatmap from the output of ``qn.extract()``.

    Each row is one genomic interval; colour encodes signal intensity.
    One panel is drawn per sample (or per group when ``groups`` is provided).
    Panels share the same colour scale and row order.

    Parameters
    ----------
    data : DataArray
        Output of ``qn.extract()``, dimensions
        ``(interval, relative_position|bin, sample)``.
    modality : {"coverage", "methylation", "variant"}, optional
        Data modality. Sets sensible defaults for ``cmap``, ``vmin``, and ``vmax``.
        Explicit kwargs always override modality defaults.
    samples : list of str, optional
        Subset of samples to plot (one panel each). Ignored when ``groups`` is set.
    sample_names : list of str, optional
        Display names (aliases) for samples, in the same order as ``samples``.
        If provided, must have the same length as ``samples``. Ignored when
        ``groups`` is set.
    groups : dict {label: [samples]}, optional
        Average samples within each group before plotting (one panel per group).
    flip_minus_strand : bool, default True
        Reverse minus-strand intervals before plotting.
    sort_by : {"mean", "max", None}, default "mean"
        Sort intervals (rows) by their mean or max signal across all positions,
        descending. Sorting is determined by the first panel and applied to all.
        ``None`` keeps the original order.
    vmin, vmax : float, optional
        Colour scale limits. Defaults to 0 and the 99th percentile across all panels.
    cmap : str, default "RdYlBu_r"
        Matplotlib colormap name.
    reference_point : float or None, default 0
        X position of the vertical reference line. ``None`` omits it.
    reference_label : str, default "TSS"
        Label for the reference line in the colorbar area.
    xlabel : str, default "Relative position"
        X-axis label (shown on the bottom panel only).
    ylabel : str, optional
        Y-axis label. Defaults to ``Intervals (n=<count>)``.
    title : str, default "Signal heatmap"
        Figure suptitle.
    figsize : tuple, optional
        Figure size. Defaults to ``(3 * n_panels + 0.5, 5)``.
    filepath : str or Path, optional
        Save figure to this path if provided.

    Returns
    -------
    axes : list of matplotlib.axes.Axes
        One Axes per panel.
    """
    import matplotlib.pyplot as plt

    # Apply modality defaults (explicit kwargs take priority)
    if modality is not None:
        if modality not in _MODALITY_DEFAULTS:
            raise ValueError(f"modality must be one of {list(_MODALITY_DEFAULTS)}, got {modality!r}")
        defaults = _MODALITY_DEFAULTS[modality]
        if vmin is None:
            vmin = defaults.get("vmin")
        if vmax is None:
            vmax = defaults.get("vmax")
        if cmap is None:
            cmap = defaults.get("cmap", "RdYlBu_r")
    if cmap is None:
        cmap = "RdYlBu_r"

    data, x, _ = _prep_extract(data, flip_minus_strand)
    all_sample_labels = list(data.coords["sample"].values)

    # Build list of (panel_label, matrix) where matrix is (n_intervals, n_positions)
    panels: "list[tuple[str, np.ndarray]]" = []
    if groups is not None:
        for group_label, group_samples in groups.items():
            missing = set(group_samples) - set(all_sample_labels)
            if missing:
                raise ValueError(f"Samples not found for group '{group_label}': {missing}")
            mat = np.stack(
                [data.sel(sample=s).values for s in group_samples], axis=0
            ).mean(axis=0)  # (n_intervals, n_positions)
            panels.append((group_label, mat))
    else:
        label_list = samples if samples is not None else all_sample_labels
        if samples is not None:
            missing = set(samples) - set(all_sample_labels)
            if missing:
                raise ValueError(f"Samples not found: {missing}")

        # Use aliases if provided
        if sample_names is not None:
            if len(sample_names) != len(label_list):
                raise ValueError(
                    f"sample_names length ({len(sample_names)}) must match "
                    f"samples length ({len(label_list)})"
                )
            display_labels = sample_names
        else:
            display_labels = label_list

        for s, label in zip(label_list, display_labels):
            panels.append((label, data.sel(sample=s).values))

    # Sort rows by first panel's signal
    if sort_by is not None and panels:
        first_mat = panels[0][1]
        if sort_by == "mean":
            order = np.argsort(np.nanmean(first_mat, axis=1))[::-1]
        elif sort_by == "max":
            order = np.argsort(np.nanmax(first_mat, axis=1))[::-1]
        else:
            raise ValueError(f"sort_by must be 'mean', 'max', or None; got {sort_by!r}")
        panels = [(lbl, mat[order]) for lbl, mat in panels]

    # Colour scale
    if vmin is None:
        vmin = 0.0
    if vmax is None:
        all_vals = np.concatenate([mat.ravel() for _, mat in panels])
        vmax = float(np.nanpercentile(all_vals, 99))

    n_panels = len(panels)
    if figsize is None:
        figsize = (3 * n_panels + 0.8, 5)

    fig, axes = plt.subplots(
        1, n_panels, figsize=figsize, sharey=True,
        gridspec_kw={"wspace": 0.05}
    )
    if n_panels == 1:
        axes = [axes]

    n_intervals = panels[0][1].shape[0]
    extent = [float(x[0]), float(x[-1]), n_intervals, 0]

    for ax, (label, mat) in zip(axes, panels):
        im = ax.imshow(
            mat,
            aspect="auto",
            origin="upper",
            extent=extent,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            interpolation="nearest",
        )
        ax.set_title(label, fontsize=9)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.tick_params(axis="y", left=False, labelleft=False)
        if reference_point is not None:
            ax.axvline(reference_point, color="white", linestyle="--", linewidth=0.8)

    axes[0].tick_params(axis="y", left=True, labelleft=True)
    ylabel = f"Intervals (n={n_intervals})" if ylabel is None else ylabel
    axes[0].set_ylabel(ylabel, fontsize=8)

    # Shared colorbar to the right
    cbar = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.02)
    cbar.set_label("Signal", fontsize=8)
    if reference_point is not None:
        axes[-1].annotate(
            reference_label,
            xy=(reference_point, 0),
            xycoords=("data", "axes fraction"),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
            color="white",
        )

    fig.suptitle(title, fontsize=11, y=1.01)

    if filepath is not None:
        fig.savefig(filepath, bbox_inches="tight")

    return axes
