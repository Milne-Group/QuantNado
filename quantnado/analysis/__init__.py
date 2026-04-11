from .core import QuantNadoDataset
from .counts import count_features
from .features import (
    annotate_intervals,
    extract_feature_ranges,
    extract_promoters,
    load_gtf,
)
from .normalise import get_library_sizes, get_mean_read_lengths, normalise
from .ranges import (
    default_position_mask,
    get_fixed_windows,
    masked_array_fromranges,
    merge_ranges,
    ranges_loader,
)
from .reduce import extract_byranges_signal, reduce_byranges_signal

__all__ = [
    "QuantNadoDataset",
    "count_features",
    "annotate_intervals",
    "extract_feature_ranges",
    "extract_promoters",
    "load_gtf",
    "plot_pca_scatter",
    "plot_pca_scree",
    "run_pca",
    "correlate",
    "heatmap",
    "metaplot",
    "tornadoplot",
    "default_position_mask",
    "get_fixed_windows",
    "masked_array_fromranges",
    "merge_ranges",
    "ranges_loader",
    "get_library_sizes",
    "get_mean_read_lengths",
    "normalise",
    "extract_byranges_signal",
    "reduce_byranges_signal",
]


def __getattr__(name):
    if name in {"plot_pca_scatter", "plot_pca_scree", "run_pca"}:
        from . import pca as _pca
        return getattr(_pca, name)
    if name in {"correlate", "heatmap", "metaplot", "tornadoplot"}:
        from . import plot as _plot
        return getattr(_plot, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
