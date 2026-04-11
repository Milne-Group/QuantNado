"""QuantNado package initialization."""

from quantnado.api import QuantNado, create_dataset, metadata_from_seqnado
from quantnado.analysis.core import QuantNadoDataset
from quantnado.analysis.normalise import get_library_sizes, normalise
from quantnado.dataset.metadata import AnchorPoint, FeatureType, ReductionMethod, array_key

open_dataset = QuantNado.open
open = QuantNado.open

__all__ = [
    "QuantNado",
    "QuantNadoDataset",
    "BamStore",
    "MethylStore",
    "VariantStore",
    "AnchorPoint",
    "FeatureType",
    "ReductionMethod",
    "array_key",
    "metaplot",
    "tornadoplot",
    "locus_plot",
    "heatmap",
    "correlate",
    "create_dataset",
    "open_dataset",
    "open",
    "metadata_from_seqnado",
    "plot_pca_scree",
    "plot_pca_scatter",
    "normalise",
    "get_library_sizes",
    "pca",
]


def __getattr__(name):
    if name == "BamStore":
        from quantnado.dataset.store_bam import BamStore
        return BamStore
    if name == "MethylStore":
        from quantnado.dataset.store_methyl import MethylStore
        return MethylStore
    if name == "VariantStore":
        from quantnado.dataset.store_variants import VariantStore
        return VariantStore
    if name in {"correlate", "heatmap", "locus_plot", "metaplot", "tornadoplot"}:
        from quantnado.analysis import plot as _plot
        return getattr(_plot, name)
    if name in {"plot_pca_scree", "plot_pca_scatter"}:
        from quantnado.analysis import pca as _pca
        return getattr(_pca, name)
    if name == "pca":
        from quantnado.analysis.pca import run_pca
        return run_pca
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
