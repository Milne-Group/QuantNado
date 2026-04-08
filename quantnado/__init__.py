"""QuantNado package initialization."""

from quantnado.api import QuantNado, metadata_from_seqnado
from quantnado.analysis.plot import correlate, heatmap, locus_plot, metaplot, tornadoplot
from quantnado.analysis.normalise import get_library_sizes, get_mean_read_lengths, normalise
from quantnado.analysis.pca import plot_pca_scree, plot_pca_scatter, run_pca as pca
from quantnado.dataset.utils import AnchorPoint, FeatureType, ReductionMethod
from quantnado.dataset.store_coverage import BamStore
from quantnado.dataset.store_methyl import MethylStore
from quantnado.dataset.store_variants import VariantStore
from quantnado.dataset.store_multiomics import MultiomicsStore

open_dataset = QuantNado.open_dataset
create_dataset = QuantNado.create_dataset
open = QuantNado.open_dataset

__all__ = [
    "QuantNado",
    "BamStore",
    "MethylStore",
    "VariantStore",
    "MultiomicsStore",
    "AnchorPoint",
    "FeatureType",
    "ReductionMethod",
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
    "get_mean_read_lengths",
    "pca",
]
