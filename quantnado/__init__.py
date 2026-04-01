"""QuantNado package initialization."""

from quantnado.api import QuantNado, correlate, heatmap, locus_plot, metaplot, metadata_from_seqnado, tornadoplot
from quantnado.analysis.normalise import get_library_sizes, get_mean_read_lengths, normalise
from quantnado.analysis.pca import plot_pca_scree, plot_pca_scatter, run_pca as pca
from quantnado.dataset.enums import AnchorPoint, FeatureType, ReductionMethod
from quantnado.dataset.store_bam import BamStore
from quantnado.dataset.store_methyl import MethylStore
from quantnado.dataset.store_variants import VariantStore
from quantnado.dataset.store_multiomics import MultiomicsStore

create_dataset = QuantNado.create_dataset
open_dataset = QuantNado.open_dataset

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
    "create_dataset",
    "open_dataset",
    "metadata_from_seqnado",
    "plot_pca_scree",
    "plot_pca_scatter",
    "normalise",
    "get_library_sizes",
    "get_mean_read_lengths",
    "heatmap",
    "correlate",
    "pca",
]