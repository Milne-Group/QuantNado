"""QuantNado package initialization."""

from quantnado.api import QuantNado, create_dataset, metadata_from_seqnado
from quantnado.analysis.core import QuantNadoDataset
from quantnado.analysis.plot import correlate, heatmap, locus_plot, metaplot, tornadoplot
from quantnado.analysis.normalise import get_library_sizes, normalise
from quantnado.analysis.pca import plot_pca_scree, plot_pca_scatter, run_pca as pca
from quantnado.dataset.metadata import AnchorPoint, FeatureType, ReductionMethod, array_key
from quantnado.dataset.store_bam import BamStore
from quantnado.dataset.store_methyl import MethylStore
from quantnado.dataset.store_variants import VariantStore

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
