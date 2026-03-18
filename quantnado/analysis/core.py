"""Deprecated module for QuantNadoDataset. Use quantnado.dataset.core instead."""

import warnings
from quantnado.dataset.core import BaseStore

class QuantNadoDataset(BaseStore):
    """Deprecated: Use quantnado.dataset.core.BaseStore instead."""
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "QuantNadoDataset is deprecated and has been consolidated into BaseStore. "
            "Please use quantnado.dataset.core.BaseStore instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
