"""Unit tests for quantnado.dataset.metadata.extract_metadata."""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import zarr

from quantnado.dataset.metadata import extract_metadata


# ---------------------------------------------------------------------------
# Helper: build a fake root with attrs
# ---------------------------------------------------------------------------

def _root_with_attrs(**attrs):
    """Fake object with .attrs dict, no meta group, no .get."""
    obj = MagicMock()
    obj.attrs = dict(attrs)
    obj.get = MagicMock(return_value=None)
    # Disable hasattr(obj, "sample") and hasattr(obj, "meta")
    del obj.sample
    del obj.meta
    del obj.root
    return obj


# ---------------------------------------------------------------------------
# Basic happy path via attrs
# ---------------------------------------------------------------------------

class TestExtractMetadataBasic:
    def test_returns_dataframe(self):
        obj = _root_with_attrs(sample_names=["s1", "s2"])
        result = extract_metadata(obj)
        assert isinstance(result, pd.DataFrame)

    def test_index_is_sample_id(self):
        obj = _root_with_attrs(sample_names=["s1", "s2"])
        result = extract_metadata(obj)
        assert result.index.name == "sample_id"
        assert list(result.index) == ["s1", "s2"]

    def test_metadata_columns_extracted(self):
        obj = _root_with_attrs(
            sample_names=["s1", "s2"],
            metadata_condition=["ctrl", "treat"],
        )
        result = extract_metadata(obj)
        assert "condition" in result.columns
        assert list(result["condition"]) == ["ctrl", "treat"]

    def test_empty_strings_become_na(self):
        obj = _root_with_attrs(
            sample_names=["s1", "s2"],
            metadata_group=["A", ""],
        )
        result = extract_metadata(obj)
        assert pd.isna(result["group"].iloc[1])

    def test_metadata_column_length_mismatch_ignored(self):
        # A metadata_ attr whose length doesn't match sample count is silently skipped
        obj = _root_with_attrs(
            sample_names=["s1", "s2"],
            metadata_short=["only_one"],
        )
        result = extract_metadata(obj)
        assert "short" not in result.columns

    def test_assay_column_moved_to_front(self):
        obj = _root_with_attrs(
            sample_names=["s1", "s2"],
            metadata_assay=["ATAC", "RNA"],
            metadata_condition=["ctrl", "treat"],
        )
        result = extract_metadata(obj)
        cols = list(result.columns)
        assert cols.index("assay") < cols.index("condition")


# ---------------------------------------------------------------------------
# Fallback: meta group with sample_names array
# ---------------------------------------------------------------------------

class TestExtractMetadataMetaFallback:
    def test_meta_sample_names_array(self):
        """When attrs has no sample_names but meta group has it as array."""
        obj = MagicMock()
        obj.attrs = {}
        obj.get = MagicMock(return_value=None)
        del obj.sample
        del obj.root
        # Set up meta group with sample_names array
        meta = MagicMock()
        meta.__contains__ = lambda self, key: key == "sample_names"
        meta.__getitem__ = lambda self, key: np.array(["a", "b"])
        obj.meta = meta
        result = extract_metadata(obj)
        assert list(result.index) == ["a", "b"]


# ---------------------------------------------------------------------------
# Fallback: sample attribute (xarray-like)
# ---------------------------------------------------------------------------

class TestExtractMetadataSampleAttrFallback:
    def test_sample_coord_fallback(self):
        """When neither attrs nor meta has sample_names, use .sample coord."""
        obj = MagicMock()
        obj.attrs = {}
        obj.get = MagicMock(return_value=None)
        del obj.meta
        del obj.root
        obj.sample.values = np.array(["x", "y"])
        result = extract_metadata(obj)
        assert list(result.index) == ["x", "y"]


# ---------------------------------------------------------------------------
# Fallback: root.attrs
# ---------------------------------------------------------------------------

class TestExtractMetadataRootFallback:
    def test_root_attrs_fallback(self):
        """When obj has .root with sample_names attr but no direct attrs."""
        obj = MagicMock()
        obj.attrs = {}
        obj.get = MagicMock(return_value=None)
        del obj.sample
        del obj.meta
        obj.root.attrs = {"sample_names": ["p", "q"]}
        result = extract_metadata(obj)
        assert list(result.index) == ["p", "q"]


# ---------------------------------------------------------------------------
# Error: unable to determine sample labels
# ---------------------------------------------------------------------------

class TestExtractMetadataError:
    def test_no_sample_labels_raises(self):
        obj = MagicMock()
        obj.attrs = {}
        obj.get = MagicMock(return_value=None)
        del obj.sample
        del obj.meta
        del obj.root
        with pytest.raises(ValueError, match="Unable to determine"):
            extract_metadata(obj)


# ---------------------------------------------------------------------------
# Zarr metadata group integration
# ---------------------------------------------------------------------------

class TestExtractMetadataZarrGroup:
    def test_completed_array_read(self, tmp_path):
        """When a real Zarr group has metadata/completed array."""
        import zarr
        store_path = str(tmp_path / "test.zarr")
        root = zarr.open_group(store_path, mode="w", zarr_format=3)
        root.attrs.update({"sample_names": ["s1", "s2"]})
        meta = root.create_group("metadata")
        meta.create_array("completed", shape=(2,), dtype=bool, fill_value=False)
        meta["completed"][:] = [True, False]

        result = extract_metadata(root)
        assert "completed" in result.columns
        assert list(result["completed"]) == [True, False]
