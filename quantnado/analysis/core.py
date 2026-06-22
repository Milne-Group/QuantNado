"""QuantNadoDataset — unified read-only xarray view over per-sample zarr stores.

Supports two layouts:

1. **Directory of per-sample zarrs** — ``QuantNadoDataset("dataset/")``
   Each ``.zarr`` has ``root.attrs["assay"]`` and ``root.attrs["sample"]``.

2. **Combined zarr** — ``QuantNadoDataset("dataset/combined.zarr")``
   Written by :meth:`QuantNadoDataset.combine`.

Both layouts may also be opened from a ``.tar.gz``/``.tgz`` archive.

Both expose the same API.  Auto-detected on open.
"""

from __future__ import annotations

import tarfile
import tempfile
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path

import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from loguru import logger

# Keys that are zarr infrastructure — excluded when listing chromosomes / assays
_META_KEYS = frozenset({"metadata"})
_COVERAGE_COLLAPSE_KEYS = frozenset({"atac", "chip", "cat"})


class DatasetInfo(dict):
    """Pretty-printable dict returned by :meth:`QuantNadoDataset.info`."""

    def __call__(self) -> "DatasetInfo":
        """Support legacy ``qn.info()`` usage while ``info`` remains a property."""
        return self

    @staticmethod
    def _format_chromsizes(chromsizes: dict[str, int]) -> str:
        return ", ".join(f"{chrom}={size:,}" for chrom, size in chromsizes.items())

    @staticmethod
    def _format_list(values: list[str]) -> str:
        return ", ".join(values)

    @staticmethod
    def _format_value(value) -> str:
        if isinstance(value, list):
            return DatasetInfo._format_list([str(v) for v in value])
        if isinstance(value, dict):
            return ", ".join(f"{k}={v}" for k, v in value.items())
        return str(value)

    def __repr__(self) -> str:
        assays = self.get("assays", [])
        chromosomes = self.get("chromosomes", [])
        chromsizes = self.get("chromsizes", {})
        per_assay = self.get("per_assay", {})
        extras = self.get("extras", {})

        lines = [
            "DatasetInfo",
            f"  assays      : {self._format_list(assays)}",
            f"  chromosomes : {self._format_list(chromosomes)}",
            f"  chromsizes  : {self._format_chromsizes(chromsizes)}",
        ]
        if extras:
            lines.append("")
            for key, value in extras.items():
                if key == "groups" and isinstance(value, dict):
                    lines.append("  groups")
                    for group_name, labels in value.items():
                        if isinstance(labels, dict):
                            label_text = self._format_list(
                                [str(v) for v in labels.get("labels", [])]
                            )
                            count_text = labels.get("n")
                            suffix = f" ({count_text})" if count_text is not None else ""
                            lines.append(f"    {group_name:<9}: {label_text}{suffix}")
                        else:
                            lines.append(f"    {group_name:<9}: {self._format_value(labels)}")
                    continue
                lines.append(f"  {key:<11} : {self._format_value(value)}")
        for assay, assay_info in per_assay.items():
            lines.append("")
            lines.append(f"  {assay}")
            if assay_info.get("n_samples") is not None:
                lines.append(f"    n        : {assay_info.get('n_samples')}")
            lines.append(f"    samples   : {self._format_list(assay_info.get('sample_names', []))}")
            lines.append(f"    keys      : {self._format_list(assay_info.get('array_keys', []))}")
            if assay_info.get("ips"):
                lines.append(f"    ips       : {self._format_list(assay_info.get('ips', []))}")
        return "\n".join(lines)

    __str__ = __repr__


class ObjectInfo(dict):
    """Pretty-printable summary for xarray / pandas objects."""

    @staticmethod
    def _fmt(value) -> str:
        if isinstance(value, dict):
            return ", ".join(f"{k}={v}" for k, v in value.items())
        if isinstance(value, (list, tuple)):
            return ", ".join(str(v) for v in value)
        return str(value)

    def __repr__(self) -> str:
        lines = ["ObjectInfo"]
        for key, value in self.items():
            lines.append(f"  {key:<10} : {self._fmt(value)}")
        return "\n".join(lines)

    __str__ = __repr__


class GroupInfo(dict):
    """Pretty-printable assay -> sample group mapping."""

    @staticmethod
    def _fmt_samples(samples: list[str]) -> str:
        return ", ".join(samples)

    def __repr__(self) -> str:
        lines = ["GroupInfo"]
        for assay, samples in self.items():
            lines.append(f"  {assay:<10}: {self._fmt_samples(samples)}")
        return "\n".join(lines)

    __str__ = __repr__


class NamedGroupInfo(dict):
    """Pretty-printable mapping of group-set name -> GroupInfo."""

    def __repr__(self) -> str:
        lines = ["NamedGroupInfo"]
        for name, groups in self.items():
            lines.append(f"  {name}")
            for label, samples in groups.items():
                lines.append(f"    {label:<9}: {', '.join(samples)}")
        return "\n".join(lines)

    __str__ = __repr__


# ---------------------------------------------------------------------------
# Layout detection helpers
# ---------------------------------------------------------------------------


def _is_combined_zarr(root: zarr.Group) -> bool:
    """True if this is a combined multi-sample store."""
    return "sample_names" in root.attrs or "assay_types" in root.attrs or "assays" in root.attrs


def _is_per_sample_zarr(root: zarr.Group) -> bool:
    """True if this is a single-sample store (has assay + sample attrs)."""
    return "assay" in root.attrs and "sample" in root.attrs


def _chrom_keys(root: zarr.Group) -> list[str]:
    """Chromosome group keys — excludes metadata and non-chrom entries."""
    return sorted(k for k in root.keys() if k not in _META_KEYS and isinstance(root[k], zarr.Group))


def _assay_keys(chrom_group: zarr.Group) -> list[str]:
    """Array keys inside a chromosome group."""
    return [k for k in chrom_group.keys() if isinstance(chrom_group[k], zarr.Array)]


def _is_tar_archive(path: Path) -> bool:
    """Return True when *path* is a readable tar archive, regardless of suffix."""
    return path.is_file() and tarfile.is_tarfile(path)


def _looks_like_zarr_store(path: Path) -> bool:
    """Return True when *path* contains zarr metadata at its root."""
    return any((path / name).exists() for name in ("zarr.json", ".zarray", ".zgroup"))


def _looks_like_dataset_directory(path: Path) -> bool:
    """Return True when *path* looks like a directory of per-sample zarr stores."""
    return path.is_dir() and any(
        child.is_dir() and child.name.endswith(".zarr") for child in path.iterdir()
    )


def _find_extracted_dataset_root(extract_dir: Path) -> Path:
    """Find the dataset root inside an extracted archive."""
    if _looks_like_zarr_store(extract_dir):
        return extract_dir

    children = [child for child in extract_dir.iterdir() if child.name != "__MACOSX"]
    if len(children) == 1 and children[0].is_dir():
        child = children[0]
        if _looks_like_zarr_store(child) or _looks_like_dataset_directory(child):
            return child

    if _looks_like_dataset_directory(extract_dir):
        return extract_dir

    zarr_children = [child for child in children if child.is_dir() and child.name.endswith(".zarr")]
    if len(zarr_children) == 1 and _looks_like_zarr_store(zarr_children[0]):
        return zarr_children[0]
    if zarr_children:
        return extract_dir

    raise ValueError(
        f"Archive extracted to {extract_dir}, but no QuantNado .zarr store "
        "or dataset directory was found."
    )


def _extract_tar_archive(path: Path, extract_dir: Path) -> Path:
    """Extract *path* safely and return the QuantNado dataset root within it."""
    try:
        with tarfile.open(path, mode="r:*") as tar:
            tar.extractall(extract_dir, filter="data")
    except tarfile.TarError as exc:
        raise ValueError(f"Could not read tar archive {path}: {exc}") from exc

    return _find_extracted_dataset_root(extract_dir)


def _copy_array_row_chunks(
    src_arr: zarr.Array,
    dst_arr: zarr.Array,
    row_idx: int,
    chrom_len: int,
    chunk_len: int,
    *,
    dtype: np.dtype | type | None = None,
) -> int:
    """Copy one source row into one destination row using bounded chunks."""
    bytes_written = 0
    itemsize = np.dtype(dst_arr.dtype).itemsize
    for start in range(0, chrom_len, chunk_len):
        end = min(start + chunk_len, chrom_len)
        data = src_arr[0, start:end]
        if dtype is not None:
            data = np.asarray(data, dtype=dtype)
        dst_arr[row_idx, start:end] = data
        bytes_written += (end - start) * itemsize
    return bytes_written


def _copy_sum_array_row_chunks(
    src_arrays: Sequence[zarr.Array],
    dst_arr: zarr.Array,
    row_idx: int,
    chrom_len: int,
    chunk_len: int,
    *,
    dtype: np.dtype | type = np.float32,
) -> int:
    """Copy the row-wise sum of one or more source arrays into a destination row."""
    if not src_arrays:
        return 0

    bytes_written = 0
    itemsize = np.dtype(dst_arr.dtype).itemsize
    for start in range(0, chrom_len, chunk_len):
        end = min(start + chunk_len, chrom_len)
        data = np.asarray(src_arrays[0][0, start:end], dtype=dtype)
        for src_arr in src_arrays[1:]:
            data += np.asarray(src_arr[0, start:end], dtype=dtype)
        dst_arr[row_idx, start:end] = data
        bytes_written += (end - start) * itemsize
    return bytes_written


def _run_copy_tasks(tasks: Sequence, n_workers: int) -> int:
    """Run row-copy tasks serially or with a thread pool and return bytes written."""
    if n_workers == 1 or len(tasks) <= 1:
        return sum(task() for task in tasks)

    bytes_written = 0
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(task) for task in tasks]
        for future in as_completed(futures):
            bytes_written += future.result()
    return bytes_written


def _format_rate(bytes_written: int, elapsed: float) -> str:
    """Human-readable throughput helper for combine logging."""
    if elapsed <= 0:
        return "n/a"
    return f"{bytes_written / elapsed / 1024**2:.1f} MiB/s"


# ---------------------------------------------------------------------------
# Internal store descriptors
# ---------------------------------------------------------------------------


class _PerSampleStore:
    """Wraps a single opened per-sample zarr root."""

    def __init__(self, path: Path, root: zarr.Group) -> None:
        self.path = path
        self.root = root
        attrs = dict(root.attrs)
        self.assay = attrs.get("assay", "")
        self.sample = attrs.get("sample", path.stem)
        self.ip = attrs.get("ip", "")
        self.stranded = attrs.get("stranded", "")
        self.chromsizes: dict[str, int] = {
            str(k): int(v) for k, v in attrs.get("chromsizes", {}).items()
        }
        self.chromosomes = _chrom_keys(root)
        self.chunk_len: int = int(attrs.get("chunk_len", 65536))
        self.viewpoints: list[str] = attrs.get("viewpoints", [])
        meta = root.get("metadata")
        self.completed = (
            bool(meta["completed"][0]) if meta is not None and "completed" in meta else False
        )
        self.total_reads: int = (
            int(meta["total_reads"][0]) if meta is not None and "total_reads" in meta else 0
        )
        self.mean_read_length: float = (
            float(meta["mean_read_length"][0])
            if meta is not None and "mean_read_length" in meta
            else 0.0
        )
        self.sparsity: float = (
            float(meta["sparsity"][0]) if meta is not None and "sparsity" in meta else 0.0
        )

    def array_keys(self) -> list[str]:
        if not self.chromosomes:
            return []
        return _assay_keys(self.root[self.chromosomes[0]])

    def get_array(self, chrom: str, key: str) -> zarr.Array:
        return self.root[chrom][key]


# ---------------------------------------------------------------------------
# QuantNadoDataset
# ---------------------------------------------------------------------------


class QuantNadoDataset:
    """Unified read-only xarray view over QuantNado zarr stores.

    Parameters
    ----------
    path:
        Path to either:
        - a directory containing per-sample ``.zarr`` stores, or
        - a single combined ``.zarr`` store written by :meth:`combine`.
        ``.tar.gz``/``.tgz`` archives containing either layout are extracted to
        a temporary directory and opened read-only.

    Examples
    --------
    >>> qn = QuantNadoDataset("dataset/")
    >>> region = qn.sel(chrom="chr1", start=1_000_000, end=1_001_000)
    >>> region["atac"].sel(sample="ATAC-SEM-1")
    >>> tree = qn.to_datatree()
    """

    def __init__(self, path: Path | str, annotation: str | Path | None = None) -> None:
        input_path = Path(path)
        self.source_path = input_path
        self.path = input_path
        self._archive_tmpdir: tempfile.TemporaryDirectory[str] | None = None
        self._stores: list[_PerSampleStore] = []
        self._combined_root: zarr.Group | None = None
        self._combined = False
        self._genes_df: pd.DataFrame | None = None
        self._exons_df: pd.DataFrame | None = None
        self._subset_samples: list[str] | None = None
        self._group_sets: dict[str, dict[str, list[str]]] = {}
        self._last_group_name: str | None = None

        if input_path.exists() and _is_tar_archive(input_path):
            self._archive_tmpdir = tempfile.TemporaryDirectory(prefix="quantnado-")
            self.path = _extract_tar_archive(input_path, Path(self._archive_tmpdir.name))

        if self.path.is_dir() and not str(self.path).endswith(".zarr"):
            # Directory of per-sample zarrs
            self._load_directory(self.path)
        elif self.path.exists():
            root = zarr.open_group(str(self.path), mode="r")
            if _is_combined_zarr(root):
                self._combined_root = root
                self._combined = True
            elif _is_per_sample_zarr(root):
                # Single per-sample zarr opened directly
                self._stores.append(_PerSampleStore(self.path, root))
            else:
                raise ValueError(f"Cannot determine store layout for {self.path}")
        else:
            raise FileNotFoundError(f"Path does not exist: {self.path}")

        if annotation is not None:
            self.set_annotation(annotation)

    def _load_directory(self, directory: Path) -> None:
        zarr_paths = sorted(directory.glob("*.zarr"))
        if not zarr_paths:
            raise FileNotFoundError(f"No .zarr stores found in {directory}")
        for zp in zarr_paths:
            try:
                root = zarr.open_group(str(zp), mode="r")
                if _is_per_sample_zarr(root):
                    store = _PerSampleStore(zp, root)
                    if not store.completed:
                        logger.warning(f"Skipping incomplete store: {zp.name}")
                        continue
                    self._stores.append(store)
                else:
                    logger.debug(f"Skipping {zp.name}: not a per-sample store")
            except Exception as e:
                logger.warning(f"Could not open {zp}: {e}")
        if not self._stores:
            raise ValueError(f"No completed per-sample stores found in {directory}")
        logger.info(f"Opened {len(self._stores)} stores from {directory}")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def sample_names(self) -> list[str]:
        if self._combined:
            if self._subset_samples is not None:
                return self._subset_samples
            names = self._combined_root.attrs.get("sample_names", [])
            return [str(s) for s in names]
        # Per-sample: all sample names across all stores (MCC may contribute multiple)
        names = []
        for store in self._stores:
            if store.viewpoints:
                names.extend(f"{store.sample}_{vp}" for vp in store.viewpoints)
            else:
                names.append(store.sample)
        return names

    @property
    def assays(self) -> list[str]:
        """Distinct biological assay types present (e.g. 'ATAC', 'RNA', 'METH')."""
        if self._combined:
            if self._subset_samples is not None:
                return sorted(set(a.upper() for a in self._get_assay_per_sample() if a))
            return sorted(set(s.upper() for s in self._combined_root.attrs.get("assay_types", [])))
        return sorted(set(s.assay.upper() for s in self._stores))

    @property
    def array_keys(self) -> list[str]:
        """All zarr data-variable names (e.g. 'atac', 'rna_fwd', 'coverage', 'AF')."""
        if self._combined:
            all_keys = [str(k) for k in self._combined_root.attrs.get("array_keys", [])]
            if self._subset_samples is None:
                return all_keys

            subset = set(self._subset_samples)
            key_to_samples = {
                str(k): {str(s) for s in v}
                for k, v in dict(self._combined_root.attrs.get("key_to_samples", {})).items()
            }
            return [
                key
                for key in all_keys
                if key == "coverage" or bool(key_to_samples.get(key, set()) & subset)
            ]
        keys: set[str] = set()
        for store in self._stores:
            keys.update(store.array_keys())
        return sorted(keys)

    @property
    def chromosomes(self) -> list[str]:
        if self._combined:
            return _chrom_keys(self._combined_root)
        if not self._stores:
            return []
        return self._stores[0].chromosomes

    @property
    def chromsizes(self) -> dict[str, int]:
        if self._combined:
            stored = self._combined_root.attrs.get("chromsizes", {})
            return {str(k): int(v) for k, v in stored.items()}
        if not self._stores:
            return {}
        return self._stores[0].chromsizes

    @property
    def completed_mask(self) -> np.ndarray:
        if self._combined:
            meta = self._combined_root.get("metadata")
            if meta is not None and "completed" in meta:
                full_mask = meta["completed"][:].astype(bool)
                if self._subset_samples is not None:
                    full_names = [str(s) for s in self._combined_root.attrs.get("sample_names", [])]
                    name_to_idx = {s: i for i, s in enumerate(full_names)}
                    return np.array(
                        [
                            full_mask[name_to_idx[s]]
                            for s in self._subset_samples
                            if s in name_to_idx
                        ],
                        dtype=bool,
                    )
                return full_mask
            return np.ones(len(self.sample_names), dtype=bool)
        return np.array([s.completed for s in self._stores], dtype=bool)

    @property
    def groups(self) -> GroupInfo:
        """Assay-based sample groups for the current dataset view."""
        return self.group_by("assay")

    @property
    def group_sets(self) -> dict[str, GroupInfo]:
        """Cached named group sets for the current dataset view."""
        return {name: GroupInfo(groups) for name, groups in self._group_sets.items()}

    def _get_assay_per_sample(self) -> list[str]:
        """Return assay type string for each sample, in sample_names order."""
        if self._combined:
            meta = self._combined_root.get("metadata")
            if meta is not None and "assay" in meta:
                raw = meta["assay"][:]
                all_assays = [s.decode() if isinstance(s, bytes) else str(s) for s in raw]
                if self._subset_samples is not None:
                    full_names = [str(s) for s in self._combined_root.attrs.get("sample_names", [])]
                    name_to_idx = {s: i for i, s in enumerate(full_names)}
                    return [
                        all_assays[name_to_idx[s]] for s in self._subset_samples if s in name_to_idx
                    ]
                return all_assays
            return [""] * len(self.sample_names)
        result = []
        for store in self._stores:
            if store.viewpoints:
                result.extend([store.assay.upper()] * len(store.viewpoints))
            else:
                result.append(store.assay.upper())
        return result

    def _get_ip_per_sample(self) -> list[str]:
        """Return IP / target label per sample, in sample_names order."""

        def _fill_missing(values: list[str], assays: list[str], samples: list[str]) -> list[str]:
            return [
                value
                if str(value).strip()
                else QuantNadoDataset._infer_ip_from_sample_name(sample, assay)
                for value, assay, sample in zip(values, assays, samples)
            ]

        if self._combined:
            meta = self._combined_root.get("metadata")
            if meta is not None and "ip" in meta:
                raw = meta["ip"][:]
                all_values = [s.decode() if isinstance(s, bytes) else str(s) for s in raw]
                if self._subset_samples is not None:
                    full_names = [str(s) for s in self._combined_root.attrs.get("sample_names", [])]
                    name_to_idx = {s: i for i, s in enumerate(full_names)}
                    samples = [s for s in self._subset_samples if s in name_to_idx]
                    values = [all_values[name_to_idx[s]] for s in samples]
                    assays = self._get_assay_per_sample()
                    return _fill_missing(values, assays, samples)
                return _fill_missing(all_values, self._get_assay_per_sample(), self.sample_names)
            return _fill_missing(
                [""] * len(self.sample_names),
                self._get_assay_per_sample(),
                self.sample_names,
            )
        result = []
        for store in self._stores:
            value = str(store.ip or "") or QuantNadoDataset._infer_ip_from_sample_name(
                store.sample, store.assay
            )
            if store.viewpoints:
                result.extend([value] * len(store.viewpoints))
            else:
                result.append(value)
        return result

    def _get_stranded_per_sample(self) -> list[str]:
        """Return strandedness label per sample, in sample_names order."""
        if self._combined:
            meta = self._combined_root.get("metadata")
            if meta is not None and "stranded" in meta:
                raw = meta["stranded"][:]
                all_values = [s.decode() if isinstance(s, bytes) else str(s) for s in raw]
                if self._subset_samples is not None:
                    full_names = [str(s) for s in self._combined_root.attrs.get("sample_names", [])]
                    name_to_idx = {s: i for i, s in enumerate(full_names)}
                    return [
                        all_values[name_to_idx[s]] for s in self._subset_samples if s in name_to_idx
                    ]
                return all_values
            return [""] * len(self.sample_names)
        result = []
        for store in self._stores:
            value = str(getattr(store, "stranded", "") or "")
            if store.viewpoints:
                result.extend([value] * len(store.viewpoints))
            else:
                result.append(value)
        return result

    @staticmethod
    def _infer_ip_from_sample_name(sample_name: str, assay_name: str) -> str:
        """Best-effort fallback for missing IP metadata from assay-style sample names."""
        assay_upper = str(assay_name or "").upper()
        if assay_upper not in {"CHIP", "CUT&TAG", "CUTTAG", "CAT", "CUT&RUN", "CUTRUN"}:
            return ""
        sample = str(sample_name)
        if "_" not in sample:
            return ""
        return sample.rsplit("_", 1)[-1]

    @property
    def metadata(self) -> pd.DataFrame:
        """Return per-sample metadata with core fields plus cached ``group_by`` labels."""
        sample_names = list(self.sample_names)
        assays = self._get_assay_per_sample()
        ips = self._get_ip_per_sample()
        stranded = self._get_stranded_per_sample()

        rows = []
        for sample, assay, ip, strand in zip(sample_names, assays, ips, stranded):
            rows.append(
                {
                    "sample_id": sample,
                    "assay": str(assay).upper(),
                    "ip": str(ip).strip() or None,
                    "stranded": str(strand).strip() or None,
                }
            )

        metadata_df = pd.DataFrame(rows)
        if metadata_df.empty:
            return metadata_df

        # Promote cached group labels to first-class metadata columns. These
        # are user-defined and should override best-effort name parsing.
        for group_name, groups in self._group_sets.items():
            if not groups:
                continue
            labels_by_sample: dict[str, str] = {}
            for label, members in groups.items():
                for sample in members:
                    if sample in labels_by_sample and labels_by_sample[sample] != label:
                        labels_by_sample[sample] = f"{labels_by_sample[sample]}|{label}"
                    else:
                        labels_by_sample[sample] = label
            metadata_df[group_name] = metadata_df["sample_id"].map(labels_by_sample)

        preferred = [
            "sample_id",
            "assay",
            "ip",
            "stranded",
        ]
        remaining = [col for col in metadata_df.columns if col not in preferred]
        metadata_df = metadata_df.loc[
            :, [col for col in preferred if col in metadata_df.columns] + sorted(remaining)
        ]

        sort_cols = [col for col in ["assay", "sample_id"] if col in metadata_df.columns]
        return metadata_df.sort_values(sort_cols, na_position="last").reset_index(drop=True)

    @staticmethod
    def _canonicalise_assay_name(assay_name: str) -> str:
        """Normalise biological assay aliases so assay filters match stored metadata."""
        assay_upper = str(assay_name or "").strip().upper()
        alias_map = {
            "CUTTAG": "CUT&TAG",
            "CAT": "CUT&TAG",
            "CUTRUN": "CUT&RUN",
        }
        return alias_map.get(assay_upper, assay_upper)

    def group_by(
        self,
        by: str = "assay",
        *,
        groups: "dict[str, list[str] | str] | None" = None,
        match: str = "exact",
        drop_empty: bool = True,
        **named_groups: "dict[str, list[str] | str]",
    ) -> "GroupInfo | NamedGroupInfo":
        """Build sample groups from metadata or an explicit mapping.

        Parameters
        ----------
        by:
            Metadata field to group on. Currently supports ``"assay"``,
            ``"ip"``, and ``"stranded"``.
        groups:
            Optional explicit label -> sample list mapping. When provided, this
            takes precedence over ``by`` and is validated against the current
            dataset view. With ``match="exact"`` (default), values are treated
            as explicit sample names. With ``match="contains"``, string or
            string-list values are treated as case-insensitive substrings to
            search for in sample names.
        match:
            How to interpret ``groups`` values. One of ``"exact"`` or
            ``"contains"``.
        drop_empty:
            If True (default), drop groups whose label is empty.
        """
        if named_groups:
            result: dict[str, GroupInfo] = {}
            for name, spec in named_groups.items():
                self._last_group_name = str(name)
                if isinstance(spec, str) and spec.lower() in {"assay", "ip", "stranded"}:
                    result[str(name)] = self.group_by(
                        by=spec,
                        drop_empty=drop_empty,
                    )
                else:
                    result[str(name)] = self.group_by(
                        groups=spec,
                        match=match,
                        drop_empty=drop_empty,
                        by="assay",
                    )
                self._last_group_name = str(name)
            return NamedGroupInfo(result)

        if groups is not None:
            if isinstance(groups, str):
                if groups.lower() in {"assay", "ip", "stranded"}:
                    return self.group_by(by=groups, drop_empty=drop_empty)
                raise ValueError(
                    "group_by(groups=...) expects a mapping of label -> samples/patterns, "
                    "or use a metadata shorthand like ip='ip'."
                )
            valid = set(self.sample_names)
            grouped: dict[str, list[str]] = {}
            if match not in {"exact", "contains"}:
                raise ValueError("group_by(match=...) currently supports 'exact' or 'contains'")

            for label, samples in groups.items():
                if match == "exact":
                    if isinstance(samples, str):
                        requested = [samples]
                    else:
                        requested = [str(s) for s in samples]
                    grouped[str(label)] = [str(s) for s in requested if str(s) in valid]
                else:
                    patterns = [samples] if isinstance(samples, str) else [str(s) for s in samples]
                    patterns_lower = [p.lower() for p in patterns]
                    grouped[str(label)] = [
                        sample
                        for sample in self.sample_names
                        if any(pattern in sample.lower() for pattern in patterns_lower)
                    ]
            result = GroupInfo({k: v for k, v in grouped.items() if v})
            cache_name = self._last_group_name or "group"
            self._group_sets[cache_name] = {k: list(v) for k, v in result.items()}
            return result

        key = by.lower()
        if key == "assay":
            values = self._get_assay_per_sample()
            labels = [str(v).upper() if v else "" for v in values]
        elif key == "ip":
            values = self._get_ip_per_sample()
            labels = [str(v) for v in values]
        elif key == "stranded":
            values = self._get_stranded_per_sample()
            labels = [str(v) for v in values]
        else:
            raise ValueError("group_by(by=...) currently supports 'assay', 'ip', or 'stranded'")

        grouped: dict[str, list[str]] = {}
        for sample, label in zip(self.sample_names, labels):
            if drop_empty and not label:
                continue
            grouped.setdefault(label or "UNKNOWN", []).append(sample)
        result = GroupInfo(grouped)
        self._group_sets[key] = {k: list(v) for k, v in result.items()}
        self._last_group_name = key
        return result

    # ------------------------------------------------------------------
    # Subset
    # ------------------------------------------------------------------

    def subset(
        self,
        assay: "str | Sequence[str] | None" = None,
        samples: "str | Sequence[str] | None" = None,
        ip: "str | Sequence[str] | None" = None,
        group: "str | Sequence[str] | dict[str, str | Sequence[str]] | None" = None,
    ) -> "QuantNadoDataset":
        """Return a new QuantNadoDataset restricted to the specified filters.

        No data is copied — the returned object shares the same zarr handles.
        Use this to avoid repeating ``assay=`` or ``samples=`` on every call::

            rna = qn.subset(assay="RNA")
            reduced = rna.reduce(intervals_path="promoters.bed")
            normalised = rna.normalise(reduced, method="cpm")

        Parameters
        ----------
        assay:
            One or more assay types (e.g. ``"RNA"``, ``["ATAC", "ChIP"]``).
        samples:
            Explicit sample name(s).
        ip:
            One or more IP / target labels (e.g. ``"MLL"``, ``["H3K27ac", "TET2"]``).
        group:
            Labels from cached group sets. Pass a string / list to use the most
            recent :meth:`group_by` namespace, or a mapping like
            ``{"treatment": "treated", "replicate": "rep1"}``.

        Returns
        -------
        QuantNadoDataset
            A lightweight view over the same stores, filtered to the resolved samples.
        """
        resolved = QuantNadoDataset._resolve_samples(
            self,
            assay=assay,
            samples=samples,
            ip=ip,
            group=group,
        )

        new: QuantNadoDataset = object.__new__(QuantNadoDataset)
        new.path = self.path
        new.source_path = getattr(self, "source_path", self.path)
        new._archive_tmpdir = getattr(self, "_archive_tmpdir", None)
        new._combined = self._combined
        new._combined_root = self._combined_root
        new._genes_df = self._genes_df
        new._exons_df = self._exons_df
        new._group_sets = {k: dict(v) for k, v in self._group_sets.items()}
        new._last_group_name = self._last_group_name

        if self._combined:
            new._stores = []
            new._subset_samples = resolved
        else:
            resolved_set = set(resolved)
            new._stores = [
                s
                for s in self._stores
                if s.sample in resolved_set
                or any(f"{s.sample}_{vp}" in resolved_set for vp in s.viewpoints)
            ]
            new._subset_samples = None

        return new

    # ------------------------------------------------------------------
    # Gene annotation
    # ------------------------------------------------------------------

    def set_annotation(self, gtf_path: str | Path) -> None:
        """Attach a GTF annotation file for gene-name-based queries.

        Parameters
        ----------
        gtf_path:
            Path to a GTF or GTF.gz file (e.g. hg38.gtf.gz).
        """
        from .features import load_gtf

        gtf = load_gtf(
            str(gtf_path),
            feature_types=["gene", "exon"],
            usecols=[
                "gene_id",
                "gene_name",
                "transcript_id",
                "gene_type",
                "gene_biotype",
                "exon_number",
            ],
        )
        df = pd.DataFrame(gtf)
        self._genes_df = df[df["feature"] == "gene"].reset_index(drop=True)
        self._exons_df = df[df["feature"] == "exon"].reset_index(drop=True)
        if self._genes_df.empty:
            logger.warning("No 'gene' features found in GTF; falling back to 'transcript'")
            transcript_gtf = load_gtf(
                str(gtf_path),
                feature_types=["transcript"],
                usecols=["gene_id", "gene_name", "transcript_id", "gene_type", "gene_biotype"],
            )
            self._genes_df = pd.DataFrame(transcript_gtf).reset_index(drop=True)
        logger.info(f"Loaded annotation: {len(self._genes_df):,} genes from {gtf_path}")

    def gene_info(self, name: str) -> dict:
        """Look up a gene by name and return its coordinates and exon structure.

        Parameters
        ----------
        name:
            Gene name (e.g. "GNAQ"). Case-sensitive; falls back to case-insensitive.

        Returns
        -------
        dict with keys: gene_name, gene_id, chrom, start, end, strand, locus, exons.
        Coordinates are 1-based inclusive.
        """
        if self._genes_df is None:
            raise RuntimeError(
                "No annotation loaded. Call set_annotation() or pass annotation= to __init__."
            )

        if "gene_name" not in self._genes_df.columns:
            raise KeyError(f"Annotation has no 'gene_name' column — cannot look up '{name}'.")

        # Case-sensitive lookup first, then case-insensitive fallback
        hits = self._genes_df[self._genes_df["gene_name"] == name]
        if hits.empty:
            hits = self._genes_df[self._genes_df["gene_name"].str.upper() == name.upper()]
        if hits.empty:
            raise KeyError(f"Gene '{name}' not found in annotation.")

        # Multiple hits (e.g. PAR regions) → take the largest span
        if len(hits) > 1:
            hits = hits.loc[[(hits["End"] - hits["Start"]).idxmax()]]
        row = hits.iloc[0]

        # PyRanges: 0-based half-open [Start, End) → 1-based inclusive [start, end]
        chrom = str(row["Chromosome"])
        start_1 = int(row["Start"]) + 1
        end_1 = int(row["End"])
        strand = str(row.get("Strand", "+"))
        gene_id = row.get("gene_id") or None
        gene_name_out = row.get("gene_name") or name

        # Exons for this gene
        exons_out: list[dict] = []
        if self._exons_df is not None and not self._exons_df.empty:
            if gene_id and "gene_id" in self._exons_df.columns:
                ex = self._exons_df[self._exons_df["gene_id"] == gene_id]
            elif "gene_name" in self._exons_df.columns:
                ex = self._exons_df[self._exons_df["gene_name"] == gene_name_out]
            else:
                ex = self._exons_df.iloc[0:0]  # empty
            for _, erow in ex.iterrows():
                exons_out.append(
                    {
                        "start": int(erow["Start"]) + 1,
                        "end": int(erow["End"]),
                        "exon_number": str(erow["exon_number"])
                        if "exon_number" in erow and pd.notna(erow.get("exon_number"))
                        else None,
                    }
                )
            exons_out.sort(key=lambda e: e["start"])

        return {
            "gene_name": gene_name_out,
            "gene_id": gene_id,
            "chrom": chrom,
            "start": start_1,
            "end": end_1,
            "strand": strand,
            "locus": f"{chrom}:{start_1}-{end_1}",
            "exons": exons_out,
        }

    def sel_gene(
        self,
        name: str,
        padding: int = 0,
        assay: "str | Sequence[str] | None" = None,
        samples: "str | Sequence[str] | None" = None,
    ) -> xr.Dataset:
        """Select a genomic region by gene name.

        Parameters
        ----------
        name:
            Gene name (e.g. "GNAQ").
        padding:
            Extra bases to add on each side of the gene body (default: 0).
        assay:
            Optional assay filter passed to :meth:`sel`.

        Returns
        -------
        xr.Dataset with gene metadata in ``.attrs``:
        ``gene_name``, ``gene_id``, ``gene_strand``, ``locus``, ``exons``.
        """
        info = self.gene_info(name)
        chrom = info["chrom"]
        chrom_len = self.chromsizes.get(chrom, 0)

        start = max(1, info["start"] - padding)
        end = info["end"] + padding
        if chrom_len:
            end = min(end, chrom_len)

        ds = self.sel(chrom, start, end, assay=assay, samples=samples)
        ds.attrs.update(
            {
                "gene_name": info["gene_name"],
                "gene_id": info["gene_id"],
                "gene_strand": info["strand"],
                "locus": f"{chrom}:{start}-{end}",
                "exons": info["exons"],
            }
        )
        return ds

    # ------------------------------------------------------------------
    # Region selection
    # ------------------------------------------------------------------

    def sel(
        self,
        chrom: str,
        start: int | None = None,
        end: int | None = None,
        assay: "str | Sequence[str] | None" = None,
        samples: "str | Sequence[str] | None" = None,
    ) -> xr.Dataset:
        """Extract a genomic region as an xr.Dataset.

        Parameters
        ----------
        chrom:
            Chromosome name (e.g. ``"chr1"``).
        start:
            1-based start position (inclusive). Defaults to 1.
        end:
            1-based end position (inclusive). Defaults to chromosome length.
        assay:
            If provided, restrict to samples whose assay type matches
            (e.g. ``"atac"``, ``"rna"``, ``"meth"``). The returned Dataset
            will only contain the array keys that belong to that assay, and
            only the samples that have that assay type attached as the
            ``assay`` coordinate.

        Returns
        -------
        xr.Dataset
            dims: sample × position
            coords: position (1-based), sample, assay (non-index on sample)
            data_vars: one per assay key (atac, chip_h3k27ac, rna_fwd, …)
        """
        chrom_len = self.chromsizes.get(chrom)
        if chrom_len is None:
            raise ValueError(f"Chromosome '{chrom}' not in store. Available: {self.chromosomes}")

        start_1 = start if start is not None else 1
        end_1 = end if end is not None else chrom_len

        if start_1 < 1:
            raise ValueError(f"start must be >= 1 (1-based), got {start_1}")
        if end_1 > chrom_len:
            raise ValueError(f"end {end_1} exceeds chromosome length {chrom_len}")
        if end_1 < start_1:
            raise ValueError(f"end {end_1} < start {start_1}")

        # 0-based slice for array indexing
        s0 = start_1 - 1
        e0 = end_1

        position_coords = np.arange(start_1, end_1 + 1, dtype=np.int64)

        if self._combined:
            ds = self._sel_combined(chrom, s0, e0, position_coords)
        else:
            ds = self._sel_per_sample(chrom, s0, e0, position_coords)

        if samples is not None:
            ds = ds.sel(sample=self._resolve_samples(samples=samples))
        elif assay is not None:
            assay_upper = {a.upper() for a in ([assay] if isinstance(assay, str) else assay)}
            if "assay" not in ds.coords:
                raise ValueError("Dataset has no 'assay' coordinate — cannot filter by assay.")
            assay_mask = np.array([a in assay_upper for a in ds.coords["assay"].values], dtype=bool)
            if not assay_mask.any():
                available = sorted(set(ds.coords["assay"].values))
                raise ValueError(f"Assay '{assay}' not found. Available: {available}")
            ds = ds.isel(sample=assay_mask)

        stranded_by_sample = dict(zip(self.sample_names, self._get_stranded_per_sample()))
        return _orient_rna_strands(ds, stranded_by_sample)

    def _sel_per_sample(
        self, chrom: str, s0: int, e0: int, position_coords: np.ndarray
    ) -> xr.Dataset:
        """Build Dataset by grouping per-sample stores by assay key."""
        # Group stores by assay key they contribute
        key_to_stores: dict[str, list[_PerSampleStore]] = {}
        for store in self._stores:
            if chrom not in store.chromosomes:
                continue
            for key in store.array_keys():
                key_to_stores.setdefault(key, []).append(store)

        # Build the full ordered sample list and assay-per-sample mapping
        all_samples: list[str] = []
        sample_assay: dict[str, str] = {}
        for store in self._stores:
            if store.viewpoints:
                for vp in store.viewpoints:
                    name = f"{store.sample}_{vp}"
                    all_samples.append(name)
                    sample_assay[name] = store.assay.upper()
            else:
                all_samples.append(store.sample)
                sample_assay[store.sample] = store.assay.upper()

        assay_coord = np.array([sample_assay.get(s, "") for s in all_samples])

        data_vars: dict[str, xr.DataArray] = {}
        for key, stores in sorted(key_to_stores.items()):
            chunks: list[da.Array] = []
            sample_labels: list[str] = []
            for store in stores:
                arr = store.get_array(chrom, key)  # (1, chrom_len)
                chunk = da.from_zarr(arr)[:, s0:e0]  # (1, region_len)
                chunks.append(chunk)
                if store.viewpoints and key.startswith("viewpoint_"):
                    vp = key[len("viewpoint_") :]
                    sample_labels.append(f"{store.sample}_{vp}")
                else:
                    sample_labels.append(store.sample)
            stacked = da.concatenate(chunks, axis=0)  # (n_samples, region_len)
            # Reindex to full sample list so all vars share the same sample coord
            da_var = xr.DataArray(
                stacked,
                dims=("sample", "position"),
                coords={"sample": sample_labels, "position": position_coords},
            ).reindex(sample=all_samples)
            data_vars[key] = da_var

        ds = xr.Dataset(data_vars)
        ds = ds.assign_coords(assay=("sample", assay_coord))
        return ds

    def _sel_combined(
        self, chrom: str, s0: int, e0: int, position_coords: np.ndarray
    ) -> xr.Dataset:
        """Build Dataset from a combined zarr."""
        root = self._combined_root
        chrom_grp = root[chrom]
        all_assays = _assay_keys(chrom_grp)
        key_to_samples: dict[str, list[str]] = dict(root.attrs.get("key_to_samples", {}))
        all_samples = self.sample_names

        # Assay label per sample from metadata (written by combine())
        meta = root.get("metadata")
        if meta is not None and "assay" in meta:
            raw = meta["assay"][:]
            assay_coord_full = np.array(
                [s.decode() if isinstance(s, bytes) else str(s) for s in raw]
            )
            full_names = [str(s) for s in root.attrs.get("sample_names", [])]
            if full_names and len(full_names) != len(all_samples):
                name_to_idx = {s: i for i, s in enumerate(full_names)}
                assay_coord = np.array(
                    [assay_coord_full[name_to_idx[s]] for s in all_samples if s in name_to_idx]
                )
            else:
                assay_coord = assay_coord_full
        else:
            assay_coord = np.array([""] * len(all_samples))

        data_vars: dict[str, xr.DataArray] = {}
        for key in all_assays:
            arr = chrom_grp[key]  # (n_samples_for_assay, chrom_len)
            dask_arr = da.from_zarr(arr)[:, s0:e0]
            sample_labels = key_to_samples.get(
                key, [f"{key}_{i}" for i in range(dask_arr.shape[0])]
            )
            da_var = xr.DataArray(
                dask_arr,
                dims=("sample", "position"),
                coords={"sample": sample_labels, "position": position_coords},
            )
            # Reindex to the full sample list so all DataArrays share the same
            # sample coordinate.  Integer dtypes are upcasted to float64 to
            # accommodate NaN for samples that don't contribute to this assay.
            data_vars[key] = da_var.reindex(sample=all_samples)

        ds = xr.Dataset(data_vars)
        ds = ds.assign_coords(assay=("sample", assay_coord))
        return ds

    # ------------------------------------------------------------------
    # DataTree
    # ------------------------------------------------------------------

    def to_datatree(self, chromosomes: list[str] | None = None, lazy: bool = True) -> xr.DataTree:
        """Return the full dataset as an xr.DataTree.

        Each chromosome is a child node containing an xr.Dataset with
        1-based position coordinates and assay data variables.

        Parameters
        ----------
        chromosomes:
            Subset of chromosomes. Defaults to all.
        lazy:
            If True (default), use lazy dask arrays without materializing
            position coordinate arrays. This is much faster for large datasets.
            If False, materializes the full position coordinate for each chromosome.
        """
        chroms = chromosomes if chromosomes is not None else self.chromosomes
        nodes: dict[str, xr.Dataset] = {"/": xr.Dataset(attrs={"chromsizes": self.chromsizes})}
        for chrom in chroms:
            chrom_len = self.chromsizes[chrom]
            ds = self.sel(chrom)

            if not lazy:
                # Materialize position coords (slow for large chromosomes)
                position_coords = np.arange(1, chrom_len + 1, dtype=np.int64)
                ds = ds.assign_coords(position=position_coords)

            nodes[chrom] = ds
        return xr.DataTree.from_dict(nodes)

    def extract(
        self,
        feature_type: str = "promoter",
        GTF_FILE: str | None = None,
        anchor_feature: str = "gene",
        fixed_width: int | None = None,
        upstream: int | None = None,
        downstream: int | None = None,
        anchor: str = "start",
        flip_strand: bool = True,
        bin_size: int = 50,
        assay: "str | Sequence[str] | None" = None,
        modality: "str | Sequence[str] | None" = None,
        samples: "str | Sequence[str] | None" = None,
    ) -> xr.DataArray:
        """Extract signal into fixed-width bins around genomic features.

        Parameters
        ----------
        feature_type : str
            Feature type: "promoter", "gene", "transcript", or "exon".
        GTF_FILE : str
            Path to GTF file.
        anchor_feature : str
            Which feature type to anchor promoters on when ``feature_type="promoter"``.
            Usually ``"gene"`` or ``"transcript"``. This is passed through to
            :func:`quantnado.analysis.features.extract_promoters`.
        fixed_width : int, optional
            If provided, expands features to this width around anchor point.
            If None, uses feature length.
        upstream, downstream : int, optional
            Window around the anchor in base pairs. When provided, positions are
            extracted from ``anchor - upstream`` to ``anchor + downstream`` and
            plotted on a signed coordinate axis (for example ``-2000 .. 0 .. 2000``).
            Cannot be used together with ``fixed_width``.
        anchor : str
            Anchor point: "start", "end", or "midpoint".
        flip_strand : bool
            If True (default), reverse minus-strand intervals after extraction
            so the returned windows are oriented 5'→3' relative to the anchor.
            This is especially useful for gene/transcript-body style plots where
            the gene body should lie to the right of the TSS.
        bin_size : int
            Width of each bin in bp (default: 50).
        assay : str, optional
            Assay type to restrict samples to (e.g. "RNA", "ATAC", "METH").
            Also accepted as the array key for backward compatibility.
        modality : str, optional
            Array key to extract (e.g. "rna_fwd", "coverage", "methyl_pct").
            Required when assay is a type name rather than an array key.
        samples : list of str, optional
            Sample names to extract. If None, uses all samples.

        Returns
        -------
        xr.DataArray
            Shape (interval, bin, sample) with binned signal.
        """
        import pandas as pd

        from .features import extract_feature_ranges, extract_promoters, load_gtf
        from .ranges import extract_signal_into_bins

        if GTF_FILE is None:
            raise ValueError("GTF_FILE is required")
        if fixed_width is not None and (upstream is not None or downstream is not None):
            raise ValueError("Cannot specify both fixed_width and upstream/downstream")

        # Resolve modality (array key) and optional assay-type sample filter.
        # Backward compat: if only assay is given and it looks like an array key,
        # treat it as modality with no sample filtering.
        array_keys = self.array_keys
        if modality is not None:
            array_key = self._resolve_modalities(modality)
        elif isinstance(assay, str) and assay.lower() in [k.lower() for k in array_keys]:
            array_key = assay
            assay = None  # not a type filter
        elif assay is not None:
            raise ValueError(
                f"modality is required when assay='{assay}' is an assay type. "
                f"Available array keys: {array_keys}"
            )
        else:
            raise ValueError("Either assay (array key) or modality must be provided.")

        if samples is None:
            if assay is not None:
                samples = self._resolve_samples(assay=assay)
            else:
                samples = self.sample_names
        else:
            samples = self._resolve_samples(samples=samples)

        # Load and extract features
        gtf = load_gtf(GTF_FILE)

        _promoter_upstream = upstream if upstream is not None else 1000
        _promoter_downstream = downstream if downstream is not None else 200

        if feature_type == "promoter":
            features_pr = extract_promoters(
                gtf,
                upstream=_promoter_upstream,
                downstream=_promoter_downstream,
                anchor_feature=anchor_feature,
            )
        else:
            features_pr = extract_feature_ranges(gtf, feature_type=feature_type)

        features_df = pd.DataFrame(features_pr)

        # Rename PyRanges columns to standard names
        if "Chromosome" in features_df.columns:
            features_df = features_df.rename(columns={"Chromosome": "chrom"})
        if "Start" in features_df.columns:
            features_df = features_df.rename(columns={"Start": "start"})
        if "End" in features_df.columns:
            features_df = features_df.rename(columns={"End": "end"})

        strand_col = next((c for c in ("Strand", "strand") if c in features_df.columns), None)
        strands = features_df[strand_col].fillna("+").astype(str).values if strand_col else None

        if feature_type == "promoter":
            # extract_promoters already sets start = TSS - upstream, end = TSS + downstream
            # for all strands, so no anchor re-computation or re-windowing is needed.
            window_upstream = _promoter_upstream
            window_downstream = _promoter_downstream
        else:
            # Apply fixed-width or upstream/downstream windowing around the chosen anchor
            if anchor == "start":
                anchor_pos = features_df["start"].values.copy()
                if strands is not None:
                    minus_mask = strands == "-"
                    anchor_pos[minus_mask] = features_df.loc[minus_mask, "end"].values
            elif anchor == "end":
                anchor_pos = features_df["end"].values.copy()
                if strands is not None:
                    plus_mask = strands == "+"
                    minus_mask = strands == "-"
                    anchor_pos[plus_mask] = features_df.loc[plus_mask, "end"].values
                    anchor_pos[minus_mask] = features_df.loc[minus_mask, "start"].values
            elif anchor == "midpoint":
                anchor_pos = (
                    (features_df["start"].values + features_df["end"].values) // 2
                ).astype(int)
            else:
                raise ValueError(f"Unknown anchor: {anchor}")

            if upstream is not None or downstream is not None:
                left = upstream if upstream is not None else 0
                right = downstream if downstream is not None else 0
                features_df["start"] = anchor_pos - left
                features_df["end"] = anchor_pos + right
                window_upstream = left
                window_downstream = right
            elif fixed_width is not None:
                half_width = fixed_width // 2
                features_df["start"] = anchor_pos - half_width
                features_df["end"] = anchor_pos + (fixed_width - half_width)
                window_upstream = half_width
                window_downstream = fixed_width - half_width
            else:
                window_upstream = None
                window_downstream = None

        # Convert to 1-based intervals (vectorised — avoids slow iterrows)
        intervals = list(
            zip(
                features_df["chrom"].tolist(),
                (features_df["start"].values + 1).tolist(),
                features_df["end"].values.tolist(),
            )
        )

        # Extract signal into bins
        signal_array = extract_signal_into_bins(intervals, self, array_key, bin_size, samples)

        # Create DataArray
        n_intervals, n_bins, n_samples = signal_array.shape
        interval_ids = np.arange(n_intervals)
        if window_upstream is not None:
            bin_ids = np.arange(n_bins, dtype=np.int64) * bin_size - int(window_upstream)
        else:
            bin_ids = np.arange(n_bins, dtype=np.int64)

        strand_values = (
            strands if strands is not None else np.array(["+"] * n_intervals, dtype=object)
        )

        if flip_strand and strands is not None:
            minus_mask = strand_values == "-"
            if np.any(minus_mask):
                signal_array = signal_array.copy()
                signal_array[minus_mask] = signal_array[minus_mask, ::-1, :]

        da = xr.DataArray(
            signal_array,
            dims=("interval", "bin", "sample"),
            coords={
                "interval": interval_ids,
                "bin": bin_ids,
                "sample": samples,
                "strand": ("interval", strand_values),
            },
            attrs={
                "upstream": window_upstream,
                "downstream": window_downstream,
                "anchor": anchor,
                "bin_size": bin_size,
                "strand_flipped": bool(flip_strand and strands is not None),
            },
        )

        return da

    # ------------------------------------------------------------------
    # Peak calling
    # ------------------------------------------------------------------

    def call_peaks(
        self,
        output_dir: "str | Path",
        method: "str | Sequence[str] | None" = None,
        assay: "str | Sequence[str] | None" = None,
        **kwargs,
    ) -> "dict[str, Path] | dict[str, dict[str, Path]]":
        """Call peaks for all completed samples in the dataset.

        Parameters
        ----------
        output_dir : str or Path
            Directory where per-sample BED files are written.
        method : {"quantile", "seacr", "lanceotron"} or sequence, optional
            Peak-calling algorithm. Pass one method or a list of methods.
            When omitted, one method is auto-selected from the biological assay
            type:

            * ATAC or unknown → ``"quantile"``
            * CUT&TAG / CUT&RUN → ``"seacr"``
            * ChIP → ``"lanceotron"``
        assay : str, optional
            Zarr array key to call peaks on (e.g. ``"coverage"``).
            Defaults to ``"coverage"`` when present in the store, otherwise
            the first array key.
        **kwargs
            Passed through to the underlying caller.  Common options:

            * ``blacklist_file`` – BED path of excluded regions (all methods)
            * ``quantile`` – threshold quantile (quantile method)
            * ``fdr_threshold`` – FDR cutoff (seacr method)
            * ``score_threshold`` – minimum classification score (lanceotron)
            * ``n_workers`` – parallel workers (seacr / lanceotron)

        Returns
        -------
        dict[str, Path] or dict[str, dict[str, Path]]
            For one method: sample name → path to the output BED file.
            For multiple methods: method name → (sample name → path).

        Examples
        --------
        >>> qn.available_peak_methods
        ['quantile', 'seacr', 'lanceotron']
        >>> beds = qn.subset(assay="ATAC").call_peaks("peaks/atac/")
        >>> beds = qn.subset(assay="CUT&TAG").call_peaks("peaks/cat/", method="seacr")
        >>> beds_by_method = qn.call_peaks("peaks/", method=["quantile", "lanceotron"], assay=["ATAC", "CHIP"])
        """
        from pathlib import Path as _Path

        from ..peak_calling.call_lanceotron_peaks import call_lanceotron_peaks_from_zarr
        from ..peak_calling.call_quantile_peaks import call_quantile_peaks_from_zarr
        from ..peak_calling.call_seacr_peaks import call_seacr_peaks_from_zarr

        output_dir = _Path(output_dir)
        methods = (
            [method] if isinstance(method, str) or method is None else [str(m) for m in method]
        )

        # Auto-select calling method from biological assay types
        if method is None:
            bio_assays = {a.upper() for a in self.assays}
            if bio_assays & {"CUT&TAG", "CUTTAG", "CAT", "CUT&RUN", "CUTRUN"}:
                methods = ["seacr"]
            elif bio_assays & {"CHIP"}:
                methods = ["lanceotron"]
            else:
                methods = ["quantile"]
            logger.info(
                f"Auto-selected peak-calling method '{methods[0]}' for assays {sorted(bio_assays)}"
            )

        def _resolve_peak_tasks(
            assay_value: "str | Sequence[str] | None",
        ) -> "list[tuple[str, list[str] | None, str]]":
            array_keys = self.array_keys
            if assay_value is None:
                default_key = (
                    "coverage"
                    if "coverage" in array_keys
                    else (array_keys[0] if array_keys else None)
                )
                return [(default_key, None, default_key)] if default_key is not None else []

            requested = [assay_value] if isinstance(assay_value, str) else list(assay_value)
            resolved: list[tuple[str, list[str] | None, str]] = []
            assay_by_sample = dict(zip(self.sample_names, self._get_assay_per_sample()))
            key_to_samples: dict[str, list[str]]
            if self._combined:
                key_to_samples = {
                    str(k): [str(s) for s in v]
                    for k, v in dict(self._combined_root.attrs.get("key_to_samples", {})).items()
                }
            else:
                key_to_samples = {}
                for store in self._stores:
                    samples_for_store = (
                        [f"{store.sample}_{vp}" for vp in store.viewpoints]
                        if store.viewpoints
                        else [store.sample]
                    )
                    for key in store.array_keys():
                        key_to_samples.setdefault(key, []).extend(samples_for_store)

            for raw in requested:
                value = str(raw)
                matched_key = next((k for k in array_keys if k.lower() == value.lower()), None)
                if matched_key is not None:
                    samples_for_key = key_to_samples.get(matched_key)
                    task = (matched_key, samples_for_key, matched_key)
                    if task not in resolved:
                        resolved.append(task)
                    continue

                assay_upper = value.upper()
                assay_samples = self._resolve_samples(assay=assay_upper)
                explicit_keys: list[str] = []
                for key in array_keys:
                    samples_for_key = key_to_samples.get(key, [])
                    if any(
                        assay_by_sample.get(sample, "").upper() == assay_upper
                        for sample in samples_for_key
                    ):
                        explicit_keys.append(key)

                if explicit_keys:
                    for key in explicit_keys:
                        samples_for_key = [
                            sample
                            for sample in key_to_samples.get(key, [])
                            if assay_by_sample.get(sample, "").upper() == assay_upper
                        ]
                        label = assay_upper.lower() if key == "coverage" else key
                        task = (key, samples_for_key or None, label)
                        if task not in resolved:
                            resolved.append(task)
                elif assay_samples:
                    coverage_key = "coverage" if "coverage" in array_keys else None
                    if coverage_key is not None:
                        label = assay_upper.lower()
                        task = (coverage_key, assay_samples, label)
                        if task not in resolved:
                            resolved.append(task)

            if not resolved:
                raise ValueError(
                    f"Could not resolve assay={assay_value!r} to any peak-callable array keys. "
                    f"Available assays: {self.assays}; available array keys: {self.array_keys}"
                )
            return resolved

        _dispatch = {
            "quantile": call_quantile_peaks_from_zarr,
            "seacr": call_seacr_peaks_from_zarr,
            "lanceotron": call_lanceotron_peaks_from_zarr,
        }
        unknown = [m for m in methods if m not in _dispatch]
        if unknown:
            raise ValueError(f"Unknown method(s) {unknown!r}. Choose from: {sorted(_dispatch)}")

        tasks = _resolve_peak_tasks(assay)
        multi_key = len(tasks) > 1
        multi_method = len(methods) > 1
        results: dict[str, dict[str, Path]] = {}
        for selected_method in methods:
            bed_paths: list[str] = []
            for array_key, selected_samples, label in tasks:
                base_output_dir = output_dir / selected_method if multi_method else output_dir
                key_output_dir = base_output_dir / label if multi_key else base_output_dir
                bed_paths.extend(
                    _dispatch[selected_method](
                        self.path,
                        key_output_dir,
                        assay=array_key,
                        samples=selected_samples,
                        **kwargs,
                    )
                )
            results[selected_method] = {_Path(p).stem: _Path(p) for p in bed_paths}

        return results[methods[0]] if len(methods) == 1 else results

    @property
    def available_peak_methods(self) -> list[str]:
        """Peak-calling methods supported by :meth:`call_peaks`."""
        return ["quantile", "seacr", "lanceotron"]

    def peak_overlap(
        self,
        peak_sets: "dict[str, str | Path]",
    ) -> "pd.DataFrame":
        """Compute overlap statistics across 2–4 peak sets.

        Parameters
        ----------
        peak_sets : dict[str, str or Path]
            Label → BED file path.

        Returns
        -------
        pd.DataFrame
            Rows are exclusive Venn regions; columns are ``combination``,
            ``n_peaks``, and ``pct_of_first``.

        Examples
        --------
        >>> counts = qn.peak_overlap({
        ...     "ATAC":    "peaks/atac/ATAC-SEM-1.bed",
        ...     "CUT&TAG": "peaks/cat/CAT-HSC_H3K27ac.bed",
        ... })
        """
        from .peaks import overlap_peaks

        return overlap_peaks(peak_sets)

    def venn_peaks(
        self,
        peak_sets: "dict[str, str | Path]",
        ax: "plt.axes.Axes | None" = None,
        title: "str | None" = None,
        colors: "list[str] | None" = None,
        alpha: float = 0.50,
        figsize: "tuple[float, float]" = (5.5, 5.5),
    ) -> "plt.axes.Axes":
        """Venn diagram for 2 or 3 peak sets.

        Parameters
        ----------
        peak_sets : dict[str, str or Path]
            Label → BED file path.  Must have exactly 2 or 3 entries.
        ax : matplotlib.axes.Axes, optional
        title : str, optional
        colors : list of str, optional
        alpha : float
        figsize : tuple

        Returns
        -------
        matplotlib.axes.Axes

        Examples
        --------
        >>> ax = qn.venn_peaks(
        ...     {"ATAC": "peaks/atac/SEM-1.bed", "ChIP": "peaks/chip/SEM-1.bed"},
        ...     title="Peak overlap",
        ... )
        """
        from .peaks import venn_plot

        return venn_plot(
            peak_sets,
            ax=ax,
            title=title,
            colors=colors,
            alpha=alpha,
            figsize=figsize,
        )

    # ------------------------------------------------------------------
    # Combine
    # ------------------------------------------------------------------

    @classmethod
    def combine(
        cls,
        src: Path | str,
        output: Path | str,
        overwrite: bool = True,
        n_workers: int = 1,
    ) -> "QuantNadoDataset":
        """Combine a directory of per-sample zarrs into a single multi-sample zarr.

        Only ``completed`` stores are included.  Same-assay arrays are stacked
        along axis 0: ``(1, chrom_len) × N → (N, chrom_len)``.

        Parameters
        ----------
        src:
            Directory containing per-sample ``.zarr`` stores.
        output:
            Path for the combined ``.zarr`` output.
        overwrite:
            Delete ``output`` if it already exists.
        n_workers:
            Number of thread workers for row-copy tasks. ``1`` preserves the
            previous serial behaviour.
        """
        from zarr.core.array_spec import ArrayConfig
        from zarr.storage import LocalStore

        n_workers = max(1, int(n_workers))
        combine_start = time.perf_counter()
        src_ds = cls(src)
        if src_ds._combined:
            raise ValueError("src is already a combined store")

        output_path = Path(output)
        if overwrite and output_path.exists():
            import shutil

            shutil.rmtree(output_path) if output_path.is_dir() else output_path.unlink()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        out_root = zarr.group(store=LocalStore(str(output_path)), overwrite=True, zarr_format=3)
        write_config = ArrayConfig(order="C", write_empty_chunks=True)

        # Collect metadata across all stores
        all_samples: list[str] = src_ds.sample_names
        all_array_keys: list[str] = [
            key for key in src_ds.array_keys if key not in _COVERAGE_COLLAPSE_KEYS
        ]
        all_assay_types: list[str] = src_ds.assays  # biological types e.g. ['atac', 'meth', 'rna']

        # Group stores by assay key; also build key→sample-names mapping for sel()
        key_to_stores: dict[str, list[_PerSampleStore]] = {}
        keys_by_store = {id(store): store.array_keys() for store in src_ds._stores}
        for store in src_ds._stores:
            for key in keys_by_store[id(store)]:
                key_to_stores.setdefault(key, []).append(store)

        key_to_samples: dict[str, list[str]] = {}
        for key, stores in key_to_stores.items():
            names: list[str] = []
            for store in stores:
                if store.viewpoints and key.startswith("viewpoint_"):
                    vp = key[len("viewpoint_") :]
                    names.append(f"{store.sample}_{vp}")
                else:
                    names.append(store.sample)
            key_to_samples[key] = names
        # Unified coverage spans all samples regardless of assay
        key_to_samples["coverage"] = all_samples
        if "coverage" not in all_array_keys:
            all_array_keys = sorted(set(all_array_keys) | {"coverage"})

        chromsizes = src_ds.chromsizes
        chunk_len = max(1, src_ds._stores[0].chunk_len if src_ds._stores else 65536)
        logger.info(
            f"Combining {len(src_ds._stores)} store(s), {len(all_samples)} sample row(s), "
            f"{len(chromsizes)} chromosome(s) -> {output_path} "
            f"(workers={n_workers}, chunk_len={chunk_len:,})"
        )

        for chrom, chrom_len in chromsizes.items():
            chrom_start = time.perf_counter()
            logger.info(f"Combining chromosome {chrom} ({chrom_len:,} bp)")
            grp = out_root.require_group(chrom)
            for key, stores in key_to_stores.items():
                if key == "coverage" or key in _COVERAGE_COLLAPSE_KEYS:
                    continue  # written below as unified across all samples
                present_stores = [store for store in stores if chrom in store.chromosomes]
                if not present_stores:
                    continue
                n = len(present_stores)
                first_arr = present_stores[0].get_array(chrom, key)
                first_dtype = first_arr.dtype
                fill = np.nan if np.issubdtype(first_dtype, np.floating) else 0
                out_arr = grp.create_array(
                    key,
                    shape=(n, chrom_len),
                    chunks=(1, chunk_len),
                    dtype=first_dtype,
                    fill_value=fill,
                    config=write_config,
                    overwrite=True,
                )
                key_start = time.perf_counter()
                tasks = [
                    partial(
                        _copy_array_row_chunks,
                        store.get_array(chrom, key),
                        out_arr,
                        row_idx,
                        chrom_len,
                        chunk_len,
                    )
                    for row_idx, store in enumerate(present_stores)
                ]
                bytes_written = _run_copy_tasks(tasks, n_workers)
                elapsed = time.perf_counter() - key_start
                logger.info(
                    f"Combined {chrom}:{key} ({n} row(s)) in {elapsed:.1f}s "
                    f"({_format_rate(bytes_written, elapsed)})"
                )

            # Unified coverage: one row per sample, using primary signal per assay.
            # METH → coverage, RNA → rna_fwd + rna_rev, SNP → DP,
            # ATAC/ChIP/CUT&TAG → first key, MCC → viewpoint_{vp} per viewpoint.
            cov_arr = grp.create_array(
                "coverage",
                shape=(len(all_samples), chrom_len),
                chunks=(1, chunk_len),
                dtype=np.float32,
                fill_value=0.0,
                config=write_config,
                overwrite=True,
            )
            cov_start = time.perf_counter()
            cov_tasks = []
            cov_row_idx = 0
            for store in src_ds._stores:
                keys = keys_by_store[id(store)]
                keys_set = set(keys)
                missing_chrom = chrom not in store.chromosomes
                if store.viewpoints:
                    for vp in store.viewpoints:
                        vp_key = f"viewpoint_{vp}"
                        if not missing_chrom and vp_key in keys_set:
                            cov_tasks.append(
                                partial(
                                    _copy_sum_array_row_chunks,
                                    (store.get_array(chrom, vp_key),),
                                    cov_arr,
                                    cov_row_idx,
                                    chrom_len,
                                    chunk_len,
                                )
                            )
                        cov_row_idx += 1
                elif missing_chrom:
                    cov_row_idx += 1
                elif "coverage" in keys_set:
                    cov_tasks.append(
                        partial(
                            _copy_sum_array_row_chunks,
                            (store.get_array(chrom, "coverage"),),
                            cov_arr,
                            cov_row_idx,
                            chrom_len,
                            chunk_len,
                        )
                    )
                    cov_row_idx += 1
                elif "rna_fwd" in keys_set:
                    src_arrays = [store.get_array(chrom, "rna_fwd")]
                    if "rna_rev" in keys_set:
                        src_arrays.append(store.get_array(chrom, "rna_rev"))
                    cov_tasks.append(
                        partial(
                            _copy_sum_array_row_chunks,
                            tuple(src_arrays),
                            cov_arr,
                            cov_row_idx,
                            chrom_len,
                            chunk_len,
                        )
                    )
                    cov_row_idx += 1
                elif "DP" in keys_set:
                    cov_tasks.append(
                        partial(
                            _copy_sum_array_row_chunks,
                            (store.get_array(chrom, "DP"),),
                            cov_arr,
                            cov_row_idx,
                            chrom_len,
                            chunk_len,
                        )
                    )
                    cov_row_idx += 1
                else:
                    first_key = keys[0]
                    cov_tasks.append(
                        partial(
                            _copy_sum_array_row_chunks,
                            (store.get_array(chrom, first_key),),
                            cov_arr,
                            cov_row_idx,
                            chrom_len,
                            chunk_len,
                        )
                    )
                    cov_row_idx += 1
            bytes_written = _run_copy_tasks(cov_tasks, n_workers)
            elapsed = time.perf_counter() - cov_start
            logger.info(
                f"Combined {chrom}:coverage ({len(all_samples)} row(s)) in {elapsed:.1f}s "
                f"({_format_rate(bytes_written, elapsed)})"
            )
            logger.info(f"Finished chromosome {chrom} in {time.perf_counter() - chrom_start:.1f}s")

        # Write combined metadata
        meta_grp = out_root.require_group("metadata")
        sn_arr = meta_grp.require_array(
            "sample_names", shape=(len(all_samples),), dtype="str", overwrite=True
        )
        sn_arr[:] = all_samples
        # Expand per-store metadata to per-sample (MCC stores contribute N viewpoints each)
        completed_list: list[bool] = []
        total_reads_list: list[int] = []
        assay_list: list[str] = []
        ip_list: list[str] = []
        mean_rl_list: list[float] = []
        sparsity_list: list[float] = []
        stranded_list: list[str] = []
        for s in src_ds._stores:
            n = len(s.viewpoints) if s.viewpoints else 1
            completed_list.extend([s.completed] * n)
            total_reads_list.extend([s.total_reads] * n)
            assay_list.extend([s.assay.upper()] * n)
            ip_list.extend([str(s.ip or "")] * n)
            mean_rl_list.extend([s.mean_read_length] * n)
            sparsity_list.extend([s.sparsity] * n)
            stranded_list.extend([str(s.stranded or "")] * n)
        completed = np.array(completed_list, dtype=bool)
        meta_grp.require_array("completed", shape=(len(all_samples),), dtype=bool, overwrite=True)
        meta_grp["completed"][:] = completed
        meta_grp.require_array(
            "total_reads", shape=(len(all_samples),), dtype=np.int64, overwrite=True
        )
        meta_grp["total_reads"][:] = np.array(total_reads_list, dtype=np.int64)
        assay_arr = meta_grp.require_array(
            "assay", shape=(len(all_samples),), dtype="str", overwrite=True
        )
        assay_arr[:] = assay_list
        ip_arr = meta_grp.require_array(
            "ip", shape=(len(all_samples),), dtype="str", overwrite=True
        )
        ip_arr[:] = ip_list
        meta_grp.require_array(
            "mean_read_length", shape=(len(all_samples),), dtype=np.float32, overwrite=True
        )
        meta_grp["mean_read_length"][:] = np.array(mean_rl_list, dtype=np.float32)
        meta_grp.require_array(
            "sparsity", shape=(len(all_samples),), dtype=np.float32, overwrite=True
        )
        meta_grp["sparsity"][:] = np.array(sparsity_list, dtype=np.float32)
        stranded_arr = meta_grp.require_array(
            "stranded", shape=(len(all_samples),), dtype="str", overwrite=True
        )
        stranded_arr[:] = stranded_list

        out_root.attrs.update(
            {
                "assay_types": all_assay_types,
                "array_keys": all_array_keys,
                "sample_names": all_samples,
                "key_to_samples": key_to_samples,
                "chromsizes": chromsizes,
                "chunk_len": chunk_len,
            }
        )

        zarr.consolidate_metadata(str(output_path))
        logger.info(
            f"Combined {len(src_ds._stores)} stores -> {output_path} "
            f"in {time.perf_counter() - combine_start:.1f}s"
        )
        return cls(output_path)

    # ------------------------------------------------------------------
    # Uniform analysis API helpers
    # ------------------------------------------------------------------

    def _resolve_samples(
        self,
        assay: "str | Sequence[str] | None" = None,
        samples: "str | Sequence[str] | None" = None,
        ip: "str | Sequence[str] | None" = None,
        group: "str | Sequence[str] | dict[str, str | Sequence[str]] | None" = None,
    ) -> "list[str]":
        """Return sample names after applying assay/sample/IP/group filters."""

        def _as_list(value):
            if value is None:
                return None
            if isinstance(value, str):
                return [value]
            if isinstance(value, Sequence):
                return [str(v) for v in value]
            return [str(value)]

        sample_list = _as_list(samples)
        assay_list = _as_list(assay)
        ip_list = _as_list(ip)
        resolved = list(self.sample_names)

        if sample_list is not None:
            requested = [s for s in sample_list if s in self.sample_names]
            if not requested:
                raise ValueError(f"No requested samples found. Available: {self.sample_names}")
            requested_set = set(requested)
            resolved = [s for s in resolved if s in requested_set]

        if assay_list is not None:
            assay_upper = {QuantNadoDataset._canonicalise_assay_name(a) for a in assay_list}
            assay_per_sample = dict(zip(self.sample_names, self._get_assay_per_sample()))
            resolved = [
                s
                for s in resolved
                if QuantNadoDataset._canonicalise_assay_name(assay_per_sample.get(s, ""))
                in assay_upper
            ]
            if not resolved:
                raise ValueError(
                    f"No samples found for assay='{assay}'. Available assays: {self.assays}"
                )

        if ip_list is not None:
            ip_upper = {v.upper() for v in ip_list}
            ip_per_sample = dict(zip(self.sample_names, self._get_ip_per_sample()))
            resolved = [s for s in resolved if ip_per_sample.get(s, "").upper() in ip_upper]
            if not resolved:
                available = sorted({v for v in self._get_ip_per_sample() if str(v).strip()})
                raise ValueError(f"No samples found for ip='{ip}'. Available IPs: {available}")

        if group is not None:
            if isinstance(group, dict):
                group_requests = {
                    str(name): _as_list(labels) or [] for name, labels in group.items()
                }
            else:
                if self._last_group_name is None:
                    raise ValueError(
                        "No cached group namespace found. Call qn.group_by(...) first, or pass group={'name': 'label'}."
                    )
                group_requests = {self._last_group_name: _as_list(group) or []}

            for group_name, group_labels in group_requests.items():
                group_set = self._group_sets.get(group_name)
                if group_set is None:
                    raise ValueError(
                        f"Unknown group set '{group_name}'. Available group sets: {sorted(self._group_sets)}"
                    )
                unknown = [label for label in group_labels if label not in group_set]
                if unknown:
                    raise ValueError(
                        f"Unknown group(s) for '{group_name}': {unknown}. Available: {sorted(group_set)}"
                    )
                allowed = {sample for label in group_labels for sample in group_set.get(label, [])}
                before = list(resolved)
                resolved = [s for s in resolved if s in allowed]
                if not resolved:
                    available_labels = sorted(group_set)
                    available_from_current = sorted(
                        set(before)
                        & {sample for samples in group_set.values() for sample in samples}
                    )
                    raise ValueError(
                        "No samples found after applying group filter "
                        f"'{group_name}={group_labels}'. "
                        f"Samples remaining before this step: {before}. "
                        f"Available labels for '{group_name}': {available_labels}. "
                        f"Samples in this group namespace that overlap the current filters: {available_from_current}."
                    )

        return resolved

    def _resolve_modalities(
        self,
        modality: "str | Sequence[str] | None" = None,
        *,
        allow_multiple: bool = False,
        default: "str | None" = None,
    ):
        """Normalise modality selection to one string or a list of strings."""
        if modality is None:
            return [] if allow_multiple and default is None else default
        if isinstance(modality, str):
            modalities = [modality]
        elif isinstance(modality, Sequence):
            modalities = [str(m) for m in modality]
        else:
            modalities = [str(modality)]
        if not allow_multiple and len(modalities) != 1:
            raise ValueError(
                "This method accepts exactly one modality; pass a single string or a single-item list."
            )
        return modalities if allow_multiple else modalities[0]

    def _sample_indices(self, sample_list: "list[str]") -> "np.ndarray":
        """Map sample names to integer indices in ``self.sample_names``."""
        idx_map = {s: i for i, s in enumerate(self.sample_names)}
        return np.array([idx_map[s] for s in sample_list if s in idx_map], dtype=np.int64)

    def _filter_sample_data(self, data, sample_list: "list[str]"):
        """Filter xarray/pandas data structures to the requested samples."""
        if hasattr(data, "coords") and "sample" in data.coords:
            keep = [s for s in sample_list if s in data.coords["sample"].values]
            return data.sel(sample=keep)
        if hasattr(data, "columns"):
            keep = [s for s in sample_list if s in data.columns]
            return data.loc[:, keep]
        return data

    # ------------------------------------------------------------------
    # Reduction / signal aggregation
    # ------------------------------------------------------------------

    def reduce(
        self,
        intervals_path: "str | None" = None,
        ranges_df=None,
        gtf_path: "str | None" = None,
        feature_type: "str | None" = None,
        reduction: str = "mean",
        assay: "str | Sequence[str] | None" = None,
        samples: "str | Sequence[str] | None" = None,
        modality: "str | Sequence[str] | None" = None,
        progress: bool = False,
        workers: int | None = None,
        **kwargs,
    ):
        """Reduce signal over genomic intervals.

        Parameters
        ----------
        intervals_path:
            Path to a BED or GTF file.
        ranges_df:
            Pre-parsed ranges DataFrame / PyRanges.
        gtf_path:
            GTF file path (used with *feature_type*).
        feature_type:
            Feature type (e.g. ``"gene"``, ``"promoter"``).
        reduction:
            One of ``"mean"``, ``"sum"``, ``"max"``, ``"min"``, ``"median"``.
        assay:
            Restrict to samples of this assay type.
        samples:
            Explicit sample names (overrides *assay*).
        modality:
            Zarr array key (e.g. ``"atac"``, ``"rna_fwd"``).
        progress:
            Show a tqdm progress bar over reduce read batches.
        workers:
            Number of chromosome/strand work items to reduce concurrently.

        Returns
        -------
        xr.Dataset
        """
        from .reduce import reduce_byranges_signal

        resolved = self._resolve_samples(assay=assay, samples=samples)
        indices = self._sample_indices(resolved)

        return reduce_byranges_signal(
            self,
            ranges_df=ranges_df,
            intervals_path=intervals_path,
            feature_type=feature_type,
            gtf_path=gtf_path,
            reduction=reduction,
            sample_indices=indices if len(indices) < len(self.sample_names) else None,
            array_key=self._resolve_modalities(modality) if modality is not None else None,
            progress=progress,
            workers=workers,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Feature counting
    # ------------------------------------------------------------------

    def count_features(
        self,
        gtf_file: "str | None" = None,
        bed_file: "str | None" = None,
        ranges_df=None,
        feature_type: str = "gene",
        engine: str = "signal",
        assay: "str | Sequence[str] | None" = None,
        samples: "str | Sequence[str] | None" = None,
        modality: "str | Sequence[str] | None" = None,
        **kwargs,
    ):
        """Count or quantify genomic features into a feature-by-sample matrix.

        Parameters
        ----------
        gtf_file:
            Path to GTF file.
        bed_file:
            Path to BED file.
        ranges_df:
            Pre-parsed ranges DataFrame.
        feature_type:
            GTF feature level (default ``"gene"``).
            Combine with ``feature_id_attr`` in ``kwargs`` for featureCounts-like
            ``-t/-g`` semantics, for example ``feature_type="exon"``,
            ``feature_id_attr="gene_name"``.
        engine:
            Counting backend.

            - ``"signal"`` (default): quantify stored QuantNado signal over features.
              This is coverage/signal-derived summarisation and is useful for
              exploratory matrices, clustering, and general signal analysis.
            - ``"bam"``: reserved for future BAM-backed, featureCounts-style
              read/fragement assignment. Not implemented yet.
        assay:
            Restrict to samples of this assay type.
        samples:
            Explicit sample names (overrides *assay*).
        modality:
            Zarr array key hint (e.g. ``"rna_fwd"``).

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            (counts_df, feature_metadata)
        """
        engine = engine.lower()
        if engine != "signal":
            if engine == "bam":
                raise NotImplementedError(
                    "count_features(engine='bam') is not implemented yet. "
                    "Use quantify_signal(...) or count_features(engine='signal') "
                    "for stored-signal quantification today."
                )
            raise ValueError("engine must be either 'signal' or 'bam'")

        from .counts import count_features as _count_features

        resolved = self._resolve_samples(assay=assay, samples=samples)
        return _count_features(
            self,
            ranges_df=ranges_df,
            bed_file=bed_file,
            gtf_file=gtf_file,
            feature_type=feature_type,
            samples=resolved if len(resolved) < len(self.sample_names) else None,
            modality=self._resolve_modalities(modality) if modality is not None else None,
            **kwargs,
        )

    def quantify_signal(
        self,
        gtf_file: "str | None" = None,
        bed_file: "str | None" = None,
        ranges_df=None,
        feature_type: str = "gene",
        assay: "str | Sequence[str] | None" = None,
        samples: "str | Sequence[str] | None" = None,
        modality: "str | Sequence[str] | None" = None,
        return_metadata: bool = True,
        **kwargs,
    ):
        """Quantify stored signal over genomic features.

        This is the explicit signal-based alternative to BAM-backed counting.
        Internally it reuses the current ``count_features(engine="signal")``
        implementation and is intended for exploratory analysis, clustering,
        PCA, and assay-agnostic feature summarisation.

        Parameters
        ----------
        gtf_file, bed_file, ranges_df, feature_type:
            Same feature selection inputs accepted by :meth:`count_features`.
        assay:
            Restrict to samples of this assay type.
        samples:
            Explicit sample names (overrides *assay*).
        modality:
            Concrete zarr array key to quantify, for example ``"coverage"``
            or ``"rna_fwd"``.
        return_metadata:
            If True (default), return ``(matrix, feature_metadata)``.
            If False, return only the quantified matrix.

        Returns
        -------
        pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]
            Feature-by-sample quantified matrix, optionally with aligned
            feature metadata.
        """
        kwargs.setdefault("integerize", False)
        matrix, feature_metadata = self.count_features(
            gtf_file=gtf_file,
            bed_file=bed_file,
            ranges_df=ranges_df,
            feature_type=feature_type,
            engine="signal",
            assay=assay,
            samples=samples,
            modality=modality,
            **kwargs,
        )
        return (matrix, feature_metadata) if return_metadata else matrix

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------

    def normalise(
        self,
        data=None,
        method: str = "cpm",
        assay: "str | Sequence[str] | None" = None,
        samples: "str | Sequence[str] | None" = None,
        library_sizes=None,
        feature_lengths=None,
    ):
        """Normalise coverage signal or feature counts.

        Parameters
        ----------
        data:
            xr.Dataset, xr.DataArray, or pd.DataFrame.
        method:
            ``"cpm"``, ``"rpkm"``, or ``"tpm"``.
        assay:
            Pre-filter data to samples of this assay type.
        samples:
            Explicit sample names (overrides *assay*).
        library_sizes:
            Total mapped reads per sample; auto-read from store if omitted.
        feature_lengths:
            Required for ``"rpkm"`` / ``"tpm"`` on DataFrames.
        """
        from .normalise import normalise as _normalise

        if data is None:
            return NormalisedQuantNadoDataset(
                self,
                method=method,
                library_sizes=library_sizes,
                feature_lengths=feature_lengths,
            )

        if assay is not None or samples is not None:
            resolved = self._resolve_samples(assay=assay, samples=samples)
            data = self._filter_sample_data(data, resolved)

        return _normalise(
            data,
            self,
            method=method,
            library_sizes=library_sizes,
            feature_lengths=feature_lengths,
        )

    def library_sizes(
        self,
        assay: "str | Sequence[str] | None" = None,
        samples: "str | Sequence[str] | None" = None,
    ):
        """Return total mapped reads per sample as a pd.Series.

        Parameters
        ----------
        assay:
            Restrict to samples of this assay type.
        samples:
            Explicit sample names (overrides *assay*).
        """
        from .normalise import get_library_sizes

        sizes = get_library_sizes(self)
        if assay is not None or samples is not None:
            resolved = self._resolve_samples(assay=assay, samples=samples)
            sizes = sizes.reindex(resolved)
        return sizes

    # ------------------------------------------------------------------
    # PCA
    # ------------------------------------------------------------------

    def pca(
        self,
        data_or_query=None,
        n_components: int = 5,
        assay: "str | Sequence[str] | None" = None,
        samples: "str | Sequence[str] | None" = None,
        modality: "str | Sequence[str] | None" = None,
        chromosome: "str | None" = None,
        nan_handling_strategy: str = "drop",
        standardize: bool = False,
        random_state: "int | None" = None,
        subset_size: "int | None" = None,
        subset_strategy: str = "random",
    ):
        """Run PCA on reduced genomic signal.

        Parameters
        ----------
        data_or_query:
            Either a 2D DataArray, a chromosome name string, or ``None``.
            When a chromosome or ``None`` is provided, data are auto-extracted
            from :meth:`sel` using ``modality`` (default ``"coverage"``).
        n_components:
            Number of principal components.
        assay:
            Pre-filter samples before PCA.
        samples:
            Explicit sample names (overrides *assay*).

        Returns
        -------
        tuple[PCA, xr.DataArray]
        """
        from .pca import run_pca as _run_pca

        pca_chromosome = chromosome

        if isinstance(data_or_query, xr.DataArray):
            data = data_or_query
            if assay is not None or samples is not None:
                data = self._filter_sample_data(
                    data, self._resolve_samples(assay=assay, samples=samples)
                )
            if not any(coord in data.coords for coord in ("contig", "chrom")):
                pca_chromosome = None
        elif isinstance(data_or_query, xr.Dataset):
            data = data_or_query
            if assay is not None or samples is not None:
                data = self._filter_sample_data(
                    data, self._resolve_samples(assay=assay, samples=samples)
                )
            resolved_modality = self._resolve_modalities(modality, default="coverage")
            if resolved_modality not in data.data_vars:
                raise ValueError(
                    f"Modality '{resolved_modality}' not found in dataset input. "
                    f"Available: {list(data.data_vars)}"
                )
            data = data[resolved_modality]
            if not any(coord in data.coords for coord in ("contig", "chrom")):
                pca_chromosome = None
        else:
            query_chrom = data_or_query if isinstance(data_or_query, str) else chromosome
            if query_chrom is None:
                raise ValueError("Provide a DataArray, a chromosome name, or set chromosome=...")
            selected = self.sel(query_chrom, assay=assay, samples=samples)
            resolved_modality = self._resolve_modalities(modality, default="coverage")
            if resolved_modality not in selected.data_vars:
                raise ValueError(
                    f"Modality '{resolved_modality}' not found for chromosome '{query_chrom}'. "
                    f"Available: {list(selected.data_vars)}"
                )
            data = selected[resolved_modality]

        return _run_pca(
            data,
            n_components=n_components,
            chromosome=pca_chromosome,
            nan_handling_strategy=nan_handling_strategy,
            standardize=standardize,
            random_state=random_state,
            subset_size=subset_size,
            subset_strategy=subset_strategy,
        )

    def pca_scree(self, pca_obj, **kwargs):
        """Plot PCA scree (explained variance)."""
        from .pca import plot_pca_scree

        return plot_pca_scree(pca_obj, **kwargs)

    def pca_scatter(self, pca_obj, pca_result, colour_by=None, shape_by=None, **kwargs):
        """Scatter plot of PCA-transformed samples."""
        from .pca import plot_pca_scatter

        return plot_pca_scatter(
            pca_obj, pca_result, colour_by=colour_by, shape_by=shape_by, **kwargs
        )

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def metaplot(
        self,
        data,
        data_rev=None,
        *,
        assay: "str | Sequence[str] | None" = None,
        samples: "str | Sequence[str] | None" = None,
        modality: "str | Sequence[str] | None" = None,
        **kwargs,
    ):
        """Plot a metagene profile. See :func:`quantnado.analysis.plot.metaplot`."""
        from .plot import metaplot as _metaplot

        if assay is not None or samples is not None:
            resolved = self._resolve_samples(assay=assay, samples=samples)
            data = self._filter_sample_data(data, resolved)
            if data_rev is not None:
                data_rev = self._filter_sample_data(data_rev, resolved)

        resolved_modality = self._resolve_modalities(modality) if modality is not None else None
        return _metaplot(data, data_rev, modality=resolved_modality, **kwargs)

    def tornadoplot(
        self,
        data,
        data_rev=None,
        *,
        assay: "str | Sequence[str] | None" = None,
        samples: "str | Sequence[str] | None" = None,
        modality: "str | Sequence[str] | None" = None,
        **kwargs,
    ):
        """Tornado / heatmap plot. See :func:`quantnado.analysis.plot.tornadoplot`."""
        from .plot import tornadoplot as _tornadoplot

        if assay is not None or samples is not None:
            data = self._filter_sample_data(
                data, self._resolve_samples(assay=assay, samples=samples)
            )

        resolved_modality = self._resolve_modalities(modality) if modality is not None else None
        return _tornadoplot(data, data_rev, modality=resolved_modality, **kwargs)

    def heatmap(
        self,
        data,
        *,
        assay: "str | Sequence[str] | None" = None,
        samples: "str | Sequence[str] | None" = None,
        exclude_zeros: bool = False,
        zscore: "int | None" = None,
        **kwargs,
    ):
        """Heatmap of reduced signal. See :func:`quantnado.analysis.plot.heatmap`."""
        from .plot import heatmap as _heatmap

        if assay is not None or samples is not None:
            data = self._filter_sample_data(
                data, self._resolve_samples(assay=assay, samples=samples)
            )

        return _heatmap(data, exclude_zeros=exclude_zeros, zscore=zscore, **kwargs)

    def correlate(
        self,
        data,
        *,
        assay: "str | Sequence[str] | None" = None,
        samples: "str | Sequence[str] | None" = None,
        **kwargs,
    ):
        """Compute and plot sample correlation. See :func:`quantnado.analysis.plot.correlate`."""
        from .plot import correlate as _correlate

        if assay is not None or samples is not None:
            data = self._filter_sample_data(
                data, self._resolve_samples(assay=assay, samples=samples)
            )

        return _correlate(data, **kwargs)

    def locus_plot(
        self,
        locus,
        sample_names,
        modality=None,
        assay: "str | Sequence[str] | None" = None,
        **kwargs,
    ):
        """Plot a genomic locus. See :func:`quantnado.analysis.plot.locus_plot`."""
        from .plot import locus_plot

        if assay is not None:
            allowed = set(self._resolve_samples(assay=assay))
            sample_names = [s for s in sample_names if s in allowed]

        if isinstance(sample_names, str):
            sample_names = [sample_names]
        else:
            sample_names = list(sample_names)

        if modality is None:
            assay_by_sample = dict(zip(self.sample_names, self._get_assay_per_sample()))
            modalities = [
                "stranded_coverage"
                if assay_by_sample.get(sample_name, "").upper() == "RNA"
                else "methylation"
                if assay_by_sample.get(sample_name, "").upper() == "METH"
                else "variant"
                if assay_by_sample.get(sample_name, "").upper() == "SNP"
                else "coverage"
                for sample_name in sample_names
            ]
        else:
            modalities = self._resolve_modalities(modality, allow_multiple=True)
        return locus_plot(locus, sample_names=sample_names, modality=modalities, **kwargs)

    # ------------------------------------------------------------------
    # Plotnado integration
    # ------------------------------------------------------------------

    def _primary_array_key_for_sample(self, sample_name: str) -> "str | None":
        """Return the zarr array key that contains data for *sample_name*.

        Uses ``key_to_samples`` (combined store) or per-sample store metadata
        so that ``extract_region`` returns the correct data variable for ChIP /
        CUT&TAG / MCC samples rather than the first alphabetical key.
        """
        if self._combined:
            key_to_samples: dict = dict(
                self._combined_root.attrs.get("key_to_samples", {})  # type: ignore[union-attr]
            )
            for key, names in sorted(
                key_to_samples.items(), key=lambda item: item[0] == "coverage"
            ):
                if sample_name in names:
                    return key
            return None
        for store in self._stores:
            if store.sample == sample_name or (
                store.viewpoints
                and any(f"{store.sample}_{vp}" == sample_name for vp in store.viewpoints)
            ):
                for k in store.array_keys():
                    if k not in _PLOTNADO_COVERAGE_SKIP:
                        return k
        return None

    def extract_region(
        self,
        region: str,
        samples=None,
        array_key: "str | None" = None,
    ) -> "xr.DataArray":
        """Extract a genomic region as an ``xr.DataArray`` for plotnado coverage tracks.

        Parameters
        ----------
        region:
            Genomic region string, e.g. ``"chr1:1000000-1001000"``.
        samples:
            Sample name(s) to include. ``None`` returns all samples.
        array_key:
            Explicit zarr array key (e.g. ``"atac"``, ``"chip_h3k27ac"``).
            When omitted, the key that owns the requested sample is used.
        """
        chrom, start, end = _parse_plotnado_region(region)
        ds = self.sel(chrom, start, end, samples=samples)
        if array_key is not None:
            return ds[array_key]

        # When a single sample is requested, look up its owning key directly
        # so ChIP / CUT&TAG samples don't fall back to the first alphabetical key.
        if samples is not None:
            requested = [samples] if isinstance(samples, str) else list(samples)
            if len(requested) == 1:
                key = self._primary_array_key_for_sample(requested[0])
                if key is not None and key in ds:
                    return ds[key]

        for key in ds.data_vars:
            if key not in _PLOTNADO_COVERAGE_SKIP:
                return ds[key]
        keys = list(ds.data_vars)
        if keys:
            return ds[keys[0]]
        raise KeyError("No array data found in this region")

    @property
    def coverage(self) -> "_PlotnadoCoverageAdapter":
        """Sub-store adapter for plotnado stranded coverage tracks (RNA)."""
        return _PlotnadoCoverageAdapter(self)

    @property
    def methylation(self) -> "_PlotnadoMethylAdapter":
        """Sub-store adapter for plotnado methylation tracks."""
        return _PlotnadoMethylAdapter(self)

    @property
    def variants(self) -> "_PlotnadoVariantsAdapter":
        """Sub-store adapter for plotnado variant tracks."""
        return _PlotnadoVariantsAdapter(self)

    def normalised(
        self,
        method: str = "cpm",
        library_sizes: "pd.Series | dict | None" = None,
    ) -> "NormalisedQuantNadoDataset":
        """Compatibility alias for :meth:`normalise` with ``data=None``."""
        return self.normalise(method=method, library_sizes=library_sizes)

    @property
    def info(self) -> DatasetInfo:
        """Concise dataset summary with notebook-friendly representation."""
        group_summary = {
            name: {
                "labels": list(groups.keys()),
                "n": len(groups),
            }
            for name, groups in self._group_sets.items()
            if groups
        }
        summary = DatasetInfo(
            {
                "assays": list(self.assays),
                "chromosomes": list(self.chromosomes),
                "chromsizes": dict(self.chromsizes),
                "extras": {
                    "layout": "combined" if self._combined else "per-sample",
                    "path": str(self.path),
                    "subset": bool(self._subset_samples is not None),
                    "groups": group_summary,
                },
                "per_assay": {},
            }
        )
        source_path = getattr(self, "source_path", self.path)
        if source_path != self.path:
            summary["extras"]["source_path"] = str(source_path)
        if self._subset_samples is not None:
            summary["extras"]["subset_samples"] = list(self._subset_samples)

        per_assay: dict[str, dict[str, list[str]]] = {}
        for assay in self.assays:
            assay_ds = self.subset(assay=assay)
            ips = []
            if hasattr(assay_ds, "_get_ip_per_sample"):
                ips = [ip for ip in dict.fromkeys(assay_ds._get_ip_per_sample()) if str(ip).strip()]
            assay_summary = {
                "n_samples": len(assay_ds.sample_names),
                "sample_names": list(assay_ds.sample_names),
                "array_keys": list(assay_ds.array_keys),
                "ips": ips,
            }
            per_assay[assay] = assay_summary

        summary["per_assay"] = per_assay
        return summary

    def info_of(self, obj) -> ObjectInfo:
        """Return a compact summary for xarray / pandas objects."""
        if isinstance(obj, xr.DataArray):
            name = obj.name or "<unnamed>"
            return ObjectInfo(
                {
                    "type": "DataArray",
                    "name": name,
                    "dims": list(obj.dims),
                    "sizes": {k: int(v) for k, v in obj.sizes.items()},
                    "dtype": str(obj.dtype),
                    "coords": list(obj.coords),
                }
            )
        if isinstance(obj, xr.Dataset):
            return ObjectInfo(
                {
                    "type": "Dataset",
                    "dims": list(obj.dims),
                    "sizes": {k: int(v) for k, v in obj.sizes.items()},
                    "data_vars": list(obj.data_vars),
                    "coords": list(obj.coords),
                }
            )
        if isinstance(obj, pd.DataFrame):
            return ObjectInfo(
                {
                    "type": "DataFrame",
                    "shape": list(obj.shape),
                    "columns": list(obj.columns),
                    "index_name": obj.index.name,
                }
            )
        raise TypeError(
            "info_of(...) expects an xarray DataArray, xarray Dataset, or pandas DataFrame"
        )


# ---------------------------------------------------------------------------
# Plotnado sub-store adapters (used by QuantNadoDataset.coverage / .methylation / .variants)
# ---------------------------------------------------------------------------

_PLOTNADO_COVERAGE_SKIP = frozenset(
    {"rna_fwd", "rna_rev", "methyl_pct", "n_methylated", "n_total", "GT", "AF", "DP", "MQ"}
)
_PLOTNADO_METHYL_ALIASES = {"methylation_pct": "methyl_pct"}
_PLOTNADO_VARIANT_ALIASES = {"genotype": "GT", "allele_frequency": "AF"}
# plotnado allele-depth variables → synthesised from AF (ref=1-AF, alt=AF)
_PLOTNADO_VARIANT_SYNTH = {"allele_depth_ref", "allele_depth_alt"}


def _parse_plotnado_region(region: str) -> "tuple[str, int, int]":
    """Parse ``'chr1:1000000-1001000'`` → ``(chr1, 1000000, 1001000)``."""
    chrom, coords = region.split(":")
    start, end = map(int, coords.replace(",", "").split("-"))
    return chrom, start, end


def _orient_rna_strands(
    ds: xr.Dataset,
    stranded_by_sample: dict[str, str],
) -> xr.Dataset:
    """Present RNA strands in intuitive transcript orientation.

    Public xarray views should expose ``rna_fwd`` / ``rna_rev`` as forward- and
    reverse-transcript signal respectively. Reverse-stranded libraries therefore
    need their stored arrays swapped on read.
    """
    if "rna_fwd" not in ds.data_vars or "rna_rev" not in ds.data_vars or "sample" not in ds.dims:
        return ds

    sample_names = [
        s.decode() if isinstance(s, bytes) else str(s) for s in ds.coords["sample"].values
    ]
    reverse_samples = {
        sample for sample in sample_names if stranded_by_sample.get(sample, "") == "R"
    }
    if not reverse_samples:
        return ds

    fwd_parts = []
    rev_parts = []
    for sample in sample_names:
        if sample in reverse_samples:
            fwd_parts.append(ds["rna_rev"].sel(sample=[sample]))
            rev_parts.append(ds["rna_fwd"].sel(sample=[sample]))
        else:
            fwd_parts.append(ds["rna_fwd"].sel(sample=[sample]))
            rev_parts.append(ds["rna_rev"].sel(sample=[sample]))

    remapped = ds.copy()
    remapped["rna_fwd"] = xr.concat(fwd_parts, dim="sample")
    remapped["rna_rev"] = xr.concat(rev_parts, dim="sample")
    return remapped


class _PlotnadoCoverageAdapter:
    """Returned by ``QuantNadoDataset.coverage``; satisfies plotnado stranded-coverage track API."""

    def __init__(self, dataset: QuantNadoDataset) -> None:
        self._ds = dataset

    def extract_region(
        self, region: str, samples=None, strand: "str | None" = None
    ) -> "xr.DataArray":
        chrom, start, end = _parse_plotnado_region(region)
        ds = self._ds.sel(chrom, start, end, samples=samples)
        if strand == "+":
            return ds["rna_fwd"]
        if strand == "-":
            return ds["rna_rev"]
        for key in ds.data_vars:
            if key not in _PLOTNADO_COVERAGE_SKIP:
                return ds[key]
        raise KeyError(f"No coverage array found; available: {list(ds.data_vars)}")


class _PlotnadoMethylAdapter:
    """Returned by ``QuantNadoDataset.methylation``; satisfies plotnado methylation track API."""

    def __init__(self, dataset: QuantNadoDataset) -> None:
        self._ds = dataset

    def extract_region(
        self, region: str, variable: str = "methyl_pct", samples=None
    ) -> "xr.DataArray":
        chrom, start, end = _parse_plotnado_region(region)
        ds = self._ds.sel(chrom, start, end, samples=samples)
        key = _PLOTNADO_METHYL_ALIASES.get(variable, variable)
        if key not in ds:
            raise KeyError(f"'{key}' not found; available: {list(ds.data_vars)}")
        return ds[key]


class _PlotnadoVariantsAdapter:
    """Returned by ``QuantNadoDataset.variants``; satisfies plotnado variant track API.

    QuantNado stores allele frequency (``AF``) directly rather than separate
    ref/alt depth arrays.  When plotnado requests ``allele_depth_ref`` or
    ``allele_depth_alt`` we synthesise them from ``AF`` so that plotnado's
    internal ``af = alt / (ref + alt)`` calculation recovers the original value:

    * ``allele_depth_ref`` → ``1 - AF``
    * ``allele_depth_alt`` → ``AF``
    """

    def __init__(self, dataset: QuantNadoDataset) -> None:
        self._ds = dataset

    def extract_region(self, region: str, variable: str = "AF", samples=None) -> "xr.DataArray":
        chrom, start, end = _parse_plotnado_region(region)
        ds = self._ds.sel(chrom, start, end, samples=samples)
        if variable in _PLOTNADO_VARIANT_SYNTH:
            if "AF" in ds:
                af = ds["AF"].astype(float)
            elif "GT" in ds:
                # AF not stored — derive from genotype: het→0.5, hom-alt→1.0, else→0.0
                gt = ds["GT"]
                af = xr.where(gt == 2, 1.0, xr.where(gt == 1, 0.5, 0.0)).astype(float)
            else:
                raise KeyError(
                    "Neither 'AF' nor 'GT' found for variant track; "
                    f"available: {list(ds.data_vars)}"
                )
            return (1.0 - af) if variable == "allele_depth_ref" else af
        key = _PLOTNADO_VARIANT_ALIASES.get(variable, variable)
        if key not in ds:
            raise KeyError(f"'{key}' not found; available: {list(ds.data_vars)}")
        return ds[key]


# ---------------------------------------------------------------------------
# NormalisedQuantNadoDataset — plotnado-compatible wrapper with CPM scaling
# ---------------------------------------------------------------------------


class NormalisedQuantNadoDataset:
    """A dataset-level normalised view of a :class:`QuantNadoDataset`."""

    def __init__(
        self,
        dataset: QuantNadoDataset,
        method: str = "cpm",
        library_sizes: "pd.Series | dict | None" = None,
        feature_lengths=None,
    ) -> None:
        self._inner = dataset
        self._method = method.lower()
        if library_sizes is not None:
            if isinstance(library_sizes, dict):
                library_sizes = pd.Series(library_sizes, name="library_size")
            self._lib_sizes: pd.Series = library_sizes.astype(float)
        else:
            from .normalise import get_library_sizes

            self._lib_sizes = get_library_sizes(dataset)
        self._feature_lengths = feature_lengths

    def __getattr__(self, name: str):
        return getattr(self._inner, name)

    @property
    def info(self) -> DatasetInfo:
        """Dataset summary annotated with normalisation state."""
        summary = DatasetInfo(self._inner.info)
        extras = dict(summary.get("extras", {}))
        extras.update(
            {
                "normalised": True,
                "normalise_method": self._method,
            }
        )
        summary["extras"] = extras
        return summary

    def _normalise_data(self, data):
        return self._inner.normalise(
            data,
            method=self._method,
            library_sizes=self._lib_sizes,
            feature_lengths=self._feature_lengths,
        )

    def count_features(self, *args, **kwargs):
        """Count features on raw signal then apply normalisation to the matrix."""
        # Disable integerize by default — normalised values are floats
        kwargs.setdefault("integerize", False)
        counts_df, feature_metadata = self._inner.count_features(*args, **kwargs)
        feature_lengths = (
            feature_metadata["range_length"] if "range_length" in feature_metadata.columns else None
        )
        normalised_df = self._inner.normalise(
            counts_df,
            method=self._method,
            library_sizes=self._lib_sizes,
            feature_lengths=feature_lengths if self._method in ("rpkm", "tpm") else None,
        )
        return normalised_df, feature_metadata

    def quantify_signal(self, *args, **kwargs):
        """Quantify stored signal on raw data then apply normalisation to the matrix."""
        kwargs.setdefault("integerize", False)
        kwargs.setdefault("return_metadata", True)
        result = self._inner.quantify_signal(*args, **kwargs)
        if isinstance(result, tuple):
            matrix, feature_metadata = result
        else:
            matrix, feature_metadata = result, None
        feature_lengths = (
            feature_metadata["range_length"]
            if feature_metadata is not None and "range_length" in feature_metadata.columns
            else None
        )
        normalised_df = self._inner.normalise(
            matrix,
            method=self._method,
            library_sizes=self._lib_sizes,
            feature_lengths=feature_lengths if self._method in ("rpkm", "tpm") else None,
        )
        return (normalised_df, feature_metadata) if feature_metadata is not None else normalised_df

    def subset(
        self,
        assay: "str | Sequence[str] | None" = None,
        samples: "str | Sequence[str] | None" = None,
    ) -> "NormalisedQuantNadoDataset":
        return NormalisedQuantNadoDataset(
            self._inner.subset(assay=assay, samples=samples),
            method=self._method,
            library_sizes=self._lib_sizes,
            feature_lengths=self._feature_lengths,
        )

    def sel(self, *args, **kwargs):
        ds = self._inner.sel(*args, **kwargs)
        return self._normalise_data(ds)

    # ------------------------------------------------------------------
    # Plotnado integration (overrides QuantNadoDataset implementations)
    # ------------------------------------------------------------------

    def extract_region(
        self,
        region: str,
        samples=None,
        array_key: "str | None" = None,
    ) -> xr.DataArray:
        result = self._inner.extract_region(region, samples=samples, array_key=array_key)
        return self._normalise_data(result)

    @property
    def coverage(self) -> "_NormalisedCoverageAdapter":
        return _NormalisedCoverageAdapter(self._inner.coverage, self._normalise_data)

    @property
    def methylation(self) -> _PlotnadoMethylAdapter:
        """Methylation is already a percentage — no scaling applied."""
        return self._inner.methylation

    @property
    def variants(self) -> _PlotnadoVariantsAdapter:
        """Variant AF / GT are already on absolute scales — no scaling applied."""
        return self._inner.variants


class _NormalisedCoverageAdapter:
    """Wraps ``_PlotnadoCoverageAdapter`` and applies normalisation to its output."""

    def __init__(self, adapter: _PlotnadoCoverageAdapter, scale_fn) -> None:
        self._adapter = adapter
        self._scale_fn = scale_fn

    def extract_region(
        self, region: str, samples=None, strand: "str | None" = None
    ) -> xr.DataArray:
        result = self._adapter.extract_region(region, samples=samples, strand=strand)
        return self._scale_fn(result)
