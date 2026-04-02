from __future__ import annotations
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="zarr")


from pathlib import Path
from typing import Iterable, Any
import os
import shutil
import hashlib
import json
import tempfile
import uuid
from enum import StrEnum

import pysam
import bamnado
import numpy as np
import pandas as pd
import zarr
from zarr.storage import LocalStore, ZipStore
from zarr.codecs import BloscCodec
from loguru import logger
from .core import BaseStore
from quantnado.utils import estimate_chunk_len, is_network_fs


def _copy_read_filter(rf: "bamnado.ReadFilter") -> "bamnado.ReadFilter":
    if hasattr(rf, "copy"):
        return rf.copy()
    new_rf = bamnado.ReadFilter()
    for attr in (
        "min_mapq",
        "proper_pair",
        "min_length",
        "max_length",
        "strand",
        "min_fragment_length",
        "max_fragment_length",
        "blacklist_bed",
        "whitelisted_barcodes",
        "read_group",
        "filter_tag",
        "filter_tag_value",
    ):
        setattr(new_rf, attr, getattr(rf, attr))
    return new_rf


def _to_str_list(items: Iterable[Any]) -> list[str]:
    import pandas as pd

    return [str(i) if not pd.isna(i) else "" for i in items]


BIN_SIZE = 1
CONSTRUCTION_ARRAY_DTYPE = np.uint32
DEFAULT_CONSTRUCTION_COMPRESSION = "default"


class Strandedness(StrEnum):
    UNSTRANDED = "U"
    REVERSE = "R"
    FORWARD = "F"


class CoverageType(StrEnum):
    UNSTRANDED = "unstranded"
    STRANDED = "stranded"
    MICRO_CAPTURE_C = "mcc"


def _process_chromosome_mcc(
    bam_file: str,
    contig: str,
    contig_size: int,
    viewpoints: list[str],
    viewpoint_tag: str,
    base_filter: "bamnado.ReadFilter",
    use_fragment: bool = False,
) -> dict[str, np.ndarray]:
    """Single-pass over a chromosome, splitting reads by viewpoint tag.

    Returns a dict mapping each viewpoint name to a uint32 coverage array of
    length ``contig_size``.  Falls back to per-viewpoint bamnado calls if a
    blacklist BED is configured (too complex to replicate in Python).
    """
    if base_filter.blacklist_bed is not None:
        # Fallback: can't replicate blacklist filtering cheaply in Python.
        result = {}
        for vp in viewpoints:
            rf = _copy_read_filter(base_filter)
            rf.filter_tag = viewpoint_tag
            rf.filter_tag_value = vp
            signal = bamnado.get_signal_for_chromosome(
                bam_path=bam_file,
                chromosome_name=contig,
                bin_size=BIN_SIZE,
                scale_factor=1.0,
                use_fragment=use_fragment,
                ignore_scaffold_chromosomes=False,
                read_filter=rf,
            )
            if signal.shape[0] != contig_size:
                signal = signal[:contig_size] if signal.shape[0] > contig_size else np.pad(signal, (0, contig_size - signal.shape[0]))
            result[vp] = signal.astype(np.uint32)
        return result

    vp_set = set(viewpoints)
    arrays: dict[str, np.ndarray] = {vp: np.zeros(contig_size, dtype=np.uint32) for vp in viewpoints}

    min_mapq = base_filter.min_mapq
    require_proper_pair = base_filter.proper_pair
    min_length = base_filter.min_length
    max_length = base_filter.max_length
    min_frag = base_filter.min_fragment_length
    max_frag = base_filter.max_fragment_length
    whitelisted = set(base_filter.whitelisted_barcodes) if base_filter.whitelisted_barcodes else None
    read_group = base_filter.read_group

    with pysam.AlignmentFile(bam_file, "rb") as bam:
        try:
            reads = bam.fetch(contig)
        except ValueError:
            return arrays  # contig not in BAM

        try:
            for read in reads:
                if read.is_unmapped:
                    continue
                if read.mapping_quality < min_mapq:
                    continue
                if require_proper_pair and not read.is_proper_pair:
                    continue
                ql = read.query_length or 0
                if ql < min_length or ql > max_length:
                    continue
                if min_frag is not None or max_frag is not None:
                    tl = abs(read.template_length)
                    if min_frag is not None and tl < min_frag:
                        continue
                    if max_frag is not None and tl > max_frag:
                        continue
                if whitelisted is not None:
                    if not read.has_tag("CB") or read.get_tag("CB") not in whitelisted:
                        continue
                if read_group is not None:
                    if not read.has_tag("RG") or read.get_tag("RG") != read_group:
                        continue
                if not read.has_tag(viewpoint_tag):
                    continue
                vp = read.get_tag(viewpoint_tag)
                if vp not in vp_set:
                    continue

                arr = arrays[vp]
                if use_fragment and read.template_length != 0:
                    start = min(read.reference_start, read.next_reference_start)
                    end = start + abs(read.template_length)
                else:
                    start = read.reference_start
                    end = read.reference_end or start
                start = max(0, start)
                end = min(contig_size, end)
                if start < end:
                    arr[start:end] += 1
        except ValueError:
            pass  # pysam spurious "Firing event N" at end of fetch — arrays are complete

    return arrays


def _get_viewpoints_from_mcc_bam(bam_path: Path | str, viewpoint_tag: str = "VP") -> list[str]:
    viewpoints = set()
    try:
        with pysam.AlignmentFile(str(bam_path), "rb") as bam:
            for read in bam.fetch(until_eof=True):
                if read.has_tag(viewpoint_tag):
                    vp = read.get_tag(viewpoint_tag)
                    viewpoints.add(vp)
    except Exception as e:
        if viewpoints:
            logger.debug(
                f"Encountered an error while scanning MCC viewpoint tags for {bam_path}, "
                f"but recovered {len(viewpoints)} viewpoint(s): {e}"
            )
        else:
            logger.warning(f"Could not extract MCC viewpoint tags from {bam_path}: {e}")
    if not viewpoints:
        raise ValueError(
            f"No MCC viewpoint tags ('{viewpoint_tag}') found in {bam_path}"
        )
    return sorted(viewpoints)


def _compute_sample_hash(sample_names: list[str]) -> str:
    canonical = "|".join(sample_names)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _compute_bam_hash(bam_path: Path | str) -> str:
    h = hashlib.md5()
    try:
        with open(bam_path, "rb") as f:
            h.update(f.read(16384))
    except (FileNotFoundError, PermissionError) as e:
        logger.warning(f"Could not compute hash for {bam_path}: {e}")
        return ""
    return h.hexdigest()


def _resolve_chunk_len(
    chromsizes: dict[str, int],
    store_path: Path,
    chunk_len: int | None,
    n_samples: int = 1,
) -> int:
    if chunk_len is not None:
        resolved = int(chunk_len)
        if resolved <= 0:
            raise ValueError("chunk_len must be a positive integer")
        return resolved

    fs_probe_path = store_path if store_path.exists() else store_path.parent
    fs_is_network = is_network_fs(fs_probe_path)
    estimate = estimate_chunk_len(
        contig_lengths=chromsizes,
        dtype_bytes=np.dtype(CONSTRUCTION_ARRAY_DTYPE).itemsize,
        n_samples=n_samples,
        fs_is_network=fs_is_network,
    )
    resolved = int(estimate["chunk_len"])
    fs_label = "network" if fs_is_network else "local"
    logger.info(
        "Resolved chunk_len={} for {} filesystem at {} ({} estimated chunks)",
        resolved,
        fs_label,
        fs_probe_path,
        estimate["num_chunks"],
    )
    return resolved


def _normalize_construction_compression(profile: str | None) -> str:
    normalized = (profile or DEFAULT_CONSTRUCTION_COMPRESSION).strip().lower()
    aliases = {"uncompressed": "none", "off": "none"}
    normalized = aliases.get(normalized, normalized)
    valid_profiles = {"default", "fast", "none"}
    if normalized not in valid_profiles:
        raise ValueError(
            f"construction_compression must be one of {sorted(valid_profiles)}, got {profile!r}"
        )
    return normalized


def _resolve_construction_compressors(
    profile: str | None,
) -> tuple[str, list[BloscCodec]]:
    normalized = _normalize_construction_compression(profile)
    if normalized == "none":
        return normalized, []
    if normalized == "fast":
        return normalized, [BloscCodec(cname="zstd", clevel=1, shuffle="shuffle")]
    return normalized, [BloscCodec(cname="zstd", clevel=3, shuffle="shuffle")]


def _resolve_staging_root(staging_dir: Path | str | None) -> Path:
    if staging_dir is not None:
        return Path(staging_dir)
    return Path(os.environ.get("TMPDIR") or tempfile.gettempdir())


def _build_staging_store_path(
    final_store_path: Path,
    staging_dir: Path | str | None,
) -> Path:
    staging_root = _resolve_staging_root(staging_dir)
    staging_root.mkdir(parents=True, exist_ok=True)
    return staging_root / f".{final_store_path.stem}.staging-{uuid.uuid4().hex}.zarr"


def _delete_store_path(store_path: Path) -> None:
    if not store_path.exists():
        return
    if store_path.is_dir():
        shutil.rmtree(store_path)
    else:
        store_path.unlink()


def _publish_staged_store(staged_store_path: Path, final_store_path: Path) -> None:
    final_store_path.parent.mkdir(parents=True, exist_ok=True)
    publish_tmp_path = final_store_path.parent / (
        f".{final_store_path.name}.publishing-{uuid.uuid4().hex}"
    )
    try:
        shutil.copytree(staged_store_path, publish_tmp_path)
        _delete_store_path(final_store_path)
        publish_tmp_path.rename(final_store_path)
    except Exception:
        _delete_store_path(publish_tmp_path)
        raise
    finally:
        _delete_store_path(staged_store_path)


def _get_chromsizes_from_bam(bam_path: Path | str) -> dict[str, int]:
    with pysam.AlignmentFile(str(bam_path), "rb") as sam:
        return {ref: length for ref, length in zip(sam.references, sam.lengths)}


def _parse_chromsizes(
    chromsizes: str | Path | dict[str, int],
    *,
    filter_chromosomes: bool = True,
    test: bool = False,
) -> dict[str, int]:
    if isinstance(chromsizes, dict):
        chromsizes_dict = chromsizes
    else:
        path = Path(chromsizes)
        if not path.exists():
            raise FileNotFoundError(f"Chromsizes file not found: {path}")
        df = pd.read_csv(path, sep="\t", header=None, names=["chrom", "size"])
        chromsizes_dict = df.set_index("chrom")["size"].to_dict()

    if filter_chromosomes:
        chromsizes_dict = {
            k: v for k, v in chromsizes_dict.items() if k.startswith("chr") and "_" not in k
        }

    if test:
        desired = ["chr21", "chr22", "chrY"]
        chromsizes_dict = {k: v for k, v in chromsizes_dict.items() if k in desired}
        logger.info(f"Test mode enabled: keeping chromosomes {list(chromsizes_dict.keys())}")

    return chromsizes_dict


class BamStore(BaseStore):
    """
    Zarr-backed BAM signal store.

    Store layout::

        root/
        ├── coverage/
        │   ├── chr1   (chrom_len, n_samples)  uint32  chunks (chunk_len, n_samples)
        │   └── ...
        ├── coverage_fwd/   (stranded only)
        │   └── ...
        ├── coverage_rev/   (stranded only)
        │   └── ...
        └── metadata/
            ├── completed        (n_samples,)  bool
            ├── sparsity         (n_samples,)  float32
            ├── total_reads      (n_samples,)  int64
            ├── mean_read_length (n_samples,)  float32
            └── sample_hashes   (n_samples, 16) uint8

    Use ``from_bam_files`` to create a new store or ``open`` to load an existing one.
    """

    def __init__(
        self,
        store_path: Path | str,
        chromsizes: dict[str, int] | Path | str,
        sample_names: list[str],
        *,
        chunk_len: int | None = None,
        construction_compression: str = DEFAULT_CONSTRUCTION_COMPRESSION,
        overwrite: bool = True,
        resume: bool = False,
        read_only: bool = False,
        coverage_type: CoverageType
        | list[CoverageType]
        | dict[str, CoverageType] = CoverageType.UNSTRANDED,
        bam_filter: bamnado.ReadFilter
        | list[bamnado.ReadFilter]
        | dict[str, bamnado.ReadFilter]
        | None = None,
        count_fragments: bool = False,
        viewpoints: set[str] | None = None,
        viewpoint_tag: str = "VP",
        sample_bam_map: dict[str, str] | None = None,
        viewpoint_map: dict[str, str] | None = None,
    ) -> None:
        self.path = Path(store_path)
        self.store_path = self._normalize_path(self.path)
        self.read_only = read_only
        self.construction_compression, self.compressors = _resolve_construction_compressors(
            construction_compression
        )

        if self.store_path.exists() and not overwrite:
            if str(self.store_path).endswith(".zarr.zip"):
                _zip_store = ZipStore(str(self.store_path), mode="r")
                self.root = zarr.open_group(store=_zip_store, mode="r")
            else:
                self.root = zarr.open_group(str(self.store_path), mode="r" if read_only else "r+")
            self._init_common_attributes(sample_names)
        else:
            self.sample_names = [str(s) for s in sample_names]
            self._setup_sample_lookup()
            self._chromsizes = _parse_chromsizes(chromsizes)
            self._chromosomes = sorted(list(self._chromsizes.keys()))
            self.completed_mask_raw = np.zeros(len(self.sample_names), dtype=bool)
            self._metadata_cache = None

        self.n_samples = len(self.sample_names)
        self.sample_hash = _compute_sample_hash(self.sample_names)
        self.coverage_type = coverage_type
        self.count_fragments = count_fragments
        self.viewpoints = viewpoints
        self.viewpoint_tag = viewpoint_tag
        self.sample_bam_map = sample_bam_map
        self.viewpoint_map = viewpoint_map

        if not bam_filter:
            self.bam_filter = bamnado.ReadFilter()
        else:
            self.bam_filter = bam_filter

        if self.n_samples == 0:
            raise ValueError("sample_names must not be empty")

        if self.store_path.exists():
            if overwrite:
                if read_only:
                    raise ValueError("Cannot overwrite store in read-only mode.")
                logger.warning(f"Deleting existing store at: {self.store_path}")
                if self.store_path.is_dir():
                    shutil.rmtree(self.store_path)
                else:
                    self.store_path.unlink()
                self.chunk_len = _resolve_chunk_len(
                    self.chromsizes, self.store_path, chunk_len, self.n_samples
                )
                self._init_store()
            elif resume:
                self._load_existing()
                self.chunk_len = int(
                    self.root.attrs.get(
                        "chunk_len",
                        _resolve_chunk_len(
                            self.chromsizes, self.store_path, chunk_len, self.n_samples
                        ),
                    )
                )
                self._validate_sample_names()
            else:
                raise FileExistsError(
                    f"Store already exists at {self.store_path}; set overwrite=True or resume=True"
                )
        else:
            if read_only:
                raise FileNotFoundError(
                    f"Store does not exist at {self.store_path} (read_only=True)"
                )
            self.chunk_len = _resolve_chunk_len(
                self.chromsizes, self.store_path, chunk_len, self.n_samples
            )
            self._init_store()

    @property
    def bam_filter_map(self) -> dict[str, bamnado.ReadFilter]:
        match self.bam_filter:
            case bamnado.ReadFilter():
                return {s: self.bam_filter for s in self.sample_names}
            case list():
                if not all(isinstance(v, bamnado.ReadFilter) for v in self.bam_filter):
                    raise ValueError(
                        "If bam_filter is a list, all values must be bamnado.ReadFilter instances."
                    )
                if len(self.bam_filter) != len(self.sample_names):
                    raise ValueError(
                        "If bam_filter is a list, it must have the same length as sample_names."
                    )
                return {s: bf for s, bf in zip(self.sample_names, self.bam_filter)}
            case dict():
                if not all(isinstance(v, bamnado.ReadFilter) for v in self.bam_filter.values()):
                    raise ValueError(
                        "If bam_filter is a dict, all values must be bamnado.ReadFilter instances."
                    )
                unknown = set(self.bam_filter) - set(self.sample_names)
                if unknown:
                    raise ValueError(f"Unknown sample names in bam_filter dict: {sorted(unknown)}")
                return {
                    s: self.bam_filter.get(s) if self.bam_filter.get(s) else bamnado.ReadFilter()
                    for s in self.sample_names
                }
            case None:
                return {s: bamnado.ReadFilter() for s in self.sample_names}

    @property
    def coverage_type_map(self) -> dict[str, CoverageType]:
        match self.coverage_type:
            case CoverageType():
                return {s: self.coverage_type for s in self.sample_names}
            case list():
                if not all(isinstance(v, CoverageType) for v in self.coverage_type):
                    raise ValueError(
                        "If coverage_type is a list, all values must be CoverageType enum members."
                    )
                if len(self.coverage_type) != len(self.sample_names):
                    raise ValueError(
                        "If coverage_type is a list, it must have the same length as sample_names."
                    )
                return {s: lt for s, lt in zip(self.sample_names, self.coverage_type)}
            case dict():
                if not all(isinstance(v, CoverageType) for v in self.coverage_type.values()):
                    raise ValueError(
                        "If coverage_type is a dict, all values must be CoverageType enum members."
                    )
                unknown = set(self.coverage_type) - set(self.sample_names)
                if unknown:
                    raise ValueError(
                        f"Unknown sample names in coverage_type dict: {sorted(unknown)}"
                    )
                return {
                    s: self.coverage_type.get(s, CoverageType.UNSTRANDED) for s in self.sample_names
                }
            case None:
                return {s: CoverageType.UNSTRANDED for s in self.sample_names}

    @classmethod
    def open(cls, store_path: str | Path, read_only: bool = True) -> "BamStore":
        store_path = cls._normalize_path(store_path)
        if not store_path.exists():
            raise FileNotFoundError(f"Store does not exist at {store_path}")
        mode = "r" if read_only else "r+"
        if str(store_path).endswith(".zarr.zip"):
            store = ZipStore(str(store_path), mode="r")
            group = zarr.open_group(store=store, mode="r")
        else:
            group = zarr.open_group(str(store_path), mode=mode)
        try:
            sample_names = list(group.attrs["sample_names"])
            chromsizes = {str(k): int(v) for k, v in group.attrs["chromsizes"].items()}
            chunk_len = int(group.attrs["chunk_len"])
        except KeyError as e:
            raise ValueError(f"Missing required attribute in store: {e}")
        raw_strand = group.attrs.get("stranded")
        if isinstance(raw_strand, dict):
            coverage_type: CoverageType | dict[str, CoverageType] = {
                s: (
                    CoverageType.STRANDED if v == CoverageType.STRANDED else CoverageType.UNSTRANDED
                )
                for s, v in raw_strand.items()
            }
            if all(bt == CoverageType.UNSTRANDED for bt in coverage_type.values()):
                coverage_type = CoverageType.UNSTRANDED
        else:
            coverage_type = CoverageType.UNSTRANDED
        return cls(
            store_path=store_path,
            chromsizes=chromsizes,
            sample_names=sample_names,
            chunk_len=chunk_len,
            overwrite=False,
            resume=True,
            read_only=read_only,
            coverage_type=coverage_type,
        )

    def _check_writable(self):
        if getattr(self, "read_only", False):
            raise RuntimeError(
                "Store is in read-only mode. Reopen with read_only=False to allow modifications."
            )

    @staticmethod
    def _normalize_path(path: Path | str) -> Path:
        path = Path(path)
        if str(path).endswith(".zarr.zip") or str(path).endswith(".zarr"):
            return path
        return path.with_suffix(".zarr")

    def _init_store(self) -> None:
        """Create a new combined store with coverage/ groups (position, n_samples) arrays."""
        store = LocalStore(str(self.store_path))
        self.root = zarr.group(store=store, overwrite=True, zarr_format=3)
        self.meta = self.root.create_group("metadata")

        cov_grp = self.root.create_group("coverage")
        for chrom, size in self.chromsizes.items():
            cov_grp.create_array(
                name=chrom,
                shape=(size, self.n_samples),
                chunks=(self.chunk_len, self.n_samples),
                dtype=CONSTRUCTION_ARRAY_DTYPE,
                compressors=self.compressors,
                fill_value=0,
                overwrite=True,
            )

        has_stranded = any(bt == CoverageType.STRANDED for bt in self.coverage_type_map.values())
        if has_stranded:
            for suffix in ("coverage_fwd", "coverage_rev"):
                grp = self.root.create_group(suffix)
                for chrom, size in self.chromsizes.items():
                    grp.create_array(
                        name=chrom,
                        shape=(size, self.n_samples),
                        chunks=(self.chunk_len, self.n_samples),
                        dtype=np.uint32,
                        compressors=self.compressors,
                        fill_value=0,
                        overwrite=True,
                    )

        self.meta.create_array(
            "completed", shape=(self.n_samples,), dtype=bool, fill_value=False, overwrite=True
        )
        self.meta.create_array(
            "sparsity", shape=(self.n_samples,), dtype=np.float32, fill_value=np.nan, overwrite=True
        )
        self.meta.create_array(
            "sample_hashes",
            shape=(self.n_samples, 16),
            dtype=np.uint8,
            fill_value=0,
            overwrite=True,
        )
        self.meta.create_array(
            "total_reads", shape=(self.n_samples,), dtype=np.int64, fill_value=0, overwrite=True
        )
        self.meta.create_array(
            "mean_read_length",
            shape=(self.n_samples,),
            dtype=np.float32,
            fill_value=np.nan,
            overwrite=True,
        )

        strandedness_for_attrs = {
            s: (bt.value if bt == CoverageType.STRANDED else "")
            for s, bt in self.coverage_type_map.items()
        }
        _data_type_map = {
            CoverageType.STRANDED: "stranded_coverage",
            CoverageType.MICRO_CAPTURE_C: "mcc",
        }
        data_type_list = [
            _data_type_map.get(self.coverage_type_map[s], "coverage") for s in self.sample_names
        ]
        self.root.attrs.update(
            {
                "chromosomes": self.chromosomes,
                "chromsizes": self.chromsizes,
                "n_samples": self.n_samples,
                "chunk_len": self.chunk_len,
                "construction_compression": self.construction_compression,
                "structure": "coverage groups (position x sample)",
                "bin_size": BIN_SIZE,
                "sample_names": self.sample_names,
                "sample_names_hash": self.sample_hash,
                "stranded": strandedness_for_attrs,
                "metadata_data_type": data_type_list,
            }
        )
        logger.info(f"Initialized BamStore at {self.store_path}")

    def _init_sample_store_at(self, path: Path, is_stranded: bool) -> zarr.Group:
        """Create a 1D per-sample temp store (no sample dimension)."""
        store = LocalStore(str(path))
        root = zarr.group(store=store, overwrite=True, zarr_format=3)
        cov_grp = root.create_group("coverage")
        for chrom, size in self.chromsizes.items():
            cov_grp.create_array(
                name=chrom,
                shape=(size,),
                dtype=CONSTRUCTION_ARRAY_DTYPE,
                compressors=self.compressors,
                fill_value=0,
                overwrite=True,
            )
        if is_stranded:
            for suffix in ("coverage_fwd", "coverage_rev"):
                grp = root.create_group(suffix)
                for chrom, size in self.chromsizes.items():
                    grp.create_array(
                        name=chrom,
                        shape=(size,),
                        dtype=np.uint32,
                        compressors=self.compressors,
                        fill_value=0,
                        overwrite=True,
                    )
        return root

    def _load_existing(self) -> None:
        if str(self.store_path).endswith(".zarr.zip"):
            store = ZipStore(str(self.store_path), mode="r")
            self.root = zarr.open_group(store=store, mode="r")
        else:
            store = LocalStore(str(self.store_path))
            self.root = zarr.open_group(store=store, mode="a")
        self.meta = self.root.get("metadata")
        logger.info(f"Resuming existing store at {self.store_path}")

    def _validate_sample_names(self) -> None:
        stored_names = self.root.attrs.get("sample_names")
        stored_hash = self.root.attrs.get("sample_names_hash")

        if stored_names is None:
            raise ValueError("Existing store missing sample_names attribute; cannot validate")

        stored_names = [str(s) for s in stored_names]
        if stored_names != self.sample_names:
            raise ValueError(
                "Provided sample_names do not match stored sample_names; refusing to resume to prevent corruption"
            )

        if stored_hash and stored_hash != self.sample_hash:
            raise ValueError(
                f"Sample names hash mismatch (stored={stored_hash}, current={self.sample_hash})"
            )

    @property
    def completed_mask(self) -> np.ndarray:
        if not hasattr(self, "_completed_mask") or not self.read_only:
            self._completed_mask = self.meta["completed"][:].astype(bool)
        return self._completed_mask

    @property
    def sample_hashes(self) -> list[str]:
        if not hasattr(self, "_sample_hashes") or not self.read_only:
            arr = self.meta["sample_hashes"][:]
            self._sample_hashes = ["".join(f"{b:02x}" for b in row) for row in arr]
        return self._sample_hashes

    @property
    def library_sizes(self) -> pd.Series | None:
        if "total_reads" not in self.meta:
            return None
        reads = self.meta["total_reads"][:].astype(float)
        reads[~self.completed_mask] = np.nan
        return pd.Series(reads, index=self.sample_names, name="library_size")

    @property
    def mean_read_lengths(self) -> pd.Series | None:
        if "mean_read_length" not in self.meta:
            return None
        lengths = self.meta["mean_read_length"][:].astype(float)
        lengths[~self.completed_mask] = np.nan
        return pd.Series(lengths, index=self.sample_names, name="mean_read_length")

    def _process_chromosome(
        self,
        bam_file: str,
        contig: str,
        contig_size: int,
        is_stranded: bool,
        use_fragment: bool = False,
        read_filter: bamnado.ReadFilter | None = None,
    ) -> tuple[float, np.ndarray, np.ndarray | None]:
        if not read_filter:
            read_filter = bamnado.ReadFilter()

        if is_stranded:
            read_filter_fwd = _copy_read_filter(read_filter)
            read_filter_fwd.strand = Strandedness.FORWARD.value
            read_filter_rev = _copy_read_filter(read_filter)
            read_filter_rev.strand = Strandedness.REVERSE.value
            filters = [read_filter_fwd, read_filter_rev]
        else:
            filters = [read_filter]

        signal_arrays = []
        for f in filters:
            signal_arrays.append(
                bamnado.get_signal_for_chromosome(
                    bam_path=bam_file,
                    chromosome_name=contig,
                    bin_size=BIN_SIZE,
                    scale_factor=1.0,
                    use_fragment=use_fragment,
                    ignore_scaffold_chromosomes=False,
                    read_filter=f,
                )
            )

        signal_arrays_aligned = []
        for signal in signal_arrays:
            len_signal = signal.shape[0]
            if len_signal != contig_size:
                logger.warning(
                    f"Signal length for {contig} differs from chromsizes ({len_signal} vs {contig_size}); aligning"
                )
                if len_signal > contig_size:
                    signal = signal[:contig_size]
                else:
                    signal = np.pad(signal, (0, contig_size - len_signal), mode="constant")
            signal_arrays_aligned.append(signal)

        del signal_arrays

        if is_stranded:
            max_val = signal_arrays_aligned[0].max()
            min_val = signal_arrays_aligned[0].min()
            dtype_fwd = np.uint16 if max_val <= np.iinfo(np.uint16).max else np.uint32
            if min_val < 0:
                dtype_fwd = (
                    np.int16
                    if np.iinfo(np.int16).min <= min_val and max_val <= np.iinfo(np.int16).max
                    else np.int32
                )
            fwd_data = signal_arrays_aligned[0].astype(dtype_fwd, copy=False)
            sparsity_fwd = float((np.sum(fwd_data == 0) / fwd_data.size) * 100)
            del signal_arrays_aligned[0]
            rev_data = signal_arrays_aligned[0].astype(dtype_fwd, copy=False)
            sparsity_rev = float((np.sum(rev_data == 0) / rev_data.size) * 100)
            sparsity = (sparsity_fwd + sparsity_rev) / 2
            del signal_arrays_aligned[0]
            return sparsity, fwd_data, rev_data
        else:
            max_val = signal_arrays_aligned[0].max()
            dtype = np.uint16 if max_val <= np.iinfo(np.uint16).max else np.uint32
            data = signal_arrays_aligned[0].astype(dtype, copy=False)
            sparsity = float((np.sum(data == 0) / data.size) * 100)
            del signal_arrays_aligned[0]
            return sparsity, data, None

    def _write_sample_store(
        self,
        bam_file: str,
        sample_name: str,
        tmp_path: Path,
        max_workers: int = 1,
    ) -> None:
        """Process a single BAM and write coverage to a 1D per-sample temp store."""
        logger.info(f"Processing sample: {sample_name}")

        coverage_type = self.coverage_type_map.get(sample_name, CoverageType.UNSTRANDED)
        is_stranded = coverage_type == CoverageType.STRANDED
        use_fragment = self.count_fragments

        read_filter = self.bam_filter_map[sample_name]
        if coverage_type == CoverageType.MICRO_CAPTURE_C and self.viewpoint_map:
            vp = self.viewpoint_map.get(sample_name)
            if vp:
                read_filter = _copy_read_filter(read_filter)
                read_filter.filter_tag = self.viewpoint_tag
                read_filter.filter_tag_value = vp

        tmp_root = self._init_sample_store_at(tmp_path, is_stranded)
        sparsity_values: list[float] = []
        for contig, size in self.chromsizes.items():
            sparsity, fwd, rev = self._process_chromosome(
                bam_file,
                contig,
                size,
                is_stranded,
                use_fragment=use_fragment,
                read_filter=read_filter,
            )
            if rev is not None:
                tmp_root["coverage_fwd"][contig][: fwd.shape[0]] = fwd
                tmp_root["coverage_rev"][contig][: rev.shape[0]] = rev
            else:
                tmp_root["coverage"][contig][: fwd.shape[0]] = fwd
            sparsity_values.append(sparsity)

        bam_hash = _compute_bam_hash(bam_file)
        total_reads: int | None = None
        mean_read_length: float | None = None
        try:
            with pysam.AlignmentFile(bam_file, "rb") as bam:
                total_reads = bam.mapped
                lengths = []
                for read in bam.fetch():
                    if not read.is_unmapped and read.query_length:
                        lengths.append(read.query_length)
                        if len(lengths) >= 10_000:
                            break
                if lengths:
                    mean_read_length = float(np.mean(lengths))
        except Exception as e:
            logger.warning(f"Could not compute BAM stats for {bam_file}: {e}")

        tmp_root.attrs.update(
            {
                "sample_name": sample_name,
                "sparsity": float(np.mean(sparsity_values)) if sparsity_values else float("nan"),
                "total_reads": total_reads or 0,
                "mean_read_length": mean_read_length
                if mean_read_length is not None
                else float("nan"),
                "bam_hash": bam_hash,
            }
        )

    def _write_sample_direct(self, sample_idx: int, bam_file: str, sample_name: str) -> None:
        """Process a single BAM and write coverage directly into the final store (no temp store)."""
        logger.info(f"Processing sample: {sample_name}")

        coverage_type = self.coverage_type_map.get(sample_name, CoverageType.UNSTRANDED)
        is_stranded = coverage_type == CoverageType.STRANDED
        use_fragment = self.count_fragments

        read_filter = self.bam_filter_map[sample_name]
        if coverage_type == CoverageType.MICRO_CAPTURE_C and self.viewpoint_map:
            vp = self.viewpoint_map.get(sample_name)
            if vp:
                read_filter = _copy_read_filter(read_filter)
                read_filter.filter_tag = self.viewpoint_tag
                read_filter.filter_tag_value = vp

        sparsity_values: list[float] = []
        for contig, size in self.chromsizes.items():
            sparsity, fwd, rev = self._process_chromosome(
                bam_file,
                contig,
                size,
                is_stranded,
                use_fragment=use_fragment,
                read_filter=read_filter,
            )
            if rev is not None:
                self.root["coverage_fwd"][contig][: fwd.shape[0], sample_idx] = fwd
                self.root["coverage_rev"][contig][: rev.shape[0], sample_idx] = rev
            else:
                self.root["coverage"][contig][: fwd.shape[0], sample_idx] = fwd
            sparsity_values.append(sparsity)

        bam_hash = _compute_bam_hash(bam_file)
        total_reads: int | None = None
        mean_read_length: float | None = None
        try:
            with pysam.AlignmentFile(bam_file, "rb") as bam:
                total_reads = bam.mapped
                lengths = []
                for read in bam.fetch():
                    if not read.is_unmapped and read.query_length:
                        lengths.append(read.query_length)
                        if len(lengths) >= 10_000:
                            break
                if lengths:
                    mean_read_length = float(np.mean(lengths))
        except Exception as e:
            logger.warning(f"Could not compute BAM stats for {bam_file}: {e}")

        self.meta["sparsity"][sample_idx] = float(np.mean(sparsity_values)) if sparsity_values else float("nan")
        self.meta["total_reads"][sample_idx] = total_reads or 0
        self.meta["mean_read_length"][sample_idx] = mean_read_length if mean_read_length is not None else float("nan")
        if bam_hash:
            hash_bytes = bytes.fromhex(bam_hash)
            self.meta["sample_hashes"][sample_idx, : len(hash_bytes)] = np.frombuffer(hash_bytes, dtype=np.uint8)
        self.meta["completed"][sample_idx] = True

    def _process_mcc_samples(
        self,
        bam_file: str,
        pending: list[tuple[int, str, str]],
    ) -> None:
        """Process all MCC viewpoints from a single BAM in one pass per chromosome.

        ``pending`` is a list of ``(sample_idx, bam_file, sample_name)`` tuples
        for the viewpoints that still need processing (incomplete samples).
        """
        # Build viewpoint → sample_idx mapping for the pending set only.
        assert self.viewpoint_map is not None
        vp_to_idx: dict[str, int] = {
            self.viewpoint_map[sn]: idx for idx, _, sn in pending
        }
        viewpoints = list(vp_to_idx.keys())
        base_filter = next(iter(self.bam_filter_map.values()))
        # Strip any existing VP tag filter — we split by VP ourselves.
        base_filter = _copy_read_filter(base_filter)
        base_filter.filter_tag = None
        base_filter.filter_tag_value = None

        logger.info(f"Processing {len(viewpoints)} MCC viewpoint(s) from {bam_file} in single-pass mode")

        sparsity_per_vp: dict[str, list[float]] = {vp: [] for vp in viewpoints}
        n_chroms = len(self.chromsizes)

        for i, (contig, size) in enumerate(self.chromsizes.items(), 1):
            logger.debug(f"MCC single-pass: {contig} ({i}/{n_chroms})")
            vp_arrays = _process_chromosome_mcc(
                bam_file=bam_file,
                contig=contig,
                contig_size=size,
                viewpoints=viewpoints,
                viewpoint_tag=self.viewpoint_tag,
                base_filter=base_filter,
                use_fragment=self.count_fragments,
            )
            for vp, arr in vp_arrays.items():
                sample_idx = vp_to_idx[vp]
                self.root["coverage"][contig][:size, sample_idx] = arr
                sparsity_per_vp[vp].append(float((np.sum(arr == 0) / arr.size) * 100))

        # Write metadata for each viewpoint sample.
        bam_hash = _compute_bam_hash(bam_file)
        total_reads: int | None = None
        mean_read_length: float | None = None
        try:
            with pysam.AlignmentFile(bam_file, "rb") as bam:
                total_reads = bam.mapped
                lengths = []
                for read in bam.fetch():
                    if not read.is_unmapped and read.query_length:
                        lengths.append(read.query_length)
                        if len(lengths) >= 10_000:
                            break
                if lengths:
                    mean_read_length = float(np.mean(lengths))
        except Exception as e:
            logger.warning(f"Could not compute BAM stats for {bam_file}: {e}")

        for vp, sparsity_vals in sparsity_per_vp.items():
            sample_idx = vp_to_idx[vp]
            self.meta["sparsity"][sample_idx] = float(np.mean(sparsity_vals)) if sparsity_vals else float("nan")
            self.meta["total_reads"][sample_idx] = total_reads or 0
            self.meta["mean_read_length"][sample_idx] = mean_read_length if mean_read_length is not None else float("nan")
            if bam_hash:
                hash_bytes = bytes.fromhex(bam_hash)
                self.meta["sample_hashes"][sample_idx, : len(hash_bytes)] = np.frombuffer(hash_bytes, dtype=np.uint8)
            self.meta["completed"][sample_idx] = True

    def _write_sample_to_final(self, sample_idx: int, tmp_root: zarr.Group) -> None:
        """Copy a per-sample 1D temp store into column sample_idx of the final 2D store."""
        has_stranded = "coverage_fwd" in tmp_root
        for chrom in self.chromosomes:
            if has_stranded:
                self.root["coverage_fwd"][chrom][:, sample_idx] = tmp_root["coverage_fwd"][chrom][:]
                self.root["coverage_rev"][chrom][:, sample_idx] = tmp_root["coverage_rev"][chrom][:]
            else:
                self.root["coverage"][chrom][:, sample_idx] = tmp_root["coverage"][chrom][:]

        attrs = dict(tmp_root.attrs)
        self.meta["sparsity"][sample_idx] = attrs.get("sparsity", float("nan"))
        self.meta["total_reads"][sample_idx] = attrs.get("total_reads", 0)
        self.meta["mean_read_length"][sample_idx] = attrs.get("mean_read_length", float("nan"))
        bam_hash = attrs.get("bam_hash", "")
        if bam_hash:
            hash_bytes = bytes.fromhex(bam_hash)
            self.meta["sample_hashes"][sample_idx, : len(hash_bytes)] = np.frombuffer(
                hash_bytes,
                dtype=np.uint8,
            )
        self.meta["completed"][sample_idx] = True

    def process_samples(
        self,
        bam_files: list[str] | None = None,
        max_workers: int = 1,
    ) -> None:
        """Process BAM files and write signal data to the zarr store.

        Each sample is written to a temporary 1D store, then combined into the
        final (position, n_samples) arrays. Incomplete samples are skipped.
        """
        if self.sample_bam_map is not None:
            resolved_bam_files = [self.sample_bam_map[s] for s in self.sample_names]
        elif bam_files is not None:
            if len(bam_files) != self.n_samples:
                raise ValueError("bam_files length must match number of sample_names")
            resolved_bam_files = bam_files
        else:
            raise ValueError("bam_files must be provided when sample_bam_map is not set")

        completed = self.completed_mask

        pending: list[tuple[int, str, str]] = []
        for sample_idx, (bam_file, sample_name) in enumerate(
            zip(resolved_bam_files, self.sample_names)
        ):
            if completed[sample_idx]:
                logger.info(f"Skipping completed sample '{sample_name}'")
                continue
            pending.append((sample_idx, bam_file, sample_name))

        if self.n_samples == 1 and pending:
            # Single-sample store: write directly into the final arrays, no temp store needed.
            sample_idx, bam_file, sample_name = pending[0]
            self._write_sample_direct(sample_idx, bam_file, sample_name)
        elif self.sample_bam_map is not None and pending:
            # MCC: all virtual samples share the same BAM file(s). Group pending
            # samples by BAM path and process each BAM once in a single pass.
            pending_by_bam: dict[str, list[tuple[int, str, str]]] = {}
            for idx, bf, sn in pending:
                pending_by_bam.setdefault(bf, []).append((idx, bf, sn))
            for bam_file, bam_pending in pending_by_bam.items():
                self._process_mcc_samples(bam_file, bam_pending)
        else:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_dir = Path(tmpdir)
                pending_with_paths = [
                    (idx, bf, sn, tmp_dir / f"sample_{idx}.zarr") for idx, bf, sn in pending
                ]
                for sample_idx, bam_file, sample_name, tmp_path in pending_with_paths:
                    self._write_sample_store(
                        bam_file,
                        sample_name,
                        tmp_path,
                        max_workers=1,
                    )
                    tmp_root = zarr.open_group(str(tmp_path), mode="r")
                    self._write_sample_to_final(sample_idx, tmp_root)

        all_sparsity = self.meta["sparsity"][:]
        if np.isfinite(all_sparsity).any():
            self.root.attrs["average_sparsity"] = float(np.nanmean(all_sparsity))

    @classmethod
    def from_bam_files(
        cls,
        bam_files: list[str],
        store_path: Path | str | None = None,
        *,
        chromsizes: str | Path | dict[str, int] | None = None,
        metadata: pd.DataFrame | Path | str | list[Path | str] | None = None,
        bam_sample_names: list[str] | None = None,
        bam_filters: bamnado.ReadFilter | list[bamnado.ReadFilter] | None = None,
        coverage_type: CoverageType
        | list[CoverageType]
        | dict[str, CoverageType] = CoverageType.UNSTRANDED,
        count_fragments: bool = False,
        filter_chromosomes: bool = True,
        overwrite: bool = True,
        resume: bool = False,
        sample_column: str = "sample_id",
        chunk_len: int | None = None,
        construction_compression: str = DEFAULT_CONSTRUCTION_COMPRESSION,
        local_staging: bool = False,
        staging_dir: Path | str | None = None,
        max_workers: int = 1,
        log_file: Path | None = None,
        test: bool = False,
    ) -> "BamStore":
        if log_file is not None:
            from quantnado.utils import setup_logging

            setup_logging(Path(log_file), verbose=False)

        if chromsizes is None:
            if not bam_files:
                raise ValueError("bam_files list is empty; cannot extract chromsizes from BAM.")
            logger.info(f"Extracting chromsizes from {bam_files[0]}")
            chromsizes_raw = _get_chromsizes_from_bam(bam_files[0])
        else:
            chromsizes_raw = chromsizes

        chromsizes_dict = _parse_chromsizes(
            chromsizes_raw, filter_chromosomes=filter_chromosomes, test=test
        )

        if bam_sample_names is not None:
            if len(bam_sample_names) != len(bam_files):
                raise ValueError(
                    f"bam_sample_names length ({len(bam_sample_names)}) != bam_files length ({len(bam_files)})"
                )
            sample_names = bam_sample_names
        else:
            sample_names = [Path(f).stem for f in bam_files]

        if isinstance(coverage_type, CoverageType):
            per_file_coverage_types: list[CoverageType] = [coverage_type] * len(bam_files)
        elif isinstance(coverage_type, list):
            per_file_coverage_types = coverage_type
        elif isinstance(coverage_type, dict):
            per_file_coverage_types = [
                coverage_type.get(s, CoverageType.UNSTRANDED) for s in sample_names
            ]
        else:
            per_file_coverage_types = [CoverageType.UNSTRANDED] * len(bam_files)

        sample_bam_map: dict[str, str] | None = None
        viewpoint_map: dict[str, str] | None = None
        viewpoints: set[str] | None = None
        if any(bt == CoverageType.MICRO_CAPTURE_C for bt in per_file_coverage_types):
            _sample_bam_map: dict[str, str] = {}
            _viewpoint_map: dict[str, str] = {}
            _viewpoints: set[str] = set()
            expanded_sample_names: list[str] = []
            expanded_coverage_type: dict[str, CoverageType] = {}
            for bam_file, sample_name, bt in zip(bam_files, sample_names, per_file_coverage_types):
                if bt == CoverageType.MICRO_CAPTURE_C:
                    for vp in _get_viewpoints_from_mcc_bam(bam_file):
                        virtual = f"{sample_name}_{vp}"
                        expanded_sample_names.append(virtual)
                        _sample_bam_map[virtual] = str(bam_file)
                        _viewpoint_map[virtual] = vp
                        expanded_coverage_type[virtual] = CoverageType.MICRO_CAPTURE_C
                        _viewpoints.add(vp)
                else:
                    expanded_sample_names.append(sample_name)
                    _sample_bam_map[sample_name] = str(bam_file)
                    expanded_coverage_type[sample_name] = bt
            sample_names = expanded_sample_names
            coverage_type = expanded_coverage_type
            sample_bam_map = _sample_bam_map
            viewpoint_map = _viewpoint_map
            viewpoints = _viewpoints

        if store_path is None:
            raise ValueError("store_path must be provided.")

        final_store_path = cls._normalize_path(store_path)
        staging_enabled = local_staging or staging_dir is not None

        if resume and staging_enabled:
            raise ValueError("resume=True is not supported with local staging")

        if staging_enabled and final_store_path.exists() and not overwrite:
            raise FileExistsError(
                f"Store already exists at {final_store_path}; set overwrite=True to publish staged output"
            )

        build_store_path = (
            _build_staging_store_path(final_store_path, staging_dir)
            if staging_enabled
            else final_store_path
        )
        resolved_chunk_len = _resolve_chunk_len(
            chromsizes_dict, final_store_path, chunk_len, len(sample_names)
        )

        if staging_enabled:
            logger.info(f"Building under staging path {build_store_path}")

        try:
            store = cls(
                store_path=build_store_path,
                chromsizes=chromsizes_dict,
                sample_names=sample_names,
                chunk_len=resolved_chunk_len,
                construction_compression=construction_compression,
                overwrite=True if staging_enabled else overwrite,
                resume=False if staging_enabled else resume,
                coverage_type=coverage_type,
                bam_filter=bam_filters,
                count_fragments=count_fragments,
                viewpoints=viewpoints,
                sample_bam_map=sample_bam_map,
                viewpoint_map=viewpoint_map,
            )
            if sample_bam_map is not None:
                store.process_samples(max_workers=max_workers)
            else:
                store.process_samples(bam_files, max_workers=max_workers)

            if metadata is not None:
                if isinstance(metadata, list):
                    metadata_df = cls._combine_metadata_files(metadata)
                elif isinstance(metadata, (str, Path)):
                    metadata_df = pd.read_csv(metadata)
                else:
                    metadata_df = metadata
                store.set_metadata(metadata_df, sample_column=sample_column)

            if staging_enabled:
                _publish_staged_store(build_store_path, final_store_path)
                logger.info(f"Published staged dataset to {final_store_path}")
                zarr.consolidate_metadata(str(final_store_path))
                return cls.open(final_store_path, read_only=False)

            zarr.consolidate_metadata(str(store.store_path))
            return store
        except Exception:
            if staging_enabled:
                _delete_store_path(build_store_path)
            raise

    @staticmethod
    def _combine_metadata_files(metadata_files: list[Path | str]) -> pd.DataFrame:
        if not metadata_files:
            raise ValueError("No metadata files provided")
        frames = []
        for file_path in metadata_files:
            file_path = Path(file_path)
            if not file_path.exists():
                logger.warning(f"Metadata file not found: {file_path}, skipping")
                continue
            frames.append(pd.read_csv(file_path))
        if not frames:
            raise ValueError("No valid metadata files found")
        all_columns = set()
        for df in frames:
            for col in df.columns:
                if "r1" not in col.lower() and "r2" not in col.lower():
                    all_columns.add(col)
        for i in range(len(frames)):
            for col in all_columns:
                if col not in frames[i].columns:
                    frames[i][col] = ""
            frames[i] = frames[i][list(all_columns)]
        return pd.concat(frames, ignore_index=True)

    @property
    def dataset(self):
        return self.root

    @property
    def n_completed(self) -> int:
        return int(self.completed_mask.sum())

    @classmethod
    def metadata_from_csv(cls, path: Path | str, **kwargs) -> pd.DataFrame:
        return pd.read_csv(path, **kwargs)

    @classmethod
    def metadata_from_json(cls, path: Path | str) -> pd.DataFrame:
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            return pd.DataFrame(data)
        return pd.DataFrame([data])
