"""BamStore — per-sample Zarr store constructed from a single BAM file.

Store layout (per-sample)::

    {SampleName}.zarr/
    ├── {chrom}/
    │   └── {array_key}  (1, chrom_len)  uint32  chunks=(1, chunk_len)
    ├── metadata/
    │   ├── completed        bool    (1,)
    │   ├── total_reads      int64   (1,)
    │   ├── mean_read_length float32 (1,)
    │   └── sparsity         float32 (1,)
    └── root.attrs: assay, sample, ip, chromsizes, chunk_len, stranded, viewpoints

Array keys:
    ATAC                 → "atac"
    ChIP + ip=H3K27ac   → "chip_h3k27ac"
    CUT&TAG + ip=MLLN   → "cat_mlln"
    RNA (stranded)       → "rna_fwd", "rna_rev"
    MCC                  → "viewpoint_{viewpoint}" per viewpoint
"""

from __future__ import annotations

import hashlib
import shutil
import tempfile
import uuid
from enum import StrEnum
from pathlib import Path

import bamnado
import numpy as np
import pandas as pd
import pysam
import zarr
from loguru import logger
from zarr.codecs import BloscCodec
from zarr.storage import LocalStore, ZipStore

from quantnado.utils import estimate_chunk_len, is_network_fs
from .metadata import _parse_chromsizes, array_key, create_metadata_group


# ---------------------------------------------------------------------------
# Constants / enums
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Compression helpers
# ---------------------------------------------------------------------------


def _resolve_compressors(profile: str | None) -> list[BloscCodec]:
    profile = (profile or DEFAULT_CONSTRUCTION_COMPRESSION).strip().lower()
    if profile in ("none", "uncompressed", "off"):
        return []
    if profile == "fast":
        return [BloscCodec(cname="zstd", clevel=1, shuffle="shuffle")]
    return [BloscCodec(cname="zstd", clevel=3, shuffle="shuffle")]


def _resolve_chunk_len(chromsizes: dict[str, int], store_path: Path, chunk_len: int | None) -> int:
    if chunk_len is not None:
        if chunk_len <= 0:
            raise ValueError("chunk_len must be a positive integer")
        return int(chunk_len)
    probe = store_path if store_path.exists() else store_path.parent
    est = estimate_chunk_len(
        contig_lengths=chromsizes,
        dtype_bytes=np.dtype(CONSTRUCTION_ARRAY_DTYPE).itemsize,
        n_samples=1,
        fs_is_network=is_network_fs(probe),
    )
    resolved = int(est["chunk_len"])
    logger.info("Resolved chunk_len={} ({} chunks)", resolved, est["num_chunks"])
    return resolved


# ---------------------------------------------------------------------------
# BAM helpers
# ---------------------------------------------------------------------------


def _copy_read_filter(rf: bamnado.ReadFilter) -> bamnado.ReadFilter:
    if hasattr(rf, "copy"):
        return rf.copy()
    new_rf = bamnado.ReadFilter()
    for attr in (
        "min_mapq", "proper_pair", "min_length", "max_length", "strand",
        "min_fragment_length", "max_fragment_length", "blacklist_bed",
        "whitelisted_barcodes", "read_group", "filter_tag", "filter_tag_value",
    ):
        setattr(new_rf, attr, getattr(rf, attr))
    return new_rf


def _compute_bam_hash(bam_path: Path | str) -> str:
    h = hashlib.md5()
    try:
        with open(bam_path, "rb") as f:
            h.update(f.read(16384))
    except (FileNotFoundError, PermissionError) as e:
        logger.warning(f"Could not compute hash for {bam_path}: {e}")
        return ""
    return h.hexdigest()


def _collect_bam_stats(bam_file: str) -> tuple[str, int, float]:
    """Return (bam_hash, total_reads, mean_read_length)."""
    bam_hash = _compute_bam_hash(bam_file)
    total_reads = 0
    mean_read_length = float("nan")
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
    return bam_hash, total_reads, mean_read_length


def _get_chromsizes_from_bam(bam_path: Path | str) -> dict[str, int]:
    with pysam.AlignmentFile(str(bam_path), "rb") as sam:
        return {ref: length for ref, length in zip(sam.references, sam.lengths)}


# ---------------------------------------------------------------------------
# MCC helpers
# ---------------------------------------------------------------------------


def _get_viewpoints_from_mcc_bam(
    bam_path: Path | str,
    viewpoint_tag: str = "VP",
    scan_limit: int = 500_000,
) -> list[str]:
    """Return sorted unique viewpoint tag values from the BAM."""
    viewpoints: set[str] = set()
    try:
        with pysam.AlignmentFile(str(bam_path), "rb") as bam:
            for comment in bam.header.get("CO", []):
                prefix = "QuantNado:viewpoints="
                if comment.startswith(prefix):
                    return sorted(comment[len(prefix):].split(","))
            for i, read in enumerate(bam.fetch(until_eof=True)):
                if i >= scan_limit:
                    break
                if read.has_tag(viewpoint_tag):
                    viewpoints.add(read.get_tag(viewpoint_tag))
    except Exception as e:
        if viewpoints:
            logger.debug(f"Partial VP scan for {bam_path}: {e}")
        else:
            logger.warning(f"Could not extract MCC viewpoints from {bam_path}: {e}")
    if not viewpoints:
        raise ValueError(
            f"No MCC viewpoint tags ('{viewpoint_tag}') found in {bam_path} "
            f"(scanned first {scan_limit:,} reads)"
        )
    return sorted(viewpoints)



# ---------------------------------------------------------------------------
# Staging helpers
# ---------------------------------------------------------------------------


def _delete_path(p: Path) -> None:
    if p.exists():
        shutil.rmtree(p) if p.is_dir() else p.unlink()


def _publish_staged(staged: Path, final: Path) -> None:
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = final.parent / f".{final.name}.publishing-{uuid.uuid4().hex}"
    try:
        shutil.copytree(staged, tmp)
        _delete_path(final)
        tmp.rename(final)
    except Exception:
        _delete_path(tmp)
        raise
    finally:
        _delete_path(staged)


# ---------------------------------------------------------------------------
# BamStore
# ---------------------------------------------------------------------------


class BamStore:
    """Per-sample Zarr store for BAM-derived genomic coverage.

    One zarr per BAM file. Each chromosome is a zarr group containing
    ``(1, chrom_len)`` arrays — one row per sample (always 1 for per-sample stores).

    Use :meth:`from_bam_files` to create a new store or :meth:`open` to read one.
    """

    def __init__(
        self,
        store_path: Path | str,
        assay: str,
        sample: str,
        chromsizes: dict[str, int],
        *,
        ip: str | None = None,
        stranded: str | None = None,
        viewpoints: list[str] | None = None,
        chunk_len: int,
        compressors: list,
        overwrite: bool = True,
    ) -> None:
        self.store_path = _normalize_path(store_path)
        self.assay = assay
        self.sample = sample
        self.ip = ip
        self.chromsizes = chromsizes
        self.chromosomes = sorted(chromsizes.keys())
        self.stranded = stranded if isinstance(stranded, str) and stranded in ("R", "F") else None
        self.viewpoints = viewpoints or []
        self.chunk_len = chunk_len
        self.compressors = compressors
        self._array_key = array_key(assay, ip)

        if overwrite:
            _delete_path(self.store_path)

        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        local_store = LocalStore(str(self.store_path))
        self.root = zarr.group(store=local_store, overwrite=True, zarr_format=3)
        self.meta = create_metadata_group(self.root, sample)
        self._init_arrays()
        self.root.attrs.update({
            "assay": assay,
            "sample": sample,
            "ip": ip or "",
            "chromsizes": chromsizes,
            "chunk_len": chunk_len,
            "stranded": stranded or "",
            "viewpoints": self.viewpoints,
        })

    def _init_arrays(self) -> None:
        """Create per-chromosome zarr groups with appropriately-keyed arrays."""
        for chrom, chrom_len in self.chromsizes.items():
            grp = self.root.require_group(chrom)
            if self.assay.upper() == "MCC" or CoverageType.MICRO_CAPTURE_C in self.assay.lower():
                for vp in self.viewpoints:
                    grp.require_array(
                        f"viewpoint_{vp}",
                        shape=(1, chrom_len),
                        chunks=(1, self.chunk_len),
                        dtype=CONSTRUCTION_ARRAY_DTYPE,
                        compressors=self.compressors,
                        fill_value=0,
                        overwrite=True,
                    )
            elif self.stranded:
                for key in ("rna_fwd", "rna_rev"):
                    grp.require_array(
                        key,
                        shape=(1, chrom_len),
                        chunks=(1, self.chunk_len),
                        dtype=CONSTRUCTION_ARRAY_DTYPE,
                        compressors=self.compressors,
                        fill_value=0,
                        overwrite=True,
                    )
            else:
                grp.require_array(
                    self._array_key,
                    shape=(1, chrom_len),
                    chunks=(1, self.chunk_len),
                    dtype=CONSTRUCTION_ARRAY_DTYPE,
                    compressors=self.compressors,
                    fill_value=0,
                    overwrite=True,
                )

    def _process_chromosome(
        self,
        bam_file: str,
        chrom: str,
        chrom_size: int,
        read_filter: bamnado.ReadFilter,
        use_fragment: bool,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Return (fwd_signal, rev_signal | None) for one chromosome.

        For stranded RNA, 'fwd' means forward gene strand:
          - "R" (reverse/dUTP library): fwd gene ← reverse-strand reads, rev gene ← forward-strand reads
          - "F" (forward library):       fwd gene ← forward-strand reads, rev gene ← reverse-strand reads
        """
        if self.stranded:
            # bamnado expects "forward" / "reverse" (full lowercase strings)
            if self.stranded == "R":
                fwd_bam_strand, rev_bam_strand = "reverse", "forward"
            else:  # "F"
                fwd_bam_strand, rev_bam_strand = "forward", "reverse"
            rf_fwd = _copy_read_filter(read_filter)
            rf_fwd.strand = fwd_bam_strand
            rf_rev = _copy_read_filter(read_filter)
            rf_rev.strand = rev_bam_strand
            filters = [rf_fwd, rf_rev]
        else:
            filters = [read_filter]

        signals = []
        for f in filters:
            sig = bamnado.get_signal_for_chromosome(
                bam_path=bam_file,
                chromosome_name=chrom,
                bin_size=BIN_SIZE,
                scale_factor=1.0,
                use_fragment=use_fragment,
                ignore_scaffold_chromosomes=False,
                read_filter=f,
            )
            if sig.shape[0] != chrom_size:
                logger.warning(
                    f"Signal length for {chrom} differs ({sig.shape[0]} vs {chrom_size}); aligning"
                )
                sig = sig[:chrom_size] if sig.shape[0] > chrom_size else np.pad(
                    sig, (0, chrom_size - sig.shape[0])
                )
            signals.append(sig.astype(CONSTRUCTION_ARRAY_DTYPE))

        return (signals[0], signals[1] if self.stranded else None)

    def _write_coverage(self, bam_file: str, read_filter: bamnado.ReadFilter, use_fragment: bool) -> float:
        """Write coverage for all chromosomes. Returns mean sparsity."""
        sparsity_vals = []
        for chrom, chrom_size in self.chromsizes.items():
            fwd, rev = self._process_chromosome(bam_file, chrom, chrom_size, read_filter, use_fragment)
            grp = self.root[chrom]
            if rev is not None:
                grp["rna_fwd"][0, :] = fwd
                grp["rna_rev"][0, :] = rev
                sparsity_vals.append(
                    float((np.sum(fwd == 0) + np.sum(rev == 0)) / (2 * fwd.size) * 100)
                )
            else:
                grp[self._array_key][0, :] = fwd
                sparsity_vals.append(float(np.sum(fwd == 0) / fwd.size * 100))
        return float(np.mean(sparsity_vals)) if sparsity_vals else float("nan")

    def _write_viewpoint_coverage(self, bam_file: str, read_filter: bamnado.ReadFilter, use_fragment: bool) -> float:
        """Write one array per viewpoint per chromosome. Returns mean sparsity.

        Iterates viewpoint × chromosome (not chromosome × viewpoint) so only one
        chromosome-length array is in memory at a time, keeping peak memory ~O(chrom_len)
        instead of O(n_viewpoints × chrom_len).
        """
        sparsity_per_vp: dict[str, list[float]] = {vp: [] for vp in self.viewpoints}
        base_filter = _copy_read_filter(read_filter)
        base_filter.filter_tag = None
        base_filter.filter_tag_value = None

        for vp in self.viewpoints:
            rf = _copy_read_filter(base_filter)
            rf.filter_tag = self._viewpoint_tag
            rf.filter_tag_value = vp
            for chrom, chrom_size in self.chromsizes.items():
                signal = bamnado.get_signal_for_chromosome(
                    bam_path=bam_file,
                    chromosome_name=chrom,
                    bin_size=BIN_SIZE,
                    scale_factor=1.0,
                    use_fragment=use_fragment,
                    ignore_scaffold_chromosomes=False,
                    read_filter=rf,
                )
                if signal.shape[0] != chrom_size:
                    signal = (
                        signal[:chrom_size]
                        if signal.shape[0] > chrom_size
                        else np.pad(signal, (0, chrom_size - signal.shape[0]))
                    )
                arr = signal.astype(np.uint32)
                self.root[chrom][f"viewpoint_{vp}"][0, :] = arr
                sparsity_per_vp[vp].append(float(np.sum(arr == 0) / arr.size * 100))
                del arr, signal

        all_sparsity = [v for vals in sparsity_per_vp.values() for v in vals]
        return float(np.mean(all_sparsity)) if all_sparsity else float("nan")

    def _finalise(self, bam_file: str, sparsity: float) -> None:
        bam_hash, total_reads, mean_read_length = _collect_bam_stats(bam_file)
        self.meta["completed"][0] = True
        self.meta["total_reads"][0] = total_reads
        self.meta["mean_read_length"][0] = mean_read_length
        self.meta["sparsity"][0] = sparsity
        zarr.consolidate_metadata(str(self.store_path))
        logger.info(f"Completed {self.sample}: {total_reads:,} mapped reads")

    @classmethod
    def open(cls, store_path: str | Path, read_only: bool = True) -> "BamStore":
        """Open an existing per-sample BamStore zarr (read-only by default)."""
        store_path = _normalize_path(store_path)
        if not store_path.exists():
            raise FileNotFoundError(f"Store not found: {store_path}")
        mode = "r" if read_only else "r+"
        if str(store_path).endswith(".zarr.zip"):
            root = zarr.open_group(store=ZipStore(str(store_path), mode="r"), mode="r")
        else:
            root = zarr.open_group(str(store_path), mode=mode)
        # Return a thin wrapper — just expose the root for QuantNadoDataset
        obj = object.__new__(cls)
        obj.store_path = store_path
        obj.root = root
        attrs = dict(root.attrs)
        obj.assay = attrs.get("assay", "")
        obj.sample = attrs.get("sample", "")
        obj.ip = attrs.get("ip", "")
        obj.chromsizes = {str(k): int(v) for k, v in attrs.get("chromsizes", {}).items()}
        obj.chromosomes = sorted(obj.chromsizes.keys())
        obj.chunk_len = int(attrs.get("chunk_len", 65536))
        raw_stranded = attrs.get("stranded", "")
        obj.stranded = raw_stranded if raw_stranded in ("R", "F") else None
        obj.viewpoints = attrs.get("viewpoints", [])
        obj.meta = root.get("metadata")
        obj._array_key = array_key(obj.assay, obj.ip)
        return obj

    @classmethod
    def from_bam_files(
        cls,
        bam_path: str | Path,
        store_path: Path | str,
        assay: str,
        sample: str,
        chromsizes: str | Path | dict[str, int] | None = None,
        *,
        ip: str | None = None,
        stranded: str | None = None,
        bam_filter: bamnado.ReadFilter | None = None,
        count_fragments: bool = False,
        viewpoint_tag: str = "VP",
        chunk_len: int | None = None,
        construction_compression: str = DEFAULT_CONSTRUCTION_COMPRESSION,
        overwrite: bool = True,
        filter_chromosomes: bool = True,
        test: bool = False,
        staging_dir: Path | str | None = None,
        log_file: Path | None = None,
    ) -> "BamStore":
        """Create a per-sample BamStore zarr from a single BAM file.

        Parameters
        ----------
        bam_path:
            Path to the aligned BAM file.
        store_path:
            Output .zarr directory.
        assay:
            Assay type string (e.g. "ATAC", "ChIP", "CUT&TAG", "RNA", "MCC").
        sample:
            Sample name (used in metadata and as sample coordinate).
        chromsizes:
            Path to .chrom.sizes file, dict, or None to infer from BAM header.
        ip:
            IP target (ChIP/CUT&TAG only). Combined with assay to form array key.
        stranded:
            Strand orientation: "R" (reverse), "F" (forward), or None (unstranded).
            Required for RNA assays.
        """
        if log_file is not None:
            from quantnado.utils import setup_logging
            setup_logging(Path(log_file), verbose=False)

        bam_path = str(bam_path)
        is_mcc = assay.upper() == "MCC"

        if chromsizes is None:
            logger.info(f"Extracting chromsizes from {bam_path}")
            chromsizes = _get_chromsizes_from_bam(bam_path)

        chromsizes_dict = _parse_chromsizes(chromsizes, filter_chromosomes=filter_chromosomes, test=test)
        read_filter = bam_filter or bamnado.ReadFilter()
        use_fragment = count_fragments

        viewpoints: list[str] = []
        if is_mcc:
            logger.info(f"Scanning MCC viewpoints from {bam_path}")
            viewpoints = _get_viewpoints_from_mcc_bam(bam_path, viewpoint_tag=viewpoint_tag)
            logger.info(f"Found {len(viewpoints)} viewpoints: {viewpoints}")

        resolved_chunk_len = _resolve_chunk_len(chromsizes_dict, Path(store_path), chunk_len)
        compressors = _resolve_compressors(construction_compression)

        final_path = _normalize_path(store_path)
        if staging_dir is not None:
            staged_path = Path(staging_dir) / f".{final_path.stem}.staging-{uuid.uuid4().hex}.zarr"
            build_path = staged_path
        else:
            build_path = final_path

        store = cls(
            store_path=build_path,
            assay=assay,
            sample=sample,
            chromsizes=chromsizes_dict,
            ip=ip,
            stranded=stranded,
            viewpoints=viewpoints,
            chunk_len=resolved_chunk_len,
            compressors=compressors,
            overwrite=overwrite,
        )
        store._viewpoint_tag = viewpoint_tag

        if is_mcc:
            sparsity = store._write_viewpoint_coverage(bam_path, read_filter, use_fragment)
        else:
            sparsity = store._write_coverage(bam_path, read_filter, use_fragment)

        store._finalise(bam_path, sparsity)

        if staging_dir is not None:
            _publish_staged(build_path, final_path)
            return cls.open(final_path, read_only=False)

        return store

    @property
    def completed(self) -> bool:
        if self.meta is None:
            return False
        return bool(self.meta["completed"][0])

    @property
    def total_reads(self) -> int:
        if self.meta is None:
            return 0
        return int(self.meta["total_reads"][0])

    @property
    def mean_read_length(self) -> float:
        if self.meta is None:
            return float("nan")
        return float(self.meta["mean_read_length"][0])

    @property
    def sparsity(self) -> float:
        if self.meta is None:
            return float("nan")
        return float(self.meta["sparsity"][0])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_path(path: Path | str) -> Path:
    path = Path(path)
    if str(path).endswith((".zarr.zip", ".zarr")):
        return path
    return path.with_suffix(".zarr")
