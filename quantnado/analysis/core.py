"""QuantNadoDataset — unified read-only xarray view over per-sample zarr stores.

Supports two layouts:

1. **Directory of per-sample zarrs** — ``QuantNadoDataset("dataset/")``
   Each ``.zarr`` has ``root.attrs["assay"]`` and ``root.attrs["sample"]``.

2. **Combined zarr** — ``QuantNadoDataset("dataset/combined.zarr")``
   Written by :meth:`QuantNadoDataset.combine`.

Both expose the same API.  Auto-detected on open.
"""

from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence

import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from loguru import logger


# Keys that are zarr infrastructure — excluded when listing chromosomes / assays
_META_KEYS = frozenset({"metadata"})
_COVERAGE_COLLAPSE_KEYS = frozenset({"atac", "chip", "cat"})


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
    return sorted(
        k for k in root.keys()
        if k not in _META_KEYS and isinstance(root[k], zarr.Group)
    )


def _assay_keys(chrom_group: zarr.Group) -> list[str]:
    """Array keys inside a chromosome group."""
    return [k for k in chrom_group.keys() if isinstance(chrom_group[k], zarr.Array)]


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
        self.chromsizes: dict[str, int] = {
            str(k): int(v) for k, v in attrs.get("chromsizes", {}).items()
        }
        self.chromosomes = _chrom_keys(root)
        self.chunk_len: int = int(attrs.get("chunk_len", 65536))
        self.viewpoints: list[str] = attrs.get("viewpoints", [])
        meta = root.get("metadata")
        self.completed = bool(meta["completed"][0]) if meta is not None and "completed" in meta else False
        self.total_reads: int = int(meta["total_reads"][0]) if meta is not None and "total_reads" in meta else 0
        self.mean_read_length: float = float(meta["mean_read_length"][0]) if meta is not None and "mean_read_length" in meta else 0.0
        self.sparsity: float = float(meta["sparsity"][0]) if meta is not None and "sparsity" in meta else 0.0

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

    Examples
    --------
    >>> qn = QuantNadoDataset("dataset/")
    >>> region = qn.sel(chrom="chr1", start=1_000_000, end=1_001_000)
    >>> region["atac"].sel(sample="ATAC-SEM-1")
    >>> tree = qn.to_datatree()
    """

    def __init__(self, path: Path | str, annotation: str | Path | None = None) -> None:
        self.path = Path(path)
        self._stores: list[_PerSampleStore] = []
        self._combined_root: zarr.Group | None = None
        self._combined = False
        self._genes_df: pd.DataFrame | None = None
        self._exons_df: pd.DataFrame | None = None
        self._subset_samples: list[str] | None = None

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
            return sorted(set(s.upper() for s in self._combined_root.attrs.get("assay_types", [])))
        return sorted(set(s.assay.upper() for s in self._stores))

    @property
    def array_keys(self) -> list[str]:
        """All zarr data-variable names (e.g. 'atac', 'rna_fwd', 'coverage', 'AF')."""
        if self._combined:
            return list(self._combined_root.attrs.get("array_keys", []))
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
                        [full_mask[name_to_idx[s]] for s in self._subset_samples if s in name_to_idx],
                        dtype=bool,
                    )
                return full_mask
            return np.ones(len(self.sample_names), dtype=bool)
        return np.array([s.completed for s in self._stores], dtype=bool)

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
                    return [all_assays[name_to_idx[s]] for s in self._subset_samples if s in name_to_idx]
                return all_assays
            return [""] * len(self.sample_names)
        result = []
        for store in self._stores:
            if store.viewpoints:
                result.extend([store.assay.upper()] * len(store.viewpoints))
            else:
                result.append(store.assay.upper())
        return result

    # ------------------------------------------------------------------
    # Subset
    # ------------------------------------------------------------------

    def subset(
        self,
        assay: "str | Sequence[str] | None" = None,
        samples: "str | Sequence[str] | None" = None,
    ) -> "QuantNadoDataset":
        """Return a new QuantNadoDataset restricted to the specified assay/samples.

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
            Explicit sample name(s) — takes precedence over *assay*.

        Returns
        -------
        QuantNadoDataset
            A lightweight view over the same stores, filtered to the resolved samples.
        """
        resolved = self._resolve_samples(assay=assay, samples=samples)

        new: QuantNadoDataset = object.__new__(QuantNadoDataset)
        new.path = self.path
        new._combined = self._combined
        new._combined_root = self._combined_root
        new._genes_df = self._genes_df
        new._exons_df = self._exons_df

        if self._combined:
            new._stores = []
            new._subset_samples = resolved
        else:
            resolved_set = set(resolved)
            new._stores = [
                s for s in self._stores
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
            usecols=["gene_id", "gene_name", "transcript_id", "gene_type", "gene_biotype", "exon_number"],
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
                exons_out.append({
                    "start": int(erow["Start"]) + 1,
                    "end": int(erow["End"]),
                    "exon_number": str(erow["exon_number"]) if "exon_number" in erow and pd.notna(erow.get("exon_number")) else None,
                })
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
        ds.attrs.update({
            "gene_name": info["gene_name"],
            "gene_id": info["gene_id"],
            "gene_strand": info["strand"],
            "locus": f"{chrom}:{start}-{end}",
            "exons": info["exons"],
        })
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

        return ds

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
                    vp = key[len("viewpoint_"):]
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
            assay_coord_full = np.array([s.decode() if isinstance(s, bytes) else str(s) for s in raw])
            full_names = [str(s) for s in root.attrs.get("sample_names", [])]
            if full_names and len(full_names) != len(all_samples):
                name_to_idx = {s: i for i, s in enumerate(full_names)}
                assay_coord = np.array([assay_coord_full[name_to_idx[s]] for s in all_samples if s in name_to_idx])
            else:
                assay_coord = assay_coord_full
        else:
            assay_coord = np.array([""] * len(all_samples))

        data_vars: dict[str, xr.DataArray] = {}
        for key in all_assays:
            arr = chrom_grp[key]  # (n_samples_for_assay, chrom_len)
            dask_arr = da.from_zarr(arr)[:, s0:e0]
            sample_labels = key_to_samples.get(key, [f"{key}_{i}" for i in range(dask_arr.shape[0])])
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
        nodes: dict[str, xr.Dataset] = {
            "/": xr.Dataset(attrs={"chromsizes": self.chromsizes})
        }
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
        from .features import load_gtf, extract_feature_ranges, extract_promoters
        from .ranges import extract_signal_into_bins
        import pandas as pd

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

        if feature_type == "promoter":
            features_pr = extract_promoters(
                gtf,
                upstream=upstream if upstream is not None else 1000,
                downstream=downstream if downstream is not None else 200,
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

        # Apply fixed-width or upstream/downstream windowing
        strand_col = next((c for c in ("Strand", "strand") if c in features_df.columns), None)
        strands = features_df[strand_col].fillna("+").astype(str).values if strand_col else None
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
            anchor_pos = ((features_df["start"].values + features_df["end"].values) // 2).astype(int)
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

        # Convert to 1-based intervals
        intervals = [
            (row["chrom"], int(row["start"]) + 1, int(row["end"]))
            for _, row in features_df.iterrows()
        ]

        # Extract signal into bins
        signal_array = extract_signal_into_bins(
            intervals, self, array_key, bin_size, samples
        )

        # Create DataArray
        n_intervals, n_bins, n_samples = signal_array.shape
        interval_ids = np.arange(n_intervals)
        if window_upstream is not None:
            bin_ids = np.arange(n_bins, dtype=np.int64) * bin_size - int(window_upstream)
        else:
            bin_ids = np.arange(n_bins, dtype=np.int64)

        strand_values = strands if strands is not None else np.array(["+"] * n_intervals, dtype=object)

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
    # Combine
    # ------------------------------------------------------------------


    @classmethod
    def combine(
        cls,
        src: Path | str,
        output: Path | str,
        overwrite: bool = True,
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
        """
        from zarr.core.array_spec import ArrayConfig
        from zarr.storage import LocalStore

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
        for store in src_ds._stores:
            for key in store.array_keys():
                key_to_stores.setdefault(key, []).append(store)

        key_to_samples: dict[str, list[str]] = {}
        for key, stores in key_to_stores.items():
            names: list[str] = []
            for store in stores:
                if store.viewpoints and key.startswith("viewpoint_"):
                    vp = key[len("viewpoint_"):]
                    names.append(f"{store.sample}_{vp}")
                else:
                    names.append(store.sample)
            key_to_samples[key] = names
        # Unified coverage spans all samples regardless of assay
        key_to_samples["coverage"] = all_samples
        if "coverage" not in all_array_keys:
            all_array_keys = sorted(set(all_array_keys) | {"coverage"})

        chromsizes = src_ds.chromsizes
        chunk_len = src_ds._stores[0].chunk_len if src_ds._stores else 65536

        for chrom, chrom_len in chromsizes.items():
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
                for row_idx, store in enumerate(present_stores):
                    out_arr[row_idx, :] = store.get_array(chrom, key)[0, :]

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
            cov_row_idx = 0
            for store in src_ds._stores:
                keys_set = set(store.array_keys())
                missing_chrom = chrom not in store.chromosomes
                if store.viewpoints:
                    for vp in store.viewpoints:
                        vp_key = f"viewpoint_{vp}"
                        if not missing_chrom and vp_key in keys_set:
                            cov_arr[cov_row_idx, :] = np.asarray(
                                store.get_array(chrom, vp_key)[0, :], dtype=np.float32
                            )
                        cov_row_idx += 1
                elif missing_chrom:
                    cov_row_idx += 1
                elif "coverage" in keys_set:
                    cov_arr[cov_row_idx, :] = np.asarray(
                        store.get_array(chrom, "coverage")[0, :], dtype=np.float32
                    )
                    cov_row_idx += 1
                elif "rna_fwd" in keys_set:
                    fwd = np.asarray(store.get_array(chrom, "rna_fwd")[0, :], dtype=np.float32)
                    if "rna_rev" in keys_set:
                        fwd += np.asarray(store.get_array(chrom, "rna_rev")[0, :], dtype=np.float32)
                    cov_arr[cov_row_idx, :] = fwd
                    cov_row_idx += 1
                elif "DP" in keys_set:
                    cov_arr[cov_row_idx, :] = np.asarray(
                        store.get_array(chrom, "DP")[0, :], dtype=np.float32
                    )
                    cov_row_idx += 1
                else:
                    first_key = store.array_keys()[0]
                    cov_arr[cov_row_idx, :] = np.asarray(
                        store.get_array(chrom, first_key)[0, :], dtype=np.float32
                    )
                    cov_row_idx += 1

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
        mean_rl_list: list[float] = []
        sparsity_list: list[float] = []
        for s in src_ds._stores:
            n = len(s.viewpoints) if s.viewpoints else 1
            completed_list.extend([s.completed] * n)
            total_reads_list.extend([s.total_reads] * n)
            assay_list.extend([s.assay.upper()] * n)
            mean_rl_list.extend([s.mean_read_length] * n)
            sparsity_list.extend([s.sparsity] * n)
        completed = np.array(completed_list, dtype=bool)
        meta_grp.require_array("completed", shape=(len(all_samples),), dtype=bool, overwrite=True)
        meta_grp["completed"][:] = completed
        meta_grp.require_array("total_reads", shape=(len(all_samples),), dtype=np.int64, overwrite=True)
        meta_grp["total_reads"][:] = np.array(total_reads_list, dtype=np.int64)
        assay_arr = meta_grp.require_array(
            "assay", shape=(len(all_samples),), dtype="str", overwrite=True
        )
        assay_arr[:] = assay_list
        meta_grp.require_array("mean_read_length", shape=(len(all_samples),), dtype=np.float32, overwrite=True)
        meta_grp["mean_read_length"][:] = np.array(mean_rl_list, dtype=np.float32)
        meta_grp.require_array("sparsity", shape=(len(all_samples),), dtype=np.float32, overwrite=True)
        meta_grp["sparsity"][:] = np.array(sparsity_list, dtype=np.float32)

        out_root.attrs.update({
            "assay_types": all_assay_types,
            "array_keys": all_array_keys,
            "sample_names": all_samples,
            "key_to_samples": key_to_samples,
            "chromsizes": chromsizes,
            "chunk_len": chunk_len,
        })

        zarr.consolidate_metadata(str(output_path))
        logger.info(f"Combined {len(src_ds._stores)} stores → {output_path}")
        return cls(output_path)

    # ------------------------------------------------------------------
    # Uniform analysis API helpers
    # ------------------------------------------------------------------

    def _resolve_samples(
        self,
        assay: "str | Sequence[str] | None" = None,
        samples: "str | Sequence[str] | None" = None,
    ) -> "list[str]":
        """Return sample names after applying assay/samples filters."""
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

        if sample_list is not None:
            resolved = [s for s in sample_list if s in self.sample_names]
            if not resolved:
                raise ValueError(f"No requested samples found. Available: {self.sample_names}")
            return resolved
        if assay_list is not None:
            assay_upper = {a.upper() for a in assay_list}
            assay_per_sample = self._get_assay_per_sample()
            resolved = [
                s for s, a in zip(self.sample_names, assay_per_sample)
                if a.upper() in assay_upper
            ]
            if not resolved:
                raise ValueError(f"No samples found for assay='{assay}'. Available assays: {self.assays}")
            return resolved
        return list(self.sample_names)

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
            raise ValueError("This method accepts exactly one modality; pass a single string or a single-item list.")
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
        assay: "str | Sequence[str] | None" = None,
        samples: "str | Sequence[str] | None" = None,
        modality: "str | Sequence[str] | None" = None,
        **kwargs,
    ):
        """Count reads over genomic features (DESeq2-compatible matrix).

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
            data, self, method=method,
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
                data = self._filter_sample_data(data, self._resolve_samples(assay=assay, samples=samples))
            if not any(coord in data.coords for coord in ("contig", "chrom")):
                pca_chromosome = None
        elif isinstance(data_or_query, xr.Dataset):
            data = data_or_query
            if assay is not None or samples is not None:
                data = self._filter_sample_data(data, self._resolve_samples(assay=assay, samples=samples))
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
        return plot_pca_scatter(pca_obj, pca_result, colour_by=colour_by, shape_by=shape_by, **kwargs)

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
            data = self._filter_sample_data(data, self._resolve_samples(assay=assay, samples=samples))

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
            data = self._filter_sample_data(data, self._resolve_samples(assay=assay, samples=samples))

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
            data = self._filter_sample_data(data, self._resolve_samples(assay=assay, samples=samples))

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
                "stranded_coverage" if assay_by_sample.get(sample_name, "").upper() == "RNA"
                else "methylation" if assay_by_sample.get(sample_name, "").upper() == "METH"
                else "variant" if assay_by_sample.get(sample_name, "").upper() == "SNP"
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
            for key, names in sorted(key_to_samples.items(), key=lambda item: item[0] == "coverage"):
                if sample_name in names:
                    return key
            return None
        for store in self._stores:
            if store.sample == sample_name or (
                store.viewpoints and any(
                    f"{store.sample}_{vp}" == sample_name for vp in store.viewpoints
                )
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

    def extract_region(
        self, region: str, variable: str = "AF", samples=None
    ) -> "xr.DataArray":
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

    def _normalise_data(self, data):
        return self._inner.normalise(
            data,
            method=self._method,
            library_sizes=self._lib_sizes,
            feature_lengths=self._feature_lengths,
        )

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
