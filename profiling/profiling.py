import contextlib
import cProfile
import io
import pstats
import re
import shutil
import tracemalloc
from pathlib import Path

import pandas as pd
from loguru import logger

import quantnado as qn
from quantnado.analysis.core import QuantNadoDataset


MCC_ROW = {
    "assay": "MCC",
    "sample_id": "MCC-OCI-AML3",
    "bam_path": "tests/data/OCI-AML3-control-1-1-A_subsample.bam",
}


def _path_size_mb(path: Path) -> float:
    if path.is_file():
        return path.stat().st_size / 1e6
    if path.is_dir():
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1e6
    return 0.0


@contextlib.contextmanager
def _profiled(log_file: Path, n_stats: int = 50):
    """Context manager: profiles the block and logs cProfile + peak memory to log_file."""
    logger.remove()
    handler_id = logger.add(
        log_file,
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} [{level: <8}] {message}",
        mode="w",
        colorize=False,
    )
    try:
        tracemalloc.start()
        pr = cProfile.Profile()
        pr.enable()
        yield pr
        pr.disable()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        s = io.StringIO()
        pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(n_stats)
        logger.debug(f"Peak memory: {peak / 1024**3:.2f} GB")
        logger.debug(f"\n\n====== cProfile (top {n_stats}) ======\n" + s.getvalue())
    finally:
        logger.remove(handler_id)


def build_metadata(output_path: Path) -> pd.DataFrame:
    metadata = qn.metadata_from_seqnado("/Users/catherine/work/datasets")
    metadata = metadata[metadata.sample_id.str.contains("SEM")]
    metadata = metadata[~metadata.sample_id.str.contains("CM")]
    metadata = metadata[metadata.ip.isin(["MLLN", "H3K27ac"]) | metadata.ip.isna()]
    metadata = metadata[metadata.sample_id.str.contains("-1|2")]
    metadata.sort_values("sample_id", ascending=True, inplace=True)

    mcc_row = pd.DataFrame([MCC_ROW]).reindex(columns=metadata.columns)
    metadata = pd.concat([metadata, mcc_row], ignore_index=True)
    metadata.to_csv(output_path / "quantnado_metadata.csv", index=False)
    return metadata


def profile_stores(
    metadata: pd.DataFrame, output_path: Path, overwrite: bool = False, test: bool = False
) -> None:
    """
    Profiles the creation of individual sample stores by running qn.create_dataset within a cProfile context and logging to files.
    If a store already exists for a sample and overwrite=False, that sample will be skipped.
    Args:
        metadata: DataFrame containing sample metadata, including columns like 'sample_id', 'assay', and paths to input files.
        output_path: Path to save the stores (Zarr format).
        overwrite: If False, samples with existing stores will be skipped. If True, existing stores will be overwritten.
        test: If True, only profile a subset of chromosomes.

    Returns:
        None (stores are created on disk and profiling logs are saved to OUT_DIR)
    """
    for _, row in metadata.iterrows():
        sample = row["sample_id"]
        store_path = output_path / f"{sample}.zarr"

        if store_path.exists() and not overwrite:
            print(f"✓ {sample}")
            continue

        shutil.rmtree(store_path, ignore_errors=True)

        try:
            with _profiled(output_path / f"profile_store_{sample}.log", n_stats=3000):
                qn.create_dataset(
                    sample_id=sample,
                    assay=row["assay"],
                    output_path=store_path,
                    bam_path=row.get("bam_path"),
                    methyl_path=row.get("methylation_path"),
                    variants_path=row.get("variant_path"),
                    stranded=row.get("stranded", False),
                    test=test,
                )
            print(f"✓ {sample}")
        except Exception:
            logger.exception(f"✗ Failed to profile {sample}")


def profile_combine(
    samples_dir: Path, output_path: Path, overwrite: bool = False
) -> QuantNadoDataset:
    """
    Profiles QuantNadoDataset.combine by running it within a cProfile context and logging to a file.
    If the combined dataset already exists at output_path and overwrite=False, it will be loaded instead of re-combined.

    Args:
        samples_dir: Directory containing individual sample stores to combine.
        output_path: Path to save the combined dataset (Zarr format).
        overwrite: If False and output_path exists, load the existing dataset instead of recombining.
    Returns:
        The combined QuantNadoDataset.
    """
    log_path = output_path.parent / "profile_combine_stores.log"

    if output_path.exists() and not overwrite:
        print(f"Loading existing combined dataset from {output_path}...")
        try:
            return QuantNadoDataset(output_path)
        except ValueError as exc:
            logger.warning(
                f"Existing combined dataset at {output_path} is invalid or incomplete; "
                f"rebuilding it. Original error: {exc}"
            )
            shutil.rmtree(output_path, ignore_errors=True)
    print("Profiling QuantNadoDataset.combine...")
    with _profiled(log_path):
        combined = QuantNadoDataset.combine(src=samples_dir, output=output_path)
    print(f"✓ Done: combined {len(combined.sample_names)} samples")
    return combined


def parse_logs(
    metadata: pd.DataFrame, output_dir: Path, stores: Path, plot: bool = True
) -> pd.DataFrame:
    log_paths = {
        log_file.name: log_file
        for base_dir in (output_dir, stores)
        for log_file in base_dir.glob("profile_store_*.log")
    }
    profile_df = metadata[["sample_id", "assay"]].drop_duplicates().copy()
    records = []

    for sample_id in metadata.sample_id.unique():
        log_file_name = f"profile_store_{sample_id}.log"
        log_path = log_paths.get(log_file_name)
        if log_path is None:
            logger.warning(f"Log file for sample_id {sample_id} not found: {log_file_name}")
            continue

        record = {"sample_id": sample_id}
        store_path = stores / f"{sample_id}.zarr"
        if store_path.exists():
            record["store_size_mb"] = _path_size_mb(store_path)
        else:
            logger.warning(f"Store for sample_id {sample_id} not found: {store_path.name}")

        with open(log_path, "r") as f:
            for line in f:
                if m := re.search(r"in (\d+\.\d+) seconds", line):
                    record["seconds"] = float(m.group(1))
                if m := re.search(r"([\d,]+) mapped reads", line):
                    record["mapped_reads"] = float(m.group(1).replace(",", "")) / 1e6
                if m := re.search(r"([\d,]+) variants", line):
                    record["variants"] = float(m.group(1).replace(",", "")) / 1e6
                if m := re.search(r"Peak memory: (\d+\.\d+) GB", line):
                    record["peak_memory"] = float(m.group(1))

        records.append(record)

    if records:
        profile_df = profile_df.merge(pd.DataFrame(records), on="sample_id", how="left")

    if not plot:
        return profile_df

    if "seconds" in profile_df.columns:
        reads_df = (
            profile_df.dropna(subset=["seconds", "mapped_reads"])
            if "mapped_reads" in profile_df.columns
            else pd.DataFrame()
        )
        variants_df = (
            profile_df.dropna(subset=["seconds", "variants"])
            if "variants" in profile_df.columns
            else pd.DataFrame()
        )

        if not reads_df.empty or not variants_df.empty:
            import matplotlib.pyplot as plt
            import seaborn as sns
            from matplotlib.lines import Line2D

            fig, ax = plt.subplots(figsize=(6, 4))

            if not reads_df.empty:
                sns.scatterplot(
                    data=reads_df, x="seconds", y="mapped_reads", hue="assay", s=50, ax=ax
                )

            ax.set_xlabel("Seconds")
            ax.set_ylabel("Mapped Reads (Millions)")
            ax.set_title("QuantNado Store Profiling")
            max_sec = profile_df["seconds"].dropna().max()
            if max_sec > 120:
                ax.set_xticks(range(0, int(max_sec + 60), 60))
            ax.grid(True, which="both", ls="--", lw=0.5)

            if not variants_df.empty:
                ax2 = ax.twinx()
                variants_df = variants_df.sort_values("seconds")
                ax2.scatter(
                    variants_df["seconds"],
                    variants_df["variants"],
                    color="black",
                    s=50,
                    zorder=3,
                )
                ax2.set_ylabel("Variants (Millions)")
                ax2.tick_params(axis="y", colors="0.25")
                ax2.spines["right"].set_color("0.4")

            if ax.get_legend() is not None:
                handles, labels = ax.get_legend_handles_labels()
                if not variants_df.empty:
                    handles.append(
                        Line2D(
                            [0],
                            [0],
                            marker="o",
                            linestyle="None",
                            color="black",
                            markerfacecolor="black",
                            markersize=7,
                            label="SNP",
                        )
                    )
                    labels.append("SNP")
                ax.legend(handles, labels, title="Assay", loc="upper left", bbox_to_anchor=(1.2, 1))
            fig.tight_layout()
            plt.show()
        else:
            logger.warning(
                "No parsed log rows contained plottable profiling metrics; skipping plot."
            )
    else:
        logger.warning("No profiling metrics were parsed; skipping plot.")

    return profile_df


if __name__ == "__main__":
    output_path = Path("profiling/output")
    output_path.mkdir(parents=True, exist_ok=True)

    metadata = build_metadata(output_path)
    profile_stores(metadata, output_path, overwrite=False, test=True)
    profile_combine(output_path, output_path / "dataset.zarr", overwrite=False)
