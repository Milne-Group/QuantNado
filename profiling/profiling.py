import contextlib
import cProfile
import io
import pstats
import re
import shutil
import tracemalloc
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from loguru import logger

import quantnado as qn
from quantnado.analysis.core import QuantNadoDataset

OUT_DIR = Path("profiling/output")
SAMPLES_DIR = OUT_DIR / "samples"

MCC_ROW = {
    "assay": "MCC",
    "sample_id": "MCC-OCI-AML3",
    "bam_path": "tests/data/OCI-AML3-control-1-1-A_subsample.bam",
}


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


def build_metadata() -> pd.DataFrame:
    metadata = qn.metadata_from_seqnado("/Users/catherine/work/datasets")
    metadata = metadata[metadata.sample_id.str.contains("SEM")]
    metadata = metadata[~metadata.sample_id.str.contains("CM")]
    metadata = metadata[metadata.ip.isin(["MLLN", "H3K27ac"]) | metadata.ip.isna()]
    metadata = metadata[metadata.sample_id.str.contains("-1|2")]
    metadata.sort_values("sample_id", ascending=True, inplace=True)

    mcc_row = pd.DataFrame([MCC_ROW]).reindex(columns=metadata.columns)
    metadata = pd.concat([metadata, mcc_row], ignore_index=True)
    metadata.to_csv(OUT_DIR / "quantnado_metadata.csv", index=False)
    return metadata


def profile_stores(metadata: pd.DataFrame, overwrite: bool = False) -> None:
    """
    Profiles the creation of individual sample stores by running qn.create_dataset within a cProfile context and logging to files.
    If a store already exists for a sample and overwrite=False, that sample will be skipped.
    Args:
        metadata: DataFrame containing sample metadata, including columns like 'sample_id', 'assay', and paths to input files.
        overwrite: If False, samples with existing stores will be skipped. If True, existing stores will be overwritten.

    Returns:
        None (stores are created on disk and profiling logs are saved to OUT_DIR)
    """
    for _, row in metadata.iterrows():
        sample = row["sample_id"]
        store_path = SAMPLES_DIR / f"{sample}.zarr"

        if store_path.exists() and not overwrite:
            print(f"Skipping {sample} (store already exists)")
            continue

        print(f"Profiling {sample}...")
        shutil.rmtree(store_path, ignore_errors=True)

        try:
            with _profiled(OUT_DIR / f"profile_store_{sample}.log", n_stats=3000):
                qn.create_dataset(
                    sample_id=sample,
                    assay=row["assay"],
                    output_path=store_path,
                    bam_path=row.get("bam_path"),
                    methyl_path=row.get("methylation_path"),
                    variants_path=row.get("variant_path"),
                    stranded=row.get("stranded", False),
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
    if output_path.exists() and not overwrite:
        print(f"Loading existing combined dataset from {output_path}...")
        return QuantNadoDataset(output_path)
    print("Profiling QuantNadoDataset.combine...")
    with _profiled(OUT_DIR / "profile_combine_stores.log"):
        combined = QuantNadoDataset.combine(src=samples_dir, output=output_path)
    print(f"✓ Done: combined {len(combined.sample_names)} samples")
    return combined


def parse_logs(metadata: pd.DataFrame, output_dir: Path) -> dict:
    logfiles = set(log_file.stem for log_file in output_dir.glob("profile_store_*.log"))
    profile_df = metadata[["sample_id", "assay"]].drop_duplicates().copy()

    for sample_id in metadata.sample_id.unique():
        log_file_name = f"profile_store_{sample_id}.log"
        if log_file_name not in logfiles:
            logger.warning(f"Log file for sample_id {sample_id} not found: {log_file_name}")
            continue
        record = {"sample_id": sample_id}
        with open(output_dir / log_file_name, "r") as f:
            for line in f:
                if m := re.search(r"in (\d+\.\d+) seconds", line):
                    record["seconds"] = float(m.group(1))
                if m := re.search(r"([\d,]+) mapped reads", line):
                    record["mapped_reads"] = float(m.group(1).replace(",", "")) / 1e6
                if m := re.search(r"([\d,]+) variants", line):
                    record["variants"] = float(m.group(1).replace(",", ""))
                if m := re.search(r"Peak memory: (\d+\.\d+) GB", line):
                    record["peak_memory"] = float(m.group(1))

        profile_df = profile_df.merge(pd.DataFrame([record]), on="sample_id", how="left")

    plt.figure(figsize=(5.5, 4))
    sns.scatterplot(data=profile_df, x="seconds", y="mapped_reads", hue="assay", s=50)
    plt.xlabel("Seconds")
    plt.ylabel("Mapped Reads (Millions)")
    plt.title("QuantNado Store Profiling")
    max_sec = profile_df["seconds"].max()
    plt.xticks(range(0, int(max_sec + 60), 60))
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.legend(title="Assay", loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()
    return profile_df


if __name__ == "__main__":
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    metadata = build_metadata()
    profile_stores(metadata, overwrite=False)
    profile_combine(SAMPLES_DIR, OUT_DIR / "dataset.zarr", overwrite=False)
