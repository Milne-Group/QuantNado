import cProfile
import contextlib
import io
import pstats
import shutil
import tracemalloc
from pathlib import Path

import pandas as pd
from loguru import logger

import quantnado as qn
from quantnado.analysis.core import QuantNadoDataset

OUT_DIR = Path("profiling_output")
SAMPLES_DIR = OUT_DIR / "samples"
LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} [{level: <8}] {message}"


@contextlib.contextmanager
def _profiled(log_file: Path, n_stats: int = 50):
    """Context manager: yields (pr, peak_gb) after the block, logs to log_file."""
    logger.remove()
    handler_id = logger.add(log_file, level="DEBUG", format=LOG_FORMAT, mode="w", colorize=False)
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
    metadata = (
        qn.metadata_from_seqnado("/Users/catherine/work/datasets", output_dir=str(OUT_DIR))
        .sort_values("sample_id")
        .groupby("assay")
        .head(2)
        .reset_index(drop=True)
    )
    mcc_row = pd.DataFrame([{
        "assay": "MCC",
        "sample_id": "MCC-OCI-AML3",
        "bam_path": "tests/data/OCI-AML3-control-1-1-A_subsample.bam",
    }]).reindex(columns=metadata.columns)
    metadata = pd.concat([metadata, mcc_row], ignore_index=True)
    metadata.to_csv(OUT_DIR / "quantnado_metadata.csv", index=False)
    return metadata


def profile_stores(metadata: pd.DataFrame) -> None:
    for _, row in metadata.iterrows():
        sample = row["sample_id"]
        assay = row["assay"]
        store_path = SAMPLES_DIR / f"{sample}.zarr"
        log_file = OUT_DIR / f"profile_store_{sample}.log"

        print(f"Profiling {sample}...")
        shutil.rmtree(store_path, ignore_errors=True)

        try:
            with _profiled(log_file, n_stats=3000):
                qn.create_dataset(
                    sample_id=sample,
                    assay=assay,
                    output_path=store_path,
                    bam_path=row.get("bam_path"),
                    methyl_path=row.get("methylation_path"),
                    variants_path=row.get("variant_path"),
                )
            logger.info(f"✓ Successfully profiled {sample}")
        except Exception as e:
            logger.error(f"✗ Failed to profile {sample}: {e}")
            logger.exception("Traceback:")


def profile_combine(samples_dir: Path, output_path: Path) -> QuantNadoDataset:
    log_file = OUT_DIR / "profile_combine_stores.log"
    with _profiled(log_file):
        combined = QuantNadoDataset.combine(src=samples_dir, output=output_path, overwrite=True)
    print(f"✓ Done: combined {len(combined.sample_names)} samples")
    return combined


if __name__ == "__main__":
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    metadata = build_metadata()
    profile_stores(metadata)

    output_path = OUT_DIR / "dataset.zarr"
    combined_qn = profile_combine(SAMPLES_DIR, output_path)
