import cProfile
import io
import pstats
import shutil
import tracemalloc
from pathlib import Path

import pandas as pd
from loguru import logger

import quantnado as qn

out_dir = "profiling_output"
samples_dir = Path(out_dir) / "samples"
samples_dir.mkdir(parents=True, exist_ok=True)
out_file = Path(out_dir) / "dataset.zarr.zip"
log_format = "{time:YYYY-MM-DD HH:mm:ss} [{level: <8}] {message}"


metadata = (
    qn.metadata_from_seqnado("/Users/catherine/work/datasets", output_dir=out_dir)
    .sort_values("sample_id")
    .groupby("assay")
    .head(2)
    .reset_index(drop=True)
)

mcc_row = pd.DataFrame(
    [
        {
            "assay": "MCC",
            "sample_id": "MCC-OCI-AML3",
            "bam_path": "tests/data/OCI-AML3-control-1-1-A_subsample.bam",
        }
    ]
).reindex(columns=metadata.columns)

metadata = pd.concat([metadata, mcc_row], ignore_index=True)

metadata.to_csv(f"{out_dir}/quantnado_metadata.csv", index=False)

for sample in metadata["sample_id"].unique():
    profile_store = Path(samples_dir / f"{sample}.zarr")
    log_file = Path(out_dir) / f"profile_store_{sample}.log"
    if profile_store.exists():
        print(f"Skipping {sample} — store already exists")
        continue

    print(f"Profiling sample {sample}...")
    shutil.rmtree(profile_store, ignore_errors=True)

    logger.remove()
    handler_id = logger.add(log_file, level="DEBUG", format=log_format, mode="w", colorize=False)

    tracemalloc.start()
    pr = cProfile.Profile()
    pr.enable()
    qn.create_dataset(
        store_dir=profile_store,
        metadata=metadata,
        sample=sample,
        overwrite=True,
        filter_chromosomes=True,
        construction_compression="fast",
        test=True,
    )
    pr.disable()
    _, peak = tracemalloc.get_traced_memory()

    tracemalloc.stop()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("tottime")
    ps.print_stats(3000)
    logger.debug(f"Peak memory: {peak / 1024**3:.2f} GB")
    logger.debug("\n\n====== cProfile ======\n" + s.getvalue())
    logger.remove(handler_id)
