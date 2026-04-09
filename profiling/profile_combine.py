import cProfile
import glob
import io
import pstats
import tracemalloc
from pathlib import Path

from loguru import logger

from quantnado.dataset.combine_multiomics import combine_multiomics_stores

out_dir = "profiling_output"

log_format = "{time:YYYY-MM-DD HH:mm:ss} [{level: <8}] {message}"
log_file = Path(out_dir) / "profile_combine_stores.log"
logger.remove()
handler_id = logger.add(log_file, level="DEBUG", format=log_format, mode="w", colorize=False)

def profile_combine_stores(store_paths: list[str | Path], output_path: str | Path, overwrite: bool = True):   
    tracemalloc.start()
    pr = cProfile.Profile()
    pr.enable()

    combined = combine_multiomics_stores(
        stores=store_paths,
        output_path=output_path,
        overwrite=True,
    )

    pr.disable()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("tottime")
    ps.print_stats(50)

    logger.debug(f"Peak memory: {peak / 1024**3:.2f} GB")
    logger.debug("\n\n====== cProfile (top 50) ======\n" + s.getvalue())
    logger.remove(handler_id)

    print(f"✓ Done: {len(store_paths)} stores combined")
    print(f"✓ Peak memory: {peak / 1024**3:.2f} GB")

    return combined

if __name__ == "__main__":
    samples_dir = Path(out_dir) / "samples"
    output_path = Path(out_dir) / "dataset.zarr"


    logger.info("Profiling combine_multiomics_stores")

    store_paths = sorted(glob.glob(str(samples_dir / "*.zarr")))
    logger.info(f"_paths to combine:\n  - {'\n  - '.join(store_paths)}")

    logger.catch(lambda: profile_combine_stores(store_paths=store_paths, output_path=output_path))
    combined_ds = profile_combine_stores(store_paths=store_paths, output_path=output_path)

