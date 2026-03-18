"""
Tests for SEACR peak calling from QuantNado vs the original SEACR_1.3.sh script.

CTCF with IgG control (norm, chr1 subset):
  bash SEACR_1.3.sh CTCF_DE_chr1_100Mb.bedgraph.txt IgG_DE_chr1_100Mb.bedgraph.txt norm stringent CTCF_DE_chr1_100Mb
  bash SEACR_1.3.sh CTCF_DE_chr1_100Mb.bedgraph.txt IgG_DE_chr1_100Mb.bedgraph.txt norm relaxed CTCF_DE_chr1_100Mb

SEM samples without IgG control (0.01 FDR, non, chr7/9/13 subset):
  bash SEACR_1.3.sh CT-SEM_H3K27ac_chr7_9_13.bdg 0.01 non stringent CT-SEM_H3K27ac_chr7_9_13
  bash SEACR_1.3.sh CT-SEM_H3K27ac_chr7_9_13.bdg 0.01 non relaxed CT-SEM_H3K27ac_chr7_9_13
  bash SEACR_1.3.sh CT-SEM_MLL_chr7_9_13.bdg 0.01 non stringent CT-SEM_MLL_chr7_9_13
  bash SEACR_1.3.sh CT-SEM_MLL_chr7_9_13.bdg 0.01 non relaxed CT-SEM_MLL_chr7_9_13
"""

import os
import subprocess
import time
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest
import zarr
from zarr.storage import LocalStore
from loguru import logger

from quantnado.peak_calling.call_seacr_peaks import call_seacr_peaks_from_zarr
from quantnado.peak_calling.call_quantile_peaks import call_peaks_from_zarr as call_quantile_peaks_from_zarr

# Setup logging to show only info-level messages and above, without timestamps or other metadata
logger.remove()
logger.add(lambda msg: print(msg, flush=True), level="INFO")

SEACR_DIR = Path(__file__).parent / "seacr_data"
TESTFILES = SEACR_DIR / "Testfiles"
EXP_BG = TESTFILES / "CTCF_DE_chr1_100Mb.bedgraph.txt"
CTRL_BG = TESTFILES / "IgG_DE_chr1_100Mb.bedgraph.txt"
SEM_H3K27AC_BG = TESTFILES / "CT-SEM_H3K27ac_chr7_9_13.bdg"
SEM_MLL_BG = TESTFILES / "CT-SEM_MLL_chr7_9_13.bdg"
SEM_H3K27AC_NAME = "CT-SEM_H3K27ac_chr7_9_13"
SEM_MLL_NAME = "CT-SEM_MLL_chr7_9_13"
FDR_THRESHOLD = 0.01
QUANTILE_TILESIZE = 128
QUANTILE_WINDOW_OVERLAP = 8
QUANTILE_THRESHOLD = 0.98
QUANTILE_MERGE = True
OUTPUT_DIR = SEACR_DIR / "test_output"
BED_DIR = OUTPUT_DIR / "bedfiles_seacr"
PY_BED_DIR = OUTPUT_DIR / "bedfiles_python"
QUANTILE_BED_DIR = OUTPUT_DIR / "bedfiles_quantile"
REPORT_DIR = OUTPUT_DIR / "diff_reports"
SAMPLE_NAME = "CTCF_DE_chr1_100Mb"

OUTPUT_DIR.mkdir(exist_ok=True)
BED_DIR.mkdir(exist_ok=True)
PY_BED_DIR.mkdir(exist_ok=True)
QUANTILE_BED_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

# Timing store populated by each test; written by test_write_benchmark_report()
_TIMINGS: dict[str, float] = {}

# ---------------------------------------------------------------------------
# Shell SEACR tests — write peak files via the original bash script
# ---------------------------------------------------------------------------


def test_seacr_stringent():
    BED_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    subprocess.run(
        [
            "bash",
            str(SEACR_DIR / "SEACR_1.3.sh"),
            str(EXP_BG),
            str(CTRL_BG),
            "norm",
            "stringent",
            str(BED_DIR / f"seacr_{SAMPLE_NAME}"),
        ],
        check=True,
        cwd=str(SEACR_DIR),
        env={**os.environ, "LC_ALL": "C", "LANG": "C"},
    )
    _TIMINGS["shell_CTCF_stringent"] = time.perf_counter() - t0
    out = BED_DIR / f"seacr_{SAMPLE_NAME}.stringent.bed"
    n = sum(1 for _ in out.open())
    logger.info(f"Shell stringent: {n} peaks → {out}  [{_TIMINGS['shell_CTCF_stringent']:.1f}s]")


def test_seacr_relaxed():
    BED_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    subprocess.run(
        [
            "bash",
            str(SEACR_DIR / "SEACR_1.3.sh"),
            str(EXP_BG),
            str(CTRL_BG),
            "norm",
            "relaxed",
            str(BED_DIR / f"seacr_{SAMPLE_NAME}"),
        ],
        check=True,
        cwd=str(SEACR_DIR),
        env={**os.environ, "LC_ALL": "C", "LANG": "C"},
    )
    _TIMINGS["shell_CTCF_relaxed"] = time.perf_counter() - t0
    out = BED_DIR / f"seacr_{SAMPLE_NAME}.relaxed.bed"
    n = sum(1 for _ in out.open())
    logger.info(f"Shell relaxed: {n} peaks → {out}  [{_TIMINGS['shell_CTCF_relaxed']:.1f}s]")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bedgraph_to_array(path: Path) -> tuple[np.ndarray, str]:
    """Convert a single-chromosome bedgraph to a per-base float32 coverage array."""
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["chrom", "start", "end", "value"],
        dtype={"start": np.int32, "end": np.int32, "value": np.float32},
    )
    chrom = str(df["chrom"].iloc[0])
    arr = np.zeros(int(df["end"].max()), dtype=np.float32)
    for start, end, val in zip(
        df["start"].to_numpy(), df["end"].to_numpy(), df["value"].to_numpy()
    ):
        arr[start:end] = val
    return arr, chrom


def _bedgraph_to_arrays(path: Path) -> dict[str, np.ndarray]:
    """Convert a multi-chromosome bedgraph to per-base float32 coverage arrays keyed by chrom."""
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["chrom", "start", "end", "value"],
        dtype={"start": np.int32, "end": np.int32, "value": np.float32},
    )
    result: dict[str, np.ndarray] = {}
    for chrom, grp in df.groupby("chrom", sort=False):
        arr = np.zeros(int(grp["end"].max()), dtype=np.float32)
        for start, end, val in zip(
            grp["start"].to_numpy(), grp["end"].to_numpy(), grp["value"].to_numpy()
        ):
            arr[start:end] = val
        result[str(chrom)] = arr
    return result


def _write_single_sample_zarr(
    store_path: Path,
    chrom: str,
    cov: np.ndarray,
    sample_name: str,
) -> Path:
    """Create a minimal QuantNado-compatible zarr store for one sample."""
    return _write_multi_chrom_zarr(store_path, {chrom: cov}, sample_name)


def _write_multi_chrom_zarr(
    store_path: Path,
    cov_by_chrom: dict[str, np.ndarray],
    sample_name: str,
) -> Path:
    """Create a QuantNado-compatible zarr store for one sample covering multiple chromosomes."""
    store_path = store_path.with_suffix(".zarr")
    if store_path.exists():
        shutil.rmtree(store_path)

    root = zarr.group(store=LocalStore(str(store_path)), overwrite=True, zarr_format=3)
    chromsizes: dict[str, int] = {}
    chromosomes = sorted(cov_by_chrom)

    for chrom, cov in cov_by_chrom.items():
        clen = int(cov.shape[0])
        chunk_len = min(65536, clen)
        root.create_array(
            name=chrom,
            shape=(1, clen),
            chunks=(1, chunk_len),
            dtype=np.float32,
            fill_value=0,
            overwrite=True,
        )
        root[chrom][0, :] = cov.astype(np.float32, copy=False)
        chromsizes[chrom] = clen

    total_reads = max(
        1,
        int(
            sum(np.asarray(cov, dtype=np.float64).sum() for cov in cov_by_chrom.values())
        ),
    )

    meta = root.create_group("metadata")
    meta.create_array(name="completed", shape=(1,), dtype=bool, fill_value=False, overwrite=True)
    meta["completed"][0] = True
    meta.create_array(name="total_reads", shape=(1,), dtype=np.int64, fill_value=0, overwrite=True)
    meta["total_reads"][0] = total_reads

    root.attrs.update(
        {
            "sample_names": [sample_name],
            "chromsizes": chromsizes,
            "chunk_len": min(65536, max(chromsizes.values())),
            "chromosomes": chromosomes,
        }
    )
    return store_path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def coverages():
    exp_cov, chrom = _bedgraph_to_array(EXP_BG)
    ctrl_cov, _ = _bedgraph_to_array(CTRL_BG)
    return exp_cov, ctrl_cov, chrom


@pytest.fixture(scope="module")
def seacr_zarr_paths(coverages):
    exp_cov, ctrl_cov, chrom = coverages
    fixture_dir = OUTPUT_DIR / "seacr_zarr_fixtures"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    exp_store = _write_single_sample_zarr(fixture_dir / "exp", chrom, exp_cov, SAMPLE_NAME)
    ctrl_store = _write_single_sample_zarr(
        fixture_dir / "ctrl", chrom, ctrl_cov, f"{SAMPLE_NAME}_ctrl"
    )
    return exp_store, ctrl_store


@pytest.fixture(scope="module")
def sem_h3k27ac_zarr():
    cov_by_chrom = _bedgraph_to_arrays(SEM_H3K27AC_BG)
    fixture_dir = OUTPUT_DIR / "sem_zarr_fixtures"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    return _write_multi_chrom_zarr(fixture_dir / "sem_h3k27ac", cov_by_chrom, SEM_H3K27AC_NAME)


@pytest.fixture(scope="module")
def sem_mll_zarr():
    cov_by_chrom = _bedgraph_to_arrays(SEM_MLL_BG)
    fixture_dir = OUTPUT_DIR / "sem_zarr_fixtures"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    return _write_multi_chrom_zarr(fixture_dir / "sem_mll", cov_by_chrom, SEM_MLL_NAME)


# ---------------------------------------------------------------------------
# Python SEACR tests — write peak files via the Python implementation
# ---------------------------------------------------------------------------


def test_python_seacr_stringent(seacr_zarr_paths):
    exp_store, ctrl_store = seacr_zarr_paths
    t0 = time.perf_counter()
    result_files = call_seacr_peaks_from_zarr(
        zarr_path=exp_store,
        output_dir=PY_BED_DIR,
        control_zarr_path=ctrl_store,
        norm="norm",
        stringency="stringent",
    )
    _TIMINGS["python_CTCF_stringent"] = time.perf_counter() - t0
    assert result_files, "No output file produced by call_seacr_peaks_from_zarr"

    out = Path(result_files[0])
    assert out.exists(), f"Expected output BED does not exist: {out}"
    line_count = sum(1 for _ in out.open())
    logger.info(
        f"Python stringent (zarr API): {line_count} peaks → {out}  [{_TIMINGS['python_CTCF_stringent']:.1f}s]"
    )


def test_python_seacr_relaxed(seacr_zarr_paths):
    exp_store, ctrl_store = seacr_zarr_paths
    t0 = time.perf_counter()
    result_files = call_seacr_peaks_from_zarr(
        zarr_path=exp_store,
        output_dir=PY_BED_DIR,
        control_zarr_path=ctrl_store,
        norm="norm",
        stringency="relaxed",
    )
    _TIMINGS["python_CTCF_relaxed"] = time.perf_counter() - t0
    assert result_files, "No output file produced by call_seacr_peaks_from_zarr"

    out = Path(result_files[0])
    assert out.exists(), f"Expected output BED does not exist: {out}"
    line_count = sum(1 for _ in out.open())
    logger.info(
        f"Python relaxed (zarr API): {line_count} peaks → {out}  [{_TIMINGS['python_CTCF_relaxed']:.1f}s]"
    )


def _read_raw_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _coord_key(line: str) -> str:
    parts = line.split()
    return "\t".join(parts[:3])


def _run_quantile_peak_call(zarr_path: Path, sample_name: str) -> Path:
    output_dir = QUANTILE_BED_DIR / sample_name
    result_files = call_quantile_peaks_from_zarr(
        zarr_path=zarr_path,
        output_dir=output_dir,
        tilesize=QUANTILE_TILESIZE,
        window_overlap=QUANTILE_WINDOW_OVERLAP,
        quantile=QUANTILE_THRESHOLD,
        merge=QUANTILE_MERGE,
    )
    assert result_files, "No output file produced by call_quantile_peaks_from_zarr"

    out = Path(result_files[0])
    assert out.exists(), f"Expected quantile output BED does not exist: {out}"
    return out


def _write_diff_report(sample_name: str, mode: str) -> Path:
    shell_file = BED_DIR / f"seacr_{sample_name}.{mode}.bed"
    py_file = PY_BED_DIR / f"{sample_name}.{mode}.bed"
    assert shell_file.exists(), f"Shell {mode} output not found: {shell_file}"
    assert py_file.exists(), f"Python {mode} output not found: {py_file}"

    shell_lines = _read_raw_lines(shell_file)
    py_lines = _read_raw_lines(py_file)
    shell_set = set(shell_lines)
    py_set = set(py_lines)

    shared_exact = sorted(shell_set & py_set)
    only_shell = sorted(shell_set - py_set)
    only_py = sorted(py_set - shell_set)

    shell_coords = {_coord_key(line) for line in shell_set}
    py_coords = {_coord_key(line) for line in py_set}
    shared_coords = sorted(shell_coords & py_coords)
    only_shell_coords = sorted(shell_coords - py_coords)
    only_py_coords = sorted(py_coords - shell_coords)

    cols = ["Chromosome", "Start", "End", "AUC", "MaxSignal", "MaxRegion"]
    shell_df = pd.read_csv(shell_file, sep=r"\s+", header=None, names=cols, engine="python")
    py_df = pd.read_csv(py_file, sep=r"\s+", header=None, names=cols, engine="python")
    merged = shell_df.merge(py_df, on=["Chromosome", "Start", "End"], suffixes=("_shell", "_py"))

    if not merged.empty:
        auc_abs = (merged["AUC_py"] - merged["AUC_shell"]).abs()
        sig_abs = (merged["MaxSignal_py"] - merged["MaxSignal_shell"]).abs()
        auc_median = float(auc_abs.median())
        auc_max = float(auc_abs.max())
        sig_median = float(sig_abs.median())
        sig_max = float(sig_abs.max())
        max_region_exact = int((merged["MaxRegion_py"] == merged["MaxRegion_shell"]).sum())
    else:
        auc_median = auc_max = sig_median = sig_max = float("nan")
        max_region_exact = 0

    report_file = REPORT_DIR / f"{sample_name}.{mode}.diff_report.txt"
    only_shell_file = REPORT_DIR / f"{sample_name}.{mode}.only_in_shell.bed"
    only_py_file = REPORT_DIR / f"{sample_name}.{mode}.only_in_python.bed"

    report_lines = [
        f"mode\t{mode}",
        f"shell_file\t{shell_file}",
        f"python_file\t{py_file}",
        f"shell_peak_count\t{len(shell_lines)}",
        f"python_peak_count\t{len(py_lines)}",
        f"shared_exact_lines\t{len(shared_exact)}",
        f"only_in_shell_lines\t{len(only_shell)}",
        f"only_in_python_lines\t{len(only_py)}",
        f"shared_coords\t{len(shared_coords)}",
        f"only_in_shell_coords\t{len(only_shell_coords)}",
        f"only_in_python_coords\t{len(only_py_coords)}",
        f"shared_coord_rows_for_value_compare\t{len(merged)}",
        f"auc_abs_median\t{auc_median}",
        f"auc_abs_max\t{auc_max}",
        f"maxsignal_abs_median\t{sig_median}",
        f"maxsignal_abs_max\t{sig_max}",
        f"maxregion_exact_matches\t{max_region_exact}",
        "",
        "examples_only_in_shell_coords",
    ]
    report_lines.extend(only_shell_coords[:20])
    report_lines.append("")
    report_lines.append("examples_only_in_python_coords")
    report_lines.extend(only_py_coords[:20])

    report_file.write_text("\n".join(report_lines) + "\n")
    only_shell_file.write_text("\n".join(only_shell) + ("\n" if only_shell else ""))
    only_py_file.write_text("\n".join(only_py) + ("\n" if only_py else ""))

    logger.info(f"Wrote {mode} diff report → {report_file}")
    logger.info(f"Wrote shell-only peaks → {only_shell_file}")
    logger.info(f"Wrote python-only peaks → {only_py_file}")
    return report_file


def test_diff_report_stringent():
    report_file = _write_diff_report(SAMPLE_NAME, "stringent")
    assert report_file.exists(), f"Diff report not written: {report_file}"


def test_diff_report_relaxed():
    report_file = _write_diff_report(SAMPLE_NAME, "relaxed")
    assert report_file.exists(), f"Diff report not written: {report_file}"


# ---------------------------------------------------------------------------
# SEM samples — no IgG control, numeric FDR threshold
# ---------------------------------------------------------------------------


def test_seacr_sem_h3k27ac_stringent():
    BED_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    subprocess.run(
        [
            "bash",
            str(SEACR_DIR / "SEACR_1.3.sh"),
            str(SEM_H3K27AC_BG),
            str(FDR_THRESHOLD),
            "non",
            "stringent",
            str(BED_DIR / f"seacr_{SEM_H3K27AC_NAME}"),
        ],
        check=True,
        cwd=str(SEACR_DIR),
        env={**os.environ, "LC_ALL": "C", "LANG": "C"},
    )
    _TIMINGS["shell_SEM_H3K27ac_stringent"] = time.perf_counter() - t0
    out = BED_DIR / f"seacr_{SEM_H3K27AC_NAME}.stringent.bed"
    n = sum(1 for _ in out.open())
    logger.info(
        f"Shell SEM H3K27ac stringent: {n} peaks → {out}  [{_TIMINGS['shell_SEM_H3K27ac_stringent']:.1f}s]"
    )


def test_seacr_sem_h3k27ac_relaxed():
    BED_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    subprocess.run(
        [
            "bash",
            str(SEACR_DIR / "SEACR_1.3.sh"),
            str(SEM_H3K27AC_BG),
            str(FDR_THRESHOLD),
            "non",
            "relaxed",
            str(BED_DIR / f"seacr_{SEM_H3K27AC_NAME}"),
        ],
        check=True,
        cwd=str(SEACR_DIR),
        env={**os.environ, "LC_ALL": "C", "LANG": "C"},
    )
    _TIMINGS["shell_SEM_H3K27ac_relaxed"] = time.perf_counter() - t0
    out = BED_DIR / f"seacr_{SEM_H3K27AC_NAME}.relaxed.bed"
    n = sum(1 for _ in out.open())
    logger.info(
        f"Shell SEM H3K27ac relaxed: {n} peaks → {out}  [{_TIMINGS['shell_SEM_H3K27ac_relaxed']:.1f}s]"
    )


def test_seacr_sem_mll_stringent():
    BED_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    subprocess.run(
        [
            "bash",
            str(SEACR_DIR / "SEACR_1.3.sh"),
            str(SEM_MLL_BG),
            str(FDR_THRESHOLD),
            "non",
            "stringent",
            str(BED_DIR / f"seacr_{SEM_MLL_NAME}"),
        ],
        check=True,
        cwd=str(SEACR_DIR),
        env={**os.environ, "LC_ALL": "C", "LANG": "C"},
    )
    _TIMINGS["shell_SEM_MLL_stringent"] = time.perf_counter() - t0
    out = BED_DIR / f"seacr_{SEM_MLL_NAME}.stringent.bed"
    n = sum(1 for _ in out.open())
    logger.info(
        f"Shell SEM MLL stringent: {n} peaks → {out}  [{_TIMINGS['shell_SEM_MLL_stringent']:.1f}s]"
    )


def test_seacr_sem_mll_relaxed():
    BED_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    subprocess.run(
        [
            "bash",
            str(SEACR_DIR / "SEACR_1.3.sh"),
            str(SEM_MLL_BG),
            str(FDR_THRESHOLD),
            "non",
            "relaxed",
            str(BED_DIR / f"seacr_{SEM_MLL_NAME}"),
        ],
        check=True,
        cwd=str(SEACR_DIR),
        env={**os.environ, "LC_ALL": "C", "LANG": "C"},
    )
    _TIMINGS["shell_SEM_MLL_relaxed"] = time.perf_counter() - t0
    out = BED_DIR / f"seacr_{SEM_MLL_NAME}.relaxed.bed"
    n = sum(1 for _ in out.open())
    logger.info(
        f"Shell SEM MLL relaxed: {n} peaks → {out}  [{_TIMINGS['shell_SEM_MLL_relaxed']:.1f}s]"
    )


def test_python_sem_h3k27ac_stringent(sem_h3k27ac_zarr):
    t0 = time.perf_counter()
    result_files = call_seacr_peaks_from_zarr(
        zarr_path=sem_h3k27ac_zarr,
        output_dir=PY_BED_DIR,
        control_zarr_path=None,
        fdr_threshold=FDR_THRESHOLD,
        norm="non",
        stringency="stringent",
    )
    _TIMINGS["python_SEM_H3K27ac_stringent"] = time.perf_counter() - t0
    assert result_files, "No output file produced by call_seacr_peaks_from_zarr"
    out = Path(result_files[0])
    assert out.exists(), f"Expected output BED does not exist: {out}"
    line_count = sum(1 for _ in out.open())
    logger.info(
        f"Python SEM H3K27ac stringent: {line_count} peaks → {out}  [{_TIMINGS['python_SEM_H3K27ac_stringent']:.1f}s]"
    )


def test_python_sem_h3k27ac_relaxed(sem_h3k27ac_zarr):
    t0 = time.perf_counter()
    result_files = call_seacr_peaks_from_zarr(
        zarr_path=sem_h3k27ac_zarr,
        output_dir=PY_BED_DIR,
        control_zarr_path=None,
        fdr_threshold=FDR_THRESHOLD,
        norm="non",
        stringency="relaxed",
    )
    _TIMINGS["python_SEM_H3K27ac_relaxed"] = time.perf_counter() - t0
    assert result_files, "No output file produced by call_seacr_peaks_from_zarr"
    out = Path(result_files[0])
    assert out.exists(), f"Expected output BED does not exist: {out}"
    line_count = sum(1 for _ in out.open())
    logger.info(
        f"Python SEM H3K27ac relaxed: {line_count} peaks → {out}  [{_TIMINGS['python_SEM_H3K27ac_relaxed']:.1f}s]"
    )


def test_python_sem_mll_stringent(sem_mll_zarr):
    t0 = time.perf_counter()
    result_files = call_seacr_peaks_from_zarr(
        zarr_path=sem_mll_zarr,
        output_dir=PY_BED_DIR,
        control_zarr_path=None,
        fdr_threshold=FDR_THRESHOLD,
        norm="non",
        stringency="stringent",
    )
    _TIMINGS["python_SEM_MLL_stringent"] = time.perf_counter() - t0
    assert result_files, "No output file produced by call_seacr_peaks_from_zarr"
    out = Path(result_files[0])
    assert out.exists(), f"Expected output BED does not exist: {out}"
    line_count = sum(1 for _ in out.open())
    logger.info(
        f"Python SEM MLL stringent: {line_count} peaks → {out}  [{_TIMINGS['python_SEM_MLL_stringent']:.1f}s]"
    )


def test_python_sem_mll_relaxed(sem_mll_zarr):
    t0 = time.perf_counter()
    result_files = call_seacr_peaks_from_zarr(
        zarr_path=sem_mll_zarr,
        output_dir=PY_BED_DIR,
        control_zarr_path=None,
        fdr_threshold=FDR_THRESHOLD,
        norm="non",
        stringency="relaxed",
    )
    _TIMINGS["python_SEM_MLL_relaxed"] = time.perf_counter() - t0
    assert result_files, "No output file produced by call_seacr_peaks_from_zarr"
    out = Path(result_files[0])
    assert out.exists(), f"Expected output BED does not exist: {out}"
    line_count = sum(1 for _ in out.open())
    logger.info(
        f"Python SEM MLL relaxed: {line_count} peaks → {out}  [{_TIMINGS['python_SEM_MLL_relaxed']:.1f}s]"
    )


def test_diff_report_sem_h3k27ac_stringent():
    report_file = _write_diff_report(SEM_H3K27AC_NAME, "stringent")
    assert report_file.exists(), f"Diff report not written: {report_file}"


def test_diff_report_sem_h3k27ac_relaxed():
    report_file = _write_diff_report(SEM_H3K27AC_NAME, "relaxed")
    assert report_file.exists(), f"Diff report not written: {report_file}"


def test_diff_report_sem_mll_stringent():
    report_file = _write_diff_report(SEM_MLL_NAME, "stringent")
    assert report_file.exists(), f"Diff report not written: {report_file}"


def test_diff_report_sem_mll_relaxed():
    report_file = _write_diff_report(SEM_MLL_NAME, "relaxed")
    assert report_file.exists(), f"Diff report not written: {report_file}"


# ---------------------------------------------------------------------------
# Quantile peak calling — SEM samples
# ---------------------------------------------------------------------------


def test_quantile_sem_h3k27ac(sem_h3k27ac_zarr):
    t0 = time.perf_counter()
    out = _run_quantile_peak_call(sem_h3k27ac_zarr, SEM_H3K27AC_NAME)
    _TIMINGS["quantile_SEM_H3K27ac"] = time.perf_counter() - t0
    line_count = sum(1 for _ in out.open())
    assert line_count > 0, f"Quantile peak output is empty: {out}"
    logger.info(
        f"Quantile SEM H3K27ac: {line_count} peaks → {out}  [{_TIMINGS['quantile_SEM_H3K27ac']:.1f}s]"
    )


def test_quantile_sem_mll(sem_mll_zarr):
    t0 = time.perf_counter()
    out = _run_quantile_peak_call(sem_mll_zarr, SEM_MLL_NAME)
    _TIMINGS["quantile_SEM_MLL"] = time.perf_counter() - t0
    line_count = sum(1 for _ in out.open())
    assert line_count > 0, f"Quantile peak output is empty: {out}"
    logger.info(
        f"Quantile SEM MLL: {line_count} peaks → {out}  [{_TIMINGS['quantile_SEM_MLL']:.1f}s]"
    )


# ---------------------------------------------------------------------------
# Benchmark report — must run last
# ---------------------------------------------------------------------------


def test_write_benchmark_report():
    """Write a TSV comparing shell vs Python wall-clock times for each dataset/mode."""
    benchmark_file = OUTPUT_DIR / "benchmark_times.tsv"

    # Group keys into (dataset, mode) pairs
    datasets = [
        ("CTCF", "stringent", "shell_CTCF_stringent", "python_CTCF_stringent"),
        ("CTCF", "relaxed", "shell_CTCF_relaxed", "python_CTCF_relaxed"),
        ("SEM_H3K27ac", "stringent", "shell_SEM_H3K27ac_stringent", "python_SEM_H3K27ac_stringent"),
        ("SEM_H3K27ac", "relaxed", "shell_SEM_H3K27ac_relaxed", "python_SEM_H3K27ac_relaxed"),
        ("SEM_MLL", "stringent", "shell_SEM_MLL_stringent", "python_SEM_MLL_stringent"),
        ("SEM_MLL", "relaxed", "shell_SEM_MLL_relaxed", "python_SEM_MLL_relaxed"),
        ("SEM_H3K27ac", "quantile", None, "quantile_SEM_H3K27ac"),
        ("SEM_MLL", "quantile", None, "quantile_SEM_MLL"),
    ]

    header = "dataset\tmode\tshell_s\tpython_s\tspeedup"
    rows = [header]
    for dataset, mode, shell_key, python_key in datasets:
        shell_t = _TIMINGS.get(shell_key, float("nan")) if shell_key else float("nan")
        python_t = _TIMINGS.get(python_key, float("nan"))
        if shell_key and np.isfinite(shell_t) and np.isfinite(python_t):
            speedup = f"{shell_t / python_t:.2f}x"
        else:
            speedup = "n/a"
        rows.append(f"{dataset}\t{mode}\t{shell_t:.2f}\t{python_t:.2f}\t{speedup}")

    benchmark_file.write_text("\n".join(rows) + "\n")
    logger.info(f"Benchmark report → {benchmark_file}")
    logger.info("\n" + "\n".join(rows))
    assert benchmark_file.exists()
