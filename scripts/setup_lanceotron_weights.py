#!/usr/bin/env python3
"""One-time script to download and convert LanceOtron v5.03 weights for QuantNado.

Produces (in quantnado/peak_calling/static/lanceotron/):
    lanceotron_v5_03.pt       – PyTorch state_dict
    wide_scaler_mean.npy      – (12,)
    wide_scaler_scale.npy     – (12,)
    deep_scaler_mean.npy      – (2000,)
    deep_scaler_scale.npy     – (2000,)

Requirements (not needed at runtime):
    pip install torch h5py

Run from the repo root:
    python scripts/setup_lanceotron_weights.py
"""

import pickle
import sys
import urllib.request
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "quantnado" / "peak_calling" / "static" / "lanceotron"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LANCEOTRON_RAW = (
    "https://raw.githubusercontent.com/LHentges/LanceOtron/master/lanceotron/lanceotron/static/"
)
FILES = {
    "wide_and_deep_fully_trained_v5_03.h5": LANCEOTRON_RAW + "wide_and_deep_fully_trained_v5_03.h5",
    "standard_scaler_wide_v5_03.p": LANCEOTRON_RAW + "standard_scaler_wide_v5_03.p",
    "standard_scaler_deep_v5_03.p": LANCEOTRON_RAW + "standard_scaler_deep_v5_03.p",
}


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


def _download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  already exists: {dest.name}")
        return
    print(f"  downloading {dest.name} …", end="", flush=True)
    urllib.request.urlretrieve(url, dest)
    print(f" {dest.stat().st_size:,} bytes")


# ---------------------------------------------------------------------------
# Scaler extraction — version-agnostic, no sklearn import needed
# ---------------------------------------------------------------------------


def _extract_scaler(pickle_path: Path, prefix: str) -> None:
    """Load a sklearn StandardScaler pickle and save mean_/scale_ as .npy.

    Uses only the stdlib pickle module — no sklearn import required.
    The mean_ and scale_ attributes are plain numpy arrays regardless of
    the sklearn version used to create the pickle.
    """
    with open(pickle_path, "rb") as f:
        obj = pickle.load(f)  # noqa: S301 — trusted file from LanceOtron repo
    mean_ = np.asarray(obj.mean_, dtype=np.float64)
    scale_ = np.asarray(obj.scale_, dtype=np.float64)
    np.save(OUT_DIR / f"{prefix}_scaler_mean.npy", mean_)
    np.save(OUT_DIR / f"{prefix}_scaler_scale.npy", scale_)
    print(f"  {prefix} scaler: mean shape={mean_.shape}, scale shape={scale_.shape}")


# ---------------------------------------------------------------------------
# Keras h5 → PyTorch state_dict conversion
# ---------------------------------------------------------------------------


def _h5_get(f, *path_parts: str) -> np.ndarray:
    """Read a dataset from an open h5py File using slash-joined path parts."""
    key = "/".join(path_parts)
    return f[key][()]


def _convert_weights(h5_path: Path) -> None:
    """Convert Keras .h5 weights to a PyTorch state_dict and save as .pt.

    Layer paths are hardcoded from inspecting the actual h5 file structure:

        conv1d / conv1d_1..4                   — 5 conv layers
        batch_normalization / _1.._7           — 8 BN layers
        dense / dense_1 / dense_2              — 3 numbered dense layers
        shape_classification                   — shape output head
        pvalue_classification                  — pvalue output head
        overall_classification                 — overall output head
    """
    import h5py
    import torch

    sys.path.insert(0, str(REPO_ROOT))
    from quantnado.peak_calling.call_lanceotron_peaks import _build_model

    ModelClass = _build_model()
    model = ModelClass()

    with h5py.File(h5_path, "r") as f:

        def conv(h5_name: str, module) -> None:
            """Load Conv1d weights into a _ConvBlock.
            _ConvBlock.conv is _SamePadConv1d; the actual nn.Conv1d is .conv.conv.
            Keras kernel shape: (k, Cin, Cout) → PyTorch: (Cout, Cin, k).
            """
            k = _h5_get(f, h5_name, h5_name, "kernel:0")
            b = _h5_get(f, h5_name, h5_name, "bias:0")
            module.conv.conv.weight.data = torch.from_numpy(k.transpose(2, 1, 0)).float()
            module.conv.conv.bias.data   = torch.from_numpy(b).float()
            print(f"    conv  {h5_name:30s} kernel{k.shape} → weight{tuple(module.conv.conv.weight.shape)}")

        def bn(h5_name: str, module) -> None:
            """Load BatchNorm1d from Keras BN weights."""
            module.weight.data       = torch.from_numpy(_h5_get(f, h5_name, h5_name, "gamma:0")).float()
            module.bias.data         = torch.from_numpy(_h5_get(f, h5_name, h5_name, "beta:0")).float()
            module.running_mean.data = torch.from_numpy(_h5_get(f, h5_name, h5_name, "moving_mean:0")).float()
            module.running_var.data  = torch.from_numpy(_h5_get(f, h5_name, h5_name, "moving_variance:0")).float()
            print(f"    bn    {h5_name}")

        def dense(h5_name: str, module) -> None:
            """Load Linear: Keras (Cin, Cout) → PyTorch weight (Cout, Cin)."""
            k = _h5_get(f, h5_name, h5_name, "kernel:0")
            b = _h5_get(f, h5_name, h5_name, "bias:0")
            module.weight.data = torch.from_numpy(k.T).float()
            module.bias.data   = torch.from_numpy(b).float()
            print(f"    dense {h5_name:30s} kernel{k.shape} → weight{tuple(module.weight.shape)}")

        # ── Conv layers (architectural order) ────────────────────────
        conv("conv1d",   model.deep_entry)
        conv("conv1d_1", model.deep_blocks[0])
        conv("conv1d_2", model.deep_blocks[1])
        conv("conv1d_3", model.deep_blocks[2])
        conv("conv1d_4", model.deep_blocks[3])

        # ── BatchNorm layers (architectural order) ────────────────────
        bn("batch_normalization",   model.deep_entry.bn)
        bn("batch_normalization_1", model.deep_blocks[0].bn)
        bn("batch_normalization_2", model.deep_blocks[1].bn)
        bn("batch_normalization_3", model.deep_blocks[2].bn)
        bn("batch_normalization_4", model.deep_blocks[3].bn)
        bn("batch_normalization_5", model.deep_dense_bn)
        bn("batch_normalization_6", model.combined1_bn)
        bn("batch_normalization_7", model.combined2_bn)

        # ── Dense / output layers ─────────────────────────────────────
        dense("dense",                   model.deep_dense)
        dense("dense_1",                 model.combined1)
        dense("dense_2",                 model.combined2)
        dense("shape_classification",    model.shape_out)
        dense("pvalue_classification",   model.pvalue_out)
        dense("overall_classification",  model.overall_out)

    out = OUT_DIR / "lanceotron_v5_03.pt"
    torch.save(model.state_dict(), out)
    print(f"\n  saved: {out}  ({out.stat().st_size:,} bytes)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=== LanceOtron weight setup ===")

    # 1. Download
    print("\n[1] Downloading LanceOtron assets …")
    for fname, url in FILES.items():
        _download(url, OUT_DIR / fname)

    # 2. Extract scalers
    print("\n[2] Extracting scaler parameters …")
    _extract_scaler(OUT_DIR / "standard_scaler_wide_v5_03.p", "wide")
    _extract_scaler(OUT_DIR / "standard_scaler_deep_v5_03.p", "deep")

    # 3. Convert model weights
    print("\n[3] Converting Keras weights to PyTorch …")
    _convert_weights(OUT_DIR / "wide_and_deep_fully_trained_v5_03.h5")

    # 4. Remove intermediate downloads — only .pt and .npy are needed at runtime
    print("\n[4] Removing intermediate downloads …")
    for fname in FILES:
        p = OUT_DIR / fname
        if p.exists():
            p.unlink()
            print(f"  removed: {fname}")

    print("\nDone. Assets are in:", OUT_DIR)


if __name__ == "__main__":
    main()
