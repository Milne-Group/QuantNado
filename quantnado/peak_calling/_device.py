"""Shared GPU device detection and array transfer utilities for QuantNado peak calling.

All torch imports are deferred to function call time to avoid the ~2 s cold-start
penalty when torch is installed but not needed.
"""

from __future__ import annotations

from functools import lru_cache

from loguru import logger


@lru_cache(maxsize=1)
def get_device(override: str | None = None) -> str:
    """Auto-detect and return the best available compute device.

    Priority: CUDA > MPS > CPU.  MPS availability is validated with a small
    test operation before committing — some older PyTorch builds have incomplete
    MPS support.

    Parameters
    ----------
    override:
        Explicit device string ('cuda', 'mps', 'cpu').  When provided the
        auto-detection is skipped and the override is returned directly
        (after normalising to lowercase).

    Returns
    -------
    str
        One of 'cuda', 'mps', or 'cpu'.
    """
    if override is not None:
        device = override.lower()
        logger.info(f"Using device (manual override): {device}")
        return device

    try:
        import torch

        if torch.cuda.is_available():
            logger.info("Using device: cuda")
            return "cuda"

        if torch.backends.mps.is_available():
            # Validate MPS with a small test op — some builds are incomplete.
            try:
                t = torch.zeros(1, device="mps")
                _ = (t + 1.0).cpu()
                logger.info("Using device: mps")
                return "mps"
            except Exception as exc:
                logger.warning(
                    f"MPS device available but validation failed ({exc}); falling back to CPU."
                )

    except ImportError:
        pass

    logger.info("Using device: cpu")
    return "cpu"


def gpu_available() -> bool:
    """Return True if a non-CPU device is available."""
    return get_device() != "cpu"


def to_torch(arr, device: str, dtype=None):
    """Convert a NumPy array to a PyTorch tensor on *device*.

    Parameters
    ----------
    arr:
        Input NumPy ndarray.
    device:
        Target device string ('cpu', 'cuda', 'mps', …).
    dtype:
        Optional torch dtype (e.g. ``torch.float32``).  Defaults to float32.

    Returns
    -------
    torch.Tensor
    """
    import torch

    if dtype is None:
        dtype = torch.float32
    return torch.from_numpy(arr).to(dtype=dtype, device=device)


def to_numpy(tensor) -> "np.ndarray":  # noqa: F821
    """Convert a PyTorch tensor to a NumPy array (CPU, detached)."""
    return tensor.cpu().detach().numpy()
