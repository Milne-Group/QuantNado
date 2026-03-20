"""Unit tests for quantnado.peak_calling._device."""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh_get_device(*args, **kwargs):
    """Call get_device with a cleared lru_cache each time."""
    from quantnado.peak_calling._device import get_device
    get_device.cache_clear()
    return get_device(*args, **kwargs)


# ---------------------------------------------------------------------------
# get_device tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_get_device_override_cpu():
    result = _fresh_get_device("cpu")
    assert result == "cpu"


@pytest.mark.unit
def test_get_device_override_mps():
    result = _fresh_get_device("mps")
    assert result == "mps"


@pytest.mark.unit
def test_get_device_override_case_insensitive():
    result = _fresh_get_device("CPU")
    assert result == "cpu"


@pytest.mark.unit
def test_get_device_cpu_fallback_no_torch():
    """When torch is not importable, must return 'cpu'."""
    from quantnado.peak_calling._device import get_device
    get_device.cache_clear()

    with patch.dict("sys.modules", {"torch": None}):
        # Reimport inside patch context so the deferred import sees the stub
        import importlib
        import quantnado.peak_calling._device as _dev
        importlib.reload(_dev)
        _dev.get_device.cache_clear()
        result = _dev.get_device()
        assert result == "cpu"
        _dev.get_device.cache_clear()

    # Reload again to restore normal state
    importlib.reload(_dev)


@pytest.mark.unit
def test_get_device_auto_cuda():
    """Auto-detect returns 'cuda' when torch.cuda.is_available() is True."""
    from quantnado.peak_calling._device import get_device
    get_device.cache_clear()

    torch_mock = MagicMock()
    torch_mock.cuda.is_available.return_value = True

    with patch.dict("sys.modules", {"torch": torch_mock}):
        import importlib
        import quantnado.peak_calling._device as _dev
        importlib.reload(_dev)
        _dev.get_device.cache_clear()
        result = _dev.get_device()
        assert result == "cuda"
        _dev.get_device.cache_clear()

    importlib.reload(_dev)


@pytest.mark.unit
def test_get_device_auto_mps():
    """Auto-detect returns 'mps' when CUDA unavailable but MPS validates."""
    from quantnado.peak_calling._device import get_device
    get_device.cache_clear()

    # MPS validation: zeros(1, device='mps') + 1.0
    tensor_mock = MagicMock()
    tensor_mock.__add__ = lambda self, other: MagicMock()
    tensor_mock.cpu = MagicMock(return_value=MagicMock())

    torch_mock = MagicMock()
    torch_mock.cuda.is_available.return_value = False
    torch_mock.backends.mps.is_available.return_value = True
    torch_mock.zeros.return_value = tensor_mock

    with patch.dict("sys.modules", {"torch": torch_mock}):
        import importlib
        import quantnado.peak_calling._device as _dev
        importlib.reload(_dev)
        _dev.get_device.cache_clear()
        result = _dev.get_device()
        assert result == "mps"
        _dev.get_device.cache_clear()

    importlib.reload(_dev)


@pytest.mark.unit
def test_mps_validation_failure_falls_back_to_cpu():
    """Broken MPS (validation raises) must fall back to CPU."""
    from quantnado.peak_calling._device import get_device
    get_device.cache_clear()

    torch_mock = MagicMock()
    torch_mock.cuda.is_available.return_value = False
    torch_mock.backends.mps.is_available.return_value = True
    torch_mock.zeros.side_effect = RuntimeError("MPS broken")

    with patch.dict("sys.modules", {"torch": torch_mock}):
        import importlib
        import quantnado.peak_calling._device as _dev
        importlib.reload(_dev)
        _dev.get_device.cache_clear()
        result = _dev.get_device()
        assert result == "cpu"
        _dev.get_device.cache_clear()

    importlib.reload(_dev)


# ---------------------------------------------------------------------------
# to_torch / to_numpy roundtrip
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_to_torch_roundtrip():
    """numpy → torch → numpy should preserve values."""
    pytest.importorskip("torch")
    from quantnado.peak_calling._device import to_torch, to_numpy

    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    t = to_torch(arr, device="cpu")
    result = to_numpy(t)
    np.testing.assert_array_almost_equal(result, arr)


@pytest.mark.unit
def test_to_torch_dtype_float32():
    pytest.importorskip("torch")
    import torch
    from quantnado.peak_calling._device import to_torch

    arr = np.array([1, 2, 3], dtype=np.int32)
    t = to_torch(arr, device="cpu")
    assert t.dtype == torch.float32


# ---------------------------------------------------------------------------
# gpu_available
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_gpu_available_cpu_only():
    from quantnado.peak_calling._device import get_device, gpu_available
    get_device.cache_clear()
    # Override to cpu explicitly — gpu_available must be False
    get_device("cpu")  # prime cache with "cpu"
    # gpu_available uses get_device() (cached), but since we primed with override
    # we need to reset and use the real auto-detect; instead just test the contract
    get_device.cache_clear()
    # With no GPU hardware in CI, auto-detect should return cpu
    # This assertion holds on machines without GPU; skip on GPU machines
    device = get_device()
    assert gpu_available() == (device != "cpu")
