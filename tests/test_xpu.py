"""Tests for Intel XPU support.

``torch.linalg.eigh`` has no XPU kernel, so on XPU ``_psd_batch`` runs a
batched Jacobi eigensolver on device. The solver is device-agnostic, so its
numerics are checked here on CPU against LAPACK; device dispatch is checked
with mocks so the tests run without an XPU.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
import torch

from rctd import _irwls
from rctd._irwls import _jacobi_eigh, _jacobi_schedule, _psd_batch, _psd_jacobi
from rctd._types import auto_batch_size, resolve_device


@pytest.fixture
def _eager_jacobi(monkeypatch):
    """Run the Jacobi rounds eagerly (no torch.compile warmup in unit tests)."""
    monkeypatch.setattr(_irwls, "_JACOBI_USE_COMPILE", False)


def _make_hessians(N: int, K: int, seed: int = 0) -> torch.Tensor:
    """Indefinite symmetric batch with zero rows and already-diagonal members."""
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(N, K, K, generator=g, dtype=torch.float64)
    H = (A @ A.transpose(-1, -2) / K - 0.3 * torch.eye(K, dtype=torch.float64)) * 1e3
    H[0, 0, :] = 0
    H[0, :, 0] = 0
    H[1] = torch.diag(torch.rand(K, generator=g, dtype=torch.float64))
    H[2] = 0
    return H


@pytest.mark.parametrize("n", [2, 4, 6, 18, 38])
def test_jacobi_schedule_covers_all_pairs(n):
    """Each sweep rotates every index pair exactly once and returns to the
    starting slot order."""
    order, moves = _jacobi_schedule(n, torch.device("cpu"))
    assert len(moves) == n - 1
    current = order.tolist()
    pairs = []
    for move in moves:
        pairs += [tuple(sorted(current[i : i + 2])) for i in range(0, n, 2)]
        current = [current[j] for j in move.tolist()]
    assert current == order.tolist()
    assert len(pairs) == len(set(pairs)) == n * (n - 1) // 2


@pytest.mark.parametrize("K", [3, 13, 22, 38])
def test_jacobi_eigh_matches_lapack(K, _eager_jacobi):
    H = _make_hessians(64, K, seed=K)
    evals, evecs = _jacobi_eigh(H)
    ref = torch.linalg.eigh(H).eigenvalues
    scale = H.abs().amax(dim=(1, 2)).clamp(min=1.0)[:, None]
    np.testing.assert_array_less(((evals.sort(dim=-1).values - ref).abs() / scale).numpy(), 1e-12)
    recon = evecs @ torch.diag_embed(evals) @ evecs.transpose(-1, -2)
    np.testing.assert_array_less(((recon - H).abs().amax(-1) / scale).numpy(), 1e-12)
    eye = torch.eye(K, dtype=H.dtype)
    np.testing.assert_allclose(
        (evecs.transpose(-1, -2) @ evecs).numpy(), eye.expand_as(H).numpy(), atol=1e-12
    )


@pytest.mark.parametrize("K", [5, 22])
def test_psd_jacobi_matches_cpu_psd_batch(K, _eager_jacobi):
    H = _make_hessians(48, K, seed=1)
    H_psd, max_eig = _psd_jacobi(H, 1e-3)
    H_ref, eig_ref = _psd_batch(H)
    np.testing.assert_allclose(H_psd.numpy(), H_ref.numpy(), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(max_eig.numpy(), eig_ref.numpy(), rtol=1e-12, atol=1e-12)


def test_psd_jacobi_handles_non_finite(_eager_jacobi):
    H = _make_hessians(8, 17, seed=3)
    H[3, 0, 0] = float("nan")
    H[5, 4, 4] = float("inf")
    H_psd, max_eig = _psd_jacobi(H, 1e-3)
    assert torch.isfinite(H_psd).all()
    assert torch.isfinite(max_eig).all()
    assert torch.allclose(max_eig[3], torch.tensor(1e-3, dtype=H.dtype))


def test_psd_jacobi_chunks_large_batches(monkeypatch, _eager_jacobi):
    monkeypatch.setattr(_irwls, "_JACOBI_CHUNK", 16)
    H = _make_hessians(40, 9, seed=5)
    H_psd, max_eig = _psd_jacobi(H, 1e-3)
    H_ref, eig_ref = _psd_batch(H)
    assert H_psd.shape == H.shape and max_eig.shape == (40,)
    np.testing.assert_allclose(H_psd.numpy(), H_ref.numpy(), rtol=1e-10, atol=1e-10)


def test_xpu_eigh_threshold_override(monkeypatch):
    """``RCTDConfig.eigh_threshold`` caps the on-device K on XPU as on CUDA."""
    monkeypatch.setattr(_irwls, "_EIGH_THRESHOLD_OVERRIDE", None)
    assert _irwls._xpu_eigh_threshold() == _irwls._XPU_EIGH_THRESHOLD
    monkeypatch.setattr(_irwls, "_EIGH_THRESHOLD_OVERRIDE", 0)
    assert _irwls._xpu_eigh_threshold() == 0


def test_jacobi_not_used_when_compile_disabled(monkeypatch):
    """Eager Jacobi is slower than the CPU offload, so compile=False keeps the
    offload on XPU."""
    monkeypatch.setattr(_irwls, "_USE_COMPILE", False)
    assert not _irwls._jacobi_compile_usable()


# ─── Device resolution / batch sizing ───────────────────────────────────


def test_resolve_device_auto_prefers_cuda_then_xpu():
    with patch("torch.cuda.is_available", return_value=True):
        assert resolve_device("auto").type == "cuda"
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("rctd._types.xpu_available", return_value=True),
    ):
        assert resolve_device("auto").type == "xpu"
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("rctd._types.xpu_available", return_value=False),
    ):
        assert resolve_device("auto").type == "cpu"
    assert resolve_device("xpu").type == "xpu"


def test_auto_batch_size_uses_xpu_memory():
    G, K = 500, 20
    per_pixel = (3 * G + 5 * K + 2 * K * K) * 8
    free = 2_000 * per_pixel  # budget below the XPU cap (= default)
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("rctd._types.xpu_available", return_value=True),
        patch("torch.xpu.mem_get_info", create=True, return_value=(free, free)),
    ):
        assert auto_batch_size(G, K, 8, budget_fraction=1.0) == 2_000
        assert auto_batch_size(G, K, 8, budget_fraction=1.0, device="xpu") == 2_000
        # Plenty of memory: capped at the default on XPU
        assert auto_batch_size(G, K, 8, budget_fraction=1.0, default=1500) == 1500
        # An explicit CPU run on an XPU host keeps the CPU default
        assert auto_batch_size(G, K, 8, device="cpu") == 10000
