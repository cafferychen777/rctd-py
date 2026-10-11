"""Batched IRWLS solver for RCTD decomposition.

Ports solveIRWLS.weights, solveWLS, get_der_fast, psd from R spacexr.
Uses PyTorch for GPU acceleration.
"""

import warnings

import torch

from rctd._likelihood import _pop_inductor_codegen_failed, calc_q_all
from rctd._simplex import project_simplex, project_simplex_batch


def _get_derivatives(
    S: torch.Tensor,
    Y: torch.Tensor,
    prediction: torch.Tensor,
    Q_mat: torch.Tensor,
    SQ_mat: torch.Tensor,
    x_vals: torch.Tensor,
    bulk_mode: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute gradient and Hessian for IRWLS step.

    Ports get_der_fast from R prob_model.R.

    Args:
        S: design matrix (G, K)
        Y: observed counts (G,)
        prediction: current predicted values (G,)
        Q_mat, SQ_mat, x_vals: likelihood tables
        bulk_mode: if True, use Gaussian approximation

    Returns:
        (grad, hess): gradient (K,) and Hessian (K, K)
    """
    if bulk_mode:
        d1_vec = -2.0 * (torch.log(prediction) - torch.log(Y + 1e-10)) / prediction
        d2_vec = -2.0 * (1.0 - torch.log(prediction) + torch.log(Y + 1e-10)) / prediction**2
    else:
        _, d1_vec, d2_vec = calc_q_all(Y, prediction, Q_mat, SQ_mat, x_vals)

    # R: grad = -d1_vec %*% S  (grad is (K,))
    grad = -(d1_vec @ S)

    # R: hess = S.T @ diag(-d2_vec) @ S
    # Vectorized: H = S.T @ ((-d2_vec)[:, None] * S)
    hess = S.T @ ((-d2_vec)[:, None] * S)

    return grad, hess


def _psd(H: torch.Tensor, epsilon: float = 1e-3) -> torch.Tensor:
    """Project matrix to positive semi-definite.

    Ports psd from R prob_model.R.
    Uses eigendecomposition, clamping eigenvalues to >= epsilon.
    Matches R's fixed epsilon=1e-3; the box-constrained QP solver
    prevents divergence along ill-conditioned directions.
    """
    if H.device.type == "xpu":
        # No XPU eigh kernel; a single K x K matrix is cheapest on CPU LAPACK.
        eigenvalues, eigenvectors = (t.to(H.device) for t in torch.linalg.eigh(H.cpu()))
    else:
        eigenvalues, eigenvectors = torch.linalg.eigh(H)
    eigenvalues = torch.clamp(eigenvalues, min=epsilon)
    return eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.T


def _solve_box_qp(
    D: torch.Tensor,
    d: torch.Tensor,
    lower_bound: torch.Tensor,
    n_sweeps: int = 50,
) -> torch.Tensor:
    """Solve box-constrained QP: min 0.5 x^T D x - d^T x  s.t. x >= lb.

    Matches R's quadprog::solve.QP with only lower-bound constraints.
    Uses Gauss-Seidel coordinate descent which converges for PSD D.

    Args:
        D: PSD matrix (K, K)
        d: linear term (K,)
        lower_bound: lower bounds (K,)
        n_sweeps: number of full coordinate sweeps

    Returns:
        Optimal x (K,)
    """
    K = d.shape[0]
    # Warm start: diagonal approximation (stable for ill-conditioned D).
    # Using solve(D, d) blows up when cond(D) >> 1 (e.g. 5e6 for 45 types),
    # producing delta_w ~ 1e5 that causes unconstrained weights to diverge.
    # R's quadprog::solve.QP (active-set) doesn't have this issue.
    x = torch.maximum(d / torch.diagonal(D), lower_bound)

    for _ in range(n_sweeps):
        for i in range(K):
            # Exact minimization along coordinate i with others fixed:
            # x_i* = (d_i - sum_{j!=i} D_{ij} x_j) / D_{ii}
            residual = d[i] - torch.dot(D[i], x) + D[i, i] * x[i]
            x_i_opt = residual / D[i, i]
            x[i] = torch.max(x_i_opt, lower_bound[i])

    return x


@torch.no_grad()
def solve_irwls(
    S: torch.Tensor,
    Y: torch.Tensor,
    nUMI: float,
    Q_mat: torch.Tensor,
    SQ_mat: torch.Tensor,
    x_vals: torch.Tensor,
    max_iter: int = 50,
    min_change: float = 0.001,
    step_size: float = 0.3,
    constrain: bool = True,
    bulk_mode: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve IRWLS for a single pixel.

    Ports solveIRWLS.weights from R IRWLS.R.

    Args:
        S: design matrix (G, K) = cell_type_profiles * nUMI
        Y: observed counts (G,)
        nUMI: total UMI for this pixel (scalar)
        Q_mat, SQ_mat, x_vals: likelihood tables
        max_iter: maximum iterations
        min_change: convergence tolerance
        step_size: damping factor (alpha=0.3 in R)
        constrain: if True, project weights onto simplex (sum=1, non-neg)
        bulk_mode: if True, use Gaussian likelihood

    Returns:
        (weights, converged): weights (K,) and convergence flag (scalar bool tensor)
    """
    K = S.shape[1]
    dtype = S.dtype
    device = S.device
    # R: if(!bulk_mode) B[B > K_val] <- K_val
    if not bulk_mode:
        k_val = Q_mat.shape[0] - 3
        Y = torch.clamp(Y, max=k_val)
    w = torch.ones(K, dtype=dtype, device=device) / K

    change = 1.0  # initial sentinel ensures at least one iteration
    for _ in range(max_iter):
        if change <= min_change:
            break
        # R: solution <- pmax(initialSol, 0)
        solution = torch.clamp(w, min=0.0)
        # R: prediction = abs(S %*% solution)
        prediction = torch.abs(S @ solution)
        # R: threshold = max(1e-4, nUMI * 1e-7)
        threshold = max(1e-4, float(nUMI) * 1e-7)
        prediction = torch.clamp(prediction, min=threshold)

        grad, hess = _get_derivatives(S, Y, prediction, Q_mat, SQ_mat, x_vals, bulk_mode)
        hess = _psd(hess)

        # R: norm_factor <- norm(D_mat, "2")
        norm_factor = torch.linalg.norm(hess, ord=2)
        norm_factor = max(norm_factor.item(), 1e-10)
        # R: D_mat <- D_mat / norm_factor
        # R: d_vec <- -derivatives$grad / norm_factor  (note: R negates grad)
        D = hess / norm_factor + 1e-7 * torch.eye(K, dtype=dtype, device=device)
        d = -grad / norm_factor

        # Solve box-constrained QP: min 0.5 x^T D x - d^T x  s.t. x >= -solution
        delta_w = _solve_box_qp(D, d, -solution)

        # Damped update (R: solution <- solution + alpha * delta_w)
        w_new = solution + step_size * delta_w

        # For constrained mode, additionally project onto simplex
        if constrain:
            w_new = project_simplex(w_new)

        # R uses norm(as.matrix(x)) which defaults to L1 norm (sum of abs)
        # for a column vector, NOT L2 norm
        change = torch.sum(torch.abs(w_new - w)).item()
        w = w_new

    converged = change <= min_change
    return w, torch.tensor(converged, dtype=torch.bool, device=device)


def _get_derivatives_batch(
    S: torch.Tensor,
    Y: torch.Tensor,
    prediction: torch.Tensor,
    Q_mat: torch.Tensor,
    SQ_mat: torch.Tensor,
    x_vals: torch.Tensor,
    bulk_mode: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched gradient and Hessian computation.

    Args:
        S: (N, G, K) design matrices
        Y: (N, G) observed counts
        prediction: (N, G) current predicted values
        Q_mat, SQ_mat, x_vals: likelihood tables
        bulk_mode: if True, use Gaussian approximation

    Returns:
        (grad, hess): gradient (N, K) and Hessian (N, K, K)
    """
    N, G, K = S.shape
    if bulk_mode:
        d1_vec = -2.0 * (torch.log(prediction) - torch.log(Y + 1e-10)) / prediction
        d2_vec = -2.0 * (1.0 - torch.log(prediction) + torch.log(Y + 1e-10)) / prediction**2
    else:
        # Flatten (N, G) → (N*G,) for calc_q_all, then reshape back
        Y_flat = Y.reshape(-1)
        pred_flat = prediction.reshape(-1)
        _, d1_flat, d2_flat = calc_q_all(Y_flat, pred_flat, Q_mat, SQ_mat, x_vals)
        d1_vec = d1_flat.reshape(N, G)
        d2_vec = d2_flat.reshape(N, G)

    # grad[n] = -(d1[n] @ S[n]) → (N, 1, G) @ (N, G, K) → (N, 1, K) → (N, K)
    grad = -torch.bmm(d1_vec.unsqueeze(1), S).squeeze(1)

    # hess[n] = S[n].T @ diag(-d2[n]) @ S[n]
    # = (S[n] * (-d2[n])[:, None]).T @ S[n]
    # Batched: (N, K, G) @ (N, G, K) where S_weighted = S * (-d2)[:, :, None]
    S_weighted = S * (-d2_vec).unsqueeze(2)  # (N, G, K)
    hess = torch.bmm(S_weighted.transpose(1, 2), S)  # (N, K, K)

    return grad, hess


def _psd_2x2(H: torch.Tensor, epsilon: float = 1e-3) -> tuple[torch.Tensor, torch.Tensor]:
    """Analytical PSD projection for 2×2 symmetric matrices. H: (N, 2, 2).

    Closed-form eigendecomposition avoids cuSOLVER entirely.
    For a symmetric [[a, b], [b, d]]:
      eigenvalues = (a+d)/2 ± sqrt(((a-d)/2)² + b²)
    """
    a = H[:, 0, 0]
    b = H[:, 0, 1]
    d = H[:, 1, 1]

    half_trace = (a + d) * 0.5
    disc = torch.sqrt(((a - d) * 0.5) ** 2 + b * b)

    lam1 = torch.clamp(half_trace - disc, min=epsilon)
    lam2 = torch.clamp(half_trace + disc, min=epsilon)

    # Reconstruct H_psd = V @ diag(lam) @ V.T analytically
    # For 2×2 symmetric, H_psd = lam1 * v1 v1^T + lam2 * v2 v2^T
    # where v1, v2 are normalized eigenvectors.
    # Direct formula: H_psd[i,j] = f(lam1, lam2, a, b, d)
    # Faster: just reconstruct from clamped eigenvalues
    # H_psd = half_trace_new * I + disc_new * [[cos2t, sin2t], [sin2t, -cos2t]]
    # where cos2t = (a-d)/(2*disc), sin2t = b/disc
    safe_disc = torch.clamp(disc, min=1e-30)
    cos2t = (a - d) * 0.5 / safe_disc
    sin2t = b / safe_disc

    half_lam_sum = (lam1 + lam2) * 0.5
    half_lam_diff = (lam2 - lam1) * 0.5

    H_psd = torch.empty_like(H)
    H_psd[:, 0, 0] = half_lam_sum + half_lam_diff * cos2t
    H_psd[:, 0, 1] = half_lam_diff * sin2t
    H_psd[:, 1, 0] = H_psd[:, 0, 1]
    H_psd[:, 1, 1] = half_lam_sum - half_lam_diff * cos2t

    return H_psd, lam2


# cuSOLVER's batched syevBatched has an undocumented batch-size limit on
# certain GPU architectures (e.g. Blackwell / compute capability 12.0,
# confirmed with CUDA 12.8 + PyTorch 2.9).  The exact threshold depends on K
# (e.g. ~31650 for K=3, ~27430 for K=16) but stays above 27000 for K≤16.
# We sub-batch eigh calls at 25000 to stay safely below the limit for all K.
_MAX_EIGH_BATCH: int = 25000


def _eigh_safe(H: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """torch.linalg.eigh with sub-batching to avoid cuSOLVER batch-size limits."""
    N = H.shape[0]
    if N <= _MAX_EIGH_BATCH or H.device.type != "cuda":
        return torch.linalg.eigh(H)
    # Sub-batch to stay under cuSOLVER limit
    all_evals = []
    all_evecs = []
    for start in range(0, N, _MAX_EIGH_BATCH):
        evals, evecs = torch.linalg.eigh(H[start : start + _MAX_EIGH_BATCH])
        all_evals.append(evals)
        all_evecs.append(evecs)
    return torch.cat(all_evals), torch.cat(all_evecs)


# User override for the GPU-eigh K cutoff. None = arch-based default.
# Set by RCTD.__init__ from RCTDConfig.eigh_threshold (issue #22).
_EIGH_THRESHOLD_OVERRIDE: int | None = None


def _cuda_eigh_threshold(device: torch.device) -> int:
    """Per-arch K threshold for staying on GPU eigh vs offloading to CPU OpenBLAS.

    Benchmarks at K=45 on Ampere/L40S showed CPU OpenBLAS beats cuSOLVER's
    batched syevj — hence the historical K<=16 cutoff. On Hopper (sm_90+)
    and Blackwell (sm_100+), cuSOLVER's batched eigh is much faster, and
    CPU offload causes a major perf cliff (8h+ stalls observed at K=78).
    Bump the threshold on newer arches.

    If ``_EIGH_THRESHOLD_OVERRIDE`` is set, it wins over the arch default —
    e.g. ``--eigh-threshold 64`` on V100/L20 to force GPU eigh at K=38
    instead of falling into the (often slower) CPU offload path.
    """
    if device.type != "cuda":
        return 0
    if _EIGH_THRESHOLD_OVERRIDE is not None:
        return _EIGH_THRESHOLD_OVERRIDE
    try:
        major, _ = torch.cuda.get_device_capability(device)
    except Exception:
        return 16
    return 128 if major >= 9 else 16


# Intel XPU: torch.linalg.eigh has no XPU kernel (torch 2.12 silently falls
# back to CPU LAPACK, one matrix at a time), so _psd_batch would pay a
# device->host->device round trip per IRWLS iteration with no speedup. On XPU
# we instead run a batched cyclic Jacobi eigensolver on device. Each round
# applies K/2 disjoint plane rotations to every matrix at once (round-robin
# ordering); the round is compiled with torch.compile, which fuses it into a
# few kernels. Eager-mode rounds launch too many tiny kernels to beat LAPACK,
# so without torch.compile the CPU offload is kept.
_JACOBI_MAX_SWEEPS: int = 30
# Default K cutoff on XPU (Max 1100, float64): compiled Jacobi is ~4x faster
# than the CPU offload at K~20, ~1.3x at K=38 and break-even near K=50.
_XPU_EIGH_THRESHOLD: int = 48
# Below ~1000 matrices a Jacobi call is launch-bound and the CPU offload wins
# (e.g. the 100-pixel batches of sigma estimation).
_JACOBI_MIN_BATCH: int = 1024
_JACOBI_CHUNK: int = 32768  # bounds the (N, K, K) temporaries per call
_JACOBI_SCHEDULES: dict[tuple[int, str], tuple[torch.Tensor, list[torch.Tensor]]] = {}


def _jacobi_schedule(n: int, device: torch.device) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Round-robin pairing for even ``n``.

    Returns the slot order of the first round (pairs sit in adjacent slots
    ``(2i, 2i+1)``) and, per round, the index that moves slots into the next
    round's order. After ``n - 1`` rounds the order is back to the first one.
    """
    key = (n, str(device))
    if key not in _JACOBI_SCHEDULES:
        players = list(range(n))
        orders = []
        for _ in range(n - 1):
            orders.append([p for i in range(n // 2) for p in (players[i], players[n - 1 - i])])
            players = [players[0], players[-1]] + players[1:-1]
        orders.append(orders[0])
        moves = []
        for r in range(n - 1):
            slot = {orig: j for j, orig in enumerate(orders[r])}
            moves.append(torch.tensor([slot[o] for o in orders[r + 1]], device=device))
        _JACOBI_SCHEDULES[key] = (torch.tensor(orders[0], device=device), moves)
    return _JACOBI_SCHEDULES[key]


def _jacobi_round_impl(
    A: torch.Tensor, V: torch.Tensor, move: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """One parallel Jacobi round: rotate every adjacent slot pair (2i, 2i+1).

    A: (N, n, n) symmetric, V: (N, n, n) accumulated eigenvectors (columns),
    move: (n,) slot permutation into the next round's pairing.
    """
    N, n, _ = A.shape
    m = n // 2
    diag = A.diagonal(dim1=1, dim2=2)
    app, aqq = diag[:, 0::2], diag[:, 1::2]
    apq = A.diagonal(offset=1, dim1=1, dim2=2)[:, 0::2]
    # Classical Jacobi rotation zeroing A[p, q]: t = tan(theta), smaller root
    tau = aqq - app
    sign = torch.where(tau >= 0, 1.0, -1.0).to(A.dtype)
    den = tau.abs() + torch.hypot(tau, 2 * apq)
    t = torch.where(
        apq == 0,
        torch.zeros_like(apq),
        2 * apq * sign / torch.where(den == 0, torch.ones_like(den), den),
    )
    c = torch.rsqrt(1 + t * t)
    s = t * c
    # Rows p, q
    R = A.view(N, m, 2, n)
    Ap, Aq = R[:, :, 0], R[:, :, 1]
    c3, s3 = c.unsqueeze(-1), s.unsqueeze(-1)
    A = torch.stack((c3 * Ap - s3 * Aq, s3 * Ap + c3 * Aq), dim=2).view(N, n, n)
    # Columns p, q
    C = A.view(N, n, m, 2)
    Bp, Bq = C[..., 0], C[..., 1]
    c2, s2 = c.unsqueeze(1), s.unsqueeze(1)
    A = torch.stack((c2 * Bp - s2 * Bq, s2 * Bp + c2 * Bq), dim=-1).view(N, n, n)
    W = V.view(N, n, m, 2)
    Vp, Vq = W[..., 0], W[..., 1]
    V = torch.stack((c2 * Vp - s2 * Vq, s2 * Vp + c2 * Vq), dim=-1).view(N, n, n)
    return A[:, move][:, :, move], V[:, :, move]


# Lazy-compile state for the Jacobi round: None = not yet attempted
_JACOBI_USE_COMPILE: bool | None = None

try:
    _jacobi_round_compiled = torch.compile(_jacobi_round_impl, dynamic=True)
except Exception:
    _jacobi_round_compiled = None


def _jacobi_compile_usable() -> bool:
    """True unless torch.compile is disabled or known to fail (see _USE_COMPILE)."""
    return (
        _jacobi_round_compiled is not None
        and _USE_COMPILE is not False
        and _JACOBI_USE_COMPILE is not False
    )


def _jacobi_round(
    A: torch.Tensor, V: torch.Tensor, move: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compiled Jacobi round; falls back to eager if compilation fails."""
    global _JACOBI_USE_COMPILE
    if not _jacobi_compile_usable():
        return _jacobi_round_impl(A, V, move)
    if _JACOBI_USE_COMPILE is True:
        return _jacobi_round_compiled(A, V, move)
    try:
        result = _jacobi_round_compiled(A, V, move)
    except RuntimeError:
        _JACOBI_USE_COMPILE = False
        warnings.warn(
            "torch.compile failed for the XPU Jacobi eigensolver; falling back to "
            "CPU eigh. Use RCTDConfig(compile=False) to suppress.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _jacobi_round_impl(A, V, move)
    if _pop_inductor_codegen_failed():
        _JACOBI_USE_COMPILE = False
        return _jacobi_round_impl(A, V, move)
    _JACOBI_USE_COMPILE = True
    return result


def _jacobi_eigh(H: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched symmetric eigendecomposition by cyclic Jacobi (unsorted).

    H: (N, K, K) symmetric → (eigenvalues (N, K), eigenvectors (N, K, K) as
    columns). Odd K is padded with a decoupled zero row/column, which no
    rotation touches (its rotations are exact identities).
    """
    N, K, _ = H.shape
    n = K + (K % 2)
    A = H.new_zeros(N, n, n)
    A[:, :K, :K] = H
    order, moves = _jacobi_schedule(n, H.device)
    A = A[:, order][:, :, order]
    V = torch.eye(n, dtype=H.dtype, device=H.device)[:, order].expand(N, n, n).contiguous()
    fro = torch.linalg.matrix_norm(A)
    tol = n * torch.finfo(H.dtype).eps
    for _ in range(_JACOBI_MAX_SWEEPS):
        for move in moves:
            A, V = _jacobi_round(A, V, move)
        diag = A.diagonal(dim1=1, dim2=2)
        off = torch.linalg.matrix_norm(A - torch.diag_embed(diag))
        if bool((off <= tol * fro).all()):
            break
    # After a full sweep the slots are back in the first round's order.
    back = torch.argsort(order)
    eigenvalues = A.diagonal(dim1=1, dim2=2)[:, back][:, :K]
    eigenvectors = V[:, :, back][:, :K, :K]
    return eigenvalues, eigenvectors


def _psd_jacobi(H: torch.Tensor, epsilon: float) -> tuple[torch.Tensor, torch.Tensor]:
    """On-device PSD projection via _jacobi_eigh (used on Intel XPU)."""
    K = H.shape[-1]
    # Same guard as the CPU branch: non-finite matrices become epsilon*I.
    bad_mask = (~torch.isfinite(H)).any(dim=-1).any(dim=-1)
    if bad_mask.any():
        H = H.clone()
        H[bad_mask] = epsilon * torch.eye(K, dtype=H.dtype, device=H.device)
    H_psd_parts, max_eig_parts = [], []
    for start in range(0, H.shape[0], _JACOBI_CHUNK):
        eigenvalues, eigenvectors = _jacobi_eigh(H[start : start + _JACOBI_CHUNK])
        eigenvalues = torch.clamp(eigenvalues, min=epsilon)
        scaled = eigenvectors * eigenvalues.unsqueeze(-2).sqrt()
        H_psd_parts.append(scaled @ scaled.transpose(-1, -2))
        max_eig_parts.append(eigenvalues.max(dim=-1).values)
    if len(H_psd_parts) == 1:
        return H_psd_parts[0], max_eig_parts[0]
    return torch.cat(H_psd_parts), torch.cat(max_eig_parts)


def _xpu_eigh_threshold() -> int:
    """K cutoff for the on-device Jacobi eigensolver on Intel XPU.

    ``_EIGH_THRESHOLD_OVERRIDE`` (from ``RCTDConfig.eigh_threshold``) wins over
    the default, as on CUDA.
    """
    if _EIGH_THRESHOLD_OVERRIDE is not None:
        return _EIGH_THRESHOLD_OVERRIDE
    return _XPU_EIGH_THRESHOLD


def _psd_batch(H: torch.Tensor, epsilon: float = 1e-3) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched PSD projection. H: (N, K, K) → (H_psd (N, K, K), max_eig (N,)).

    For K=1, trivial scalar clamp (avoids cuSOLVER entirely).
    For K=2, uses analytical closed-form (no cuSOLVER overhead).
    For K below the per-arch threshold (see ``_cuda_eigh_threshold``), runs
    eigh directly on GPU. For larger K on older arches, offloads to CPU
    OpenBLAS — but **only with OMP_NUM_THREADS bounded**, otherwise BLAS
    oversubscribes all cores and stalls the IRWLS loop.
    """
    K = H.shape[-1]
    if K == 1:
        # 1×1: eigenvalue = the single element; no decomposition needed
        H_psd = torch.clamp(H, min=epsilon)
        max_eig = H_psd[:, 0, 0]
        return H_psd, max_eig
    if K == 2:
        return _psd_2x2(H, epsilon)
    if (
        H.device.type == "xpu"
        and K <= _xpu_eigh_threshold()
        and H.shape[0] >= _JACOBI_MIN_BATCH
        and _jacobi_compile_usable()
    ):
        return _psd_jacobi(H, epsilon)
    if H.device.type == "cuda" and K <= _cuda_eigh_threshold(H.device):
        # Small matrices: GPU eigh avoids CPU↔GPU transfer overhead
        # Guard: cuSOLVER crashes on NaN input instead of propagating it.
        # Replace NaN with epsilon*I to produce a valid (if meaningless) PSD matrix;
        # the IRWLS outer loop handles the resulting near-zero delta_w gracefully.
        nan_mask = torch.isnan(H).any(dim=-1).any(dim=-1)  # (N,)
        if nan_mask.any():
            H = H.clone()
            H[nan_mask] = epsilon * torch.eye(K, dtype=H.dtype, device=H.device)
        eigenvalues, eigenvectors = _eigh_safe(H)
        eigenvalues = torch.clamp(eigenvalues, min=epsilon)
        # Fused reconstruction: V * sqrt(λ) → (V*sqrt(λ)) @ (V*sqrt(λ))^T
        # Equivalent to V @ diag(λ) @ V^T but avoids diag_embed
        scaled = eigenvectors * eigenvalues.unsqueeze(-2).sqrt()  # (N, K, K)
        H_psd = scaled @ scaled.transpose(-1, -2)
        max_eig = eigenvalues[:, -1]
        return H_psd, max_eig
    else:
        # Large K or CPU: offload to CPU for eigh
        orig_device = H.device
        orig_dtype = H.dtype
        H_cpu = H if H.device.type == "cpu" else H.cpu()

        # Guard: LAPACK syevd raises "_LinAlgError (error code: 99)" on
        # non-finite or near-degenerate batches (issue #20 — K=49 doublet
        # mode on Xenium MINOR types). Mirror the GPU branch's NaN guard,
        # extended to ±Inf.
        bad_mask = (~torch.isfinite(H_cpu)).any(dim=-1).any(dim=-1)  # (N,)
        if bad_mask.any():
            H_cpu = H_cpu.clone()
            H_cpu[bad_mask] = epsilon * torch.eye(K, dtype=H_cpu.dtype)

        # Cap intra-op threads to 1 around the eigh call (issue #22). On hosts
        # with many CPU cores, default OpenBLAS thread count oversubscribes
        # under batched syevd and caused a ~91× stall on V100 at K=38 (smei,
        # 3086s → 33.8s with OMP_NUM_THREADS=1). torch.set_num_threads(1)
        # routes through the same OMP layer OpenBLAS uses, so the effect is
        # equivalent — confirmed locally: 1-thread eigh on (5000, 38, 38) is
        # 64% faster than 32-thread on a 64-core host.
        prev_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            try:
                eigenvalues, eigenvectors = torch.linalg.eigh(H_cpu)
            except torch._C._LinAlgError:
                # Add a tiny diagonal jitter and retry. Jitter << epsilon clamp
                # (1e-3), so well-conditioned batch elements are unaffected after
                # the clamp below.
                I_K = torch.eye(K, dtype=H_cpu.dtype)
                for jitter in (1e-6, 1e-4):
                    try:
                        eigenvalues, eigenvectors = torch.linalg.eigh(H_cpu + jitter * I_K)
                        break
                    except torch._C._LinAlgError:
                        continue
                else:
                    # Last resort: replace the whole batch with epsilon*I. The IRWLS
                    # outer loop handles the resulting near-zero delta_w gracefully
                    # (same contract as the GPU NaN path above).
                    H_safe = (epsilon * I_K).expand_as(H_cpu).contiguous()
                    eigenvalues, eigenvectors = torch.linalg.eigh(H_safe)
        finally:
            torch.set_num_threads(prev_threads)

        eigenvalues = torch.clamp(eigenvalues, min=epsilon)
        H_psd = eigenvectors @ torch.diag_embed(eigenvalues) @ eigenvectors.transpose(-1, -2)
        max_eig = eigenvalues[:, -1]
        if H.device.type == "cpu":
            return H_psd, max_eig
        return H_psd.to(device=orig_device, dtype=orig_dtype), max_eig.to(
            device=orig_device, dtype=orig_dtype
        )


def _solve_box_qp_batch_impl(
    D: torch.Tensor,
    d: torch.Tensor,
    lower_bound: torch.Tensor,
    n_sweeps: int = 50,
) -> torch.Tensor:
    """Gauss-Seidel coordinate descent for batched box-constrained QP.

    When compiled via torch.compile(dynamic=True), fuses the inner coordinate
    descent operations into optimized kernels via Triton/Inductor, reducing
    memory traffic and kernel launch overhead.
    """
    K = d.shape[1]
    D_diag = torch.diagonal(D, dim1=-2, dim2=-1)  # (N, K)
    x = torch.maximum(d / D_diag, lower_bound)  # (N, K)

    for _ in range(n_sweeps):
        for i in range(K):
            Dx_i = (D[:, i, :] * x).sum(-1)  # (N,)
            residual = d[:, i] - Dx_i + D[:, i, i] * x[:, i]
            x[:, i] = torch.maximum(residual / D[:, i, i], lower_bound[:, i])

    return x


# Lazy-compile state: None = not yet attempted, True/False = cached result
_USE_COMPILE: bool | None = None

try:
    _solve_box_qp_batch_compiled = torch.compile(_solve_box_qp_batch_impl, dynamic=True)
except Exception:
    _solve_box_qp_batch_compiled = None


@torch.jit.script
def _solve_box_qp_batch_adaptive_jit(
    D: torch.Tensor,
    d: torch.Tensor,
    lower_bound: torch.Tensor,
    n_sweeps: int = 50,
    check_interval: int = 5,
    tol: float = 1e-4,
) -> torch.Tensor:
    """JIT-compiled Gauss-Seidel with batch-level early exit.

    Checks convergence every check_interval sweeps: if the maximum absolute
    change across all pixels and coordinates is below tol, exits early.
    """
    K = d.shape[1]
    D_diag = torch.diagonal(D, dim1=-2, dim2=-1)
    x = torch.maximum(d / D_diag, lower_bound)

    for sweep in range(n_sweeps):
        x_prev = x.clone()
        for i in range(K):
            Dx_i = (D[:, i, :] * x).sum(-1)
            residual = d[:, i] - Dx_i + D[:, i, i] * x[:, i]
            x[:, i] = torch.maximum(residual / D[:, i, i], lower_bound[:, i])

        if (sweep + 1) % check_interval == 0:
            max_change = (x - x_prev).abs().max()
            if max_change < tol:
                break

    return x


def _solve_box_qp_2(
    D: torch.Tensor,
    d: torch.Tensor,
    lower_bound: torch.Tensor,
) -> torch.Tensor:
    """Analytical box-constrained QP for K=2 via Cramer's rule + clamping.

    For 2 variables the unconstrained optimum is D^{-1}d (closed-form).
    Then clamp to box constraints with one correction sweep to handle
    cross-variable coupling from the clamp.
    """
    D00 = D[:, 0, 0]
    D01 = D[:, 0, 1]
    D11 = D[:, 1, 1]
    det = D00 * D11 - D01 * D01
    det = torch.clamp(det.abs(), min=1e-30) * det.sign()  # safe division

    # Unconstrained optimum via Cramer's rule
    x0 = (D11 * d[:, 0] - D01 * d[:, 1]) / det
    x1 = (D00 * d[:, 1] - D01 * d[:, 0]) / det

    # Clamp to box constraints
    x0 = torch.maximum(x0, lower_bound[:, 0])
    x1 = torch.maximum(x1, lower_bound[:, 1])

    # One correction sweep: re-optimize each coordinate given the other's clamped value
    x0_corr = (d[:, 0] - D01 * x1) / D00
    x0 = torch.maximum(x0_corr, lower_bound[:, 0])
    x1_corr = (d[:, 1] - D01 * x0) / D11
    x1 = torch.maximum(x1_corr, lower_bound[:, 1])

    return torch.stack([x0, x1], dim=1)


def _solve_box_qp_batch(
    D: torch.Tensor,
    d: torch.Tensor,
    lower_bound: torch.Tensor,
    n_sweeps: int = 50,
) -> torch.Tensor:
    """Batched box-constrained QP: min 0.5 x^T D x - d^T x  s.t. x >= lb.

    For K=2, uses analytical closed-form (no iterative sweeps needed).
    Otherwise, Gauss-Seidel coordinate descent, vectorized across pixels.
    Uses torch.compile when available; auto-falls back to eager on failure.

    Args:
        D: (N, K, K) PSD matrices
        d: (N, K) linear terms
        lower_bound: (N, K) lower bounds
        n_sweeps: number of coordinate sweeps (ignored for K=2)

    Returns:
        x: (N, K) optimal solutions
    """
    K = d.shape[1]
    if K == 2:
        return _solve_box_qp_2(D, d, lower_bound)

    global _USE_COMPILE
    if _USE_COMPILE is False or _solve_box_qp_batch_compiled is None:
        # TorchScript-fused Gauss-Seidel with batch-level early exit. This
        # path avoids the millions of small CUDA launches the eager Python
        # implementation would dispatch at K>16 (50 sweeps × K coords ×
        # IRWLS iters), which dominates wall time on Hopper/Blackwell.
        return _solve_box_qp_batch_adaptive_jit(D, d, lower_bound, n_sweeps)

    if _USE_COMPILE is True:
        return _solve_box_qp_batch_compiled(D, d, lower_bound, n_sweeps)

    # First call: try compiled, fall back to eager on failure
    try:
        result = _solve_box_qp_batch_compiled(D, d, lower_bound, n_sweeps)
    except RuntimeError:
        _USE_COMPILE = False
        warnings.warn(
            "torch.compile failed for box-QP solver (missing CUDA headers or Triton); "
            "falling back to eager mode. Use RCTDConfig(compile=False) to suppress.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _solve_box_qp_batch_impl(D, d, lower_bound, n_sweeps)

    if _pop_inductor_codegen_failed():
        # Compiled ran but inductor codegen silently degraded to eager (#27,
        # torch>=2.10 on some platforms). Disable compile so subsequent calls
        # take the quiet TorchScript JIT path (top of function); return the
        # eager result for this call.
        _USE_COMPILE = False
        return _solve_box_qp_batch_impl(D, d, lower_bound, n_sweeps)

    _USE_COMPILE = True
    return result


@torch.no_grad()
def solve_irwls_batch(
    S_batch: torch.Tensor,
    Y_batch: torch.Tensor,
    nUMI_batch: torch.Tensor,
    Q_mat: torch.Tensor,
    SQ_mat: torch.Tensor,
    x_vals: torch.Tensor,
    max_iter: int = 50,
    min_change: float = 0.001,
    step_size: float = 0.3,
    constrain: bool = True,
    bulk_mode: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve IRWLS for a batch of pixels, fully vectorized.

    All N pixels are processed simultaneously using batched matrix operations.

    Args:
        S_batch: (N, G, K) design matrices
        Y_batch: (N, G) observed counts
        nUMI_batch: (N,) total UMI per pixel
        Q_mat, SQ_mat, x_vals: shared likelihood tables
        max_iter: maximum IRWLS iterations
        min_change: convergence tolerance (L1 norm of weight change)
        step_size: damping factor
        constrain: project weights onto simplex
        bulk_mode: use Gaussian likelihood

    Returns:
        (weights, converged): (N, K) and (N,) bool arrays
    """
    N, G, K = S_batch.shape
    dtype = S_batch.dtype
    device = S_batch.device

    if not bulk_mode:
        k_val = Q_mat.shape[0] - 3
        Y_batch = torch.clamp(Y_batch, max=k_val)

    w = torch.ones(N, K, dtype=dtype, device=device) / K
    eye_K = torch.eye(K, dtype=dtype, device=device)  # shared identity

    # Per-pixel convergence tracking
    change = torch.full((N,), float("inf"), dtype=dtype, device=device)
    converged = torch.zeros(N, dtype=torch.bool, device=device)

    # Threshold per pixel: max(1e-4, nUMI * 1e-7)
    threshold = torch.clamp(nUMI_batch * 1e-7, min=1e-4)  # (N,)

    for it in range(max_iter):
        # Only process non-converged pixels
        active = ~converged
        if not active.any():
            break

        solution = torch.clamp(w, min=0.0)  # (N, K)

        # prediction = |S @ w|: (N, G, K) @ (N, K, 1) → (N, G)
        prediction = torch.abs(torch.bmm(S_batch, solution.unsqueeze(2)).squeeze(2))
        prediction = torch.clamp(prediction, min=threshold.unsqueeze(1))

        grad, hess = _get_derivatives_batch(
            S_batch, Y_batch, prediction, Q_mat, SQ_mat, x_vals, bulk_mode
        )

        hess, norm_factor = _psd_batch(hess)
        norm_factor = torch.clamp(norm_factor, min=1e-10)

        D = hess / norm_factor[:, None, None] + 1e-7 * eye_K.unsqueeze(0)
        d = -grad / norm_factor[:, None]

        delta_w = _solve_box_qp_batch(D, d, -solution)

        w_new = solution + step_size * delta_w

        if constrain:
            w_new = project_simplex_batch(w_new)

        # Per-pixel L1 change
        change = torch.sum(torch.abs(w_new - w), dim=1)  # (N,)
        newly_converged = change <= min_change

        # Freeze converged pixels: only update active ones
        w = torch.where(active.unsqueeze(1), w_new, w)
        converged = converged | newly_converged

    return w, converged


@torch.no_grad()
def solve_irwls_batch_shared(
    P: torch.Tensor,
    Y_batch: torch.Tensor,
    nUMI_batch: torch.Tensor,
    Q_mat: torch.Tensor,
    SQ_mat: torch.Tensor,
    x_vals: torch.Tensor,
    max_iter: int = 50,
    min_change: float = 0.001,
    step_size: float = 0.3,
    constrain: bool = True,
    bulk_mode: bool = False,
    _calc_q_fn=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Optimized IRWLS for shared reference profiles.

    When all pixels share the same profile matrix P, this replaces bmm with
    matmul for prediction, gradient, and Hessian. Active pixel compaction
    skips converged pixels entirely each iteration.

    Args:
        P: (G, K) shared reference profiles (NOT pre-scaled by nUMI)
        Y_batch: (N, G) observed counts
        nUMI_batch: (N,) total UMI per pixel
        Q_mat, SQ_mat, x_vals: shared likelihood tables
        max_iter: maximum IRWLS iterations
        min_change: convergence tolerance (L1 norm of weight change)
        step_size: damping factor
        constrain: project weights onto simplex
        bulk_mode: use Gaussian likelihood

    Returns:
        (weights, converged): (N, K) and (N,) bool arrays
    """
    _q_fn = _calc_q_fn if _calc_q_fn is not None else calc_q_all

    N = Y_batch.shape[0]
    if N == 0:
        K = P.shape[1]
        return (
            torch.empty(0, K, dtype=P.dtype, device=P.device),
            torch.empty(0, dtype=torch.bool, device=P.device),
        )

    G, K = P.shape
    dtype = P.dtype
    device = P.device

    if not bulk_mode:
        k_val = Q_mat.shape[0] - 3
        Y_batch = torch.clamp(Y_batch, max=k_val)

    w = torch.ones(N, K, dtype=dtype, device=device) / K
    eye_K = torch.eye(K, dtype=dtype, device=device)
    converged = torch.zeros(N, dtype=torch.bool, device=device)

    # Precompute frequently used derived matrices
    P_T = P.T.contiguous()  # (K, G) — avoids repeated non-contiguous transpose
    P_outer = (P[:, :, None] * P[:, None, :]).reshape(G, K * K)

    threshold = torch.clamp(nUMI_batch * 1e-7, min=1e-4)

    # Active pixel state — start with all pixels
    active_idx = torch.arange(N, device=device)
    Y_act = Y_batch
    nUMI_act = nUMI_batch
    thresh_act = threshold
    w_act = w.clone()

    for it in range(max_iter):
        n_act = active_idx.shape[0]
        if n_act == 0:
            break

        solution = torch.clamp(w_act, min=0.0)

        # Prediction: matmul replaces bmm
        # S @ w = (nUMI * P) @ w = nUMI * (w @ P.T)
        prediction = torch.abs(nUMI_act.unsqueeze(1) * (solution @ P_T))
        prediction = torch.clamp(prediction, min=thresh_act.unsqueeze(1))

        # Derivatives
        if bulk_mode:
            d1_vec = -2.0 * (torch.log(prediction) - torch.log(Y_act + 1e-10)) / prediction
            d2_vec = -2.0 * (1.0 - torch.log(prediction) + torch.log(Y_act + 1e-10)) / prediction**2
        else:
            _, d1_flat, d2_flat = _q_fn(
                Y_act.reshape(-1), prediction.reshape(-1), Q_mat, SQ_mat, x_vals
            )
            d1_vec = d1_flat.reshape(n_act, G)
            d2_vec = d2_flat.reshape(n_act, G)

        # Gradient: matmul replaces bmm
        # grad = -(d1 @ S) = -(d1 * nUMI) @ P
        grad = -((d1_vec * nUMI_act.unsqueeze(1)) @ P)

        # Hessian via P outer product: matmul replaces bmm
        # H[n,i,j] = nUMI[n]^2 * sum_g (-d2[n,g]) * P[g,i] * P[g,j]
        d2_w = (-d2_vec) * (nUMI_act**2).unsqueeze(1)  # (n_act, G)
        hess = (d2_w @ P_outer).reshape(n_act, K, K)
        del d2_w, d1_vec, d2_vec, prediction  # free large intermediates

        hess, norm_factor = _psd_batch(hess)
        norm_factor = torch.clamp(norm_factor, min=1e-10)

        D = hess / norm_factor[:, None, None] + 1e-7 * eye_K.unsqueeze(0)
        d = -grad / norm_factor[:, None]

        delta_w = _solve_box_qp_batch(D, d, -solution)
        w_new = solution + step_size * delta_w

        if constrain:
            w_new = project_simplex_batch(w_new)

        change = torch.sum(torch.abs(w_new - w_act), dim=1)
        newly_converged = change <= min_change
        w_act = w_new

        # Scatter results back to full arrays
        w[active_idx] = w_act
        converged[active_idx] = converged[active_idx] | newly_converged

        # Compact to still-active pixels
        still_active = ~newly_converged
        if newly_converged.all():
            break
        if newly_converged.any():
            active_idx = active_idx[still_active]
            w_act = w_act[still_active]
            Y_act = Y_act[still_active]
            nUMI_act = nUMI_act[still_active]
            thresh_act = thresh_act[still_active]

    return w, converged
