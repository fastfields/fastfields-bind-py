"""Tests for fastfields.dlpack.

Runnable via ``pytest`` or ``python tests/test_bindings.py``.
"""

import numpy as np

import fastfields.dlpack as ff


def _edt_reference(inp, voxel_spacing, cost):
    """Brute-force distance transform along the last axis.

    out[..., i] = min_j (inp[..., j] + cost(voxel_spacing * (i - j)))
    """
    out = np.full_like(inp, np.inf)
    n = inp.shape[-1]
    flat = inp.reshape(-1, n)
    ref = out.reshape(-1, n)
    for r in range(flat.shape[0]):
        for i in range(n):
            best = np.inf
            for j in range(n):
                best = min(best, flat[r, j] + cost(voxel_spacing * (i - j)))
            ref[r, i] = best
    return out


def test_dt_euclidean():
    inp = np.array(
        [
            [0, np.inf, np.inf, 0, np.inf, np.inf, np.inf],
            [np.inf, np.inf, 0, np.inf, np.inf, 0, np.inf],
        ],
        dtype=np.float32,
    )
    ref = _edt_reference(inp, 1.0, lambda d: d * d)
    ff.dt_euclidean(inp, 1.0)
    np.testing.assert_allclose(inp, ref, rtol=1e-5, atol=1e-5)


def test_dt_l1():
    inp = np.array(
        [
            [0, np.inf, np.inf, 0, np.inf, np.inf, np.inf],
            [np.inf, np.inf, 0, np.inf, np.inf, 0, np.inf],
        ],
        dtype=np.float32,
    )
    ref = _edt_reference(inp, 1.0, lambda d: abs(d))
    ff.dt_l1(inp, 1.0)
    np.testing.assert_allclose(inp, ref, rtol=1e-5, atol=1e-5)


def test_spline_coeff_order1_noop_and_shape():
    inp = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float64)
    original = inp.copy()
    ff.spline_coeff(inp, 1, 3)  # order 1 -> no-op
    assert inp.shape == original.shape
    assert inp.dtype == original.dtype
    np.testing.assert_allclose(inp, original, rtol=1e-12, atol=1e-12)

    # A real (cubic) prefilter must at least run and preserve shape/dtype.
    inp3 = original.copy()
    ff.spline_coeff(inp3, 3, 3)
    assert inp3.shape == original.shape
    assert inp3.dtype == original.dtype


def test_resample_identity():
    inp = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    out = np.zeros_like(inp)
    # factor-1, linear spline -> sampling at integer nodes is the identity
    ff.resample(out, inp, spline=1, bound=3, shift=0.0, scale=[1.0], ndim=1)
    assert out.shape == inp.shape
    np.testing.assert_allclose(out, inp, rtol=1e-6, atol=1e-6)


def _pack_symmetric(mats):
    """Dense (B,C,C) symmetric -> compact (B, C*(C+1)/2) diagonal-then-rows."""
    B, C, _ = mats.shape
    packed = np.zeros((B, C * (C + 1) // 2), dtype=mats.dtype)
    for b in range(B):
        idx = 0
        for k in range(C):  # diagonal first
            packed[b, idx] = mats[b, k, k]
            idx += 1
        for i in range(C):  # then rows (upper off-diagonal)
            for j in range(i + 1, C):
                packed[b, idx] = mats[b, i, j]
                idx += 1
    return packed


def test_sym_matvec():
    for C in (2, 3):
        B = 4
        rng = np.random.default_rng(C)
        mats = rng.standard_normal((B, C, C))
        mats = mats + np.transpose(mats, (0, 2, 1))  # symmetrise
        vec = rng.standard_normal((B, C))

        hessian = _pack_symmetric(mats)
        out = np.zeros((B, C), dtype=np.float64)
        ff.sym_matvec(out, hessian, vec)

        ref = np.einsum("bij,bj->bi", mats, vec)
        np.testing.assert_allclose(out, ref, rtol=1e-8, atol=1e-8)


def test_shared_helpers():
    """The pure-Python argument-normalisation helpers used by the wrappers."""
    # order / bound accept int, enum or friendly name
    assert ff.as_spline("cubic") == 3
    assert ff.as_spline(ff.Spline.Linear) == 1
    assert ff.as_bound("dct2") == int(ff.Bound.DCT2)
    assert ff.as_bound("wrap") == int(ff.Bound.DFT)
    for bad in ("nope", 99):
        try:
            ff.as_spline(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError from as_spline")

    # ndim inference + range check
    assert ff.infer_ndim(None, None, [4, 4]) == 2
    assert ff.infer_ndim(None, 2.0, None) == 1
    assert ff.infer_ndim(3, None, None) == 3
    try:
        ff.check_ndim(0, 2)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError from check_ndim")

    # factor / shape -> output spatial shape
    assert ff.normalize_shape(4, 2) == [4, 4]
    assert ff.resolve_out_spatial((5,), 1, 2, None) == (10,)
    assert ff.resolve_out_spatial((5, 5), 2, None, 10) == (10, 10)
    assert ff.resolve_out_spatial((7,), 1, None, None) == (7,)  # identity

    # anchor -> (per-dim scale, scalar shift), matching interpol.resize
    assert ff.anchor_scale_shift("centers", (8,), (4,), 1) == ([7 / 3], 0.0)
    assert ff.anchor_scale_shift("edges", (8,), (4,), 1) == ([2.0], 0.5)
    assert ff.anchor_scale_shift("first", (8,), (4,), 1) == ([2.0], 0.0)
    assert ff.anchor_scale_shift("last", (8,), (4,), 1) == ([2.0], 1.0)
    try:
        ff.anchor_scale_shift("nope", (8,), (4,), 1)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError from anchor_scale_shift")


def test_pull_identity():
    # A linear pull at an identity grid returns the input coefficients.
    inp = np.arange(6.0).reshape(6, 1)  # (inshape=6, C=1)
    grid = np.arange(6.0).reshape(6, 1)  # (outshape=6, D=1) identity
    out = np.zeros((6, 1))
    ff.pull(out, inp, grid, spline=1, bound=3, extrapolate=1)
    np.testing.assert_allclose(out[:, 0], inp[:, 0], rtol=1e-10, atol=1e-10)


def test_count_identity():
    grid = np.arange(6.0).reshape(6, 1)
    cnt = np.zeros((6, 1))
    ff.count(cnt, grid, spline=1, bound=3, extrapolate=1)
    # linear splat of ones at integer coords -> exactly one per voxel
    np.testing.assert_allclose(cnt[:, 0], np.ones(6), rtol=1e-10, atol=1e-10)


def test_push_is_pull_adjoint():
    # <pull(x), y> == <x, push(y)> for the same grid (push is pull's adjoint).
    grid = np.arange(6.0).reshape(6, 1)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((6, 1))
    y = rng.standard_normal((6, 1))
    px = np.zeros((6, 1))
    ff.pull(px, x, grid, spline=2, bound=3, extrapolate=1)
    py = np.zeros((6, 1))  # push accumulates -> pre-zeroed
    ff.push(py, y, grid, spline=2, bound=3, extrapolate=1)
    np.testing.assert_allclose(
        float((px * y).sum()), float((x * py).sum()), rtol=1e-8, atol=1e-8
    )


def test_field_matvec_absolute_is_scaling():
    # absolute-only regulariser is a per-channel scaling out[...,c] = a[c]*inp.
    rng = np.random.default_rng(1)
    inp = rng.standard_normal((8, 2))  # (spatial=8, C=2)
    out = np.zeros_like(inp)
    ff.field_matvec(out, inp, absolute=[2.0, 3.0], ndim=1, bound=3)
    np.testing.assert_allclose(out[:, 0], 2.0 * inp[:, 0], rtol=1e-10)
    np.testing.assert_allclose(out[:, 1], 3.0 * inp[:, 1], rtol=1e-10)


def test_field_diag_absolute():
    # diagonal of an absolute-only operator is the constant absolute weight.
    out = np.zeros((8, 2))
    ff.field_diag(out, absolute=[2.0, 3.0], ndim=1, bound=3)
    np.testing.assert_allclose(out[:, 0], 2.0, rtol=1e-10)
    np.testing.assert_allclose(out[:, 1], 3.0, rtol=1e-10)


def test_field_kernel_is_matvec_impulse_response():
    # The per-channel field stencil equals field_matvec applied to a unit
    # impulse in the interior (field channels are independent).
    def check(kd, order, absolute, membrane, bending):
        C = 2
        ap = absolute
        mp = membrane if order >= 2 else None
        bp = bending if order >= 3 else None
        K = np.zeros((kd, kd, C), np.float64)
        ff.field_kernel(
            K, absolute=ap, membrane=mp, bending=bp, bound=3, ndim=2
        )
        N, cc, half = 2 * kd + 1, kd, kd // 2
        for c0 in range(C):
            x = np.zeros((N, N, C))
            x[cc, cc, c0] = 1.0
            o = np.zeros((N, N, C))
            ff.field_matvec(
                o, x, absolute=ap, membrane=mp, bending=bp, bound=3, ndim=2
            )
            for a in range(kd):
                for b in range(kd):
                    for c in range(C):
                        got = o[cc + a - half, cc + b - half, c]
                        kern = K[a, b, c] if c == c0 else 0.0
                        np.testing.assert_allclose(got, kern, atol=1e-10)

    check(1, 1, [2.5, 1.5], None, None)
    check(3, 2, [0.3, 0.4], [1.0, 0.7], None)
    check(5, 3, [0.3, 0.4], [0.5, 0.6], [1.0, 0.8])


def test_field_relax_reduces_residual():
    # field_relax refines `sol` in place towards (H + L) x = g. Check the
    # residual against a field_matvec/sym_matvec-derived reference: it must
    # shrink monotonically with the sweep count and converge to ~0.
    rng = np.random.default_rng(3)
    N, C = 12, 2
    absolute, membrane = [0.4, 0.6], [1.0, 0.8]
    # SPD compact-symmetric Hessian: diagonal-dominant (diag, then off-diag).
    hes = np.stack(
        [
            2.0 + rng.random((N, N)),
            2.0 + rng.random((N, N)),
            0.5 * (rng.random((N, N)) - 0.5),
        ],
        axis=-1,
    )
    grd = rng.standard_normal((N, N, C))

    def residual(x):
        r = np.zeros_like(x)
        ff.sym_matvec(r, hes, x)  # H @ x
        lx = np.zeros_like(x)
        ff.field_matvec(
            lx, x, absolute=absolute, membrane=membrane, bound=3, ndim=2
        )
        return np.abs(r + lx - grd).max()

    prev = residual(np.zeros((N, N, C)))
    sol = np.zeros((N, N, C))
    for _ in range(6):
        out = ff.field_relax(
            sol,
            hes,
            grd,
            absolute=absolute,
            membrane=membrane,
            bound=3,
            ndim=2,
            nb_iter=8,
        )
        assert out is None  # in-place, mutates `sol`
        cur = residual(sol)
        assert cur < prev
        prev = cur
    assert prev < 1e-8


def test_flow_matvec_absolute_is_scaling():
    # absolute-only flow regulariser scales the whole field by `absolute`.
    rng = np.random.default_rng(2)
    inp = rng.standard_normal((8, 1))  # (spatial=8, D=1)
    out = np.zeros_like(inp)
    ff.flow_matvec(out, inp, absolute=2.5, ndim=1, bound=3)
    np.testing.assert_allclose(out, 2.5 * inp, rtol=1e-10)


def test_flow_kernel_is_matvec_impulse_response():
    # The materialised stencil equals flow_matvec applied to a unit impulse,
    # windowed around it in the interior (translation-invariant there).
    def check(kd, a, m, b, s, d):
        is_matrix = s != 0.0 or d != 0.0
        C = 2
        kshape = (kd, kd, C, C) if is_matrix else (kd, kd, C)
        K = np.zeros(kshape, np.float64)
        ff.flow_kernel(
            K,
            absolute=a,
            membrane=m,
            bending=b,
            shears=s,
            div=d,
            bound=3,
            ndim=2,
        )
        N, cc, half = 2 * kd + 1, kd, kd // 2
        for j0 in range(C):
            x = np.zeros((N, N, C))
            x[cc, cc, j0] = 1.0
            o = np.zeros((N, N, C))
            ff.flow_matvec(
                o,
                x,
                absolute=a,
                membrane=m,
                bending=b,
                shears=s,
                div=d,
                bound=3,
                ndim=2,
            )
            for aa in range(kd):
                for bb in range(kd):
                    for i in range(C):
                        got = o[cc + aa - half, cc + bb - half, i]
                        kern = (
                            K[aa, bb, i, j0]
                            if is_matrix
                            else (K[aa, bb, i] if i == j0 else 0.0)
                        )
                        np.testing.assert_allclose(got, kern, atol=1e-10)

    check(1, 2.5, 0, 0, 0, 0)  # absolute -> (1,1,2)
    check(3, 0, 1.0, 0, 0, 0)  # membrane -> (3,3,2)
    check(5, 0, 0, 1.0, 0, 0)  # bending  -> (5,5,2)
    check(3, 0, 0, 0, 1.3, 0.7)  # lame     -> (3,3,2,2)
    check(5, 0.3, 0.5, 0.4, 1.3, 0.7)  # all    -> (5,5,2,2)


# ---------------------------------------------------------------------------
# RLS/JRLS-weighted flow regulariser
# ---------------------------------------------------------------------------
#
# `wgt` is always joint here (trailing size-1 axis): the flow components are
# the components of one displacement vector, so one weight is shared across
# them. These mirror the oracles in fastfields-cpu-lib's tests/test_reg_flow.
# What they buy at *this* layer is proof that the bindings forward the right
# arguments in the right order to the right C symbol.

_FLOW_RLS_PENALTIES = [
    dict(absolute=1.75),  # absolute only
    dict(absolute=0.3, membrane=1.0),  # membrane_jrls path
    dict(shears=1.3, div=0.7),  # lame_jrls path
    dict(absolute=0.5, membrane=0.9, shears=1.3, div=0.7),
]


def _flow_rls_setup(seed, H=5, W=6, D=2):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((H, W, D))
    y = rng.standard_normal((H, W, D))
    w = 0.25 + rng.random((H, W, 1))  # strictly positive weight map
    return x, y, w


def test_flow_matvec_rls_is_self_adjoint():
    # L(w) is symmetric for a fixed weight map: <L x, y> == <x, L y>.
    for kw in _FLOW_RLS_PENALTIES:
        x, y, w = _flow_rls_setup(11)
        lx, ly = np.zeros_like(x), np.zeros_like(y)
        ff.flow_matvec_rls(lx, x, w, bound=3, ndim=2, **kw)
        ff.flow_matvec_rls(ly, y, w, bound=3, ndim=2, **kw)
        np.testing.assert_allclose(
            float((lx * y).sum()), float((x * ly).sum()), rtol=1e-10
        )
        # and it is genuinely weighted: a non-constant w changes the operator
        flat = np.zeros_like(x)
        ff.flow_matvec_rls(flat, x, np.ones_like(w), bound=3, ndim=2, **kw)
        assert not np.allclose(lx, flat)


def test_flow_diag_rls_matches_the_operator_diagonal():
    # diag[i] == e_i^T L(w) e_i, checked on interior voxels only (the
    # boundary condition makes edge voxels stencil-dependent).
    for kw in _FLOW_RLS_PENALTIES:
        x, _, w = _flow_rls_setup(12, H=6, W=7)
        H, W, D = x.shape
        diag = np.zeros((H, W, D))
        ff.flow_diag_rls(diag, w, bound=3, ndim=2, **kw)
        for i in range(1, H - 1):
            for j in range(1, W - 1):
                for c in range(D):
                    e = np.zeros((H, W, D))
                    e[i, j, c] = 1.0
                    o = np.zeros((H, W, D))
                    ff.flow_matvec_rls(o, e, w, bound=3, ndim=2, **kw)
                    np.testing.assert_allclose(
                        o[i, j, c], diag[i, j, c], atol=1e-10
                    )


def test_flow_relax_rls_solves_the_weighted_system():
    # Relaxation drives (H + L(w)) x -> g, with L(w) the *weighted* operator:
    # the same oracle as fastfields-cpu-lib's run_2d_relax_rls. Checked as a
    # residual after a fixed sweep budget rather than a strictly monotone
    # sequence, which would be brittle once the residual nears eps.
    rng = np.random.default_rng(13)
    N, D = 8, 2
    kw = dict(absolute=0.3, membrane=0.7, shears=1.0, div=0.5)
    hes = np.stack(
        [
            6.0 + rng.random((N, N)),
            6.0 + rng.random((N, N)),
            0.5 * (rng.random((N, N)) - 0.5),
        ],
        axis=-1,
    )
    grd = rng.standard_normal((N, N, D))
    w = 0.25 + rng.random((N, N, 1))

    def residual(x):
        r = np.zeros_like(x)
        ff.sym_matvec(r, hes, x)
        lx = np.zeros_like(x)
        ff.flow_matvec_rls(lx, x, w, bound=3, ndim=2, **kw)
        return np.linalg.norm(r + lx - grd) / np.linalg.norm(grd)

    sol = np.zeros((N, N, D))
    out = ff.flow_relax_rls(
        sol, hes, grd, w, bound=3, ndim=2, nb_iter=250, **kw
    )
    assert out is None  # in-place, mutates `sol`
    assert np.any(sol != 0.0)
    assert residual(sol) < 3e-3


def test_flow_rls_rejects_bending():
    # No jrls bending kernel exists at the impl layer (as in jitfields), so
    # the library rejects it rather than silently ignoring the penalty.
    x, _, w = _flow_rls_setup(14)
    out = np.zeros_like(x)
    for call in (
        lambda: ff.flow_matvec_rls(out, x, w, bending=1.0, bound=3, ndim=2),
        lambda: ff.flow_diag_rls(out, w, bending=1.0, bound=3, ndim=2),
    ):
        try:
            call()
        except (ValueError, RuntimeError):
            pass
        else:
            raise AssertionError("bending must be rejected with weighting")


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def test_all_lists_every_re_exported_binding():
    """``__all__`` must not drift from the names the package actually binds.

    It listed only 28 of 48 bindings before fastfields-lib#69 -- the whole
    pushpull and regulariser families were missing.
    """
    bound = {n for n in dir(ff._core) if not n.startswith("_")}
    missing = sorted(bound - set(ff.__all__))
    assert not missing, f"bindings absent from __all__: {missing}"
    stale = sorted(n for n in ff.__all__ if not hasattr(ff, n))
    assert not stale, f"__all__ names that do not exist: {stale}"
    assert len(ff.__all__) == len(set(ff.__all__)), "__all__ has duplicates"


# ---------------------------------------------------------------------------
# `stream` must be a 64-bit handle on every binding
# ---------------------------------------------------------------------------
#
# A real CUDA stream handle is a pointer: `torch.cuda.Stream.cuda_stream` and
# `cupy.cuda.Stream.ptr` are full 64-bit values. nanobind's `int` caster
# *range-checks* rather than truncating, so a binding declared `int stream`
# raises TypeError on essentially every genuine GPU stream while looking fine
# on CPU (where `stream=0` is always passed). `field_relax` was exactly that
# straggler -- see fastfields-lib#69.
#
# The value is never dereferenced on the CPU path, so these calls simply have
# to survive argument conversion.
BIG_STREAM = 1 << 40

_SPATIAL = (5, 5)
_C = 2
_NDIM = 2
_PACKED = _C * (_C + 1) // 2  # compact-symmetric Hessian, C=2


def _f(*shape):
    return np.zeros(shape, dtype=np.float64)


def _field_kw():
    return dict(absolute=[1.0] * _C, bound=3, ndim=_NDIM)


def _flow_kw():
    return dict(absolute=1.0, bound=3, ndim=_NDIM)


# One minimal, valid call per stream-taking regulariser binding. Keep this in
# lock-step with the `m.def`s in src/ext.cpp.
_STREAM_CASES = {
    "field_matvec": lambda s: ff.field_matvec(
        _f(*_SPATIAL, _C), _f(*_SPATIAL, _C), stream=s, **_field_kw()
    ),
    "field_diag": lambda s: ff.field_diag(
        _f(*_SPATIAL, _C), stream=s, **_field_kw()
    ),
    "field_relax": lambda s: ff.field_relax(
        _f(*_SPATIAL, _C),
        _f(*_SPATIAL, _PACKED) + 1.0,
        _f(*_SPATIAL, _C),
        stream=s,
        **_field_kw(),
    ),
    "field_kernel": lambda s: ff.field_kernel(
        _f(1, 1, _C), stream=s, **_field_kw()
    ),
    "field_matvec_rls": lambda s: ff.field_matvec_rls(
        _f(*_SPATIAL, _C),
        _f(*_SPATIAL, _C),
        _f(*_SPATIAL, 1) + 1.0,
        stream=s,
        **_field_kw(),
    ),
    "field_diag_rls": lambda s: ff.field_diag_rls(
        _f(*_SPATIAL, _C), _f(*_SPATIAL, 1) + 1.0, stream=s, **_field_kw()
    ),
    "field_relax_rls": lambda s: ff.field_relax_rls(
        _f(*_SPATIAL, _C),
        _f(*_SPATIAL, _PACKED) + 1.0,
        _f(*_SPATIAL, _C),
        _f(*_SPATIAL, 1) + 1.0,
        stream=s,
        **_field_kw(),
    ),
    "field_addmatvec_": lambda s: ff.field_addmatvec_(
        _f(*_SPATIAL, _C), _f(*_SPATIAL, _C), stream=s, **_field_kw()
    ),
    "field_submatvec_": lambda s: ff.field_submatvec_(
        _f(*_SPATIAL, _C), _f(*_SPATIAL, _C), stream=s, **_field_kw()
    ),
    "field_adddiag_": lambda s: ff.field_adddiag_(
        _f(*_SPATIAL, _C), stream=s, **_field_kw()
    ),
    "field_subdiag_": lambda s: ff.field_subdiag_(
        _f(*_SPATIAL, _C), stream=s, **_field_kw()
    ),
    "field_addkernel_": lambda s: ff.field_addkernel_(
        _f(1, 1, _C), stream=s, **_field_kw()
    ),
    "field_subkernel_": lambda s: ff.field_subkernel_(
        _f(1, 1, _C), stream=s, **_field_kw()
    ),
    "flow_matvec": lambda s: ff.flow_matvec(
        _f(*_SPATIAL, _NDIM), _f(*_SPATIAL, _NDIM), stream=s, **_flow_kw()
    ),
    "flow_diag": lambda s: ff.flow_diag(
        _f(*_SPATIAL, _NDIM), stream=s, **_flow_kw()
    ),
    "flow_relax": lambda s: ff.flow_relax(
        _f(*_SPATIAL, _NDIM),
        _f(*_SPATIAL, _PACKED) + 1.0,
        _f(*_SPATIAL, _NDIM),
        stream=s,
        **_flow_kw(),
    ),
    "flow_kernel": lambda s: ff.flow_kernel(
        _f(1, 1, _NDIM), stream=s, **_flow_kw()
    ),
    "flow_matvec_rls": lambda s: ff.flow_matvec_rls(
        _f(*_SPATIAL, _NDIM),
        _f(*_SPATIAL, _NDIM),
        _f(*_SPATIAL, 1) + 1.0,
        stream=s,
        **_flow_kw(),
    ),
    "flow_diag_rls": lambda s: ff.flow_diag_rls(
        _f(*_SPATIAL, _NDIM), _f(*_SPATIAL, 1) + 1.0, stream=s, **_flow_kw()
    ),
    "flow_relax_rls": lambda s: ff.flow_relax_rls(
        _f(*_SPATIAL, _NDIM),
        _f(*_SPATIAL, _PACKED) + 1.0,
        _f(*_SPATIAL, _NDIM),
        _f(*_SPATIAL, 1) + 1.0,
        stream=s,
        **_flow_kw(),
    ),
    "flow_addmatvec_": lambda s: ff.flow_addmatvec_(
        _f(*_SPATIAL, _NDIM), _f(*_SPATIAL, _NDIM), stream=s, **_flow_kw()
    ),
    "flow_submatvec_": lambda s: ff.flow_submatvec_(
        _f(*_SPATIAL, _NDIM), _f(*_SPATIAL, _NDIM), stream=s, **_flow_kw()
    ),
    "flow_adddiag_": lambda s: ff.flow_adddiag_(
        _f(*_SPATIAL, _NDIM), stream=s, **_flow_kw()
    ),
    "flow_subdiag_": lambda s: ff.flow_subdiag_(
        _f(*_SPATIAL, _NDIM), stream=s, **_flow_kw()
    ),
    "flow_addkernel_": lambda s: ff.flow_addkernel_(
        _f(1, 1, _NDIM), stream=s, **_flow_kw()
    ),
    "flow_subkernel_": lambda s: ff.flow_subkernel_(
        _f(1, 1, _NDIM), stream=s, **_flow_kw()
    ),
}


def test_every_reg_binding_is_covered_by_a_stream_case():
    """A new field_*/flow_* binding must be added to _STREAM_CASES."""
    bound = {
        n
        for n in dir(ff._core)
        if n.startswith(("field_", "flow_")) and not n.startswith("_")
    }
    assert bound == set(_STREAM_CASES)


def test_reg_bindings_accept_a_64bit_stream_handle():
    """Regression for fastfields-lib#69: `int stream` rejected real handles."""
    # Sanity: the same calls work with the CPU default, so a failure below is
    # about the stream argument and nothing else.
    for _, call in sorted(_STREAM_CASES.items()):
        call(0)
    for name, call in sorted(_STREAM_CASES.items()):
        try:
            call(BIG_STREAM)
        except TypeError as exc:  # pragma: no cover - the bug being guarded
            raise AssertionError(
                f"{name} narrows `stream` to a 32-bit int: a real CUDA "
                f"stream handle ({BIG_STREAM}) is rejected -- {exc}"
            ) from None


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("ALL TESTS PASSED")
