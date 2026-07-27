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
        is_matrix = (s != 0.0 or d != 0.0)
        C = 2
        kshape = (kd, kd, C, C) if is_matrix else (kd, kd, C)
        K = np.zeros(kshape, np.float64)
        ff.flow_kernel(K, absolute=a, membrane=m, bending=b, shears=s, div=d,
                       bound=3, ndim=2)
        N, cc, half = 2 * kd + 1, kd, kd // 2
        for j0 in range(C):
            x = np.zeros((N, N, C))
            x[cc, cc, j0] = 1.0
            o = np.zeros((N, N, C))
            ff.flow_matvec(o, x, absolute=a, membrane=m, bending=b, shears=s,
                           div=d, bound=3, ndim=2)
            for aa in range(kd):
                for bb in range(kd):
                    for i in range(C):
                        got = o[cc + aa - half, cc + bb - half, i]
                        kern = (K[aa, bb, i, j0] if is_matrix
                                else (K[aa, bb, i] if i == j0 else 0.0))
                        np.testing.assert_allclose(got, kern, atol=1e-10)

    check(1, 2.5, 0, 0, 0, 0)       # absolute -> (1,1,2)
    check(3, 0, 1.0, 0, 0, 0)       # membrane -> (3,3,2)
    check(5, 0, 0, 1.0, 0, 0)       # bending  -> (5,5,2)
    check(3, 0, 0, 0, 1.3, 0.7)     # lame     -> (3,3,2,2)
    check(5, 0.3, 0.5, 0.4, 1.3, 0.7)  # all    -> (5,5,2,2)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("ALL TESTS PASSED")
