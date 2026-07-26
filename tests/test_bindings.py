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


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("ALL TESTS PASSED")
