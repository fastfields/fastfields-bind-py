"""fastfields.dlpack: nanobind DLPack bindings to the fastfields-lib C++ lib.

All functions accept any array object exposing ``__dlpack__`` (numpy, torch,
cupy, ...). Tensors marked in-place / as outputs are written through their
DLPack data pointers.
"""

from __future__ import annotations

import ctypes
import os
import sys

_this_dir = os.path.dirname(os.path.abspath(__file__))
_lib_dir = os.path.join(_this_dir, "lib")

# Shared-library suffix used by the fastfields-lib Makefile, per platform.
if sys.platform == "darwin":
    _LIBEXT = "dylib"
elif sys.platform == "win32":
    _LIBEXT = "dll"
else:
    _LIBEXT = "so"

# cpu first, then the hub library that depends on it.
_NATIVE_LIBS = ("libfastfields-cpu." + _LIBEXT, "libfastfields." + _LIBEXT)


def _preload_native_libs() -> None:
    """Make the shipped shared libraries loadable by the ``_core`` extension.

    On ELF/Mach-O we preload cpu first, then the hub library, by absolute path
    with ``RTLD_GLOBAL``: this registers each under its soname so the loader
    reuses the copies when ``_core`` is imported, regardless of the RUNPATH
    baked into libfastfields. On Windows there is no ``RTLD_GLOBAL`` and no
    rpath, so we add the lib dir to the DLL search path (``_core`` and the
    hub DLL then resolve their dependencies from there) and also load the DLLs
    eagerly so a missing dependency surfaces here rather than as an opaque
    extension-import failure.
    """
    if sys.platform == "win32":
        if os.path.isdir(_lib_dir):
            os.add_dll_directory(_lib_dir)
        for name in _NATIVE_LIBS:
            path = os.path.join(_lib_dir, name)
            if os.path.exists(path):
                ctypes.CDLL(path)
        return
    for name in _NATIVE_LIBS:
        path = os.path.join(_lib_dir, name)
        if os.path.exists(path):
            ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)


_preload_native_libs()

from . import _core  # noqa: E402  (must follow the preload above)

# Re-export every binding under its pythonic (== ff::) name.
dt_euclidean = _core.dt_euclidean
dt_l1 = _core.dt_l1
dt_spline_table = _core.dt_spline_table
dt_spline_brent = _core.dt_spline_brent
dt_spline_gaussnewton = _core.dt_spline_gaussnewton
dt_mesh = _core.dt_mesh

sym_matvec = _core.sym_matvec
sym_matvec_backward = _core.sym_matvec_backward
sym_addmatvec_ = _core.sym_addmatvec_
sym_submatvec_ = _core.sym_submatvec_
sym_solve = _core.sym_solve
sym_solve_ = _core.sym_solve_
sym_invert = _core.sym_invert
sym_invert_ = _core.sym_invert_

resample = _core.resample
restriction = _core.restriction
spline_coeff = _core.spline_coeff

# Shared enums + pure-Python argument-normalisation helpers, used by every
# wrapper (numpy/torch/cupy) so the resample/spline-coeff argument handling
# lives in one place. See ``fastfields.dlpack._helpers``.
from ._helpers import (  # noqa: E402
    Bound,
    Spline,
    anchor_scale_shift,
    as_bound,
    as_spline,
    check_ndim,
    infer_ndim,
    normalize_shape,
    resolve_out_spatial,
)

__all__ = [
    "dt_euclidean",
    "dt_l1",
    "dt_spline_table",
    "dt_spline_brent",
    "dt_spline_gaussnewton",
    "dt_mesh",
    "sym_matvec",
    "sym_matvec_backward",
    "sym_addmatvec_",
    "sym_submatvec_",
    "sym_solve",
    "sym_solve_",
    "sym_invert",
    "sym_invert_",
    "resample",
    "restriction",
    "spline_coeff",
    "Spline",
    "Bound",
    # shared argument-normalisation helpers (fastfields.dlpack._helpers)
    "as_spline",
    "as_bound",
    "normalize_shape",
    "infer_ndim",
    "check_ndim",
    "resolve_out_spatial",
    "anchor_scale_shift",
]
