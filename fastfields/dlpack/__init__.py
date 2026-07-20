"""fastfields.dlpack: nanobind bindings from DLPack to the fastfields-lib C++ library.

All functions accept any array object exposing ``__dlpack__`` (numpy, torch,
cupy, ...). Tensors marked in-place / as outputs are written through their
DLPack data pointers.
"""

from __future__ import annotations

import ctypes
import os
from enum import IntEnum

_this_dir = os.path.dirname(os.path.abspath(__file__))
_lib_dir = os.path.join(_this_dir, "lib")


def _preload_native_libs() -> None:
    """Preload the shipped shared libraries with RTLD_GLOBAL.

    We load ``libfastfields-cpu.so`` first, then ``libfastfields.so`` (which
    depends on it). Preloading by absolute path registers each library under
    its soname, so when the ``_core`` extension is imported the dynamic loader
    reuses the already-loaded copies and never has to resolve them via rpath.
    This makes import robust regardless of the (transitive) RUNPATH baked into
    libfastfields.so.
    """
    for name in ("libfastfields-cpu.so", "libfastfields.so"):
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


class Spline(IntEnum):
    """Spline interpolation order (passed as the ``spline`` argument)."""

    Nearest = 0
    Linear = 1
    Quadratic = 2
    Cubic = 3
    FourthOrder = 4
    FifthOrder = 5
    SixthOrder = 6
    SeventhOrder = 7


class Bound(IntEnum):
    """Boundary condition (passed as the ``bound`` argument)."""

    Zero = 0       # zero outside the FOV
    Replicate = 1  # clip coordinates
    DCT1 = 2       # symmetric w.r.t. voxel centre
    DCT2 = 3       # symmetric w.r.t. voxel edge (Neumann)
    DST1 = 4       # antisymmetric w.r.t. voxel centre
    DST2 = 5       # antisymmetric w.r.t. voxel edge (Dirichlet)
    DFT = 6        # circular / wrap around
    NoCheck = 7    # assume coordinates are inbound


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
]
