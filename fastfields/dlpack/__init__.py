"""fastfields.dlpack: nanobind DLPack bindings to the fastfields-lib C++ lib.

All functions accept any array object exposing ``__dlpack__`` (numpy, torch,
cupy, ...). Tensors marked in-place / as outputs are written through their
DLPack data pointers.
"""

from __future__ import annotations

import ctypes
import os
import re as _re
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

# -- version + compute backend ---------------------------------------------
# ``_version.py`` is generated at build time by versioningit; a bare source
# tree that was never built has no such file.
try:
    from ._version import __version__
except ImportError:  # pragma: no cover - source tree without a build
    __version__ = "0+unknown"


_BACKEND_RE = _re.compile(r"\A(cpu|cu[0-9]+)\Z")


def _parse_backend(version: str) -> str | None:
    """Extract the compute-backend label from a PEP 440 version string.

    This distribution is the only backend-specific one in the stack (it ships
    the compiled ``libfastfields*``), and it records which build it is in the
    version's *local* segment, PyTorch-style: ``0.0.0.dev1+cpu``. A build made
    off-tag carries the git distance in the same segment
    (``0.0.0.dev1+cpu.3.gdeadbee``), so only the leading component is the
    backend.

    Parameters
    ----------
    version : str
        A PEP 440 version string.

    Returns
    -------
    str or None
        ``"cpu"``, ``"cu128"``, ... or ``None`` for a build with no backend
        label (e.g. an unversioned source checkout).
    """
    try:
        from packaging.version import InvalidVersion, parse
    except ImportError:  # pragma: no cover - packaging always installed
        return None
    try:
        local = parse(version).local
    except InvalidVersion:  # pragma: no cover
        return None
    if not local:
        return None
    head = local.split(".")[0]
    return head if _BACKEND_RE.match(head) else None


#: Compute backend this build of the native libraries was compiled for --
#: ``"cpu"``, ``"cu128"``, ... or ``None`` if unknown. Mirrors the role of
#: ``torch.version.cuda``: a friendly derived attribute so callers never have
#: to parse ``__version__`` themselves.
backend = _parse_backend(__version__)

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

pull = _core.pull
push = _core.push
count = _core.count
grad = _core.grad
pull_backward = _core.pull_backward
push_backward = _core.push_backward
count_backward = _core.count_backward
grad_backward = _core.grad_backward

field_matvec = _core.field_matvec
field_diag = _core.field_diag
field_kernel = _core.field_kernel
flow_matvec = _core.flow_matvec
flow_diag = _core.flow_diag
flow_relax = _core.flow_relax
flow_kernel = _core.flow_kernel

# In-place-only accumulate primitives (jitfields op '+' / '-'). These
# read-modify-write the caller's `out`; the trailing underscore marks that, as
# for sym_addmatvec_ / sym_submatvec_. There is no out-of-place counterpart at
# this level -- the numpy/torch/cupy wrappers get it by cloning first.
field_addmatvec_ = _core.field_addmatvec_
field_submatvec_ = _core.field_submatvec_
field_adddiag_ = _core.field_adddiag_
field_subdiag_ = _core.field_subdiag_
field_addkernel_ = _core.field_addkernel_
field_subkernel_ = _core.field_subkernel_
flow_addmatvec_ = _core.flow_addmatvec_
flow_submatvec_ = _core.flow_submatvec_
flow_adddiag_ = _core.flow_adddiag_
flow_subdiag_ = _core.flow_subdiag_
flow_addkernel_ = _core.flow_addkernel_
flow_subkernel_ = _core.flow_subkernel_

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
    "__version__",
    "backend",
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
