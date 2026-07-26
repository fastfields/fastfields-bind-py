"""Pure-Python argument-normalisation helpers shared by the wrappers.

The ``fastfields.numpy`` / ``fastfields.torch`` / ``fastfields.cupy`` wrappers
all need the same backend-agnostic logic to normalise ``order``/``bound``
arguments and to resolve ``factor``/``shape``/``anchor`` into the per-dim
``scale`` + scalar ``shift`` that the resize binding consumes. These helpers
live here -- in the common ``fastfields.dlpack`` dependency -- so each wrapper
imports them instead of carrying its own copy.

Everything here is pure Python (only the :class:`Spline`/:class:`Bound` enums,
no numpy/torch/cupy), so importing this module never pulls in an array backend.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Optional, Sequence


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

    Zero = 0  # zero outside the FOV
    Replicate = 1  # clip coordinates
    DCT1 = 2  # symmetric w.r.t. voxel centre
    DCT2 = 3  # symmetric w.r.t. voxel edge (Neumann)
    DST1 = 4  # antisymmetric w.r.t. voxel centre
    DST2 = 5  # antisymmetric w.r.t. voxel edge (Dirichlet)
    DFT = 6  # circular / wrap around
    NoCheck = 7  # assume coordinates are inbound


# --------------------------------------------------------------------------- #
# order / bound normalisation                                                 #
# --------------------------------------------------------------------------- #

_SPLINE_ALIASES = {
    "nearest": Spline.Nearest,
    "constant": Spline.Nearest,
    "linear": Spline.Linear,
    "quadratic": Spline.Quadratic,
    "cubic": Spline.Cubic,
    "fourth": Spline.FourthOrder,
    "fifth": Spline.FifthOrder,
    "sixth": Spline.SixthOrder,
    "seventh": Spline.SeventhOrder,
}

_BOUND_ALIASES = {
    "zero": Bound.Zero,
    "zeros": Bound.Zero,
    "replicate": Bound.Replicate,
    "nearest": Bound.Replicate,
    "dct1": Bound.DCT1,
    "dct2": Bound.DCT2,
    "neumann": Bound.DCT2,
    "reflect": Bound.DCT2,
    "dst1": Bound.DST1,
    "dst2": Bound.DST2,
    "dirichlet": Bound.DST2,
    "dft": Bound.DFT,
    "wrap": Bound.DFT,
    "circular": Bound.DFT,
    "nocheck": Bound.NoCheck,
}


def as_spline(value: int | str | Spline) -> int:
    """Normalise a spline-order argument to an ``int`` in ``0..7``.

    Accepts an integer, a :class:`Spline` enum, or a friendly string alias
    (e.g. ``"cubic"``). Raises ``ValueError`` for an unknown alias or an
    out-of-range integer.
    """
    if isinstance(value, str):
        key = value.strip().lower()
        if key not in _SPLINE_ALIASES:
            raise ValueError(
                f"unknown spline order {value!r}; "
                f"expected an int 0..7 or one of {sorted(_SPLINE_ALIASES)}"
            )
        return int(_SPLINE_ALIASES[key])
    ivalue = int(value)
    if not 0 <= ivalue <= 7:
        raise ValueError(f"spline order must be in 0..7, got {ivalue}")
    return ivalue


def as_bound(value: int | str | Bound) -> int:
    """Normalise a boundary-condition argument to an ``int`` in ``0..7``.

    Accepts an integer, a :class:`Bound` enum, or a friendly string alias
    (e.g. ``"dct2"``, ``"wrap"``). Raises ``ValueError`` for an unknown alias
    or an out-of-range integer.
    """
    if isinstance(value, str):
        key = value.strip().lower()
        if key not in _BOUND_ALIASES:
            raise ValueError(
                f"unknown boundary condition {value!r}; "
                f"expected an int 0..7 or one of {sorted(_BOUND_ALIASES)}"
            )
        return int(_BOUND_ALIASES[key])
    ivalue = int(value)
    if not 0 <= ivalue <= 7:
        raise ValueError(f"boundary condition must be in 0..7, got {ivalue}")
    return ivalue


# --------------------------------------------------------------------------- #
# resize shape / ndim resolution                                              #
# --------------------------------------------------------------------------- #


def normalize_shape(shape: int | Sequence[int], ndim: int) -> list[int]:
    """Normalise a shape argument to a list of ``int`` of length ``ndim``.

    A scalar ``int`` is broadcast to ``ndim`` entries. Raises ``ValueError`` if
    an explicit sequence does not have length ``ndim``.
    """
    if isinstance(shape, int):
        shape = [shape] * ndim
    shape = list(shape)
    if len(shape) != ndim:
        raise ValueError(f"Expected shape of length ndim={ndim}, got {shape}.")
    return [int(s) for s in shape]


def infer_ndim(
    ndim: Optional[int],
    factor: float | Sequence[float] | None,
    shape: int | Sequence[int] | None,
) -> int:
    """Infer the number of trailing spatial dimensions to resize.

    An explicit ``ndim`` wins; otherwise a sequence ``shape`` or ``factor``
    implies its length; failing that, ``1``.
    """
    if ndim is not None:
        return int(ndim)
    if shape is not None and not isinstance(shape, int):
        return len(list(shape))
    if factor is not None and not isinstance(factor, (int, float)):
        return len(list(factor))
    return 1


def check_ndim(ndim: int, arr_ndim: int) -> None:
    """Validate that ``ndim`` is in ``1..arr_ndim`` (else ``ValueError``)."""
    if ndim < 1 or ndim > arr_ndim:
        raise ValueError(f"ndim must be in 1..{arr_ndim}, got {ndim}")


def resolve_out_spatial(
    spatial_in: Sequence[int],
    ndim: int,
    factor: float | Sequence[float] | None,
    shape: int | Sequence[int] | None,
) -> tuple[int, ...]:
    """Resolve the output spatial shape from ``factor`` or ``shape``.

    ``factor`` and ``shape`` are mutually exclusive; with neither, the output
    keeps the input spatial shape (identity). A scalar ``factor`` is broadcast
    to ``ndim`` entries; each output size is ``max(1, round(in * factor))``.
    """
    if shape is not None:
        return tuple(normalize_shape(shape, ndim))
    if factor is not None:
        if isinstance(factor, (int, float)):
            factors = [float(factor)] * ndim
        else:
            factors = [float(f) for f in factor]
        if len(factors) != ndim:
            raise ValueError(
                f"Expected factor of length ndim={ndim}, got {factors}."
            )
        return tuple(
            max(1, int(round(n * f))) for n, f in zip(spatial_in, factors)
        )
    return tuple(int(n) for n in spatial_in)  # identity


# --------------------------------------------------------------------------- #
# torch-interpol anchor conventions                                           #
# --------------------------------------------------------------------------- #

# Each anchor is identified by its first (lower-cased) letter, so both the full
# name ("centers") and the abbreviation ("c") are accepted, mirroring
# ``interpol.resize``.
_ANCHORS = ("c", "e", "f", "l")
_ANCHOR_SHIFT = {"e": 0.5, "f": 0.0, "l": 1.0}


def anchor_scale_shift(
    anchor: str,
    inshape: Sequence[int],
    outshape: Sequence[int],
    ndim: int,
) -> tuple[list[float], float]:
    """Map a torch-interpol ``anchor`` to a per-dim scale and scalar shift.

    The fastfields resize kernel samples input coordinate
    ``scale[d] * loc + shift * (scale[d] - 1)`` for output index ``loc``. The
    four anchors of ``interpol.resize`` map onto ``(scale, shift)`` as:

    ==========  =================  =======
    anchor      scale[d]           shift
    ==========  =================  =======
    ``centers`` ``(in-1)/(out-1)`` ``0.0``
    ``edges``   ``in/out``         ``0.5``
    ``first``   ``in/out``         ``0.0``
    ``last``    ``in/out``         ``1.0``
    ==========  =================  =======

    Parameters
    ----------
    anchor : str
        Anchor name or abbreviation (``centers``/``edges``/``first``/``last``
        or ``c``/``e``/``f``/``l``); matched case-insensitively on the first
        letter.
    inshape, outshape : sequence of int
        Input and output spatial sizes (length ``ndim``).
    ndim : int
        Number of spatial dimensions.

    Returns
    -------
    scale : list of float
        Per-dim input-index step per output-index step.
    shift : float
        Scalar sampling shift shared across dimensions.

    Raises
    ------
    ValueError
        If ``anchor`` is empty or its first letter is not one of ``c/e/f/l``.
    """
    key = str(anchor)[:1].lower()
    if key not in _ANCHORS:
        raise ValueError(
            f"anchor must be one of centers/edges/first/last, got {anchor!r}"
        )
    if key == "c":
        scale = [
            ((inshape[d] - 1) / (outshape[d] - 1))
            if (inshape[d] > 1 and outshape[d] > 1)
            else 1.0
            for d in range(ndim)
        ]
        return scale, 0.0
    scale = [float(inshape[d]) / float(outshape[d]) for d in range(ndim)]
    return scale, _ANCHOR_SHIFT[key]
