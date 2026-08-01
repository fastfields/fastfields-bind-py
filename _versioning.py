"""Custom ``versioningit`` hooks that stamp the compute-backend variant.

``fastfields-dlpack`` is the only distribution in the stack that ships compiled
artifacts (``libfastfields``/``libfastfields-cpu``, and later
``libfastfields-cuda``), so it is the only one that is *backend-specific*. The
pure-Python wrappers are universal wheels with no variant axis.

Following PyTorch's convention, the backend is encoded in the wheel's
:pep:`440` **local version identifier**::

    fastfields_dlpack-0.0.0.dev1+cpu-cp311-cp311-manylinux_2_17_x86_64.whl
                                ^^^

The label is set by the ``FF_WHEEL_VARIANT`` environment variable at build time
(the release matrix sets one value per backend) and must match the wheel
index's backend folder names exactly -- ``cpu``, ``cu118``, ``cu126``,
``cu128`` -- because ``fastfields/whl``'s ``generate.py`` buckets wheels on it.
Only alphanumerics are accepted: ``-`` and ``_`` normalise away in a local
segment, so allowing them would let two spellings collide.

When ``FF_WHEEL_VARIANT`` is unset (an ordinary local build, or any of the
pure-Python packages) these hooks reproduce versioningit's stock behaviour, so
a developer build is unchanged.

Why two hooks
-------------
versioningit only calls its ``format`` step when the checkout is *not* exactly
at a tag -- see ``versioningit/core.py``::

    if description.state == "exact":
        version = base_version
    else:
        version = self.do_format(...)

A release build is by construction a clean checkout at the tag, i.e. the
``exact`` state, so a ``format`` hook alone would never fire for the builds
that actually need the label. The variant therefore has to be applied in
``tag2version``, which runs unconditionally.

That in turn means ``base_version`` may already carry a local segment, and the
stock ``format`` templates (``{base_version}+{distance}.{vcs}{rev}``) would
then emit a second ``+`` and produce an invalid version. So ``format`` is
overridden too, purely to fold both parts into the one local segment PEP 440
allows:

===========================  ===============================
state                        version
===========================  ===============================
exact                        ``0.0.0.dev1+cpu``
distance                     ``0.0.0.dev1+cpu.3.gdeadbee``
dirty / distance-dirty       ``0.0.0.dev1+cpu.3.gdeadbee.dirty``
===========================  ===============================
"""

from __future__ import annotations

import os
import re
from typing import Any

#: Environment variable naming the compute backend for this build.
VARIANT_ENV = "FF_WHEEL_VARIANT"

#: A variant label must be purely alphanumeric (see module docstring).
_VARIANT_RE = re.compile(r"\A[A-Za-z0-9]+\Z")

#: Per-state suffix appended inside the local segment. Mirrors the templates
#: that used to live in ``[tool.versioningit.format]``.
_SUFFIXES = {
    "distance": "{distance}.{vcs}{rev}",
    "dirty": "{distance}.{vcs}{rev}.dirty",
    "distance-dirty": "{distance}.{vcs}{rev}.dirty",
}


def get_variant() -> str | None:
    """Return the normalised backend label, or ``None`` if unset.

    Returns
    -------
    str or None
        The lower-cased value of ``$FF_WHEEL_VARIANT`` (e.g. ``"cpu"``), or
        ``None`` when the variable is unset or empty.

    Raises
    ------
    ValueError
        If the value is not purely alphanumeric. Failing loudly is deliberate:
        a silently-dropped label would publish a wheel that the index files
        under the wrong backend.
    """
    raw = os.environ.get(VARIANT_ENV, "").strip()
    if not raw:
        return None
    if not _VARIANT_RE.match(raw):
        raise ValueError(
            f"{VARIANT_ENV}={raw!r} is not a valid PEP 440 local version "
            "label for this project: use alphanumerics only (cpu, cu128, ...)"
        )
    return raw.lower()


def tag2version(*, tag: str, params: dict[str, Any]) -> str:
    """Turn a git tag into a base version, appending the backend variant.

    Parameters
    ----------
    tag : str
        The tag ``git describe`` resolved to (e.g. ``0.0.0.dev1``).
    params : dict
        Extra ``[tool.versioningit.tag2version]`` keys. Unused.

    Returns
    -------
    str
        ``<version>`` or ``<version>+<variant>``.
    """
    base = tag.lstrip("v")
    variant = get_variant()
    return f"{base}+{variant}" if variant else base


def format(  # noqa: A001 - the versioningit step really is called "format"
    *,
    description: Any,
    base_version: str,
    next_version: str,
    params: dict[str, Any],
) -> str:
    """Render a non-exact version, folding variant + distance into one label.

    Parameters
    ----------
    description : versioningit.VCSDescription
        The VCS state; ``.state`` selects the suffix and ``.fields`` supplies
        ``distance``/``vcs``/``rev``.
    base_version : str
        Output of :func:`tag2version`, so it may already end in ``+<variant>``.
    next_version : str
        Unused; this project does not do next-version guessing.
    params : dict
        Extra ``[tool.versioningit.format]`` keys. Unused.

    Returns
    -------
    str
        e.g. ``0.0.0.dev1+cpu.3.gdeadbee.dirty``.
    """
    try:
        suffix = _SUFFIXES[description.state]
    except KeyError:
        raise ValueError(
            f"no version format for VCS state {description.state!r}"
        ) from None
    base, _, local = base_version.partition("+")
    parts = [p for p in (local, suffix.format_map(description.fields)) if p]
    return f"{base}+{'.'.join(parts)}"
