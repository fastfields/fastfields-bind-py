"""Tests for the compute-backend local version label.

The backend a wheel was built for is carried in its PEP 440 *local* version
identifier (``0.0.0.dev1+cpu``), which is also what the wheel index buckets on
-- so a silent regression here misfiles published wheels. These tests pin both
halves: the build-time ``versioningit`` hooks in ``_versioning.py``, and the
runtime ``fastfields.dlpack.backend`` attribute derived from them.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest
from packaging.version import Version

# Load the hook module by path rather than putting the repo root on sys.path:
# doing the latter would shadow the *installed* fastfields.dlpack with the
# unbuilt source tree (it is a PEP 420 namespace package, so the source dir
# wins and `_core` is then missing).
_HOOKS = pathlib.Path(__file__).resolve().parent.parent / "_versioning.py"
_spec = importlib.util.spec_from_file_location("_ff_versioning", _HOOKS)
_versioning = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_versioning)


class FakeDescription:
    """Stand-in for ``versioningit.VCSDescription``."""

    def __init__(self, state: str) -> None:
        self.state = state
        self.fields = {"distance": 3, "vcs": "g", "rev": "deadbee"}


@pytest.fixture
def variant(monkeypatch):
    """Return a setter for ``$FF_WHEEL_VARIANT``."""

    def _set(value):
        if value is None:
            monkeypatch.delenv(_versioning.VARIANT_ENV, raising=False)
        else:
            monkeypatch.setenv(_versioning.VARIANT_ENV, value)

    return _set


@pytest.mark.parametrize(
    "tag,label,expected",
    [
        ("0.0.0.dev1", "cpu", "0.0.0.dev1+cpu"),
        ("v0.0.0.dev1", "cpu", "0.0.0.dev1+cpu"),
        ("0.0.0.dev1", "cu128", "0.0.0.dev1+cu128"),
        ("0.0.0.dev1", "CPU", "0.0.0.dev1+cpu"),
        ("0.0.0.dev1", None, "0.0.0.dev1"),
        ("0.0.0.dev1", "", "0.0.0.dev1"),
    ],
)
def test_tag2version(variant, tag, label, expected):
    variant(label)
    assert _versioning.tag2version(tag=tag, params={}) == expected


@pytest.mark.parametrize("bad", ["cu-128", "cu_128", "cu 128", "cu.128"])
def test_tag2version_rejects_non_alphanumeric(variant, bad):
    # '-' and '_' normalise away in a local segment, so two spellings would
    # collide; fail loudly rather than misfile the wheel.
    variant(bad)
    with pytest.raises(ValueError):
        _versioning.tag2version(tag="0.0.0.dev1", params={})


@pytest.mark.parametrize(
    "state,expected",
    [
        ("distance", "0.0.0.dev1+cpu.3.gdeadbee"),
        ("dirty", "0.0.0.dev1+cpu.3.gdeadbee.dirty"),
        ("distance-dirty", "0.0.0.dev1+cpu.3.gdeadbee.dirty"),
    ],
)
def test_format_folds_variant_and_distance(variant, state, expected):
    # PEP 440 allows exactly one '+', so the variant and the distance suffix
    # have to end up in the *same* local segment.
    variant("cpu")
    base = _versioning.tag2version(tag="0.0.0.dev1", params={})
    got = _versioning.format(
        description=FakeDescription(state),
        base_version=base,
        next_version="",
        params={},
    )
    assert got == expected
    assert Version(got)  # parses as valid PEP 440


def test_format_without_variant_matches_stock_templates(variant):
    variant(None)
    base = _versioning.tag2version(tag="0.0.0.dev1", params={})
    got = _versioning.format(
        description=FakeDescription("distance"),
        base_version=base,
        next_version="",
        params={},
    )
    assert got == "0.0.0.dev1+3.gdeadbee"


@pytest.mark.parametrize(
    "version,expected",
    [
        ("0.0.0.dev1+cpu", "cpu"),
        ("0.0.0.dev1+cu128", "cu128"),
        ("0.0.0.dev1+cpu.3.gdeadbee", "cpu"),
        ("0.0.0.dev1+cpu.3.gdeadbee.dirty", "cpu"),
        ("0.0.0.dev1", None),
        ("0.0.0.dev1+3.gdeadbee", None),
        ("0+unknown", None),
    ],
)
def test_parse_backend(version, expected):
    from fastfields.dlpack import _parse_backend

    assert _parse_backend(version) == expected


def test_backend_attribute_agrees_with_version():
    import fastfields.dlpack as ff

    assert ff.backend == ff._parse_backend(ff.__version__)
