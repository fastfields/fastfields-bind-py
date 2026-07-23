#!/usr/bin/env python3
"""Stamp a compute-backend **local version label** onto a built wheel.

Release wheels encode their backend in the PEP 440 *local version* segment
(PyTorch convention): ``fastfields_dlpack-0.1.0+cu128-cp311-…whl``. Neither
cibuildwheel nor ``python -m build`` has a hook to set that, so this repacks
the wheel: rewrite the ``Version`` in ``METADATA``, rename the ``.dist-info``
directory, and regenerate ``RECORD`` via ``wheel pack``.

Usage
-----
``python scripts/apply_local_version.py <wheel-or-glob> <dest_dir> <label>``

e.g. ``python scripts/apply_local_version.py dist/*.whl wheelhouse cu128``.

The label *replaces* any existing local segment, so release builds should start
from a clean tagged version (``0.1.0``), giving ``0.1.0+cu128``.
"""

from __future__ import annotations

import glob
import subprocess
import sys
import tempfile
from pathlib import Path


def _rewrite(unpacked_root: Path, base_name: str, new_version: str) -> Path:
    """Rewrite METADATA + rename the dist-info dir to ``new_version``.

    Parameters
    ----------
    unpacked_root : pathlib.Path
        Directory ``wheel unpack`` produced (``<name>-<oldver>/``).
    base_name : str
        Distribution name as it appears in the dist-info dir.
    new_version : str
        The version to stamp (e.g. ``0.1.0+cu128``).

    Returns
    -------
    pathlib.Path
        The renamed unpacked root (``<name>-<new_version>/``).
    """
    old_di = next(unpacked_root.glob("*.dist-info"))
    meta = old_di / "METADATA"
    lines = meta.read_text().splitlines()
    meta.write_text(
        "\n".join(
            f"Version: {new_version}" if ln.startswith("Version:") else ln
            for ln in lines
        )
        + "\n"
    )
    new_di = old_di.with_name(f"{base_name}-{new_version}.dist-info")
    old_di.rename(new_di)
    new_root = unpacked_root.with_name(f"{base_name}-{new_version}")
    unpacked_root.rename(new_root)
    return new_root


def apply(wheel: Path, dest_dir: Path, label: str) -> Path:
    """Repack ``wheel`` with local version ``label`` into ``dest_dir``.

    Parameters
    ----------
    wheel : pathlib.Path
        The input wheel.
    dest_dir : pathlib.Path
        Where the relabelled wheel is written.
    label : str
        The local version label (``cpu``, ``cu128``, …).

    Returns
    -------
    pathlib.Path
        Path to the written wheel.
    """
    name, version = wheel.name.split("-")[:2]
    base_version = version.split("+")[0]
    new_version = f"{base_version}+{label}"
    dest_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        subprocess.check_call(
            [sys.executable, "-m", "wheel", "unpack", str(wheel), "-d", tmp]
        )
        unpacked = next(tmp_path.iterdir())
        new_root = _rewrite(unpacked, name, new_version)
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "wheel",
                "pack",
                str(new_root),
                "-d",
                str(dest_dir),
            ]
        )
    out = next(dest_dir.glob(f"{name}-{new_version}-*.whl"))
    print(f"stamped {wheel.name} -> {out.name}")
    return out


def main(argv: list[str]) -> int:
    """CLI entry point."""
    if len(argv) != 3:
        print(__doc__)
        return 2
    pattern, dest, label = argv
    wheels = [Path(p) for p in glob.glob(pattern)]
    if not wheels:
        print(f"no wheels matched {pattern!r}", file=sys.stderr)
        return 1
    for wheel in wheels:
        apply(wheel, Path(dest), label)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
