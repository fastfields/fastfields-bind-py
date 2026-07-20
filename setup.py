"""Manual (no-CMake) build for fastfields_bind.

Custom build_ext that:
  1. builds the fastfields-lib shared libraries (via its Makefile) if missing,
  2. copies libfastfields.so + libfastfields-cpu.so into fastfields_bind/lib/,
  3. compiles the nanobind extension (src/ext.cpp + nb_combined.cpp) against
     ./fastfields, linking -lfastfields with an $ORIGIN/lib rpath,
  4. ships the .so libraries as package data.

./fastfields is treated as the fastfields-lib source tree; it works whether
that path is a symlink (dev) or a git submodule (release).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import sysconfig

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

HERE = os.path.dirname(os.path.abspath(__file__))
FASTFIELDS_DIR = os.path.join(HERE, "fastfields")
FASTFIELDS_BUILD = os.path.join(FASTFIELDS_DIR, "build")
PKG_DIR = os.path.join(HERE, "fastfields_bind")
PKG_LIB_DIR = os.path.join(PKG_DIR, "lib")

# Shared libraries produced by the fastfields-lib Makefile.
MAIN_LIB = "libfastfields.so"
CPU_LIB = "libfastfields-cpu.so"
MAIN_LIB_PATH = os.path.join(FASTFIELDS_BUILD, MAIN_LIB)
CPU_LIB_PATH = os.path.join(FASTFIELDS_BUILD, "lib", CPU_LIB)

CXX = os.environ.get("CXX", "clang++")


def _nanobind_paths():
    import nanobind

    nb_root = os.path.dirname(nanobind.__file__)
    return {
        "include": nanobind.include_dir(),
        "robin_map": os.path.join(nb_root, "ext", "robin_map", "include"),
        "combined_src": os.path.join(nb_root, "src", "nb_combined.cpp"),
    }


class BuildExt(build_ext):
    def run(self):
        self._ensure_fastfields_libs()
        self._copy_libs_into_package()
        super().run()

    # -- step 1: build the C++ libraries if they are not there yet -----------
    def _ensure_fastfields_libs(self):
        if os.path.exists(MAIN_LIB_PATH) and os.path.exists(CPU_LIB_PATH):
            return
        if not os.path.isdir(FASTFIELDS_DIR):
            raise RuntimeError(
                f"fastfields source tree not found at {FASTFIELDS_DIR!r}. "
                "Initialise the git submodule (or create the symlink)."
            )
        print(f"[fastfields_bind] building fastfields-lib in {FASTFIELDS_DIR}")
        subprocess.check_call(
            ["make", "-C", FASTFIELDS_DIR, f"CXX={CXX}"]
        )
        if not (os.path.exists(MAIN_LIB_PATH) and os.path.exists(CPU_LIB_PATH)):
            raise RuntimeError("fastfields-lib build did not produce the expected .so files")

    # -- step 2: ship the libraries inside the package -----------------------
    def _copy_libs_into_package(self):
        os.makedirs(PKG_LIB_DIR, exist_ok=True)
        for src in (MAIN_LIB_PATH, CPU_LIB_PATH):
            dst = os.path.join(PKG_LIB_DIR, os.path.basename(src))
            shutil.copyfile(src, dst)
            shutil.copymode(src, dst)
        # Mirror into the build tree so a wheel build also picks them up.
        if getattr(self, "build_lib", None):
            build_pkg_lib = os.path.join(self.build_lib, "fastfields_bind", "lib")
            os.makedirs(build_pkg_lib, exist_ok=True)
            for name in (MAIN_LIB, CPU_LIB):
                shutil.copyfile(
                    os.path.join(PKG_LIB_DIR, name),
                    os.path.join(build_pkg_lib, name),
                )

    # -- step 3: compile the nanobind extension ------------------------------
    def build_extension(self, ext):
        nb = _nanobind_paths()

        include_dirs = [
            nb["include"],
            nb["robin_map"],
            sysconfig.get_path("include"),
            FASTFIELDS_DIR,
        ]
        sources = list(ext.sources) + [nb["combined_src"]]

        ext_path = self.get_ext_fullpath(ext.name)
        os.makedirs(os.path.dirname(ext_path), exist_ok=True)

        cmd = [
            CXX,
            "-std=c++17",
            "-fPIC",
            "-O2",
            "-shared",
            "-fvisibility=hidden",
        ]
        for inc in include_dirs:
            cmd += ["-I", inc]
        cmd += sources
        cmd += [
            "-L", FASTFIELDS_BUILD,
            "-lfastfields",
            "-Wl,-rpath,$ORIGIN/lib",
            "-o", ext_path,
        ]
        print("[fastfields_bind] " + " ".join(cmd))
        subprocess.check_call(cmd)

        # Ensure the libs sit next to the freshly-built extension too (covers
        # both in-place/editable builds and staged wheel builds).
        ext_lib_dir = os.path.join(os.path.dirname(ext_path), "lib")
        os.makedirs(ext_lib_dir, exist_ok=True)
        for name in (MAIN_LIB, CPU_LIB):
            shutil.copyfile(
                os.path.join(PKG_LIB_DIR, name),
                os.path.join(ext_lib_dir, name),
            )


ext_modules = [
    Extension(
        "fastfields_bind._core",
        sources=[os.path.join("src", "ext.cpp")],
    )
]

setup(
    cmdclass={"build_ext": BuildExt},
    ext_modules=ext_modules,
)
