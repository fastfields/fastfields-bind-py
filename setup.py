"""Manual (no-CMake) build for fastfields_bind.

Custom build_ext that:
  1. builds the fastfields-lib shared libraries (via its Makefile) if missing,
  2. copies libfastfields.so + libfastfields-cpu.so into fastfields_bind/lib/,
  3. compiles the nanobind extension (src/ext.cpp + nb_combined.cpp) against
     ./fastfields, linking -lfastfields with an $ORIGIN/lib rpath,
  4. ships the .so libraries as package data.

./_fastfields_lib is treated as the fastfields-lib source tree; it works
whether that path is a symlink (dev) or a git submodule (release). The Python
package it builds is the PEP 420 namespace subpackage ``fastfields.dlpack``
(the compiled extension imports as ``fastfields.dlpack._core``); no
``fastfields/__init__.py`` is created, so other distributions can merge into
the same ``fastfields`` namespace.
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
FASTFIELDS_DIR = os.path.join(HERE, "_fastfields_lib")
FASTFIELDS_BUILD = os.path.join(FASTFIELDS_DIR, "build")
PKG_DIR = os.path.join(HERE, "fastfields", "dlpack")
PKG_LIB_DIR = os.path.join(PKG_DIR, "lib")

# The fastfields-lib Makefile names its shared libraries per platform, exactly
# as the loader expects: .so on Linux, .dylib on macOS, .dll on Windows. Match
# that here (a hard-coded ".so" silently broke the macOS/Windows CI legs). The
# extension is found next to its shipped libs via an rpath that also differs by
# platform ($ORIGIN on ELF, @loader_path on Mach-O); Windows has no rpath and
# instead relies on os.add_dll_directory at import time (see __init__.py).
if sys.platform == "darwin":
    LIBEXT = "dylib"
    RPATH_FLAG = "-Wl,-rpath,@loader_path/lib"
elif sys.platform == "win32":
    LIBEXT = "dll"
    RPATH_FLAG = None
else:
    LIBEXT = "so"
    RPATH_FLAG = "-Wl,-rpath,$ORIGIN/lib"

# Shared libraries produced by the fastfields-lib Makefile.
MAIN_LIB = "libfastfields." + LIBEXT
CPU_LIB = "libfastfields-cpu." + LIBEXT
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
        subprocess.check_call(["make", "-C", FASTFIELDS_DIR, f"CXX={CXX}"])
        if not (
            os.path.exists(MAIN_LIB_PATH) and os.path.exists(CPU_LIB_PATH)
        ):
            raise RuntimeError(
                "fastfields-lib build did not produce the expected .so files"
            )

    # -- step 2: ship the libraries inside the package -----------------------
    def _copy_libs_into_package(self):
        os.makedirs(PKG_LIB_DIR, exist_ok=True)
        for src in (MAIN_LIB_PATH, CPU_LIB_PATH):
            dst = os.path.join(PKG_LIB_DIR, os.path.basename(src))
            shutil.copyfile(src, dst)
            shutil.copymode(src, dst)
        # Mirror into the build tree so a wheel build also picks them up.
        if getattr(self, "build_lib", None):
            build_pkg_lib = os.path.join(
                self.build_lib, "fastfields", "dlpack", "lib"
            )
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

        cmd = [CXX, "-std=c++17", "-O2", "-shared", "-fvisibility=hidden"]
        # Position-independent code is POSIX-only (default on Windows PE/COFF).
        if sys.platform != "win32":
            cmd.append("-fPIC")
        for inc in include_dirs:
            cmd += ["-I", inc]
        cmd += sources
        if sys.platform == "win32":
            # No import library is produced for libfastfields.dll, so link the
            # extension directly against the DLL by path; the DLL is then
            # located at import time via os.add_dll_directory.
            cmd += [MAIN_LIB_PATH]
        else:
            cmd += ["-L", FASTFIELDS_BUILD, "-lfastfields"]
        if RPATH_FLAG:
            cmd += [RPATH_FLAG]
        cmd += ["-o", ext_path]
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
        "fastfields.dlpack._core",
        sources=[os.path.join("src", "ext.cpp")],
    )
]

setup(
    cmdclass={"build_ext": BuildExt},
    ext_modules=ext_modules,
)
