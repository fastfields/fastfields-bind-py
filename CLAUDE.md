# fastfields-bind-py  (imports as `fastfields.dlpack`)

**nanobind** bindings from DLPack to the `fastfields-lib` C++/CUDA library. This
is the base of the Python stack: the raw, in-place bindings that every friendly
wrapper (numpy/cupy/torch/any) sits on.

```
… ─ lib ─ bind-py ← (you are here) ─ {numpy,cupy,torch} ─ fastfields
```

- Submodule `_fastfields_lib -> fastfields-lib` (symlink in dev; real submodule
  in a release checkout, pinned to branch
  `claude/jitfields-fastfields-migration-v5r416`).
- Distribution name `fastfields-dlpack`; imports as **`fastfields.dlpack`**.

## Philosophy / role
- Thin, close-to-the-metal bindings. Functions accept **any `__dlpack__`
  object** (numpy / torch / cupy) and operate **in place** / write through
  pre-allocated output arrays via the DLPack data pointer.
- Exposes the same operation families as `fastfields-lib`: distance transforms &
  point-to-spline/mesh distance, posdef (compact-symmetric) linear algebra,
  resampling (`resample`/`restriction`/`spline_coeff`), pushpull, regularisers.
- Enum helpers `ff.Spline` (0=Nearest … 7) and `ff.Bound`
  (0=Zero,1=Replicate,2=DCT1,3=DCT2,4=DST1,5=DST2,6=DFT,7=NoCheck) document the
  integer `spline`/`bound` arguments.

## Layout
- `src/ext.cpp` — the nanobind extension source (builds as
  `fastfields.dlpack._core`).
- `fastfields/dlpack/__init__.py` — Python surface; `lib/*.so` holds the shipped
  shared libraries.
- `setup.py` — custom `build_ext` (no CMake). `tests/test_bindings.py`.

## Build & test
```
pip install .                     # regular install (NOT editable — see caveat)
python -m pytest tests/ -q        # import from a neutral cwd
```
`setup.py` builds `libfastfields.so` + `libfastfields-cpu.so` via the
`_fastfields_lib` Makefile **if missing**, copies them into
`fastfields/dlpack/lib/`, and compiles the nanobind extension against them with
an `$ORIGIN/lib` rpath. It respects `CXX` (default `clang++`); the extension is
built `-std=c++17` even though the C++ libs are C++11.

## Conventions & caveats
- **PEP 420 namespace package**: this distribution ships only
  `fastfields/dlpack/` and **never a `fastfields/__init__.py`**, so it merges
  with `fastfields-numpy`/`-cupy`/`-torch`/`fastfields` into one `fastfields`
  namespace. Do not add a top-level `__init__.py`.
- **Editable installs break the native-namespace merge** across the sibling
  distributions — prefer a regular `pip install .`. Import from a neutral cwd
  (not the repo root) so the installed package, not the source dir, is found.
- Ruff config in `pyproject.toml`: line-length 79, select B/E/F/I/W.

## Pointers
- Hierarchy: `/home/user/.github/profile/README.md`.
- Underlying C++ API + status: `/home/user/fastfields-lib/MIGRATION.md`.
