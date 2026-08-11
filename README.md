# fastfields-dlpack

nanobind bindings from DLPack to the [`fastfields-lib`](./_fastfields_lib) C++
library. Imports as **`fastfields.dlpack`** (a PEP 420 namespace subpackage; this
distribution ships only `fastfields/dlpack/` and never a `fastfields/__init__.py`,
so it merges with `fastfields-numpy`/`-torch`/`-cupy`/`fastfields`).

All functions accept any array exposing `__dlpack__` (numpy, torch, cupy).
Arguments documented as in-place / outputs are written through their DLPack
data pointers.

```python
import numpy as np
import fastfields.dlpack as ff

x = np.array([0, np.inf, np.inf, 0, np.inf], dtype=np.float32)
ff.dt_euclidean(x)          # in-place squared Euclidean distance transform
```

On CUDA arrays the calls are asynchronous: they enqueue work on the `stream`
you pass and return. **Every array must be kept alive by the caller until that
stream is synchronized** — see [`docs/cuda.md`](./docs/cuda.md).

Structurally malformed arrays (0-d, null data, vector dtypes, or a single call
mixing devices) are rejected with a `ValueError` naming the function and the
argument; shape/dtype contracts are checked by `fastfields-lib` one layer down.

## Build

```bash
pip install .
```

(Editable installs can break the native-namespace merge across the sibling
`fastfields-*` distributions; prefer a regular `pip install .`.)

`./_fastfields_lib` must be the fastfields-lib source tree (a git submodule in a
release checkout, a symlink in this dev tree). `setup.py` builds its shared
libraries via `make` if they are missing, ships them inside the wheel under
`fastfields/dlpack/lib/`, and compiles the nanobind extension
(`fastfields.dlpack._core`) against them with an `$ORIGIN/lib` rpath.

Enum helpers `ff.Spline` (0=Nearest .. 7) and `ff.Bound`
(0=Zero,1=Replicate,2=DCT1,3=DCT2,4=DST1,5=DST2,6=DFT,7=NoCheck) document the
integer `spline`/`bound` arguments.
