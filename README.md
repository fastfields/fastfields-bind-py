# fastfields-bind

nanobind bindings from DLPack to the [`fastfields-lib`](./fastfields) C++ library.

All functions accept any array exposing `__dlpack__` (numpy, torch, cupy).
Arguments documented as in-place / outputs are written through their DLPack
data pointers.

```python
import numpy as np
import fastfields_bind as ff

x = np.array([0, np.inf, np.inf, 0, np.inf], dtype=np.float32)
ff.dt_euclidean(x)          # in-place squared Euclidean distance transform
```

## Build

```bash
pip install -e .
```

`./fastfields` must be the fastfields-lib source tree (a git submodule in a
release checkout, a symlink in this dev tree). `setup.py` builds its shared
libraries via `make` if they are missing, ships them inside the wheel under
`fastfields_bind/lib/`, and compiles the nanobind extension against them.

Enum helpers `ff.Spline` (0=Nearest .. 7) and `ff.Bound`
(0=Zero,1=Replicate,2=DCT1,3=DCT2,4=DST1,5=DST2,6=DFT,7=NoCheck) document the
integer `spline`/`bound` arguments.
