# fastfields-dlpack

`fastfields-dlpack` provides **nanobind** bindings from DLPack to the `fastfields-lib` C++/CUDA library. It imports as **`fastfields.dlpack`** and is the base of the Python stack: raw, in-place bindings that every friendly wrapper (numpy / cupy / torch / any) sits on. All functions accept any array exposing `__dlpack__` (numpy, torch, cupy) and operate in place / write through pre-allocated outputs.

## Installation

```bash
pip install fastfields-dlpack
```

## Usage

```python
import numpy as np
import fastfields.dlpack as ff

x = np.array([0, np.inf, np.inf, 0, np.inf], dtype=np.float32)
ff.dt_euclidean(x)          # in-place squared Euclidean distance transform
```

See the [API reference](api/index.md) for the full list of operations.
