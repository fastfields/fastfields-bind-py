# fastfields-dlpack

**fastfields-dlpack** is the low-level core of the fastfields project. It imports
as `fastfields.dlpack` and exposes the field operators as raw, in-place calls
that work directly on any array — NumPy, PyTorch or CuPy — sharing memory with
zero copies.

Most people don't use this package directly. The friendly, array-returning
wrappers — [`fastfields.numpy`](https://fastfields.github.io/fastfields-numpy/),
[`fastfields.torch`](https://fastfields.github.io/fastfields-torch/),
[`fastfields.cupy`](https://fastfields.github.io/fastfields-cupy/) and the
unified [`fastfields.auto`](https://fastfields.github.io/fastfields/) — are built
on top of it and are what you normally want. Reach for `fastfields.dlpack` when
you want the thinnest possible layer and are happy to manage output buffers
yourself.

## Install

```sh
pip install fastfields-dlpack \
    --extra-index-url https://fastfields.github.io/whl/cpu/
```

## Use it

Functions write results in place or into a pre-allocated output you pass in:

```python
import numpy as np
import fastfields.dlpack as ff

x = np.array([0, np.inf, np.inf, 0, np.inf], dtype=np.float32)
ff.dt_euclidean(x)          # in-place squared Euclidean distance transform
```

## What's inside

The same operation families as the higher-level packages, in their raw in-place
form: distance transforms (`dt_euclidean`, `dt_l1`, the `dt_spline_*` and
`dt_mesh` point distances), positive-definite linear algebra (`sym_matvec`,
`sym_addmatvec_`, `sym_solve`, `sym_invert`, …), and resampling (`resample`,
`restriction`, `spline_coeff`). The `Spline` and `Bound` enums document the
integer order/boundary arguments.

See the [API reference](api/index.md) for full signatures and options.
