// nanobind bindings for fastfields-lib.
//
// We accept untyped `nb::ndarray<>` arguments: nanobind auto-imports any
// object exposing `__dlpack__` (numpy / torch / cupy / ...). Each ndarray is
// converted to a DLTensor with `to_dltensor` and handed to the C++ library,
// which operates in place / writes outputs through the DLTensor pointers.
//
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <cstring>
#include <optional>
#include <vector>

// All five public headers share the FF_LIB_BOUND_SPLINE_T guard, so they
// co-include cleanly and provide the correctly-namespaced ff:: declarations.
#include "distance.h"   // ff::dt_euclidean / dt_l1 / dt_spline_* / dt_mesh
#include "posdef.h"     // ff::sym_matvec / sym_solve / sym_invert / ...
#include "resize.h"     // ff::resample  + dlpack.h (DLTensor)
#include "restrict.h"   // ff::restriction
#include "splinc.h"     // ff::spline_coeff
#include "pushpull.h"   // ff::pull / push / count / grad
#include "reg_field.h"  // ff::field_matvec / field_diag
#include "reg_flow.h"   // ff::flow_matvec / flow_diag

namespace nb = nanobind;
using namespace nb::literals;
using arr = nb::ndarray<>;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Fill a DLTensor that views the memory of a nanobind ndarray. The shape and
// stride pointers alias nanobind's internal storage, which stays alive for the
// duration of the wrapped call (the ndarray argument outlives it), so no copy
// is required. Strides are already in elements (DLPack convention).
static DLTensor to_dltensor(arr &a) {
    DLTensor t;
    std::memset(&t, 0, sizeof(t));
    t.data = a.data_handle();
    t.device.device_type = (DLDeviceType)a.device_type();
    t.device.device_id = a.device_id();
    t.ndim = (int32_t)a.ndim();
    t.dtype.code = a.dtype().code;
    t.dtype.bits = a.dtype().bits;
    t.dtype.lanes = a.dtype().lanes;
    t.shape = const_cast<int64_t *>(a.shape_ptr());
    t.strides = const_cast<int64_t *>(a.stride_ptr());
    t.byte_offset = 0;
    return t;
}

// A DLTensor with null data: the library treats this as "absent" for the
// optional weight (posdef) and nearest_vertex (mesh) arguments.
static DLTensor null_dltensor() {
    DLTensor t;
    std::memset(&t, 0, sizeof(t));
    t.data = nullptr;
    return t;
}

static DLTensor opt_to_dltensor(std::optional<arr> &a) {
    return a.has_value() ? to_dltensor(*a) : null_dltensor();
}

// A `const double *` view of an optional Python sequence, or nullptr when the
// sequence is absent/empty (the library reads a null pointer as "use the
// default": all-ones voxel size, or a disabled penalty). Used by the
// regulariser bindings for voxel_size / absolute / membrane / bending.
static const double *vec_ptr(std::optional<std::vector<double>> &v) {
    return (v.has_value() && !v->empty()) ? v->data() : nullptr;
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------
NB_MODULE(_core, m) {
    m.doc() = "nanobind bindings to fastfields-lib (DLPack in/out).\n"
              "Arrays may be any object exposing __dlpack__ (numpy/torch/cupy).\n"
              "spline order: 0=Nearest 1=Linear 2=Quadratic 3=Cubic ... 7.\n"
              "bound: 0=Zero 1=Replicate 2=DCT1 3=DCT2 4=DST1 5=DST2 6=DFT 7=NoCheck.";

    // ----- distance.h -----
    m.def(
        "dt_euclidean",
        [](arr inp_out, double voxel_spacing, int stream) {
            DLTensor t = to_dltensor(inp_out);
            ff::dt_euclidean(t, voxel_spacing, stream);
        },
        "inp_out"_a, "voxel_spacing"_a = 1.0, "stream"_a = 0,
        "In-place Euclidean distance transform along the last axis "
        "(float32/float64; 0 at features, +inf elsewhere).");

    m.def(
        "dt_l1",
        [](arr inp_out, double voxel_spacing, int stream) {
            DLTensor t = to_dltensor(inp_out);
            ff::dt_l1(t, voxel_spacing, stream);
        },
        "inp_out"_a, "voxel_spacing"_a = 1.0, "stream"_a = 0,
        "In-place L1 distance transform along the last axis.");

    m.def(
        "dt_spline_table",
        [](arr time, arr dist, arr loc, arr coeff, arr times, int8_t spline,
           int8_t bound, int stream) {
            DLTensor t = to_dltensor(time), d = to_dltensor(dist),
                     l = to_dltensor(loc), c = to_dltensor(coeff),
                     ts = to_dltensor(times);
            ff::dt_spline_table(t, d, l, c, ts, spline, bound, stream);
        },
        "time"_a, "dist"_a, "loc"_a, "coeff"_a, "times"_a, "spline"_a = 3,
        "bound"_a = 3, "stream"_a = 0,
        "Point-to-spline distance via a dictionary of candidate times.");

    m.def(
        "dt_spline_brent",
        [](arr time, arr dist, arr loc, arr coeff, int64_t max_iter, double tol,
           double step, int8_t spline, int8_t bound, int stream) {
            DLTensor t = to_dltensor(time), d = to_dltensor(dist),
                     l = to_dltensor(loc), c = to_dltensor(coeff);
            ff::dt_spline_brent(t, d, l, c, max_iter, tol, step, spline, bound,
                              stream);
        },
        "time"_a, "dist"_a, "loc"_a, "coeff"_a, "max_iter"_a, "tol"_a, "step"_a,
        "spline"_a = 3, "bound"_a = 3, "stream"_a = 0,
        "Point-to-spline distance via Brent's method.");

    m.def(
        "dt_spline_gaussnewton",
        [](arr time, arr dist, arr loc, arr coeff, int64_t max_iter, double tol,
           int8_t spline, int8_t bound, int stream) {
            DLTensor t = to_dltensor(time), d = to_dltensor(dist),
                     l = to_dltensor(loc), c = to_dltensor(coeff);
            ff::dt_spline_gaussnewton(t, d, l, c, max_iter, tol, spline, bound,
                                    stream);
        },
        "time"_a, "dist"_a, "loc"_a, "coeff"_a, "max_iter"_a, "tol"_a,
        "spline"_a = 3, "bound"_a = 3, "stream"_a = 0,
        "Point-to-spline distance via Gauss-Newton optimization.");

    m.def(
        "dt_mesh",
        [](arr dist, std::optional<arr> nearest_vertex, arr loc, arr vertices,
           arr faces, bool signed_, bool naive, int stream) {
            DLTensor d = to_dltensor(dist);
            DLTensor nv = opt_to_dltensor(nearest_vertex);
            DLTensor l = to_dltensor(loc), v = to_dltensor(vertices),
                     f = to_dltensor(faces);
            ff::dt_mesh(d, nv, l, v, f, signed_, naive, stream);
        },
        "dist"_a, "nearest_vertex"_a.none() = nb::none(), "loc"_a, "vertices"_a,
        "faces"_a, "signed_"_a = true, "naive"_a = false, "stream"_a = 0,
        "Point-to-triangular-mesh (squared) distance; nearest_vertex optional.");

    // ----- posdef.h -----
    m.def(
        "sym_matvec",
        [](arr out, arr hessian, arr inp, int stream) {
            DLTensor o = to_dltensor(out), h = to_dltensor(hessian),
                     i = to_dltensor(inp);
            ff::sym_matvec(o, h, i, stream);
        },
        "out"_a, "hessian"_a, "inp"_a, "stream"_a = 0,
        "out = H @ inp (H is compact-symmetric, diagonal-then-rows packed).");

    m.def(
        "sym_matvec_backward",
        [](arr out, arr grd, arr inp, int stream) {
            DLTensor o = to_dltensor(out), g = to_dltensor(grd),
                     i = to_dltensor(inp);
            ff::sym_matvec_backward(o, g, i, stream);
        },
        "out"_a, "grd"_a, "inp"_a, "stream"_a = 0,
        "Backward of sym_matvec wrt the matrix.");

    m.def(
        "sym_addmatvec_",
        [](arr out, arr hessian, arr inp, int stream) {
            DLTensor o = to_dltensor(out), h = to_dltensor(hessian),
                     i = to_dltensor(inp);
            ff::sym_addmatvec_(o, h, i, stream);
        },
        "out"_a, "hessian"_a, "inp"_a, "stream"_a = 0, "out += H @ inp.");

    m.def(
        "sym_submatvec_",
        [](arr out, arr hessian, arr inp, int stream) {
            DLTensor o = to_dltensor(out), h = to_dltensor(hessian),
                     i = to_dltensor(inp);
            ff::sym_submatvec_(o, h, i, stream);
        },
        "out"_a, "hessian"_a, "inp"_a, "stream"_a = 0, "out -= H @ inp.");

    m.def(
        "sym_solve",
        [](arr out, arr hessian, arr inp, std::optional<arr> weight,
           int stream) {
            DLTensor o = to_dltensor(out), h = to_dltensor(hessian),
                     i = to_dltensor(inp);
            DLTensor w = opt_to_dltensor(weight);
            ff::sym_solve(o, h, i, w, stream);
        },
        "out"_a, "hessian"_a, "inp"_a, "weight"_a.none() = nb::none(),
        "stream"_a = 0, "out = (H + diag(weight)) \\ inp (weight optional).");

    m.def(
        "sym_solve_",
        [](arr inp_out, arr hessian, std::optional<arr> weight, int stream) {
            DLTensor io = to_dltensor(inp_out), h = to_dltensor(hessian);
            DLTensor w = opt_to_dltensor(weight);
            ff::sym_solve_(io, h, w, stream);
        },
        "inp_out"_a, "hessian"_a, "weight"_a.none() = nb::none(),
        "stream"_a = 0,
        "In-place: inp_out = (H + diag(weight)) \\ inp_out (weight optional).");

    m.def(
        "sym_invert",
        [](arr out, arr hessian, int stream) {
            DLTensor o = to_dltensor(out), h = to_dltensor(hessian);
            ff::sym_invert(o, h, stream);
        },
        "out"_a, "hessian"_a, "stream"_a = 0,
        "out = inv(H) (both compact-symmetric).");

    m.def(
        "sym_invert_",
        [](arr hessian, int stream) {
            DLTensor h = to_dltensor(hessian);
            ff::sym_invert_(h, stream);
        },
        "hessian"_a, "stream"_a = 0,
        "In-place: hessian = inv(hessian) (compact-symmetric).");

    // ----- resize.h / restrict.h (scale as a Python sequence) -----
    auto resize_like = [](arr &out, arr &inp, int8_t spline, int8_t bound,
                          double shift, std::optional<std::vector<double>> &scale,
                          int ndim, int stream, bool restriction) {
        DLTensor o = to_dltensor(out), i = to_dltensor(inp);
        const double *scale_ptr = nullptr;
        if (scale.has_value() && !scale->empty())
            scale_ptr = scale->data();
        if (restriction)
            ff::restriction(o, i, spline, bound, shift, scale_ptr, ndim, stream);
        else
            ff::resample(o, i, spline, bound, shift, scale_ptr, ndim, stream);
    };

    m.def(
        "resample",
        [resize_like](arr out, arr inp, int8_t spline, int8_t bound,
                      double shift, std::optional<std::vector<double>> scale,
                      int ndim, int stream) {
            resize_like(out, inp, spline, bound, shift, scale, ndim, stream,
                        false);
        },
        "out"_a, "inp"_a, "spline"_a = 2, "bound"_a = 3, "shift"_a = 0.0,
        "scale"_a.none() = nb::none(), "ndim"_a = 1, "stream"_a = 0,
        "Spline resample (prolongation). scale is a per-dim sequence of length "
        "ndim (input-index per output-index).");

    m.def(
        "restriction",
        [resize_like](arr out, arr inp, int8_t spline, int8_t bound,
                      double shift, std::optional<std::vector<double>> scale,
                      int ndim, int stream) {
            resize_like(out, inp, spline, bound, shift, scale, ndim, stream,
                        true);
        },
        "out"_a, "inp"_a, "spline"_a = 2, "bound"_a = 3, "shift"_a = 0.0,
        "scale"_a.none() = nb::none(), "ndim"_a = 1, "stream"_a = 0,
        "Restriction (adjoint of resample). out is accumulated into and must be "
        "pre-zeroed by the caller.");

    // ----- splinc.h -----
    m.def(
        "spline_coeff",
        [](arr inp_out, int8_t spline, int8_t bound, int stream) {
            DLTensor t = to_dltensor(inp_out);
            ff::spline_coeff(t, spline, bound, stream);
        },
        "inp_out"_a, "spline"_a = 3, "bound"_a = 3, "stream"_a = 0,
        "In-place spline-coefficient prefilter along the last axis "
        "(orders 0/1 are no-ops).");

    // ----- pushpull.h -----
    // Channel-last, x-first coordinate convention. `grid` holds sampling
    // coordinates in voxels; D (its last dim) is the spatial rank (1/2/3).
    m.def(
        "pull",
        [](arr out, arr inp, arr grid, int8_t spline, int8_t bound,
           int8_t extrapolate, int stream) {
            DLTensor o = to_dltensor(out), i = to_dltensor(inp),
                     g = to_dltensor(grid);
            ff::pull(o, i, g, spline, bound, extrapolate, stream);
        },
        "out"_a, "inp"_a, "grid"_a, "spline"_a = 2, "bound"_a = 3,
        "extrapolate"_a = 1, "stream"_a = 0,
        "Sample (pull) a spline-encoded volume at grid coordinates. "
        "inp (*batch,*inshape,C), grid (*batch,*outshape,D), "
        "out (*batch,*outshape,C).");

    m.def(
        "push",
        [](arr out, arr inp, arr grid, int8_t spline, int8_t bound,
           int8_t extrapolate, int stream) {
            DLTensor o = to_dltensor(out), i = to_dltensor(inp),
                     g = to_dltensor(grid);
            ff::push(o, i, g, spline, bound, extrapolate, stream);
        },
        "out"_a, "inp"_a, "grid"_a, "spline"_a = 2, "bound"_a = 3,
        "extrapolate"_a = 1, "stream"_a = 0,
        "Splat (push) values into a volume; adjoint of pull. out "
        "(*batch,*inshape,C) is accumulated into and must be pre-zeroed.");

    m.def(
        "count",
        [](arr out, arr grid, int8_t spline, int8_t bound, int8_t extrapolate,
           int stream) {
            DLTensor o = to_dltensor(out), g = to_dltensor(grid);
            ff::count(o, g, spline, bound, extrapolate, stream);
        },
        "out"_a, "grid"_a, "spline"_a = 2, "bound"_a = 3, "extrapolate"_a = 1,
        "stream"_a = 0,
        "Splat ones (push of an all-ones input). out (*batch,*inshape,1) must "
        "be pre-zeroed.");

    m.def(
        "grad",
        [](arr out, arr inp, arr grid, int8_t spline, int8_t bound,
           int8_t extrapolate, bool abs, int stream) {
            DLTensor o = to_dltensor(out), i = to_dltensor(inp),
                     g = to_dltensor(grid);
            ff::grad(o, i, g, spline, bound, extrapolate, abs, stream);
        },
        "out"_a, "inp"_a, "grid"_a, "spline"_a = 2, "bound"_a = 3,
        "extrapolate"_a = 1, "abs"_a = false, "stream"_a = 0,
        "Sample spatial gradients of a spline-encoded volume. "
        "out (*batch,*outshape,C,D).");

    // ----- reg_field.h (multi-channel field; per-channel penalty vectors) -----
    // voxel_size is a length-ndim sequence; absolute/membrane/bending are
    // length-C sequences (any may be omitted -> that penalty is disabled).
    m.def(
        "field_matvec",
        [](arr out, arr inp, std::optional<std::vector<double>> voxel_size,
           std::optional<std::vector<double>> absolute,
           std::optional<std::vector<double>> membrane,
           std::optional<std::vector<double>> bending, int8_t bound, int ndim,
           int stream) {
            DLTensor o = to_dltensor(out), i = to_dltensor(inp);
            ff::field_matvec(o, i, vec_ptr(voxel_size), vec_ptr(absolute),
                             vec_ptr(membrane), vec_ptr(bending), bound, ndim,
                             stream);
        },
        "out"_a, "inp"_a, "voxel_size"_a.none() = nb::none(),
        "absolute"_a.none() = nb::none(), "membrane"_a.none() = nb::none(),
        "bending"_a.none() = nb::none(), "bound"_a = 3, "ndim"_a = 1,
        "stream"_a = 0,
        "Apply a spatial regulariser to a multi-channel field "
        "(*batch,*spatial,C).");

    m.def(
        "field_diag",
        [](arr out, std::optional<std::vector<double>> voxel_size,
           std::optional<std::vector<double>> absolute,
           std::optional<std::vector<double>> membrane,
           std::optional<std::vector<double>> bending, int8_t bound, int ndim,
           int stream) {
            DLTensor o = to_dltensor(out);
            ff::field_diag(o, vec_ptr(voxel_size), vec_ptr(absolute),
                           vec_ptr(membrane), vec_ptr(bending), bound, ndim,
                           stream);
        },
        "out"_a, "voxel_size"_a.none() = nb::none(),
        "absolute"_a.none() = nb::none(), "membrane"_a.none() = nb::none(),
        "bending"_a.none() = nb::none(), "bound"_a = 3, "ndim"_a = 1,
        "stream"_a = 0,
        "Diagonal (preconditioner) of the field regulariser operator.");

    m.def(
        "field_kernel",
        [](arr out, std::optional<std::vector<double>> voxel_size,
           std::optional<std::vector<double>> absolute,
           std::optional<std::vector<double>> membrane,
           std::optional<std::vector<double>> bending, int8_t bound, int ndim,
           int stream) {
            DLTensor o = to_dltensor(out);
            ff::field_kernel(o, vec_ptr(voxel_size), vec_ptr(absolute),
                             vec_ptr(membrane), vec_ptr(bending), bound, ndim,
                             stream);
        },
        "out"_a, "voxel_size"_a.none() = nb::none(),
        "absolute"_a.none() = nb::none(), "membrane"_a.none() = nb::none(),
        "bending"_a.none() = nb::none(), "bound"_a = 3, "ndim"_a = 1,
        "stream"_a = 0,
        "Materialise the Toeplitz convolution kernel of the field "
        "regulariser.");

    // ----- reg_flow.h (vector flow field; scalar penalties) -----
    m.def(
        "flow_matvec",
        [](arr out, arr inp, std::optional<std::vector<double>> voxel_size,
           double absolute, double membrane, double bending, double shears,
           double div, int8_t bound, int ndim, int stream) {
            DLTensor o = to_dltensor(out), i = to_dltensor(inp);
            ff::flow_matvec(o, i, vec_ptr(voxel_size), absolute, membrane,
                            bending, shears, div, bound, ndim, stream);
        },
        "out"_a, "inp"_a, "voxel_size"_a.none() = nb::none(),
        "absolute"_a = 0.0, "membrane"_a = 0.0, "bending"_a = 0.0,
        "shears"_a = 0.0, "div"_a = 0.0,
        "bound"_a = 3, "ndim"_a = 1, "stream"_a = 0,
        "Apply a spatial regulariser to a vector flow field.");

    m.def(
        "flow_diag",
        [](arr out, std::optional<std::vector<double>> voxel_size,
           double absolute, double membrane, double bending, double shears,
           double div, int8_t bound, int ndim, int stream) {
            DLTensor o = to_dltensor(out);
            ff::flow_diag(o, vec_ptr(voxel_size), absolute, membrane, bending,
                          shears, div, bound, ndim, stream);
        },
        "out"_a, "voxel_size"_a.none() = nb::none(), "absolute"_a = 0.0,
        "membrane"_a = 0.0, "bending"_a = 0.0, "shears"_a = 0.0, "div"_a = 0.0,
        "bound"_a = 3, "ndim"_a = 1, "stream"_a = 0,
        "Diagonal (preconditioner) of the flow regulariser operator.");

    m.def(
        "flow_relax",
        [](arr sol, arr hes, arr grd,
           std::optional<std::vector<double>> voxel_size, double absolute,
           double membrane, double bending, double shears, double div,
           int8_t bound, int ndim, int nb_iter, int stream) {
            DLTensor s = to_dltensor(sol), h = to_dltensor(hes),
                     g = to_dltensor(grd);
            ff::flow_relax(s, h, g, vec_ptr(voxel_size), absolute, membrane,
                           bending, shears, div, bound, ndim, nb_iter, stream);
        },
        "sol"_a, "hes"_a, "grd"_a, "voxel_size"_a.none() = nb::none(),
        "absolute"_a = 0.0, "membrane"_a = 0.0, "bending"_a = 0.0,
        "shears"_a = 0.0, "div"_a = 0.0, "bound"_a = 3, "ndim"_a = 1,
        "nb_iter"_a = 1, "stream"_a = 0,
        "In-place relaxation sweeps solving (H + L) x = g for the flow field.");

    m.def(
        "flow_kernel",
        [](arr out, std::optional<std::vector<double>> voxel_size,
           double absolute, double membrane, double bending, double shears,
           double div, int8_t bound, int ndim, int stream) {
            DLTensor o = to_dltensor(out);
            ff::flow_kernel(o, vec_ptr(voxel_size), absolute, membrane,
                            bending, shears, div, bound, ndim, stream);
        },
        "out"_a, "voxel_size"_a.none() = nb::none(), "absolute"_a = 0.0,
        "membrane"_a = 0.0, "bending"_a = 0.0, "shears"_a = 0.0, "div"_a = 0.0,
        "bound"_a = 3, "ndim"_a = 1, "stream"_a = 0,
        "Materialise the Toeplitz convolution kernel of the flow regulariser.");
}
