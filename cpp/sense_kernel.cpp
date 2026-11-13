#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <stdexcept>

namespace py = pybind11;

void require_same_shape(const py::array_t<float>& left, const py::array_t<float>& right) {
    if (left.ndim() != right.ndim()) {
        throw std::runtime_error("array ranks differ");
    }
    for (ssize_t axis = 0; axis < left.ndim(); ++axis) {
        if (left.shape(axis) != right.shape(axis)) {
            throw std::runtime_error("array shapes differ");
        }
    }
}

py::tuple image_to_coils(
    py::array_t<float> image_real,
    py::array_t<float> image_imag,
    py::array_t<float> sens_real,
    py::array_t<float> sens_imag) {
    require_same_shape(image_real, image_imag);
    require_same_shape(sens_real, sens_imag);
    if (image_real.ndim() != 2 || sens_real.ndim() != 3) {
        throw std::runtime_error("expected image (H, W) and sensitivities (C, H, W)");
    }
    auto image_r = image_real.unchecked<2>();
    auto image_i = image_imag.unchecked<2>();
    auto sens_r = sens_real.unchecked<3>();
    auto sens_i = sens_imag.unchecked<3>();
    const ssize_t coils = sens_r.shape(0);
    const ssize_t height = sens_r.shape(1);
    const ssize_t width = sens_r.shape(2);
    if (image_r.shape(0) != height || image_r.shape(1) != width) {
        throw std::runtime_error("image and sensitivity spatial shapes differ");
    }

    py::array_t<float> coil_real({coils, height, width});
    py::array_t<float> coil_imag({coils, height, width});
    auto out_r = coil_real.mutable_unchecked<3>();
    auto out_i = coil_imag.mutable_unchecked<3>();
    for (ssize_t coil = 0; coil < coils; ++coil) {
        for (ssize_t y = 0; y < height; ++y) {
            for (ssize_t x = 0; x < width; ++x) {
                const float gain_r = sens_r(coil, y, x);
                const float gain_i = sens_i(coil, y, x);
                const float value_r = image_r(y, x);
                const float value_i = image_i(y, x);
                out_r(coil, y, x) = gain_r * value_r - gain_i * value_i;
                out_i(coil, y, x) = gain_r * value_i + gain_i * value_r;
            }
        }
    }
    return py::make_tuple(coil_real, coil_imag);
}

py::tuple coils_to_image(
    py::array_t<float> sens_real,
    py::array_t<float> sens_imag,
    py::array_t<float> coil_real,
    py::array_t<float> coil_imag) {
    require_same_shape(sens_real, sens_imag);
    require_same_shape(coil_real, coil_imag);
    require_same_shape(sens_real, coil_real);
    if (sens_real.ndim() != 3) {
        throw std::runtime_error("expected arrays shaped (C, H, W)");
    }
    auto sens_r = sens_real.unchecked<3>();
    auto sens_i = sens_imag.unchecked<3>();
    auto coil_r = coil_real.unchecked<3>();
    auto coil_i = coil_imag.unchecked<3>();
    const ssize_t coils = sens_r.shape(0);
    const ssize_t height = sens_r.shape(1);
    const ssize_t width = sens_r.shape(2);

    py::array_t<float> image_real({height, width});
    py::array_t<float> image_imag({height, width});
    auto out_r = image_real.mutable_unchecked<2>();
    auto out_i = image_imag.mutable_unchecked<2>();
    for (ssize_t y = 0; y < height; ++y) {
        for (ssize_t x = 0; x < width; ++x) {
            float sum_r = 0.f;
            float sum_i = 0.f;
            for (ssize_t coil = 0; coil < coils; ++coil) {
                const float gain_r = sens_r(coil, y, x);
                const float gain_i = sens_i(coil, y, x);
                const float value_r = coil_r(coil, y, x);
                const float value_i = coil_i(coil, y, x);
                sum_r += gain_r * value_r + gain_i * value_i;
                sum_i += gain_r * value_i - gain_i * value_r;
            }
            out_r(y, x) = sum_r;
            out_i(y, x) = sum_i;
        }
    }
    return py::make_tuple(image_real, image_imag);
}

py::tuple apply_phase_mask(
    py::array_t<float> k_real,
    py::array_t<float> k_imag,
    py::array_t<float> mask) {
    require_same_shape(k_real, k_imag);
    if (k_real.ndim() != 3 || mask.ndim() != 1) {
        throw std::runtime_error("expected k-space (C, Ky, Kx) and mask (Ky,)");
    }
    if (mask.shape(0) != k_real.shape(1)) {
        throw std::runtime_error("mask length does not match phase-encode count");
    }
    auto in_r = k_real.unchecked<3>();
    auto in_i = k_imag.unchecked<3>();
    auto weights = mask.unchecked<1>();
    const ssize_t coils = in_r.shape(0);
    const ssize_t height = in_r.shape(1);
    const ssize_t width = in_r.shape(2);
    py::array_t<float> out_real({coils, height, width});
    py::array_t<float> out_imag({coils, height, width});
    auto dst_r = out_real.mutable_unchecked<3>();
    auto dst_i = out_imag.mutable_unchecked<3>();
    for (ssize_t coil = 0; coil < coils; ++coil) {
        for (ssize_t y = 0; y < height; ++y) {
            const float weight = weights(y);
            for (ssize_t x = 0; x < width; ++x) {
                dst_r(coil, y, x) = in_r(coil, y, x) * weight;
                dst_i(coil, y, x) = in_i(coil, y, x) * weight;
            }
        }
    }
    return py::make_tuple(out_real, out_imag);
}

PYBIND11_MODULE(sense_kernel, module) {
    module.doc() = "Coil combination and phase-encode mask for multi-coil MRI.";
    module.def("image_to_coils", &image_to_coils, "Apply coil sensitivities to one image.");
    module.def("coils_to_image", &coils_to_image, "Adjoint coil combination.");
    module.def("apply_phase_mask", &apply_phase_mask, "Keep sampled phase-encode lines.");
}