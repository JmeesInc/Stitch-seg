#include <torch/extension.h>

// Forward declarations
torch::Tensor fused_warp_perspective(
    torch::Tensor src,
    torch::Tensor M,
    int h_out, int w_out,
    int mode);

torch::Tensor fast_gradient_mask(
    torch::Tensor inv_mask,
    int radius,
    int downscale);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_warp_perspective", &fused_warp_perspective,
          "Fused warp perspective (CUDA)",
          py::arg("src"), py::arg("M"),
          py::arg("h_out"), py::arg("w_out"),
          py::arg("mode") = 1);
    m.def("fast_gradient_mask", &fast_gradient_mask,
          "Fast gradient mask via iterated dilation (CUDA)",
          py::arg("inv_mask"), py::arg("radius"),
          py::arg("downscale") = 4);
}
