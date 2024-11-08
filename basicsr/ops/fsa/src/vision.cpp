#include "fsa_cuda.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fsa_vertical_forward", &fsa_vertical_cuda_forward, "fsa_vertical_forward");
    m.def("fsa_vertical_backward", &fsa_vertical_cuda_backward, "fsa_vertical_backward");
    m.def("fsa_horizontal_forward", &fsa_horizontal_cuda_forward, "fsa_horizontal_forward");
    m.def("fsa_horizontal_backward", &fsa_horizontal_cuda_backward, "fsa_horizontal_backward");
}