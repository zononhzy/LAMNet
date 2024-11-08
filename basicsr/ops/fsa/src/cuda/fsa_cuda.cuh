//
// Created by 胡振宇 on 2024-02-04.
//

#ifndef OPS_FSA_CUDA_CUH
#define OPS_FSA_CUDA_CUH

#include <torch/extension.h>

at::Tensor fsa_vertical_cuda_forward(const at::Tensor &input, const at::Tensor &kernel, const int kernel_size,
                                       const at::Tensor &kernel_map, const int group, const int group_channel);

std::vector<at::Tensor>
fsa_vertical_cuda_backward(const at::Tensor &grad_output, const at::Tensor &input, const at::Tensor &kernel,
                             const int kernel_size, const at::Tensor &kernel_map, const int group,
                             const int group_channel);

at::Tensor fsa_horizontal_cuda_forward(const at::Tensor &input, const at::Tensor &kernel, const int kernel_size,
                                         const at::Tensor &kernel_map, const int group, const int group_channel);

std::vector<at::Tensor>
fsa_horizontal_cuda_backward(const at::Tensor &grad_output, const at::Tensor &input, const at::Tensor &kernel,
                               const int kernel_size, const at::Tensor &kernel_map, const int group,
                               const int group_channel);
#endif //OPS_FSA_CUDA_CUH
