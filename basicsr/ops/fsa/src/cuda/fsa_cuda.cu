//
// Created by 胡振宇 on 2024-02-04.
//

#include "fsa_cuda.cuh"
#include "fsa_im2col_cuda.cuh"
#include <vector>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

at::Tensor fsa_vertical_cuda_forward(const at::Tensor &input, const at::Tensor &kernel, const int kernel_size,
                                       const at::Tensor &kernel_map, const int group, const int group_channel) {
    // input: (N, C, H, W)
    // kernel: (N, kernel_size*group, H, W)
    // kernel_map: (kernel_size, 2)

    AT_ASSERTM(input.is_contiguous(), "input tensor has to be contiguous");
    AT_ASSERTM(kernel.is_contiguous(), "kernel tensor has to be contiguous");
    AT_ASSERT(kernel_map.is_contiguous(), "kernel_map tensor has to be contiguous");
    AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor");
    AT_ASSERTM(kernel.type().is_cuda(), "kernel must be a CUDA tensor");
    AT_ASSERTM(kernel_map.type().is_cuda(), "kernel_map must be a CUDA tensor");

    AT_ASSERT(kernel.size(1)==kernel_size*group, "kernel size and group size mismatch");
    AT_ASSERT(input.size(1)==group*group_channel, "input channel and group size mismatch");

    const int batch = input.size(0);
    const int channel = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);

    auto output = at::zeros_like(input);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.type(), "fsa_vertical_forward_cuda", ([&] {
        fsa_vertical_im2col_cuda(
                at::cuda::getCurrentCUDAStream(),
                input.data_ptr<scalar_t>(),
                kernel.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(), batch, channel,
                height, width, kernel_size, kernel_map.data_ptr<int>(), group, group_channel);
        })
    );

    return output;
}


std::vector<at::Tensor>
fsa_vertical_cuda_backward(const at::Tensor &grad_output, const at::Tensor &input, const at::Tensor &kernel,
                             const int kernel_size, const at::Tensor &kernel_map, const int group,
                             const int group_channel) {
    AT_ASSERTM(grad_output.is_contiguous(), "grad_output tensor has to be contiguous");
    AT_ASSERTM(input.is_contiguous(), "input tensor has to be contiguous");
    AT_ASSERTM(kernel.is_contiguous(), "kernel tensor has to be contiguous");
    AT_ASSERTM(kernel_map.is_contiguous(), "kernel_map tensor has to be contiguous");
    AT_ASSERTM(grad_output.type().is_cuda(), "grad_output must be a CUDA tensor");
    AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor");
    AT_ASSERTM(kernel.type().is_cuda(), "kernel must be a CUDA tensor");

    AT_ASSERT(kernel.size(1)==kernel_size*group, "kernel size and group size mismatch");
    AT_ASSERT(input.size(1)==group*group_channel, "input channel and group size mismatch");

    const int batch = input.size(0);
    const int channel = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);

    auto grad_input = at::zeros_like(input);
    auto grad_kernel = at::zeros_like(kernel);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.type(), "fsa_vertical_backward_cuda", ([&] {
        fsa_vertical_col2im_cuda(
                at::cuda::getCurrentCUDAStream(),
                grad_output.data_ptr<scalar_t>(),
                input.data_ptr<scalar_t>(),
                kernel.data_ptr<scalar_t>(),
                grad_input.data_ptr<scalar_t>(),
                grad_kernel.data_ptr<scalar_t>(),
                batch, channel, height, width, kernel_size, kernel_map.data_ptr<int>(), group, group_channel);
        })
    );

    return {grad_input, grad_kernel};
}

at::Tensor fsa_horizontal_cuda_forward(const at::Tensor &input, const at::Tensor &kernel, const int kernel_size,
                                         const at::Tensor &kernel_map, const int group, const int group_channel) {
    // input: (N, C, H, W)
    // kernel: (N, kernel_size*group, H, W)
    // kernel_map: (kernel_size, 2)

    AT_ASSERTM(input.is_contiguous(), "input tensor has to be contiguous");
    AT_ASSERTM(kernel.is_contiguous(), "kernel tensor has to be contiguous");
    AT_ASSERT(kernel_map.is_contiguous(), "kernel_map tensor has to be contiguous");
    AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor");
    AT_ASSERTM(kernel.type().is_cuda(), "kernel must be a CUDA tensor");
    AT_ASSERTM(kernel_map.type().is_cuda(), "kernel_map must be a CUDA tensor");

    AT_ASSERT(kernel.size(1)==kernel_size*group, "kernel size and group size mismatch");
    AT_ASSERT(input.size(1)==group*group_channel, "input channel and group size mismatch");

    const int batch = input.size(0);
    const int channel = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);

    auto output = at::zeros_like(input);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.type(), "fsa_horizontal_forward_cuda", ([&] {
        fsa_horizontal_im2col_cuda(
                at::cuda::getCurrentCUDAStream(),
                input.data_ptr<scalar_t>(),
                kernel.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(), batch, channel,
                height, width, kernel_size, kernel_map.data_ptr<int>(), group, group_channel);
        })
    );

    return output;
}

std::vector<at::Tensor>
fsa_horizontal_cuda_backward(const at::Tensor &grad_output, const at::Tensor &input, const at::Tensor &kernel,
                               const int kernel_size, const at::Tensor &kernel_map, const int group,
                               const int group_channel) {
    AT_ASSERTM(grad_output.is_contiguous(), "grad_output tensor has to be contiguous");
    AT_ASSERTM(input.is_contiguous(), "input tensor has to be contiguous");
    AT_ASSERTM(kernel.is_contiguous(), "kernel tensor has to be contiguous");
    AT_ASSERTM(kernel_map.is_contiguous(), "kernel_map tensor has to be contiguous");
    AT_ASSERTM(grad_output.type().is_cuda(), "grad_output must be a CUDA tensor");
    AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor");
    AT_ASSERTM(kernel.type().is_cuda(), "kernel must be a CUDA tensor");

    AT_ASSERT(kernel.size(1)==kernel_size*group, "kernel size and group size mismatch");
    AT_ASSERT(input.size(1)==group*group_channel, "input channel and group size mismatch");

    const int batch = input.size(0);
    const int channel = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);

    auto grad_input = at::zeros_like(input);
    auto grad_kernel = at::zeros_like(kernel);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.type(), "fsa_horizontal_backward_cuda", ([&] {
        fsa_horizontal_col2im_cuda(
                at::cuda::getCurrentCUDAStream(),
                grad_output.data_ptr<scalar_t>(),
                input.data_ptr<scalar_t>(),
                kernel.data_ptr<scalar_t>(),
                grad_input.data_ptr<scalar_t>(),
                grad_kernel.data_ptr<scalar_t>(),
                batch, channel, height, width, kernel_size, kernel_map.data_ptr<int>(), group, group_channel);
        })
    );

    return {grad_input, grad_kernel};
}