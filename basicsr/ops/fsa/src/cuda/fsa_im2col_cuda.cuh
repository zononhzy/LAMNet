// Created by 胡振宇 on 2024-02-04.
//

#ifndef OPS_FSA_IM2COL_CUDA_CUH
#define OPS_FSA_IM2COL_CUDA_CUH

#include <torch/extension.h>
#include <THC/THCAtomics.cuh>

#define CUDA_KERNEL_LOOP(i, n)                                                 \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);               \
         i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 512;

inline int GET_BLOCKS(const int N, const int num_threads) {
    return (N + num_threads - 1) / num_threads;
}

template<typename scalar_t>
__global__ void
fsa_vertical_im2col_kernel(const int n, const scalar_t *data_im, const scalar_t *kernel, scalar_t *data_col,
                             const int channel, const int height, const int width, const int kernel_size,
                             const int *kernel_map, const int group, const int group_channel) {
    CUDA_KERNEL_LOOP(index, n) {
        int _temp = index;
        const int w = _temp % width;
        _temp /= width;
        const int h = _temp % height;
        _temp /= height;
        const int c = _temp % channel;
        const int b = _temp / channel;
        const int g = c / group_channel;

        const int input_size = height * width;
        const scalar_t *data_im_ptr = data_im + index;
        const scalar_t *kernel_ptr = kernel + ((b * group + g) * kernel_size * height + h) * width + w;
        scalar_t *data_col_ptr = data_col + index;

        scalar_t sum = 0;
        for (int kh = 0; kh < kernel_size; kh++) {
            const int start = kernel_map[kh * 2];
            const int len = kernel_map[kh * 2 + 1];

            const scalar_t weight = kernel_ptr[kh * input_size];

            scalar_t val = 0;
            const int direction = start >= 0 ? 1 : -1;
            for (int i = 0; i < len; i++) {
                const int offset = start + i * direction;
                const int h_im = h + offset;
                if (h_im >= 0 && h_im < height) {
                    val += data_im_ptr[offset * width];
                }
            }
            sum += val * weight / len;
        }
        *data_col_ptr = sum;
    }
}

template<typename scalar_t>
__global__ void
fsa_vertical_col2im_kernel(const int n, const scalar_t *grad_output, const scalar_t *data_im, const scalar_t *kernel,
                             scalar_t *grad_input, scalar_t *grad_kernel, const int channel, const int height,
                             const int width,
                             const int kernel_size, const int *kernel_map, const int group,
                             const int group_channel) {
    CUDA_KERNEL_LOOP(index, n) {
        int _temp = index;
        const int w = _temp % width;
        _temp /= width;
        const int h = _temp % height;
        _temp /= height;
        const int c = _temp % channel;
        const int b = _temp / channel;
        const int g = c / group_channel;

        const int input_size = height * width;
        scalar_t *grad_input_ptr = grad_input + index;
        scalar_t *grad_kernel_ptr = grad_kernel + ((b * group + g) * kernel_size * height + h) * width + w;
        const scalar_t top_grad = grad_output[index];
        const scalar_t *data_im_ptr = data_im + index;
        const scalar_t *kernel_ptr = kernel + ((b * group + g) * kernel_size * height + h) * width + w;

        for (int kh = 0; kh < kernel_size; kh++) {
            const int start = kernel_map[kh * 2];
            const int len = kernel_map[kh * 2 + 1];

            const scalar_t weight = kernel_ptr[kh * input_size];

            scalar_t val = 0;
            const int direction = start >= 0 ? 1 : -1;
            for (int i = 0; i < len; i++) {
                const int offset = start + i * direction;
                const int h_im = h + offset;
                if (h_im >= 0 && h_im < height) {
                    val += data_im_ptr[offset * width];
                    atomicAdd(grad_input_ptr + offset * width, weight * top_grad / len);
                }
            }
            atomicAdd(grad_kernel_ptr + kh * input_size, val * top_grad / len);
        }
    }
}


template<typename scalar_t>
void fsa_vertical_im2col_cuda(cudaStream_t stream, const scalar_t *input, const scalar_t *kernel, scalar_t *output,
                                const int batch, const int channel, const int height, const int width,
                                const int kernel_size, const int *kernel_map, const int group,
                                const int group_channel) {
    const int num_kernels = height * width * channel * batch;

    fsa_vertical_im2col_kernel<scalar_t><<<GET_BLOCKS(num_kernels, CUDA_NUM_THREADS), CUDA_NUM_THREADS, 0, stream>>>(
            num_kernels, input, kernel, output, channel, height, width, kernel_size, kernel_map, group, group_channel);

    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("error in fsa_vertical_im2col_cuda: %s\n", cudaGetErrorString(err));
    }
}

template<typename scalar_t>
void fsa_vertical_col2im_cuda(cudaStream_t stream, const scalar_t *grad_output, const scalar_t *input,
                                const scalar_t *kernel,
                                scalar_t *grad_input, scalar_t *grad_kernel, const int batch, const int channel,
                                const int height, const int width,
                                const int kernel_size, const int *kernel_map, const int group,
                                const int group_channel) {
    const int num_kernels = height * width * channel * batch;

    fsa_vertical_col2im_kernel<scalar_t><<<GET_BLOCKS(num_kernels, CUDA_NUM_THREADS), CUDA_NUM_THREADS, 0, stream>>>(
            num_kernels, grad_output, input, kernel, grad_input, grad_kernel, channel, height, width, kernel_size,
            kernel_map, group, group_channel);

    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("error in fsa_vertical_col2im_cuda: %s\n", cudaGetErrorString(err));
    }
}

template<typename scalar_t>
__global__ void fsa_horizontal_im2col_kernel(const int n, const scalar_t *data_im, const scalar_t *kernel,
                                               scalar_t *data_col, const int channel, const int height, const int width,
                                               const int kernel_size, const int *kernel_map, const int group,
                                               const int group_channel) {
    CUDA_KERNEL_LOOP(index, n) {
        int _temp = index;
        const int w = _temp % width;
        _temp /= width;
        const int h = _temp % height;
        _temp /= height;
        const int c = _temp % channel;
        const int b = _temp / channel;
        const int g = c / group_channel;

        const int input_size = height * width;
        const scalar_t *data_im_ptr = data_im + index;
        const scalar_t *kernel_ptr = kernel + ((b * group + g) * kernel_size * height + h) * width + w;
        scalar_t *data_col_ptr = data_col + index;

        scalar_t sum = 0;
        for (int kw = 0; kw < kernel_size; kw++) {
            const int start = kernel_map[kw * 2];
            const int len = kernel_map[kw * 2 + 1];

            const scalar_t weight = kernel_ptr[kw * input_size];

            scalar_t val = 0;
            const int direction = start >= 0 ? 1 : -1;
            for (int i = 0; i < len; i++) {
                const int offset = start + i * direction;
                const int w_im = w + offset;
                if (w_im >= 0 && w_im < width) {
                    val += data_im_ptr[offset];
                }
            }
            sum += val * weight / len;
        }
        *data_col_ptr = sum;
    }
}

template<typename scalar_t>
__global__ void fsa_horizontal_col2im_kernel(const int n, const scalar_t *grad_output, const scalar_t *data_im,
                                               const scalar_t *kernel, scalar_t *grad_input, scalar_t *grad_kernel,
                                               const int channel, const int height, const int width,
                                               const int kernel_size,
                                               const int *kernel_map, const int group, const int group_channel) {
    CUDA_KERNEL_LOOP(index, n) {
        int _temp = index;
        const int w = _temp % width;
        _temp /= width;
        const int h = _temp % height;
        _temp /= height;
        const int c = _temp % channel;
        const int b = _temp / channel;
        const int g = c / group_channel;

        const int input_size = height * width;
        scalar_t *grad_input_ptr = grad_input + index;
        scalar_t *grad_kernel_ptr = grad_kernel + ((b * group + g) * kernel_size * height + h) * width + w;
        const scalar_t top_grad = grad_output[index];
        const scalar_t *data_im_ptr = data_im + index;
        const scalar_t *kernel_ptr = kernel + ((b * group + g) * kernel_size * height + h) * width + w;

        for (int kw = 0; kw < kernel_size; kw++) {
            const int start = kernel_map[kw * 2];
            const int len = kernel_map[kw * 2 + 1];

            const scalar_t weight = kernel_ptr[kw * input_size];

            scalar_t val = 0;
            const int direction = start >= 0 ? 1 : -1;
            for (int i = 0; i < len; i++) {
                const int offset = start + i * direction;
                const int w_im = w + offset;
                if (w_im >= 0 && w_im < width) {
                    val += data_im_ptr[offset];
                    atomicAdd(grad_input_ptr + offset, weight * top_grad / len);
                }
            }
            atomicAdd(grad_kernel_ptr + kw * input_size, val * top_grad / len);
        }
    }
}

template<typename scalar_t>
void fsa_horizontal_im2col_cuda(cudaStream_t stream, const scalar_t *input, const scalar_t *kernel, scalar_t *output,
                                  const int batch, const int channel, const int height, const int width,
                                  const int kernel_size, const int *kernel_map, const int group,
                                  const int group_channel) {
    const int num_kernels = height * width * channel * batch;

    fsa_horizontal_im2col_kernel<scalar_t><<<GET_BLOCKS(num_kernels,
                                                          CUDA_NUM_THREADS), CUDA_NUM_THREADS, 0, stream>>>(
            num_kernels, input, kernel, output, channel, height, width, kernel_size, kernel_map, group, group_channel);

    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("error in fsa_horizontal_im2col_cuda: %s\n", cudaGetErrorString(err));
    }
}

template<typename scalar_t>
void fsa_horizontal_col2im_cuda(cudaStream_t stream, const scalar_t *grad_output, const scalar_t *input,
                                  const scalar_t *kernel,
                                  scalar_t *grad_input, scalar_t *grad_kernel, const int batch, const int channel,
                                  const int height, const int width,
                                  const int kernel_size, const int *kernel_map, const int group,
                                  const int group_channel) {
    const int num_kernels = height * width * channel * batch;

    fsa_horizontal_col2im_kernel<scalar_t><<<GET_BLOCKS(num_kernels,
                                                          CUDA_NUM_THREADS), CUDA_NUM_THREADS, 0, stream>>>(
            num_kernels, grad_output, input, kernel, grad_input, grad_kernel, channel, height, width, kernel_size,
            kernel_map, group, group_channel);

    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("error in fsa_horizontal_col2im_cuda: %s\n", cudaGetErrorString(err));
    }
}

#endif //OPS_FSA_IM2COL_CUDA_CUH
