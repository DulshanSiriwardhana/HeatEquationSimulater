#include "../include/initial.h"
#include <cuda_runtime.h>
#include <math.h>
#include <stddef.h>

/**
 * @brief Device Gaussian function for initial condition.
 */
__device__ double gaussian_device(double x, double y, double x0, double y0, double sigma) {
    double dx = x - x0;
    double dy = y - y0;
    return exp(-(dx*dx + dy*dy) / (2.0 * sigma * sigma));
}

/**
 * @brief CUDA kernel to set the initial temperature distribution on the grid.
 */
__global__ void set_initial_conditions_kernel(double *u, int Nx, int Ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < Nx && j < Ny) {
        double sigma = Nx / 10.0;
        double base_temp = 20.0;
        double temp = base_temp;
        temp += 80.0 * gaussian_device(i, j, Nx / 3.0, Ny / 3.0, sigma);
        temp += 100.0 * gaussian_device(i, j, Nx / 2.0, Ny / 2.0, sigma);
        temp += 70.0 * gaussian_device(i, j, 2.0 * Nx / 3.0, 2.0 * Ny / 3.0, sigma);
        u[j * Nx + i] = temp;
    }
}

/**
 * @brief Set the initial temperature distribution on the grid (CUDA).
 */
void set_initial_conditions_cuda(double *d_u, int Nx, int Ny) {
    if (!d_u || Nx < 1 || Ny < 1) return;
    dim3 threads_per_block(16, 16);
    dim3 num_blocks((Nx + threads_per_block.x - 1) / threads_per_block.x,
                    (Ny + threads_per_block.y - 1) / threads_per_block.y);
    set_initial_conditions_kernel<<<num_blocks, threads_per_block>>>(d_u, Nx, Ny);
    cudaDeviceSynchronize();
}