#include "../include/boundary.h"
#include <cuda_runtime.h>
#include <stddef.h>

/**
 * @brief CUDA kernel to apply boundary conditions to the grid.
 *
 * This implementation copies the values from the adjacent interior cells to the boundaries.
 */
__global__ void apply_boundary_kernel(double *u, int Nx, int Ny) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Left and right boundaries
    if (idx < Ny) {
        int j = idx;
        u[j * Nx] = u[j * Nx + 1];                    // Left boundary
        u[j * Nx + (Nx - 1)] = u[j * Nx + (Nx - 2)]; // Right boundary
    }
    
    // Top and bottom boundaries
    if (idx < Nx) {
        int i = idx;
        u[i] = u[Nx + i];                             // Top boundary
        u[(Ny - 1) * Nx + i] = u[(Ny - 2) * Nx + i]; // Bottom boundary
    }
}

/**
 * @brief Apply boundary conditions to the grid on the device (CUDA).
 */
void apply_boundary_conditions_cuda(double *d_u, int Nx, int Ny, double boundary_temp) {
    if (!d_u || Nx < 2 || Ny < 2) return;
    int max_dim = (Nx > Ny) ? Nx : Ny;
    int threads_per_block = 256;
    int blocks = (max_dim + threads_per_block - 1) / threads_per_block;
    
    apply_boundary_kernel<<<blocks, threads_per_block>>>(d_u, Nx, Ny);
    cudaDeviceSynchronize();
}