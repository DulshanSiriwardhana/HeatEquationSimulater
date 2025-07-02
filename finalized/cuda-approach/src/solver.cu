#include "../include/solver.h"
#include <cuda_runtime.h>
#include <stddef.h>

/**
 * @brief CUDA kernel to advance the solution by one time step using the finite difference method.
 */
__global__ void advance_time_step_kernel(const double *u, double *u_new, int Nx, int Ny,
                                         double rdx2, double rdy2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Only process interior points
    if (i > 0 && i < Nx-1 && j > 0 && j < Ny-1) {
        int idx = j * Nx + i;
        
        double u_center = u[idx];
        double u_left = u[j * Nx + (i - 1)];
        double u_right = u[j * Nx + (i + 1)];
        double u_up = u[(j - 1) * Nx + i];
        double u_down = u[(j + 1) * Nx + i];
        
        u_new[idx] = u_center + rdx2 * (u_left - 2 * u_center + u_right)
                              + rdy2 * (u_up - 2 * u_center + u_down);
    }
}

/**
 * @brief CUDA kernel to copy boundary values.
 */
__global__ void copy_boundaries_kernel(const double *u, double *u_new, int Nx, int Ny) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Copy top and bottom boundaries
    if (idx < Nx) {
        u_new[idx] = u[idx];                              // Top boundary
        u_new[(Ny - 1) * Nx + idx] = u[(Ny - 1) * Nx + idx]; // Bottom boundary
    }
    
    // Copy left and right boundaries
    if (idx < Ny) {
        u_new[idx * Nx] = u[idx * Nx];                    // Left boundary
        u_new[idx * Nx + (Nx - 1)] = u[idx * Nx + (Nx - 1)]; // Right boundary
    }
}

/**
 * @brief Advance the solution by one time step using the finite difference method (CUDA).
 */
void advance_time_step_cuda(double *d_u, double *d_u_new, int Nx, int Ny,
                           double dx, double dy, double dt, double alpha) {
    if (!d_u || !d_u_new || Nx < 2 || Ny < 2) return;
    double rdx2 = alpha * dt / (dx * dx);
    double rdy2 = alpha * dt / (dy * dy);
    
    // Launch kernel for interior points
    dim3 threads_per_block(16, 16);
    dim3 num_blocks((Nx + threads_per_block.x - 1) / threads_per_block.x,
                    (Ny + threads_per_block.y - 1) / threads_per_block.y);
    
    advance_time_step_kernel<<<num_blocks, threads_per_block>>>(d_u, d_u_new, Nx, Ny, rdx2, rdy2);
    
    // Copy boundary values
    int max_dim = (Nx > Ny) ? Nx : Ny;
    int threads_per_block_1d = 256;
    int blocks_1d = (max_dim + threads_per_block_1d - 1) / threads_per_block_1d;
    
    copy_boundaries_kernel<<<blocks_1d, threads_per_block_1d>>>(d_u, d_u_new, Nx, Ny);
    
    cudaDeviceSynchronize();
}