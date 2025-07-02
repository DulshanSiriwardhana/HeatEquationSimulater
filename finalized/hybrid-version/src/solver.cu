#include "../include/solver.h"
#include <cuda_runtime.h>
#include <omp.h>
#include <stdio.h>

__global__ void heat_update(float *u, float *u_new, int nx, int ny, float alpha) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i > 0 && i < nx-1 && j > 0 && j < ny-1) {
        int idx = j*nx + i;
        u_new[idx] = u[idx] + alpha * (
            u[(j+1)*nx + i] + u[(j-1)*nx + i] +
            u[j*nx + (i+1)] + u[j*nx + (i-1)] - 4*u[idx]
        );
    }
}
e
void solve_heat_equation(float *u, int nx, int ny, int timesteps, float alpha) {
    float *d_u, *d_u_new;
    size_t size = nx * ny * sizeof(float);
    cudaMalloc(&d_u, size);
    cudaMalloc(&d_u_new, size);
    cudaMemcpy(d_u, u, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((nx+15)/16, (ny+15)/16);

    for (int t = 0; t < timesteps; ++t) {
        heat_update<<<numBlocks, threadsPerBlock>>>(d_u, d_u_new, nx, ny, alpha);
        cudaDeviceSynchronize();
        float *tmp = d_u;
        d_u = d_u_new;
        d_u_new = tmp;
    }
    cudaMemcpy(u, d_u, size, cudaMemcpyDeviceToHost);
    cudaFree(d_u);
    cudaFree(d_u_new);
} 