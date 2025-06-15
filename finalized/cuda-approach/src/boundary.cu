#include "boundary.h"

__global__ void apply_boundary_conditions(double *u, int Nx, int Ny, double boundary_temp) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < Ny) {
        u[idx * Nx] = boundary_temp;
        u[idx * Nx + (Nx - 1)] = boundary_temp;
    }

    if (idx < Nx) {
        u[idx] = boundary_temp;
        u[(Ny - 1) * Nx + idx] = boundary_temp;
    }
}
