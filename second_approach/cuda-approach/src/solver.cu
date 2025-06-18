#include "solver.h"

__global__ void advance_time_step(const double *u, double *u_new, int Nx, int Ny,
                                  double dx, double dy, double dt, double alpha) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= 1 && i < Nx - 1 && j >= 1 && j < Ny - 1) {
        int idx = j * Nx + i;

        double u_center = u[idx];
        double u_left = u[j * Nx + (i - 1)];
        double u_right = u[j * Nx + (i + 1)];
        double u_up = u[(j - 1) * Nx + i];
        double u_down = u[(j + 1) * Nx + i];

        double rdx2 = alpha * dt / (dx * dx);
        double rdy2 = alpha * dt / (dy * dy);

        u_new[idx] = u_center + rdx2 * (u_left - 2 * u_center + u_right)
                                 + rdy2 * (u_up - 2 * u_center + u_down);
    }
}
