#include "solver.h"
#include <omp.h>

void advance_time_step(const double *u, double *u_new, int Nx, int Ny,
                       double dx, double dy, double dt, double alpha) {
    double rdx2 = alpha * dt / (dx * dx);
    double rdy2 = alpha * dt / (dy * dy);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int j = 1; j < Ny - 1; j++) {
        for (int i = 1; i < Nx - 1; i++) {
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

    #pragma omp parallel for
    for (int i = 0; i < Nx; i++) {
        u_new[i] = u[i];
        u_new[(Ny - 1) * Nx + i] = u[(Ny - 1) * Nx + i];
    }
    #pragma omp parallel for
    for (int j = 0; j < Ny; j++) {
        u_new[j * Nx] = u[j * Nx];
        u_new[j * Nx + (Nx - 1)] = u[j * Nx + (Nx - 1)];
    }
}
