#include "boundary.h"

void apply_boundary_conditions(double *u, int Nx, int Ny, double boundary_temp) {
    for (int j = 0; j < Ny; j++) {
        u[j * Nx] = u[j * Nx + 1];
        u[j * Nx + (Nx - 1)] = u[j * Nx + (Nx - 2)];
    }

    for (int i = 0; i < Nx; i++) {
        u[i] = u[Nx + i];
        u[(Ny - 1) * Nx + i] = u[(Ny - 2) * Nx + i];
    }
}
