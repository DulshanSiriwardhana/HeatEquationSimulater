#include "boundary.h"

// Apply fixed temperature at edges (Dirichlet BC)
void apply_boundary_conditions(double *u, int Nx, int Ny, double boundary_temp) {
    // Left and right: copy adjacent interior points
    for (int j = 0; j < Ny; j++) {
        u[j * Nx] = u[j * Nx + 1];             // left boundary
        u[j * Nx + (Nx - 1)] = u[j * Nx + (Nx - 2)]; // right boundary
    }

    // Top and bottom rows
    for (int i = 0; i < Nx; i++) {
        u[i] = u[Nx + i];                     // top boundary
        u[(Ny - 1) * Nx + i] = u[(Ny - 2) * Nx + i];  // bottom boundary
    }
}
