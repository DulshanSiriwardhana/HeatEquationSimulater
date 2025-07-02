#include "boundary.h"
#include <stddef.h>
#include <omp.h>

/**
 * @brief Apply boundary conditions to the grid (OpenMP parallelized).
 *
 * This implementation copies the values from the adjacent interior cells to the boundaries.
 *
 * @param u Pointer to the grid array (size Nx*Ny)
 * @param Nx Number of grid points in x-direction
 * @param Ny Number of grid points in y-direction
 * @param boundary_temp Value to set at the boundaries (currently unused)
 */
void apply_boundary_conditions(double *u, int Nx, int Ny, double boundary_temp) {
    if (!u || Nx < 2 || Ny < 2) return;
    // Left and right boundaries
    #pragma omp parallel for
    for (int j = 0; j < Ny; j++) {
        u[j * Nx] = u[j * Nx + 1];
        u[j * Nx + (Nx - 1)] = u[j * Nx + (Nx - 2)];
    }
    // Top and bottom boundaries
    #pragma omp parallel for
    for (int i = 0; i < Nx; i++) {
        u[i] = u[Nx + i];
        u[(Ny - 1) * Nx + i] = u[(Ny - 2) * Nx + i];
    }
}
