#include "../include/boundary.h"
#include <stddef.h>
#include <omp.h>

/**
 * @brief Apply boundary conditions to the grid (OpenMP parallelized).
 *
 * Sets the boundary values of the grid to zero (Dirichlet) on all sides.
 *
 * @param u Pointer to the grid array (size nx*ny)
 * @param nx Number of grid points in x-direction
 * @param ny Number of grid points in y-direction
 */
void apply_boundary_conditions(float *u, int nx, int ny) {
    if (!u || nx < 2 || ny < 2) return;
    // Top and bottom boundaries
    #pragma omp parallel for
    for (int i = 0; i < nx; ++i) {
        u[i] = 0.0f; // Top boundary
        u[(ny-1)*nx + i] = 0.0f; // Bottom boundary
    }
    // Left and right boundaries
    #pragma omp parallel for
    for (int j = 0; j < ny; ++j) {
        u[j*nx] = 0.0f; // Left boundary
        u[j*nx + (nx-1)] = 0.0f; // Right boundary
    }
} 