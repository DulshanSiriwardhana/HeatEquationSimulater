#include "../include/boundary.h"
#include <omp.h>

void apply_boundary_conditions(float *u, int nx, int ny) {
    // Parallelize boundary application using OpenMP
    #pragma omp parallel for
    for (int i = 0; i < nx; ++i) {
        u[i] = 0.0f; // Top boundary
        u[(ny-1)*nx + i] = 0.0f; // Bottom boundary
    }
    #pragma omp parallel for
    for (int j = 0; j < ny; ++j) {
        u[j*nx] = 0.0f; // Left boundary
        u[j*nx + (nx-1)] = 0.0f; // Right boundary
    }
} 