#include "../include/initial.h"
#include <stddef.h>
#include <omp.h>

/**
 * @brief Set the initial temperature distribution on the grid (OpenMP parallelized).
 *
 * Initializes the grid with a hot spot in the center.
 *
 * @param u Pointer to the grid array (size nx*ny)
 * @param nx Number of grid points in x-direction
 * @param ny Number of grid points in y-direction
 */
void set_initial_conditions(float *u, int nx, int ny) {
    if (!u || nx < 1 || ny < 1) return;
    #pragma omp parallel for
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            u[j*nx + i] = 0.0f;
        }
    }
    // Set a hot spot in the center
    int cx = nx / 2;
    int cy = ny / 2;
    u[cy*nx + cx] = 100.0f;
} 