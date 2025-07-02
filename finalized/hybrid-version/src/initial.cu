#include "../include/initial.h"
#include <omp.h>

void set_initial_conditions(float *u, int nx, int ny) {
    #pragma omp parallel for
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            u[j*nx + i] = 0.0f;
        }
    }
    // Example: set a hot spot in the center
    int cx = nx / 2;
    int cy = ny / 2;
    u[cy*nx + cx] = 100.0f;
} 