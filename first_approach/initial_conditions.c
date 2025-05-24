#include "initial_conditions.h"

void set_initial_conditions(double *u, int Nx, int Ny) {
    int N = Nx * Ny;
    for (int i = 0; i < N; ++i)
        u[i] = 100.0;

    int cx = Nx / 2;
    int cy = Ny / 2;
    int radius = 5;
    for (int j = 0; j < Ny; ++j) {
        for (int i = 0; i < Nx; ++i) {
            int idx = j * Nx + i;
            int dx_ = i - cx;
            int dy_ = j - cy;
            if (dx_ * dx_ + dy_ * dy_ <= radius * radius) {
                u[idx] = 100.0;
            }
        }
    }
}
