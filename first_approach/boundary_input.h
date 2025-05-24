#ifndef BOUNDARY_INPUT_H
#define BOUNDARY_INPUT_H

static inline void apply_boundary(double* boundary, int Nx, int Ny, double T, int x_start, int x_end, int y_pos) {
    // Dummy example: set temperature T on a horizontal line y = y_pos between x_start and x_end
    for (int i = x_start; i < x_end; ++i) {
        int idx = y_pos * Nx + i;
        if (idx < Nx*Ny) boundary[idx] = T;
    }
}

#endif
