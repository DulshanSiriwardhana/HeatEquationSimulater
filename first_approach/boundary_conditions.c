#include "boundary_conditions.h"
#include "boundary_input.h"  // your real boundary apply function header

void set_boundary_conditions(double *boundary, int Nx, int Ny) {
    apply_boundary(boundary, Nx, Ny, 100, 0, 50, 50);
}
