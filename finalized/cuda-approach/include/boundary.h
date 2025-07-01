#ifndef BOUNDARY_H
#define BOUNDARY_H

void apply_boundary_conditions_cuda(double *d_u, int Nx, int Ny, double boundary_temp);

#endif