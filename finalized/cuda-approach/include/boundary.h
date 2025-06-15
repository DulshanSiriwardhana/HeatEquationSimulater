#ifndef BOUNDARY_H
#define BOUNDARY_H

__global__ void apply_boundary_conditions(double *u, int Nx, int Ny, double boundary_temp);

#endif
