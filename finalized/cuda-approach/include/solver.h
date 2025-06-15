#ifndef SOLVER_H
#define SOLVER_H

__global__ void advance_time_step(const double *u, double *u_new, int Nx, int Ny,
                                  double dx, double dy, double dt, double alpha);

#endif
