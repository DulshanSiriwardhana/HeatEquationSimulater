#ifndef SOLVER_H
#define SOLVER_H

void advance_time_step_cuda(double *d_u, double *d_u_new, int Nx, int Ny,
                            double dx, double dy, double dt, double alpha);

#endif