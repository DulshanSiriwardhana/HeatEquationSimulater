#ifndef SOLVER_H
#define SOLVER_H

/**
 * @brief Advance the solution by one time step using the finite difference method (CUDA).
 *
 * Computes the next time step for the heat equation on a 2D grid on the device.
 *
 * @param d_u Pointer to the current device grid (size Nx*Ny)
 * @param d_u_new Pointer to the new device grid (size Nx*Ny)
 * @param Nx Number of grid points in x-direction
 * @param Ny Number of grid points in y-direction
 * @param dx Grid spacing in x-direction
 * @param dy Grid spacing in y-direction
 * @param dt Time step size
 * @param alpha Diffusion coefficient
 */
void advance_time_step_cuda(double *d_u, double *d_u_new, int Nx, int Ny,
                            double dx, double dy, double dt, double alpha);

#endif