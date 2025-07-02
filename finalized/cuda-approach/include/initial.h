#ifndef INITIAL_H
#define INITIAL_H

/**
 * @brief Set the initial temperature distribution on the grid (CUDA).
 *
 * Initializes the device grid with the desired initial condition (e.g., sum of Gaussians).
 *
 * @param d_u Pointer to the device grid array (size Nx*Ny)
 * @param Nx Number of grid points in x-direction
 * @param Ny Number of grid points in y-direction
 */
void set_initial_conditions_cuda(double *d_u, int Nx, int Ny);

#endif