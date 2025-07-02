#ifndef BOUNDARY_H
#define BOUNDARY_H

/**
 * @brief Apply boundary conditions to the grid on the device (CUDA).
 *
 * Sets the boundary values of the grid according to the chosen scheme.
 *
 * @param d_u Pointer to the device grid array (size Nx*Ny)
 * @param Nx Number of grid points in x-direction
 * @param Ny Number of grid points in y-direction
 * @param boundary_temp Value to set at the boundaries (if used)
 */
void apply_boundary_conditions_cuda(double *d_u, int Nx, int Ny, double boundary_temp);

#endif