#ifndef INITIAL_H
#define INITIAL_H

/**
 * @brief Set the initial temperature distribution on the grid.
 *
 * Initializes the grid with the desired initial condition (e.g., sum of Gaussians).
 *
 * @param u Pointer to the grid array (size Nx*Ny)
 * @param Nx Number of grid points in x-direction
 * @param Ny Number of grid points in y-direction
 */
void set_initial_conditions(double *u, int Nx, int Ny);

#endif
