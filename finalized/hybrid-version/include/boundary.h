#ifndef BOUNDARY_H
#define BOUNDARY_H

/**
 * @brief Apply boundary conditions to the grid (OpenMP parallelized).
 *
 * Sets the boundary values of the grid according to the chosen scheme.
 *
 * @param u Pointer to the grid array (size nx*ny)
 * @param nx Number of grid points in x-direction
 * @param ny Number of grid points in y-direction
 */
void apply_boundary_conditions(float *u, int nx, int ny);

#endif // BOUNDARY_H 