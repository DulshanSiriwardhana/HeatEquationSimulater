#ifndef INITIAL_H
#define INITIAL_H

/**
 * @brief Set the initial temperature distribution on the grid (OpenMP parallelized).
 *
 * Initializes the grid with the desired initial condition (e.g., hot spot in the center).
 *
 * @param u Pointer to the grid array (size nx*ny)
 * @param nx Number of grid points in x-direction
 * @param ny Number of grid points in y-direction
 */
void set_initial_conditions(float *u, int nx, int ny);

#endif // INITIAL_H