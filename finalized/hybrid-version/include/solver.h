#ifndef SOLVER_H
#define SOLVER_H

/**
 * @brief Solve the heat equation using a hybrid CUDA+OpenMP approach.
 *
 * Advances the solution for a given number of timesteps on the device, using CUDA for computation.
 *
 * @param u Pointer to the grid array (size nx*ny)
 * @param nx Number of grid points in x-direction
 * @param ny Number of grid points in y-direction
 * @param timesteps Number of time steps to simulate
 * @param alpha Diffusion coefficient
 */
void solve_heat_equation(float *u, int nx, int ny, int timesteps, float alpha);

#endif // SOLVER_H 