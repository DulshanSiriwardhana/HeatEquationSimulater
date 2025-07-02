#ifndef EXPORT_H
#define EXPORT_H

/**
 * @brief Export the current grid to a CSV file.
 *
 * Writes the grid data to a CSV file for a given time step.
 *
 * @param folder Output directory
 * @param u Pointer to the grid array (size Nx*Ny)
 * @param Nx Number of grid points in x-direction
 * @param Ny Number of grid points in y-direction
 * @param timestep Current time step (used in filename)
 */
void export_to_csv(const char *folder, const double *u, int Nx, int Ny, int timestep);

#endif
