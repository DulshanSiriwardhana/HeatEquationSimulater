#ifndef EXPORT_H
#define EXPORT_H

/**
 * @brief Export the current grid to a CSV file for a given time step.
 *
 * Writes the grid data to a CSV file for a given time step.
 *
 * @param folder Output directory
 * @param u Pointer to the grid array (size nx*ny)
 * @param nx Number of grid points in x-direction
 * @param ny Number of grid points in y-direction
 * @param timestep Current time step (used in filename)
 */
void export_to_csv(const char *folder, const float *u, int nx, int ny, int timestep);

#endif // EXPORT_H 