#ifndef EXPORT_H
#define EXPORT_H

/**
 * @brief Export the current device grid to a CSV file.
 *
 * Writes the grid data to a CSV file for a given time step, copying from device to host as needed.
 *
 * @param folder Output directory
 * @param d_u Pointer to the device grid array (size Nx*Ny)
 * @param Nx Number of grid points in x-direction
 * @param Ny Number of grid points in y-direction
 * @param timestep Current time step (used in filename)
 */
void export_to_csv_cuda(const char *folder, const double *d_u, int Nx, int Ny, int timestep);

#endif