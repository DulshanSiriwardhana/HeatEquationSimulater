#ifndef EXPORT_H
#define EXPORT_H

void export_to_csv_cuda(const char *folder, const double *d_u, int Nx, int Ny, int timestep);

#endif