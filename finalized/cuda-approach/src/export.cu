#include "../include/export.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <stddef.h>

/**
 * @brief Export the current device grid to a CSV file.
 *
 * Writes the grid data to a CSV file for a given time step, copying from device to host as needed.
 */
void export_to_csv_cuda(const char *folder, const double *d_u, int Nx, int Ny, int timestep) {
    if (!folder || !d_u || Nx < 1 || Ny < 1) return;
    char filename[256];
    snprintf(filename, sizeof(filename), "%s/solution_t%04d.csv", folder, timestep);
    double *h_u = (double*)malloc(Nx * Ny * sizeof(double));
    if (!h_u) {
        fprintf(stderr, "Host memory allocation failed for export\n");
        return;
    }
    cudaError_t err = cudaMemcpy(h_u, d_u, Nx * Ny * sizeof(double), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA memcpy failed: %s\n", cudaGetErrorString(err));
        free(h_u);
        return;
    }
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Error opening file %s for writing\n", filename);
        free(h_u);
        return;
    }
    for (int j = 0; j < Ny; j++) {
        for (int i = 0; i < Nx; i++) {
            fprintf(fp, "%.5f", h_u[j * Nx + i]);
            if (i < Nx - 1) fprintf(fp, ",");
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
    free(h_u);
}