#include "../include/export.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

void export_to_csv_cuda(const char *folder, const double *d_u, int Nx, int Ny, int timestep) {
    char filename[256];
    snprintf(filename, sizeof(filename), "%s/solution_t%04d.csv", folder, timestep);

    // Allocate host memory
    double *h_u = (double*)malloc(Nx * Ny * sizeof(double));
    if (!h_u) {
        fprintf(stderr, "Host memory allocation failed for export\n");
        return;
    }

    // Copy data from device to host
    cudaError_t err = cudaMemcpy(h_u, d_u, Nx * Ny * sizeof(double), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA memcpy failed: %s\n", cudaGetErrorString(err));
        free(h_u);
        return;
    }

    FILE *fp = fopen(filename, "w");
    if (!fp) {
        perror("Error opening file for export");
        free(h_u);
        exit(EXIT_FAILURE);
    }

    for (int j = 0; j < Ny; j++) {
        for (int i = 0; i < Nx; i++) {
            fprintf(fp, "%.5f", h_u[j * Nx + i]);
            if (i < Nx - 1)
                fprintf(fp, ",");
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
    free(h_u);
}