#include "export.h"
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>

/**
 * @brief Export the current grid to a CSV file.
 *
 * Writes the grid data to a CSV file for a given time step.
 */
void export_to_csv(const char *folder, const double *u, int Nx, int Ny, int timestep) {
    if (!folder || !u || Nx < 1 || Ny < 1) return;
    char filename[256];
    snprintf(filename, sizeof(filename), "%s/solution_t%04d.csv", folder, timestep);
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Error opening file %s for writing\n", filename);
        return;
    }
    for (int j = 0; j < Ny; j++) {
        for (int i = 0; i < Nx; i++) {
            fprintf(fp, "%.5f", u[j * Nx + i]);
            if (i < Nx - 1) fprintf(fp, ",");
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
}
