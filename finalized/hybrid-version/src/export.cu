#include "../include/export.h"
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>

/**
 * @brief Export the current grid to a CSV file for a given time step.
 *
 * Writes the grid data to a CSV file for a given time step.
 */
void export_to_csv(const char *folder, const float *u, int nx, int ny, int timestep) {
    if (!folder || !u || nx < 1 || ny < 1) return;
    char filename[256];
    snprintf(filename, sizeof(filename), "%s/solution_t%04d.csv", folder, timestep);
    FILE *fp = fopen(filename, "w");
    if (!fp) return;
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            fprintf(fp, "%f", u[j*nx + i]);
            if (i < nx - 1) fprintf(fp, ",");
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
}