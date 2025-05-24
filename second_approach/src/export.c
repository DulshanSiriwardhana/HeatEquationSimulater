#include "export.h"
#include <stdio.h>
#include <stdlib.h>

void export_to_csv(const char *folder, const double *u, int Nx, int Ny, int timestep) {
    char filename[256];
    snprintf(filename, sizeof(filename), "%s/solution_t%04d.csv", folder, timestep);

    FILE *fp = fopen(filename, "w");
    if (!fp) {
        perror("Error opening file for export");
        exit(EXIT_FAILURE);
    }

    for (int j = 0; j < Ny; j++) {
        for (int i = 0; i < Nx; i++) {
            fprintf(fp, "%.5f", u[j * Nx + i]);
            if (i < Nx - 1)
                fprintf(fp, ",");
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
}
