#ifndef EXPORT_H
#define EXPORT_H

#include <stdio.h>

static inline void export_solution(double* u, int Nx, int Ny, int t_step) {
    char filename[128];
    sprintf(filename, "output/solution_t%03d.csv", t_step);

    FILE* fp = fopen(filename, "w");
    if (!fp) {
        perror("Failed to open export file");
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

#endif
