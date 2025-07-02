#include "../include/export.h"
#include <stdio.h>

void export_results(const float *u, int nx, int ny, const char *filename) {
    FILE *fp = fopen(filename, "w");
    if (!fp) return;
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            fprintf(fp, "%f ", u[j*nx + i]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
} 