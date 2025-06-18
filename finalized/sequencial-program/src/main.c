#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#include "boundary.h"
#include "initial.h"
#include "solver.h"
#include "export.h"

void create_folder_if_not_exists(const char *folder) {
#ifdef _WIN32
    _mkdir(folder);
#else
    struct stat st = {0};
    if (stat(folder, &st) == -1) {
        mkdir(folder, 0777);
    }
#endif
}

void save_all_data_as_csv(const char *folder, double *all_data, int Nx, int Ny, int steps) {
    char filename[256];
    for (int t = 0; t <= steps; t++) {
        snprintf(filename, sizeof(filename), "%s/solution_t%04d.csv", folder, t);
        FILE *fp = fopen(filename, "w");
        if (!fp) {
            fprintf(stderr, "Error opening file %s\n", filename);
            continue;
        }
        for (int j = 0; j < Ny; j++) {
            for (int i = 0; i < Nx; i++) {
                int index = t * Nx * Ny + j * Nx + i;
                fprintf(fp, "%.5f", all_data[index]);
                if (i < Nx - 1) fprintf(fp, ",");
            }
            fprintf(fp, "\n");
        }
        fclose(fp);
    }
}

int main() {
    const int Nx = 1000;
    const int Ny = 1000;
    const int steps = 100;

    const double dx = 1.0;
    const double dy = 1.0;
    const double dt = 30;
    const double alpha = 0.01;

    const char *output_folder = "output";
    create_folder_if_not_exists(output_folder);

    int N = Nx * Ny;
    double *u = calloc(N, sizeof(double));
    double *u_new = calloc(N, sizeof(double));
    double *all_data = calloc((steps + 1) * N, sizeof(double));

    if (!u || !u_new || !all_data) {
        fprintf(stderr, "Memory allocation failed\n");
        return EXIT_FAILURE;
    }

    set_initial_conditions(u, Nx, Ny);
    apply_boundary_conditions(u, Nx, Ny, 0.0);

    clock_t start = clock();

    for (int t = 0; t <= steps; t++) {
        // Save current u to buffer
        memcpy(&all_data[t * N], u, N * sizeof(double));
        advance_time_step(u, u_new, Nx, Ny, dx, dy, dt, alpha);
        double *temp = u;
        u = u_new;
        u_new = temp;
    }

    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Simulation completed in %.2f seconds.\n", elapsed);

    clock_t start2 = clock();
    save_all_data_as_csv(output_folder, all_data, Nx, Ny, steps);
    clock_t end2 = clock();
    double elapsed2 = (double)(end2 - start2) / CLOCKS_PER_SEC;
    printf("CSV save completed in %.4f seconds\n", elapsed2);
    printf("Total Time spent: %.4f seconds.\n", elapsed+elapsed2);

    free(u);
    free(u_new);
    free(all_data);

    return 0;
}
