#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <omp.h>

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

int main() {
    const int Nx = 50;
    const int Ny = 50;
    const int steps = 1000000;

    const double dx = 1.0;
    const double dy = 1.0;
    const double dt = 1;
    const double alpha = 0.01;

    const char *output_folder = "output";
    create_folder_if_not_exists(output_folder);

    int N = Nx * Ny;
    double *u = calloc(N, sizeof(double));
    double *u_new = calloc(N, sizeof(double));
    if (!u || !u_new) {
        fprintf(stderr, "Memory allocation failed\n");
        return EXIT_FAILURE;
    }

    set_initial_conditions(u, Nx, Ny);
    apply_boundary_conditions(u, Nx, Ny, 0.0);

    double start = omp_get_wtime();

    for (int t = 0; t <= steps; t++) {
        //export_to_csv(output_folder, u, Nx, Ny, t);
        advance_time_step(u, u_new, Nx, Ny, dx, dy, dt, alpha);
        // Swap pointers
        double *temp = u;
        u = u_new;
        u_new = temp;
    }

    double end = omp_get_wtime();
    double elapsed = end - start;
    printf("Simulation completed in %.2f seconds.\n", elapsed);

    free(u);
    free(u_new);

    return 0;
}
