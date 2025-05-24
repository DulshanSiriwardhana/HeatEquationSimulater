#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#include "weight.h"
#include "boundary_input.h"
#include "rhs_vector.h"
#include "export.h"
#include "boundary_conditions.h"
#include "initial_conditions.h"

void create_output_folder(const char *folder) {
#ifdef _WIN32
    _mkdir(folder);
#else
    mkdir(folder, 0777);
#endif
}

int main() {
    int Nx = 50, Ny = 50, steps = 100;
    double dx = 1.0, dy = 1.0, dt = 0.1, alpha = 0.01;

    int N = Nx * Ny;
    double *u = calloc(N, sizeof(double));
    double *u_new = calloc(N, sizeof(double));
    double *boundary = calloc(N, sizeof(double));
    double *rhs = calloc(N, sizeof(double));

    create_output_folder("output");

    set_boundary_conditions(boundary, Nx, Ny);
    set_initial_conditions(u, Nx, Ny);

    clock_t start = clock();

    for (int t = 0; t <= steps; ++t) {
        Weights w = compute_weights(dx, dy, dt, alpha);
        compute_rhs_vector(u, boundary, Nx, Ny, w.a, w.b, w.c, w.d, w.e, rhs);

        memcpy(u_new, rhs, N * sizeof(double));
        export_solution(u_new, Nx, Ny, t);

        memcpy(u, u_new, N * sizeof(double));
    }

    clock_t end = clock();
    double time_taken = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Simulation completed in %.2f seconds.\n", time_taken);

    free(u);
    free(u_new);
    free(boundary);
    free(rhs);

    return 0;
}
