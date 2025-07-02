#include "../include/initial.h"
#include "../include/boundary.h"
#include "../include/solver.h"
#include "../include/export.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    int nx = 256, ny = 256, timesteps = 1000;
    float alpha = 0.1f;
    if (argc > 1) nx = atoi(argv[1]);
    if (argc > 2) ny = atoi(argv[2]);
    if (argc > 3) timesteps = atoi(argv[3]);
    if (argc > 4) alpha = atof(argv[4]);

    float *u = (float*)malloc(nx * ny * sizeof(float));
    if (!u) {
        fprintf(stderr, "Memory allocation failed!\n");
        return 1;
    }

    set_initial_conditions(u, nx, ny);
    apply_boundary_conditions(u, nx, ny);

    solve_heat_equation(u, nx, ny, timesteps, alpha);

    export_results(u, nx, ny, "output_hybrid.txt");

    free(u);
    printf("Simulation complete. Results written to output_hybrid.txt\n");
    return 0;
} 