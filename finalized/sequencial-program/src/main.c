#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <errno.h>

#include "boundary.h"
#include "initial.h"
#include "solver.h"
#include "export.h"

#define DEFAULT_NX 1000
#define DEFAULT_NY 1000
#define DEFAULT_STEPS 100
#define DEFAULT_DX 1.0
#define DEFAULT_DY 1.0
#define DEFAULT_DT 30.0
#define DEFAULT_ALPHA 0.01
#define DEFAULT_OUTPUT_FOLDER "output"

/**
 * @brief Create a folder if it does not exist.
 */
static void create_folder_if_not_exists(const char *folder) {
#ifdef _WIN32
    _mkdir(folder);
#else
    struct stat st = {0};
    if (stat(folder, &st) == -1) {
        if (mkdir(folder, 0777) != 0 && errno != EEXIST) {
            fprintf(stderr, "Failed to create output directory: %s\n", folder);
            exit(EXIT_FAILURE);
        }
    }
#endif
}

/**
 * @brief Print usage instructions.
 */
static void print_usage(const char *progname) {
    printf("Usage: %s [Nx Ny steps dx dy dt alpha output_folder]\n", progname);
    printf("Defaults: Nx=%d Ny=%d steps=%d dx=%.1f dy=%.1f dt=%.1f alpha=%.2f output_folder=%s\n",
           DEFAULT_NX, DEFAULT_NY, DEFAULT_STEPS, DEFAULT_DX, DEFAULT_DY, DEFAULT_DT, DEFAULT_ALPHA, DEFAULT_OUTPUT_FOLDER);
}

int main(int argc, char *argv[]) {
    // Parse command-line arguments
    int Nx = DEFAULT_NX, Ny = DEFAULT_NY, steps = DEFAULT_STEPS;
    double dx = DEFAULT_DX, dy = DEFAULT_DY, dt = DEFAULT_DT, alpha = DEFAULT_ALPHA;
    const char *output_folder = DEFAULT_OUTPUT_FOLDER;
    if (argc > 1) Nx = atoi(argv[1]);
    if (argc > 2) Ny = atoi(argv[2]);
    if (argc > 3) steps = atoi(argv[3]);
    if (argc > 4) dx = atof(argv[4]);
    if (argc > 5) dy = atof(argv[5]);
    if (argc > 6) dt = atof(argv[6]);
    if (argc > 7) alpha = atof(argv[7]);
    if (argc > 8) output_folder = argv[8];
    if (argc > 1 && (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0)) {
        print_usage(argv[0]);
        return 0;
    }
    if (Nx < 2 || Ny < 2 || steps < 1 || dx <= 0 || dy <= 0 || dt <= 0 || alpha <= 0) {
        fprintf(stderr, "Invalid parameters.\n");
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    create_folder_if_not_exists(output_folder);

    int N = Nx * Ny;
    double *u = calloc(N, sizeof(double));
    double *u_new = calloc(N, sizeof(double));
    if (!u || !u_new) {
        fprintf(stderr, "Memory allocation failed.\n");
        free(u); free(u_new);
        return EXIT_FAILURE;
    }

    set_initial_conditions(u, Nx, Ny);
    apply_boundary_conditions(u, Nx, Ny, 0.0);

    clock_t start = clock();
    for (int t = 0; t <= steps; t++) {
        export_to_csv(output_folder, u, Nx, Ny, t);
        advance_time_step(u, u_new, Nx, Ny, dx, dy, dt, alpha);
        double *temp = u; u = u_new; u_new = temp;
    }
    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Simulation and export completed in %.2f seconds.\n", elapsed);

    free(u);
    free(u_new);
    return 0;
}
