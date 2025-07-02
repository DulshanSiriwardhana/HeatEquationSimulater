#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <errno.h>

#include "../include/initial.h"
#include "../include/boundary.h"
#include "../include/solver.h"
#include "../include/export.h"

#define DEFAULT_NX 256
#define DEFAULT_NY 256
#define DEFAULT_STEPS 1000
#define DEFAULT_ALPHA 0.1f
#define DEFAULT_OUTPUT_FOLDER "output_hybrid"

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
    printf("Usage: %s [nx ny timesteps alpha output_folder]\n", progname);
    printf("Defaults: nx=%d ny=%d timesteps=%d alpha=%.2f output_folder=%s\n",
           DEFAULT_NX, DEFAULT_NY, DEFAULT_STEPS, DEFAULT_ALPHA, DEFAULT_OUTPUT_FOLDER);
}

int main(int argc, char *argv[]) {
    // Parse command-line arguments
    int nx = DEFAULT_NX, ny = DEFAULT_NY, timesteps = DEFAULT_STEPS;
    float alpha = DEFAULT_ALPHA;
    const char *output_folder = DEFAULT_OUTPUT_FOLDER;
    if (argc > 1) nx = atoi(argv[1]);
    if (argc > 2) ny = atoi(argv[2]);
    if (argc > 3) timesteps = atoi(argv[3]);
    if (argc > 4) alpha = atof(argv[4]);
    if (argc > 5) output_folder = argv[5];
    if (argc > 1 && (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0)) {
        print_usage(argv[0]);
        return 0;
    }
    if (nx < 2 || ny < 2 || timesteps < 1 || alpha <= 0) {
        fprintf(stderr, "Invalid parameters.\n");
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    create_folder_if_not_exists(output_folder);

    float *u = (float*)malloc(nx * ny * sizeof(float));
    if (!u) {
        fprintf(stderr, "Memory allocation failed!\n");
        return 1;
    }

    set_initial_conditions(u, nx, ny);
    apply_boundary_conditions(u, nx, ny);

    float *u_new = (float*)malloc(nx * ny * sizeof(float));
    if (!u_new) {
        fprintf(stderr, "Memory allocation failed!\n");
        free(u);
        return 1;
    }

    for (int t = 0; t <= timesteps; ++t) {
        export_to_csv(output_folder, u, nx, ny, t);
        solve_heat_equation(u, nx, ny, 1, alpha); // advance one step
        float *tmp = u; u = u_new; u_new = tmp;
    }

    free(u);
    free(u_new);
    printf("Simulation complete. Results written to %s/solution_tXXXX.csv\n", output_folder);
    return 0;
}