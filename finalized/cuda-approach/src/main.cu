#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <errno.h>
#include <cuda_runtime.h>

#include "../include/boundary.h"
#include "../include/initial.h"
#include "../include/solver.h"
#include "../include/export.h"

#define DEFAULT_NX 1000
#define DEFAULT_NY 1000
#define DEFAULT_STEPS 100
#define DEFAULT_DX 1.0
#define DEFAULT_DY 1.0
#define DEFAULT_DT 30.0
#define DEFAULT_ALPHA 0.01
#define DEFAULT_OUTPUT_FOLDER "output_cuda"

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

/**
 * @brief Check CUDA error and exit if failed.
 */
static void check_cuda_error(cudaError_t err, const char* operation) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", operation, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
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

    // Print CUDA device info
    int device;
    check_cuda_error(cudaGetDevice(&device), "cudaGetDevice");
    cudaDeviceProp prop;
    check_cuda_error(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
    printf("Using CUDA device: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("Global memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));

    size_t N = Nx * Ny;
    size_t grid_size = N * sizeof(double);

    // Allocate device memory
    double *d_u = NULL, *d_u_new = NULL;
    check_cuda_error(cudaMalloc(&d_u, grid_size), "cudaMalloc d_u");
    check_cuda_error(cudaMalloc(&d_u_new, grid_size), "cudaMalloc d_u_new");

    // Initialize conditions
    set_initial_conditions_cuda(d_u, Nx, Ny);
    apply_boundary_conditions_cuda(d_u, Nx, Ny, 0.0);

    // Start timing
    cudaEvent_t start, stop;
    check_cuda_error(cudaEventCreate(&start), "cudaEventCreate start");
    check_cuda_error(cudaEventCreate(&stop), "cudaEventCreate stop");
    check_cuda_error(cudaEventRecord(start), "cudaEventRecord start");

    // Main simulation loop
    for (int t = 0; t <= steps; t++) {
        export_to_csv_cuda(output_folder, d_u, Nx, Ny, t);
        if (t < steps) { // Don't advance on last iteration
            advance_time_step_cuda(d_u, d_u_new, Nx, Ny, dx, dy, dt, alpha);
            // Swap pointers
            double *temp = d_u;
            d_u = d_u_new;
            d_u_new = temp;
        }
        if (t % 10 == 0) {
            printf("Completed timestep %d/%d\n", t, steps);
        }
    }

    check_cuda_error(cudaEventRecord(stop), "cudaEventRecord stop");
    check_cuda_error(cudaEventSynchronize(stop), "cudaEventSynchronize stop");
    float elapsed_ms;
    check_cuda_error(cudaEventElapsedTime(&elapsed_ms, start, stop), "cudaEventElapsedTime");
    printf("GPU simulation and export completed in %.2f seconds.\n", elapsed_ms / 1000.0f);

    // Cleanup
    cudaFree(d_u);
    cudaFree(d_u_new);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}