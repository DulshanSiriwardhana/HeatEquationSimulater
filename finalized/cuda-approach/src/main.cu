#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <cuda_runtime.h>

#include "../include/boundary.h"
#include "../include/initial.h"
#include "../include/solver.h"
#include "../include/export.h"

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

void save_all_data_as_csv_cuda(const char *folder, double *d_all_data, int Nx, int Ny, int steps) {
    size_t single_grid_size = Nx * Ny * sizeof(double);
    double *h_grid = (double*)malloc(single_grid_size);
    
    if (!h_grid) {
        fprintf(stderr, "Host memory allocation failed for batch export\n");
        return;
    }
    
    char filename[256];
    for (int t = 0; t <= steps; t++) {
        // Copy single timestep from device to host
        cudaError_t err = cudaMemcpy(h_grid, &d_all_data[t * Nx * Ny], 
                                    single_grid_size, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA memcpy failed for timestep %d: %s\n", t, cudaGetErrorString(err));
            continue;
        }
        
        snprintf(filename, sizeof(filename), "%s/solution_t%04d.csv", folder, t);
        FILE *fp = fopen(filename, "w");
        if (!fp) {
            fprintf(stderr, "Error opening file %s\n", filename);
            continue;
        }
        
        for (int j = 0; j < Ny; j++) {
            for (int i = 0; i < Nx; i++) {
                fprintf(fp, "%.5f", h_grid[j * Nx + i]);
                if (i < Nx - 1) fprintf(fp, ",");
            }
            fprintf(fp, "\n");
        }
        fclose(fp);
    }
    
    free(h_grid);
}

void check_cuda_error(cudaError_t err, const char* operation) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", operation, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
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

    const char *output_folder = "output_cuda";
    create_folder_if_not_exists(output_folder);

    // Print CUDA device info
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("Using CUDA device: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("Global memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));

    size_t N = Nx * Ny;
    size_t grid_size = N * sizeof(double);
    size_t total_size = (steps + 1) * grid_size;

    // Allocate device memory
    double *d_u, *d_u_new, *d_all_data;
    check_cuda_error(cudaMalloc(&d_u, grid_size), "d_u allocation");
    check_cuda_error(cudaMalloc(&d_u_new, grid_size), "d_u_new allocation");
    check_cuda_error(cudaMalloc(&d_all_data, total_size), "d_all_data allocation");

    printf("Allocated %.2f GB on GPU\n", total_size / (1024.0 * 1024.0 * 1024.0));

    // Initialize conditions
    set_initial_conditions_cuda(d_u, Nx, Ny);
    apply_boundary_conditions_cuda(d_u, Nx, Ny, 0.0);

    // Start timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Main simulation loop
    for (int t = 0; t <= steps; t++) {
        // Save current state to buffer
        cudaMemcpy(&d_all_data[t * N], d_u, grid_size, cudaMemcpyDeviceToDevice);
        
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

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    printf("GPU simulation completed in %.2f seconds.\n", elapsed_ms / 1000.0f);

    // Export data
    clock_t start2 = clock();
    save_all_data_as_csv_cuda(output_folder, d_all_data, Nx, Ny, steps);
    clock_t end2 = clock();
    double elapsed2 = (double)(end2 - start2) / CLOCKS_PER_SEC;
    printf("CSV export completed in %.4f seconds\n", elapsed2);
    printf("Total time: %.4f seconds.\n", elapsed_ms / 1000.0f + elapsed2);

    // Cleanup
    cudaFree(d_u);
    cudaFree(d_u_new);
    cudaFree(d_all_data);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}