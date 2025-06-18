// main.cu
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "boundary.h"
#include "initial.h"
#include "solver.h"

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

    int N = Nx * Ny;
    size_t size = N * sizeof(double);

    double *h_u = (double *)calloc(N, sizeof(double));
    double *h_u_new = (double *)calloc(N, sizeof(double));
    double *d_u, *d_u_new;

    cudaMalloc((void **)&d_u, size);
    cudaMalloc((void **)&d_u_new, size);

    set_initial_conditions(h_u, Nx, Ny);
    cudaMemcpy(d_u, h_u, size, cudaMemcpyHostToDevice);

    apply_boundary_conditions<<<(Nx * Ny + 255) / 256, 256>>>(d_u, Nx, Ny, 0.0);
    cudaDeviceSynchronize();

    dim3 blockSize(16, 16);
    dim3 gridSize((Nx + blockSize.x - 1) / blockSize.x,
                  (Ny + blockSize.y - 1) / blockSize.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int t = 0; t < steps; t++) {
        advance_time_step<<<gridSize, blockSize>>>(d_u, d_u_new, Nx, Ny, dx, dy, dt, alpha);
        cudaDeviceSynchronize();
        apply_boundary_conditions<<<(Nx * Ny + 255) / 256, 256>>>(d_u_new, Nx, Ny, 0.0);
        cudaDeviceSynchronize();
        double *temp = d_u;
        d_u = d_u_new;
        d_u_new = temp;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("Simulation completed in %.2f seconds.\n", elapsed / 1000.0f);

    cudaMemcpy(h_u, d_u, size, cudaMemcpyDeviceToHost);

    free(h_u);
    free(h_u_new);
    cudaFree(d_u);
    cudaFree(d_u_new);

    return 0;
}
