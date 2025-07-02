#include <math.h>
#include "initial.h"
#include <stddef.h>

/**
 * @brief Gaussian function for initial condition.
 */
static double gaussian(double x, double y, double x0, double y0, double sigma) {
    double dx = x - x0;
    double dy = y - y0;
    return exp(-(dx*dx + dy*dy) / (2.0 * sigma * sigma));
}

/**
 * @brief Set the initial temperature distribution on the grid.
 *
 * Initializes the grid with a sum of Gaussians.
 */
void set_initial_conditions(double *u, int Nx, int Ny) {
    if (!u || Nx < 1 || Ny < 1) return;
    double sigma = Nx / 10.0;
    double base_temp = 20.0;
    for (int j = 0; j < Ny; j++) {
        for (int i = 0; i < Nx; i++) {
            double temp = base_temp;
            temp += 80.0 * gaussian(i, j, Nx / 3.0, Ny / 3.0, sigma);
            temp += 100.0 * gaussian(i, j, Nx / 2.0, Ny / 2.0, sigma);
            temp += 70.0 * gaussian(i, j, 2.0 * Nx / 3.0, 2.0 * Ny / 3.0, sigma);
            u[j * Nx + i] = temp;
        }
    }
}
