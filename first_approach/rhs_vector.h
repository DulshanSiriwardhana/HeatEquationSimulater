#ifndef RHS_VECTOR_H
#define RHS_VECTOR_H

static inline void compute_rhs_vector(const double *u, const double *boundary, int Nx, int Ny,
                                      double a, double b, double c, double d, double e, double *rhs) {
    // Dummy example: simple copy with boundary overlay — replace with your actual solver rhs computation
    int N = Nx * Ny;
    for (int i = 0; i < N; ++i) {
        rhs[i] = u[i];
        if (boundary[i] != 0.0) {
            rhs[i] = boundary[i];  // enforce boundary condition
        }
    }
}

#endif
