#ifndef WEIGHT_H
#define WEIGHT_H

typedef struct {
    double a, b, c, d, e;
} Weights;

static inline Weights compute_weights(double dx, double dy, double dt, double alpha) {
    Weights w;
    // Dummy example weights — replace with your real computation
    w.a = alpha * dt / (dx * dx);
    w.b = alpha * dt / (dy * dy);
    w.c = 1.0 - 2.0 * (w.a + w.b);
    w.d = w.a;
    w.e = w.b;
    return w;
}

#endif
