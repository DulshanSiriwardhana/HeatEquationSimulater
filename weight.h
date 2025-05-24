#include <stdio.h>
#include <math.h>

typedef struct {
    double a;
    double b;
    double c;
    double d;
    double e;
} Result;

Result computeValues(double deltaX, double deltaY, double deltaT, double alpha) {
    Result res;
    
    res.a = (-deltaT * alpha)/ (deltaY*deltaY);
    res.b = (-deltaT * alpha)/ (deltaX*deltaX);
    res.c = res.a;
    res.d = res.b;
    res.e = 1-2*res.a -2*res.b;

    return res;
}
