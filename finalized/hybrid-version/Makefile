# Makefile for hybrid CUDA + OpenMP heat equation solver

NVCC = nvcc
CFLAGS = -Xcompiler -fopenmp -O2
INCLUDES = -Iinclude
SRC = src/main.cu src/initial.cu src/boundary.cu src/solver.cu src/export.cu
TARGET = heat_sim_hybrid

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CFLAGS) $(INCLUDES) -o $@ $^

clean:
	rm -f $(TARGET) output_hybrid.txt 