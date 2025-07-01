# CUDA Heat Diffusion Simulation Makefile

# Compiler and flags
NVCC = nvcc
CXXFLAGS = -O3 -std=c++11
CUDAFLAGS = -arch=sm_60 -O3 -Iinclude

# Source files
SOURCES = src/main.cu src/boundary.cu src/export.cu src/initial.cu src/solver.cu
HEADERS = include/boundary.h include/export.h include/initial.h include/solver.h

# Target executable
TARGET = heat_diffusion_cuda

# Default target
all: $(TARGET)

# Build target
$(TARGET): $(SOURCES) $(HEADERS)
	$(NVCC) $(CUDAFLAGS) $(CXXFLAGS) -o $(TARGET) $(SOURCES)

# Clean target
clean:
	rm -f $(TARGET)
	rm -rf output_cuda/

# Run target
run: $(TARGET)
	./$(TARGET)

# Debug build
debug: CUDAFLAGS += -g -G
debug: $(TARGET)

# Check CUDA installation
check-cuda:
	@nvcc --version
	@nvidia-smi

.PHONY: all clean run debug check-cuda