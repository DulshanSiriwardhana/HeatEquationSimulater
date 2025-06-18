# Compiler
NVCC = nvcc

# Include header files
INCLUDES = -Iinclude

# Source files
SRC = src/main.cu src/initial.cu src/solver.cu src/boundary.cu

# Output binary
TARGET = heat_sim

# Compilation rule
all:
	$(NVCC) $(INCLUDES) $(SRC) -o $(TARGET)

# Clean rule
clean:
	rm -f $(TARGET)
