import os
import numpy as np
final = 100

def getFile(output_folder):
    filename = os.path.join(output_folder, f"solution_t{final:04d}.csv")
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        return None
    data = np.loadtxt(filename, delimiter=",")
    print(data)
    return data

sequencial = getFile('../openMP-approach/output')
openmp = getFile('../sequencial-program/output')

def compareMatrix(seq, omp):
    count = 0
    Ny = len(seq)
    Nx = len(seq[0])

    for y in range(Ny):
        for x in range(Nx):
            if seq[y][x]!=omp[y][x]:
                count+=1
    return count

print("Number of different values: ", compareMatrix(sequencial, openmp))
