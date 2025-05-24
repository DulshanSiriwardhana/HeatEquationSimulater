import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D

# Load data from CSV inside "output" folder
def load_solution(t_step, Nx, Ny, folder="output"):
    filename = os.path.join(folder, f"solution_t{t_step:03d}.csv")
    if not os.path.exists(filename):
        print(f"File {filename} not found!")
        return None

    data = np.loadtxt(filename, delimiter=",")
    if data.shape != (Ny, Nx):
        print(f"Unexpected shape in {filename}")
        return None
    return data

# Set grid size
Nx = 50
Ny = 50
max_time_step = 100  # Adjust based on your simulation

# Generate coordinate grids
x = np.arange(Nx)
y = np.arange(Ny)
X, Y = np.meshgrid(x, y)

# Initial data
initial_data = load_solution(0, Nx, Ny)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Initial surface plot
surf = ax.plot_surface(X, Y, initial_data, cmap='viridis')

# Add labels and color bar
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Temperature')
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

# Slider setup
ax_slider = plt.axes([0.2, 0.02, 0.6, 0.03])
slider = Slider(ax_slider, 'Time Step', 0, max_time_step, valinit=0, valstep=1)

# Update function
def update(val):
    t = int(slider.val)
    new_data = load_solution(t, Nx, Ny)
    if new_data is None:
        return
    ax.clear()
    surf = ax.plot_surface(X, Y, new_data, cmap='viridis')
    ax.set_title(f"Time step: {t}")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Temperature')

slider.on_changed(update)
plt.tight_layout()
plt.show()
