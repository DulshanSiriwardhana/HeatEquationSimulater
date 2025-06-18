import matplotlib
matplotlib.use('TkAgg')  # Use Tkinter backend for interactive plots

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D

# Parameters (should match C program)
Nx = 500
Ny = 500
max_time_step = 1000  # Same as steps in your simulation
output_folder = "output"

# Load CSV for a given time step
def load_solution(t_step):
    filename = os.path.join(output_folder, f"solution_t{t_step:04d}.csv")
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        return None
    data = np.loadtxt(filename, delimiter=",")
    if data.shape != (Ny, Nx):
        print(f"Unexpected data shape in {filename}, expected ({Ny},{Nx}) got {data.shape}")
        return None
    return data

# Prepare grid
x = np.arange(Nx)
y = np.arange(Ny)
X, Y = np.meshgrid(x, y)

# Initial data
Z = load_solution(0)
if Z is None:
    raise FileNotFoundError("Initial CSV file missing.")

# Plot setup
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='viridis')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Temperature')
ax.set_title('Heat Simulation at t=0')

cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
cbar.set_label('Temperature')

# Slider axis
slider_ax = plt.axes([0.2, 0.05, 0.6, 0.03])
time_slider = Slider(slider_ax, 'Time Step', 0, max_time_step, valinit=0, valstep=1)

# Update function for slider
def update(val):
    t = int(time_slider.val)
    Z = load_solution(t)
    if Z is None:
        return
    ax.clear()
    surf = ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Temperature')
    ax.set_title(f'Heat Simulation at t={t}')
    fig.canvas.draw_idle()

time_slider.on_changed(update)

plt.show()
