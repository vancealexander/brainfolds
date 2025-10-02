import numpy as np
import pyvista as pv
from pyvistaqt import BackgroundPlotter
import time

# Initialize a 2D grid
n_points = 50
x = np.linspace(-1, 1, n_points)
y = np.linspace(-1, 1, n_points)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)  # Start with a flat surface

# Create a StructuredGrid
grid = pv.StructuredGrid(X, Y, Z)
plotter = BackgroundPlotter()
actor = plotter.add_mesh(grid, cmap='cool', show_edges=True)  # Store the actor for reference

# Define sphere constraint
sphere_radius = 1.5

# Add initial random perturbations to seed folding
Z += 0.1 * np.random.normal(0, 0.1, Z.shape)

# Animation loop
for t in range(100):
    # Growth with a folding factor
    growth_factor = 0.05 * (1 + 0.1 * np.sin(t * 0.1))  # Varying growth rate
    Z_new = Z + growth_factor * (1 - np.sqrt(X**2 + Y**2 + Z**2) / sphere_radius)
    
    # Enhance folding with sinusoidal pattern
    fold_strength = 0.02
    Z_new += fold_strength * np.sin(5 * X) * np.cos(5 * Y)
    
    # Enforce spherical constraint
    mask = np.sqrt(X**2 + Y**2 + Z_new**2) > sphere_radius
    Z_new[mask] = sphere_radius * Z[mask] / np.sqrt(X[mask]**2 + Y[mask]**2 + Z[mask]**2)
    
    Z = Z_new
    
    # Update the grid points
    grid.points[:, 2] = Z.ravel()  # Update z-coordinates
    
    # Update the plot (no need for SetGeometry)
    plotter.update()
    
    # Process events to keep the window responsive
    plotter.app.processEvents()
    time.sleep(0.01)
    print(f"Step: {t}, Max Z: {Z.max()}")  # Debug: Check if Z is changing

# Keep the window open by running the Qt event loop
plotter.app.exec_()