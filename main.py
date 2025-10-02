import numpy as np
import pyvista as pv
from pyvistaqt import BackgroundPlotter
import time
from skimage.measure import marching_cubes

# Initialize a 2D grid
n_points = 50
x = np.linspace(-1, 1, n_points)
y = np.linspace(-1, 1, n_points)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)  # Start with a flat surface

# Create a StructuredGrid
grid = pv.StructuredGrid(X, Y, Z)
plotter = BackgroundPlotter()
actor = plotter.add_mesh(grid, cmap='cool', show_edges=True)  # Store the actor

# Define sphere constraint
sphere_radius = 1.5

# Add initial random perturbations to seed folding
Z += 0.1 * np.random.normal(0, 0.1, Z.shape)

# Animation loop
for t in range(100):
    # Growth with a folding factor
    growth_factor = 0.05 * (1 + 0.1 * np.sin(t * 0.1))  # Varying growth rate
    Z_new = Z + growth_factor * (1 - np.sqrt(X**2 + Y**2 + Z**2) / sphere_radius)
    
    # Enhance fold complexity with multiple sinusoidal patterns
    fold_strength = 0.03  # Increased for more pronounced folds
    Z_new += fold_strength * (np.sin(5 * X) * np.cos(5 * Y) + 0.5 * np.cos(10 * X) * np.sin(10 * Y))
    
    # Enforce spherical constraint
    mask = np.sqrt(X**2 + Y**2 + Z_new**2) > sphere_radius
    Z_new[mask] = sphere_radius * Z[mask] / np.sqrt(X[mask]**2 + Y[mask]**2 + Z[mask]**2)
    
    Z = Z_new
    
    # Update the grid points
    grid.points[:, 2] = Z.ravel()  # Update z-coordinates
    
    # Compute curvature (simple estimate using gradients)
    dz_dx, dz_dy = np.gradient(Z)
    curvature = np.sqrt(dz_dx**2 + dz_dy**2)  # Magnitude of gradient as curvature proxy
    grid.point_data["curvature"] = curvature.ravel()  # Add as scalar field
    
    # Update the actor with curvature coloring
    actor = plotter.add_mesh(grid, cmap='viridis', scalars="curvature", show_edges=True, render=True)
    
    # Update the plot
    plotter.update()
    plotter.app.processEvents()
    time.sleep(0.01)
    print(f"Step: {t}, Max Z: {Z.max()}")  # Debug: Check progress

# Optional: Estimate fractal dimension (post-processing after animation)
def estimate_fractal_dimension(surface_grid):
    # Convert to volume for marching_cubes (add z-dimension)
    z_range = np.linspace(Z.min(), Z.max(), 20)
    volume = np.zeros((n_points, n_points, len(z_range)))
    for i, z in enumerate(z_range):
        volume[:, :, i] = Z + (z - Z.mean())  # Offset for height
    verts, faces, _, _ = marching_cubes(volume, 0, spacing=(2/n_points, 2/n_points, (Z.max()-Z.min())/20))
    return len(faces) / len(verts)  # Rough fractal dimension estimate

final_fractal_dim = estimate_fractal_dimension(grid)
print(f"Estimated Fractal Dimension: {final_fractal_dim}")

# Keep the window open by running the Qt event loop
plotter.app.exec_()