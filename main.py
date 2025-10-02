# brainfolds
# PyVista simulation of gyrification process, modeling the formation of gyri and sulci in brain development.
#
# License: This software is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0).
#          You are free to use, modify, and distribute this software, provided you give appropriate credit to the
#          author, Vance Alexander (ORCID: https://orcid.org/0000-0002-0232-1428), in any publications or derivatives.
#          See the LICENSE file for full details: https://creativecommons.org/licenses/by/4.0/
#
# Author: Vance Alexander
# ORCID: https://orcid.org/0000-0002-0232-1428

import numpy as np
import pyvista as pv
from pyvistaqt import BackgroundPlotter
import time
from skimage.measure import marching_cubes
import nibabel as nib  # For loading NIfTI files
from scipy.ndimage import zoom  # For resizing atlas data
from skimage.transform import resize  # Fallback for resizing

# Define grid size first
n_points = 60  # High resolution for detailed folds [Hofman, 1991]

# Load fetal head atlas for realistic constraint
atlas_file = 'atlas/fetal-t2w-head-atlas.nii'  # Path to the downloaded file in subfolder 'atlas'
atlas = nib.load(atlas_file)
atlas_data = atlas.get_fdata()  # 3D array of intensities

# Compute mean along axis=2
atlas_mean = atlas_data.mean(axis=2)  # Mean along z-axis for 2D projection

# Resize atlas_mean to match the grid size (n_points x n_points)
# Calculate zoom factors to scale to n_points x n_points
zoom_factor_x = n_points / atlas_mean.shape[0]
zoom_factor_y = n_points / atlas_mean.shape[1]
atlas_mean_resized = zoom(atlas_mean, (zoom_factor_x, zoom_factor_y))  # Resize using zoom factors

# Ensure the resized array matches n_points x n_points (trim or pad if necessary)
if atlas_mean_resized.shape != (n_points, n_points):
    atlas_mean_resized = resize(atlas_mean_resized, (n_points, n_points), order=1, mode='constant', cval=0, anti_aliasing=True)

# Initialize a 2D grid with layered structure
x = np.linspace(-1, 1, n_points)
y = np.linspace(-1, 1, n_points)
X, Y = np.meshgrid(x, y)
Z_outer = np.interp(atlas_mean_resized.ravel(), (atlas_mean_resized.min(), atlas_mean_resized.max()), (0, 0.5))  # Normalize atlas to initial Z range
Z_outer = Z_outer.reshape(n_points, n_points)  # Reshape to match X, Y
Z_inner = np.zeros_like(X)  # Inner white matter layer (slower growth)

# Create a StructuredGrid for the outer layer
grid = pv.StructuredGrid(X, Y, Z_outer)
plotter = BackgroundPlotter()
actor = plotter.add_mesh(grid, cmap='viridis', show_edges=True)

# Simulation parameters
# sphere_radius: Represents the skull constraint, now based on fetal MRI atlas [Matthew et al., 2024]
sphere_radius = 1.5 + 0.1 * atlas_mean_resized.mean()  # Adjust base radius with atlas data
# max_growth_rate_outer: Faster growth for outer cortex, reflecting progenitor cell proliferation [Cao et al., 2017]
max_growth_rate_outer = 0.1
# max_growth_rate_inner: Slower growth for inner layer, part of differential tangential growth (DTG) [Budday et al., 2015]
max_growth_rate_inner = 0.06
# fold_strength_base: Base intensity of folding, influenced by genetic and mechanical factors [Leyva-Mendivil et al., 2020]
fold_strength_base = 0.04
# pressure_increase: Simulates increasing skull pressure over development [Matthew et al., 2024]
pressure_increase = 0.015

# Initial perturbations and genetic variability
# Initial perturbations simulate early cellular irregularities [Hofman, 1991]
Z_outer += 0.1 * np.random.normal(0, 0.1, Z_outer.shape)
# genetic_modifier: Random variations to mimic genetic influences on proliferation [Cao et al., 2017]
genetic_modifier = 1 + 0.2 * np.random.normal(0, 0.1, Z_outer.shape)

# Regional growth factor
# Higher growth in the center mimics frontal lobe proliferation [Budday et al., 2015]
regional_factor = 1 + 0.3 * np.exp(-((X**2 + Y**2) / 0.5))

# Animation loop for predictive development
for t in range(150):  # Increased steps to model gradual fetal development [Cao et al., 2017]
    # Differential tangential growth (DTG) between outer and inner layers
    # growth_factor_outer: Varies with sinusoidal function to mimic cyclic cellular activity [Budday et al., 2015]
    # Multiplied by genetic_modifier to reflect individual variability [Cao et al., 2017]
    growth_factor_outer = max_growth_rate_outer * (1 + 0.1 * np.sin(t * 0.05)) * regional_factor * genetic_modifier
    # growth_factor_inner: Slower, opposing phase growth for white matter [Budday et al., 2015]
    growth_factor_inner = max_growth_rate_inner * (1 + 0.05 * np.cos(t * 0.05)) * regional_factor
    Z_outer_new = Z_outer + growth_factor_outer * (1 - np.sqrt(X**2 + Y**2 + Z_outer**2) / sphere_radius)
    Z_inner_new = Z_inner + growth_factor_inner * (1 - np.sqrt(X**2 + Y**2 + Z_inner**2) / sphere_radius)
    
    # Fold complexity with temporal evolution
    # fold_strength: Varies over time to simulate evolving genetic and mechanical influences [Leyva-Mendivil et al., 2020]
    # Multiple terms create gyri and sulci; high-frequency folds fade with stabilization [Hofman, 1991]
    fold_strength = fold_strength_base * (1 + 0.2 * np.sin(t * 0.1))
    Z_outer_new += fold_strength * (np.sin(5 * X) * np.cos(5 * Y) + 0.5 * np.cos(10 * X) * np.sin(10 * Y))
    Z_outer_new += 0.02 * np.cos(15 * X) * np.sin(15 * Y) * np.exp(-t / 150)
    
    # Enforce spherical constraint with dynamic pressure
    # mask: Identifies points exceeding the boundary, mimicking skull limit [Matthew et al., 2024]
    # Pressure increase simulates developmental constraint [Budday et al., 2015]
    # Modify mask with atlas data for non-spherical, realistic skull shape, inspired by space-filling curve concept [Pirkowski, 2025]
    atlas_constraint = atlas_mean_resized / atlas_mean_resized.max()  # Normalize resized atlas for constraint
    mask = np.sqrt(X**2 + Y**2 + Z_outer_new**2) > sphere_radius - pressure_increase * (t / 150) + 0.2 * atlas_constraint
    Z_outer_new[mask] = (sphere_radius - pressure_increase * (t / 150)) * Z_outer[mask] / np.sqrt(X[mask]**2 + Y[mask]**2 + Z_outer[mask]**2)
    
    Z_outer = Z_outer_new
    Z_inner = Z_inner_new  # Inner layer influences outer buckling indirectly
    
    # Update the grid with outer layer
    grid.points[:, 2] = Z_outer.ravel()
    
    # Compute curvature to highlight gyri and sulci
    # Improved estimate with second derivatives for mean curvature [Leyva-Mendivil et al., 2020]
    dz_dx, dz_dy = np.gradient(Z_outer)
    curvature = np.sqrt(dz_dx**2 + dz_dy**2 + np.gradient(dz_dx)[1]**2 + np.gradient(dz_dy)[0]**2)
    grid.point_data["curvature"] = curvature.ravel()
    
    # Update actor with curvature coloring
    actor = plotter.add_mesh(grid, cmap='viridis', scalars="curvature", show_edges=True, render=True)
    plotter.update()
    plotter.app.processEvents()
    time.sleep(0.01)
    print(f"Step: {t}, Max Z: {Z_outer.max()}")

    # Dynamic fractal dimension tracking
    # Computed every 10 steps to monitor fold complexity evolution [Hofman, 1991]
    if t % 10 == 0:
        z_range = np.linspace(Z_outer.min(), Z_outer.max(), 30)
        volume = np.zeros((n_points, n_points, len(z_range)))
        for i, z in enumerate(z_range):
            volume[:, :, i] = Z_outer + (z - Z_outer.mean())
        verts, faces, _, _ = marching_cubes(volume, 0, spacing=(2/n_points, 2/n_points, (Z_outer.max()-Z_outer.min())/30))
        fractal_dim = len(faces) / len(verts)
        print(f"Step {t} Fractal Dimension: {fractal_dim}")

# Final fractal dimension
def estimate_fractal_dimension(surface_grid):
    z_range = np.linspace(Z_outer.min(), Z_outer.max(), 30)
    volume = np.zeros((n_points, n_points, len(z_range)))
    for i, z in enumerate(z_range):
        volume[:, :, i] = Z_outer + (z - Z_outer.mean())
    verts, faces, _, _ = marching_cubes(volume, 0, spacing=(2/n_points, 2/n_points, (Z_outer.max()-Z_outer.min())/30))
    return len(faces) / len(verts)

final_fractal_dim = estimate_fractal_dimension(grid)
print(f"Estimated Final Fractal Dimension: {final_fractal_dim}")

# Keep the window open
plotter.app.exec_()