"""
Generate example images for documentation.

Run this script to create the example images shown in the documentation.
These images will be saved to docs/images/ directory.

Usage:
    python generate_example_images.py
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import skewnorm

from veldist import KinematicSolver, compute_moments

# Create images directory if it doesn't exist
os.makedirs("images", exist_ok=True)

print("Generating example images for documentation...")
print("=" * 60)

# ============================================================================
# Example 1: Basic Gaussian Distribution Recovery
# ============================================================================

print("\n1. Generating Example 1: Gaussian Recovery...")

np.random.seed(42)

# Generate data
true_mean = 0.0
true_std = 10.0
n_stars = 500

true_velocities = np.random.normal(true_mean, true_std, n_stars)
measurement_errors = np.ones(n_stars) * 2.0
observed_velocities = true_velocities + np.random.normal(0, measurement_errors)

# Run inference
solver = KinematicSolver()
solver.setup_grid(center=0.0, width=100.0, n_bins=50)
solver.add_data(vel=observed_velocities, err=measurement_errors)
samples = solver.run(num_warmup=500, num_samples=1000, gpu=False)

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ax = axes[0]
ax.hist(
    observed_velocities,
    bins=30,
    density=True,
    alpha=0.5,
    label="Observed Data",
    color="gray",
)
solver.plot_result(ax=ax, true_intrinsic=true_velocities)
ax.set_title("Recovered Distribution")
ax.legend()

ax = axes[1]
ax.hist(samples["smoothness_sigma"], bins=30, alpha=0.7, color="steelblue")
ax.axvline(
    np.mean(samples["smoothness_sigma"]), color="red", linestyle="--", label="Mean"
)
ax.set_xlabel("Smoothness σ")
ax.set_ylabel("Frequency")
ax.set_title("Inferred Smoothness Parameter")
ax.legend()

plt.tight_layout()
plt.savefig("images/example_gaussian.png", dpi=150)
plt.close()

print("   ✓ Saved: images/example_gaussian.png")

# ============================================================================
# Example 2: Deconvolution with High Noise
# ============================================================================

print("\n2. Generating Example 2: High Noise Deconvolution...")

np.random.seed(123)

# Generate data
true_mean = 0.0
true_std = 5.0
n_stars = 1000

true_velocities = np.random.normal(true_mean, true_std, n_stars)
measurement_errors = np.ones(n_stars) * 7.0
observed_velocities = true_velocities + np.random.normal(0, measurement_errors)

# Run inference
solver = KinematicSolver()
solver.setup_grid(center=0.0, width=100.0, n_bins=50)
solver.add_data(vel=observed_velocities, err=measurement_errors)
samples = solver.run(num_warmup=500, num_samples=1000, gpu=False)

# Create figure
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Panel 1: Observed data
ax = axes[0]
naive_std = np.std(observed_velocities)
ax.hist(
    observed_velocities,
    bins=40,
    density=True,
    alpha=0.6,
    color="gray",
    label="Observed",
)
x = np.linspace(-30, 30, 200)
ax.plot(
    x,
    np.exp(-0.5 * x**2 / naive_std**2) / (naive_std * np.sqrt(2 * np.pi)),
    "k--",
    label=f"Gaussian fit (σ={naive_std:.1f})",
)
ax.set_xlabel("Velocity (km/s)")
ax.set_ylabel("Density")
ax.set_title("Observed Data (Naive)")
ax.legend()

# Panel 2: Deconvolved result
ax = axes[1]
solver.plot_result(ax=ax, true_intrinsic=true_velocities)
ax.set_title("Deconvolved Distribution")

# Panel 3: Comparison
ax = axes[2]
bins = solver.grid["centers"]
pdf_samples = samples["intrinsic_pdf"] / solver.grid["width"]
mean_pdf = np.mean(pdf_samples, axis=0)

true_pdf = np.exp(-0.5 * bins**2 / true_std**2) / (true_std * np.sqrt(2 * np.pi))

ax.plot(bins, mean_pdf, "g-", linewidth=2, label="Inferred")
ax.plot(bins, true_pdf, "k--", linewidth=2, label="True")
ax.fill_between(
    bins,
    np.percentile(pdf_samples, 16, axis=0),
    np.percentile(pdf_samples, 84, axis=0),
    alpha=0.3,
    color="green",
)
ax.set_xlabel("Velocity (km/s)")
ax.set_ylabel("Probability Density")
ax.set_title("Deconvolution Recovers True Width")
ax.legend()

plt.tight_layout()
plt.savefig("images/example_deconvolution.png", dpi=150)
plt.close()

print("   ✓ Saved: images/example_deconvolution.png")

# ============================================================================
# Example 3: Multi-Component System
# ============================================================================

print("\n3. Generating Example 3: Multi-Component System...")

np.random.seed(456)

# Generate data
n1, n2, n3 = 600, 300, 100
mean1, std1 = 0.0, 5.0
mean2, std2 = 0.0, 15.0
mean3, std3 = 30.0, 20.0

v1 = np.random.normal(mean1, std1, n1)
v2 = np.random.normal(mean2, std2, n2)
v3 = np.random.normal(mean3, std3, n3)

true_velocities = np.concatenate([v1, v2, v3])

errors = np.concatenate(
    [
        np.random.uniform(2, 4, n1),
        np.random.uniform(3, 6, n2),
        np.random.uniform(5, 10, n3),
    ]
)

observed_velocities = true_velocities + np.random.normal(0, errors)

# Run inference
solver = KinematicSolver()
solver.setup_grid(center=10.0, width=120.0, n_bins=60)
solver.add_data(vel=observed_velocities, err=errors)
samples = solver.run(num_warmup=500, num_samples=1000, gpu=False)

# Find peaks
pdf_density = samples["intrinsic_pdf"] / solver.grid["width"]
mean_pdf = np.mean(pdf_density, axis=0)
centers = solver.grid["centers"]
peaks, _ = find_peaks(mean_pdf, distance=5, prominence=0.001)

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Panel 1: Observed data colored by component
ax = axes[0, 0]
ax.hist(
    v1 + np.random.normal(0, errors[:n1]),
    bins=30,
    alpha=0.5,
    label="Cold disk",
    density=True,
)
ax.hist(
    v2 + np.random.normal(0, errors[n1 : n1 + n2]),
    bins=30,
    alpha=0.5,
    label="Warm disk",
    density=True,
)
ax.hist(
    v3 + np.random.normal(0, errors[n1 + n2 :]),
    bins=30,
    alpha=0.5,
    label="Halo",
    density=True,
)
ax.set_xlabel("Observed Velocity (km/s)")
ax.set_ylabel("Density")
ax.set_title("Observed Data (with errors)")
ax.legend()

# Panel 2: Inferred distribution
ax = axes[0, 1]
solver.plot_result(ax=ax, true_intrinsic=true_velocities)
ax.plot(centers[peaks], mean_pdf[peaks], "r*", markersize=15, label="Detected Peaks")
ax.set_title(f"Inferred Distribution ({len(peaks)} components)")
ax.legend()

# Panel 3: Error distribution
ax = axes[1, 0]
ax.scatter(observed_velocities, errors, alpha=0.3, s=10)
ax.set_xlabel("Observed Velocity (km/s)")
ax.set_ylabel("Measurement Error (km/s)")
ax.set_title("Heteroscedastic Errors")
ax.grid(True, alpha=0.3)

# Panel 4: Smoothness parameter evolution
ax = axes[1, 1]
ax.plot(samples["smoothness_sigma"], alpha=0.5, linewidth=0.5)
ax.axhline(
    np.mean(samples["smoothness_sigma"]), color="red", linestyle="--", label="Mean"
)
ax.set_xlabel("MCMC Sample")
ax.set_ylabel("Smoothness σ")
ax.set_title("Smoothness Parameter Trace")
ax.legend()

plt.tight_layout()
plt.savefig("images/example_multicomponent.png", dpi=150)
plt.close()

print("   ✓ Saved: images/example_multicomponent.png")

# ============================================================================
# Example 4: Complete Workflow
# ============================================================================

print("\n4. Generating Example 4: Complete Workflow...")

np.random.seed(789)

# Generate data
true_velocities = skewnorm.rvs(a=2, loc=5, scale=8, size=400)
errors = 2.0 + 0.1 * np.abs(true_velocities) + np.random.uniform(0, 1, 400)
observed_velocities = true_velocities + np.random.normal(0, errors)

# Grid setup
data_min, data_max = np.percentile(observed_velocities, [1, 99])
data_range = data_max - data_min
grid_center = np.median(observed_velocities)
grid_width = data_range * 1.5
typical_error = np.median(errors)
n_bins = int(grid_width / (2 * typical_error))
n_bins = np.clip(n_bins, 30, 100)

# Run inference
solver = KinematicSolver()
solver.setup_grid(center=grid_center, width=grid_width, n_bins=n_bins)
solver.add_data(vel=observed_velocities, err=errors)
samples = solver.run(num_warmup=1000, num_samples=2000, gpu=False)

# Create figure
fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# Main result
ax1 = fig.add_subplot(gs[0:2, :])
solver.plot_result(ax=ax1, true_intrinsic=true_velocities)
ax1.hist(
    observed_velocities,
    bins=50,
    density=True,
    alpha=0.3,
    color="gray",
    label="Observed (with errors)",
    zorder=0,
)
ax1.set_xlabel("Velocity (km/s)", fontsize=12)
ax1.set_ylabel("Probability Density", fontsize=12)
ax1.set_title("Inferred Line-of-Sight Velocity Distribution", fontsize=14)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.2)

# Convergence trace
smooth_samples = samples["smoothness_sigma"]
ax2 = fig.add_subplot(gs[2, 0])
ax2.plot(smooth_samples, alpha=0.7, linewidth=0.5)
ax2.axhline(np.mean(smooth_samples), color="red", linestyle="--")
ax2.axhline(np.percentile(smooth_samples, 16), color="red", linestyle=":", alpha=0.5)
ax2.axhline(np.percentile(smooth_samples, 84), color="red", linestyle=":", alpha=0.5)
ax2.set_xlabel("MCMC Iteration", fontsize=10)
ax2.set_ylabel("Smoothness σ", fontsize=10)
ax2.set_title("Parameter Trace", fontsize=11)

# Uncertainty map
centers = solver.grid["centers"]
pdf_density = samples["intrinsic_pdf"] / solver.grid["width"]
lower_90 = np.percentile(pdf_density, 5, axis=0)
upper_90 = np.percentile(pdf_density, 95, axis=0)

ax3 = fig.add_subplot(gs[2, 1])
ax3.fill_between(centers, lower_90, upper_90, alpha=0.3, color="steelblue")
ax3.plot(centers, np.mean(pdf_density, axis=0), "b-", linewidth=1.5)
ax3.set_xlabel("Velocity (km/s)", fontsize=10)
ax3.set_ylabel("Probability Density", fontsize=10)
ax3.set_title("90% Credible Interval", fontsize=11)
ax3.grid(True, alpha=0.2)

plt.savefig("images/example_complete_workflow.png", dpi=150, bbox_inches="tight")
plt.close()

print("   ✓ Saved: images/example_complete_workflow.png")

print("\n" + "=" * 60)
print("✓ All example images generated successfully!")
print("\nImages saved to: docs/images/")
print("\nYou can now build the documentation with: cd docs && make html")
