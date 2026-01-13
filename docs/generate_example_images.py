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

print("\n1. Generating Example 1: Gaussian Recovery (Naive vs Deconvolution)...")

np.random.seed(42)

# Generate data
true_mean = 0.0
true_std = 10.0
n_stars = 500

true_velocities = np.random.normal(true_mean, true_std, n_stars)
measurement_errors = np.ones(n_stars) * 2.0
observed_velocities = true_velocities + np.random.normal(0, measurement_errors)

# Naive fit
naive_mean = np.mean(observed_velocities)
naive_std = np.std(observed_velocities)


# Define true PDF function
def true_pdf_ex1(x):
    return np.exp(-0.5 * ((x - true_mean) / true_std) ** 2) / (
        true_std * np.sqrt(2 * np.pi)
    )


# Run inference
solver = KinematicSolver()
solver.setup_grid(center=0.0, width=100.0, n_bins=50)
solver.add_data(vel=observed_velocities, err=measurement_errors)
samples = solver.run(num_warmup=500, num_samples=1000, gpu=False)

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left panel: Naive approach
ax = axes[0]
x_plot = np.linspace(-40, 40, 200)
naive_pdf = np.exp(-0.5 * ((x_plot - naive_mean) / naive_std) ** 2) / (
    naive_std * np.sqrt(2 * np.pi)
)

ax.hist(
    observed_velocities,
    bins=30,
    density=True,
    alpha=0.4,
    label="Observed Data",
    color="gray",
)
ax.plot(x_plot, naive_pdf, "r-", linewidth=2, label=f"Naive Fit (σ={naive_std:.1f})")
ax.plot(x_plot, true_pdf_ex1(x_plot), "k--", linewidth=2, label=f"True (σ={true_std:.1f})")
ax.set_xlabel("Velocity (km/s)")
ax.set_ylabel("Probability Density")
ax.set_title("Naive Approach: Gaussian Fit to Observed Data")
ax.legend()
ax.grid(True, alpha=0.2)

# Right panel: Deconvolution
ax = axes[1]
ax.hist(
    observed_velocities,
    bins=30,
    density=True,
    alpha=0.4,
    label="Observed Data",
    color="gray",
)
solver.plot_result(ax=ax, true_intrinsic=true_pdf_ex1)
ax.set_title("Bayesian Deconvolution: Recovers True Distribution")
ax.grid(True, alpha=0.2)

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

# Naive fit
naive_mean = np.mean(observed_velocities)
naive_std = np.std(observed_velocities)


# Define true PDF function
def true_pdf(x):
    return np.exp(-0.5 * ((x - true_mean) / true_std) ** 2) / (
        true_std * np.sqrt(2 * np.pi)
    )


# Run inference
solver = KinematicSolver()
solver.setup_grid(center=0.0, width=100.0, n_bins=50)
solver.add_data(vel=observed_velocities, err=measurement_errors)
samples = solver.run(num_warmup=500, num_samples=1000, gpu=False)

moments = compute_moments(samples["intrinsic_pdf"], solver.grid["centers"])
deconvolved_std = moments["std"][0]

# Create figure
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# Panel 1: Observed data with naive fit
ax = axes[0]
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
    "r-",
    linewidth=2,
    label=f"Naive Fit (σ={naive_std:.1f})",
)
ax.plot(x, true_pdf(x), "k--", linewidth=2, label=f"True (σ={true_std:.1f})")
ax.set_xlabel("Velocity (km/s)")
ax.set_ylabel("Probability Density")
ax.set_title("Observed Data: Naive Fit")
ax.legend()
ax.grid(True, alpha=0.2)

# Panel 2: Deconvolved result
ax = axes[1]
ax.hist(
    observed_velocities,
    bins=40,
    density=True,
    alpha=0.4,
    label="Observed",
    color="gray",
)
solver.plot_result(ax=ax, true_intrinsic=true_pdf)
ax.set_title("Deconvolved Distribution")
ax.grid(True, alpha=0.2)

# Panel 3: Direct comparison
ax = axes[2]
bins = solver.grid["centers"]
pdf_samples = samples["intrinsic_pdf"] / solver.grid["width"]
mean_pdf = np.mean(pdf_samples, axis=0)
true_pdf_vals = true_pdf(bins)

ax.plot(bins, mean_pdf, "g-", linewidth=2, label=f"Inferred (σ={deconvolved_std:.1f})")
ax.plot(bins, true_pdf_vals, "k--", linewidth=2, label=f"True (σ={true_std:.1f})")
ax.plot(
    x,
    np.exp(-0.5 * x**2 / naive_std**2) / (naive_std * np.sqrt(2 * np.pi)),
    "r:",
    linewidth=2,
    alpha=0.7,
    label=f"Naive (σ={naive_std:.1f})",
)
ax.fill_between(
    bins,
    np.percentile(pdf_samples, 16, axis=0),
    np.percentile(pdf_samples, 84, axis=0),
    alpha=0.3,
    color="green",
)
ax.set_xlabel("Velocity (km/s)")
ax.set_ylabel("Probability Density")
ax.set_title("Comparison: Deconvolution Recovers True Width")
ax.legend()
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig("images/example_deconvolution.png", dpi=150)
plt.close()

print("   ✓ Saved: images/example_deconvolution.png")

# ============================================================================
# Example 3: Bimodal Distribution (Two Separated Gaussians)
# ============================================================================

print("\n3. Generating Example 3: Bimodal Distribution...")

np.random.seed(456)

# Generate data - two well-separated Gaussians
n1, n2 = 400, 300
mean1, std1 = -15.0, 6.0
mean2, std2 = 20.0, 8.0

v1 = np.random.normal(mean1, std1, n1)
v2 = np.random.normal(mean2, std2, n2)

true_velocities = np.concatenate([v1, v2])
n_stars = len(true_velocities)

errors = np.random.uniform(3, 6, n_stars)
observed_velocities = true_velocities + np.random.normal(0, errors)

# Naive fit
naive_mean = np.mean(observed_velocities)
naive_std = np.std(observed_velocities)


# Define true PDF function (mixture of 2 Gaussians)
def true_pdf(x):
    pdf1 = np.exp(-0.5 * ((x - mean1) / std1) ** 2) / (std1 * np.sqrt(2 * np.pi))
    pdf2 = np.exp(-0.5 * ((x - mean2) / std2) ** 2) / (std2 * np.sqrt(2 * np.pi))
    w1, w2 = n1 / n_stars, n2 / n_stars
    return w1 * pdf1 + w2 * pdf2


# Run inference
solver = KinematicSolver()
solver.setup_grid(center=2.5, width=80.0, n_bins=60)
solver.add_data(vel=observed_velocities, err=errors)
samples = solver.run(num_warmup=500, num_samples=1000, gpu=False)

# Find peaks
pdf_density = samples["intrinsic_pdf"] / solver.grid["width"]
mean_pdf = np.mean(pdf_density, axis=0)
centers = solver.grid["centers"]
peaks, _ = find_peaks(mean_pdf, distance=5, prominence=0.003)

# Create figure
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# Panel 1: Observed data with naive fit
ax = axes[0]
ax.hist(observed_velocities, bins=40, density=True, alpha=0.5, color="gray", label="Observed")
x = np.linspace(-40, 50, 200)
ax.plot(
    x,
    np.exp(-0.5 * ((x - naive_mean) / naive_std) ** 2) / (naive_std * np.sqrt(2 * np.pi)),
    "r-",
    linewidth=2,
    label="Naive Fit (1 Gaussian)",
)
ax.set_xlabel("Velocity (km/s)")
ax.set_ylabel("Probability Density")
ax.set_title("Naive Approach: Misses Bimodality")
ax.legend()
ax.grid(True, alpha=0.2)

# Panel 2: Deconvolved result
ax = axes[1]
ax.hist(observed_velocities, bins=40, density=True, alpha=0.4, label="Observed", color="gray")
solver.plot_result(ax=ax, true_intrinsic=true_pdf)
ax.plot(
    centers[peaks],
    mean_pdf[peaks],
    "r*",
    markersize=15,
    label=f"Detected Peaks ({len(peaks)})",
)
ax.set_title("Deconvolution: Resolves Both Components")
ax.legend()
ax.grid(True, alpha=0.2)

# Panel 3: Components shown separately
ax = axes[2]
ax.hist(v1, bins=25, alpha=0.4, label=f"True Pop. A (n={n1})", density=True)
ax.hist(v2, bins=25, alpha=0.4, label=f"True Pop. B (n={n2})", density=True)
ax.plot(x, true_pdf(x), "k--", linewidth=2, label="True Combined")
ax.plot(centers, mean_pdf, "g-", linewidth=2, label="Inferred")
ax.set_xlabel("Velocity (km/s)")
ax.set_ylabel("Probability Density")
ax.set_title("True Components vs Inferred")
ax.legend()
ax.grid(True, alpha=0.2)

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
true_a, true_loc, true_scale = 2, 5, 8
true_velocities = skewnorm.rvs(a=true_a, loc=true_loc, scale=true_scale, size=400)
errors = 2.0 + 0.1 * np.abs(true_velocities) + np.random.uniform(0, 1, 400)
observed_velocities = true_velocities + np.random.normal(0, errors)

# Naive fit
naive_mean = np.mean(observed_velocities)
naive_std = np.std(observed_velocities)


# Define true PDF function
def true_pdf(x):
    return skewnorm.pdf(x, a=true_a, loc=true_loc, scale=true_scale)


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
x_plot = np.linspace(data_min - 5, data_max + 5, 200)
ax1.plot(
    x_plot,
    np.exp(-0.5 * ((x_plot - naive_mean) / naive_std) ** 2)
    / (naive_std * np.sqrt(2 * np.pi)),
    "r:",
    linewidth=2,
    alpha=0.7,
    label="Naive Fit",
)
solver.plot_result(ax=ax1, true_intrinsic=true_pdf)
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

# Error distribution
ax2 = fig.add_subplot(gs[2, 0])
ax2.scatter(observed_velocities, errors, alpha=0.3, s=10)
ax2.set_xlabel("Observed Velocity (km/s)", fontsize=10)
ax2.set_ylabel("Measurement Error (km/s)", fontsize=10)
ax2.set_title("Heteroscedastic Errors", fontsize=11)
ax2.grid(True, alpha=0.3)

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
