"""
Generate ALL example images for documentation (including intermediate plots).

Run this script to create all the images shown in the documentation,
including data generation plots, intermediate results, and final comparisons.
These images will be saved to docs/images/ directory.

Usage:
    cd docs
    python generate_all_example_images.py
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

print("Generating ALL example images for documentation...")
print("=" * 70)

# ============================================================================
# EXAMPLE 1: Basic Gaussian Distribution Recovery
# ============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 1: Basic Gaussian Recovery")
print("=" * 70)

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

def true_pdf_ex1(x):
    return np.exp(-0.5 * ((x - true_mean) / true_std) ** 2) / (
        true_std * np.sqrt(2 * np.pi)
    )

# --- Plot 1a: Data generation visualization ---
print("  → Generating ex1_data_generation.png...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.hist(true_velocities, bins=30, alpha=0.5, label='True intrinsic', density=True)
ax1.hist(observed_velocities, bins=30, alpha=0.5, label='Observed (with errors)', density=True)
ax1.set_xlabel('Velocity (km/s)')
ax1.set_ylabel('Probability Density')
ax1.set_title('Generated Data')
ax1.legend()
ax1.grid(True, alpha=0.2)

ax2.scatter(true_velocities, observed_velocities - true_velocities, alpha=0.3, s=10)
ax2.axhline(0, color='red', linestyle='--', alpha=0.5)
ax2.set_xlabel('True Velocity (km/s)')
ax2.set_ylabel('Measurement Error (km/s)')
ax2.set_title('Observational Noise')
ax2.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig('images/ex1_data_generation.png', dpi=150, bbox_inches='tight')
plt.close()

# Run inference
solver = KinematicSolver()
solver.setup_grid(center=0.0, width=100.0, n_bins=50)
solver.add_data(vel=observed_velocities, err=measurement_errors)
samples = solver.run(num_warmup=500, num_samples=1000, gpu=False)

# --- Plot 1b: Intermediate results (posterior analysis) ---
print("  → Generating ex1_posterior_analysis.png...")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Left: Posterior distribution of the velocity dispersion
ax = axes[0]
std_samples = [compute_moments(samples['intrinsic_pdf'][i:i+1], solver.grid['centers'])['std'][0] 
               for i in range(samples['intrinsic_pdf'].shape[0])]
ax.hist(std_samples, bins=40, alpha=0.7, edgecolor='black')
ax.axvline(true_std, color='red', linestyle='--', linewidth=2, label=f'True σ={true_std:.1f}')
ax.axvline(naive_std, color='orange', linestyle=':', linewidth=2, label=f'Naive σ={naive_std:.1f}')
ax.axvline(np.mean(std_samples), color='green', linewidth=2, label=f'Inferred σ={np.mean(std_samples):.1f}')
ax.set_xlabel('Velocity Dispersion (km/s)')
ax.set_ylabel('Posterior Samples')
ax.set_title('Recovered Dispersion')
ax.legend()
ax.grid(True, alpha=0.2)

# Middle: Smoothness parameter posterior
ax = axes[1]
ax.hist(samples['smoothness_sigma'], bins=40, alpha=0.7, edgecolor='black', color='steelblue')
ax.set_xlabel('Smoothness σ')
ax.set_ylabel('Posterior Samples')
ax.set_title('Inferred Smoothness Prior')
ax.grid(True, alpha=0.2)

# Right: Sample of posterior PDFs
ax = axes[2]
n_samples_to_plot = 50
indices = np.random.choice(samples['intrinsic_pdf'].shape[0], n_samples_to_plot, replace=False)
centers = solver.grid['centers']
pdf_density = samples['intrinsic_pdf'] / solver.grid['width']
for idx in indices:
    ax.plot(centers, pdf_density[idx], 'gray', alpha=0.1, linewidth=1)
ax.plot(centers, np.mean(pdf_density, axis=0), 'green', linewidth=2, label='Mean posterior')
ax.plot(centers, true_pdf_ex1(centers), 'r--', linewidth=2, label='True')
ax.set_xlabel('Velocity (km/s)')
ax.set_ylabel('Probability Density')
ax.set_title('Posterior PDF Samples')
ax.legend()
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('images/ex1_posterior_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

# --- Plot 1c: Final comparison ---
print("  → Generating example_gaussian.png (final comparison)...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
x_plot = np.linspace(-40, 40, 200)
naive_pdf = np.exp(-0.5 * ((x_plot - naive_mean) / naive_std) ** 2) / (
    naive_std * np.sqrt(2 * np.pi)
)
ax.hist(observed_velocities, bins=30, density=True, alpha=0.4, label="Observed Data", color="gray")
ax.plot(x_plot, naive_pdf, "r-", linewidth=2, label=f"Naive Fit (σ={naive_std:.1f})")
ax.plot(x_plot, true_pdf_ex1(x_plot), "k--", linewidth=2, label=f"True (σ={true_std:.1f})")
ax.set_xlabel("Velocity (km/s)")
ax.set_ylabel("Probability Density")
ax.set_title("Naive Approach: Gaussian Fit to Observed Data")
ax.legend()
ax.grid(True, alpha=0.2)

ax = axes[1]
ax.hist(observed_velocities, bins=30, density=True, alpha=0.4, label="Observed Data", color="gray")
solver.plot_result(ax=ax, true_intrinsic=true_pdf_ex1)
ax.set_title("Bayesian Deconvolution: Recovers True Distribution")
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig("images/example_gaussian.png", dpi=150, bbox_inches='tight')
plt.close()

print("  ✓ Example 1 complete (3 plots)")

# ============================================================================
# EXAMPLE 2: Deconvolution with High Noise
# ============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 2: High Noise Deconvolution")
print("=" * 70)

np.random.seed(123)

# Generate data
true_mean = 0.0
true_std = 5.0
n_stars = 1000

true_velocities = np.random.normal(true_mean, true_std, n_stars)
measurement_errors = np.ones(n_stars) * 7.0
observed_velocities = true_velocities + np.random.normal(0, measurement_errors)

naive_mean = np.mean(observed_velocities)
naive_std = np.std(observed_velocities)
obs_std = np.std(observed_velocities)

def true_pdf_ex2(x):
    return np.exp(-0.5 * ((x - true_mean) / true_std) ** 2) / (
        true_std * np.sqrt(2 * np.pi)
    )

# --- Plot 2a: Data generation visualization ---
print("  → Generating ex2_data_generation.png...")
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(true_velocities, bins=40, alpha=0.6, label=f'True (σ={true_std:.1f} km/s)', density=True)
ax.hist(observed_velocities, bins=40, alpha=0.6, label=f'Observed (σ={obs_std:.1f} km/s)', density=True)
ax.axvline(0, color='black', linestyle='--', alpha=0.3, label='True mean')
ax.set_xlabel('Velocity (km/s)')
ax.set_ylabel('Probability Density')
ax.set_title('High Noise Case: Observed Distribution is Much Wider')
ax.legend()
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig('images/ex2_data_generation.png', dpi=150, bbox_inches='tight')
plt.close()

# Run inference
solver = KinematicSolver()
solver.setup_grid(center=0.0, width=100.0, n_bins=50)
solver.add_data(vel=observed_velocities, err=measurement_errors)
samples = solver.run(num_warmup=500, num_samples=1000, gpu=False)

moments = compute_moments(samples['intrinsic_pdf'], solver.grid['centers'])
deconvolved_std = moments['std'][0]

# --- Plot 2b: Deconvolution process visualization ---
print("  → Generating ex2_deconvolution_process.png...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: Show the deconvolution process conceptually
ax = ax1
x = np.linspace(-30, 30, 200)

true_pdf_vals = np.exp(-0.5*(x/true_std)**2) / (true_std*np.sqrt(2*np.pi))
ax.fill_between(x, true_pdf_vals, alpha=0.3, color='blue', label=f'True (σ={true_std:.1f})')

error_kernel = np.exp(-0.5*(x/measurement_errors[0])**2) / (measurement_errors[0]*np.sqrt(2*np.pi))
ax.plot(x, error_kernel * 0.5, 'r--', linewidth=2, label=f'Measurement blur (σ={measurement_errors[0]:.1f})')

obs_pdf = np.exp(-0.5*(x/naive_std)**2) / (naive_std*np.sqrt(2*np.pi))
ax.plot(x, obs_pdf, 'orange', linewidth=2, label=f'Observed (σ={naive_std:.1f})')

centers = solver.grid['centers']
pdf_density = samples['intrinsic_pdf'] / solver.grid['width']
mean_pdf = np.mean(pdf_density, axis=0)
ax.plot(centers, mean_pdf, 'g-', linewidth=2.5, label=f'Deconvolved (σ={deconvolved_std:.1f})')

ax.set_xlabel('Velocity (km/s)')
ax.set_ylabel('Probability Density')
ax.set_title('Deconvolution Process')
ax.legend()
ax.grid(True, alpha=0.2)

# Right: Posterior uncertainty in recovered width
ax = ax2
std_samples = [compute_moments(samples['intrinsic_pdf'][i:i+1], solver.grid['centers'])['std'][0] 
               for i in range(samples['intrinsic_pdf'].shape[0])]
ax.hist(std_samples, bins=50, alpha=0.7, edgecolor='black', color='green')
ax.axvline(true_std, color='red', linestyle='--', linewidth=2.5, label=f'True σ={true_std:.1f}')
ax.axvline(naive_std, color='orange', linestyle=':', linewidth=2.5, label=f'Naive σ={naive_std:.1f}')
ax.set_xlabel('Inferred Velocity Dispersion (km/s)')
ax.set_ylabel('Posterior Samples')
ax.set_title('Posterior: Recovered Width')
ax.legend()
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('images/ex2_deconvolution_process.png', dpi=150, bbox_inches='tight')
plt.close()

# --- Plot 2c: Final comparison ---
print("  → Generating example_deconvolution.png (final comparison)...")
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

ax = axes[0]
ax.hist(observed_velocities, bins=40, density=True, alpha=0.6, color='gray', label='Observed')
x = np.linspace(-30, 30, 200)
ax.plot(x, np.exp(-0.5*x**2/naive_std**2)/(naive_std*np.sqrt(2*np.pi)), 
        'r-', linewidth=2, label=f'Naive Fit (σ={naive_std:.1f})')
ax.plot(x, true_pdf_ex2(x), 'k--', linewidth=2, label=f'True (σ={true_std:.1f})')
ax.set_xlabel('Velocity (km/s)')
ax.set_ylabel('Probability Density')
ax.set_title('Observed Data: Naive Fit')
ax.legend()
ax.grid(True, alpha=0.2)

ax = axes[1]
ax.hist(observed_velocities, bins=40, density=True, alpha=0.4, label='Observed', color='gray')
solver.plot_result(ax=ax, true_intrinsic=true_pdf_ex2)
ax.set_title('Deconvolved Distribution')
ax.grid(True, alpha=0.2)

ax = axes[2]
bins = solver.grid['centers']
pdf_samples = samples['intrinsic_pdf'] / solver.grid['width']
mean_pdf = np.mean(pdf_samples, axis=0)
true_pdf_vals = true_pdf_ex2(bins)

ax.plot(bins, mean_pdf, 'g-', linewidth=2, label=f'Inferred (σ={deconvolved_std:.1f})')
ax.plot(bins, true_pdf_vals, 'k--', linewidth=2, label=f'True (σ={true_std:.1f})')
ax.plot(x, np.exp(-0.5*x**2/naive_std**2)/(naive_std*np.sqrt(2*np.pi)), 
        'r:', linewidth=2, alpha=0.7, label=f'Naive (σ={naive_std:.1f})')
ax.fill_between(bins, 
                np.percentile(pdf_samples, 16, axis=0),
                np.percentile(pdf_samples, 84, axis=0),
                alpha=0.3, color='green')
ax.set_xlabel('Velocity (km/s)')
ax.set_ylabel('Probability Density')
ax.set_title('Comparison: Deconvolution Recovers True Width')
ax.legend()
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('images/example_deconvolution.png', dpi=150, bbox_inches='tight')
plt.close()

print("  ✓ Example 2 complete (3 plots)")

# ============================================================================
# EXAMPLE 3: Bimodal Distribution
# ============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 3: Bimodal Distribution")
print("=" * 70)

np.random.seed(456)

# Generate data
n1 = 400
mean1, std1 = -15.0, 6.0
n2 = 300
mean2, std2 = 20.0, 8.0

v1 = np.random.normal(mean1, std1, n1)
v2 = np.random.normal(mean2, std2, n2)

true_velocities = np.concatenate([v1, v2])
n_stars = len(true_velocities)

errors = np.random.uniform(3, 6, n_stars)
observed_velocities = true_velocities + np.random.normal(0, errors)

naive_mean = np.mean(observed_velocities)
naive_std = np.std(observed_velocities)

def true_pdf_ex3(x):
    pdf1 = np.exp(-0.5 * ((x - mean1) / std1)**2) / (std1 * np.sqrt(2 * np.pi))
    pdf2 = np.exp(-0.5 * ((x - mean2) / std2)**2) / (std2 * np.sqrt(2 * np.pi))
    w1, w2 = n1 / n_stars, n2 / n_stars
    return w1 * pdf1 + w2 * pdf2

# --- Plot 3a: Data generation visualization ---
print("  → Generating ex3_data_generation.png...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.hist(v1, bins=30, alpha=0.5, label=f'Pop. A: μ={mean1:.0f}', density=True, color='blue')
ax1.hist(v2, bins=30, alpha=0.5, label=f'Pop. B: μ={mean2:.0f}', density=True, color='orange')
ax1.hist(true_velocities, bins=50, alpha=0.3, label='Combined', density=True, color='gray')
ax1.set_xlabel('Velocity (km/s)')
ax1.set_ylabel('Probability Density')
ax1.set_title('True Bimodal Distribution')
ax1.legend()
ax1.grid(True, alpha=0.2)

ax2.hist(observed_velocities, bins=50, alpha=0.6, density=True, color='gray')
ax2.set_xlabel('Velocity (km/s)')
ax2.set_ylabel('Probability Density')
ax2.set_title('Observed Data (with measurement errors)')
ax2.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig('images/ex3_data_generation.png', dpi=150, bbox_inches='tight')
plt.close()

# Run inference
solver = KinematicSolver()
solver.setup_grid(center=2.5, width=80.0, n_bins=60)
solver.add_data(vel=observed_velocities, err=errors)
samples = solver.run(num_warmup=500, num_samples=1000, gpu=False)

# Find peaks
pdf_density = samples['intrinsic_pdf'] / solver.grid['width']
mean_pdf = np.mean(pdf_density, axis=0)
centers = solver.grid['centers']
peaks, properties = find_peaks(mean_pdf, distance=5, prominence=0.003)

# --- Plot 3b: Posterior uncertainty ---
print("  → Generating ex3_posterior_uncertainty.png...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: Posterior samples showing bimodal recovery
ax = ax1
n_samples_to_plot = 100
indices = np.random.choice(pdf_density.shape[0], n_samples_to_plot, replace=False)
for idx in indices:
    ax.plot(centers, pdf_density[idx], 'gray', alpha=0.05, linewidth=0.5)
ax.plot(centers, mean_pdf, 'green', linewidth=2.5, label='Mean posterior', zorder=10)
ax.plot(centers, true_pdf_ex3(centers), 'r--', linewidth=2, label='True bimodal', zorder=10)
ax.plot(centers[peaks], mean_pdf[peaks], 'k*', markersize=15, label='Detected peaks', zorder=11)
ax.set_xlabel('Velocity (km/s)')
ax.set_ylabel('Probability Density')
ax.set_title('Posterior: 100 Samples + Mean')
ax.legend()
ax.grid(True, alpha=0.2)

# Right: Peak locations across posterior samples
ax = ax2
peak1_locations = []
peak2_locations = []
for i in range(min(500, pdf_density.shape[0])):
    peaks_i, _ = find_peaks(pdf_density[i], distance=5, prominence=0.002)
    if len(peaks_i) >= 2:
        sorted_peaks = sorted(peaks_i, key=lambda p: centers[p])
        peak1_locations.append(centers[sorted_peaks[0]])
        peak2_locations.append(centers[sorted_peaks[1]])

ax.hist(peak1_locations, bins=30, alpha=0.6, color='blue', label=f'Component 1 (true: {mean1:.0f})')
ax.hist(peak2_locations, bins=30, alpha=0.6, color='orange', label=f'Component 2 (true: {mean2:.0f})')
ax.axvline(mean1, color='blue', linestyle='--', linewidth=2)
ax.axvline(mean2, color='orange', linestyle='--', linewidth=2)
ax.set_xlabel('Peak Velocity (km/s)')
ax.set_ylabel('Posterior Samples')
ax.set_title('Uncertainty in Peak Locations')
ax.legend()
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('images/ex3_posterior_uncertainty.png', dpi=150, bbox_inches='tight')
plt.close()

# --- Plot 3c: Final comparison ---
print("  → Generating example_multicomponent.png (final comparison)...")
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

ax = axes[0]
ax.hist(observed_velocities, bins=40, density=True, alpha=0.5, color='gray', label='Observed')
x = np.linspace(-40, 50, 200)
ax.plot(x, np.exp(-0.5*((x-naive_mean)/naive_std)**2)/(naive_std*np.sqrt(2*np.pi)), 
        'r-', linewidth=2, label='Naive Fit (1 Gaussian)')
ax.set_xlabel('Velocity (km/s)')
ax.set_ylabel('Probability Density')
ax.set_title('Naive Approach: Misses Bimodality')
ax.legend()
ax.grid(True, alpha=0.2)

ax = axes[1]
ax.hist(observed_velocities, bins=40, density=True, alpha=0.4, label='Observed', color='gray')
solver.plot_result(ax=ax, true_intrinsic=true_pdf_ex3)
ax.plot(centers[peaks], mean_pdf[peaks], 'r*', markersize=15, label=f'Detected Peaks ({len(peaks)})')
ax.set_title('Deconvolution: Resolves Both Components')
ax.legend()
ax.grid(True, alpha=0.2)

ax = axes[2]
ax.hist(v1, bins=25, alpha=0.4, label=f'True Pop. A (n={n1})', density=True)
ax.hist(v2, bins=25, alpha=0.4, label=f'True Pop. B (n={n2})', density=True)
ax.plot(x, true_pdf_ex3(x), 'k--', linewidth=2, label='True Combined')
ax.plot(centers, mean_pdf, 'g-', linewidth=2, label='Inferred')
ax.set_xlabel('Velocity (km/s)')
ax.set_ylabel('Probability Density')
ax.set_title('True Components vs Inferred')
ax.legend()
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('images/example_multicomponent.png', dpi=150, bbox_inches='tight')
plt.close()

print("  ✓ Example 3 complete (3 plots)")

# ============================================================================
# EXAMPLE 4: Real-World Workflow
# ============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 4: Real-World Workflow with Quality Checks")
print("=" * 70)

np.random.seed(789)

# Generate data
true_a, true_loc, true_scale = 2, 5, 8
true_velocities = skewnorm.rvs(a=true_a, loc=true_loc, scale=true_scale, size=400)

errors = np.random.uniform(2.0, 5.0, 400)
observed_velocities = true_velocities + np.random.normal(0, errors)

def true_pdf_ex4(x):
    return skewnorm.pdf(x, a=true_a, loc=true_loc, scale=true_scale)

# --- Plot 4a: Data generation visualization ---
print("  → Generating ex4_data_generation.png...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.hist(true_velocities, bins=40, alpha=0.5, label='True (skewed)', density=True)
ax1.hist(observed_velocities, bins=40, alpha=0.5, label='Observed', density=True)
x = np.linspace(-10, 40, 200)
ax1.plot(x, skewnorm.pdf(x, a=true_a, loc=true_loc, scale=true_scale), 
         'k--', linewidth=2, label='True PDF')
ax1.set_xlabel('Velocity (km/s)')
ax1.set_ylabel('Probability Density')
ax1.set_title('Asymmetric Distribution (Positive Skew)')
ax1.legend()
ax1.grid(True, alpha=0.2)

ax2.scatter(observed_velocities, errors, alpha=0.4, s=15)
ax2.axhline(np.median(errors), color='red', linestyle='--', linewidth=2, label='Median error')
ax2.set_xlabel('Observed Velocity (km/s)')
ax2.set_ylabel('Measurement Error (km/s)')
ax2.set_title('Heteroscedastic Errors')
ax2.legend()
ax2.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig('images/ex4_data_generation.png', dpi=150, bbox_inches='tight')
plt.close()

# Grid setup
data_min, data_max = np.percentile(observed_velocities, [1, 99])
data_range = data_max - data_min
grid_center = np.median(observed_velocities)
grid_width = data_range * 1.5
typical_error = np.median(errors)
n_bins = int(grid_width / (2 * typical_error))
n_bins = np.clip(n_bins, 30, 100)

naive_mean = np.mean(observed_velocities)
naive_std = np.std(observed_velocities)

# Run inference
solver = KinematicSolver()
solver.setup_grid(center=grid_center, width=grid_width, n_bins=n_bins)
solver.add_data(vel=observed_velocities, err=errors)
samples = solver.run(num_warmup=1000, num_samples=2000, gpu=False)

# --- Plot 4b: Posterior moments ---
print("  → Generating ex4_posterior_moments.png...")

# Compute moments for each posterior sample
n_posterior = samples['intrinsic_pdf'].shape[0]
mean_samples = np.zeros(n_posterior)
std_samples = np.zeros(n_posterior)
skew_samples = np.zeros(n_posterior)

for i in range(n_posterior):
    m = compute_moments(samples['intrinsic_pdf'][i:i+1], solver.grid['centers'])
    mean_samples[i] = m['mean'][0]
    std_samples[i] = m['std'][0]
    skew_samples[i] = m['skewness'][0]

true_mean_val = skewnorm.mean(a=true_a, loc=true_loc, scale=true_scale)
true_std_val = skewnorm.std(a=true_a, loc=true_loc, scale=true_scale)
true_skew_val = skewnorm.stats(a=true_a, loc=true_loc, scale=true_scale, moments='s')

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

ax = axes[0]
ax.hist(mean_samples, bins=40, alpha=0.7, edgecolor='black', color='steelblue')
ax.axvline(true_mean_val, color='red', linestyle='--', linewidth=2.5, label=f'True: {true_mean_val:.1f}')
ax.axvline(np.mean(mean_samples), color='green', linewidth=2, label=f'Inferred: {np.mean(mean_samples):.1f}')
ax.set_xlabel('Mean Velocity (km/s)')
ax.set_ylabel('Posterior Samples')
ax.set_title('Posterior: Mean')
ax.legend()
ax.grid(True, alpha=0.2)

ax = axes[1]
ax.hist(std_samples, bins=40, alpha=0.7, edgecolor='black', color='coral')
ax.axvline(true_std_val, color='red', linestyle='--', linewidth=2.5, label=f'True: {true_std_val:.1f}')
ax.axvline(np.mean(std_samples), color='green', linewidth=2, label=f'Inferred: {np.mean(std_samples):.1f}')
ax.set_xlabel('Dispersion (km/s)')
ax.set_ylabel('Posterior Samples')
ax.set_title('Posterior: Dispersion')
ax.legend()
ax.grid(True, alpha=0.2)

ax = axes[2]
ax.hist(skew_samples, bins=40, alpha=0.7, edgecolor='black', color='gold')
ax.axvline(true_skew_val, color='red', linestyle='--', linewidth=2.5, label=f'True: {true_skew_val:.2f}')
ax.axvline(np.mean(skew_samples), color='green', linewidth=2, label=f'Inferred: {np.mean(skew_samples):.2f}')
ax.set_xlabel('Skewness')
ax.set_ylabel('Posterior Samples')
ax.set_title('Posterior: Skewness')
ax.legend()
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('images/ex4_posterior_moments.png', dpi=150, bbox_inches='tight')
plt.close()

# --- Plot 4c: Final publication figure ---
print("  → Generating example_complete_workflow.png (final figure)...")
centers = solver.grid['centers']
pdf_density = samples['intrinsic_pdf'] / solver.grid['width']
lower_90 = np.percentile(pdf_density, 5, axis=0)
upper_90 = np.percentile(pdf_density, 95, axis=0)

fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

ax1 = fig.add_subplot(gs[0:2, :])
x_plot = np.linspace(data_min - 5, data_max + 5, 200)
ax1.plot(x_plot, np.exp(-0.5*((x_plot-naive_mean)/naive_std)**2)/(naive_std*np.sqrt(2*np.pi)),
         'r:', linewidth=2, alpha=0.7, label='Naive Fit')
solver.plot_result(ax=ax1, true_intrinsic=true_pdf_ex4)
ax1.hist(observed_velocities, bins=50, density=True, alpha=0.3, 
         color='gray', label='Observed (with errors)', zorder=0)
ax1.set_xlabel('Velocity (km/s)', fontsize=12)
ax1.set_ylabel('Probability Density', fontsize=12)
ax1.set_title('Inferred Line-of-Sight Velocity Distribution', fontsize=14)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.2)

ax2 = fig.add_subplot(gs[2, 0])
ax2.scatter(observed_velocities, errors, alpha=0.3, s=10)
ax2.axhline(np.median(errors), color='red', linestyle='--', alpha=0.5, label='Median')
ax2.set_xlabel('Observed Velocity (km/s)', fontsize=10)
ax2.set_ylabel('Measurement Error (km/s)', fontsize=10)
ax2.set_title('Heteroscedastic Errors', fontsize=11)
ax2.legend()
ax2.grid(True, alpha=0.3)

ax3 = fig.add_subplot(gs[2, 1])
ax3.fill_between(centers, lower_90, upper_90, alpha=0.3, color='steelblue', label='90% CI')
ax3.plot(centers, np.mean(pdf_density, axis=0), 'b-', linewidth=1.5, label='Mean')
ax3.set_xlabel('Velocity (km/s)', fontsize=10)
ax3.set_ylabel('Probability Density', fontsize=10)
ax3.set_title('90% Credible Interval', fontsize=11)
ax3.legend()
ax3.grid(True, alpha=0.2)

plt.savefig('images/example_complete_workflow.png', dpi=150, bbox_inches='tight')
plt.close()

print("  ✓ Example 4 complete (3 plots)")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 70)
print("✓ ALL PLOTS GENERATED SUCCESSFULLY")
print("=" * 70)
print("\nGenerated files in docs/images/:")
print("  Example 1 (Basic Gaussian):")
print("    - ex1_data_generation.png")
print("    - ex1_posterior_analysis.png")
print("    - example_gaussian.png")
print("\n  Example 2 (High Noise):")
print("    - ex2_data_generation.png")
print("    - ex2_deconvolution_process.png")
print("    - example_deconvolution.png")
print("\n  Example 3 (Bimodal):")
print("    - ex3_data_generation.png")
print("    - ex3_posterior_uncertainty.png")
print("    - example_multicomponent.png")
print("\n  Example 4 (Real Workflow):")
print("    - ex4_data_generation.png")
print("    - ex4_posterior_moments.png")
print("    - example_complete_workflow.png")
print("\nTotal: 12 images")
print("=" * 70)
