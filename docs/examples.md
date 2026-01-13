# End-to-End Examples

This page provides complete, runnable examples demonstrating how to use `veldist` for various scientific use cases.

## Example 1: Basic Gaussian Distribution Recovery

This example demonstrates the core capability of `veldist`: recovering the intrinsic velocity distribution from noisy observations. We compare the naive approach (fitting a Gaussian to the observed data) with proper Bayesian deconvolution.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from veldist import KinematicSolver

# Set random seed for reproducibility
np.random.seed(42)

# ========================================
# 1. Generate Synthetic Data
# ========================================

# True distribution parameters
true_mean = 0.0  # km/s
true_std = 10.0  # km/s
n_stars = 500

# Generate true velocities from a Gaussian
true_velocities = np.random.normal(true_mean, true_std, n_stars)

# Add observational errors
measurement_errors = np.ones(n_stars) * 2.0  # km/s
observed_velocities = true_velocities + np.random.normal(0, measurement_errors)

print(f"Generated {n_stars} stars")
print(f"True mean: {true_mean:.1f} km/s, True std: {true_std:.1f} km/s")
print(f"Observed mean: {np.mean(observed_velocities):.1f} km/s")
print(f"Observed std: {np.std(observed_velocities):.1f} km/s")

# ========================================
# 2. Naive Approach: Fit Gaussian to Observed Data
# ========================================

# Simple Gaussian fit (ignores measurement errors)
naive_mean = np.mean(observed_velocities)
naive_std = np.std(observed_velocities)

print(f"\n=== Naive Gaussian Fit ===")
print(f"Fitted mean: {naive_mean:.2f} km/s")
print(f"Fitted std:  {naive_std:.2f} km/s")

# ========================================
# 3. Deconvolution with veldist
# ========================================

solver = KinematicSolver()
solver.setup_grid(center=0.0, width=100.0, n_bins=50)
solver.add_data(vel=observed_velocities, err=measurement_errors)

samples = solver.run(num_warmup=500, num_samples=1000, gpu=False)

# ========================================
# 4. Analyze Results
# ========================================

from veldist import compute_moments

moments = compute_moments(samples['intrinsic_pdf'], solver.grid['centers'])

print(f"\n=== Bayesian Deconvolution ===")
print(f"Inferred mean: {moments['mean'][0]:.2f} ± {moments['mean'][1]:.2f} km/s")
print(f"Inferred std:  {moments['std'][0]:.2f} ± {moments['std'][1]:.2f} km/s")
print(f"\n=== True Values ===")
print(f"True mean:     {true_mean:.2f} km/s")
print(f"True std:      {true_std:.2f} km/s")

# ========================================
# 5. Visualize Comparison
# ========================================

# Define true PDF function for comparison
def true_pdf(x):
    return np.exp(-0.5 * ((x - true_mean) / true_std)**2) / (true_std * np.sqrt(2 * np.pi))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left panel: Naive approach
ax = axes[0]
x_plot = np.linspace(-40, 40, 200)
naive_pdf = np.exp(-0.5 * ((x_plot - naive_mean) / naive_std)**2) / (naive_std * np.sqrt(2 * np.pi))

ax.hist(observed_velocities, bins=30, density=True, alpha=0.4, 
        label='Observed Data', color='gray')
ax.plot(x_plot, naive_pdf, 'r-', linewidth=2, label=f'Naive Fit (σ={naive_std:.1f})')
ax.plot(x_plot, true_pdf(x_plot), 'k--', linewidth=2, label=f'True (σ={true_std:.1f})')
ax.set_xlabel('Velocity (km/s)')
ax.set_ylabel('Probability Density')
ax.set_title('Naive Approach: Gaussian Fit to Observed Data')
ax.legend()
ax.grid(True, alpha=0.2)

# Right panel: Deconvolution
ax = axes[1]
ax.hist(observed_velocities, bins=30, density=True, alpha=0.4, 
        label='Observed Data', color='gray')
solver.plot_result(ax=ax, true_intrinsic=true_pdf)
ax.set_title('Bayesian Deconvolution: Recovers True Distribution')
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('docs/images/example_gaussian.png', dpi=150)
plt.show()
```

### Expected Output

**Console output:**

```
Generated 500 stars
True mean: 0.0 km/s, True std: 10.0 km/s
Observed mean: 0.1 km/s
Observed std: 10.2 km/s

=== Naive Gaussian Fit ===
Fitted mean: 0.12 km/s
Fitted std:  10.20 km/s

Computing Design Matrix for 500 stars...
Matrix ready. Shape: (500, 50)

Starting NUTS MCMC...
Inference Complete.

=== Bayesian Deconvolution ===
Inferred mean: 0.12 ± 0.45 km/s
Inferred std:  9.98 ± 0.35 km/s

=== True Values ===
True mean:     0.00 km/s
True std:      10.00 km/s
```

**Visual output:**

![Example 1: Gaussian Recovery](images/example_gaussian.png)

The left panel shows the naive approach: fitting a Gaussian directly to observed data (red line) overestimates the width due to measurement errors. The right panel shows Bayesian deconvolution correctly recovering the true intrinsic distribution (green matches the black dashed true curve).

---

## Example 2: Deconvolution with High Noise

This example demonstrates `veldist`'s key strength: recovering the intrinsic distribution when measurement errors are comparable to or larger than the intrinsic spread.

```python
import numpy as np
import matplotlib.pyplot as plt
from veldist import KinematicSolver, compute_moments

np.random.seed(123)

# ========================================
# 1. Generate Data with High Noise
# ========================================

# Narrow intrinsic distribution
true_mean = 0.0
true_std = 5.0  # Small intrinsic width
n_stars = 1000

true_velocities = np.random.normal(true_mean, true_std, n_stars)

# Large measurement errors (comparable to intrinsic width)
measurement_errors = np.ones(n_stars) * 7.0  # Larger than intrinsic!
observed_velocities = true_velocities + np.random.normal(0, measurement_errors)

# Compare spreads
obs_std = np.std(observed_velocities)
print(f"True intrinsic dispersion: {true_std:.1f} km/s")
print(f"Measurement error:         {measurement_errors[0]:.1f} km/s")
print(f"Observed dispersion:       {obs_std:.1f} km/s")
print(f"Noise is {measurement_errors[0]/true_std:.1f}x larger than signal!")

# ========================================
# 2. Naive vs Deconvolution
# ========================================

# Naive approach
naive_mean = np.mean(observed_velocities)
naive_std = np.std(observed_velocities)

# Define true PDF function for comparison
def true_pdf(x):
    return np.exp(-0.5 * ((x - true_mean) / true_std)**2) / (true_std * np.sqrt(2 * np.pi))

# Deconvolution
solver = KinematicSolver()
solver.setup_grid(center=0.0, width=100.0, n_bins=50)
solver.add_data(vel=observed_velocities, err=measurement_errors)
samples = solver.run(num_warmup=500, num_samples=1000, gpu=False)

moments = compute_moments(samples['intrinsic_pdf'], solver.grid['centers'])
deconvolved_std = moments['std'][0]

print(f"\n=== Dispersion Recovery ===")
print(f"Naive (observed):    {naive_std:.2f} km/s")
print(f"Deconvolved:         {deconvolved_std:.2f} ± {moments['std'][1]:.2f} km/s")
print(f"True intrinsic:      {true_std:.2f} km/s")
print(f"\nError reduction: {abs(naive_std - true_std):.2f} → "
      f"{abs(deconvolved_std - true_std):.2f} km/s")

# ========================================
# 3. Visualization
# ========================================

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# Panel 1: Observed data with naive fit
ax = axes[0]
ax.hist(observed_velocities, bins=40, density=True, alpha=0.6, 
        color='gray', label='Observed')
x = np.linspace(-30, 30, 200)
ax.plot(x, np.exp(-0.5*x**2/naive_std**2)/(naive_std*np.sqrt(2*np.pi)), 
        'r-', linewidth=2, label=f'Naive Fit (σ={naive_std:.1f})')
ax.plot(x, true_pdf(x), 'k--', linewidth=2, label=f'True (σ={true_std:.1f})')
ax.set_xlabel('Velocity (km/s)')
ax.set_ylabel('Probability Density')
ax.set_title('Observed Data: Naive Fit')
ax.legend()
ax.grid(True, alpha=0.2)

# Panel 2: Deconvolved result
ax = axes[1]
ax.hist(observed_velocities, bins=40, density=True, alpha=0.4, 
        label='Observed', color='gray')
solver.plot_result(ax=ax, true_intrinsic=true_pdf)
ax.set_title('Deconvolved Distribution')
ax.grid(True, alpha=0.2)

# Panel 3: Direct comparison
ax = axes[2]
bins = solver.grid['centers']
pdf_samples = samples['intrinsic_pdf'] / solver.grid['width']
mean_pdf = np.mean(pdf_samples, axis=0)
true_pdf_vals = true_pdf(bins)

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
plt.savefig('docs/images/example_deconvolution.png', dpi=150)
plt.show()
```

### Visual Output

![Example 2: Deconvolution with High Noise](images/example_deconvolution.png)

This figure demonstrates the power of deconvolution. The left panel shows naive Gaussian fitting to observed data (red) is heavily biased by measurement errors. The middle panel shows the deconvolved result successfully recovering the narrower true width. The right panel directly compares all three: naive fit (red dotted), inferred (green), and true (black dashed), clearly showing deconvolution works even when noise > signal.

---

## Example 3: Bimodal Distribution (Two Separated Gaussians)

This example shows how `veldist` can recover complex, non-Gaussian distributions with multiple components—common in stellar systems with multiple populations.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from veldist import KinematicSolver

np.random.seed(456)

# ========================================
# 1. Create Bimodal Distribution
# ========================================

# Two well-separated Gaussian components
# Component 1: Population A
n1 = 400
mean1, std1 = -15.0, 6.0

# Component 2: Population B
n2 = 300
mean2, std2 = 20.0, 8.0

# Generate samples
v1 = np.random.normal(mean1, std1, n1)
v2 = np.random.normal(mean2, std2, n2)

true_velocities = np.concatenate([v1, v2])
n_stars = len(true_velocities)

# Measurement errors
errors = np.random.uniform(3, 6, n_stars)
observed_velocities = true_velocities + np.random.normal(0, errors)

print(f"Component 1: {n1} stars at μ={mean1:.1f}, σ={std1:.1f} km/s")
print(f"Component 2: {n2} stars at μ={mean2:.1f}, σ={std2:.1f} km/s")
print(f"Total: {n_stars} stars")
print(f"Measurement errors: {errors.min():.1f} - {errors.max():.1f} km/s")

# ========================================
# 2. Naive vs Deconvolution
# ========================================

# Naive: single Gaussian fit
naive_mean = np.mean(observed_velocities)
naive_std = np.std(observed_velocities)

# Define true PDF function (mixture of 2 Gaussians)
def true_pdf(x):
    pdf1 = np.exp(-0.5 * ((x - mean1) / std1)**2) / (std1 * np.sqrt(2 * np.pi))
    pdf2 = np.exp(-0.5 * ((x - mean2) / std2)**2) / (std2 * np.sqrt(2 * np.pi))
    w1, w2 = n1 / n_stars, n2 / n_stars
    return w1 * pdf1 + w2 * pdf2

# Deconvolution
solver = KinematicSolver()
solver.setup_grid(center=2.5, width=80.0, n_bins=60)
solver.add_data(vel=observed_velocities, err=errors)
samples = solver.run(num_warmup=500, num_samples=1000, gpu=False)

# Find peaks in the inferred distribution
pdf_density = samples['intrinsic_pdf'] / solver.grid['width']
mean_pdf = np.mean(pdf_density, axis=0)
centers = solver.grid['centers']
peaks, properties = find_peaks(mean_pdf, distance=5, prominence=0.003)

print(f"\n=== Detected {len(peaks)} components ===")
for i, peak_idx in enumerate(peaks, 1):
    peak_vel = centers[peak_idx]
    print(f"Component {i}: velocity ≈ {peak_vel:.1f} km/s")

# ========================================
# 3. Visualization
# ========================================

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# Panel 1: Observed data with naive fit
ax = axes[0]
ax.hist(observed_velocities, bins=40, density=True, alpha=0.5, 
        color='gray', label='Observed')
x = np.linspace(-40, 50, 200)
ax.plot(x, np.exp(-0.5*((x-naive_mean)/naive_std)**2)/(naive_std*np.sqrt(2*np.pi)), 
        'r-', linewidth=2, label='Naive Fit (1 Gaussian)')
ax.set_xlabel('Velocity (km/s)')
ax.set_ylabel('Probability Density')
ax.set_title('Naive Approach: Misses Bimodality')
ax.legend()
ax.grid(True, alpha=0.2)

# Panel 2: Deconvolved result
ax = axes[1]
ax.hist(observed_velocities, bins=40, density=True, alpha=0.4, 
        label='Observed', color='gray')
solver.plot_result(ax=ax, true_intrinsic=true_pdf)
ax.plot(centers[peaks], mean_pdf[peaks], 'r*', markersize=15, 
        label=f'Detected Peaks ({len(peaks)})')
ax.set_title('Deconvolution: Resolves Both Components')
ax.legend()
ax.grid(True, alpha=0.2)

# Panel 3: Components shown separately
ax = axes[2]
# Show individual components
ax.hist(v1, bins=25, alpha=0.4, label=f'True Pop. A (n={n1})', density=True)
ax.hist(v2, bins=25, alpha=0.4, label=f'True Pop. B (n={n2})', density=True)
ax.plot(x, true_pdf(x), 'k--', linewidth=2, label='True Combined')
ax.plot(centers, mean_pdf, 'g-', linewidth=2, label='Inferred')
ax.set_xlabel('Velocity (km/s)')
ax.set_ylabel('Probability Density')
ax.set_title('True Components vs Inferred')
ax.legend()
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('docs/images/example_multicomponent.png', dpi=150)
plt.show()
```

### Visual Output

![Example 3: Bimodal Distribution](images/example_multicomponent.png)

**Left panel:** Naive single-Gaussian fit completely misses the bimodal nature of the data. **Middle panel:** Deconvolution successfully resolves both components (marked with red stars). **Right panel:** Comparison showing the true underlying populations and how well the inferred distribution (green) matches the true combined distribution (black dashed).

---

## Example 4: Real-World Workflow with Quality Checks

This example demonstrates a complete analysis pipeline with diagnostic checks and uncertainty quantification.

```python
import numpy as np
import matplotlib.pyplot as plt
from veldist import KinematicSolver, compute_moments

np.random.seed(789)

# ========================================
# 1. Simulate "Real" Data
# ========================================

# Asymmetric distribution (e.g., infalling stream)
from scipy.stats import skewnorm

# True distribution parameters
true_a, true_loc, true_scale = 2, 5, 8
true_velocities = skewnorm.rvs(a=true_a, loc=true_loc, scale=true_scale, size=400)

# Define true PDF function for comparison
def true_pdf(x):
    return skewnorm.pdf(x, a=true_a, loc=true_loc, scale=true_scale)

# Realistic error model (depends on magnitude, etc.)
# Better measurements for slow-moving stars
errors = 2.0 + 0.1 * np.abs(true_velocities) + np.random.uniform(0, 1, 400)
observed_velocities = true_velocities + np.random.normal(0, errors)

print("=== Dataset Summary ===")
print(f"N stars: {len(true_velocities)}")
print(f"Velocity range: {observed_velocities.min():.1f} to "
      f"{observed_velocities.max():.1f} km/s")
print(f"Median error: {np.median(errors):.2f} km/s")
print(f"Error range: {errors.min():.1f} - {errors.max():.1f} km/s")

# ========================================
# 2. Pre-Analysis Checks
# ========================================

# Check for outliers
q1, q3 = np.percentile(observed_velocities, [25, 75])
iqr = q3 - q1
outliers = (observed_velocities < q1 - 3*iqr) | (observed_velocities > q3 + 3*iqr)
print(f"\nOutliers detected: {outliers.sum()} stars")

if outliers.sum() > 0:
    print("Consider removing or flagging outliers")

# Error budget
signal = np.std(observed_velocities)
noise = np.median(errors)
snr = signal / noise
print(f"\nSignal-to-noise: {snr:.2f}")
if snr < 2:
    print("Warning: Low SNR may lead to high uncertainties")

# ========================================
# 3. Grid Selection
# ========================================

# Choose grid based on data range
data_min, data_max = np.percentile(observed_velocities, [1, 99])
data_range = data_max - data_min
grid_center = np.median(observed_velocities)
grid_width = data_range * 1.5  # Add 50% buffer

print(f"\n=== Grid Setup ===")
print(f"Data 99% range: {data_min:.1f} to {data_max:.1f} km/s")
print(f"Grid center: {grid_center:.1f} km/s")
print(f"Grid width: {grid_width:.1f} km/s")

# Choose number of bins
# Rule of thumb: ~3-5 bins per expected feature width
typical_error = np.median(errors)
n_bins = int(grid_width / (2 * typical_error))
n_bins = np.clip(n_bins, 30, 100)  # Keep reasonable
print(f"Number of bins: {n_bins}")

# ========================================
# 4. Naive vs Deconvolution
# ========================================

# Naive approach
naive_mean = np.mean(observed_velocities)
naive_std = np.std(observed_velocities)

# Run inference
solver = KinematicSolver()
solver.setup_grid(center=grid_center, width=grid_width, n_bins=n_bins)
solver.add_data(vel=observed_velocities, err=errors)
samples = solver.run(num_warmup=1000, num_samples=2000, gpu=False)

# ========================================
# 5. Results and Uncertainties
# ========================================

moments = compute_moments(samples['intrinsic_pdf'], solver.grid['centers'])

print("\n=== Results ===")
print(f"Mean velocity: {moments['mean'][0]:.2f} ± {moments['mean'][1]:.2f} km/s")
print(f"Dispersion:    {moments['std'][0]:.2f} ± {moments['std'][1]:.2f} km/s")
print(f"Skewness:      {moments['skewness'][0]:.2f} ± {moments['skewness'][1]:.2f}")
print(f"Kurtosis:      {moments['kurtosis'][0]:.2f} ± {moments['kurtosis'][1]:.2f}")

# Credible intervals
centers = solver.grid['centers']
pdf_density = samples['intrinsic_pdf'] / solver.grid['width']
lower_90 = np.percentile(pdf_density, 5, axis=0)
upper_90 = np.percentile(pdf_density, 95, axis=0)
relative_unc = (upper_90 - lower_90) / (np.mean(pdf_density, axis=0) + 1e-10)

print(f"\nMedian relative uncertainty: {np.median(relative_unc):.1%}")

# ========================================
# 6. Publication-Quality Visualization
# ========================================

fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# Main result
ax1 = fig.add_subplot(gs[0:2, :])
x_plot = np.linspace(data_min - 5, data_max + 5, 200)
ax1.plot(x_plot, np.exp(-0.5*((x_plot-naive_mean)/naive_std)**2)/(naive_std*np.sqrt(2*np.pi)),
         'r:', linewidth=2, alpha=0.7, label='Naive Fit')
solver.plot_result(ax=ax1, true_intrinsic=true_pdf)
ax1.hist(observed_velocities, bins=50, density=True, alpha=0.3, 
         color='gray', label='Observed (with errors)', zorder=0)
ax1.set_xlabel('Velocity (km/s)', fontsize=12)
ax1.set_ylabel('Probability Density', fontsize=12)
ax1.set_title('Inferred Line-of-Sight Velocity Distribution', fontsize=14)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.2)

# Error distribution
ax2 = fig.add_subplot(gs[2, 0])
ax2.scatter(observed_velocities, errors, alpha=0.3, s=10)
ax2.set_xlabel('Observed Velocity (km/s)', fontsize=10)
ax2.set_ylabel('Measurement Error (km/s)', fontsize=10)
ax2.set_title('Heteroscedastic Errors', fontsize=11)
ax2.grid(True, alpha=0.3)

# Uncertainty map
ax3 = fig.add_subplot(gs[2, 1])
ax3.fill_between(centers, lower_90, upper_90, alpha=0.3, color='steelblue')
ax3.plot(centers, np.mean(pdf_density, axis=0), 'b-', linewidth=1.5)
ax3.set_xlabel('Velocity (km/s)', fontsize=10)
ax3.set_ylabel('Probability Density', fontsize=10)
ax3.set_title('90% Credible Interval', fontsize=11)
ax3.grid(True, alpha=0.2)

plt.savefig('docs/images/example_complete_workflow.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n✓ Analysis complete.")
```

### Visual Output

![Example 4: Complete Workflow](images/example_complete_workflow.png)

**Top panel:** Publication-quality plot comparing naive fit (red dotted), inferred distribution (green with credible intervals), and true asymmetric distribution (black dashed). The deconvolution successfully captures the skewness that the naive Gaussian fit misses. **Bottom left:** Heteroscedastic error distribution showing measurement quality varies across the dataset. **Bottom right:** 90% credible interval visualization showing where the inference is most/least certain.
