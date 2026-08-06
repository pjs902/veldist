# veldist

Non-parametric Bayesian inference of the line-of-sight velocity distribution
(LOSVD) from discrete stellar velocities.

Given individual stellar velocities and per-star measurement uncertainties,
`veldist` recovers the intrinsic LOSVD as a histogram posterior, marginalising
over a smoothing hyperparameter that adapts to the signal-to-noise of the
data. The default prior (`gaussian_core`) shrinks toward a Gaussian LOSVD
wherever the data are uninformative, rather than toward a flat histogram;
see [Methodology](theory) for why that matters. Designed for resolved
stellar kinematics in globular clusters, dwarf galaxies, and the extended
halos of nearby galaxies. Includes a batch pipeline for Voronoi-binned data
and a writer for the Dynamite `BayesLOSVD` input format.

## Installation

(pypi coming soon)

```bash
pip install -e .
```

Or for development:

```bash
git clone https://github.com/pjs902/veldist.git
cd veldist
pip install -e .
```

## Quick Start

```python
import veldist
from veldist import KinematicSolver

# Make 4 CPU devices visible, so the 4 sampling chains run in parallel.
# Must come before any other JAX work: the device count is fixed when JAX
# initialises its backend. Results are identical without it, just ~4x slower.
veldist.set_host_devices(4)

# Initialize solver
solver = KinematicSolver()

# Set up velocity grid
solver.setup_grid(center=0.0, width=100.0, n_bins=50)

# Add your observational data
solver.add_data(vel=observed_velocities, err=velocity_errors)

# Run inference
samples = solver.run(num_warmup=500, num_samples=3000)

# Plot results
solver.plot_result()
```

## Batch workflow (Voronoi bins) and Dynamite output

Use `fit_all_bins` to run the full inference pipeline across many bins, then
export directly to Dynamite `BayesLOSVD` files with
`write_dynamite_kinematics`.

```python
from veldist import fit_all_bins, write_dynamite_kinematics

solvers = fit_all_bins(
    bin_data_list,
    grid_kwargs={"center": 0.0, "width": 600.0, "n_bins": 60},
    run_kwargs={"num_warmup": 500, "num_samples": 3000, "gpu": False, "seed": 5567},
    min_stars=10,
)

write_dynamite_kinematics(
    solvers=solvers,
    output_dir="dynamite_input",
    voronoi_bin_metadata=voronoi_bin_metadata,
    bin_flux_mode="nstars",
)
```

Bins with fewer than `min_stars` stars are returned as `None` and written as
masked pixels in `bins.dat`.  The Dynamite writer requires `astropy`.  See
the [Examples](examples) page for a complete walkthrough including spatial
map extraction.

```{toctree}
:hidden:
:caption: Documentation

theory
examples
validation
api
```

## License

MIT License - see LICENSE file for details.
