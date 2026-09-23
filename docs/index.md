# veldist

Non-parametric Bayesian inference of the line-of-sight velocity distribution
(LOSVD) from discrete stellar velocities.

`veldist` takes individual stellar velocities and their measurement errors
and returns a posterior for the intrinsic LOSVD as a histogram. The amount of
smoothing is inferred from the data rather than set by hand. The default
prior (`gaussian_core`) falls back to a Gaussian wherever the data say
little, instead of to a flat histogram; [Methodology](theory) explains why
that matters.

It is built for resolved stellar kinematics: globular clusters, dwarf
galaxies and the outer halos of nearby galaxies. It includes a batch
pipeline for Voronoi-binned data and a writer for Dynamite's `BayesLOSVD`
input format.

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

`fit_all_bins` runs the full inference on every bin, and
`write_dynamite_kinematics` writes the results as Dynamite `BayesLOSVD`
files.

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

Bins with fewer than `min_stars` stars come back as `None` and are masked in
`bins.dat`. The Dynamite writer needs `astropy`. [Examples](examples) has a
complete walkthrough, including kinematic maps.

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
