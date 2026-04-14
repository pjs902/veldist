# veldist

[![Tests](https://github.com/pjs902/veldist/actions/workflows/tests.yml/badge.svg)](https://github.com/pjs902/veldist/actions/workflows/tests.yml)
[![Documentation Status](https://readthedocs.org/projects/veldist/badge/?version=latest)](https://veldist.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

Non-parametric Bayesian inference of the line-of-sight velocity distribution
(LOSVD) from discrete stellar velocities.

Given a set of individual stellar velocities and their per-star measurement
uncertainties, `veldist` recovers the intrinsic LOSVD as a histogram
posterior, marginalising over a smoothing hyperparameter that adapts to the
signal-to-noise of the data.  Output is a full MCMC posterior over the LOSVD
shape, from which scalar summaries ($V$, $\sigma$, skewness, kurtosis, tail
weight) and their uncertainties are derived without additional assumptions.

Designed for resolved stellar kinematics in globular clusters, dwarf
galaxies, and the extended halos of nearby galaxies.  Includes a batch
pipeline for Voronoi-binned IFU-like data and a writer for the Dynamite
`histLOSVD` / BayesLOSVD input format.

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
from veldist import KinematicSolver

# Initialize solver
solver = KinematicSolver()

# Set up velocity grid
solver.setup_grid(center=0.0, width=100.0, n_bins=50)

# Add your observational data
solver.add_data(vel=observed_velocities, err=velocity_errors)

# Run inference
samples = solver.run(num_warmup=500, num_samples=1000)

# Plot results
solver.plot_result()
```

## Batch fitting and Dynamite export

The latest API adds a batch pipeline for Voronoi-binned data and a writer for
Dynamite `BayesLOSVD` input files.

```python
from veldist import fit_all_bins, write_dynamite_kinematics

# 1) Solve all bins (skips bins with too few stars as None)
solvers = fit_all_bins(
    bin_data_list,  # [{'vel': ..., 'err': ...}, ...]
    grid_kwargs={"center": 0.0, "width": 600.0, "n_bins": 60},
    run_kwargs={"num_warmup": 500, "num_samples": 1000, "gpu": False, "seed": 5567},
    min_stars=10,
)

# 2) Optional post-processing per bin (only if needed)
# for s in solvers:
#     if s is not None:
#         s.truncate_losvd(n_sigma=3.0)

# 3) Write Dynamite inputs (ECSV + aperture.dat + bins.dat)
write_dynamite_kinematics(
    solvers=solvers,
    output_dir="dynamite_input",
    voronoi_bin_metadata=voronoi_bin_metadata,
    bin_flux_mode="nstars",  # or "uniform" / "custom"
)
```

Notes:

- `fit_all_bins` uses `seed + bin_index` internally to avoid chain correlation across bins.
- `clip_uncertainties()` is run as part of the batch pipeline and enforces a floor on LOSVD uncertainties.
- `truncate_losvd()` is optional and intended only for diagnosed tail contamination cases.
- `write_dynamite_kinematics` requires `astropy` (now included in project dependencies).

## License

MIT License - see LICENSE file for details.

## Attribution

This will eventually be described in a paper. If this code is useful for your research, please cite the paper (when available) or the GitHub repository.
