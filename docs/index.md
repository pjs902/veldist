# veldist

Infer velocity distributions from noisy stellar kinematics.

More generally, do unbiased, non-parametric density estimation.

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

## Batch workflow (Voronoi bins) and Dynamite output

Use `fit_all_bins` to run the full inference pipeline across many bins, then
export directly to Dynamite `BayesLOSVD` files with
`write_dynamite_kinematics`.

```python
from veldist import fit_all_bins, write_dynamite_kinematics

solvers = fit_all_bins(
    bin_data_list,
    grid_kwargs={"center": 0.0, "width": 600.0, "n_bins": 60},
    run_kwargs={"num_warmup": 500, "num_samples": 1000, "gpu": False, "seed": 5567},
    min_stars=10,
)

write_dynamite_kinematics(
    solvers=solvers,
    output_dir="dynamite_input",
    voronoi_bin_metadata=voronoi_bin_metadata,
    bin_flux_mode="nstars",
)
```

Important details:

- `fit_all_bins` skips under-populated bins (`None`) and those are masked automatically in `bins.dat`.
- `clip_uncertainties()` is applied to enforce non-zero LOSVD uncertainties for robust downstream NNLS behavior.
- `truncate_losvd()` is available as an optional diagnostic fix for tail contamination; it is not part of the default batch pipeline.
- The Dynamite writer requires `astropy`.

For complete, runnable examples covering various use cases, see the [Examples](examples) page.

```{toctree}
:hidden:
:caption: Documentation

theory
examples
api
```

## License

MIT License - see LICENSE file for details.
