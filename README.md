# veldist

Infer velocity distributions from noisy stellar kinematics using Bayesian methods.

More generally, do unbiased, non-parametric density estimation.

## Installation

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

## License

MIT License - see LICENSE file for details.
