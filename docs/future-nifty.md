# Future direction: NIFTy.re for 3D / 6D

Status: **not planned.** Evaluated 2026-10-16 and parked. Revisit only if we
go beyond 2D velocity space.

NIFTy.re (Edenhofer et al. 2024, JOSS 9, 6593; <https://github.com/nifty-ppl/nifty>,
read at commit `1f3cb28`) is a JAX library offering FFT/ICR Gaussian-process
priors with a learned power spectrum (`CorrelatedFieldMaker`), plus MGVI/geoVI
variational inference. NUTS is available through BlackJAX. The `nifty/re`
files are GPL-2+ OR BSD-2, which is compatible with our MIT license.

## Why not now (1D/2D)

- **No engine gain.** Our likelihood `sum log(M @ softmax(x))` is a per-star
  mixture and matches none of NIFTy's built-in likelihoods. A custom
  `Likelihood` with only `energy` supports only `blackjax_nuts`, which is the
  same algorithm we already run in NumPyro.
- **VI risks calibration.** MGVI/geoVI need a Fisher metric. Ours has no
  closed form; the empirical Fisher (per-star gradient `M_i / (M_i·π)` chained
  through softmax) is the stand-in. Even then, VI posteriors are known to
  come out too narrow, which conflicts with the SBC and coverage gates.
- **Our problems are small** (≤ ~840 cells). NUTS copes with that. The
  bottleneck has been memory and draw volume, not sampler speed.

## The Gaussian-core prior ports cleanly

NIFTy writes every prior as a deterministic map from standard-normal
latents, which is the shape our prior already has:

- `v0`, `s0`, `rho0`: `NormalPrior`, `LogNormalPrior` and `UniformPrior` are
  built in. `sigma3 ~ Exponential` takes one line by inverse CDF,
  `-log(norm.sf(xi)) / rate`.
- Core (Gaussian cell mass, Gauss-Legendre + `logsumexp`) and deviation (RW3
  cumsums / GMRF solve + quadratic null-space projection) are plain JAX and
  move over unchanged. The mass invariant holds as long as the field stays
  latent under `softmax`.
- Optionally, the deviation could become a correlated field with a learned
  spectrum. That field is built on a Hartley transform and so treats the grid
  as periodic: velocity grids need padding, or ICR, to stop one tail wrapping
  into the other. The null-space projection is still needed.

## Where it would matter

| | Cells (K≈20–30) | What breaks in veldist |
|---|---|---|
| 2D velocity (now) | ~840 | nothing |
| 3D velocity per spatial bin | 8k–27k | dense `L` (n², ~6 GB at 27k), dense M, slow NUTS |
| 6D phase space | 6×10⁷+ | everything; GPU VI is the only option |

NIFTy's contributions at that scale: O(n log n) priors; ICR for non-periodic
grids; outer-product spectra (repeated `add_fluctuations`), which give a
separable space × velocity prior coupling neighbouring spatial bins; and
multi-GPU VI.

Work we would need anyway, whichever library we use:

- Sparse M: each star's kernel touches only ~(few·σ/h)³ cells.
- Position–velocity coupled errors in 6D (a parallax error scales the
  tangential velocity and moves the star in space), which breaks the
  separable precomputed-M design.
- A selection function.
- A calibration strategy when NUTS can't serve as ground truth: SBC at small
  K, then extrapolate.

Suggested order if pursued: 3D velocity per spatial bin with sparse M and an
FFT/ICR prior first, then 6D as a separate project. Baseline to beat: Bovy's
extreme deconvolution, a Gaussian mixture rather than a grid.

None of the above has been run; the empirical-Fisher MGVI idea in particular
is untested.
