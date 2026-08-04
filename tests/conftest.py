"""Pytest configuration shared across the test suite.

Two global settings, both of which must be applied before any JAX array
operation runs — which is why they live here rather than in a test module.

1. **64-bit mode.** Several tests assert numerical properties (e.g. that a
   curve is exactly quadratic to within 1e-6) that are below the precision
   floor of JAX's default float32, particularly after chained cumsum/QR
   operations.

   Note this makes the suite blind by default to float32 failures that
   *production* can hit, since ``KinematicSolver`` runs in float32. A
   design-matrix cancellation bug survived exactly that way. Tests targeting
   numerical robustness should wrap the call in ``jax.enable_x64(False)`` —
   see ``test_design_matrix_is_strictly_positive_in_float32``.

2. **Host device count.** ``KinematicSolver.run`` defaults to
   ``num_chains=4`` (multiple chains are the only source of r_hat). With a
   single visible device those chains run *sequentially*, making every slow
   test about 4x slower for identical results. Requesting the devices up front
   makes them genuinely parallel.

   The device count is frozen when JAX initialises its backend, and merely
   *querying* it (``jax.local_device_count()``) is enough to trigger that — so
   this call must come before any other JAX use, and must not be guarded by a
   device-count check.
"""

import numpyro

from veldist.veldist import NUM_CHAINS

numpyro.set_host_device_count(NUM_CHAINS)

import jax  # noqa: E402  (must follow the device-count request)

jax.config.update("jax_enable_x64", True)
