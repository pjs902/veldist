"""Pytest configuration shared across the test suite.

Enables JAX's 64-bit mode. Several tests assert numerical properties (e.g.
that a curve is exactly quadratic to within 1e-6) that are below the
precision floor of JAX's default float32, particularly after chained
cumsum/QR operations. This must be set before any JAX array operations run,
so it lives in conftest.py rather than in an individual test module.
"""

import jax

jax.config.update("jax_enable_x64", True)
