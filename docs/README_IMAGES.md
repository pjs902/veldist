# Generating Documentation Images

This directory contains scripts to generate all the example images used in the documentation.

## Quick Start

To regenerate all example images:

```bash
cd docs
python generate_all_example_images.py
```

This will create 12 images in `docs/images/`:
- 8 intermediate visualization plots (data generation and posterior analysis)
- 4 final comparison plots

## What Gets Generated

### Example 1: Basic Gaussian Recovery
- `ex1_data_generation.png` - True vs observed distributions + noise
- `ex1_posterior_analysis.png` - Posterior dispersion, smoothness, and PDF samples
- `example_gaussian.png` - Final naive vs deconvolution comparison

### Example 2: High Noise Deconvolution
- `ex2_data_generation.png` - High noise case visualization
- `ex2_deconvolution_process.png` - Deconvolution process and posterior width
- `example_deconvolution.png` - Final three-panel comparison

### Example 3: Bimodal Distribution
- `ex3_data_generation.png` - Two populations separately and combined
- `ex3_posterior_uncertainty.png` - Posterior samples and peak locations
- `example_multicomponent.png` - Final multi-component recovery

### Example 4: Real-World Workflow
- `ex4_data_generation.png` - Skewed distribution with heteroscedastic errors
- `ex4_posterior_moments.png` - Posteriors for mean, dispersion, and skewness
- `example_complete_workflow.png` - Final publication-quality figure

## Runtime

The script takes approximately 2-3 minutes to run as it performs MCMC inference for all 4 examples.

## Requirements

All dependencies are already in the project requirements:
- numpy
- matplotlib
- scipy
- veldist (from src/)

## Updating Images

If you modify the examples in `examples.md`, update the corresponding code in 
`generate_all_example_images.py` to keep the images in sync.
