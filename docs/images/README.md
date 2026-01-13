# Example Images

This directory contains images for the documentation examples.

## Generating Images

To regenerate all example images, run from the `docs/` directory:

```bash
python generate_example_images.py
```

This will create:

- `example_gaussian.png` - Basic Gaussian recovery example
- `example_deconvolution.png` - High noise deconvolution example
- `example_multicomponent.png` - Multi-component system example
- `example_complete_workflow.png` - Complete analysis workflow example

## Note

These images are automatically generated from the code examples in `examples.md`. They can be regenerated at any time to ensure documentation stays up-to-date with the codebase.
