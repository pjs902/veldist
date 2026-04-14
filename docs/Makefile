# Minimal makefile for Sphinx documentation

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Figure scripts that don't require inference (fast; scipy + matplotlib only)
FIG_IMAGES_QUICK = \
	images/fig_prior.png \
	images/fig_design_matrix.png \
	images/fig_summary_metrics.png \
	images/fig_bimodal.png

# Figure scripts that require a live KinematicSolver (JAX + NumPyro)
FIG_IMAGES_INF = \
	images/fig_deconvolution.png \
	images/fig_kin_maps.png

.PHONY: help figures figures-quick html html-quick clean Makefile

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Generate all figures including inference (requires JAX/NumPyro)
figures: $(FIG_IMAGES_QUICK) $(FIG_IMAGES_INF)
	@echo "All figures generated."

# Generate only the no-inference figures (no JAX needed)
figures-quick: $(FIG_IMAGES_QUICK)
	@echo "Quick figures done.  Run 'make figures' to also build inference figures."

images/fig_prior.png: fig_prior.py
	python $<

images/fig_design_matrix.png: fig_design_matrix.py
	python $<

images/fig_summary_metrics.png: fig_summary_metrics.py
	python $<

images/fig_bimodal.png: fig_bimodal.py
	python $<

images/fig_deconvolution.png: fig_deconvolution.py
	python $<

images/fig_kin_maps.png: fig_kin_maps.py
	python $<

# Build HTML docs; generates all figures first
html: figures
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Build HTML docs without the inference figures
html-quick: figures-quick
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	@$(SPHINXBUILD) -M clean "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	rm -f $(FIG_IMAGES_QUICK) $(FIG_IMAGES_INF)
	@echo "Figures and build artefacts removed."

# Tell Make that .py files are source files; they don't need to be built.
# Without this, the catch-all below would try to run Sphinx on them.
%.py: ;

# Catch-all for other Sphinx targets (latexpdf, etc.)
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
