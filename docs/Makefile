# Minimal Makefile for Sphinx docs (HTML + LaTeX/PDF)
# Use: make html | make latexpdf

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Turn warnings into errors for the HTML build, and report all of them rather
# than stopping at the first. Without this an autodoc directive that fails to
# import its module is only a warning, so the build stays green while shipping
# an API chapter containing no signatures at all.
STRICTOPTS    = -W --keep-going

.PHONY: help html latexpdf clean

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

html:
	@$(SPHINXBUILD) -b html $(STRICTOPTS) $(SPHINXOPTS) "$(SOURCEDIR)" "$(BUILDDIR)/html"

latexpdf:
	@$(SPHINXBUILD) -b latex $(SPHINXOPTS) "$(SOURCEDIR)" "$(BUILDDIR)/latex"
	$(MAKE) -C "$(BUILDDIR)/latex" all-pdf

clean:
	rm -rf "$(BUILDDIR)"
