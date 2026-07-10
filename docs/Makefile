# Minimal Makefile for Sphinx docs (HTML + LaTeX/PDF)
# Use: make html | make latexpdf

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

.PHONY: help html latexpdf clean

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

html:
	@$(SPHINXBUILD) -b html $(SPHINXOPTS) "$(SOURCEDIR)" "$(BUILDDIR)/html"

latexpdf:
	@$(SPHINXBUILD) -b latex $(SPHINXOPTS) "$(SOURCEDIR)" "$(BUILDDIR)/latex"
	$(MAKE) -C "$(BUILDDIR)/latex" all-pdf

clean:
	rm -rf "$(BUILDDIR)"
