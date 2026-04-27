# Makefile for Krum documentation

.PHONY: help html clean servedocs

help:
	@echo "Available targets:"
	@echo "  html       - Build HTML documentation"
	@echo "  clean      - Remove generated files"
	@echo "  servedocs  - Serve documentation locally"

html:
	uv run sphinx-build -b html . _build/html

clean:
	rm -rf _build

servedocs:
	uv run sphinx-build -b html . _build/html && uv run python -m http.server 8000 -d _build/html