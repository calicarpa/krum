# Makefile for Krum documentation

.PHONY: help html clean servedocs live watch

help:
	@echo "Available targets:"
	@echo "  html       - Build HTML documentation"
	@echo "  clean      - Remove generated files"
	@echo "  serve      - Build and serve static documentation"
	@echo "  watch      - Watch for changes and serve (auto-rebuild)"

html:
	uv run sphinx-build -b html . _build/html

clean:
	rm -rf _build

serve:
	uv run sphinx-build -b html . _build/html && uv run python -m http.server 8000 -d _build/html

watch:
	uv run sphinx-autobuild -b html . _build/html --port 8000 --host 127.0.0.1
