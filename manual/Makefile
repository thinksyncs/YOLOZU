SHELL := /bin/bash

PDF_NAME := yolozu_manual
BUILD_DIR := build
TEX := main.tex

.PHONY: pdf graphs clean

graphs:
	python3 ../tools/generate_manual_import_graphs.py

pdf: graphs
	mkdir -p $(BUILD_DIR) $(BUILD_DIR)/chapters
	latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=$(BUILD_DIR) -jobname=$(PDF_NAME) $(TEX)
	@echo "Wrote $(BUILD_DIR)/$(PDF_NAME).pdf"

clean:
	latexmk -C -outdir=$(BUILD_DIR) -jobname=$(PDF_NAME) $(TEX) || true
	rm -rf $(BUILD_DIR)
