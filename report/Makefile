# Build the report:  make   (tables from results/, then the PDF)
PY ?= python3

all: report.pdf

report.pdf: report.tex tables
	latexmk -pdf -interaction=nonstopmode -halt-on-error report.tex

tables:
	$(PY) make_tables.py

clean:
	latexmk -C

.PHONY: all tables clean
