#!/bin/bash
# Compile LaTeX manuscript to PDF
# Run from the paper/ directory

pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex

# Clean auxiliary files (optional)
# rm -f *.aux *.bbl *.blg *.log *.out

echo "Compilation complete: main.pdf"
