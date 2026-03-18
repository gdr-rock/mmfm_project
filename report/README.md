# LaTeX Report

This folder contains a paper-style LaTeX draft for the `vlwm_planning` project.

## Files

- `main.tex`: main report source
- `references.bib`: bibliography entries used by the draft

## What is already written

- A full research-paper structure matching the course requirements
- Project-specific methodology based on the current codebase
- Tables for datasets, hyperparameters, and results
- An appendix template for individual contributions

## What still needs to be filled in

- Author names and affiliation
- Final quantitative results in the results table
- Any qualitative examples or figures you want to include
- Exact team-member contribution details in the appendix

## Compile

Example:

```powershell
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```
