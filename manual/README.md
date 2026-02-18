# YOLOZU LaTeX Manual

This folder contains a **direct TeX** (hand-written LaTeX) manual for YOLOZU.

## Build

Requirements:
- A LaTeX distribution (MacTeX / TeX Live)
- `latexmk` (usually included with TeX Live)

Build PDF:

```bash
cd manual
make pdf
```

Clean:

```bash
cd manual
make clean
```

Output:
- `build/yolozu_manual.pdf`

## Editing

- Entry point: `main.tex`
- Chapters: `chapters/*.tex`

This manual is designed to mirror the repo docs organization (see `docs/README.md`) while being
printable/searchable as a single PDF.
