# YOLOZU LaTeX Manual

This folder contains a **direct TeX** (hand-written LaTeX) manual for YOLOZU.

Smoke-first verification path (repo checkout):

```bash
bash scripts/smoke.sh
```

This validates the committed offline assets under `data/smoke` and writes
`reports/smoke_coco_eval_dry_run.json`.

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
