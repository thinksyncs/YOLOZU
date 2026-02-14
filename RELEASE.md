# Releasing to PyPI

YOLOZU is configured for **PyPI Trusted Publishing** (OIDC) via GitHub Actions.
This avoids long-lived PyPI API tokens.

## One-time setup (PyPI)

1) Create a PyPI account and enable **2FA**.

2) Configure a **Trusted Publisher** for this repo.
   - For first release of a new project name, use a **pending publisher** so the project is created on first publish.
   - Recommended:
     - Project name: `yolozu`
     - Repository: `<owner>/YOLOZU`
     - Workflow file: `.github/workflows/publish.yml`
     - Environment: `pypi`

## Each release

1) Bump version in `yolozu/__init__.py` (`__version__`).

2) Run tests locally:

```bash
python3 -m unittest -q
```

3) Push to `main`.

4) Create an annotated git tag and push it:

```bash
git tag -a vX.Y.Z -m "vX.Y.Z"
git push origin vX.Y.Z
```

This triggers `.github/workflows/publish.yml` which builds and publishes to PyPI.

5) Optional: create a GitHub Release for human-friendly notes (does not affect publishing).

## Notes

- You cannot upload the same version twice to PyPI. Always bump `__version__` before releasing.
- If you prefer manual publishing, use `python -m build` + `python -m twine upload dist/*` with a PyPI API token.
