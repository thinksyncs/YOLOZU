import argparse
import hashlib
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path


HF_DATASETS_BASE = "https://huggingface.co/datasets"


DATASET_DEFAULT_ARCHIVES: dict[str, list[str]] = {
    # Common BOP dataset repos on HF use <dataset>_base.zip plus split-specific archives.
    # We keep this list minimal and allow overriding via --archives.
    "tless": ["tless_base.zip", "tless_train_primesense.zip"],
    "lm": ["lm_base.zip", "lm_train_pbr.zip"],
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, out_path: Path, *, force: bool) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not force:
        return
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    if tmp.exists() and force:
        tmp.unlink()
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as r, tmp.open("wb") as f:
        shutil.copyfileobj(r, f)
    tmp.replace(out_path)


def _extract_zip(zip_path: Path, out_dir: Path, *, force: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = out_dir / f".extracted_{zip_path.name}.sha256"
    digest = _sha256(zip_path)
    if stamp.exists() and stamp.read_text(encoding="utf-8").strip() == digest and not force:
        return
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
    stamp.write_text(digest + "\n", encoding="utf-8")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download BOP dataset archives from the bop-benchmark HuggingFace repo.")
    p.add_argument("--dataset", required=True, help="Dataset id (e.g., tless, lm).")
    p.add_argument(
        "--archives",
        default=None,
        help="Comma-separated archive filenames to download (overrides defaults).",
    )
    p.add_argument("--out", required=True, help="Output directory (will contain extracted dataset folder).")
    p.add_argument("--cache", default=None, help="Optional cache directory for zips (default: <out>/zips).")
    p.add_argument("--force", action="store_true", help="Re-download and re-extract even if present.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    dataset = str(args.dataset).strip().lower()
    out_dir = Path(str(args.out))
    cache_dir = Path(str(args.cache)) if args.cache else (out_dir / "zips")
    if not out_dir.is_absolute():
        out_dir = Path.cwd() / out_dir
    if not cache_dir.is_absolute():
        cache_dir = out_dir / cache_dir

    if args.archives:
        archives = [a.strip() for a in str(args.archives).split(",") if a.strip()]
    else:
        archives = list(DATASET_DEFAULT_ARCHIVES.get(dataset, []))
        if not archives:
            raise SystemExit(
                f"no default archives known for dataset={dataset!r}. "
                "Pass --archives (comma-separated), e.g. --archives tless_base.zip,tless_train_primesense.zip"
            )

    for name in archives:
        url = f"{HF_DATASETS_BASE}/bop-benchmark/{dataset}/resolve/main/{name}"
        zip_path = cache_dir / name
        print(f"download: {url}")
        _download(url, zip_path, force=bool(args.force))
        print(f"extract: {zip_path} -> {out_dir}")
        _extract_zip(zip_path, out_dir, force=bool(args.force))

    # Print dataset root hint if present.
    ds_dir = out_dir / dataset
    if ds_dir.exists():
        print(ds_dir)
    else:
        print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

