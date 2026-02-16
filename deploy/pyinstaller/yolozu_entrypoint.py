from __future__ import annotations


def main() -> int:
    from yolozu.cli import main as _main

    return int(_main())


if __name__ == "__main__":
    raise SystemExit(main())

