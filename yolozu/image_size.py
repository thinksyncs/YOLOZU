from __future__ import annotations

from pathlib import Path


class ImageSizeError(RuntimeError):
    pass


def get_image_size(path: str | Path) -> tuple[int, int]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".png",):
        return _png_size(path)
    if suffix in (".jpg", ".jpeg"):
        return _jpeg_size(path)
    raise ImageSizeError(f"unsupported image type: {path.suffix}")


def _png_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    if len(data) < 24:
        raise ImageSizeError("invalid PNG (too short)")
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ImageSizeError("invalid PNG signature")
    # IHDR is the first chunk; width/height are big-endian uint32.
    if data[12:16] != b"IHDR":
        raise ImageSizeError("invalid PNG (missing IHDR)")
    width = int.from_bytes(data[16:20], "big", signed=False)
    height = int.from_bytes(data[20:24], "big", signed=False)
    if width <= 0 or height <= 0:
        raise ImageSizeError("invalid PNG dimensions")
    return width, height


def _jpeg_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    if len(data) < 4:
        raise ImageSizeError("invalid JPEG (too short)")
    if data[0:2] != b"\xff\xd8":
        raise ImageSizeError("invalid JPEG signature")

    i = 2
    while i + 4 <= len(data):
        if data[i] != 0xFF:
            i += 1
            continue

        # Skip fill bytes 0xFF
        while i < len(data) and data[i] == 0xFF:
            i += 1
        if i >= len(data):
            break

        marker = data[i]
        i += 1

        # Standalone markers (no length payload)
        if marker in (0xD8, 0xD9):  # SOI, EOI
            continue
        if marker == 0xDA:  # SOS: start of scan, size info is before this
            break

        if i + 2 > len(data):
            break
        seg_len = int.from_bytes(data[i : i + 2], "big", signed=False)
        if seg_len < 2:
            raise ImageSizeError("invalid JPEG segment length")
        seg_start = i + 2
        seg_end = seg_start + (seg_len - 2)
        if seg_end > len(data):
            break

        # SOF markers that contain width/height (baseline/progressive + variants)
        if marker in (
            0xC0,
            0xC1,
            0xC2,
            0xC3,
            0xC5,
            0xC6,
            0xC7,
            0xC9,
            0xCA,
            0xCB,
            0xCD,
            0xCE,
            0xCF,
        ):
            if seg_start + 7 > len(data):
                break
            height = int.from_bytes(data[seg_start + 1 : seg_start + 3], "big", signed=False)
            width = int.from_bytes(data[seg_start + 3 : seg_start + 5], "big", signed=False)
            if width <= 0 or height <= 0:
                raise ImageSizeError("invalid JPEG dimensions")
            return width, height

        i = seg_end

    raise ImageSizeError("could not determine JPEG size (no SOF marker found)")

