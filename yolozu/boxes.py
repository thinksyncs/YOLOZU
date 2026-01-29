from __future__ import annotations


def xyxy_to_cxcywh_abs(xyxy: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = xyxy
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    return cx, cy, w, h


def cxcywh_abs_to_norm(
    cxcywh: tuple[float, float, float, float], *, width: int, height: int
) -> tuple[float, float, float, float]:
    cx, cy, w, h = cxcywh
    if width <= 0 or height <= 0:
        raise ValueError("width/height must be positive")
    return cx / width, cy / height, w / width, h / height


def xyxy_abs_to_cxcywh_norm(
    xyxy: tuple[float, float, float, float], *, width: int, height: int
) -> tuple[float, float, float, float]:
    return cxcywh_abs_to_norm(xyxy_to_cxcywh_abs(xyxy), width=width, height=height)

