import math
import time
from typing import Callable


def run_benchmark(iterations=100, sleep_s=0.0):
    start = time.perf_counter()
    for _ in range(iterations):
        if sleep_s:
            time.sleep(sleep_s)
    elapsed = time.perf_counter() - start
    if elapsed <= 0.0:
        return float("inf")
    return iterations / elapsed


def _percentile(values: list[float], percent: float) -> float:
    if not values:
        return 0.0
    if percent <= 0:
        return float(min(values))
    if percent >= 100:
        return float(max(values))
    values_sorted = sorted(values)
    k = (len(values_sorted) - 1) * (percent / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(values_sorted[int(k)])
    d0 = values_sorted[int(f)] * (c - k)
    d1 = values_sorted[int(c)] * (k - f)
    return float(d0 + d1)


def measure_latency(
    *,
    iterations: int = 100,
    warmup: int = 10,
    sleep_s: float = 0.0,
    step: Callable[[], None] | None = None,
) -> dict[str, float | int | dict[str, float]]:
    if iterations <= 0:
        raise ValueError("iterations must be > 0")
    if warmup < 0:
        raise ValueError("warmup must be >= 0")

    if step is None:
        if sleep_s <= 0.0:
            def step():
                return None
        else:
            def step():
                time.sleep(sleep_s)

    for _ in range(warmup):
        step()

    durations: list[float] = []
    start = time.perf_counter()
    for _ in range(iterations):
        t0 = time.perf_counter()
        step()
        durations.append(time.perf_counter() - t0)
    elapsed = time.perf_counter() - start

    fps = float("inf") if elapsed <= 0 else float(iterations / elapsed)
    latency_ms = [d * 1000.0 for d in durations]
    mean_ms = float(sum(latency_ms) / len(latency_ms)) if latency_ms else 0.0

    return {
        "iterations": int(iterations),
        "warmup": int(warmup),
        "total_sec": float(elapsed),
        "fps": float(fps),
        "latency_ms": {
            "mean": round(mean_ms, 3),
            "p50": round(_percentile(latency_ms, 50), 3),
            "p90": round(_percentile(latency_ms, 90), 3),
            "p95": round(_percentile(latency_ms, 95), 3),
            "p99": round(_percentile(latency_ms, 99), 3),
            "min": round(min(latency_ms), 3) if latency_ms else 0.0,
            "max": round(max(latency_ms), 3) if latency_ms else 0.0,
        },
    }
