import time


def run_benchmark(iterations=100, sleep_s=0.0):
    start = time.perf_counter()
    for _ in range(iterations):
        if sleep_s:
            time.sleep(sleep_s)
    elapsed = time.perf_counter() - start
    if elapsed <= 0.0:
        return float("inf")
    return iterations / elapsed
