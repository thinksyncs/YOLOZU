import json

from yolozu.benchmark import run_benchmark


def main():
    fps = run_benchmark(iterations=100, sleep_s=0.0)
    print(json.dumps({"fps": fps}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
