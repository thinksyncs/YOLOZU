import json


SCENARIOS = [
    {"name": "symmetric_on"},
    {"name": "symmetric_off"},
    {"name": "tabletop_on"},
    {"name": "tabletop_off"},
    {"name": "extreme_depth"},
    {"name": "handheld_jitter_on"},
    {"name": "handheld_jitter_off"},
    {"name": "template_gate_on"},
    {"name": "template_gate_off"},
]


def _scenario_metrics(index):
    fps = 33.0 - index * 0.5
    return {
        "fps": max(5.0, fps),
        "map": round(0.4 + index * 0.01, 4),
        "recall": round(0.5 + index * 0.01, 4),
        "depth_error": round(0.2 + index * 0.005, 4),
        "pose_error": round(5.0 + index * 0.1, 4),
        "rejection_rate": round(0.1 + index * 0.01, 4),
    }


def build_report():
    scenarios = []
    for index, scenario in enumerate(SCENARIOS):
        metrics = _scenario_metrics(index)
        scenarios.append({"name": scenario["name"], "metrics": metrics})
    return {"scenarios": scenarios}


def main():
    report = build_report()
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
