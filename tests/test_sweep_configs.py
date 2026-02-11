#!/usr/bin/env python3
"""
Test suite for sweep configuration examples.
Validates JSON structure and parameter combinations.
"""
import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]


def validate_sweep_config(config_path: Path) -> bool:
    """Validate a sweep configuration JSON file."""
    print(f"\nValidating: {config_path.name}")
    
    try:
        config = json.loads(config_path.read_text())
    except Exception as e:
        print(f"  ✗ Failed to parse JSON: {e}")
        return False
    
    # Check required fields
    required = ["base_cmd", "result_jsonl", "result_csv", "result_md"]
    missing = [f for f in required if f not in config]
    if missing:
        print(f"  ✗ Missing required fields: {missing}")
        return False
    
    # Check param_grid or param_list exists
    if "param_grid" not in config and "param_list" not in config:
        print(f"  ✗ Must have either 'param_grid' or 'param_list'")
        return False
    
    # Validate param_grid structure
    if "param_grid" in config:
        grid = config["param_grid"]
        if not isinstance(grid, dict):
            print(f"  ✗ param_grid must be a dict")
            return False
        
        for key, values in grid.items():
            if not isinstance(values, list):
                print(f"  ✗ param_grid['{key}'] must be a list")
                return False
            if not values:
                print(f"  ✗ param_grid['{key}'] is empty")
                return False
        
        # Calculate total runs
        total_runs = 1
        for values in grid.values():
            total_runs *= len(values)
        print(f"  ✓ param_grid valid: {len(grid)} parameters, {total_runs} total runs")
    
    # Check base_cmd has placeholders
    base_cmd = config["base_cmd"]
    if "param_grid" in config:
        for param in config["param_grid"].keys():
            placeholder = "{" + param + "}"
            if placeholder not in base_cmd:
                print(f"  ⚠ Warning: parameter '{param}' not used in base_cmd")
    
    # Validate metrics structure if present
    if "metrics" in config:
        metrics = config["metrics"]
        if "path" not in metrics:
            print(f"  ✗ metrics.path is required")
            return False
        print(f"  ✓ metrics configuration valid")
    
    print(f"  ✓ {config_path.name} is valid")
    return True


def main():
    docs_dir = repo_root / "docs"
    
    # Find all sweep example configs
    sweep_configs = [
        docs_dir / "sweep_ttt_example.json",
        docs_dir / "sweep_threshold_example.json",
        docs_dir / "sweep_gate_weights_example.json",
        docs_dir / "hpo_sweep_example.json",
    ]
    
    all_valid = True
    for config_path in sweep_configs:
        if not config_path.exists():
            print(f"\n✗ Config not found: {config_path}")
            all_valid = False
            continue
        
        if not validate_sweep_config(config_path):
            all_valid = False
    
    if all_valid:
        print("\n✓ All sweep configs are valid!")
        return 0
    else:
        print("\n✗ Some sweep configs are invalid")
        return 1


if __name__ == "__main__":
    sys.exit(main())
