#!/usr/bin/env python3
"""
Test suite for sweep configuration examples.
Validates JSON structure and parameter combinations.
"""
import json
import sys
import unittest
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]


class TestSweepConfigs(unittest.TestCase):
    """Test suite for sweep configuration validation."""

    def _validate_sweep_config(self, config_path: Path) -> None:
        """Validate a sweep configuration JSON file."""
        
        # Check file exists
        self.assertTrue(config_path.exists(), f"Config not found: {config_path}")
        
        # Parse JSON
        try:
            config = json.loads(config_path.read_text())
        except Exception as e:
            self.fail(f"Failed to parse JSON in {config_path.name}: {e}")
        
        # Check required fields
        required = ["base_cmd", "result_jsonl", "result_csv", "result_md"]
        missing = [f for f in required if f not in config]
        self.assertEqual([], missing, f"Missing required fields in {config_path.name}: {missing}")
        
        # Check param_grid or param_list exists
        self.assertTrue(
            "param_grid" in config or "param_list" in config,
            f"{config_path.name} must have either 'param_grid' or 'param_list'"
        )
        
        # Validate param_grid structure
        if "param_grid" in config:
            grid = config["param_grid"]
            self.assertIsInstance(grid, dict, f"param_grid must be a dict in {config_path.name}")
            
            for key, values in grid.items():
                self.assertIsInstance(
                    values, list,
                    f"param_grid['{key}'] must be a list in {config_path.name}"
                )
                self.assertGreater(
                    len(values), 0,
                    f"param_grid['{key}'] is empty in {config_path.name}"
                )
            
            # Calculate total runs
            total_runs = 1
            for values in grid.values():
                total_runs *= len(values)
            
            # Check base_cmd has placeholders for all params
            base_cmd = config["base_cmd"]
            for param in grid.keys():
                placeholder = "{" + param + "}"
                if placeholder not in base_cmd:
                    # This is a warning, not a failure
                    print(f"  ⚠ Warning: parameter '{param}' not used in base_cmd in {config_path.name}")
        
        # Validate metrics structure if present
        if "metrics" in config:
            metrics = config["metrics"]
            self.assertIn("path", metrics, f"metrics.path is required in {config_path.name}")

    def test_sweep_ttt_example(self):
        """Test TTT sweep configuration."""
        config_path = repo_root / "docs" / "sweep_ttt_example.json"
        self._validate_sweep_config(config_path)

    def test_sweep_threshold_example(self):
        """Test threshold sweep configuration."""
        config_path = repo_root / "docs" / "sweep_threshold_example.json"
        self._validate_sweep_config(config_path)

    def test_sweep_gate_weights_example(self):
        """Test gate weights sweep configuration."""
        config_path = repo_root / "docs" / "sweep_gate_weights_example.json"
        self._validate_sweep_config(config_path)

    def test_hpo_sweep_example(self):
        """Test HPO sweep configuration (if exists)."""
        config_path = repo_root / "docs" / "hpo_sweep_example.json"
        if config_path.exists():
            self._validate_sweep_config(config_path)


if __name__ == "__main__":
    unittest.main()
