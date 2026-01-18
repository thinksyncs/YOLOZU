import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yolozu.jitter import default_jitter_profile, jitter_off, sample_extrinsics_jitter, sample_intrinsics_jitter


class TestJitter(unittest.TestCase):
    def test_jitter_off(self):
        self.assertEqual(jitter_off(), {"dfx": 0.0, "dfy": 0.0, "dcx": 0.0, "dcy": 0.0})

    def test_intrinsics_jitter_bounds(self):
        profile = default_jitter_profile()
        jitter = sample_intrinsics_jitter(profile, seed=123)
        intr = profile["intrinsics"]
        self.assertLessEqual(abs(jitter["dfx"]), intr["dfx"])
        self.assertLessEqual(abs(jitter["dfy"]), intr["dfy"])
        self.assertLessEqual(abs(jitter["dcx"]), intr["dcx"])
        self.assertLessEqual(abs(jitter["dcy"]), intr["dcy"])

    def test_extrinsics_jitter_bounds(self):
        profile = default_jitter_profile()
        jitter = sample_extrinsics_jitter(profile, seed=456)
        ext = profile["extrinsics"]
        self.assertLessEqual(abs(jitter["dx"]), ext["dx"])
        self.assertLessEqual(abs(jitter["dy"]), ext["dy"])
        self.assertLessEqual(abs(jitter["dz"]), ext["dz"])
        self.assertLessEqual(abs(jitter["droll"]), ext["droll"])
        self.assertLessEqual(abs(jitter["dpitch"]), ext["dpitch"])
        self.assertLessEqual(abs(jitter["dyaw"]), ext["dyaw"])


if __name__ == "__main__":
    unittest.main()
