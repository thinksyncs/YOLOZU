import unittest

from rtdetr_pose.matcher import linear_sum_assignment


class TestMatcher(unittest.TestCase):
    def test_linear_sum_assignment_square(self):
        cost = [
            [4, 1, 3],
            [2, 0, 5],
            [3, 2, 2],
        ]
        rows, cols = linear_sum_assignment(cost)
        pairs = sorted(zip(rows, cols))
        # One optimal solution: (0,1), (1,0), (2,2)
        self.assertEqual(pairs, [(0, 1), (1, 0), (2, 2)])

    def test_linear_sum_assignment_rectangular(self):
        cost = [
            [10, 3, 2, 9],
            [7, 1, 4, 8],
        ]
        rows, cols = linear_sum_assignment(cost)
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(cols), 2)
        self.assertEqual(len(set(rows)), 2)
        self.assertEqual(len(set(cols)), 2)


if __name__ == "__main__":
    unittest.main()
