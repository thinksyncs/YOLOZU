import unittest


class TestReplayBuffer(unittest.TestCase):
    def test_capacity_zero_disables_storage(self):
        from yolozu.replay_buffer import ReplayBuffer

        buf = ReplayBuffer(capacity=0, seed=0)
        for i in range(10):
            buf.add({"image_path": f"/tmp/{i}.jpg", "labels": []})
        self.assertEqual(len(buf), 0)
        self.assertEqual(buf.seen, 10)

    def test_add_with_info(self):
        from yolozu.replay_buffer import ReplayBuffer

        buf = ReplayBuffer(capacity=2, seed=0)

        inserted, replaced = buf.add_with_info({"image_path": "/tmp/0.jpg"})
        self.assertTrue(inserted)
        self.assertIsNone(replaced)
        self.assertEqual(len(buf), 1)
        self.assertEqual(buf.seen, 1)

        inserted, replaced = buf.add_with_info({"image_path": "/tmp/1.jpg"})
        self.assertTrue(inserted)
        self.assertIsNone(replaced)
        self.assertEqual(len(buf), 2)
        self.assertEqual(buf.seen, 2)

        inserted, replaced = buf.add_with_info({"image_path": "/tmp/2.jpg"})
        self.assertTrue(inserted)
        self.assertIsNotNone(replaced)
        self.assertEqual(replaced.get("image_path"), "/tmp/1.jpg")
        self.assertEqual(len(buf), 2)
        self.assertEqual(buf.seen, 3)

        inserted, replaced = buf.add_with_info({"image_path": "/tmp/3.jpg"})
        self.assertFalse(inserted)
        self.assertIsNone(replaced)
        self.assertEqual(len(buf), 2)
        self.assertEqual(buf.seen, 4)

    def test_reservoir_basic_properties(self):
        from yolozu.replay_buffer import ReplayBuffer

        buf = ReplayBuffer(capacity=10, seed=123)
        for i in range(100):
            buf.add({"image_path": f"/tmp/{i}.jpg", "labels": [{"class_id": 0, "bbox": {"cx": 0.5, "cy": 0.5, "w": 1.0, "h": 1.0}}]})
        self.assertEqual(len(buf), 10)
        self.assertEqual(buf.seen, 100)

        sample = buf.sample(5)
        self.assertEqual(len(sample), 5)
        self.assertEqual(len({r["image_path"] for r in sample}), 5)

        summary = buf.summary()
        self.assertEqual(summary["capacity"], 10)
        self.assertEqual(summary["size"], 10)
        self.assertEqual(len(summary["images"]), 10)

    def test_sample_per_task_cap(self):
        from yolozu.replay_buffer import ReplayBuffer

        buf = ReplayBuffer(capacity=100, seed=0)
        for i in range(7):
            buf.add({"image_path": f"/tmp/a_{i}.jpg", "__task": "A"})
        for i in range(5):
            buf.add({"image_path": f"/tmp/b_{i}.jpg", "__task": "B"})

        sample = buf.sample(task_key="__task", per_task_cap=2)
        self.assertEqual(len(sample), 4)
        counts = {"A": 0, "B": 0}
        for rec in sample:
            counts[rec.get("__task")] += 1
        self.assertLessEqual(counts["A"], 2)
        self.assertLessEqual(counts["B"], 2)

        sample_k = buf.sample(3, task_key="__task", per_task_cap=2)
        self.assertEqual(len(sample_k), 3)
        counts_k = {"A": 0, "B": 0}
        for rec in sample_k:
            counts_k[rec.get("__task")] += 1
        self.assertLessEqual(counts_k["A"], 2)
        self.assertLessEqual(counts_k["B"], 2)


if __name__ == "__main__":
    unittest.main()
