import unittest

from yolozu.long_tail_recipe import (
    build_class_balanced_weights,
    build_logit_adjustment_bias,
    build_long_tail_recipe,
)


class TestLongTailRecipe(unittest.TestCase):
    def _records(self):
        return [
            {
                "image": "a.jpg",
                "labels": [
                    {"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2},
                    {"class_id": 1, "cx": 0.3, "cy": 0.3, "w": 0.1, "h": 0.1},
                ],
            },
            {
                "image": "b.jpg",
                "labels": [{"class_id": 0, "cx": 0.6, "cy": 0.5, "w": 0.2, "h": 0.2}],
            },
            {
                "image": "c.jpg",
                "labels": [{"class_id": 2, "cx": 0.4, "cy": 0.4, "w": 0.2, "h": 0.2}],
            },
        ]

    def test_class_balanced_weights_and_logit_bias(self):
        counts = {0: 10, 1: 2, 2: 1}
        weights = build_class_balanced_weights(counts, beta=0.9)
        self.assertGreater(weights[2], weights[0])

        bias = build_logit_adjustment_bias(counts, tau=1.0)
        self.assertLess(bias[2], bias[0])

    def test_build_recipe_contains_required_sections(self):
        recipe = build_long_tail_recipe(
            self._records(),
            seed=7,
            stage1_epochs=20,
            stage2_epochs=10,
            rebalance_sampler="class_balanced",
            loss_plugin="focal",
            logit_adjustment_tau=1.0,
            lort_tau=0.3,
            class_balanced_beta=0.999,
            focal_gamma=2.0,
            ldam_margin=0.5,
        )

        self.assertEqual(recipe.get("kind"), "yolozu_long_tail_recipe")
        self.assertIn("stages", recipe)
        self.assertIn("plugins", recipe)
        self.assertIn("dataset_distribution", recipe)
        self.assertIn("recipe_hash", recipe)

        stages = recipe["stages"]
        self.assertTrue(stages["stage1_representation"]["enabled"])
        self.assertTrue(stages["stage2_classifier_retrain"]["decoupled"])

        plugins = recipe["plugins"]
        self.assertEqual(plugins["sampler"]["name"], "class_balanced")
        self.assertEqual(plugins["loss"]["name"], "focal")
        self.assertTrue(plugins["logit_adjustment"]["enabled"])
        self.assertTrue(plugins["lort"]["enabled"])


if __name__ == "__main__":
    unittest.main()
