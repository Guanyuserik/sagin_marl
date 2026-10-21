import unittest

from scripts.experiments.more_gus.run_review_campaign import (
    conditions, learning_config, normalize_episode,
)


class ReviewCampaignTests(unittest.TestCase):
    def setUp(self):
        self.base = dict(seed=45211, structured_actor_backbone="topology_aware",
            critic_value_mode="relational", task_arrival_rate=4e5, b_acc=4e6,
            b_backhaul_per_sat=1e7, b_sat_total=1e7, sat_cpu_freq=5e10,
            hotspot_rho=4.0, hotspot_on_mean_steps=20.0, T_steps=250,
            safety_shield_enabled=True)

    def test_bottleneck_scans_change_one_resource_and_keep_alias_consistent(self):
        points = conditions(self.base)
        self.assertEqual(len(points), 21)
        self.assertEqual(len({p["name"] for p in points}), 21)
        for point in points:
            changes = point["changes"]
            if point["name"].startswith("access_"):
                self.assertEqual(set(changes), {"b_acc"})
            if point["name"].startswith("cpu_"):
                self.assertEqual(set(changes), {"sat_cpu_freq"})
            if point["name"].startswith("backhaul_"):
                self.assertEqual(set(changes), {"b_backhaul_per_sat", "b_sat_total"})
                self.assertEqual(changes["b_sat_total"], changes["b_backhaul_per_sat"])
            if point["name"].startswith("hotspot_"):
                self.assertNotIn("task_arrival_rate", changes)

    def test_learning_control_keeps_reward_safety_and_scenario(self):
        modified = learning_config(self.base, 61723)
        changed = {key for key in self.base if modified[key] != self.base[key]}
        self.assertEqual(changed, {"seed", "structured_actor_backbone", "critic_value_mode"})
        self.assertTrue(modified["safety_shield_enabled"])

    def episode(self):
        return dict(arrival_sum=1000, episode_length=250, sat_processed_sum=800,
            drop_sum=50, terminal_queue_sum=150, queue_total_mean=2e6,
            bw_weighted_workload_level_sum=-5000, reward_sum=90,
            collision_episode_fraction=0, processed_ratio_eval=0.999)

    def test_report_uses_cumulative_workload_and_original_cost(self):
        result = normalize_episode(self.episode(), 250)
        self.assertEqual(result["processed_workload_fraction"], 0.8)
        self.assertEqual(result["objective_cost_per_observed_slot"], 20)
        self.assertEqual(result["mean_queue_mbit"], 2)
        self.assertEqual(result["terminal_inventory_fraction"], 0.15)
        self.assertAlmostEqual(result["conservation_error"], 0)

    def test_missing_inventory_or_arrivals_cannot_silently_pass(self):
        bad = self.episode()
        bad["terminal_queue_sum"] = 0
        with self.assertRaises(ValueError):
            normalize_episode(bad, 250)
        bad = self.episode()
        bad["arrival_sum"] = 0
        with self.assertRaises(ValueError):
            normalize_episode(bad, 250)

    def test_early_termination_remains_explicit(self):
        row = self.episode()
        row["episode_length"] = 80
        row["collision_episode_fraction"] = 1
        result = normalize_episode(row, 250)
        self.assertEqual(result["early_termination"], 1)
        self.assertEqual(result["complete_horizon"], 0)
        self.assertEqual(result["collision"], 1)


if __name__ == "__main__":
    unittest.main()
