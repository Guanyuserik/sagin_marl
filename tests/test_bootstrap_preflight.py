import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.experiments.more_gus import run_bootstrap_training as runner


class PreflightSeedRegressionTests(unittest.TestCase):
    def test_internal_external_parity_uses_the_frozen_config_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            small = root / "preflight" / "small_resume"
            small.mkdir(parents=True)
            (small / "training_stop.json").write_text(json.dumps({"completed_updates": 3}))
            (root / "small_config.yaml").write_text("checkpoint_eval_episode_seed_base: 20420000\n")
            metrics = {"reward_sum": 32., "processed_ratio_eval": .7, "drop_ratio_eval": .1,
                "pre_backlog_steps_eval": 2., "collision_episode_fraction": 0.}
            with (small / "checkpoint_eval.csv").open("w") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(metrics))
                writer.writeheader()
                writer.writerow(metrics)
            report = {"stages": {name: {"optimizer_steps": 1} for name in ("accel", "sat", "bw")}}
            with patch.object(runner, "check_training", return_value=report), \
                    patch.object(runner, "evaluate", return_value=metrics) as evaluate, \
                    patch.object(runner, "run_command", return_value=0.), \
                    patch.object(runner, "train_command", return_value=[]):
                runner.preflight(root, root, SimpleNamespace())
            self.assertEqual(evaluate.call_args.kwargs["seed"], 20420000)
            self.assertEqual(json.loads((root / "preflight.json").read_text())["evaluation_seed"], 20420000)


if __name__ == "__main__":
    unittest.main()
