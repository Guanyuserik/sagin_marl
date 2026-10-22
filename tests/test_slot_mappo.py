import math
import unittest
from types import SimpleNamespace

import torch

from sagin_marl.rl.slot_mappo import agent_ppo_loss, hybrid_logprobs, prepare_slot_batch, slot_gae


class SlotMAPPOTests(unittest.TestCase):
    def test_opposite_agent_changes_do_not_cancel_before_clipping(self):
        new = torch.tensor([[math.log(2), math.log(0.5)]], requires_grad=True)
        loss, info = agent_ppo_loss(new, torch.zeros_like(new), torch.ones(1), 0.2)
        self.assertAlmostEqual(float(loss.detach()), -0.85, places=6)
        self.assertEqual(float(info["clip_frac"]), 1.0)
        loss.backward()
        self.assertAlmostEqual(float(new.grad[0, 0]), 0.0)
        self.assertAlmostEqual(float(new.grad[0, 1]), -0.25)
        # A joint ratio over all UAVs would cancel 2 * .5 and produce loss -1.
        self.assertNotAlmostEqual(float(loss.detach()), -1.0)

    def test_slot_gae_terminal_and_env_separation(self):
        rewards = torch.tensor([[1., 10.], [2., 20.], [4., 40.]])
        values = torch.zeros_like(rewards)
        done = torch.tensor([[False, True], [True, False], [True, True]])
        returns, _ = slot_gae(rewards, values, torch.full_like(values, 99.), done, gamma=0.5, gae_lambda=1.)
        # Successor values are irrelevant after a terminal. Other rows use the
        # supplied bootstrap, so use a consistent zero-value successor below.
        self.assertEqual(float(returns[0, 1]), 10.)
        returns, _ = slot_gae(rewards, values, values, done, gamma=0.5, gae_lambda=1.)
        torch.testing.assert_close(returns, torch.tensor([[2., 10.], [2., 40.], [4., 40.]]))

    def test_rollout_cut_bootstraps_but_finite_episode_end_does_not(self):
        reward = torch.tensor([[1., 1.]])
        value = torch.tensor([[0.2, 0.2]])
        successor = torch.tensor([[0.5, 0.5]])
        returns, _ = slot_gae(reward, value, successor, torch.tensor([[False, True]]), gamma=.9, gae_lambda=.95)
        torch.testing.assert_close(returns, torch.tensor([[1.45, 1.0]]))

    def test_hybrid_likelihood_uses_raw_bw_density(self):
        class Policy:
            def __init__(self, value):
                self.value = value

            def evaluate_actions(self, local, action, **kwargs):
                n = action.shape[0]
                return SimpleNamespace(logprob=torch.full((n,), self.value),
                    logprob_raw=torch.full((n,), self.value * 7), entropy=torch.ones(n), entropy_raw=torch.full((n,), 7.))

        from dataclasses import make_dataclass
        Local = make_dataclass("Local", [("x", torch.Tensor)])
        local = Local(torch.zeros(4, 1))
        stages = {s: SimpleNamespace(num_agents=2, local_batch=local,
            actions=torch.zeros(2, 2, 2) if s != 1 else torch.zeros(2, 2, dtype=torch.long),
            latent_actions=torch.zeros(2, 2, 2)) for s in range(3)}
        actor = SimpleNamespace(accel_policy=Policy(-1.), sat_subset_policy=Policy(-2.), bw_policy=Policy(3.))
        parts, entropy, collector = hybrid_logprobs(actor, stages, torch.arange(2))
        torch.testing.assert_close(parts[0, 0], torch.tensor([-1., -2., 21.]))
        self.assertEqual(float(collector[2][0, 0]), 3.)
        self.assertEqual(float(entropy[0, 0, 2]), 7.)

    def test_nonfinite_likelihood_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            agent_ppo_loss(torch.tensor([[float("nan")]]), torch.zeros(1, 1), torch.ones(1), .2)

    def test_native_placeholder_values_are_recomputed_before_gae(self):
        from dataclasses import make_dataclass
        Local = make_dataclass("Local", [("x", torch.Tensor)])

        class Policy:
            def evaluate_actions(self, local, action, **kwargs):
                n = action.shape[0]
                return SimpleNamespace(logprob=torch.zeros(n), logprob_raw=torch.zeros(n),
                    entropy=torch.zeros(n), entropy_raw=torch.zeros(n))

        actor = SimpleNamespace(accel_policy=Policy(), sat_subset_policy=Policy(), bw_policy=Policy())
        critic = SimpleNamespace(value_accel=lambda world: world.x[:, 0])
        envs = __import__("numpy").array([0, 1, 0, 1])
        stages = {s: SimpleNamespace(num_samples=4, num_agents=2, env_indices=envs,
            transition_indices=__import__("numpy").arange(4) * 3 + s,
            local_batch=Local(torch.zeros(8, 1)), world_batch=Local(torch.full((4, 1), 2.)),
            actions=torch.zeros(4, 2, 2) if s != 1 else torch.zeros(4, 2, dtype=torch.long),
            old_logprobs=torch.zeros(4), latent_actions=torch.zeros(4, 2, 2), values=torch.zeros(4),
            rewards=torch.ones(4), terminated=torch.tensor([False, False, True, True]),
            truncated=torch.zeros(4, dtype=torch.bool)) for s in range(3)}
        views = SimpleNamespace(training_view=SimpleNamespace(stage_batches=stages),
            return_view=SimpleNamespace(stage_batches={2: SimpleNamespace(next_world_batch=Local(torch.full((4, 1), 2.)))}))
        batch = prepare_slot_batch(actor, critic, views, num_envs=2, gamma=1., gae_lambda=1., chunk_size=3)
        torch.testing.assert_close(batch.old_values, torch.full((4,), 2.))
        torch.testing.assert_close(batch.returns, torch.tensor([2., 2., 1., 1.]))


if __name__ == "__main__":
    unittest.main()
