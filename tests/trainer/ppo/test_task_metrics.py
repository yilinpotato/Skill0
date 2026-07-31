from types import SimpleNamespace

import numpy as np
import torch

from verl.trainer.ppo.metric_utils import compute_data_metrics


def test_compute_data_metrics_reports_per_task_rollout_tokens_and_wins():
    """Task totals count episodes once and token requests once per action."""
    batch = SimpleNamespace(
        batch={
            "token_level_scores": torch.zeros((3, 2)),
            "token_level_rewards": torch.zeros((3, 2)),
            "advantages": torch.zeros((3, 2)),
            "returns": torch.zeros((3, 2)),
            "responses": torch.ones((3, 2), dtype=torch.long),
            # One prompt token plus two response tokens for every decision.
            "attention_mask": torch.ones((3, 3), dtype=torch.long),
        },
        non_tensor_batch={
            # Two actions belong to the same successful clean trajectory.
            "traj_uid": np.asarray(["clean-episode", "clean-episode", "heat-episode"], dtype=object),
            "episode_rewards": np.asarray([10.0, 10.0, 0.0], dtype=np.float32),
            "episode_lengths": np.asarray([2, 2, 1], dtype=np.float32),
            "episode_success": np.asarray([1.0, 1.0, 0.0], dtype=np.float32),
            "tool_callings": np.asarray([0, 0, 0], dtype=np.float32),
            "is_action_valid": np.asarray([True, True, False]),
            "episode_task_type": np.asarray(["clean", "clean", "heat"], dtype=object),
        },
    )

    metrics = compute_data_metrics(batch, use_critic=False)

    assert metrics["episode/clean/episodes"] == 1
    assert metrics["episode/clean/wins"] == 1
    assert metrics["episode/clean/success_rate"] == 1.0
    assert metrics["episode/heat/episodes"] == 1
    assert metrics["episode/heat/wins"] == 0
    assert metrics["tokens/small_model/by_task_type/clean/prompt"] == 2
    assert metrics["tokens/small_model/by_task_type/clean/response"] == 4
    assert metrics["tokens/small_model/by_task_type/clean/total"] == 6
    assert metrics["tokens/small_model/by_task_type/heat/total"] == 3
