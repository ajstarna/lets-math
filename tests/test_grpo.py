import pytest
import torch

from lets_math.grpo import compute_group_advantages, exact_answer_rewards


def test_normalizes_rewards_within_each_prompt_group():
    rewards = torch.tensor([[0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]])

    advantages = compute_group_advantages(rewards)

    expected = torch.tensor([[-1.0, 1.0, -1.0, 1.0], [-1.0, -1.0, 1.0, 1.0]])
    assert torch.allclose(advantages, expected, atol=1e-3)


def test_equal_reward_group_has_zero_advantages():
    rewards = torch.tensor([[1.0, 1.0, 1.0, 1.0]])

    advantages = compute_group_advantages(rewards)

    assert torch.equal(advantages, torch.zeros_like(rewards))


def test_rewards_only_exact_well_formed_answers():
    rewards = exact_answer_rewards(
        ["Work\n#### 4", "Work\n#### 5", "no final answer"],
        ["Gold\n#### 4", "Gold\n#### 4", "Gold\n#### 4"],
    )

    assert torch.equal(rewards, torch.tensor([1.0, 0.0, 0.0]))


def test_rejects_non_grouped_rewards():
    with pytest.raises(ValueError, match="shape"):
        compute_group_advantages(torch.tensor([0.0, 1.0]))
