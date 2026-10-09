"""Small, testable building blocks for a hand-rolled GRPO trainer."""

import torch


def compute_group_advantages(rewards: torch.Tensor, epsilon: float = 1e-4) -> torch.Tensor:
    """Normalize rewards within each prompt's completion group.

    ``rewards`` has shape ``(prompt_batch_size, group_size)``. A group whose
    completions all receive the same reward has zero advantages and therefore
    supplies no relative learning signal.
    """
    if rewards.ndim != 2:
        raise ValueError("rewards must have shape (prompt_batch_size, group_size)")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    group_mean = rewards.mean(dim=1, keepdim=True)
    group_std = rewards.std(dim=1, keepdim=True, correction=0)
    return (rewards - group_mean) / (group_std + epsilon)
