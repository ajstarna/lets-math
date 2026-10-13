"""A small hand-rolled GRPO trainer for exact-answer GSM8K rewards."""

from argparse import Namespace
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from time import perf_counter
from typing import Any

import torch
from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .evaluate import evaluate_gsm8k, score_completion
from .sft import _append_metric, _device, _save_adapter, _write_summary, format_gsm8k_example, load_lora_model


@dataclass(frozen=True)
class GRPOConfig:
    model_name: str
    adapter: str
    output_dir: str
    batch_size: int = 1
    group_size: int = 4
    learning_rate: float = 1e-5
    max_steps: int = 100
    updates_per_rollout: int = 4
    max_new_tokens: int = 384
    evaluation_max_new_tokens: int = 512
    temperature: float = 0.7
    beta: float = 0.04
    clip_epsilon: float = 0.2
    format_reward: float = 0.05
    max_grad_norm: float = 1.0
    log_every: int = 1
    save_every: int = 25
    evaluate_every: int = 25
    evaluation_examples: int = 100
    generation_batch_size: int = 8
    seed: int = 42


def compute_group_advantages(rewards: torch.Tensor, epsilon: float = 1e-4) -> torch.Tensor:
    """Normalize rewards within each prompt's completion group."""
    if rewards.ndim != 2:
        raise ValueError("rewards must have shape (prompt_batch_size, group_size)")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    group_mean = rewards.mean(dim=1, keepdim=True)
    group_std = rewards.std(dim=1, keepdim=True, correction=0)
    return (rewards - group_mean) / (group_std + epsilon)


def exact_answer_rewards(completions: list[str], answers: list[str]) -> torch.Tensor:
    """Return one binary exact-answer verifier reward per completion."""
    if len(completions) != len(answers):
        raise ValueError("completions and answers must have the same length")
    return torch.tensor(
        [
            float(score_completion(completion, answer).is_correct)
            for completion, answer in zip(completions, answers, strict=True)
        ]
    )


def format_rewards(completions: list[str], answers: list[str]) -> torch.Tensor:
    """Return one binary reward for the required trailing ``#### answer`` contract."""
    if len(completions) != len(answers):
        raise ValueError("completions and answers must have the same length")
    return torch.tensor(
        [
            float(score_completion(completion, answer).has_valid_format)
            for completion, answer in zip(completions, answers, strict=True)
        ]
    )


def grpo_rewards(
    completions: list[str], answers: list[str], format_reward: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return total, exact-answer, and format reward components."""
    if format_reward < 0:
        raise ValueError("format_reward cannot be negative")
    exact = exact_answer_rewards(completions, answers)
    formatted = format_rewards(completions, answers)
    return exact + format_reward * formatted, exact, formatted


@contextmanager
def _reference_adapter(model: PreTrainedModel, reference: dict[str, torch.Tensor]):
    """Temporarily use the fixed initial LoRA weights without another base model."""
    parameters = dict(model.named_parameters())
    current = {name: parameters[name].data for name in reference}
    try:
        for name, value in reference.items():
            parameters[name].data = value
        yield
    finally:
        for name, value in current.items():
            parameters[name].data = value


def _completion_log_probs(
    model: PreTrainedModel, sequences: torch.Tensor, attention_mask: torch.Tensor, completion_mask: torch.Tensor
) -> torch.Tensor:
    """Per-token log probabilities for generated completion tokens."""
    logits = model(input_ids=sequences, attention_mask=attention_mask).logits[:, :-1]
    target = sequences[:, 1:]
    log_probs = logits.log_softmax(-1).gather(-1, target.unsqueeze(-1)).squeeze(-1)
    return log_probs[:, -completion_mask.shape[1] :] * completion_mask


def _sample_groups(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    rows: list[dict[str, str]],
    config: GRPOConfig,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[str],
    list[str],
    torch.Tensor,
]:
    prompts = [format_gsm8k_example(row)["prompt"] for row in rows]
    prompts = [prompt for prompt in prompts for _ in range(config.group_size)]
    answers = [row["answer"] for row in rows for _ in range(config.group_size)]
    device = next(model.parameters()).device
    original_padding_side = tokenizer.padding_side
    was_training = model.training
    tokenizer.padding_side = "left"
    try:
        inputs = tokenizer(
            prompts, add_special_tokens=False, padding=True, return_tensors="pt"
        ).to(device)
        prompt_width = inputs["input_ids"].shape[1]
        model.eval()
        with torch.inference_mode():
            sequences = model.generate(
                **inputs,
                do_sample=True,
                temperature=config.temperature,
                max_new_tokens=config.max_new_tokens,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
    finally:
        tokenizer.padding_side = original_padding_side
        model.train(was_training)

    # generate() ran under inference_mode; clone its token IDs before using them
    # as targets in the gradient-tracked policy forward pass below.
    sequences = sequences.clone()
    completion_ids = sequences[:, prompt_width:]
    completion_mask = (completion_ids != tokenizer.pad_token_id).to(torch.float32)
    attention_mask = torch.cat(
        (inputs["attention_mask"], completion_mask.to(inputs["attention_mask"].dtype)), dim=1
    )
    completions = [
        tokenizer.decode(ids[mask.bool()], skip_special_tokens=True)
        for ids, mask in zip(completion_ids, completion_mask, strict=True)
    ]
    truncated = ~completion_ids.eq(tokenizer.eos_token_id).any(dim=1)
    return sequences, attention_mask, completion_mask, completions, answers, truncated


def _prompt_batches(dataset: Dataset, batch_size: int, seed: int):
    epoch = 0
    while True:
        indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed + epoch)).tolist()
        for start in range(0, len(indices) - batch_size + 1, batch_size):
            yield [dataset[index] for index in indices[start : start + batch_size]]
        epoch += 1


def train_grpo(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset: Dataset,
    validation_dataset: Dataset,
    config: GRPOConfig,
    wandb_run: Any | None = None,
) -> None:
    """Train an SFT LoRA adapter with sampled, group-relative exact-answer rewards."""
    if min(
        config.batch_size,
        config.group_size,
        config.max_steps,
        config.updates_per_rollout,
        config.max_new_tokens,
        config.evaluation_max_new_tokens,
        config.log_every,
        config.save_every,
        config.evaluate_every,
        config.evaluation_examples,
        config.generation_batch_size,
    ) < 1:
        raise ValueError("batch sizes, step counts, and intervals must be at least 1")
    if (
        config.temperature <= 0
        or config.beta < 0
        or config.clip_epsilon < 0
        or config.format_reward < 0
        or config.max_grad_norm <= 0
    ):
        raise ValueError(
            "temperature and max_grad_norm must be positive; beta, clip_epsilon, and format_reward cannot be negative"
        )

    torch.manual_seed(config.seed)
    random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    model.to(_device())
    model.config.use_cache = False
    reference = {name: parameter.detach().clone() for name, parameter in model.named_parameters() if parameter.requires_grad}
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=config.learning_rate)
    batches = _prompt_batches(train_dataset, config.batch_size, config.seed)
    validation_dataset = validation_dataset.select(
        range(min(config.evaluation_examples, len(validation_dataset)))
    )
    original_padding_side = tokenizer.padding_side
    model.eval()
    tokenizer.padding_side = "left"
    try:
        initial_accuracy = evaluate_gsm8k(
            model,
            tokenizer,
            validation_dataset,
            config.evaluation_max_new_tokens,
            config.generation_batch_size,
        ).accuracy
    finally:
        tokenizer.padding_side = original_padding_side
    best_accuracy = initial_accuracy
    _save_adapter(model, tokenizer, Path(config.output_dir) / "checkpoint-initial")
    _save_adapter(model, tokenizer, Path(config.output_dir) / "checkpoint-best")
    _append_metric(
        config.output_dir,
        {"step": 0, "event": "generation_validation", "accuracy": initial_accuracy},
    )
    if wandb_run is not None:
        wandb_run.log({"generation/accuracy": initial_accuracy}, step=0)
    print(f"GRPO initial generation accuracy: {initial_accuracy:.1%}")
    start = perf_counter()
    rollout_count = (config.max_steps + config.updates_per_rollout - 1) // config.updates_per_rollout
    print(
        f"GRPO training on {_device()}; {config.max_steps} optimizer steps = "
        f"{rollout_count} rollout batches × up to {config.updates_per_rollout} updates "
        f"({rollout_count * config.batch_size * config.group_size} sampled completions maximum)"
    )

    step = 0
    while step < config.max_steps:
        rows = next(batches)
        (
            sequences,
            attention_mask,
            completion_mask,
            completions,
            answers,
            truncated,
        ) = _sample_groups(model, tokenizer, rows, config)
        rewards, exact_rewards, formatting_rewards = grpo_rewards(
            completions, answers, config.format_reward
        )
        rewards = rewards.to(sequences.device)
        exact_rewards = exact_rewards.to(sequences.device)
        formatting_rewards = formatting_rewards.to(sequences.device)
        advantages = compute_group_advantages(
            rewards.view(config.batch_size, config.group_size)
        ).flatten()
        rollout_truncation_rate = truncated.float().mean().item()
        nonzero_advantage_fraction = (advantages != 0).float().mean().item()
        model.eval()  # Eval mode keeps rollout and update log-probabilities dropout-free.
        with torch.inference_mode():
            old_log_probs = _completion_log_probs(model, sequences, attention_mask, completion_mask)
            with _reference_adapter(model, reference):
                reference_log_probs = _completion_log_probs(model, sequences, attention_mask, completion_mask)

        for _ in range(min(config.updates_per_rollout, config.max_steps - step)):
            current_log_probs = _completion_log_probs(model, sequences, attention_mask, completion_mask)
            ratio = (current_log_probs - old_log_probs).exp()
            clipped_ratio = ratio.clamp(1 - config.clip_epsilon, 1 + config.clip_epsilon)
            policy_loss = -torch.minimum(ratio * advantages[:, None], clipped_ratio * advantages[:, None])
            log_ratio = reference_log_probs - current_log_probs
            kl = log_ratio.exp() - log_ratio - 1
            token_count = completion_mask.sum().clamp_min(1)
            loss = ((policy_loss + config.beta * kl) * completion_mask).sum() / token_count
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite GRPO loss at step {step + 1}: {loss.item()}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                (parameter for parameter in model.parameters() if parameter.requires_grad),
                config.max_grad_norm,
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            reward = rewards.mean().item()
            if step % config.log_every == 0:
                print(
                    f"GRPO step {step}/{config.max_steps}: reward {reward:.3f}, "
                    f"exact {exact_rewards.mean().item():.1%}, loss {loss.item():.4f}"
                )
                metrics = {
                    "train/reward": reward,
                    "train/exact_reward": exact_rewards.mean().item(),
                    "train/format_reward": formatting_rewards.mean().item(),
                    "train/loss": loss.item(),
                    "train/kl": (kl * completion_mask).sum().item() / token_count.item(),
                    "train/grad_norm": grad_norm.item(),
                    "rollout/truncation_rate": rollout_truncation_rate,
                    "rollout/nonzero_advantage_fraction": nonzero_advantage_fraction,
                }
                _append_metric(config.output_dir, {"step": step, "event": "train", **metrics})
                if wandb_run is not None:
                    wandb_run.log(metrics, step=step)

            if step % config.evaluate_every == 0 or step == config.max_steps:
                padding_side = tokenizer.padding_side
                tokenizer.padding_side = "left"
                try:
                    accuracy = evaluate_gsm8k(
                        model,
                        tokenizer,
                        validation_dataset,
                        config.evaluation_max_new_tokens,
                        config.generation_batch_size,
                    ).accuracy
                finally:
                    tokenizer.padding_side = padding_side
                print(f"GRPO step {step}/{config.max_steps}: generation accuracy {accuracy:.1%}")
                _append_metric(config.output_dir, {"step": step, "event": "generation_validation", "accuracy": accuracy})
                if wandb_run is not None:
                    wandb_run.log({"generation/accuracy": accuracy}, step=step)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    _save_adapter(model, tokenizer, Path(config.output_dir) / "checkpoint-best")

            if step % config.save_every == 0 or step == config.max_steps:
                _save_adapter(model, tokenizer, Path(config.output_dir) / "checkpoint-last")

    summary = {"final_step": config.max_steps, "best_generation_accuracy": best_accuracy, "runtime_seconds": perf_counter() - start}
    _write_summary(config.output_dir, summary)
    if wandb_run is not None:
        wandb_run.summary["generation/best_accuracy"] = best_accuracy


def run(args: Namespace) -> None:
    """CLI entry point for GRPO from an SFT adapter."""
    config = GRPOConfig(
        model_name=args.model, adapter=args.adapter, output_dir=args.output_dir, batch_size=args.batch_size,
        group_size=args.group_size, learning_rate=args.learning_rate, max_steps=args.max_steps,
        updates_per_rollout=args.updates_per_rollout, max_new_tokens=args.max_new_tokens,
        evaluation_max_new_tokens=args.evaluation_max_new_tokens,
        temperature=args.temperature, beta=args.beta, clip_epsilon=args.clip_epsilon,
        format_reward=args.format_reward, max_grad_norm=args.max_grad_norm,
        log_every=args.log_every, save_every=args.save_every,
        evaluate_every=args.evaluate_every, evaluation_examples=args.evaluation_examples,
        generation_batch_size=args.generation_batch_size, seed=args.seed,
    )
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(config.output_dir) / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    from .data import load_experiment_dataset
    from .sft import SFTConfig

    datasets = load_experiment_dataset(args)
    model, tokenizer = load_lora_model(SFTConfig(model_name=config.model_name, resume_from_checkpoint=config.adapter))
    wandb_run = None
    if args.wandb_project:
        import wandb
        wandb_run = wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=asdict(config))
    try:
        train_grpo(model, tokenizer, datasets["train"], datasets["validation"], config, wandb_run)
    finally:
        if wandb_run is not None:
            wandb_run.finish()
