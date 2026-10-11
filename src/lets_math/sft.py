"""Data preparation and a small raw-PyTorch SFT trainer."""

from argparse import Namespace
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
from time import perf_counter
from typing import Any

import torch
from datasets import Dataset, DatasetDict
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


PROMPT_TEMPLATE = """Solve the following math problem. Show your work, then end with `#### <answer>`.

Problem: {question}

Solution:
"""


@dataclass(frozen=True)
class SFTConfig:
    """Parameters for one raw-PyTorch SFT training run."""

    model_name: str
    max_length: int = 576
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-4
    max_steps: int = 100
    validation_examples: int = 256
    validate_every: int = 25
    log_every: int = 5
    seed: int = 42
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    output_dir: str | None = None
    save_every: int = 100
    resume_from_checkpoint: str | None = None
    generation_validate_every: int = 200
    generation_validation_examples: int = 100
    generation_max_new_tokens: int = 512
    generation_batch_size: int = 8


@dataclass(frozen=True)
class TokenLengthStatistics:
    """Summary of tokenized prompt-plus-completion sequence lengths."""

    example_count: int
    minimum: int
    median: int
    p90: int
    p95: int
    p99: int
    maximum: int
    within_max_length: int


def format_gsm8k_example(example: Mapping[str, str]) -> dict[str, str]:
    """Convert one GSM8K row to the prompt/completion format used for SFT."""
    return {
        "prompt": PROMPT_TEMPLATE.format(question=example["question"]),
        "completion": example["answer"],
    }


def tokenize_sft_example(
    example: Mapping[str, str], tokenizer: PreTrainedTokenizerBase
) -> dict[str, list[int]]:
    """Tokenize a prompt/completion pair and mask prompt tokens from the loss.

    The model receives both prompt and completion tokens. Labels for prompt tokens
    are ``-100``, which PyTorch's cross-entropy loss ignores; consequently SFT
    learns only to predict the worked solution and final answer.
    """
    prompt_ids = tokenizer(example["prompt"], add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(example["completion"], add_special_tokens=False)["input_ids"]
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define an EOS token for SFT")
    completion_ids = [*completion_ids, tokenizer.eos_token_id]

    input_ids = [*prompt_ids, *completion_ids]
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + completion_ids,
    }


def token_length_statistics(
    dataset: Dataset, tokenizer: PreTrainedTokenizerBase, max_length: int
) -> TokenLengthStatistics:
    """Measure lengths of the exact sequences that SFT would train on."""
    if len(dataset) == 0:
        raise ValueError("Cannot compute token statistics for an empty dataset")
    if max_length < 1:
        raise ValueError("max_length must be positive")

    lengths = sorted(
        len(tokenize_sft_example(format_gsm8k_example(example), tokenizer)["input_ids"])
        for example in dataset
    )

    def percentile(percent: float) -> int:
        return lengths[math.ceil(percent * len(lengths)) - 1]

    return TokenLengthStatistics(
        example_count=len(lengths),
        minimum=lengths[0],
        median=percentile(0.5),
        p90=percentile(0.9),
        p95=percentile(0.95),
        p99=percentile(0.99),
        maximum=lengths[-1],
        within_max_length=sum(length <= max_length for length in lengths),
    )


def prepare_gsm8k_sft_dataset(
    dataset: Dataset, tokenizer: PreTrainedTokenizerBase, max_length: int
) -> Dataset:
    """Format, tokenize, and discard examples exceeding ``max_length`` tokens."""
    if max_length < 1:
        raise ValueError("max_length must be positive")

    formatted = dataset.map(format_gsm8k_example, remove_columns=dataset.column_names)
    tokenized = formatted.map(
        lambda example: tokenize_sft_example(example, tokenizer),
        remove_columns=formatted.column_names,
    )
    return tokenized.filter(lambda example: len(example["input_ids"]) <= max_length)


def load_lora_model(config: SFTConfig) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a BF16 causal LM and attach trainable LoRA adapters."""
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, padding_side="right")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        dtype=torch.bfloat16,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    if config.resume_from_checkpoint is not None:
        model = PeftModel.from_pretrained(
            model, config.resume_from_checkpoint, is_trainable=True
        )
        print(f"Loaded trainable LoRA adapter from {config.resume_from_checkpoint}")
    else:
        model = get_peft_model(
            model,
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=["q_proj", "v_proj"],
            ),
        )
    model.print_trainable_parameters()
    return model, tokenizer


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _data_loader(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
    *,
    shuffle: bool,
    seed: int | None = None,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        collate_fn=DataCollatorForTokenClassification(
            tokenizer, label_pad_token_id=-100, return_tensors="pt"
        ),
    )


def validate_sft(
    model: PreTrainedModel,
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
) -> float:
    """Return mean next-token loss over non-masked completion tokens."""
    loader = _data_loader(dataset, tokenizer, batch_size, shuffle=False)
    device = next(model.parameters()).device
    total_loss = 0.0
    total_target_tokens = 0
    was_training = model.training
    model.eval()

    with torch.inference_mode():
        for batch in loader:
            batch = {name: value.to(device) for name, value in batch.items()}
            target_tokens = (batch["labels"] != -100).sum().item()
            total_loss += model(**batch).loss.item() * target_tokens
            total_target_tokens += target_tokens

    if was_training:
        model.train()
    return total_loss / total_target_tokens


def _infinite_train_batches(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
    seed: int,
    start_epoch: int = 0,
    start_batch_offset: int = 0,
) -> Iterator[tuple[dict[str, torch.Tensor], int, int]]:
    """Yield batches plus the reproducible position after each yielded batch."""
    epoch = start_epoch
    batch_offset = start_batch_offset
    while True:
        loader = _data_loader(dataset, tokenizer, batch_size, shuffle=True, seed=seed + epoch)
        batches = iter(loader)
        for _ in range(batch_offset):
            next(batches)
        for batch_offset, batch in enumerate(batches, start=batch_offset + 1):
            yield batch, epoch, batch_offset
        epoch += 1
        batch_offset = 0


def _append_metric(output_dir: str | None, record: dict[str, Any]) -> None:
    """Append one machine-readable training event when local output is enabled."""
    if output_dir is None:
        return
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with (output_path / "metrics.jsonl").open("a") as file:
        file.write(json.dumps(record) + "\n")


def _write_summary(output_dir: str | None, summary: dict[str, Any]) -> None:
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def _save_adapter(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    checkpoint_dir: Path,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)


def _save_best_checkpoint(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, output_dir: str
) -> None:
    """Overwrite the single best-validation LoRA adapter checkpoint."""
    checkpoint_dir = Path(output_dir) / "checkpoint-best"
    _save_adapter(model, tokenizer, checkpoint_dir)
    print(f"Saved best checkpoint to {checkpoint_dir}")


def _save_best_accuracy_checkpoint(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, output_dir: str
) -> None:
    """Overwrite the single checkpoint selected by generated-answer accuracy."""
    checkpoint_dir = Path(output_dir) / "checkpoint-best-accuracy"
    _save_adapter(model, tokenizer, checkpoint_dir)
    print(f"Saved best accuracy checkpoint to {checkpoint_dir}")


def _save_last_checkpoint(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    optimizer: torch.optim.Optimizer,
    output_dir: str,
    optimizer_step: int,
    epoch: int,
    batch_offset: int,
    seed: int,
    best_validation_loss: float,
    best_generation_accuracy: float | None,
) -> None:
    """Overwrite the resumable checkpoint, including all training state."""
    checkpoint_dir = Path(output_dir) / "checkpoint-last"
    _save_adapter(model, tokenizer, checkpoint_dir)
    state: dict[str, Any] = {
        "optimizer": optimizer.state_dict(),
        "optimizer_step": optimizer_step,
        "epoch": epoch,
        "batch_offset": batch_offset,
        "seed": seed,
        "best_validation_loss": best_validation_loss,
        "best_generation_accuracy": best_generation_accuracy,
        "python_rng_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda_rng_states"] = torch.cuda.get_rng_state_all()
    torch.save(state, checkpoint_dir / "training_state.pt")
    print(f"Saved resumable checkpoint to {checkpoint_dir}")


def _load_training_state(
    checkpoint_dir: str,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    seed: int,
) -> tuple[int, int, int, float, float | None]:
    """Restore optimizer/RNG/data-position state from ``checkpoint-last``."""
    state_path = Path(checkpoint_dir) / "training_state.pt"
    if not state_path.is_file():
        raise ValueError(
            f"{checkpoint_dir} has no training_state.pt; use checkpoint-last, not checkpoint-best"
        )
    state = torch.load(state_path, map_location=device, weights_only=False)
    if state["seed"] != seed:
        raise ValueError(
            f"Checkpoint seed ({state['seed']}) does not match --seed ({seed})"
        )
    optimizer.load_state_dict(state["optimizer"])
    random.setstate(state["python_rng_state"])
    torch.set_rng_state(state["torch_rng_state"])
    if "cuda_rng_states" in state:
        if not torch.cuda.is_available():
            raise ValueError("Checkpoint contains CUDA RNG state, but CUDA is unavailable")
        torch.cuda.set_rng_state_all(state["cuda_rng_states"])
    return (
        state["optimizer_step"],
        state["epoch"],
        state["batch_offset"],
        state["best_validation_loss"],
        state.get("best_generation_accuracy"),
    )


def _generated_validation_accuracy(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset: Dataset,
    config: SFTConfig,
) -> float:
    """Run deterministic answer evaluation without changing later training mode."""
    from .evaluate import evaluate_gsm8k

    was_training = model.training
    original_padding_side = tokenizer.padding_side
    model.eval()
    tokenizer.padding_side = "left"
    try:
        result = evaluate_gsm8k(
            model,
            tokenizer,
            dataset,
            config.generation_max_new_tokens,
            config.generation_batch_size,
        )
    finally:
        tokenizer.padding_side = original_padding_side
        if was_training:
            model.train()
    return result.accuracy


def train_sft(
    model: PreTrainedModel,
    datasets: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    config: SFTConfig,
    generation_validation_dataset: Dataset,
    wandb_run: Any | None = None,
) -> None:
    """Train LoRA parameters with explicit validation and resumable state."""
    if config.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be at least 1")
    if config.max_steps < 1 or config.validation_examples < 1:
        raise ValueError("max_steps and validation_examples must be at least 1")
    if min(config.validate_every, config.log_every, config.save_every) < 1:
        raise ValueError("validation, logging, and save intervals must be at least 1")
    if min(config.generation_validate_every, config.generation_validation_examples) < 1:
        raise ValueError("generation validation intervals and examples must be at least 1")

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    device = _device()
    model.to(device)
    validation_dataset = datasets["validation"].select(
        range(min(config.validation_examples, len(datasets["validation"])))
    )
    generation_validation_dataset = generation_validation_dataset.select(
        range(min(config.generation_validation_examples, len(generation_validation_dataset)))
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
    )
    print(
        f"Training on {device}; effective batch size: "
        f"{config.batch_size * config.gradient_accumulation_steps}"
    )

    optimizer_step = 0
    epoch = 0
    batch_offset = 0
    best_generation_accuracy: float | None = None
    if config.resume_from_checkpoint is not None:
        (
            optimizer_step,
            epoch,
            batch_offset,
            best_validation_loss,
            best_generation_accuracy,
        ) = _load_training_state(
            config.resume_from_checkpoint, optimizer, device, config.seed
        )
        if optimizer_step >= config.max_steps:
            raise ValueError(
                f"Checkpoint is already at step {optimizer_step}, which is >= --max-steps {config.max_steps}"
            )
        print(
            f"Resumed at step {optimizer_step}, epoch {epoch}, batch offset {batch_offset}"
        )
    else:
        best_validation_loss = validate_sft(
            model, validation_dataset, tokenizer, config.batch_size
        )
        print(f"Validation loss before training: {best_validation_loss:.4f}")
        if config.output_dir is not None:
            _save_best_checkpoint(model, tokenizer, config.output_dir)
            _save_last_checkpoint(
                model, tokenizer, optimizer, config.output_dir, 0, 0, 0,
                config.seed, best_validation_loss, best_generation_accuracy,
            )
        _append_metric(
            config.output_dir,
            {"step": 0, "event": "validation", "loss": best_validation_loss},
        )
        if wandb_run is not None:
            wandb_run.log({"validation/loss": best_validation_loss}, step=0)

    run_start = perf_counter()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    batches = _infinite_train_batches(
        datasets["train"], tokenizer, config.batch_size, config.seed, epoch, batch_offset
    )

    for step in range(optimizer_step + 1, config.max_steps + 1):
        accumulated_loss = 0.0
        for _ in range(config.gradient_accumulation_steps):
            batch, epoch, batch_offset = next(batches)
            batch = {name: value.to(device) for name, value in batch.items()}
            loss = model(**batch).loss
            accumulated_loss += loss.item()
            (loss / config.gradient_accumulation_steps).backward()

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        mean_train_loss = accumulated_loss / config.gradient_accumulation_steps

        if step % config.log_every == 0:
            print(f"SFT step {step}/{config.max_steps}: train loss {mean_train_loss:.4f}")
            _append_metric(
                config.output_dir,
                {"step": step, "event": "train", "loss": mean_train_loss},
            )
            if wandb_run is not None:
                wandb_run.log({"train/loss": mean_train_loss}, step=step)

        if step % config.validate_every == 0 or step == config.max_steps:
            validation_loss = validate_sft(
                model, validation_dataset, tokenizer, config.batch_size
            )
            print(f"SFT step {step}/{config.max_steps}: validation loss {validation_loss:.4f}")
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                if config.output_dir is not None:
                    _save_best_checkpoint(model, tokenizer, config.output_dir)
            _append_metric(
                config.output_dir,
                {"step": step, "event": "validation", "loss": validation_loss},
            )
            if wandb_run is not None:
                wandb_run.log({"validation/loss": validation_loss}, step=step)

        if step % config.generation_validate_every == 0 or step == config.max_steps:
            accuracy = _generated_validation_accuracy(
                model, tokenizer, generation_validation_dataset, config
            )
            print(f"SFT step {step}/{config.max_steps}: generation accuracy {accuracy:.1%}")
            if best_generation_accuracy is None or accuracy > best_generation_accuracy:
                best_generation_accuracy = accuracy
                if config.output_dir is not None:
                    _save_best_accuracy_checkpoint(model, tokenizer, config.output_dir)
            _append_metric(
                config.output_dir,
                {"step": step, "event": "generation_validation", "accuracy": accuracy},
            )
            if wandb_run is not None:
                wandb_run.log({"generation/accuracy": accuracy}, step=step)

        if step % config.save_every == 0 or step == config.max_steps:
            if config.output_dir is not None:
                _save_last_checkpoint(
                    model, tokenizer, optimizer, config.output_dir, step, epoch,
                    batch_offset, config.seed, best_validation_loss, best_generation_accuracy,
                )

    summary = {
        "final_step": config.max_steps,
        "best_validation_loss": best_validation_loss,
        "best_generation_accuracy": best_generation_accuracy,
        "runtime_seconds": perf_counter() - run_start,
    }
    _write_summary(config.output_dir, summary)
    if wandb_run is not None:
        wandb_run.summary["validation/best_loss"] = best_validation_loss
        wandb_run.summary["generation/best_accuracy"] = best_generation_accuracy


def run(args: Namespace) -> None:
    """Prepare GSM8K and run one configured SFT experiment."""
    config = SFTConfig(
        model_name=args.model,
        max_length=args.max_length,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        validation_examples=args.validation_examples,
        validate_every=args.validate_every,
        log_every=args.log_every,
        seed=args.seed,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        output_dir=args.output_dir,
        save_every=args.save_every,
        resume_from_checkpoint=args.resume_from_checkpoint,
        generation_validate_every=args.generation_validate_every,
        generation_validation_examples=args.generation_validation_examples,
        generation_max_new_tokens=args.generation_max_new_tokens,
        generation_batch_size=args.generation_batch_size,
    )
    if config.output_dir is not None:
        output_path = Path(config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")

    # LoRA parameters are initialized during model loading, so seed beforehand.
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    from .main import load_experiment_dataset

    raw_datasets = load_experiment_dataset(args)
    model, tokenizer = load_lora_model(config)
    datasets = DatasetDict(
        {
            name: prepare_gsm8k_sft_dataset(raw_datasets[name], tokenizer, config.max_length)
            for name in ("train", "validation")
        }
    )
    for name in datasets:
        print(f"SFT {name}: {len(datasets[name]):,} examples")

    wandb_run = None
    if args.wandb_project:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=asdict(config),
        )
    try:
        train_sft(
            model,
            datasets,
            tokenizer,
            config,
            generation_validation_dataset=raw_datasets["validation"],
            wandb_run=wandb_run,
        )
    finally:
        if wandb_run is not None:
            wandb_run.finish()
