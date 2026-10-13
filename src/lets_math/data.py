"""Dataset loading and experiment entry points."""

from argparse import Namespace
from collections.abc import Mapping
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer

from .sft import completion_length_statistics, token_length_statistics


def _format_example(example: Mapping[str, Any]) -> str:
    return "\n".join(f"  {key}: {value!r}" for key, value in example.items())


def make_experiment_split(
    dataset: DatasetDict, validation_fraction: float, seed: int
) -> DatasetDict:
    """Split the source training set while preserving any official test split."""
    if not 0 < validation_fraction < 1:
        raise ValueError("--validation-fraction must be between 0 and 1")
    if "train" not in dataset:
        raise ValueError("Dataset must provide a 'train' split for experiment splitting")

    split = dataset["train"].train_test_split(test_size=validation_fraction, seed=seed)
    experiment_dataset = DatasetDict(train=split["train"], validation=split["test"])
    if "test" in dataset:
        experiment_dataset["test"] = dataset["test"]
    return experiment_dataset


def _print_dataset_details(dataset: DatasetDict) -> None:
    print("Experiment splits:")
    for name, split in dataset.items():
        print(f"  {name}: {len(split):,} examples")
    print(f"Features: {dataset['train'].features}")


def _print_token_statistics(dataset: Dataset, model: str, max_length: int) -> None:
    print(f"\nLoading tokenizer from {model}...")
    tokenizer = AutoTokenizer.from_pretrained(model)
    sequence_stats = token_length_statistics(dataset, tokenizer, max_length)
    completion_stats = completion_length_statistics(dataset, tokenizer, max_length)
    print(f"SFT sequence lengths (prompt + completion + EOS; max length {max_length}):")
    print(
        f"  examples: {sequence_stats.example_count:,} | min: {sequence_stats.minimum} | "
        f"p50: {sequence_stats.median} | p90: {sequence_stats.p90} | p95: {sequence_stats.p95} | "
        f"p99: {sequence_stats.p99} | max: {sequence_stats.maximum}"
    )
    print(
        f"  fitting max length: {sequence_stats.within_max_length:,}/{sequence_stats.example_count:,}"
    )
    print("Gold completion lengths (solution + final answer + EOS):")
    print(
        f"  min: {completion_stats.minimum} | p50: {completion_stats.median} | "
        f"p90: {completion_stats.p90} | p95: {completion_stats.p95} | "
        f"p99: {completion_stats.p99} | max: {completion_stats.maximum}"
    )


def load_experiment_dataset(args: Namespace) -> DatasetDict:
    """Load source data and make the fixed experiment train/validation split."""
    config = args.dataset_config or None
    print(f"Loading {args.dataset}" + (f" ({config})" if config else "") + "...")
    source_dataset = load_dataset(args.dataset, config)
    return make_experiment_split(
        source_dataset, validation_fraction=args.validation_fraction, seed=args.seed
    )


def inspect_data(args: Namespace) -> None:
    """Load a dataset, create an experiment split, and display examples."""
    if args.examples < 0:
        raise ValueError("--examples must be >= 0")

    dataset = load_experiment_dataset(args)
    _print_dataset_details(dataset)
    if args.show_token_stats:
        _print_token_statistics(dataset["train"], args.model, args.max_length)

    if args.split not in dataset:
        available = ", ".join(dataset.keys())
        raise ValueError(f"Unknown split {args.split!r}; available splits: {available}")

    split: Dataset = dataset[args.split]
    example_count = min(args.examples, len(split))
    print(f"\nExamples from {args.split} ({example_count} shown):")
    for index in range(example_count):
        print(f"\n[{index}]")
        print(_format_example(split[index]))
