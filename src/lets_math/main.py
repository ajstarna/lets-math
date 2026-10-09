"""Dataset loading and experiment entry points."""

from argparse import Namespace
from collections.abc import Mapping
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer

from .sft import token_length_statistics


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
    stats = token_length_statistics(dataset, tokenizer, max_length)
    print(f"SFT token lengths (prompt + completion + EOS; max length {max_length}):")
    print(
        f"  examples: {stats.example_count:,} | min: {stats.minimum} | "
        f"p50: {stats.median} | p90: {stats.p90} | p95: {stats.p95} | "
        f"p99: {stats.p99} | max: {stats.maximum}"
    )
    print(f"  fitting max length: {stats.within_max_length:,}/{stats.example_count:,}")


def load_experiment_dataset(args: Namespace) -> DatasetDict:
    """Load source data and make the fixed experiment train/validation split."""
    config = args.dataset_config or None
    print(f"Loading {args.dataset}" + (f" ({config})" if config else "") + "...")
    source_dataset = load_dataset(args.dataset, config)
    return make_experiment_split(
        source_dataset, validation_fraction=args.validation_fraction, seed=args.seed
    )


def run(args: Namespace) -> None:
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
