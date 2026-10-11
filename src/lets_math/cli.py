"""Command-line interface for lets-math experiments."""

import argparse

from . import data, evaluate, sft


DEFAULT_DATASET = "openai/gsm8k"
DEFAULT_DATASET_CONFIG = "main"
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B"


def _add_data_arguments(
    parser: argparse.ArgumentParser, *, split_default: str = "train", examples_default: int = 2
) -> None:
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face causal-LM checkpoint")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Hugging Face dataset name")
    parser.add_argument(
        "--dataset-config",
        default=DEFAULT_DATASET_CONFIG,
        help="Optional Hugging Face dataset configuration",
    )
    parser.add_argument("--split", default=split_default, help="Dataset split to use")
    parser.add_argument("--examples", type=int, default=examples_default, help="Number of examples to use")
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.1,
        help="Fraction of the source training split reserved for validation",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for deterministic splitting and training"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=576,
        help="Maximum SFT sequence length",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lets-math",
        description="SFT and hand-rolled GRPO experiments for verifiable math problems.",
    )
    parser.add_argument("--version", action="version", version="lets-math 0.1.0")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect-data", help="Inspect dataset splits and examples")
    _add_data_arguments(inspect_parser)
    inspect_parser.add_argument(
        "--show-token-stats",
        action="store_true",
        help="Load --model's tokenizer and report SFT sequence-length statistics",
    )
    inspect_parser.set_defaults(func=data.inspect_data)

    sft_parser = subparsers.add_parser("train-sft", help="Run raw-PyTorch LoRA supervised fine-tuning")
    _add_data_arguments(sft_parser)
    sft_parser.add_argument("--batch-size", type=int, default=1)
    sft_parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    sft_parser.add_argument("--learning-rate", type=float, default=1e-4)
    sft_parser.add_argument("--max-steps", type=int, default=100)
    sft_parser.add_argument("--validation-examples", type=int, default=256)
    sft_parser.add_argument("--validate-every", type=int, default=25)
    sft_parser.add_argument("--log-every", type=int, default=5)
    sft_parser.add_argument("--lora-r", type=int, default=8)
    sft_parser.add_argument("--lora-alpha", type=int, default=16)
    sft_parser.add_argument("--lora-dropout", type=float, default=0.05)
    sft_parser.add_argument("--output-dir", required=True, help="Directory for checkpoints")
    sft_parser.add_argument("--save-every", type=int, default=100)
    sft_parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Resumable checkpoint-last directory from a prior SFT run",
    )
    sft_parser.add_argument("--generation-validate-every", type=int, default=200)
    sft_parser.add_argument("--generation-validation-examples", type=int, default=100)
    sft_parser.add_argument("--generation-max-new-tokens", type=int, default=512)
    sft_parser.add_argument("--generation-batch-size", type=int, default=8)
    sft_parser.add_argument(
        "--wandb-project", default=None, help="Optional Weights & Biases project name"
    )
    sft_parser.add_argument("--wandb-run-name", default=None)
    sft_parser.set_defaults(func=sft.run)

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Greedily generate and exact-score GSM8K completions"
    )
    _add_data_arguments(evaluate_parser, split_default="validation", examples_default=100)
    evaluate_parser.add_argument("--adapter", default=None, help="Optional LoRA adapter checkpoint")
    evaluate_parser.add_argument("--max-new-tokens", type=int, default=256)
    evaluate_parser.add_argument("--generation-batch-size", type=int, default=8)
    evaluate_parser.add_argument("--show-examples", type=int, default=3)
    evaluate_parser.add_argument(
        "--output-json", default=None, help="Optional path for metrics and per-example results"
    )
    evaluate_parser.set_defaults(func=evaluate.run)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)
