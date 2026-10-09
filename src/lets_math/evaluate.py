"""Deterministic exact-answer evaluation for base, SFT, and RL checkpoints."""

from argparse import Namespace
from dataclasses import dataclass
from statistics import mean

import torch
from datasets import Dataset
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from .answers import extract_gsm8k_answer
from .sft import format_gsm8k_example


@dataclass(frozen=True)
class ScoredCompletion:
    """Verifier outcome for one generated completion."""

    expected_answer: str
    predicted_answer: str | None
    has_valid_format: bool
    is_correct: bool


@dataclass(frozen=True)
class EvaluationExample:
    question: str
    completion: str
    score: ScoredCompletion
    completion_tokens: int


@dataclass(frozen=True)
class EvaluationResult:
    examples: list[EvaluationExample]

    @property
    def accuracy(self) -> float:
        return mean(example.score.is_correct for example in self.examples)

    @property
    def format_rate(self) -> float:
        return mean(example.score.has_valid_format for example in self.examples)

    @property
    def mean_completion_tokens(self) -> float:
        return mean(example.completion_tokens for example in self.examples)


def score_completion(completion: str, gold_solution: str) -> ScoredCompletion:
    """Score one model completion against a GSM8K worked solution."""
    expected_answer = extract_gsm8k_answer(gold_solution)
    try:
        predicted_answer = extract_gsm8k_answer(completion)
    except ValueError:
        return ScoredCompletion(
            expected_answer=expected_answer,
            predicted_answer=None,
            has_valid_format=False,
            is_correct=False,
        )
    return ScoredCompletion(
        expected_answer=expected_answer,
        predicted_answer=predicted_answer,
        has_valid_format=True,
        is_correct=predicted_answer == expected_answer,
    )


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_for_evaluation(
    model_name: str, adapter: str | None = None
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a BF16 base model, optionally with a saved LoRA adapter."""
    # Decoder-only batched generation needs left padding so each prompt's final
    # real token is aligned at the final input position.
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
    model.config.pad_token_id = tokenizer.pad_token_id
    if adapter is not None:
        model = PeftModel.from_pretrained(model, adapter)
    model.to(_device())
    model.eval()
    return model, tokenizer


def generate_completions(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    max_new_tokens: int,
) -> list[tuple[str, int]]:
    """Greedily generate completion-only text for a left-padded prompt batch."""
    if not prompts:
        return []

    device = next(model.parameters()).device
    inputs = tokenizer(
        prompts, add_special_tokens=False, padding=True, return_tensors="pt"
    ).to(device)
    input_width = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    completions = []
    for sequence in generated_ids:
        completion_ids = sequence[input_width:].tolist()
        # Generation pads shorter completed sequences to the batch's longest
        # sequence. Exclude that padding (and a terminal EOS when EOS is pad).
        while completion_ids and completion_ids[-1] == tokenizer.pad_token_id:
            completion_ids.pop()
        completions.append(
            (tokenizer.decode(completion_ids, skip_special_tokens=True), len(completion_ids))
        )
    return completions


def evaluate_gsm8k(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset: Dataset,
    max_new_tokens: int,
    generation_batch_size: int,
) -> EvaluationResult:
    """Generate and score every example in a GSM8K dataset split."""
    if len(dataset) == 0:
        raise ValueError("Cannot evaluate an empty dataset")
    if max_new_tokens < 1 or generation_batch_size < 1:
        raise ValueError("max_new_tokens and generation_batch_size must be at least 1")

    examples = []
    progress = tqdm(total=len(dataset), desc="Evaluating", colour="blue")
    for start in range(0, len(dataset), generation_batch_size):
        rows = dataset[start : start + generation_batch_size]
        prompts = [
            format_gsm8k_example({"question": question, "answer": answer})["prompt"]
            for question, answer in zip(rows["question"], rows["answer"], strict=True)
        ]
        completions = generate_completions(model, tokenizer, prompts, max_new_tokens)
        for question, answer, (completion, completion_tokens) in zip(
            rows["question"], rows["answer"], completions, strict=True
        ):
            examples.append(
                EvaluationExample(
                    question=question,
                    completion=completion,
                    score=score_completion(completion, answer),
                    completion_tokens=completion_tokens,
                )
            )
        progress.update(len(rows["question"]))
    progress.close()
    return EvaluationResult(examples)


def _print_example(index: int, example: EvaluationExample) -> None:
    score = example.score
    print(f"\n[{index}] correct={score.is_correct} format={score.has_valid_format}")
    print(f"Question: {example.question}")
    print(f"Expected: {score.expected_answer!r}")
    print(f"Predicted: {score.predicted_answer!r}")
    print(f"Completion ({example.completion_tokens} tokens):\n{example.completion}")


def run(args: Namespace) -> None:
    """CLI entry point for deterministic GSM8K evaluation."""
    from .main import load_experiment_dataset

    datasets = load_experiment_dataset(args)
    if args.split not in datasets:
        available = ", ".join(datasets.keys())
        raise ValueError(f"Unknown split {args.split!r}; available splits: {available}")
    if args.examples < 1:
        raise ValueError("--examples must be at least 1 for evaluation")

    dataset = datasets[args.split].select(range(min(args.examples, len(datasets[args.split]))))
    model, tokenizer = load_model_for_evaluation(args.model, args.adapter)
    result = evaluate_gsm8k(
        model, tokenizer, dataset, args.max_new_tokens, args.generation_batch_size
    )

    correct = sum(example.score.is_correct for example in result.examples)
    valid_format = sum(example.score.has_valid_format for example in result.examples)
    print(f"\nExact-answer accuracy: {correct}/{len(result.examples)} ({result.accuracy:.1%})")
    print(f"Valid #### format: {valid_format}/{len(result.examples)} ({result.format_rate:.1%})")
    print(f"Mean completion length: {result.mean_completion_tokens:.1f} tokens")

    for index, example in enumerate(result.examples[: args.show_examples]):
        _print_example(index, example)
