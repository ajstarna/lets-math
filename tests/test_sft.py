from datasets import Dataset

from lets_math.sft import (
    completion_length_statistics,
    format_gsm8k_example,
    token_length_statistics,
    tokenize_sft_example,
)


class FakeTokenizer:
    eos_token_id = 99

    def __call__(self, text, add_special_tokens):
        assert add_special_tokens is False
        return {"input_ids": [ord(character) for character in text]}


def test_formats_gsm8k_as_prompt_and_completion():
    formatted = format_gsm8k_example(
        {"question": "What is 2 + 2?", "answer": "2 + 2 = 4.\n#### 4"}
    )

    assert formatted["prompt"].endswith("Problem: What is 2 + 2?\n\nSolution:\n")
    assert formatted["completion"] == "2 + 2 = 4.\n#### 4"


def test_masks_prompt_tokens_and_trains_on_completion_tokens():
    tokenized = tokenize_sft_example(
        {"prompt": "Q", "completion": "A"}, tokenizer=FakeTokenizer()
    )

    assert tokenized == {
        "input_ids": [ord("Q"), ord("A"), 99],
        "attention_mask": [1, 1, 1],
        "labels": [-100, ord("A"), 99],
    }


def test_reports_lengths_for_the_formatted_sft_sequences():
    dataset = Dataset.from_list(
        [
            {"question": "Short?", "answer": "#### 1"},
            {"question": "A longer question?", "answer": "Reasoning\n#### 2"},
        ]
    )

    stats = token_length_statistics(dataset, FakeTokenizer(), max_length=200)

    assert stats.example_count == 2
    assert stats.minimum <= stats.median <= stats.maximum
    assert stats.p90 <= stats.p95 <= stats.p99 == stats.maximum
    assert stats.within_max_length == 2


def test_reports_completion_only_lengths():
    dataset = Dataset.from_list(
        [
            {"question": "One?", "answer": "#### 1"},
            {"question": "Two?", "answer": "Reasoning\n#### 2"},
        ]
    )

    stats = completion_length_statistics(dataset, FakeTokenizer(), max_length=200)

    # Completion labels contain only the answer text plus the EOS token.
    assert stats.minimum == len("#### 1") + 1
    assert stats.maximum == len("Reasoning\n#### 2") + 1
    assert stats.within_max_length == 2
