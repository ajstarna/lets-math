import pytest

from lets_math.answers import extract_gsm8k_answer


def test_extracts_final_answer_after_worked_solution():
    solution = "First compute 8 + 4 = 12.\n#### 12\n"

    assert extract_gsm8k_answer(solution) == "12"


def test_preserves_non_integer_answer_text():
    assert extract_gsm8k_answer("Reasoning\n#### -1.5") == "-1.5"


def test_rejects_solution_without_trailing_marker():
    with pytest.raises(ValueError, match="trailing"):
        extract_gsm8k_answer("Reasoning only\n#### 12\nMore text")
