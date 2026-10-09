"""Answer extraction for datasets with explicit final-answer markers."""

import re


_GSM8K_FINAL_ANSWER = re.compile(r"####\s*([^\n]+)\s*$")


def extract_gsm8k_answer(solution: str) -> str:
    """Return GSM8K's final answer, without its trailing ``####`` marker.

    GSM8K solutions may contain arbitrary worked reasoning, but their final line
    must be ``#### <answer>``. Keeping this parser strict prevents malformed
    training data from quietly receiving a reward.
    """
    match = _GSM8K_FINAL_ANSWER.search(solution)
    if match is None:
        raise ValueError("GSM8K solution has no trailing '#### <answer>' marker")
    return match.group(1).strip()
