from lets_math.evaluate import score_completion


def test_scores_correct_well_formed_completion():
    score = score_completion("Work\n#### 72", "Gold reasoning\n#### 72")

    assert score.has_valid_format is True
    assert score.is_correct is True
    assert score.predicted_answer == "72"


def test_scores_incorrect_but_well_formed_completion():
    score = score_completion("Work\n#### 71", "Gold reasoning\n#### 72")

    assert score.has_valid_format is True
    assert score.is_correct is False


def test_scores_completion_without_a_final_marker_as_invalid():
    score = score_completion("Work but no final answer", "Gold reasoning\n#### 72")

    assert score.has_valid_format is False
    assert score.predicted_answer is None
    assert score.is_correct is False
