import pytest

from structured_reasoning import Step, extract_steps, group_lcs_rewards, maxflow_reward


def test_extract_steps_ignores_unclosed_tags():
    steps = extract_steps("<plan>Use x.</plan><check>Verify.</check><broken>oops")
    assert [(step.tag, step.content) for step in steps] == [
        ("plan", "Use x."),
        ("check", "Verify."),
    ]


def test_maxflow_identifies_required_middle_step():
    result = maxflow_reward([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    assert result.max_flow == pytest.approx(1.0)
    assert result.importance[1] > 0.9
    assert result.reward > 0


def test_maxflow_rejects_invalid_matrix():
    with pytest.raises(ValueError):
        maxflow_reward([[0, 1]])


def test_lcs_group_separates_correct_and_wrong_samples():
    common = [Step("plan", "let us solve the equation"), Step("answer", "x equals two")]
    different = [Step("guess", "maybe it is seven")]
    rewards = group_lcs_rewards([common, common, different], [True, True, False])
    assert rewards[0] >= 0
    assert rewards[1] >= 0
    assert rewards[2] < 0
