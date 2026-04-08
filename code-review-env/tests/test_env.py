"""
Test suite: validates OpenEnv compliance and grader correctness.
Run with: python tests/test_env.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from env import CodeReviewEnv, TASK_IDS
from models import ReviewAction, Observation, StepReward, EnvironmentState


def test_reset_returns_observation():
    for task_id in TASK_IDS:
        env = CodeReviewEnv()
        obs = env.reset(task_id)
        assert isinstance(obs, Observation), f"reset() must return Observation for {task_id}"
        assert obs.step == 0
        assert obs.task_id == task_id
        assert len(obs.review_context.files_changed) > 0
    print("✓ reset() returns valid Observation for all tasks")


def test_state_returns_environment_state():
    env = CodeReviewEnv()
    env.reset(TASK_IDS[0])
    s = env.state()
    assert isinstance(s, EnvironmentState)
    assert s.step == 0
    print("✓ state() returns EnvironmentState")


def test_step_returns_tuple():
    env = CodeReviewEnv()
    env.reset(TASK_IDS[0])
    action = ReviewAction(
        action_type="review",
        severity="critical",
        issue_type="bug",
        line_number=3,
        description="test description",
    )
    obs, reward, done, info = env.step(action)
    assert isinstance(obs, Observation)
    assert isinstance(reward, StepReward)
    assert isinstance(done, bool)
    assert isinstance(info, dict)
    print("✓ step() returns (Observation, StepReward, bool, dict)")


def test_reward_range():
    env = CodeReviewEnv()
    env.reset(TASK_IDS[0])
    for _ in range(3):
        action = ReviewAction(action_type="review", severity="minor",
                              issue_type="style", description="some issue")
        _, reward, done, _ = env.step(action)
        assert -1.0 <= reward.value <= 1.0, f"Reward {reward.value} out of range"
        if done:
            break
    print("✓ All intermediate rewards in [-1.0, 1.0]")


def test_done_on_submit():
    env = CodeReviewEnv()
    env.reset(TASK_IDS[0])
    action = ReviewAction(action_type="submit", verdict="request_changes", confidence=0.5)
    _, _, done, info = env.step(action)
    assert done is True
    assert "final_score" in info
    assert 0.0 <= info["final_score"] <= 1.0
    print("✓ Episode terminates on submit with final_score in [0.0, 1.0]")


def test_done_on_max_steps():
    env = CodeReviewEnv()
    env.reset(TASK_IDS[0])
    max_steps = env.state().max_steps
    done = False
    for _ in range(max_steps + 5):
        action = ReviewAction(action_type="comment", comment="still reviewing")
        _, _, done, info = env.step(action)
        if done:
            break
    assert done is True, "Episode should terminate at max_steps"
    print("✓ Episode terminates at max_steps")


def test_perfect_score_task1():
    env = CodeReviewEnv()
    env.reset("task_1_easy_bug_hunt")
    actions = [
        ReviewAction(action_type="review", severity="critical", issue_type="bug",
                     line_number=3, description="assignment operator = instead of == comparison operator"),
        ReviewAction(action_type="review", severity="critical", issue_type="bug",
                     line_number=6, description="off-by-one: range should be len(numbers) not len+1 IndexError"),
        ReviewAction(action_type="review", severity="major", issue_type="bug",
                     line_number=9, description="missing return statement returns None"),
        ReviewAction(action_type="patch",
                     patched_code="def find_max(numbers):\n    if len(numbers) == 0:\n        raise ValueError()\n    max_val = numbers[0]\n    for i in range(1, len(numbers)):\n        if numbers[i] > max_val:\n            max_val = numbers[i]\n    return max_val"),
        ReviewAction(action_type="submit", verdict="request_changes", confidence=0.99),
    ]
    done = False
    for a in actions:
        if done: break
        _, _, done, info = env.step(a)
    assert info["final_score"] == 1.0, f"Expected 1.0, got {info['final_score']}"
    print("✓ Task 1 perfect score achievable")


def test_zero_score_no_actions():
    env = CodeReviewEnv()
    env.reset("task_2_medium_security")
    action = ReviewAction(action_type="submit", verdict="approve", confidence=0.1)
    _, _, done, info = env.step(action)
    assert info["final_score"] < 0.1, f"Blind approve should score near 0, got {info['final_score']}"
    print("✓ Blind approve scores near 0")


def test_repetition_penalty():
    env = CodeReviewEnv()
    env.reset(TASK_IDS[0])
    same_action = ReviewAction(action_type="review", severity="minor",
                                issue_type="style", description="identical description here")
    env.step(same_action)
    _, reward2, _, _ = env.step(same_action)
    assert reward2.breakdown.get("repetition_penalty", 0) < 0, "Repetition should be penalised"
    print("✓ Repetition penalty applied for identical descriptions")


def test_state_immutability():
    """state() should return a copy, not a live reference."""
    env = CodeReviewEnv()
    env.reset(TASK_IDS[0])
    s1 = env.state()
    env.step(ReviewAction(action_type="comment", comment="hi"))
    s2 = env.state()
    assert s1.step != s2.step, "state() must return a snapshot copy"
    print("✓ state() returns immutable snapshot")


if __name__ == "__main__":
    tests = [
        test_reset_returns_observation,
        test_state_returns_environment_state,
        test_step_returns_tuple,
        test_reward_range,
        test_done_on_submit,
        test_done_on_max_steps,
        test_perfect_score_task1,
        test_zero_score_no_actions,
        test_repetition_penalty,
        test_state_immutability,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"✗ {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed")
    sys.exit(0 if passed == len(tests) else 1)
