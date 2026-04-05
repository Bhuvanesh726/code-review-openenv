"""
Tests for CodeReviewEnv — verifies OpenEnv spec compliance and grader logic.
Run: python tests/test_environment.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import (
    CodeReviewEnv, CodeReviewAction, ReviewComment,
    IssueType, IssueSeverity
)


def test_reset_returns_observation():
    env = CodeReviewEnv("easy_null_pointer")
    obs = env.reset()
    assert obs.diff, "Observation must have a diff"
    assert obs.filename == "user_service.py"
    assert obs.step == 0
    assert obs.max_steps == 3
    assert obs.issues_found_so_far == 0
    assert obs.total_issues == 2
    print("PASS test_reset_returns_observation")


def test_step_returns_reward_in_range():
    env = CodeReviewEnv("easy_null_pointer")
    env.reset()
    action = CodeReviewAction(comments=[], summary="No issues found")
    result = env.step(action)
    assert 0.0 <= result.reward <= 1.0, f"Reward out of range: {result.reward}"
    print(f"PASS test_step_returns_reward_in_range (reward={result.reward})")


def test_correct_issue_gives_reward():
    env = CodeReviewEnv("easy_null_pointer")
    env.reset()
    action = CodeReviewAction(
        comments=[
            ReviewComment(
                line=14,
                issue_type=IssueType.BUG,
                severity=IssueSeverity.HIGH,
                description="get_user() can return none causing attributeerror on user birth_date",
            )
        ],
        summary="Found null dereference"
    )
    result = env.step(action)
    assert result.reward > 0.0, "Correct issue should give positive reward"
    print(f"PASS test_correct_issue_gives_reward (reward={result.reward})")


def test_false_positive_penalizes():
    env = CodeReviewEnv("easy_null_pointer")
    env.reset()
    # Submit 5 fake issues
    action = CodeReviewAction(
        comments=[
            ReviewComment(
                line=i,
                issue_type=IssueType.BUG,
                severity=IssueSeverity.LOW,
                description=f"Fake issue {i} completely made up",
            )
            for i in range(5)
        ],
        summary="Lots of fake issues"
    )
    result = env.step(action)
    # Reward should be very low due to false positives
    assert result.reward < 0.5, f"False positives should lower reward: {result.reward}"
    print(f"PASS test_false_positive_penalizes (reward={result.reward})")


def test_done_after_max_steps():
    env = CodeReviewEnv("easy_null_pointer")
    env.reset()
    action = CodeReviewAction(comments=[], summary="")
    for i in range(3):
        result = env.step(action)
    assert result.done, "Should be done after max_steps"
    print("PASS test_done_after_max_steps")


def test_done_when_all_issues_found():
    env = CodeReviewEnv("easy_null_pointer")
    env.reset()
    action = CodeReviewAction(
        comments=[
            ReviewComment(
                line=14,
                issue_type=IssueType.BUG,
                severity=IssueSeverity.HIGH,
                description="get_user() can return none causing attributeerror on user birth_date",
            ),
            ReviewComment(
                line=32,
                issue_type=IssueType.SECURITY,
                severity=IssueSeverity.CRITICAL,
                description="hardcoded secret admin token credential plaintext in code",
            ),
        ],
        summary="Found both issues"
    )
    result = env.step(action)
    assert result.done, "Should be done when all issues found"
    assert result.reward > 0.5, f"Finding all issues should give high reward: {result.reward}"
    print(f"PASS test_done_when_all_issues_found (reward={result.reward})")


def test_state_method():
    env = CodeReviewEnv("easy_null_pointer")
    env.reset()
    state = env.state()
    assert state.task_id == "easy_null_pointer"
    assert state.step == 0
    assert not state.done
    print("PASS test_state_method")


def test_all_tasks_loadable():
    from tasks.task_registry import TASK_REGISTRY
    for task_id in TASK_REGISTRY:
        env = CodeReviewEnv(task_id)
        obs = env.reset()
        assert obs.diff, f"Task {task_id} must have a diff"
        assert obs.total_issues > 0, f"Task {task_id} must have seeded issues"
        print(f"PASS test_all_tasks_loadable [{task_id}] ({obs.total_issues} issues)")


def test_scores_in_range_all_tasks():
    from tasks.task_registry import TASK_REGISTRY
    for task_id in TASK_REGISTRY:
        env = CodeReviewEnv(task_id)
        env.reset()
        action = CodeReviewAction(comments=[], summary="no issues")
        result = env.step(action)
        assert 0.0 <= result.reward <= 1.0, f"Reward out of [0,1]: {result.reward}"
    print("PASS test_scores_in_range_all_tasks")


if __name__ == "__main__":
    tests = [
        test_reset_returns_observation,
        test_step_returns_reward_in_range,
        test_correct_issue_gives_reward,
        test_false_positive_penalizes,
        test_done_after_max_steps,
        test_done_when_all_issues_found,
        test_state_method,
        test_all_tasks_loadable,
        test_scores_in_range_all_tasks,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"FAIL {t.__name__}: {e}")
            failed += 1

    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
