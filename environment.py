"""
Code Review OpenEnv Environment
================================
A real-world environment where an AI agent reviews code diffs for bugs,
security issues, and style violations. Graded against seeded known issues.
"""

from __future__ import annotations

import random
import textwrap
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from tasks.task_registry import TASK_REGISTRY


# ---------------------------------------------------------------------------
# Pydantic models — OpenEnv spec
# ---------------------------------------------------------------------------

class IssueSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IssueType(str, Enum):
    BUG = "bug"
    SECURITY = "security"
    STYLE = "style"
    PERFORMANCE = "performance"
    LOGIC = "logic"


class ReviewComment(BaseModel):
    line: Optional[int] = Field(None, description="Line number of the issue (None = whole file)")
    issue_type: IssueType
    severity: IssueSeverity
    description: str = Field(..., description="Clear description of the issue found")
    suggestion: Optional[str] = Field(None, description="Suggested fix")


class CodeReviewAction(BaseModel):
    """Action the agent submits: a list of review comments on the diff."""
    comments: List[ReviewComment] = Field(
        default_factory=list,
        description="List of review comments. Submit empty list to pass (no issues found)."
    )
    summary: str = Field(
        default="",
        description="Overall review summary (1-3 sentences)."
    )


class CodeReviewObservation(BaseModel):
    """What the agent sees each step."""
    diff: str = Field(..., description="The code diff to review (unified diff format)")
    filename: str = Field(..., description="File being reviewed")
    language: str = Field(..., description="Programming language")
    context: str = Field(..., description="Brief context about the codebase/PR")
    step: int = Field(..., description="Current step number")
    max_steps: int = Field(..., description="Maximum steps allowed")
    last_feedback: Optional[str] = Field(None, description="Feedback from last review attempt")
    issues_found_so_far: int = Field(0, description="Count of valid issues found so far")
    total_issues: int = Field(..., description="Total seeded issues to find (revealed at end)")


class CodeReviewReward(BaseModel):
    value: float = Field(..., ge=0.0, le=1.0)
    issues_found: int
    false_positives: int
    precision: float
    recall: float
    breakdown: Dict[str, float]


class CodeReviewState(BaseModel):
    task_id: str
    filename: str
    language: str
    step: int
    max_steps: int
    seeded_issues: List[Dict[str, Any]]
    found_issue_ids: List[int]
    false_positive_count: int
    done: bool
    total_reward: float


# ---------------------------------------------------------------------------
# Step result (returned by env.step)
# ---------------------------------------------------------------------------

class StepResult(BaseModel):
    observation: CodeReviewObservation
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core Environment
# ---------------------------------------------------------------------------

class CodeReviewEnv:
    """
    OpenEnv-compliant code review environment.

    The agent receives a code diff and must identify seeded issues.
    Reward is based on precision and recall of found issues.
    """

    MAX_STEPS = 3  # agent gets 3 attempts to refine their review

    def __init__(self, task_id: str = "easy_null_pointer"):
        if task_id not in TASK_REGISTRY:
            raise ValueError(f"Unknown task: {task_id}. Available: {list(TASK_REGISTRY.keys())}")
        self._task_id = task_id
        self._task = None
        self._step = 0
        self._found_issue_ids: List[int] = []
        self._false_positives = 0
        self._done = False
        self._cumulative_reward = 0.0

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------

    def reset(self) -> CodeReviewObservation:
        """Reset environment and return initial observation."""
        task_def = TASK_REGISTRY[self._task_id]
        self._task = task_def.copy()
        self._step = 0
        self._found_issue_ids = []
        self._false_positives = 0
        self._done = False
        self._cumulative_reward = 0.0
        return self._make_observation(last_feedback=None)

    def step(self, action: CodeReviewAction) -> StepResult:
        """Process agent's review and return (observation, reward, done, info)."""
        if self._done:
            raise RuntimeError("Episode is done. Call reset() first.")

        self._step += 1
        seeded = self._task["seeded_issues"]

        # Match agent comments against seeded issues
        newly_found = []
        fp_this_step = 0

        for comment in action.comments:
            matched = self._match_issue(comment, seeded)
            if matched is not None and matched not in self._found_issue_ids:
                self._found_issue_ids.append(matched)
                newly_found.append(matched)
            elif matched is None:
                fp_this_step += 1

        self._false_positives += fp_this_step

        # Calculate step reward
        recall = len(self._found_issue_ids) / max(len(seeded), 1)
        total_comments = len(action.comments)
        precision = len(self._found_issue_ids) / max(total_comments, 1) if total_comments > 0 else 1.0

        # F1 as reward, penalized for false positives
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        fp_penalty = min(self._false_positives * 0.05, 0.3)
        raw_reward = round(f1 - fp_penalty, 4)
        reward = max(0.01, min(0.99, raw_reward))

        self._cumulative_reward = reward  # use latest (not cumulative sum) for scoring

        # Done if: max steps reached OR all issues found
        done = (self._step >= self.MAX_STEPS) or (len(self._found_issue_ids) == len(seeded))
        self._done = done

        feedback = self._generate_feedback(newly_found, fp_this_step, seeded)
        obs = self._make_observation(last_feedback=feedback)

        info = {
            "newly_found_count": len(newly_found),
            "false_positives_this_step": fp_this_step,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
        }

        return StepResult(observation=obs, reward=reward, done=done, info=info)

    def state(self) -> CodeReviewState:
        """Return current environment state."""
        return CodeReviewState(
            task_id=self._task_id,
            filename=self._task["filename"] if self._task else "",
            language=self._task["language"] if self._task else "",
            step=self._step,
            max_steps=self.MAX_STEPS,
            seeded_issues=self._task["seeded_issues"] if self._task else [],
            found_issue_ids=self._found_issue_ids,
            false_positive_count=self._false_positives,
            done=self._done,
            total_reward=self._cumulative_reward,
        )

    def close(self):
        """Clean up resources."""
        self._task = None
        self._done = True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_observation(self, last_feedback: Optional[str]) -> CodeReviewObservation:
        task = self._task
        return CodeReviewObservation(
            diff=task["diff"],
            filename=task["filename"],
            language=task["language"],
            context=task["context"],
            step=self._step,
            max_steps=self.MAX_STEPS,
            last_feedback=last_feedback,
            issues_found_so_far=len(self._found_issue_ids),
            total_issues=len(task["seeded_issues"]),
        )

    def _match_issue(self, comment: ReviewComment, seeded: List[Dict]) -> Optional[int]:
        """
        Match an agent comment to a seeded issue.
        Matching logic: issue_type must match AND description keyword overlap >= 1.
        Line proximity: within ±3 lines if line is specified.
        """
        for idx, issue in enumerate(seeded):
            type_match = comment.issue_type.value == issue["type"]
            if not type_match:
                continue

            # Keyword overlap check
            keywords = set(issue["keywords"])
            desc_words = set(comment.description.lower().split())
            if not keywords.intersection(desc_words):
                continue

            # Line proximity check (lenient: ±5 or None)
            if comment.line is not None and issue.get("line") is not None:
                if abs(comment.line - issue["line"]) > 5:
                    continue

            return idx
        return None

    def _generate_feedback(self, newly_found: List[int], fp_count: int, seeded: List[Dict]) -> str:
        parts = []
        if newly_found:
            parts.append(f"Good: you identified {len(newly_found)} new issue(s) correctly.")
        if fp_count > 0:
            parts.append(f"Watch out: {fp_count} comment(s) did not match any real issue (false positives).")
        remaining = len(seeded) - len(self._found_issue_ids)
        if remaining > 0:
            parts.append(f"There are still {remaining} issue(s) left to find.")
        else:
            parts.append("All issues found!")
        return " ".join(parts) if parts else "No new issues found this step."
