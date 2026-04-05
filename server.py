"""
FastAPI server wrapping CodeReviewEnv for OpenEnv HTTP API.
Endpoints: POST /reset, POST /step, GET /state
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from environment import CodeReviewEnv, CodeReviewAction, ReviewComment, IssueType, IssueSeverity

app = FastAPI(title="CodeReview OpenEnv", version="1.0.0")

# Global env instance per task (simple single-session server for HF Space)
_envs: Dict[str, CodeReviewEnv] = {}


def _get_or_create_env(task_id: str) -> CodeReviewEnv:
    if task_id not in _envs:
        _envs[task_id] = CodeReviewEnv(task_id=task_id)
    return _envs[task_id]


# ---------------------------------------------------------------------------
# Request/Response schemas for HTTP layer
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str = "easy_null_pointer"


class CommentPayload(BaseModel):
    line: Optional[int] = None
    issue_type: str
    severity: str
    description: str
    suggestion: Optional[str] = None


class StepRequest(BaseModel):
    task_id: str = "easy_null_pointer"
    comments: list[CommentPayload] = []
    summary: str = ""


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/reset")
async def reset(req: ResetRequest = None):
    """Reset environment and return initial observation."""
    if req is None:
        req = ResetRequest()
    try:
        env = _get_or_create_env(req.task_id)
        obs = env.reset()
        return obs.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step")
async def step(req: StepRequest):
    """Submit review action and get observation + reward."""
    env = _get_or_create_env(req.task_id)

    comments = []
    for c in req.comments:
        try:
            comments.append(ReviewComment(
                line=c.line,
                issue_type=IssueType(c.issue_type),
                severity=IssueSeverity(c.severity),
                description=c.description,
                suggestion=c.suggestion,
            ))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid comment: {e}")

    action = CodeReviewAction(comments=comments, summary=req.summary)
    try:
        result = env.step(action)
        return {
            "observation": result.observation.model_dump(),
            "reward": result.reward,
            "done": result.done,
            "info": result.info,
        }
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state")
async def state(task_id: str = "easy_null_pointer"):
    """Return current environment state."""
    env = _get_or_create_env(task_id)
    return env.state().model_dump()


@app.get("/tasks")
async def list_tasks():
    """List all available tasks."""
    from tasks.task_registry import TASK_REGISTRY
    return {
        "tasks": [
            {
                "id": tid,
                "filename": t["filename"],
                "language": t["language"],
                "num_issues": len(t["seeded_issues"]),
                "context": t["context"][:100] + "...",
            }
            for tid, t in TASK_REGISTRY.items()
        ]
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
