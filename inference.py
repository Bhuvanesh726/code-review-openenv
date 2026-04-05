"""
Inference Script — Code Review OpenEnv
========================================
MANDATORY env vars:
    API_BASE_URL   The API endpoint for the LLM
    MODEL_NAME     The model identifier
    HF_TOKEN       Your Hugging Face / API key
    HF_SPACE_URL   Your deployed HF Space URL (e.g. https://your-space.hf.space)

Stdout format (strict):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import json
import os
import textwrap
from typing import List, Optional

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_SPACE_URL = os.getenv("HF_SPACE_URL", "http://localhost:7860").rstrip("/")
BENCHMARK = "code_review_openenv"
MAX_STEPS = 3
SUCCESS_SCORE_THRESHOLD = 0.5

TASKS = [
    "easy_null_pointer",
    "medium_flask_security",
    "hard_async_race",
]

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert code reviewer. You will be shown a code diff and must identify
real bugs, security vulnerabilities, and critical issues.

For each issue you find, respond with a JSON object in this exact format:
{
  "comments": [
    {
      "line": <line_number_or_null>,
      "issue_type": "<bug|security|style|performance|logic>",
      "severity": "<low|medium|high|critical>",
      "description": "<clear description of the issue>",
      "suggestion": "<optional fix suggestion>"
    }
  ],
  "summary": "<1-3 sentence overall review>"
}

Focus on: null/None dereferences, SQL injection, hardcoded secrets, missing auth,
race conditions, cryptographic weaknesses, IDOR vulnerabilities.
Return ONLY valid JSON. No markdown, no explanation outside the JSON.
""").strip()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    # Truncate action to avoid giant lines
    action_short = action[:120].replace("\n", " ") if action else "null"
    print(
        f"[STEP] step={step} action={action_short} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Environment client
# ---------------------------------------------------------------------------
def env_reset(task_id: str) -> dict:
    resp = requests.post(f"{HF_SPACE_URL}/reset", json={"task_id": task_id}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def env_step(task_id: str, comments: list, summary: str) -> dict:
    payload = {"task_id": task_id, "comments": comments, "summary": summary}
    resp = requests.post(f"{HF_SPACE_URL}/step", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------
def build_user_prompt(obs: dict, step: int) -> str:
    return textwrap.dedent(f"""
    Step {step} of {obs['max_steps']}
    File: {obs['filename']} ({obs['language']})
    Context: {obs['context']}

    Previous feedback: {obs.get('last_feedback') or 'None — this is your first attempt.'}
    Issues found so far: {obs['issues_found_so_far']} (total to find: {obs['total_issues']})

    CODE DIFF:
    {obs['diff']}

    Identify ALL bugs and security issues. Return JSON only.
    """).strip()


def get_model_action(client: OpenAI, obs: dict, step: int) -> dict:
    user_prompt = build_user_prompt(obs, step)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=1000,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        # Strip markdown fences if present
        text = text.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(text)
        return parsed
    except Exception as exc:
        print(f"[DEBUG] Model error: {exc}", flush=True)
        return {"comments": [], "summary": "Failed to parse model output"}


# ---------------------------------------------------------------------------
# Run one task episode
# ---------------------------------------------------------------------------
def run_task(client: OpenAI, task_id: str) -> tuple[float, List[float], int, bool]:
    """Returns (score, rewards, steps_taken, success)"""
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env_reset(task_id)
        done = False

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            action_dict = get_model_action(client, obs, step)
            comments = action_dict.get("comments", [])
            summary = action_dict.get("summary", "")

            action_str = f"comments={len(comments)}_summary={summary[:60]}"
            error = None

            try:
                result = env_step(task_id, comments, summary)
                obs = result["observation"]
                reward = float(result.get("reward", 0.0))
                done = bool(result.get("done", False))
            except Exception as e:
                reward = 0.0
                done = True
                error = str(e)[:100]

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

        score = rewards[-1] if rewards else 0.0  # final step reward is the episode score
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Episode error: {e}", flush=True)

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return score, rewards, steps_taken, success


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    all_scores = []
    for task_id in TASKS:
        print(f"\n{'='*60}", flush=True)
        score, rewards, steps, success = run_task(client, task_id)
        all_scores.append(score)

    avg = sum(all_scores) / len(all_scores) if all_scores else 0.0
    print(f"\n[SUMMARY] tasks={len(TASKS)} avg_score={avg:.3f} scores={','.join(f'{s:.3f}' for s in all_scores)}", flush=True)


if __name__ == "__main__":
    main()
