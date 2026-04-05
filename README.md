# Code Review OpenEnv

A real-world [OpenEnv](https://github.com/openenv) environment where an AI agent reviews code diffs to identify bugs, security vulnerabilities, and logic errors — graded against programmatically seeded known issues.

## Why This Environment?

Code review is a high-value knowledge-worker task done millions of times daily. Automating or augmenting it with AI could save enormous engineering time. This environment provides:

- Realistic diffs from common bug patterns (OWASP Top 10, race conditions, financial logic bugs)
- Deterministic, reproducible grading via issue-matching
- Dense reward signal — partial credit for finding subsets of issues
- Genuine difficulty progression from trivial bugs to subtle concurrency flaws

---

## Action Space

The agent submits a `CodeReviewAction`:

```json
{
  "comments": [
    {
      "line": 14,
      "issue_type": "bug",
      "severity": "high",
      "description": "get_user() can return None causing AttributeError",
      "suggestion": "Add a None check before accessing user['birth_date']"
    }
  ],
  "summary": "This PR has a null dereference and a hardcoded credential."
}
```

**issue_type** values: `bug`, `security`, `style`, `performance`, `logic`  
**severity** values: `low`, `medium`, `high`, `critical`

---

## Observation Space

Each step the agent receives:

| Field | Type | Description |
|---|---|---|
| `diff` | string | Unified diff of the code |
| `filename` | string | File being reviewed |
| `language` | string | Programming language |
| `context` | string | PR description / context |
| `step` | int | Current step (1-indexed) |
| `max_steps` | int | Max steps per episode (3) |
| `last_feedback` | string\|null | Feedback from previous step |
| `issues_found_so_far` | int | Correctly identified issues so far |
| `total_issues` | int | Total seeded issues to find |

---

## Reward Function

**Reward = F1(precision, recall) − false_positive_penalty**

- **Recall** = correctly found issues / total seeded issues
- **Precision** = correctly found issues / total comments submitted
- **F1** = harmonic mean of precision and recall
- **False positive penalty** = 0.05 per false positive, capped at 0.3

This means:
- Finding all issues perfectly = 1.0
- Finding half the issues with no false positives = ~0.67
- Spamming random comments = penalized toward 0

---

## Tasks

### Easy — `easy_null_pointer`
**File:** `user_service.py` | **Seeded issues:** 2  
Review a simple Python service class. Find a `None` dereference and a hardcoded credential.  
**Expected baseline score:** ~0.75

### Medium — `medium_flask_security`
**File:** `auth_routes.py` | **Seeded issues:** 5  
Review Flask auth routes for OWASP Top 10 violations: SQL injection, MD5 password hashing, missing authorization, weak token entropy, and token leakage.  
**Expected baseline score:** ~0.55

### Hard — `hard_async_race`
**File:** `payment_processor.py` | **Seeded issues:** 4  
Review async payment processing code. Requires understanding of race conditions, TOCTOU bugs in financial code, IDOR vulnerabilities, and float precision loss on money values. Challenges frontier models.  
**Expected baseline score:** ~0.30

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/reset` | Reset environment, get initial observation |
| `POST` | `/step` | Submit review action, get reward |
| `GET` | `/state` | Get current environment state |
| `GET` | `/tasks` | List all available tasks |
| `GET` | `/health` | Health check |

### Reset request
```json
{ "task_id": "easy_null_pointer" }
```

### Step request
```json
{
  "task_id": "easy_null_pointer",
  "comments": [...],
  "summary": "..."
}
```

---

## Setup & Running

### Local (Python)
```bash
pip install -r requirements.txt
python server.py
# Server starts at http://localhost:7860
```

### Docker
```bash
docker build -t code-review-env .
docker run -p 7860:7860 code-review-env
```

### Run inference baseline
```bash
export HF_TOKEN=your_hf_token
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export HF_SPACE_URL=https://your-space.hf.space

python inference.py
```

---

## Baseline Scores

Tested with `Qwen/Qwen2.5-72B-Instruct`:

| Task | Score |
|---|---|
| easy_null_pointer | 0.75 |
| medium_flask_security | 0.55 |
| hard_async_race | 0.30 |
| **Average** | **0.53** |

---

## Project Structure

```
code-review-openenv/
├── environment.py        # Core CodeReviewEnv class (OpenEnv spec)
├── server.py             # FastAPI HTTP server
├── inference.py          # Baseline inference script (mandatory)
├── openenv.yaml          # Environment metadata
├── requirements.txt
├── Dockerfile
├── README.md
└── tasks/
    ├── __init__.py
    └── task_registry.py  # Task definitions with seeded issues
```
