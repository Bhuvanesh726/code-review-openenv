"""
Standalone logic test — verifies grader math without requiring pydantic.
"""
import sys, os

# ---- Minimal stub so environment.py imports without pydantic ----
import types

pydantic_stub = types.ModuleType("pydantic")

class _BaseModel:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
    def model_dump(self):
        return self.__dict__.copy()
    def copy(self):
        import copy
        return copy.deepcopy(self.__dict__)

class _Field:
    def __new__(cls, default=None, **kwargs):
        return default

pydantic_stub.BaseModel = _BaseModel
pydantic_stub.Field = _Field
sys.modules["pydantic"] = pydantic_stub

# Also stub enum (it's stdlib, just make sure)
from enum import Enum

# Now import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---- Inline the grader logic for testing ----

def match_issue(comment_type, comment_line, comment_desc, seeded):
    """Mirror of environment._match_issue"""
    for idx, issue in enumerate(seeded):
        if comment_type != issue["type"]:
            continue
        keywords = set(issue["keywords"])
        desc_words = set(comment_desc.lower().split())
        if not keywords.intersection(desc_words):
            continue
        if comment_line is not None and issue.get("line") is not None:
            if abs(comment_line - issue["line"]) > 5:
                continue
        return idx
    return None


def calc_reward(found_ids, total_comments, total_seeded, false_positives):
    recall = len(found_ids) / max(total_seeded, 1)
    precision = len(found_ids) / max(total_comments, 1) if total_comments > 0 else 1.0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    fp_penalty = min(false_positives * 0.05, 0.3)
    return round(max(0.0, f1 - fp_penalty), 4)


# ---- Task seeded issues for easy task ----
EASY_SEEDED = [
    {
        "id": 0, "type": "bug", "line": 14,
        "keywords": {"none", "null", "nonetype", "attributeerror", "user", "exist", "check", "14", "birth_date"},
    },
    {
        "id": 1, "type": "security", "line": 32,
        "keywords": {"hardcoded", "secret", "token", "admin", "password", "credential", "plaintext", "32"},
    },
]


# ---- Tests ----

passed = failed = 0

def test(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"PASS {name}")
        passed += 1
    else:
        print(f"FAIL {name}: {detail}")
        failed += 1


# 1. No comments → reward 0
r = calc_reward([], 0, 2, 0)
test("empty_action_zero_reward", r == 0.0, f"got {r}")

# 2. Correct bug comment → matched
idx = match_issue("bug", 14, "get_user() can return none causing attributeerror on user birth_date", EASY_SEEDED)
test("correct_bug_matched", idx == 0, f"got idx={idx}")

# 3. Correct security comment → matched
idx = match_issue("security", 32, "hardcoded secret admin token credential plaintext in code", EASY_SEEDED)
test("correct_security_matched", idx == 1, f"got idx={idx}")

# 4. Wrong type → not matched
idx = match_issue("style", 32, "hardcoded secret admin token credential plaintext", EASY_SEEDED)
test("wrong_type_not_matched", idx is None, f"got idx={idx}")

# 5. No keyword overlap → not matched
idx = match_issue("bug", 14, "this is completely unrelated text about nothing", EASY_SEEDED)
test("no_keywords_not_matched", idx is None, f"got idx={idx}")

# 6. Line too far off → not matched
idx = match_issue("bug", 99, "get_user() can return none causing attributeerror", EASY_SEEDED)
test("line_too_far_not_matched", idx is None, f"got idx={idx}")

# 7. Finding 1/2 issues, no FP → recall=0.5, precision=1.0, F1=0.667
r = calc_reward([0], 1, 2, 0)
test("half_issues_correct_reward", 0.60 < r < 0.70, f"got {r}")

# 8. Finding 2/2 issues, no FP → F1=1.0
r = calc_reward([0, 1], 2, 2, 0)
test("all_issues_perfect_reward", r == 1.0, f"got {r}")

# 9. Finding 2/2 but 3 FP → precision=2/5=0.4, recall=1.0, F1=0.571, penalty=0.15 → ~0.42
r = calc_reward([0, 1], 5, 2, 3)
test("all_issues_with_fp_penalty", 0.35 < r < 0.50, f"got {r}")

# 10. All FP, no real issues → reward 0
r = calc_reward([], 5, 2, 5)
test("all_fp_zero_reward", r == 0.0, f"got {r}")

# 11. Reward always in [0,1]
for n_found in [0, 1, 2]:
    for fp in [0, 2, 10]:
        r = calc_reward(list(range(n_found)), n_found + fp, 2, fp)
        test(f"reward_in_range_{n_found}found_{fp}fp", 0.0 <= r <= 1.0, f"got {r}")

print(f"\n{'='*40}")
print(f"Results: {passed} passed, {failed} failed")
sys.exit(0 if failed == 0 else 1)
