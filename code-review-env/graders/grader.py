"""
Programmatic graders for all three tasks.
Each grader returns a score in [0.0, 1.0] with a breakdown dict.
Grading is deterministic: keyword matching + structural checks.
"""
from __future__ import annotations
import ast
import re
from typing import Any, Dict, List, Tuple

from models import ReviewAction, EnvironmentState


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _keywords_hit(text: str, keywords: List[str]) -> bool:
    """Return True if any keyword appears in text (case-insensitive)."""
    t = text.lower()
    return any(kw.lower() in t for kw in keywords)


def _actions_mention_bug(actions: List[ReviewAction], bug: Dict[str, Any]) -> bool:
    """Check whether any action mentions the given bug via keyword matching."""
    keywords = bug["description_keywords"]
    for action in actions:
        text = " ".join(filter(None, [
            action.description or "",
            action.comment or "",
            action.issue_type or "",
        ]))
        if _keywords_hit(text, keywords):
            return True
    return False


def _patch_fixes_syntax(patched_code: str) -> bool:
    """Try to parse the patched code as valid Python."""
    try:
        ast.parse(patched_code)
        return True
    except SyntaxError:
        return False


def _patch_contains_fix(patched_code: str, fix_keywords: List[str]) -> bool:
    return _keywords_hit(patched_code, fix_keywords)


# ─── Task 1 Grader ────────────────────────────────────────────────────────────

def grade_task1(state: EnvironmentState) -> Tuple[float, Dict[str, float]]:
    """
    Score breakdown:
      - 30% : identified comparison operator bug (= vs ==)
      - 30% : identified off-by-one bug
      - 20% : identified missing return
      - 20% : patch parses correctly and contains all three fixes
    """
    from tasks.task1_easy import KNOWN_BUGS

    actions = state.actions_taken
    breakdown: Dict[str, float] = {}

    # Bug identification (60% total, 20 each)
    for bug_name, bug_info in KNOWN_BUGS.items():
        hit = _actions_mention_bug(actions, bug_info)
        weight = 0.30 if bug_info["severity"] == "critical" else 0.20
        breakdown[f"found_{bug_name}"] = weight if hit else 0.0

    # Patch quality (20%)
    patch_score = 0.0
    if state.patch_submitted:
        p = state.patch_submitted
        if _patch_fixes_syntax(p):
            patch_score += 0.10
        if "==" in p and "= 0" not in p.replace("==", ""):
            patch_score += 0.04
        if "range(1, len(numbers))" in p:
            patch_score += 0.03
        if re.search(r"return\s+max_val", p):
            patch_score += 0.03
    breakdown["patch_quality"] = patch_score

    total = sum(breakdown.values())
    return min(total, 1.0), breakdown


# ─── Task 2 Grader ────────────────────────────────────────────────────────────

def grade_task2(state: EnvironmentState) -> Tuple[float, Dict[str, float]]:
    """
    Score breakdown:
      - 20% : identified SQL injection (login)
      - 20% : identified SQL injection (register)
      - 15% : identified plaintext password
      - 10% : identified no rate limiting
      - 10% : identified sensitive data leakage
      - 05% : identified hardcoded secret
      - 20% : patch uses parameterized queries + password hashing
    """
    from tasks.task2_medium import KNOWN_VULNERABILITIES

    actions = state.actions_taken
    breakdown: Dict[str, float] = {}

    weights = {
        "sql_injection_login": 0.20,
        "sql_injection_register": 0.20,
        "plaintext_password": 0.15,
        "no_rate_limiting": 0.10,
        "sensitive_data_leak": 0.10,
        "hardcoded_secret": 0.05,
    }

    for vuln_name, vuln_info in KNOWN_VULNERABILITIES.items():
        hit = _actions_mention_bug(actions, vuln_info)
        breakdown[f"found_{vuln_name}"] = weights[vuln_name] if hit else 0.0

    # Patch quality (20%)
    patch_score = 0.0
    if state.patch_submitted:
        p = state.patch_submitted
        if _patch_fixes_syntax(p):
            patch_score += 0.05
        if "?" in p and "execute" in p:                         # parameterized
            patch_score += 0.07
        if _patch_contains_fix(p, ["generate_password_hash", "bcrypt", "argon2", "pbkdf2"]):
            patch_score += 0.05
        if _patch_contains_fix(p, ["os.environ", "environ.get", "getenv"]):
            patch_score += 0.03
    breakdown["patch_quality"] = patch_score

    total = sum(breakdown.values())
    return min(total, 1.0), breakdown


# ─── Task 3 Grader ────────────────────────────────────────────────────────────

def grade_task3(state: EnvironmentState) -> Tuple[float, Dict[str, float]]:
    """
    Score breakdown:
      - 15% : race condition
      - 15% : memory leak / missing eviction
      - 15% : N+1 query / mget
      - 10% : LRU order correctness
      - 15% : thread safety
      - 15% : pickle deserialization vulnerability
      - 15% : patch quality (structural checks)
    """
    from tasks.task3_hard import KNOWN_ISSUES

    actions = state.actions_taken
    breakdown: Dict[str, float] = {}

    weights = {
        "race_condition": 0.15,
        "memory_leak": 0.15,
        "n_plus_one": 0.15,
        "wrong_lru_order": 0.10,
        "thread_safety": 0.15,
        "pickle_injection": 0.15,
    }

    for issue_name, issue_info in KNOWN_ISSUES.items():
        hit = _actions_mention_bug(actions, issue_info)
        breakdown[f"found_{issue_name}"] = weights[issue_name] if hit else 0.0

    # Patch quality (15%)
    patch_score = 0.0
    if state.patch_submitted:
        p = state.patch_submitted
        if _patch_fixes_syntax(p):
            patch_score += 0.03
        if _patch_contains_fix(p, ["threading.Lock", "Lock()", "_lock"]):
            patch_score += 0.03
        if _patch_contains_fix(p, ["OrderedDict", "move_to_end"]):
            patch_score += 0.03
        if _patch_contains_fix(p, ["mget", "pipeline"]):
            patch_score += 0.03
        if _patch_contains_fix(p, ["json.loads", "json.dumps"]) and "pickle" not in p:
            patch_score += 0.03
    breakdown["patch_quality"] = patch_score

    total = sum(breakdown.values())
    return min(total, 1.0), breakdown


# ─── Dispatcher ──────────────────────────────────────────────────────────────

GRADERS = {
    "task_1_easy_bug_hunt": grade_task1,
    "task_2_medium_security": grade_task2,
    "task_3_hard_perf_correctness": grade_task3,
}


def grade(state: EnvironmentState) -> Tuple[float, Dict[str, float]]:
    grader = GRADERS.get(state.task_id)
    if grader is None:
        raise ValueError(f"No grader found for task_id={state.task_id!r}")
    return grader(state)
