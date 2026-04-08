"""
Task 1 (Easy): Bug Identification in a Simple Python Utility.

The agent reviews a short Python module with 3 clearly planted bugs:
  1. Off-by-one error in a loop
  2. Incorrect comparison operator (= vs ==)
  3. Missing return statement in a branch
"""
from __future__ import annotations
from typing import Any, Dict

TASK_ID = "task_1_easy_bug_hunt"
MAX_STEPS = 8

BUGGY_CODE = '''\
def find_max(numbers: list) -> int:
    """Return the maximum value in a non-empty list."""
    if len(numbers) = 0:          # BUG 1: assignment instead of == comparison
        raise ValueError("List is empty")
    max_val = numbers[0]
    for i in range(1, len(numbers) + 1):  # BUG 2: off-by-one, should be len(numbers)
        if numbers[i] > max_val:
            max_val = numbers[i]
    # BUG 3: missing return statement — falls off the end returning None


def calculate_average(numbers: list) -> float:
    """Return the arithmetic mean of a list of numbers."""
    if not numbers:
        raise ValueError("Cannot average empty list")
    total = 0
    for n in numbers:
        total += n
    return total / len(numbers)


def is_palindrome(s: str) -> bool:
    """Check whether a string is a palindrome (case-insensitive)."""
    cleaned = s.lower().replace(" ", "")
    return cleaned == cleaned[::-1]
'''

FIXED_CODE = '''\
def find_max(numbers: list) -> int:
    """Return the maximum value in a non-empty list."""
    if len(numbers) == 0:
        raise ValueError("List is empty")
    max_val = numbers[0]
    for i in range(1, len(numbers)):
        if numbers[i] > max_val:
            max_val = numbers[i]
    return max_val


def calculate_average(numbers: list) -> float:
    """Return the arithmetic mean of a list of numbers."""
    if not numbers:
        raise ValueError("Cannot average empty list")
    total = 0
    for n in numbers:
        total += n
    return total / len(numbers)


def is_palindrome(s: str) -> bool:
    """Check whether a string is a palindrome (case-insensitive)."""
    cleaned = s.lower().replace(" ", "")
    return cleaned == cleaned[::-1]
'''

KNOWN_BUGS = {
    "bug_comparison_operator": {
        "line": 3,
        "description_keywords": ["assignment", "comparison", "==", "=", "operator"],
        "severity": "critical",
        "issue_type": "bug",
    },
    "bug_off_by_one": {
        "line": 6,
        "description_keywords": ["off-by-one", "index", "range", "len", "+1", "IndexError"],
        "severity": "critical",
        "issue_type": "bug",
    },
    "bug_missing_return": {
        "line": 9,
        "description_keywords": ["return", "None", "missing", "falls off"],
        "severity": "major",
        "issue_type": "bug",
    },
}

PULL_REQUEST = {
    "pull_request_title": "Add utility functions: find_max, calculate_average, is_palindrome",
    "author": "dev-intern",
    "description": (
        "Implements three utility functions for list and string operations. "
        "Please review for correctness before merging."
    ),
    "files_changed": [
        {
            "filename": "utils.py",
            "language": "python",
            "content": BUGGY_CODE,
            "line_count": BUGGY_CODE.count("\n") + 1,
        }
    ],
    "test_results": "No tests provided.",
    "linter_output": "SyntaxError detected on line 3 (invalid syntax).",
}


def get_task_config() -> Dict[str, Any]:
    return {
        "task_id": TASK_ID,
        "max_steps": MAX_STEPS,
        "pull_request": PULL_REQUEST,
        "known_bugs": KNOWN_BUGS,
        "fixed_code": FIXED_CODE,
        "difficulty": "easy",
        "description": (
            "Review a short Python utility module. "
            "Find and describe all bugs, then submit a patched version."
        ),
    }
