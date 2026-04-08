"""
OpenEnv-compliant Pydantic models for the Code Review Environment.
"""
from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


# ─── Action Space ────────────────────────────────────────────────────────────

class ReviewAction(BaseModel):
    """Agent action: review and optionally patch code."""
    action_type: Literal["review", "patch", "comment", "submit"] = Field(
        description="Type of action the agent takes."
    )
    # For 'review': provide a structured analysis
    severity: Optional[Literal["critical", "major", "minor", "info"]] = None
    issue_type: Optional[str] = Field(
        default=None,
        description="Category: bug, security, performance, style, logic"
    )
    line_number: Optional[int] = Field(default=None, ge=1)
    description: Optional[str] = Field(default=None, max_length=500)

    # For 'patch': provide fixed code
    patched_code: Optional[str] = Field(
        default=None,
        description="Full corrected code (for patch actions)."
    )

    # For 'comment': free-form annotation
    comment: Optional[str] = Field(default=None, max_length=1000)

    # For 'submit': final verdict
    verdict: Optional[Literal["approve", "request_changes", "reject"]] = None
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)


# ─── Observation Space ───────────────────────────────────────────────────────

class CodeFile(BaseModel):
    filename: str
    language: str
    content: str
    line_count: int


class ReviewContext(BaseModel):
    pull_request_title: str
    author: str
    description: str
    files_changed: List[CodeFile]
    test_results: Optional[str] = None
    linter_output: Optional[str] = None


class Observation(BaseModel):
    """What the agent sees at each step."""
    task_id: str
    step: int
    max_steps: int
    review_context: ReviewContext
    previous_actions: List[ReviewAction] = Field(default_factory=list)
    feedback: Optional[str] = None
    issues_found_so_far: List[Dict[str, Any]] = Field(default_factory=list)
    score_so_far: float = 0.0
    done: bool = False


# ─── Reward Model ────────────────────────────────────────────────────────────

class StepReward(BaseModel):
    """Reward signal returned at each step."""
    value: float = Field(ge=-1.0, le=1.0)
    breakdown: Dict[str, float] = Field(default_factory=dict)
    explanation: str = ""


# ─── State ───────────────────────────────────────────────────────────────────

class EnvironmentState(BaseModel):
    task_id: str
    step: int
    max_steps: int
    review_context: ReviewContext
    actions_taken: List[ReviewAction] = Field(default_factory=list)
    issues_identified: List[Dict[str, Any]] = Field(default_factory=list)
    patch_submitted: Optional[str] = None
    verdict_submitted: Optional[str] = None
    total_reward: float = 0.0
    done: bool = False
    terminated_reason: Optional[str] = None
