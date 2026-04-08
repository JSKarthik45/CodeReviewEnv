"""
CodeReviewEnv — OpenEnv-compliant environment.

Implements:
  reset()  → Observation
  step(action) → (Observation, StepReward, done, info)
  state()  → EnvironmentState
"""
from __future__ import annotations

import copy
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from typing import Any, Dict, Tuple

from models import (
    CodeFile,
    EnvironmentState,
    Observation,
    ReviewAction,
    ReviewContext,
    StepReward,
)
from graders.grader import grade


# ── Task registry ────────────────────────────────────────────────────────────
def _load_task(task_id: str) -> Dict[str, Any]:
    if task_id == "task_1_easy_bug_hunt":
        from tasks.task1_easy import get_task_config
    elif task_id == "task_2_medium_security":
        from tasks.task2_medium import get_task_config
    elif task_id == "task_3_hard_perf_correctness":
        from tasks.task3_hard import get_task_config
    else:
        raise ValueError(f"Unknown task_id: {task_id!r}")
    return get_task_config()


TASK_IDS = [
    "task_1_easy_bug_hunt",
    "task_2_medium_security",
    "task_3_hard_perf_correctness",
]

# ─────────────────────────────────────────────────────────────────────────────


class CodeReviewEnv:
    """OpenEnv-compliant code review environment."""

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def reset(self, task_id: str = "task_1_easy_bug_hunt") -> Observation:
        """Reset the environment for a given task. Returns the initial observation."""
        cfg = _load_task(task_id)
        pr = cfg["pull_request"]

        files = [CodeFile(**f) for f in pr["files_changed"]]
        review_ctx = ReviewContext(
            pull_request_title=pr["pull_request_title"],
            author=pr["author"],
            description=pr["description"],
            files_changed=files,
            test_results=pr.get("test_results"),
            linter_output=pr.get("linter_output"),
        )

        self._state = EnvironmentState(
            task_id=task_id,
            step=0,
            max_steps=cfg["max_steps"],
            review_context=review_ctx,
        )
        self._cfg = cfg
        return self._make_observation()

    def step(self, action: ReviewAction) -> Tuple[Observation, StepReward, bool, Dict[str, Any]]:
        """
        Apply an action. Returns (observation, reward, done, info).
        Raises RuntimeError if called before reset().
        """
        if not hasattr(self, "_state"):
            raise RuntimeError("Call reset() before step().")

        s = self._state

        # ── Terminal check ───────────────────────────────────────────────────
        if s.done:
            obs = self._make_observation(feedback="Episode already finished.")
            return obs, StepReward(value=0.0, explanation="Episode done."), True, {}

        s.step += 1

        # ── Absorb action ────────────────────────────────────────────────────
        s.actions_taken.append(action)

        # Record issue if it is a review action
        if action.action_type == "review" and action.description:
            issue = {
                "step": s.step,
                "severity": action.severity,
                "issue_type": action.issue_type,
                "line": action.line_number,
                "description": action.description,
            }
            s.issues_identified.append(issue)

        # Record patch
        if action.action_type == "patch" and action.patched_code:
            s.patch_submitted = action.patched_code

        # Record verdict
        if action.action_type == "submit" and action.verdict:
            s.verdict_submitted = action.verdict

        # ── Reward ───────────────────────────────────────────────────────────
        reward = self._compute_step_reward(action)
        s.total_reward += reward.value

        # ── Done condition ───────────────────────────────────────────────────
        submitted = action.action_type == "submit"
        out_of_steps = s.step >= s.max_steps

        if submitted or out_of_steps:
            final_score, breakdown = grade(s)
            s.total_reward = final_score
            s.done = True
            s.terminated_reason = "submitted" if submitted else "max_steps_reached"
            reward = StepReward(
                value=final_score,
                breakdown=breakdown,
                explanation=f"Final score: {final_score:.3f}",
            )
            info = {"final_score": final_score, "breakdown": breakdown, "reason": s.terminated_reason}
        else:
            info = {"step": s.step, "cumulative_reward": s.total_reward}

        obs = self._make_observation()
        return obs, reward, s.done, info

    def state(self) -> EnvironmentState:
        if not hasattr(self, "_state"):
            raise RuntimeError("Call reset() before state().")
        return copy.deepcopy(self._state)

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _make_observation(self, feedback: str | None = None) -> Observation:
        s = self._state
        return Observation(
            task_id=s.task_id,
            step=s.step,
            max_steps=s.max_steps,
            review_context=s.review_context,
            previous_actions=list(s.actions_taken),
            feedback=feedback,
            issues_found_so_far=list(s.issues_identified),
            score_so_far=s.total_reward,
            done=s.done,
        )

    def _compute_step_reward(self, action: ReviewAction) -> StepReward:
        """
        Dense intermediate reward:
          +0.05  for a review action with a non-empty description
          +0.03  for a review action with severity='critical'
          +0.10  for a patch action with non-empty code
          -0.05  for repeated identical descriptions (loop detection)
          -0.10  step penalty (encourages efficiency)
        """
        s = self._state
        r = 0.0
        parts: Dict[str, float] = {}

        STEP_PENALTY = -0.01
        r += STEP_PENALTY
        parts["step_penalty"] = STEP_PENALTY

        if action.action_type == "review":
            if action.description:
                parts["review_description"] = 0.05
                r += 0.05
            if action.severity == "critical":
                parts["critical_severity_bonus"] = 0.03
                r += 0.03
            # Loop detection: penalise if same description appeared before
            prev_descs = [
                a.description for a in s.actions_taken[:-1]
                if a.description
            ]
            if action.description and action.description in prev_descs:
                parts["repetition_penalty"] = -0.05
                r += -0.05

        elif action.action_type == "patch":
            if action.patched_code and len(action.patched_code) > 50:
                parts["patch_submitted"] = 0.10
                r += 0.10

        elif action.action_type == "submit":
            pass  # final score handled in step()

        return StepReward(
            value=max(-1.0, min(1.0, r)),
            breakdown=parts,
            explanation=f"Step {s.step} intermediate reward",
        )
