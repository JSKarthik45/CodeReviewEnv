"""
inference.py — CodeReviewEnv baseline inference script.

Mandatory env vars:
    API_BASE_URL    The API endpoint for the LLM.
    MODEL_NAME      The model identifier to use for inference.
    HF_TOKEN        Your Hugging Face / API key.

STDOUT format (strictly followed):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import json
import os
import sys
import textwrap
from typing import Any, Dict, List, Optional

from openai import OpenAI

sys.path.insert(0, os.path.dirname(__file__))
from env import CodeReviewEnv, TASK_IDS
from models import ReviewAction

# ── Env vars ──────────────────────────────────────────────────────────────────
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK = "code-review-env"
SUCCESS_SCORE_THRESHOLD = 0.5

# ── Logging helpers ───────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    action_clean = action.replace("\n", " ").replace("\r", "")[:120]
    print(
        f"[STEP] step={step} action={action_clean} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert software engineer performing a thorough code review.
    Your job is to:
    1. Identify ALL bugs, security vulnerabilities, performance issues, and logic errors.
    2. For each issue, output a JSON action with action_type="review".
    3. After identifying all issues, output a patch with action_type="patch".
    4. Finally, output action_type="submit" with your verdict.

    Each response must be a single valid JSON object. No markdown, no explanation outside JSON.

    Schema:
    {
      "action_type": "review" | "patch" | "comment" | "submit",
      "severity": "critical" | "major" | "minor" | "info",
      "issue_type": "bug" | "security" | "performance" | "logic" | "style",
      "line_number": <int or null>,
      "description": "<description of the issue>",
      "patched_code": "<full corrected code>",
      "comment": "<optional>",
      "verdict": "approve" | "request_changes" | "reject",
      "confidence": <0.0-1.0>
    }

    Output ONE JSON object per response. Be precise and thorough.
""").strip()


def build_user_prompt(obs: Dict[str, Any]) -> str:
    ctx = obs["review_context"]
    files_text = "\n\n".join(
        f"=== {f['filename']} ({f['language']}) ===\n{f['content']}"
        for f in ctx["files_changed"]
    )
    issues_so_far = obs.get("issues_found_so_far", [])

    prompt = textwrap.dedent(f"""
        Pull Request: {ctx['pull_request_title']}
        Author: {ctx['author']}
        Description: {ctx['description']}
        Linter: {ctx.get('linter_output', 'N/A')}
        Tests: {ctx.get('test_results', 'N/A')}

        --- CODE ---
        {files_text}
        --- END CODE ---

        Step: {obs['step']} / {obs['max_steps']}
        Issues reported so far: {len(issues_so_far)}
    """).strip()

    if issues_so_far:
        prompt += "\n\nIssues already reported (do NOT repeat these):"
        for iss in issues_so_far:
            prompt += f"\n  - [{iss.get('severity','?')}] line {iss.get('line','?')}: {iss.get('description','')}"

    steps_left = obs['max_steps'] - obs['step']
    if steps_left <= 2:
        prompt += "\n\nYou are almost out of steps. Submit your patch and verdict NOW."
    elif obs['step'] == 0:
        prompt += "\n\nBegin your review. Output your first action as JSON."
    else:
        prompt += "\n\nContinue reviewing or submit if done. Output next action as JSON."

    return prompt


# ── JSON extraction ───────────────────────────────────────────────────────────

def extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in response")
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError("Unbalanced JSON in response")


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(client: OpenAI, task_id: str) -> Dict[str, Any]:
    env = CodeReviewEnv()
    obs_obj = env.reset(task_id)
    obs = obs_obj.model_dump()

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    history: List[Dict[str, str]] = []
    patch_submitted = False
    error_msg: Optional[str] = None

    try:
        for step in range(1, obs_obj.max_steps + 1):
            if obs.get("done"):
                break

            error_msg = None
            steps_left = obs["max_steps"] - obs["step"]

            # Force patch then submit near step limit
            if steps_left <= 1 and not patch_submitted:
                action_dict = {
                    "action_type": "patch",
                    "patched_code": obs["review_context"]["files_changed"][0]["content"],
                }
            elif steps_left <= 0:
                action_dict = {
                    "action_type": "submit",
                    "verdict": "request_changes",
                    "confidence": 0.5,
                }
            else:
                user_msg = build_user_prompt(obs)
                history.append({"role": "user", "content": user_msg})

                try:
                    completion = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history,
                        max_tokens=1024,
                        temperature=0.2,
                        stream=False,
                    )
                    raw = (completion.choices[0].message.content or "").strip()
                    history.append({"role": "assistant", "content": raw})
                    action_dict = extract_json(raw)
                except Exception as exc:
                    error_msg = str(exc)[:80]
                    action_dict = {
                        "action_type": "submit",
                        "verdict": "request_changes",
                        "confidence": 0.3,
                    }

            if action_dict.get("action_type") == "patch":
                patch_submitted = True

            # Validate action
            try:
                action = ReviewAction(**action_dict)
            except Exception as exc:
                error_msg = str(exc)[:80]
                action = ReviewAction(
                    action_type="submit",
                    verdict="request_changes",
                    confidence=0.3,
                )

            # Step environment
            obs_obj, reward_obj, done, info = env.step(action)
            obs = obs_obj.model_dump()

            reward = reward_obj.value
            rewards.append(reward)
            steps_taken = step

            action_summary = f"{action.action_type}:{(action.description or action.verdict or '')[:60]}"
            log_step(step=step, action=action_summary, reward=reward, done=done, error=error_msg)

            if done:
                score = info.get("final_score", 0.0)
                break

        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {"task_id": task_id, "score": score, "steps": steps_taken, "success": success}


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if not API_KEY:
        print("[ERROR] HF_TOKEN environment variable not set.", flush=True)
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    task_ids = os.getenv("TASK_IDS", ",".join(TASK_IDS)).split(",")
    task_ids = [t.strip() for t in task_ids if t.strip()]

    all_results = []
    for task_id in task_ids:
        result = run_episode(client, task_id)
        all_results.append(result)

    # Aggregate summary to stderr so it doesn't pollute stdout log format
    print("\n[SUMMARY]", file=sys.stderr)
    for r in all_results:
        print(f"  {r['task_id']}: score={r['score']:.3f} steps={r['steps']} success={r['success']}", file=sys.stderr)
    if all_results:
        avg = sum(r["score"] for r in all_results) / len(all_results)
        print(f"  aggregate: {avg:.3f}", file=sys.stderr)


if __name__ == "__main__":
    main()
