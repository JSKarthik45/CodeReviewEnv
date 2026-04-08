"""
Baseline inference script for CodeReviewEnv.

Evaluates a model (via OpenAI-compatible API) across all three tasks and
reports per-task and aggregate scores.

Usage:
    HF_TOKEN=<your_token> python agents/baseline_agent.py [--model MODEL] [--server URL]

The script uses the Hugging Face Inference API (OpenAI-compatible endpoint)
with the model specified via --model (default: Qwen/Qwen2.5-72B-Instruct).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

import requests
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "Qwen/Qwen2.5-72B-Instruct"
DEFAULT_SERVER = "http://localhost:7860"
HF_BASE_URL = "https://api-inference.huggingface.co/v1"

TASK_IDS = [
    "task_1_easy_bug_hunt",
    "task_2_medium_security",
    "task_3_hard_perf_correctness",
]

# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert software engineer performing a thorough code review.
Your task is to:
1. Carefully read the provided code.
2. Identify ALL bugs, security vulnerabilities, performance issues, and correctness problems.
3. For each issue, output a JSON action with action_type="review".
4. After all issues are identified, output a patch with action_type="patch".
5. Finally, output action_type="submit" with your verdict.

Each action must be valid JSON matching this schema:
{
  "action_type": "review" | "patch" | "comment" | "submit",
  "severity": "critical" | "major" | "minor" | "info",   // for review
  "issue_type": "bug" | "security" | "performance" | "logic" | "style",
  "line_number": <int or null>,
  "description": "<concise description of the issue>",
  "patched_code": "<full corrected code>",  // for patch
  "comment": "<optional comment>",
  "verdict": "approve" | "request_changes" | "reject",  // for submit
  "confidence": <0.0-1.0>
}

Output ONE action JSON per message. Be precise and thorough.
"""


def build_user_prompt(obs: Dict[str, Any]) -> str:
    ctx = obs["review_context"]
    files_text = "\n\n".join(
        f"=== {f['filename']} ({f['language']}) ===\n{f['content']}"
        for f in ctx["files_changed"]
    )
    prev = obs.get("previous_actions", [])
    issues_so_far = obs.get("issues_found_so_far", [])

    prompt = f"""Pull Request: {ctx['pull_request_title']}
Author: {ctx['author']}
Description: {ctx['description']}

Linter: {ctx.get('linter_output', 'N/A')}
Tests: {ctx.get('test_results', 'N/A')}

--- CODE ---
{files_text}
--- END CODE ---

Steps taken so far: {obs['step']} / {obs['max_steps']}
Issues identified so far: {len(issues_so_far)}
"""
    if issues_so_far:
        prompt += "\nIssues already reported:\n"
        for iss in issues_so_far:
            prompt += f"  - [{iss.get('severity','?')}] line {iss.get('line','?')}: {iss.get('description','')}\n"

    if obs["step"] == 0:
        prompt += "\nPlease begin your review. Output your first action as JSON."
    elif obs["step"] >= obs["max_steps"] - 2:
        prompt += "\nYou are running low on steps. Please submit a patch and final verdict now."
    else:
        prompt += "\nContinue your review or submit if done. Output next action as JSON."

    return prompt


# ── Agent loop ────────────────────────────────────────────────────────────────

def extract_json(text: str) -> Dict[str, Any]:
    """Extract first JSON object from model response."""
    # Try direct parse first
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Find JSON block
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON found in response")
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError("Unbalanced JSON")


def run_episode(
    client: OpenAI,
    model: str,
    server: str,
    task_id: str,
) -> Dict[str, Any]:
    """Run a single episode and return the result dict."""

    # 1. Reset
    resp = requests.post(f"{server}/reset", json={"task_id": task_id}, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    session_id = data["session_id"]
    obs = data["observation"]

    print(f"\n{'='*60}")
    print(f"Task: {task_id}")
    print(f"Session: {session_id}")
    print(f"{'='*60}")

    history: List[Dict[str, str]] = []
    final_score = 0.0
    done = False
    patch_submitted = False

    while not done:
        user_msg = build_user_prompt(obs)
        history.append({"role": "user", "content": user_msg})

        # Call model
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history,
                max_tokens=1024,
                temperature=0.2,
            )
            raw = completion.choices[0].message.content or ""
        except Exception as exc:
            print(f"  [Model error] {exc}")
            break

        history.append({"role": "assistant", "content": raw})

        # Parse action
        try:
            action_dict = extract_json(raw)
        except ValueError as exc:
            print(f"  [Parse error] {exc} | raw={raw[:200]!r}")
            # Force a submit to avoid infinite spin
            action_dict = {"action_type": "submit", "verdict": "request_changes", "confidence": 0.3}

        action_type = action_dict.get("action_type", "review")
        print(f"  Step {obs['step']+1}: {action_type} | {action_dict.get('description','')[:80]}")

        # Auto-submit near step limit
        if obs["step"] >= obs["max_steps"] - 1 and action_type != "submit":
            action_dict = {"action_type": "submit", "verdict": "request_changes", "confidence": 0.5}
            if not patch_submitted:
                # Submit a patch first
                action_dict = {
                    "action_type": "patch",
                    "patched_code": obs["review_context"]["files_changed"][0]["content"],
                }

        if action_type == "patch":
            patch_submitted = True

        # Step
        step_resp = requests.post(
            f"{server}/step",
            json={"session_id": session_id, "action": action_dict},
            timeout=30,
        )
        step_resp.raise_for_status()
        step_data = step_resp.json()
        obs = step_data["observation"]
        done = step_data["done"]
        info = step_data.get("info", {})

        if done:
            final_score = info.get("final_score", 0.0)
            breakdown = info.get("breakdown", {})
            print(f"\n  Final score: {final_score:.4f}")
            print(f"  Breakdown:  {json.dumps(breakdown, indent=4)}")

        time.sleep(0.3)  # be polite to the API

    # Cleanup
    requests.delete(f"{server}/session/{session_id}", timeout=10)

    return {
        "task_id": task_id,
        "final_score": final_score,
        "steps_taken": obs["step"],
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CodeReviewEnv baseline agent")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--server", default=DEFAULT_SERVER)
    parser.add_argument("--task", default=None, help="Run a single task (default: all)")
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("ERROR: HF_TOKEN environment variable not set.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(
        api_key=hf_token,
        base_url=HF_BASE_URL,
    )

    tasks = [args.task] if args.task else TASK_IDS
    results = []

    for task_id in tasks:
        result = run_episode(client, args.model, args.server, task_id)
        results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("BASELINE SUMMARY")
    print("=" * 60)
    for r in results:
        print(f"  {r['task_id']:<40} score={r['final_score']:.4f}  steps={r['steps_taken']}")

    if len(results) == len(TASK_IDS):
        avg = sum(r["final_score"] for r in results) / len(results)
        print(f"\n  Aggregate average score: {avg:.4f}")

    # Save results
    out_path = "baseline_results.json"
    with open(out_path, "w") as f:
        json.dump({"model": args.model, "results": results}, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
