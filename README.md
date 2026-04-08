---
title: CodeReviewEnv
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# Code Review Environment
My custom environment for code review tasks.

# 🔍 CodeReviewEnv

> An OpenEnv-compliant benchmark environment where AI agents act as senior engineers reviewing pull requests — catching bugs, finding security holes, and fixing broken code.

---

## Overview & Motivation

Code review is one of the highest-leverage activities in software engineering, yet it is time-consuming, inconsistent, and cognitively demanding. A model that can reliably triage pull requests, identify security vulnerabilities, and produce corrected patches would meaningfully accelerate software delivery.

**CodeReviewEnv** simulates exactly this. Three tasks of increasing difficulty present agents with realistic pull requests containing planted defects. The agent must reason over code, report issues with structured annotations, submit a corrected patch, and deliver a final verdict — all within a bounded step budget.

---

## Environment Architecture

```
code-review-env/
├── env.py               # Core OpenEnv environment (reset / step / state)
├── server.py            # FastAPI HTTP server exposing the OpenEnv interface
├── models.py            # Pydantic typed models: Action, Observation, Reward, State
├── openenv.yaml         # OpenEnv metadata
├── tasks/
│   ├── task1_easy.py    # Bug hunt: simple Python utility
│   ├── task2_medium.py  # Security audit: Flask auth endpoint
│   └── task3_hard.py    # Correctness: distributed LRU cache
├── graders/
│   └── grader.py        # Deterministic keyword + AST graders
├── agents/
│   └── baseline_agent.py  # HF Inference API baseline (OpenAI-compatible)
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Action Space

Each agent turn is a single `ReviewAction` JSON object:

| Field | Type | Description |
|---|---|---|
| `action_type` | `"review" \| "patch" \| "comment" \| "submit"` | What the agent is doing |
| `severity` | `"critical" \| "major" \| "minor" \| "info"` | Issue severity (for `review`) |
| `issue_type` | `"bug" \| "security" \| "performance" \| "logic" \| "style"` | Issue category |
| `line_number` | `int \| null` | Line the issue is on |
| `description` | `str` | Concise natural-language description of the issue |
| `patched_code` | `str \| null` | Full corrected code (for `patch` actions) |
| `comment` | `str \| null` | Free-form annotation |
| `verdict` | `"approve" \| "request_changes" \| "reject"` | Final verdict (for `submit`) |
| `confidence` | `float [0.0, 1.0]` | Agent's self-reported confidence |

---

## Observation Space

Each step returns an `Observation` containing:

| Field | Description |
|---|---|
| `task_id` | Identifier of the current task |
| `step` / `max_steps` | Current step and budget |
| `review_context` | Full PR: title, author, description, code files, linter output, test results |
| `previous_actions` | All actions taken so far this episode |
| `issues_found_so_far` | Structured list of issues reported |
| `score_so_far` | Running cumulative intermediate reward |
| `done` | Whether the episode has ended |

---

## Reward Function

Reward is **dense** — provided at every step, not only at the end.

### Intermediate (per-step)

| Signal | Value | Rationale |
|---|---|---|
| Step penalty | −0.01 | Encourages efficiency |
| Review with description | +0.05 | Rewards substantive annotations |
| Critical severity bonus | +0.03 | Rewards correct triage |
| Patch submitted | +0.10 | Rewards producing a fix |
| Repetition penalty | −0.05 | Penalises looping / copy-paste |

### Terminal (on `submit` or step exhaustion)

The programmatic grader runs and returns a score in **[0.0, 1.0]** based on which issues were correctly identified and how well the submitted patch addresses them. This final score overwrites the episode total.

---

## Tasks

### Task 1 — Easy: Bug Hunt (`task_1_easy_bug_hunt`)

**Max steps:** 8  
**File reviewed:** `utils.py` (Python, 30 lines)

A developer submits three utility functions. Three bugs are planted:

| # | Line | Bug | Severity |
|---|---|---|---|
| 1 | 3 | `=` (assignment) used instead of `==` (comparison) — causes `SyntaxError` | Critical |
| 2 | 6 | `range(1, len(numbers) + 1)` — off-by-one causes `IndexError` | Critical |
| 3 | 9 | Missing `return max_val` — function silently returns `None` | Major |

**Grading:** 30% per critical bug identified, 20% for minor, 20% for a syntactically valid patch with all three fixes applied.

---

### Task 2 — Medium: Security Audit (`task_2_medium_security`)

**Max steps:** 12  
**File reviewed:** `auth.py` (Flask, 55 lines)

A backend developer submits login and registration endpoints. Six security vulnerabilities are present:

| # | Line | Vulnerability | Severity |
|---|---|---|---|
| 1 | 23 | SQL injection in `login` query (f-string interpolation) | Critical |
| 2 | 44 | SQL injection in `register` INSERT | Critical |
| 3 | 39 | Plaintext password storage (no hashing) | Critical |
| 4 | — | No rate limiting on `/login` (brute-force possible) | Major |
| 5 | 30 | Sensitive data leakage: error distinguishes "wrong password" vs "user not found" | Major |
| 6 | 5 | Hardcoded `secret_key` in source | Major |

**Grading:** Weighted by severity. Patch checked for parameterized queries, password hashing, and environment variable use.

---

### Task 3 — Hard: Distributed Systems Correctness (`task_3_hard_perf_correctness`)

**Max steps:** 16  
**File reviewed:** `cache.py` (Python, 55 lines)

A senior engineer submits a Redis-backed LRU cache claimed to be production-ready. Six issues lurk:

| # | Issue | Type | Severity |
|---|---|---|---|
| 1 | Non-atomic `EXISTS` + `GET` creates a race condition | Concurrency | Critical |
| 2 | Local `dict` grows unboundedly — `capacity` parameter ignored | Performance | Critical |
| 3 | `get_many` calls `self.get()` in a loop (N+1 round trips) | Performance | Major |
| 4 | `dict` preserves insertion order, not access order — LRU eviction is wrong | Logic | Major |
| 5 | Shared `dict` modified without a `threading.Lock` | Concurrency | Critical |
| 6 | `pickle.loads` on bytes from Redis — arbitrary code execution | Security | Critical |

**Grading:** Equally weighted. Patch checked structurally for `threading.Lock`, `OrderedDict.move_to_end`, `mget`, and `json` instead of `pickle`.

---

## Baseline Performance

Evaluated with `Qwen/Qwen2.5-72B-Instruct` via Hugging Face Inference API:

| Task | Score |
|---|---|
| Task 1 — Easy | 0.72 |
| Task 2 — Medium | 0.55 |
| Task 3 — Hard | 0.38 |
| **Aggregate** | **0.55** |

---

## Setup & Usage

### 1. Local (Python)

```bash
git clone <repo>
cd code-review-env
pip install -r requirements.txt
python server.py
# Server running at http://localhost:7860
```

### 2. Docker

```bash
docker build -t code-review-env .
docker run -p 7860:7860 code-review-env
```

### 3. API Quickstart

```bash
# Reset to task 1
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task_1_easy_bug_hunt"}'

# Take a step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "<session_id>",
    "action": {
      "action_type": "review",
      "severity": "critical",
      "issue_type": "bug",
      "line_number": 3,
      "description": "Assignment operator = used instead of comparison == on line 3"
    }
  }'
```

### 4. Run inference script

```bash
export HF_TOKEN=hf_your_token_here
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
python inference.py
```

Expected stdout format:
```
[START] task=task_1_easy_bug_hunt env=code-review-env model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=review:assignment operator = instead of == reward=0.07 done=false error=null
[STEP] step=2 action=review:off-by-one in range reward=0.07 done=false error=null
[STEP] step=3 action=patch:fixed code reward=0.10 done=false error=null
[STEP] step=4 action=submit:request_changes reward=1.00 done=true error=null
[END] success=true steps=4 score=1.000 rewards=0.07,0.07,0.10,1.00
```

### 5. OpenEnv validation

```bash
openenv validate .
```

---

## HTTP API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Environment info |
| `GET` | `/tasks` | List all tasks |
| `POST` | `/reset` | Start a new episode |
| `POST` | `/step` | Take an action |
| `GET` | `/state/{session_id}` | Inspect full environment state |
| `DELETE` | `/session/{session_id}` | Clean up session |

---

## Hugging Face Spaces Deployment

The `Dockerfile` targets port `7860` and runs as a non-root user — compatible with HF Spaces Docker SDK out of the box. Tag the Space with `openenv`.

```yaml
# README header for HF Spaces
---
title: CodeReviewEnv
emoji: 🔍
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: false
tags:
  - openenv
---
```
