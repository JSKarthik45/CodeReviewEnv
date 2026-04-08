"""
FastAPI server exposing the CodeReviewEnv as an HTTP API.
Endpoints mirror the OpenEnv interface: /reset, /step, /state, /tasks.
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from typing import Any, Dict
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env import CodeReviewEnv, TASK_IDS
from models import ReviewAction, Observation, StepReward, EnvironmentState

app = FastAPI(
    title="CodeReviewEnv",
    description="OpenEnv-compliant environment for AI code review agents.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Session store (in-memory, single process) ────────────────────────────────
_sessions: Dict[str, CodeReviewEnv] = {}


# ── Request / Response models ─────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = "task_1_easy_bug_hunt"
    session_id: str | None = None


class ResetResponse(BaseModel):
    session_id: str
    observation: Observation


class StepRequest(BaseModel):
    session_id: str
    action: ReviewAction


class StepResponse(BaseModel):
    observation: Observation
    reward: StepReward
    done: bool
    info: Dict[str, Any]


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "CodeReviewEnv",
        "version": "1.0.0",
        "tasks": TASK_IDS,
        "spec": "OpenEnv v1",
    }


@app.get("/tasks")
def list_tasks():
    from tasks.task1_easy import get_task_config as t1
    from tasks.task2_medium import get_task_config as t2
    from tasks.task3_hard import get_task_config as t3
    tasks = []
    for fn in (t1, t2, t3):
        cfg = fn()
        tasks.append({
            "task_id": cfg["task_id"],
            "difficulty": cfg["difficulty"],
            "description": cfg["description"],
        })
    return {"tasks": tasks}


@app.post("/reset", response_model=ResetResponse)
def reset(req: ResetRequest):
    if req.task_id not in TASK_IDS:
        raise HTTPException(400, f"Unknown task_id {req.task_id!r}. Choose from {TASK_IDS}")
    session_id = req.session_id or str(uuid.uuid4())
    env = CodeReviewEnv()
    obs = env.reset(req.task_id)
    _sessions[session_id] = env
    return ResetResponse(session_id=session_id, observation=obs)


@app.post("/step", response_model=StepResponse)
def step(req: StepRequest):
    env = _sessions.get(req.session_id)
    if env is None:
        raise HTTPException(404, f"Session {req.session_id!r} not found. Call /reset first.")
    obs, reward, done, info = env.step(req.action)
    return StepResponse(observation=obs, reward=reward, done=done, info=info)


@app.get("/state/{session_id}", response_model=EnvironmentState)
def get_state(session_id: str):
    env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(404, f"Session {session_id!r} not found.")
    return env.state()


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    _sessions.pop(session_id, None)
    return {"deleted": session_id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=7860, reload=False)
