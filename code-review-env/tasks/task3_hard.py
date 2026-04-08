"""
Task 3 (Hard): Performance & Correctness Review of a Distributed LRU Cache.

The agent reviews a Python LRU cache with Redis backing containing:
  1. Race condition (non-atomic check-then-act on Redis)
  2. Memory leak (unbounded local dict grows forever)
  3. N+1 query pattern (per-key pipeline not batched)
  4. Incorrect LRU eviction (uses insertion order, not access order)
  5. Thread-safety violation (shared dict without lock)
  6. Silent data corruption (pickle loads untrusted bytes)
"""
from __future__ import annotations
from typing import Any, Dict

TASK_ID = "task_3_hard_perf_correctness"
MAX_STEPS = 16

BUGGY_CODE = '''\
import pickle
import threading
import redis

class DistributedLRUCache:
    """
    LRU cache backed by Redis for distributed deployments.
    Local dict acts as an L1 write-through layer.
    """

    def __init__(self, capacity: int, redis_url: str = "redis://localhost:6379"):
        self.capacity = capacity
        self.local = {}          # ISSUE 2 & 5: shared dict, no lock, unbounded growth
        self.redis = redis.from_url(redis_url)
        self.hits = 0
        self.misses = 0

    # ── ISSUE 5: no lock; concurrent writes race on self.local ──────────────
    def get(self, key: str):
        if key in self.local:
            self.hits += 1
            return self.local[key]            # ISSUE 4: doesn't update LRU order

        # ISSUE 1: race condition — between EXISTS and GET another process may delete key
        if self.redis.exists(key):
            raw = self.redis.get(key)
            value = pickle.loads(raw)         # ISSUE 6: deserialising untrusted bytes
            self.local[key] = value           # ISSUE 2: local dict grows without bound
            self.hits += 1
            return value

        self.misses += 1
        return None

    def put(self, key: str, value, ttl: int = 300):
        # ISSUE 2: no eviction from self.local; grows forever
        self.local[key] = value

        # ISSUE 1: non-atomic: set + expire are two separate commands
        self.redis.set(key, pickle.dumps(value))
        self.redis.expire(key, ttl)

    def get_many(self, keys: list):
        # ISSUE 3: N+1 — calls self.get() in a loop instead of using pipeline/mget
        return {k: self.get(k) for k in keys}

    def invalidate(self, key: str):
        self.local.pop(key, None)
        self.redis.delete(key)

    def stats(self):
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total else 0,
            "local_size": len(self.local),
        }
'''

FIXED_CODE = '''\
import json
import threading
from collections import OrderedDict
import redis

class DistributedLRUCache:
    """
    Thread-safe LRU cache backed by Redis.
    Uses OrderedDict for correct LRU eviction, a Lock for thread safety,
    atomic Redis SET EX commands, and mget for batch fetching.
    Serialises with JSON (not pickle) to avoid arbitrary code execution.
    """

    def __init__(self, capacity: int, redis_url: str = "redis://localhost:6379"):
        self.capacity = capacity
        self.local: OrderedDict = OrderedDict()   # correct LRU order
        self._lock = threading.Lock()             # thread safety
        self.redis = redis.from_url(redis_url)
        self.hits = 0
        self.misses = 0

    def get(self, key: str):
        with self._lock:
            if key in self.local:
                self.local.move_to_end(key)       # update LRU order
                self.hits += 1
                return self.local[key]

        raw = self.redis.get(key)                 # atomic single GET, no race
        if raw is not None:
            value = json.loads(raw)               # safe deserialisation
            with self._lock:
                self._evict_if_needed()
                self.local[key] = value
                self.hits += 1
            return value

        with self._lock:
            self.misses += 1
        return None

    def _evict_if_needed(self):
        """Call with self._lock held."""
        while len(self.local) >= self.capacity:
            self.local.popitem(last=False)        # evict LRU item

    def put(self, key: str, value, ttl: int = 300):
        payload = json.dumps(value)
        self.redis.set(key, payload, ex=ttl)      # atomic SET with TTL
        with self._lock:
            self.local[key] = value
            self.local.move_to_end(key)
            self._evict_if_needed()

    def get_many(self, keys: list):
        """Batch fetch using Redis MGET — O(1) round trips."""
        if not keys:
            return {}
        raws = self.redis.mget(keys)
        result = {}
        with self._lock:
            for key, raw in zip(keys, raws):
                if raw is not None:
                    value = json.loads(raw)
                    self._evict_if_needed()
                    self.local[key] = value
                    self.hits += 1
                    result[key] = value
                else:
                    self.misses += 1
                    result[key] = None
        return result

    def invalidate(self, key: str):
        with self._lock:
            self.local.pop(key, None)
        self.redis.delete(key)

    def stats(self):
        with self._lock:
            total = self.hits + self.misses
            return {
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": self.hits / total if total else 0,
                "local_size": len(self.local),
            }
'''

KNOWN_ISSUES = {
    "race_condition": {
        "lines": [23, 43],
        "description_keywords": ["race condition", "atomic", "exists", "set", "pipeline", "non-atomic"],
        "severity": "critical",
        "issue_type": "concurrency",
    },
    "memory_leak": {
        "lines": [13, 27, 38],
        "description_keywords": ["memory leak", "unbounded", "evict", "capacity", "grow"],
        "severity": "critical",
        "issue_type": "performance",
    },
    "n_plus_one": {
        "lines": [47],
        "description_keywords": ["n+1", "pipeline", "mget", "batch", "loop", "round trip"],
        "severity": "major",
        "issue_type": "performance",
    },
    "wrong_lru_order": {
        "lines": [21, 24],
        "description_keywords": ["lru", "order", "move_to_end", "access order", "insertion order", "OrderedDict"],
        "severity": "major",
        "issue_type": "logic",
    },
    "thread_safety": {
        "lines": [13],
        "description_keywords": ["thread", "lock", "concurrent", "race", "mutex", "atomic"],
        "severity": "critical",
        "issue_type": "concurrency",
    },
    "pickle_injection": {
        "lines": [26],
        "description_keywords": ["pickle", "deseri", "arbitrary code", "injection", "untrusted", "json"],
        "severity": "critical",
        "issue_type": "security",
    },
}

PULL_REQUEST = {
    "pull_request_title": "Introduce DistributedLRUCache with Redis backing for session store",
    "author": "senior-eng",
    "description": (
        "Implements a two-tier LRU cache (local + Redis) to reduce DB load by 60%. "
        "Designed for high-throughput production use. Please review thoroughly."
    ),
    "files_changed": [
        {
            "filename": "cache.py",
            "language": "python",
            "content": BUGGY_CODE,
            "line_count": BUGGY_CODE.count("\n") + 1,
        }
    ],
    "test_results": "Unit tests pass. Load tests not yet run.",
    "linter_output": "No issues found by flake8.",
}


def get_task_config() -> Dict[str, Any]:
    return {
        "task_id": TASK_ID,
        "max_steps": MAX_STEPS,
        "pull_request": PULL_REQUEST,
        "known_issues": KNOWN_ISSUES,
        "fixed_code": FIXED_CODE,
        "difficulty": "hard",
        "description": (
            "Review a production-grade distributed LRU cache implementation. "
            "Identify all concurrency, performance, correctness, and security issues. "
            "Provide a fully corrected implementation."
        ),
    }
