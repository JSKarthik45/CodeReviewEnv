"""
Task 2 (Medium): Security Vulnerability Review in a Flask Web Endpoint.

The agent reviews a Flask user-authentication endpoint containing:
  1. SQL injection vulnerability (string formatting into query)
  2. Plaintext password storage (no hashing)
  3. Missing rate limiting / brute-force protection
  4. Sensitive data leakage in error response
  5. Hardcoded secret key
"""
from __future__ import annotations
from typing import Any, Dict

TASK_ID = "task_2_medium_security"
MAX_STEPS = 12

BUGGY_CODE = '''\
import sqlite3
from flask import Flask, request, jsonify

app = Flask(__name__)
app.secret_key = "supersecret123"   # VULN 5: hardcoded secret key

DB_PATH = "users.db"


def get_db():
    return sqlite3.connect(DB_PATH)


@app.route("/login", methods=["POST"])
def login():
    username = request.json.get("username")
    password = request.json.get("password")

    db = get_db()
    cursor = db.cursor()

    # VULN 1: SQL injection — user input directly interpolated into query
    query = f"SELECT * FROM users WHERE username = \'{username}\' AND password = \'{password}\'"
    cursor.execute(query)
    user = cursor.fetchone()

    if user:
        return jsonify({"status": "ok", "user_id": user[0], "email": user[2]})
    else:
        # VULN 4: leaks whether username exists or password is wrong
        cursor.execute(f"SELECT id FROM users WHERE username = \'{username}\'")
        exists = cursor.fetchone()
        if exists:
            return jsonify({"error": f"Wrong password for user {username}"}), 401
        return jsonify({"error": f"User {username} does not exist"}), 404


@app.route("/register", methods=["POST"])
def register():
    username = request.json.get("username")
    password = request.json.get("password")   # VULN 2: stored in plaintext
    email = request.json.get("email")

    db = get_db()
    cursor = db.cursor()
    # VULN 1 again: SQL injection in insert
    cursor.execute(
        f"INSERT INTO users (username, password, email) VALUES (\'{username}\', \'{password}\', \'{email}\')"
    )
    db.commit()
    return jsonify({"status": "registered"})


# VULN 3: No rate limiting on login endpoint (brute-force possible)

if __name__ == "__main__":
    app.run(debug=True)
'''

FIXED_CODE = '''\
import os
import sqlite3
from flask import Flask, request, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY")  # read from env, never hardcode

limiter = Limiter(get_remote_address, app=app, default_limits=["200 per day", "50 per hour"])

DB_PATH = "users.db"


def get_db():
    return sqlite3.connect(DB_PATH)


@app.route("/login", methods=["POST"])
@limiter.limit("5 per minute")   # brute-force protection
def login():
    username = request.json.get("username")
    password = request.json.get("password")

    db = get_db()
    cursor = db.cursor()

    # Parameterised query — prevents SQL injection
    cursor.execute("SELECT id, password_hash FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()

    if user and check_password_hash(user[1], password):
        return jsonify({"status": "ok", "user_id": user[0]})
    # Generic error — does not reveal whether user exists
    return jsonify({"error": "Invalid credentials"}), 401


@app.route("/register", methods=["POST"])
def register():
    username = request.json.get("username")
    password = request.json.get("password")
    email = request.json.get("email")

    db = get_db()
    cursor = db.cursor()
    password_hash = generate_password_hash(password)
    cursor.execute(
        "INSERT INTO users (username, password_hash, email) VALUES (?, ?, ?)",
        (username, password_hash, email),
    )
    db.commit()
    return jsonify({"status": "registered"})


if __name__ == "__main__":
    app.run(debug=False)
'''

KNOWN_VULNERABILITIES = {
    "sql_injection_login": {
        "line": 23,
        "description_keywords": ["sql injection", "parameterized", "f-string", "format", "interpolat", "query"],
        "severity": "critical",
        "issue_type": "security",
    },
    "sql_injection_register": {
        "line": 44,
        "description_keywords": ["sql injection", "parameterized", "f-string", "format", "interpolat", "insert"],
        "severity": "critical",
        "issue_type": "security",
    },
    "plaintext_password": {
        "line": 39,
        "description_keywords": ["plaintext", "hash", "bcrypt", "werkzeug", "password", "store"],
        "severity": "critical",
        "issue_type": "security",
    },
    "no_rate_limiting": {
        "line": None,
        "description_keywords": ["rate limit", "brute force", "throttl", "limiter"],
        "severity": "major",
        "issue_type": "security",
    },
    "sensitive_data_leak": {
        "line": 30,
        "description_keywords": ["leak", "enumerat", "username exist", "generic error", "information disclos"],
        "severity": "major",
        "issue_type": "security",
    },
    "hardcoded_secret": {
        "line": 5,
        "description_keywords": ["hardcode", "secret", "env", "environment variable", "secret_key"],
        "severity": "major",
        "issue_type": "security",
    },
}

PULL_REQUEST = {
    "pull_request_title": "Implement user login and registration API endpoints",
    "author": "backend-dev",
    "description": (
        "Adds /login and /register REST endpoints backed by SQLite. "
        "Ready for production review."
    ),
    "files_changed": [
        {
            "filename": "auth.py",
            "language": "python",
            "content": BUGGY_CODE,
            "line_count": BUGGY_CODE.count("\n") + 1,
        }
    ],
    "test_results": "Manual testing passed on happy path.",
    "linter_output": "No linter warnings.",
}


def get_task_config() -> Dict[str, Any]:
    return {
        "task_id": TASK_ID,
        "max_steps": MAX_STEPS,
        "pull_request": PULL_REQUEST,
        "known_vulnerabilities": KNOWN_VULNERABILITIES,
        "fixed_code": FIXED_CODE,
        "difficulty": "medium",
        "description": (
            "Review a Flask authentication endpoint for security vulnerabilities. "
            "Identify all issues by category and severity, then provide a secure patched version."
        ),
    }
