#!/usr/bin/env python3
"""
TicketMind OpenEnv – Pre-Submission Validator
=============================================
Run this before submitting. It verifies:
  1. /health returns 200
  2. /tasks returns 3+ tasks
  3. reset() works for all tasks
  4. step() works and returns valid StepResult
  5. state() returns current session state
  6. All graders produce scores in [0.0, 1.0]
  7. openenv.yaml is parseable and complete

Usage:
    python validate.py [--url http://localhost:7860]
"""

import argparse
import json
import sys
import time
from typing import Any, Dict, List

import requests
import yaml

REQUIRED_TASKS = {"ticket_classification", "ticket_response", "full_resolution"}
REQUIRED_DIFFICULTIES = {"easy", "medium", "hard"}
REQUIRED_FIELDS_OBS = {
    "session_id", "task_id", "ticket", "available_actions",
    "step", "max_steps", "done", "cumulative_reward",
}
REQUIRED_FIELDS_STEP = {"observation", "reward", "done", "info"}


def check(name: str, condition: bool, detail: str = "") -> bool:
    status = "✅ PASS" if condition else "❌ FAIL"
    msg = f"  {status}  {name}"
    if detail:
        msg += f"  [{detail}]"
    print(msg)
    return condition


def validate(base_url: str) -> bool:
    results: List[bool] = []

    print(f"\n{'='*60}")
    print(f"  TicketMind OpenEnv – Validator")
    print(f"  Target: {base_url}")
    print(f"{'='*60}\n")

    # ── 1. Health ─────────────────────────────────────────────────────────────
    print("── 1. Liveness ──────────────────────────────────────────")
    try:
        r = requests.get(f"{base_url}/health", timeout=10)
        ok = r.status_code == 200
        data = r.json() if ok else {}
        results.append(check("GET /health returns 200", ok, f"status={r.status_code}"))
        results.append(check("Response has 'status' key", "status" in data))
        results.append(check("Status is 'ok'", data.get("status") == "ok"))
    except Exception as e:
        results.append(check("GET /health reachable", False, str(e)))
        print("\n[FATAL] Cannot reach environment. Is it running?")
        return False

    # ── 2. Task listing ───────────────────────────────────────────────────────
    print("\n── 2. Task Listing ──────────────────────────────────────")
    try:
        r = requests.get(f"{base_url}/tasks", timeout=10)
        results.append(check("GET /tasks returns 200", r.status_code == 200))
        tasks_data = r.json().get("tasks", [])
        task_ids = {t["task_id"] for t in tasks_data}
        difficulties = {t["difficulty"] for t in tasks_data}
        results.append(check("Has 3+ tasks", len(tasks_data) >= 3, f"count={len(tasks_data)}"))
        results.append(check(
            "Has required task IDs", REQUIRED_TASKS.issubset(task_ids),
            f"found={task_ids}",
        ))
        results.append(check(
            "Has easy/medium/hard", REQUIRED_DIFFICULTIES.issubset(difficulties),
            f"found={difficulties}",
        ))
        for t in tasks_data:
            has_rubric = bool(t.get("scoring_rubric"))
            results.append(check(
                f"Task '{t['task_id']}' has scoring_rubric", has_rubric,
            ))
    except Exception as e:
        results.append(check("GET /tasks parseable", False, str(e)))

    # ── 3. reset() ────────────────────────────────────────────────────────────
    print("\n── 3. reset() ───────────────────────────────────────────")
    sessions: Dict[str, str] = {}   # task_id → session_id
    for task_id in sorted(REQUIRED_TASKS):
        try:
            r = requests.post(
                f"{base_url}/reset",
                json={"task_id": task_id, "seed": 42},
                timeout=10,
            )
            results.append(check(
                f"reset({task_id}) returns 200", r.status_code == 200,
                f"status={r.status_code}",
            ))
            obs = r.json()
            has_required = REQUIRED_FIELDS_OBS.issubset(obs.keys())
            results.append(check(
                f"reset({task_id}) observation has required fields", has_required,
                f"missing={REQUIRED_FIELDS_OBS - obs.keys()}",
            ))
            results.append(check(
                f"reset({task_id}) done=false", obs.get("done") == False,
            ))
            results.append(check(
                f"reset({task_id}) step=0", obs.get("step") == 0,
            ))
            sessions[task_id] = obs["session_id"]
        except Exception as e:
            results.append(check(f"reset({task_id})", False, str(e)))

    # ── 4. state() ────────────────────────────────────────────────────────────
    print("\n── 4. state() ───────────────────────────────────────────")
    for task_id, session_id in sessions.items():
        try:
            r = requests.get(f"{base_url}/state/{session_id}", timeout=10)
            results.append(check(
                f"state({task_id}) returns 200", r.status_code == 200,
            ))
            s = r.json()
            results.append(check(
                f"state({task_id}) has session_id", s.get("session_id") == session_id,
            ))
        except Exception as e:
            results.append(check(f"state({task_id})", False, str(e)))

    # ── 5. step() ─────────────────────────────────────────────────────────────
    print("\n── 5. step() ────────────────────────────────────────────")
    for task_id, session_id in sessions.items():
        try:
            r = requests.post(
                f"{base_url}/step",
                json={
                    "session_id": session_id,
                    "action_type": "classify",
                    "payload": {"category": "billing", "confidence": 0.9},
                },
                timeout=10,
            )
            results.append(check(
                f"step({task_id}) returns 200", r.status_code == 200,
                f"status={r.status_code}",
            ))
            result = r.json()
            has_required = REQUIRED_FIELDS_STEP.issubset(result.keys())
            results.append(check(
                f"step({task_id}) result has required fields", has_required,
                f"missing={REQUIRED_FIELDS_STEP - result.keys()}",
            ))
            reward = result.get("reward", -999)
            results.append(check(
                f"step({task_id}) reward in [0,1]",
                isinstance(reward, (int, float)) and 0.0 <= reward <= 1.0,
                f"reward={reward}",
            ))
        except Exception as e:
            results.append(check(f"step({task_id})", False, str(e)))

    # ── 6. Grader scores in [0, 1] ────────────────────────────────────────────
    print("\n── 6. Grader Scores ─────────────────────────────────────")
    # Run a mini-episode to terminal for each task
    terminal_actions = {
        "ticket_classification": [
            {"action_type": "classify", "payload": {"category": "billing", "confidence": 0.8}},
        ],
        "ticket_response": [
            {"action_type": "classify", "payload": {"category": "technical"}},
            {"action_type": "escalate", "payload": {"reason": "Complex bug", "target_team": "engineering"}},
        ],
        "full_resolution": [
            {"action_type": "classify", "payload": {"category": "technical"}},
            {"action_type": "respond", "payload": {"message": "We are investigating the data loss. Our team is on it urgently.", "tone": "empathetic"}},
            {"action_type": "escalate", "payload": {"reason": "Data loss requires engineering", "target_team": "engineering"}},
        ],
    }

    for task_id, actions in terminal_actions.items():
        try:
            # Fresh episode
            r = requests.post(
                f"{base_url}/reset",
                json={"task_id": task_id, "seed": 99},
                timeout=10,
            )
            sid = r.json()["session_id"]

            final_score = None
            for act in actions:
                r = requests.post(
                    f"{base_url}/step",
                    json={"session_id": sid, **act},
                    timeout=10,
                )
                res = r.json()
                if res.get("done"):
                    info = res.get("info", {})
                    final_score = info.get("final_score", info.get("cumulative_reward"))

            if final_score is None:
                final_score = res.get("observation", {}).get("cumulative_reward", 0)

            in_range = isinstance(final_score, (int, float)) and 0.0 <= final_score <= 1.0
            results.append(check(
                f"grader({task_id}) score in [0,1]", in_range,
                f"score={final_score}",
            ))
        except Exception as e:
            results.append(check(f"grader({task_id})", False, str(e)))

    # ── 7. openenv.yaml ───────────────────────────────────────────────────────
    print("\n── 7. openenv.yaml ──────────────────────────────────────")
    try:
        with open("openenv.yaml") as f:
            spec = yaml.safe_load(f)
        results.append(check("openenv.yaml is parseable", True))
        results.append(check("Has 'name' field", bool(spec.get("name"))))
        results.append(check("Has 'version' field", bool(spec.get("version"))))
        results.append(check("Has 'tasks' field", bool(spec.get("tasks"))))
        results.append(check("Has 'observation_space'", bool(spec.get("observation_space"))))
        results.append(check("Has 'action_space'", bool(spec.get("action_space"))))
        spec_task_ids = {t["id"] for t in spec.get("tasks", [])}
        results.append(check(
            "YAML task IDs match API tasks", REQUIRED_TASKS.issubset(spec_task_ids),
            f"yaml_tasks={spec_task_ids}",
        ))
    except FileNotFoundError:
        results.append(check("openenv.yaml exists", False))
    except Exception as e:
        results.append(check("openenv.yaml valid YAML", False, str(e)))

    # ── Summary ───────────────────────────────────────────────────────────────
    passed = sum(results)
    total = len(results)
    failed = total - passed

    print(f"\n{'='*60}")
    print(f"  VALIDATION SUMMARY: {passed}/{total} checks passed")
    if failed == 0:
        print("  🎉 ALL CHECKS PASSED — ready to submit!")
    else:
        print(f"  ⚠️  {failed} check(s) failed — fix before submitting.")
    print(f"{'='*60}\n")

    return failed == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TicketMind pre-submission validator")
    parser.add_argument("--url", default="http://localhost:7860", help="Environment base URL")
    args = parser.parse_args()

    ok = validate(args.url)
    sys.exit(0 if ok else 1)
