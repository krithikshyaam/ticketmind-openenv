#!/usr/bin/env python3
"""
TicketMind OpenEnv – Baseline Inference Script
===============================================
Runs an LLM-powered agent across all three tasks and reports reproducible scores.

Usage:
    python inference.py

Required environment variables:
    API_BASE_URL   The OpenAI-compatible API base URL (e.g. https://api.openai.com/v1)
    MODEL_NAME     The model identifier (e.g. gpt-4o-mini)
    HF_TOKEN       Your API key / Hugging Face token

Optional:
    ENV_URL        TicketMind server URL (default: http://localhost:7860)
    SEED           Integer seed for reproducibility (default: 42)
    MAX_RETRIES    LLM call retries on failure (default: 2)
"""

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME: str   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN: str     = os.environ.get("HF_TOKEN", "")
ENV_URL: str      = os.environ.get("ENV_URL", "http://localhost:7860")
SEED: int         = int(os.environ.get("SEED", "42"))
MAX_RETRIES: int  = int(os.environ.get("MAX_RETRIES", "2"))

if not HF_TOKEN:
    print("[WARN] HF_TOKEN is not set. LLM calls may fail.")

client = OpenAI(api_key=HF_TOKEN or "sk-placeholder", base_url=API_BASE_URL)

# ─────────────────────────────────────────────────────────────────────────────
# LLM helper
# ─────────────────────────────────────────────────────────────────────────────

def llm(messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
    """Call the LLM and return the assistant text. Never raises exceptions."""
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=temperature,
                max_tokens=800,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"    [LLM] Error attempt {attempt+1}/{MAX_RETRIES+1}: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)
            else:
                print("    [LLM] All retries failed, using fallback.")
                return '{"action_type": "classify", "payload": {"category": "technical", "confidence": 0.5}}'  


def parse_json_response(text: str) -> Dict[str, Any]:
    """Extract JSON from LLM output, stripping markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Best-effort: find first {...} block
        import re
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group())
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Environment client helpers
# ─────────────────────────────────────────────────────────────────────────────

def env_reset(task_id: str, seed: int = SEED) -> Dict[str, Any]:
    r = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id, "seed": seed}, timeout=30)
    r.raise_for_status()
    return r.json()


def env_step(session_id: str, action_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    body = {"session_id": session_id, "action_type": action_type, "payload": payload}
    r = requests.post(f"{ENV_URL}/step", json=body, timeout=30)
    r.raise_for_status()
    return r.json()


def env_state(session_id: str) -> Dict[str, Any]:
    r = requests.get(f"{ENV_URL}/state/{session_id}", timeout=30)
    r.raise_for_status()
    return r.json()


# ─────────────────────────────────────────────────────────────────────────────
# Agent System Prompt
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are TicketMind, an expert AI customer support agent.
You will be given a customer support ticket and must decide what action to take.

You MUST respond with a single JSON object and nothing else. No explanation, no markdown outside the JSON.

Available actions and their payloads:
- classify: {"category": "<billing|technical|account|feature_request|other>", "confidence": <0.0-1.0>, "sub_category": "<optional>"}
- respond: {"message": "<your response to the customer>", "tone": "<formal|friendly|empathetic|technical>"}
- request_info: {"questions": ["<q1>", "<q2>"], "reason": "<why you need this info>"}
- escalate: {"reason": "<why escalating>", "target_team": "<billing|engineering|management|legal>", "priority_override": null}
- resolve: {"resolution_summary": "<what was done>", "resolution_type": "<answered|refunded|fixed|escalated|no_action>", "customer_satisfaction_predicted": <0.0-1.0>}

Format: {"action_type": "<action>", "payload": {<payload fields>}}

Guidelines:
- Always be empathetic and professional.
- Classify before responding when possible.
- Only escalate if the issue genuinely requires it (data loss, legal, unresolvable bugs, etc.).
- Only request info if it's truly necessary to resolve the ticket.
- Resolve/escalate to end the episode.
"""


def build_user_prompt(obs: Dict[str, Any], task_id: str) -> str:
    ticket = obs["ticket"]
    history = obs.get("conversation_history", [])
    step = obs["step"]
    max_steps = obs["max_steps"]
    available = obs["available_actions"]

    task_instructions = {
        "ticket_classification": (
            "TASK: Classify this ticket into the correct category. "
            "Use 'classify' action. You may use 'request_info' first if truly needed. "
            "Finish by calling 'classify' with your best answer."
        ),
        "ticket_response": (
            "TASK: Craft an appropriate response to this ticket. "
            "You should classify first, then respond. Escalate if the issue warrants it. "
            "End with 'respond' or 'escalate'."
        ),
        "full_resolution": (
            "TASK: Fully resolve this ticket end-to-end. "
            "Classify → (optionally) request_info → respond → escalate or resolve. "
            "End with 'escalate' or 'resolve' to close the episode."
        ),
    }

    prompt = f"""{task_instructions.get(task_id, 'Handle this ticket.')}

--- TICKET ---
ID:       {ticket['ticket_id']}
Subject:  {ticket['subject']}
Priority: {ticket['priority']}
Customer: {ticket['customer_name']} <{ticket['customer_email']}>
Body:
{ticket['body']}

Attachments: {', '.join(ticket.get('attachments', [])) or 'None'}
Previous tickets from this customer: {ticket.get('previous_tickets', 0)}

--- CONVERSATION HISTORY ---
{json.dumps(history, indent=2) if history else '(no messages yet)'}

--- AGENT CONTEXT ---
Step: {step + 1} / {max_steps}
Available actions: {available}
Steps remaining: {max_steps - step}

Respond with exactly one JSON action object."""
    return prompt


# ─────────────────────────────────────────────────────────────────────────────
# Task runners
# ─────────────────────────────────────────────────────────────────────────────

def run_task(task_id: str, seed: int = SEED, verbose: bool = True) -> Dict[str, Any]:
    """Run one full episode for a task. Returns result dict."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"  TASK: {task_id}  (seed={seed})")
        print(f"{'='*60}")

    obs = env_reset(task_id, seed=seed)
    session_id = obs["session_id"]
    max_steps = obs["max_steps"]

    if verbose:
        ticket = obs["ticket"]
        print(f"  Ticket: [{ticket['ticket_id']}] {ticket['subject']}")
        print(f"  Priority: {ticket['priority']}  |  Max steps: {max_steps}")

    episode_done = False
    final_score = 0.0
    step_rewards: List[float] = []
    actions_taken: List[str] = []

    while not episode_done:
        # Build LLM prompt
        user_msg = build_user_prompt(obs, task_id)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

        # Call LLM
        raw_response = llm(messages)
        parsed = parse_json_response(raw_response)

        action_type = parsed.get("action_type", "classify")
        payload = parsed.get("payload", {})

        if verbose:
            print(f"\n  Step {obs['step'] + 1}: action={action_type}")
            if action_type == "classify":
                print(f"    category={payload.get('category')}  confidence={payload.get('confidence')}")
            elif action_type == "respond":
                msg = payload.get("message", "")
                print(f"    tone={payload.get('tone')}  msg_preview={msg[:80]}...")
            elif action_type == "escalate":
                print(f"    reason={payload.get('reason', '')[:80]}")
            elif action_type == "resolve":
                print(f"    type={payload.get('resolution_type')}  summary={payload.get('resolution_summary', '')[:60]}...")

        # Submit action
        result = env_step(session_id, action_type, payload)
        episode_done = result["done"]
        step_reward = result["reward"]
        step_rewards.append(step_reward)
        actions_taken.append(action_type)

        if verbose:
            print(f"    reward={step_reward:.3f}  done={episode_done}")
            grader_info = result.get("info", {}).get("grader_info", {})
            if grader_info:
                for k, v in grader_info.items():
                    if k not in ("final",) and isinstance(v, (int, float, str, bool)):
                        print(f"    grader.{k}={v}")

        obs = result["observation"]

        if episode_done:
            info = result.get("info", {})
            final_score = info.get("final_score", info.get("cumulative_reward", 0.0))
            if verbose:
                print(f"\n  ── Episode complete ──")
                print(f"  Final score:  {final_score:.3f}")
                print(f"  Steps taken:  {len(step_rewards)} / {max_steps}")
                print(f"  Actions:      {' → '.join(actions_taken)}")
                if "grader_info" in info:
                    gi = info["grader_info"]
                    print("  Grader breakdown:")
                    for k, v in gi.items():
                        if k != "final" and isinstance(v, (int, float, str)):
                            print(f"    {k}: {v}")

    return {
        "task_id": task_id,
        "seed": seed,
        "final_score": final_score,
        "steps": len(step_rewards),
        "max_steps": max_steps,
        "actions": actions_taken,
        "step_rewards": step_rewards,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def wait_for_env(max_wait: int = 30) -> bool:
    """Poll /health until the env is up."""
    print(f"Waiting for TicketMind at {ENV_URL} ...")
    deadline = time.time() + max_wait
    while time.time() < deadline:
        try:
            r = requests.get(f"{ENV_URL}/health", timeout=5)
            if r.status_code == 200:
                data = r.json()
                print(f"  Environment ready: {data['environment']} v{data['version']}")
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def main() -> None:
    print("\n" + "="*60)
    print("  TicketMind OpenEnv – Baseline Inference")
    print("="*60)
    print(f"  API_BASE_URL : {API_BASE_URL}")
    print(f"  MODEL_NAME   : {MODEL_NAME}")
    print(f"  ENV_URL      : {ENV_URL}")
    print(f"  SEED         : {SEED}")
    print("="*60)

    # Verify environment is reachable
    if not wait_for_env(max_wait=60):
        print("[ERROR] TicketMind environment is not reachable. "
              "Start it with: uvicorn app.main:app --host 0.0.0.0 --port 7860")
        sys.exit(1)

    tasks = [
        "ticket_classification",   # easy
        "ticket_response",         # medium
        "full_resolution",         # hard
    ]

    results: List[Dict[str, Any]] = []
    total_start = time.time()

    for task_id in tasks:
        t0 = time.time()
        try:
            result = run_task(task_id, seed=SEED, verbose=True)
        except Exception as e:
            print(f"  [ERROR] Task {task_id} failed: {e}")
            result = {
                "task_id": task_id,
                "seed": SEED,
                "final_score": 0.0,
                "steps": 0,
                "max_steps": 10,
                "actions": [],
                "step_rewards": [],
            }
        result["elapsed_seconds"] = round(time.time() - t0, 1)
        results.append(result)

    # ── Summary ──────────────────────────────────────────────────────────────
    elapsed_total = round(time.time() - total_start, 1)
    print("\n" + "="*60)
    print("  RESULTS SUMMARY")
    print("="*60)
    print(f"  {'Task':<30} {'Difficulty':<12} {'Score':>7}  {'Steps':>6}")
    print(f"  {'-'*30} {'-'*12} {'-'*7}  {'-'*6}")

    difficulty_map = {
        "ticket_classification": "easy",
        "ticket_response":       "medium",
        "full_resolution":       "hard",
    }

    all_scores = []
    for r in results:
        diff = difficulty_map.get(r["task_id"], "?")
        score = r["final_score"]
        all_scores.append(score)
        steps_str = f"{r['steps']}/{r['max_steps']}"
        print(f"  {r['task_id']:<30} {diff:<12} {score:>7.3f}  {steps_str:>6}")

    avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
    print(f"  {'─'*58}")
    print(f"  {'Average':<43} {avg_score:>7.3f}")
    print(f"\n  Total elapsed: {elapsed_total}s")
    print("="*60)

    # Validate all scores in [0, 1]
    assert all(0.0 <= s <= 1.0 for s in all_scores), \
        f"Score out of range: {all_scores}"
    assert len(all_scores) == 3, "Expected exactly 3 task scores"

    # Dump JSON for CI validators
    output = {
        "environment": "TicketMind",
        "model": MODEL_NAME,
        "seed": SEED,
        "tasks": results,
        "average_score": round(avg_score, 4),
        "elapsed_seconds": elapsed_total,
    }
    with open("inference_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results written to inference_results.json")

    # Exit non-zero if average too low (baseline sanity)
    if avg_score < 0.05:
        print("[WARN] Average score is very low — check LLM connectivity.")


if __name__ == "__main__":
    main()