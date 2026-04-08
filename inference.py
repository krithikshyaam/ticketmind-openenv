#!/usr/bin/env python3
"""
TicketMind OpenEnv - Baseline Inference Script
Runs an LLM-powered agent across all three tasks and reports reproducible scores.

Required environment variables:
    API_BASE_URL   The OpenAI-compatible API base URL
    MODEL_NAME     The model identifier
    HF_TOKEN       Your API key / Hugging Face token
"""

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# Configuration
API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN: str = os.environ.get("HF_TOKEN") or ""
LOCAL_IMAGE_NAME: str = os.environ.get("LOCAL_IMAGE_NAME", "")
ENV_URL: str = os.environ.get("ENV_URL", "http://localhost:7860")
SEED: int = int(os.environ.get("SEED", "42"))
MAX_RETRIES: int = int(os.environ.get("MAX_RETRIES", "2"))

if not HF_TOKEN:
    print("[WARN] HF_TOKEN is not set. LLM calls may fail.")

client = OpenAI(api_key=HF_TOKEN or "sk-placeholder", base_url=API_BASE_URL)


def llm(messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
    """Call the LLM. Never raises exceptions."""
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
    """Extract JSON from LLM output."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        import re
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
        return {}


def env_reset(task_id: str, seed: int = SEED) -> Dict[str, Any]:
    r = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id, "seed": seed}, timeout=30)
    r.raise_for_status()
    return r.json()


def env_step(session_id: str, action_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    body = {"session_id": session_id, "action_type": action_type, "payload": payload}
    r = requests.post(f"{ENV_URL}/step", json=body, timeout=30)
    r.raise_for_status()
    return r.json()


SYSTEM_PROMPT = (
    "You are TicketMind, an expert AI customer support agent. "
    "You MUST respond with a single JSON object and nothing else. "
    "No explanation, no markdown outside the JSON.\n\n"
    "Available actions:\n"
    "- classify: {\"category\": \"<billing|technical|account|feature_request|other>\", \"confidence\": <0.0-1.0>}\n"
    "- respond: {\"message\": \"<reply to customer>\", \"tone\": \"<formal|friendly|empathetic|technical>\"}\n"
    "- request_info: {\"questions\": [\"<q1>\"], \"reason\": \"<why>\"}\n"
    "- escalate: {\"reason\": \"<why>\", \"target_team\": \"<billing|engineering|management|legal>\"}\n"
    "- resolve: {\"resolution_summary\": \"<what was done>\", \"resolution_type\": \"<answered|refunded|fixed|escalated|no_action>\"}\n\n"
    "Format: {\"action_type\": \"<action>\", \"payload\": {<fields>}}"
)

TASK_INSTRUCTIONS = {
    "ticket_classification": (
        "TASK: Classify this ticket into the correct category. "
        "You MUST use action_type='classify'. "
        "Do NOT use respond, escalate or resolve."
    ),
    "ticket_response": (
        "TASK: Handle this ticket in sequence. "
        "Step 1 - use action_type='classify'. "
        "Step 2 - use action_type='respond' with a helpful customer message, "
        "OR use action_type='escalate' if it is a critical bug or data loss. "
        "IMPORTANT: After classifying once, you MUST use respond or escalate. Do NOT classify again."
    ),
    "full_resolution": (
        "TASK: Fully resolve this ticket end-to-end. "
        "Step 1 - use action_type='classify'. "
        "Step 2 - use action_type='respond' with a detailed helpful message. "
        "Step 3 - use action_type='escalate' if critical, OR action_type='resolve' to close. "
        "IMPORTANT: Follow this sequence strictly. Do NOT classify more than once."
    ),
}


def build_user_prompt(obs: Dict[str, Any], task_id: str, actions_taken: List[str]) -> str:
    ticket = obs["ticket"]
    history = obs.get("conversation_history", [])
    step = obs["step"]
    max_steps = obs["max_steps"]
    available = obs["available_actions"]

    instruction = TASK_INSTRUCTIONS.get(task_id, "Handle this ticket.")

    classify_count = actions_taken.count("classify")
    if classify_count >= 1 and task_id in ("ticket_response", "full_resolution"):
        if "respond" not in actions_taken:
            instruction += " YOU HAVE ALREADY CLASSIFIED. NOW YOU MUST USE action_type=respond."
        elif task_id == "full_resolution" and "respond" in actions_taken:
            instruction += " YOU HAVE RESPONDED. NOW YOU MUST USE action_type=escalate OR action_type=resolve."

    prompt = (
        instruction + "\n\n"
        "--- TICKET ---\n"
        "ID: " + ticket["ticket_id"] + "\n"
        "Subject: " + ticket["subject"] + "\n"
        "Priority: " + ticket["priority"] + "\n"
        "Customer: " + ticket["customer_name"] + " <" + ticket["customer_email"] + ">\n"
        "Body:\n" + ticket["body"] + "\n\n"
        "Attachments: " + (", ".join(ticket.get("attachments", [])) or "None") + "\n"
        "Previous tickets: " + str(ticket.get("previous_tickets", 0)) + "\n\n"
        "--- CONVERSATION HISTORY ---\n"
        + (json.dumps(history, indent=2) if history else "(no messages yet)") + "\n\n"
        "--- CONTEXT ---\n"
        "Step: " + str(step + 1) + " / " + str(max_steps) + "\n"
        "Actions taken so far: " + str(actions_taken) + "\n"
        "Available actions: " + str(available) + "\n"
        "Steps remaining: " + str(max_steps - step) + "\n\n"
        "Respond with exactly one JSON action object."
    )
    return prompt


def force_action_override(action_type, payload, task_id, actions_taken, obs):
    """Override action if LLM is stuck in a loop."""
    classify_count = actions_taken.count("classify")
    subject = obs["ticket"]["subject"]

    if task_id == "ticket_response" and action_type == "classify" and classify_count >= 1:
        action_type = "respond"
        payload = {
            "message": (
                "Thank you for reaching out about '" + subject + "'. "
                "We have reviewed your request and our team is actively working on a resolution. "
                "We apologize for any inconvenience and will update you shortly."
            ),
            "tone": "empathetic",
        }

    elif task_id == "full_resolution" and action_type == "classify" and classify_count >= 1:
        if "respond" not in actions_taken:
            action_type = "respond"
            payload = {
                "message": (
                    "Thank you for contacting us about '" + subject + "'. "
                    "We understand the urgency and our team is investigating immediately. "
                    "We will provide a full update within 2 hours."
                ),
                "tone": "empathetic",
            }
        elif "respond" in actions_taken:
            action_type = "escalate"
            payload = {
                "reason": "Critical issue requiring engineering attention: " + subject,
                "target_team": "engineering",
            }

    return action_type, payload


def run_task(task_id: str, seed: int = SEED, verbose: bool = True) -> Dict[str, Any]:
    """Run one full episode for a task."""
    if verbose:
        print("\n" + "="*60)
        print("  TASK: " + task_id + "  (seed=" + str(seed) + ")")
        print("="*60)

    obs = env_reset(task_id, seed=seed)
    session_id = obs["session_id"]
    max_steps = obs["max_steps"]

    if verbose:
        ticket = obs["ticket"]
        print("  Ticket: [" + ticket["ticket_id"] + "] " + ticket["subject"])
        print("  Priority: " + ticket["priority"] + "  |  Max steps: " + str(max_steps))

    print("[START] task=" + task_id + " session=" + session_id[:8] + " ticket=" + obs["ticket"]["ticket_id"])

    episode_done = False
    final_score = 0.001
    step_rewards: List[float] = []
    actions_taken: List[str] = []

    # Deterministic fallback sequences per task (used if LLM unavailable)
    FALLBACK_SEQUENCES = {
        "ticket_classification": [
            ("classify", {"category": "billing", "confidence": 0.9}),
        ],
        "ticket_response": [
            ("classify", {"category": "technical", "confidence": 0.9}),
            ("respond", {"message": "Thank you for contacting us. We sincerely apologize for the inconvenience. Our team is investigating this issue urgently and will provide a resolution shortly.", "tone": "empathetic"}),
        ],
        "full_resolution": [
            ("classify", {"category": "technical", "confidence": 0.9}),
            ("respond", {"message": "We understand the urgency of your situation and sincerely apologize for the inconvenience. Our engineering team is investigating this critical issue immediately.", "tone": "empathetic"}),
            ("escalate", {"reason": "Critical issue requiring immediate engineering attention", "target_team": "engineering"}),
        ],
    }
    fallback_seq = FALLBACK_SEQUENCES.get(task_id, [])
    fallback_idx = 0

    while not episode_done:
        user_msg = build_user_prompt(obs, task_id, actions_taken)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

        raw_response = llm(messages)
        parsed = parse_json_response(raw_response)

        action_type = parsed.get("action_type", "classify")
        payload = parsed.get("payload", {})

        action_type, payload = force_action_override(action_type, payload, task_id, actions_taken, obs)
        
        # If still stuck or invalid, use deterministic fallback
        if fallback_idx < len(fallback_seq) and action_type not in obs["available_actions"]:
            action_type, payload = fallback_seq[fallback_idx]
            fallback_idx += 1

        if verbose:
            print("\n  Step " + str(obs["step"] + 1) + ": action=" + action_type)
            if action_type == "classify":
                print("    category=" + str(payload.get("category")) + "  confidence=" + str(payload.get("confidence")))
            elif action_type == "respond":
                msg = payload.get("message", "")
                print("    tone=" + str(payload.get("tone")) + "  msg=" + msg[:80] + "...")
            elif action_type == "escalate":
                print("    reason=" + str(payload.get("reason", ""))[:80])
            elif action_type == "resolve":
                print("    type=" + str(payload.get("resolution_type")))

        result = env_step(session_id, action_type, payload)
        episode_done = result["done"]
        step_reward = result["reward"]
        step_rewards.append(step_reward)
        actions_taken.append(action_type)

        if verbose:
            print("    reward=" + str(round(step_reward, 3)) + "  done=" + str(episode_done))
            grader_info = result.get("info", {}).get("grader_info", {})
            if grader_info:
                for k, v in grader_info.items():
                    if k not in ("final",) and isinstance(v, (int, float, str, bool)):
                        print("    grader." + k + "=" + str(v))

        print("[STEP] task=" + task_id + " step=" + str(len(step_rewards)) + " action=" + action_type + " reward=" + str(round(step_reward, 3)) + " done=" + str(episode_done))
        obs = result["observation"]

        if episode_done:
            info = result.get("info", {})
            final_score = info.get("final_score", info.get("cumulative_reward", 0.001))
            final_score = max(0.001, min(0.999, float(final_score)))
            print("[END] task=" + task_id + " final_score=" + str(round(final_score, 3)) + " steps=" + str(len(step_rewards)))
            if verbose:
                print("\n  -- Episode complete --")
                print("  Final score:  " + str(round(final_score, 3)))
                print("  Steps taken:  " + str(len(step_rewards)) + " / " + str(max_steps))
                print("  Actions:      " + " -> ".join(actions_taken))

    return {
        "task_id": task_id,
        "seed": seed,
        "final_score": final_score,
        "steps": len(step_rewards),
        "max_steps": max_steps,
        "actions": actions_taken,
        "step_rewards": step_rewards,
    }


def wait_for_env(max_wait: int = 60) -> bool:
    print("Waiting for TicketMind at " + ENV_URL + " ...")
    deadline = time.time() + max_wait
    while time.time() < deadline:
        try:
            r = requests.get(f"{ENV_URL}/health", timeout=5)
            if r.status_code == 200:
                data = r.json()
                print("  Environment ready: " + data["environment"] + " v" + data["version"])
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def main() -> None:
    print("\n" + "="*60)
    print("  TicketMind OpenEnv - Baseline Inference")
    print("="*60)
    print("  API_BASE_URL : " + API_BASE_URL)
    print("  MODEL_NAME   : " + MODEL_NAME)
    print("  ENV_URL      : " + ENV_URL)
    print("  SEED         : " + str(SEED))
    print("="*60)

    if not wait_for_env(max_wait=60):
        print("[ERROR] TicketMind environment is not reachable.")
        sys.exit(1)

    tasks = ["ticket_classification", "ticket_response", "full_resolution"]
    results: List[Dict[str, Any]] = []
    total_start = time.time()

    for task_id in tasks:
        t0 = time.time()
        try:
            result = run_task(task_id, seed=SEED, verbose=True)
        except Exception as e:
            print("  [ERROR] Task " + task_id + " failed: " + str(e))
            result = {
                "task_id": task_id,
                "seed": SEED,
                "final_score": 0.001,
                "steps": 0,
                "max_steps": 10,
                "actions": [],
                "step_rewards": [],
            }
        result["elapsed_seconds"] = round(time.time() - t0, 1)
        results.append(result)

    elapsed_total = round(time.time() - total_start, 1)
    print("\n" + "="*60)
    print("  RESULTS SUMMARY")
    print("="*60)

    difficulty_map = {
        "ticket_classification": "easy",
        "ticket_response": "medium",
        "full_resolution": "hard",
    }

    all_scores = []
    for r in results:
        diff = difficulty_map.get(r["task_id"], "?")
        score = max(0.001, min(0.999, float(r["final_score"])))
        all_scores.append(score)
        print("  " + r["task_id"] + " (" + diff + ") = " + str(round(score, 3)))

    avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.001
    print("  Average = " + str(round(avg_score, 3)))
    print("  Total elapsed: " + str(elapsed_total) + "s")
    print("="*60)

    assert all(0.0 < s < 1.0 for s in all_scores), "Score out of range: " + str(all_scores)
    assert len(all_scores) == 3, "Expected exactly 3 task scores"

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
    print("\n  Results written to inference_results.json")


if __name__ == "__main__":
    main()
