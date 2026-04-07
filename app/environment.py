"""
TicketMind – Environment Engine
Manages per-session state and implements the OpenEnv step()/reset()/state() contract.
"""

from __future__ import annotations
import random
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .graders import build_grader
from .models import (
    Action,
    ConversationMessage,
    CustomerTicket,
    Observation,
    ResetRequest,
    StateResponse,
    StepResult,
)
from .tasks import TASK_REGISTRY, get_task, get_ticket


# ─────────────────────────────────────────────────────────────────────────────
# Session dataclass (plain dict for simplicity)
# ─────────────────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_ticket(ticket_data: Dict[str, Any]) -> CustomerTicket:
    return CustomerTicket(
        ticket_id=ticket_data["ticket_id"],
        subject=ticket_data["subject"],
        body=ticket_data["body"],
        customer_name=ticket_data["customer_name"],
        customer_email=ticket_data["customer_email"],
        priority=ticket_data["priority"],
        created_at=ticket_data["created_at"],
        attachments=ticket_data.get("attachments", []),
        previous_tickets=ticket_data.get("previous_tickets", 0),
    )


class TicketMindEnv:
    """
    In-memory environment manager.

    Each session is isolated; concurrent sessions are supported.
    """

    def __init__(self) -> None:
        self._sessions: Dict[str, Dict[str, Any]] = {}

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def reset(self, req: ResetRequest) -> Observation:
        task = get_task(req.task_id)
        rng = random.Random(req.seed) if req.seed is not None else random.Random()

        # Pick a ticket for this episode
        ticket_ids = task["ticket_ids"]
        ticket_id = rng.choice(ticket_ids)
        ticket_data = get_ticket(ticket_id)
        ticket = _make_ticket(ticket_data)

        session_id = str(uuid.uuid4())
        grader = build_grader(req.task_id, ticket_id, max_steps=task["max_steps"])

        session: Dict[str, Any] = {
            "session_id": session_id,
            "task_id": req.task_id,
            "task": task,
            "ticket": ticket,
            "ticket_data": ticket_data,
            "grader": grader,
            "step": 0,
            "max_steps": task["max_steps"],
            "done": False,
            "truncated": False,
            "cumulative_reward": 0.0,
            "conversation_history": [],
            "action_history": [],
            "seed": req.seed,
        }

        self._sessions[session_id] = session

        return self._make_observation(session, info={"reset": True})

    def step(self, action: Action) -> StepResult:
        session = self._get_session(action.session_id)

        if session["done"]:
            obs = self._make_observation(session, info={"error": "Episode already done"})
            return StepResult(
                observation=obs,
                reward=0.0,
                done=True,
                truncated=session["truncated"],
                info={"error": "Episode already done"},
            )

        action_type = action.action_type
        payload = action.payload

        # Validate action type against task's action space
        allowed = session["task"]["action_space"]
        if action_type not in allowed:
            obs = self._make_observation(
                session,
                info={"error": f"Action '{action_type}' not allowed in task '{session['task_id']}'. Allowed: {allowed}"},
            )
            return StepResult(
                observation=obs,
                reward=0.0,
                done=False,
                info={"error": f"Invalid action: {action_type}"},
            )

        # Record action in history
        session["step"] += 1
        action_record = {
            "step": session["step"],
            "action_type": action_type,
            "payload": payload,
            "timestamp": _now(),
        }
        session["action_history"].append(action_record)

        # Grade the step
        grader = session["grader"]
        step_reward, grade_info = grader.grade_step(
            action_type, payload, session["step"], session["action_history"]
        )
        session["cumulative_reward"] = round(
            max(0.001, min(0.999, session["cumulative_reward"] + step_reward)), 4
        )

        # Append to conversation history for respond / request_info actions
        if action_type == "respond":
            session["conversation_history"].append(
                ConversationMessage(
                    role="agent",
                    content=payload.get("message", ""),
                    timestamp=_now(),
                    metadata={"tone": payload.get("tone", "friendly")},
                )
            )
        elif action_type == "request_info":
            questions = payload.get("questions", [])
            session["conversation_history"].append(
                ConversationMessage(
                    role="agent",
                    content="Questions: " + " | ".join(questions),
                    timestamp=_now(),
                    metadata={"type": "info_request"},
                )
            )

        # Determine termination
        done = False
        truncated = False
        final_score: Optional[float] = None
        final_info: Dict[str, Any] = {}

        # Task-specific terminal actions
        task_terminal: Dict[str, set] = {
            "ticket_classification": {"classify"},
            "ticket_response":       {"escalate", "respond"},
            "full_resolution":       {"resolve", "escalate"},
        }
        terminal_actions = task_terminal.get(session["task_id"], {"resolve", "escalate"})
        if action_type in terminal_actions or session["step"] >= session["max_steps"]:
            done = True
            truncated = session["step"] >= session["max_steps"] and action_type not in terminal_actions
            final_score, final_info = grader.final_grade(session["action_history"])
            # Clamp strictly between 0 and 1 (exclusive) as required
            final_score = round(max(0.001, min(0.999, float(final_score))), 4)
            session["cumulative_reward"] = final_score   # override with holistic score
            session["done"] = True
            session["truncated"] = truncated
            grade_info.update(final_info)

        # Merge info
        info = {
            "step_reward": step_reward,
            "cumulative_reward": session["cumulative_reward"],
            "grader_info": grade_info,
        }
        if final_score is not None:
            info["final_score"] = final_score

        obs = self._make_observation(session, info=info)
        return StepResult(
            observation=obs,
            reward=step_reward,
            done=done,
            truncated=truncated,
            info=info,
        )

    def state(self, session_id: str) -> StateResponse:
        session = self._get_session(session_id)
        return StateResponse(
            session_id=session_id,
            task_id=session["task_id"],
            step=session["step"],
            max_steps=session["max_steps"],
            done=session["done"],
            cumulative_reward=session["cumulative_reward"],
            ticket=session["ticket"],
            conversation_history=session["conversation_history"],
        )

    def delete_session(self, session_id: str) -> bool:
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def active_sessions(self) -> int:
        return len(self._sessions)

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _get_session(self, session_id: str) -> Dict[str, Any]:
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"Session '{session_id}' not found. Call /reset first.")
        return session

    def _make_observation(
        self, session: Dict[str, Any], info: Dict[str, Any] | None = None
    ) -> Observation:
        task = session["task"]
        return Observation(
            session_id=session["session_id"],
            task_id=session["task_id"],
            ticket=session["ticket"],
            conversation_history=session["conversation_history"],
            available_actions=task["action_space"],
            step=session["step"],
            max_steps=session["max_steps"],
            done=session["done"],
            cumulative_reward=session["cumulative_reward"],
            info=info or {},
        )


# Singleton environment instance shared across the FastAPI app
env = TicketMindEnv()