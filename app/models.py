"""
TicketMind OpenEnv – Typed Models
All request/response shapes for the step() / reset() / state() API.
"""

from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# Domain objects
# ─────────────────────────────────────────────

class CustomerTicket(BaseModel):
    ticket_id: str
    subject: str
    body: str
    customer_name: str
    customer_email: str
    priority: Literal["low", "medium", "high", "urgent"]
    created_at: str
    attachments: List[str] = Field(default_factory=list)
    previous_tickets: int = 0          # how many tickets this customer filed before


class ConversationMessage(BaseModel):
    role: Literal["customer", "agent", "system"]
    content: str
    timestamp: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────
# Observation – what the agent sees each step
# ─────────────────────────────────────────────

class Observation(BaseModel):
    session_id: str
    task_id: str
    ticket: CustomerTicket
    conversation_history: List[ConversationMessage] = Field(default_factory=list)
    available_actions: List[str]
    step: int
    max_steps: int
    done: bool
    cumulative_reward: float = 0.001
    info: Dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────
# Actions – what the agent can submit
# ─────────────────────────────────────────────

class ClassifyPayload(BaseModel):
    category: str                       # e.g. "billing", "technical", "account", etc.
    confidence: float = Field(ge=0.001, le=0.999, default=0.9)
    sub_category: Optional[str] = None


class RespondPayload(BaseModel):
    message: str                        # the agent's reply to the customer
    tone: Literal["formal", "friendly", "empathetic", "technical"] = "friendly"


class RequestInfoPayload(BaseModel):
    questions: List[str]                # clarifying questions to send back
    reason: str = ""


class EscalatePayload(BaseModel):
    reason: str
    target_team: Literal["billing", "engineering", "management", "legal"] = "engineering"
    priority_override: Optional[Literal["low", "medium", "high", "urgent"]] = None


class ResolvePayload(BaseModel):
    resolution_summary: str
    resolution_type: Literal["answered", "refunded", "fixed", "escalated", "no_action"]
    customer_satisfaction_predicted: float = Field(ge=0.001, le=0.999, default=0.8)


class Action(BaseModel):
    session_id: str
    action_type: Literal["classify", "respond", "request_info", "escalate", "resolve"]
    payload: Dict[str, Any]


# ─────────────────────────────────────────────
# Step result
# ─────────────────────────────────────────────

class StepResult(BaseModel):
    observation: Observation
    reward: float                       # step reward strictly between 0 and 1
    done: bool
    truncated: bool = False             # ran out of steps without resolving
    info: Dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────
# Reset / State API
# ─────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = "ticket_classification"
    seed: Optional[int] = None
    config: Dict[str, Any] = Field(default_factory=dict)


class StateResponse(BaseModel):
    session_id: str
    task_id: str = "ticket_classification"
    step: int
    max_steps: int
    done: bool
    cumulative_reward: float
    ticket: CustomerTicket
    conversation_history: List[ConversationMessage]


# ─────────────────────────────────────────────
# Task metadata
# ─────────────────────────────────────────────

class TaskInfo(BaseModel):
    task_id: str = "ticket_classification"
    name: str
    difficulty: Literal["easy", "medium", "hard"]
    description: str
    max_steps: int
    action_space: List[str]
    scoring_rubric: Dict[str, float]