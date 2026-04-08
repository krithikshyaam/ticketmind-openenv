"""
TicketMind – Agent Graders
Each grader scores 0.0 – 1.0 based on the agent's action history.
Partial credit is emitted at each step for training signal density.
"""

from __future__ import annotations
import re
from typing import Any, Dict, List, Optional, Tuple

from .tasks import get_ticket


def _clamp(score: float) -> float:
    """Ensure score is strictly between 0.0 and 1.0 (exclusive)."""
    return round(max(0.001, min(0.999, float(score))), 4)


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

_RELATED_CATEGORIES: Dict[str, List[str]] = {
    "billing": ["payment", "refund", "invoice", "charge", "subscription"],
    "technical": ["bug", "software", "crash", "error", "integration", "api", "data_loss", "migration"],
    "account": ["access", "authentication", "login", "password", "sso", "mfa"],
    "feature_request": ["feature", "product", "enhancement", "roadmap"],
    "other": [],
}


def _category_score(predicted: str, true_category: str, valid_categories: List[str]) -> float:
    predicted = predicted.lower().strip()
    true_category = true_category.lower().strip()
    if predicted == true_category:
        return 0.95
    # Check if semantically related
    true_related = _RELATED_CATEGORIES.get(true_category, [])
    if predicted in true_related or predicted in valid_categories:
        return 0.5
    # Check reverse mapping
    for cat, related in _RELATED_CATEGORIES.items():
        if predicted in related and cat == true_category:
            return 0.5
    return 0.05


def _response_relevance(response: str, key_facts: List[str]) -> float:
    """Score strictly between 0 and 1 for how many key facts the response acknowledges."""
    if not response or not key_facts:
        return 0.05
    response_lower = response.lower()
    hits = sum(1 for f in key_facts if f.lower() in response_lower)
    raw = hits / len(key_facts)
    return round(max(0.05, min(0.95, raw)), 2)


def _tone_score(response: str, tone: str) -> float:
    """Heuristic tone scoring - always returns strictly between 0 and 1."""
    response_lower = response.lower()
    empathy_markers = [
        "understand", "sorry", "apologize", "frustrat", "concern",
        "inconvenien", "sincerely", "thank you for", "appreciate",
    ]
    technical_markers = [
        "log", "version", "update", "configure", "reinstall",
        "cache", "debug", "steps to reproduce",
    ]
    formal_markers = ["dear", "sincerely", "regards", "furthermore", "hereby"]

    if tone in ("empathetic", "friendly"):
        hits = sum(1 for m in empathy_markers if m in response_lower)
        return round(max(0.05, min(0.95, hits / 4)), 3)
    if tone == "technical":
        hits = sum(1 for m in technical_markers if m in response_lower)
        return round(max(0.05, min(0.95, hits / 4)), 3)
    if tone == "formal":
        hits = sum(1 for m in formal_markers if m in response_lower)
        return round(max(0.05, min(0.95, hits / 3)), 3)
    return 0.5


def _response_length_ok(response: str) -> bool:
    words = len(response.split())
    return 30 <= words <= 400


def _resolution_summary_quality(summary: str, key_facts: List[str]) -> float:
    if not summary:
        return 0.05
    relevance = _response_relevance(summary, key_facts)
    length_ok = 0.15 if len(summary.split()) >= 15 else 0.001
    return round(max(0.05, min(0.95, relevance + length_ok)), 2)


# ─────────────────────────────────────────────────────────────────────────────
# Task 1: Ticket Classification (Easy)
# ─────────────────────────────────────────────────────────────────────────────

class ClassificationGrader:
    """
    Scores the agent on ticket classification.
    
    Rubric:
      correct_category          0.70
      confidence_calibration    0.10
      used_clarification_wisely 0.20  (penalise unnecessary request_info)
    """

    def __init__(self, ticket_id: str):
        ticket = get_ticket(ticket_id)
        self.true_category: str = ticket["true_category"]
        self.valid_categories: List[str] = ticket["valid_categories"]
        self.key_facts: List[str] = ticket.get("key_facts", [])
        self.required_info: List[str] = ticket.get("required_info", [])

    def grade_step(
        self,
        action_type: str,
        payload: Dict[str, Any],
        step: int,
        history: List[Dict[str, Any]],
    ) -> Tuple[float, Dict[str, Any]]:
        """Return (step_reward, info_dict)."""
        info: Dict[str, Any] = {}

        if action_type == "classify":
            predicted = payload.get("category", "")
            confidence = float(payload.get("confidence", 0.9))
            cat_score = _category_score(predicted, self.true_category, self.valid_categories)

            # Confidence calibration: reward certainty when correct, penalise overconfidence when wrong
            if cat_score >= 0.9:
                conf_score = round(max(0.001, min(0.999, confidence)), 3)  # high confidence + correct = good
            else:
                conf_score = round(max(0.001, min(0.999, 1.0 - confidence)), 3)  # low confidence when wrong = partially ok

            # Penalty for unnecessary request_info on tickets that have enough info
            clarif_penalty = 0.001
            used_clarif = any(h["action_type"] == "request_info" for h in history)
            if used_clarif and not self.required_info:
                clarif_penalty = 0.2  # wasted a step

            reward = (
                0.70 * cat_score
                + 0.10 * conf_score
                + 0.20 * (1.0 - clarif_penalty)
            )
            info = {
                "predicted_category": predicted,
                "true_category": self.true_category,
                "cat_score": cat_score,
                "conf_score": conf_score,
                "clarif_penalty": clarif_penalty,
            }
            return _clamp(reward), info

        if action_type == "request_info":
            # Mild positive signal if the ticket actually needs info
            reward = 0.15 if self.required_info else -0.05
            info = {"clarification_needed": bool(self.required_info)}
            return round(max(0.001, reward), 3), info

        return 0.001, {"note": "no reward for this action in classification task"}

    def final_grade(self, history: List[Dict[str, Any]]) -> Tuple[float, Dict[str, Any]]:
        """Final episode score after all steps."""
        classify_actions = [h for h in history if h["action_type"] == "classify"]
        if not classify_actions:
            return 0.001, {"error": "No classify action taken"}
        # Use the last classify action as final answer
        last = classify_actions[-1]
        reward, info = self.grade_step("classify", last["payload"], 0, history)
        info["final"] = True
        return _clamp(reward), info


# ─────────────────────────────────────────────────────────────────────────────
# Task 2: Ticket Response (Medium)
# ─────────────────────────────────────────────────────────────────────────────

class ResponseGrader:
    """
    Scores the agent on crafting an appropriate customer response.
    
    Rubric:
      response_relevance        0.35
      correct_escalation        0.25
      tone_quality              0.20
      appropriate_info_request  0.20
    """

    def __init__(self, ticket_id: str):
        ticket = get_ticket(ticket_id)
        self.true_category = ticket["true_category"]
        self.valid_categories = ticket["valid_categories"]
        self.needs_escalation: bool = ticket.get("needs_escalation", False)
        self.key_facts: List[str] = ticket.get("key_facts", [])
        self.required_info: List[str] = ticket.get("required_info", [])

        # Tracking
        self._classified_correctly: bool = False
        self._responded: bool = False
        self._escalated: bool = False
        self._requested_info: bool = False
        self._best_response_score: float = 0.001
        self._best_tone_score: float = 0.001

    def grade_step(
        self,
        action_type: str,
        payload: Dict[str, Any],
        step: int,
        history: List[Dict[str, Any]],
    ) -> Tuple[float, Dict[str, Any]]:
        info: Dict[str, Any] = {"action_type": action_type}

        if action_type == "classify":
            predicted = payload.get("category", "")
            cat_score = _category_score(predicted, self.true_category, self.valid_categories)
            self._classified_correctly = cat_score >= 0.5
            # Small positive signal for correct classification
            return round(max(0.001, 0.1 * cat_score), 3), info

        if action_type == "request_info":
            self._requested_info = True
            questions = payload.get("questions", [])
            if self.required_info:
                # Reward if agent asks about required info
                reward = 0.15 if questions else 0.05
            else:
                # Slight penalty for unnecessary clarification
                reward = 0.05
            return round(max(0.001, reward), 3), info

        if action_type == "respond":
            message = payload.get("message", "")
            tone = payload.get("tone", "friendly")
            self._responded = True

            relevance = _response_relevance(message, self.key_facts)
            tone_s = _tone_score(message, tone)
            length_ok = 0.1 if _response_length_ok(message) else 0.001

            step_reward = 0.35 * relevance + 0.20 * tone_s + length_ok
            self._best_response_score = max(self._best_response_score, relevance)
            self._best_tone_score = max(self._best_tone_score, tone_s)
            info.update({"relevance": relevance, "tone_score": tone_s})
            return round(step_reward, 3), info

        if action_type == "escalate":
            self._escalated = True
            if self.needs_escalation:
                reward = 0.25  # correct escalation
            else:
                reward = -0.10  # unnecessary escalation
            info["needs_escalation"] = self.needs_escalation
            return round(max(0.001, reward), 3), info

        return 0.001, info

    def final_grade(self, history: List[Dict[str, Any]]) -> Tuple[float, Dict[str, Any]]:
        if not self._responded:
            return 0.001, {"error": "Agent never responded to customer"}

        escalation_score = 0.05
        if self.needs_escalation and self._escalated:
            escalation_score = 0.95
        elif not self.needs_escalation and not self._escalated:
            escalation_score = 0.95
        elif self.needs_escalation and not self._escalated:
            escalation_score = 0.05
        else:
            escalation_score = 0.3  # escalated unnecessarily – mild penalty

        info_score = 0.05
        if self.required_info and self._requested_info:
            info_score = 0.95
        elif not self.required_info and not self._requested_info:
            info_score = 0.95
        elif self.required_info and not self._requested_info:
            info_score = 0.05
        else:
            info_score = 0.5

        final = (
            0.35 * self._best_response_score
            + 0.25 * escalation_score
            + 0.20 * self._best_tone_score
            + 0.20 * info_score
        )
        return _clamp(final), {
            "response_relevance": self._best_response_score,
            "escalation_score": escalation_score,
            "tone_score": self._best_tone_score,
            "info_request_score": info_score,
            "final": True,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Task 3: Full Resolution (Hard)
# ─────────────────────────────────────────────────────────────────────────────

class ResolutionGrader:
    """
    Scores the agent on full end-to-end ticket resolution.

    Rubric (weighted):
      correct_classification    0.15
      appropriate_escalation    0.25
      response_quality          0.30
      resolution_completeness   0.20
      efficiency (step penalty) 0.10
    """

    def __init__(self, ticket_id: str, max_steps: int = 10):
        ticket = get_ticket(ticket_id)
        self.true_category = ticket["true_category"]
        self.valid_categories = ticket["valid_categories"]
        self.needs_escalation: bool = ticket.get("needs_escalation", False)
        self.key_facts: List[str] = ticket.get("key_facts", [])
        self.required_info: List[str] = ticket.get("required_info", [])
        self.expected_resolution_type: str = ticket.get("resolution_type", "answered")
        self.max_steps = max_steps

        # Running state
        self._class_score: float = 0.001
        self._escalated: bool = False
        self._responded: bool = False
        self._resolved: bool = False
        self._best_response_score: float = 0.001
        self._best_tone: float = 0.001
        self._resolution_score: float = 0.001
        self._steps_used: int = 0

    def grade_step(
        self,
        action_type: str,
        payload: Dict[str, Any],
        step: int,
        history: List[Dict[str, Any]],
    ) -> Tuple[float, Dict[str, Any]]:
        self._steps_used = step
        info: Dict[str, Any] = {"action_type": action_type, "step": step}

        if action_type == "classify":
            predicted = payload.get("category", "")
            self._class_score = _category_score(predicted, self.true_category, self.valid_categories)
            return round(max(0.001, 0.05 * self._class_score), 3), info

        if action_type == "request_info":
            questions = payload.get("questions", [])
            reward = 0.05 if self.required_info and questions else 0.001
            return round(max(0.001, reward), 3), info

        if action_type == "respond":
            message = payload.get("message", "")
            tone = payload.get("tone", "empathetic")
            self._responded = True
            relevance = _response_relevance(message, self.key_facts)
            tone_s = _tone_score(message, tone)
            self._best_response_score = max(self._best_response_score, relevance)
            self._best_tone = max(self._best_tone, tone_s)
            step_reward = max(0.001, 0.10 * relevance + 0.05 * tone_s)
            info.update({"relevance": relevance, "tone": tone_s})
            return round(step_reward, 3), info

        if action_type == "escalate":
            self._escalated = True
            reward = 0.10 if self.needs_escalation else 0.001
            info["needs_escalation"] = self.needs_escalation
            return reward, info

        if action_type == "resolve":
            self._resolved = True
            summary = payload.get("resolution_summary", "")
            res_type = payload.get("resolution_type", "answered")
            summary_q = _resolution_summary_quality(summary, self.key_facts)
            type_match = 0.95 if res_type == self.expected_resolution_type else 0.3
            self._resolution_score = 0.6 * summary_q + 0.4 * type_match
            step_reward = max(0.001, 0.10 * self._resolution_score)
            info.update({
                "summary_quality": summary_q,
                "type_match": type_match,
                "resolution_type_predicted": res_type,
                "resolution_type_expected": self.expected_resolution_type,
            })
            return round(step_reward, 3), info

        return 0.001, info

    def final_grade(self, history: List[Dict[str, Any]]) -> Tuple[float, Dict[str, Any]]:
        # Escalation sub-score
        if self.needs_escalation and self._escalated:
            esc_score = 0.95
        elif not self.needs_escalation and not self._escalated:
            esc_score = 0.95
        elif self.needs_escalation and not self._escalated:
            esc_score = 0.05
        else:
            esc_score = 0.2

        # Response quality (tone + relevance average)
        resp_quality = (self._best_response_score * 0.7 + self._best_tone * 0.3) if self._responded else 0.001

        # Efficiency: fraction of steps unused (reward for finishing early)
        steps_used = max(1, self._steps_used)
        efficiency = max(0.001, min(0.999, 1.0 - (steps_used / self.max_steps)))

        # Resolution completeness
        resolution = self._resolution_score if self._resolved else 0.001

        final = (
            0.15 * self._class_score
            + 0.25 * esc_score
            + 0.30 * resp_quality
            + 0.20 * resolution
            + 0.10 * efficiency
        )
        return _clamp(final), {
            "classification_score": self._class_score,
            "escalation_score": esc_score,
            "response_quality": resp_quality,
            "resolution_completeness": resolution,
            "efficiency": efficiency,
            "steps_used": steps_used,
            "final": True,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_grader(task_id: str, ticket_id: str, max_steps: int = 10):
    if task_id == "ticket_classification":
        return ClassificationGrader(ticket_id)
    if task_id == "ticket_response":
        return ResponseGrader(ticket_id)
    if task_id == "full_resolution":
        return ResolutionGrader(ticket_id, max_steps=max_steps)
    raise ValueError(f"Unknown task_id: {task_id}")