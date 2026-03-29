<<<<<<< HEAD
# 🎫 TicketMind OpenEnv

[![OpenEnv](https://img.shields.io/badge/OpenEnv-v1.0-blue)](https://openenv.dev)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/license-MIT-orange)](LICENSE)

A **real-world OpenEnv environment** where AI agents learn to **classify, respond to, and fully resolve customer support tickets** for a SaaS company. Three tasks of increasing difficulty provide a structured curriculum — from single-step classification to multi-step end-to-end resolution.

---

## 🌍 Environment Description

TicketMind simulates the support inbox of a growing SaaS company. Each episode presents the agent with a realistic customer ticket (billing disputes, technical crashes, account lockouts, feature requests, data-loss incidents) and requires it to navigate a structured resolution workflow.

This is a **real-world task environment** — not a game or toy — designed to measure practical skills: reading comprehension, empathy, decision-making under uncertainty, and appropriate escalation.

### Why Customer Support?

- **High economic value** — companies spend billions on support; automation is impactful
- **Rich action space** — classify, respond, escalate, request info, resolve
- **Natural partial rewards** — each sub-goal (correct classification, relevant response, right escalation) can be independently graded
- **Clear ground truth** — tickets have known categories, escalation needs, and resolution types

---

## 🗂️ Observation Space

Each step the agent receives an **Observation** containing:

| Field | Type | Description |
|-------|------|-------------|
| `session_id` | `string` | Unique episode identifier |
| `task_id` | `string` | Which task is being run |
| `ticket` | `object` | Full customer ticket (see below) |
| `conversation_history` | `array` | Messages exchanged so far |
| `available_actions` | `array[string]` | Actions allowed in this task |
| `step` | `integer` | Current step (0-indexed) |
| `max_steps` | `integer` | Episode step budget |
| `done` | `boolean` | Whether the episode has ended |
| `cumulative_reward` | `float` | Reward accumulated so far |
| `info` | `object` | Grader diagnostics and metadata |

### Ticket Object

| Field | Type | Description |
|-------|------|-------------|
| `ticket_id` | `string` | Unique identifier (e.g. `TKT-1001`) |
| `subject` | `string` | One-line summary |
| `body` | `string` | Full customer message |
| `customer_name` | `string` | Customer's name |
| `customer_email` | `string` | Customer's email |
| `priority` | `enum` | `low`, `medium`, `high`, `urgent` |
| `created_at` | `ISO8601` | Ticket submission time |
| `attachments` | `array[string]` | File names (contextual clues) |
| `previous_tickets` | `integer` | How many prior tickets this customer filed |

---

## 🎮 Action Space

| Action | Payload | Description |
|--------|---------|-------------|
| `classify` | `{category, confidence, sub_category?}` | Label the ticket type |
| `respond` | `{message, tone}` | Send a reply to the customer |
| `request_info` | `{questions[], reason}` | Ask customer for clarification |
| `escalate` | `{reason, target_team, priority_override?}` | Escalate to human team |
| `resolve` | `{resolution_summary, resolution_type, customer_satisfaction_predicted}` | Close the ticket |

Valid categories: `billing`, `technical`, `account`, `feature_request`, `other`

---

## 📋 Tasks

### Task 1: Ticket Classification *(Easy)*
> `task_id: ticket_classification` | max_steps: 3 | actions: classify, request_info

Correctly label the ticket's category. Graded on:
- **Correct category** (0.70) — exact match = 1.0, semantically related = 0.5, wrong = 0.0
- **Confidence calibration** (0.10) — reward high confidence when correct; penalise overconfidence when wrong
- **Wise clarification** (0.20) — penalise unnecessary `request_info` on self-contained tickets

### Task 2: Customer Response Generation *(Medium)*
> `task_id: ticket_response` | max_steps: 6 | actions: classify, respond, request_info, escalate

Craft an appropriate response and decide whether to escalate. Graded on:
- **Response relevance** (0.35) — does the reply address the actual issue?
- **Correct escalation** (0.25) — escalate when needed, don't when not
- **Tone quality** (0.20) — empathy markers, professional language
- **Info request decision** (0.20) — ask for clarification iff actually needed

### Task 3: Full Ticket Resolution *(Hard)*
> `task_id: full_resolution` | max_steps: 10 | actions: all

End-to-end multi-step resolution. Graded on:
- **Correct classification** (0.15)
- **Appropriate escalation** (0.25)
- **Response quality** (0.30) — relevance + tone
- **Resolution completeness** (0.20) — summary quality + correct resolution type
- **Efficiency** (0.10) — bonus for finishing in fewer steps

---

## 🏆 Reward Function

Rewards are **dense** — partial credit is emitted at every step to maximise training signal:

```
Step reward:  immediate feedback after each action
Final reward: holistic episode score (0.0–1.0) written to cumulative_reward at done=true
```

Example step rewards:
- `classify` with correct category → +0.70
- `respond` with 4/5 key facts addressed, good tone → +0.42
- `escalate` when escalation was needed → +0.25
- `request_info` on a self-contained ticket → +0.0 (mild penalty applied at final grade)

---

## 🚀 Setup Instructions

### Prerequisites
- Python 3.10+
- Docker (for containerised deployment)

### Local Development

```bash
# Clone the repository
git clone https://huggingface.co/spaces/YOUR_USERNAME/ticketmind-openenv
cd ticketmind-openenv

# Install dependencies
pip install -r requirements.txt

# Start the environment server
uvicorn app.main:app --host 0.0.0.0 --port 7860 --reload

# Verify it's running
curl http://localhost:7860/health
```

### Run the Baseline Inference Script

```bash
# Set required environment variables
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="your-api-key-here"
export ENV_URL="http://localhost:7860"    # optional, default

# Run
python inference.py
```

The script produces `inference_results.json` with reproducible scores.

### Run the Validator

```bash
python validate.py --url http://localhost:7860
```

All 30+ checks must pass before submitting.

### Docker

```bash
# Build
docker build -t ticketmind-openenv .

# Run
docker run -p 7860:7860 ticketmind-openenv

# Test
curl http://localhost:7860/health
```

---

## 🔌 API Quick Start

```python
import requests

BASE = "http://localhost:7860"

# 1. Start an episode
obs = requests.post(f"{BASE}/reset", json={
    "task_id": "full_resolution",
    "seed": 42
}).json()

session_id = obs["session_id"]
print(f"Ticket: {obs['ticket']['subject']}")

# 2. Take actions
result = requests.post(f"{BASE}/step", json={
    "session_id": session_id,
    "action_type": "classify",
    "payload": {"category": "technical", "confidence": 0.95}
}).json()

print(f"Step reward: {result['reward']}")

# 3. Check state
state = requests.get(f"{BASE}/state/{session_id}").json()
print(f"Step: {state['step']} / {state['max_steps']}")

# 4. Continue until done...
```

---

## 📁 Project Structure

```
ticketmind/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI app — OpenEnv endpoints
│   ├── models.py        # Pydantic typed models
│   ├── environment.py   # Session manager, step()/reset()/state()
│   ├── tasks.py         # Task definitions & ticket corpus
│   └── graders.py       # Scoring logic for all 3 tasks
├── openenv.yaml         # OpenEnv spec file
├── inference.py         # Baseline agent inference script
├── validate.py          # Pre-submission validator
├── Dockerfile           # HF Spaces compatible container
├── requirements.txt
└── README.md
```

---

## ⚙️ Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `API_BASE_URL` | Yes (inference) | OpenAI-compatible API base URL |
| `MODEL_NAME` | Yes (inference) | Model identifier for LLM calls |
| `HF_TOKEN` | Yes (inference) | Hugging Face / OpenAI API key |
| `ENV_URL` | No | TicketMind server URL (default: `http://localhost:7860`) |
| `SEED` | No | Random seed for reproducibility (default: `42`) |

---

## 📊 Baseline Scores

Measured with `gpt-4o-mini`, seed=42:

| Task | Difficulty | Expected Score Range |
|------|------------|---------------------|
| ticket_classification | Easy | 0.65–0.90 |
| ticket_response | Medium | 0.45–0.75 |
| full_resolution | Hard | 0.35–0.65 |

---

## 📜 License

MIT — see [LICENSE](LICENSE)
=======
---
title: Ticketmind Openenv
emoji: 🐠
colorFrom: yellow
colorTo: red
sdk: docker
pinned: false
license: mit
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
>>>>>>> 3341514e75213917c6d488ce6c4eff28381cecf7
