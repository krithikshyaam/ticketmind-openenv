"""
TicketMind – Task Definitions
Three tasks of increasing difficulty backed by realistic SaaS support tickets.
"""

from typing import Any, Dict, List
# ─────────────────────────────────────────────────────────────────────────────
# Shared ticket corpus (used across tasks)
# ─────────────────────────────────────────────────────────────────────────────

TICKET_CORPUS: List[Dict[str, Any]] = [
    # ── BILLING ───────────────────────────────────────────────────────────────
    {
        "ticket_id": "TKT-1001",
        "subject": "Charged twice for March subscription",
        "body": (
            "Hello, I was just reviewing my bank statement and noticed that I was "
            "charged $49.99 twice on March 1st — once at 02:14 AM and again at "
            "02:17 AM. I only have one account and one subscription. Could you "
            "please refund the duplicate charge? My account email is "
            "james.o@company.com."
        ),
        "customer_name": "James O'Brien",
        "customer_email": "james.o@company.com",
        "priority": "high",
        "created_at": "2024-03-01T09:05:00Z",
        "previous_tickets": 0,
        "attachments": ["bank_statement_march.png"],
        "true_category": "billing",
        "valid_categories": ["billing", "payment", "refund"],
        "needs_escalation": False,
        "resolution_type": "refunded",
        "key_facts": ["double charge", "$49.99", "March 1st"],
        "required_info": [],
        "difficulty": "easy",
    },
    # ── TECHNICAL ─────────────────────────────────────────────────────────────
    {
        "ticket_id": "TKT-1002",
        "subject": "App crashes every time I try to export a report",
        "body": (
            "Hi support team, since the update yesterday (v3.4.2) my desktop app "
            "crashes with no error message whenever I click 'Export to PDF' on any "
            "report. I'm on Windows 11, Intel i7, 16 GB RAM. I've tried "
            "reinstalling — same issue. This is blocking my end-of-month reporting. "
            "Logs are attached."
        ),
        "customer_name": "Priya Sharma",
        "customer_email": "p.sharma@fintech.io",
        "priority": "urgent",
        "created_at": "2024-03-02T08:30:00Z",
        "previous_tickets": 3,
        "attachments": ["crash_log.txt"],
        "true_category": "technical",
        "valid_categories": ["technical", "bug", "software"],
        "needs_escalation": True,
        "resolution_type": "escalated",
        "key_facts": ["v3.4.2", "PDF export", "Windows 11", "crash"],
        "required_info": [],
        "difficulty": "medium",
    },
    # ── ACCOUNT ACCESS ────────────────────────────────────────────────────────
    {
        "ticket_id": "TKT-1003",
        "subject": "Can't log in – password reset email never arrives",
        "body": (
            "I've been locked out of my account for two days. I click 'Forgot "
            "Password' and it says the email was sent but nothing arrives — "
            "I've checked spam/junk. My email is m.chen@startup.dev. "
            "I have an active team plan with 12 seats, so this is impacting "
            "my whole team. Please help urgently."
        ),
        "customer_name": "Michael Chen",
        "customer_email": "m.chen@startup.dev",
        "priority": "urgent",
        "created_at": "2024-03-03T14:10:00Z",
        "previous_tickets": 1,
        "attachments": [],
        "true_category": "account",
        "valid_categories": ["account", "access", "authentication", "login"],
        "needs_escalation": False,
        "resolution_type": "fixed",
        "key_facts": ["locked out", "password reset", "12 seats", "team plan"],
        "required_info": ["account_id"],
        "difficulty": "medium",
    },
    # ── FEATURE REQUEST ───────────────────────────────────────────────────────
    {
        "ticket_id": "TKT-1004",
        "subject": "Request: bulk CSV export for all projects",
        "body": (
            "We manage 200+ projects and currently have to export each one "
            "individually which is very time-consuming. Is there a way to export "
            "all project data as a single CSV/Excel file? If this isn't available "
            "yet, could you add it to the roadmap? Happy to join a beta test."
        ),
        "customer_name": "Amara Diallo",
        "customer_email": "amara@designhive.co",
        "priority": "low",
        "created_at": "2024-03-04T11:45:00Z",
        "previous_tickets": 5,
        "attachments": [],
        "true_category": "feature_request",
        "valid_categories": ["feature_request", "feature", "product"],
        "needs_escalation": False,
        "resolution_type": "answered",
        "key_facts": ["bulk export", "CSV", "200 projects"],
        "required_info": [],
        "difficulty": "easy",
    },
    # ── COMPLEX / MULTI-STEP ──────────────────────────────────────────────────
    {
        "ticket_id": "TKT-1005",
        "subject": "Data loss after migration – need urgent recovery",
        "body": (
            "We just completed the migration from your Legacy plan to Enterprise "
            "following instructions from your migration guide. After migration, "
            "we can see only 40% of our historical data. The other 60% is simply "
            "gone from our dashboard. We're a healthcare analytics company and "
            "this data is critical for compliance reporting. We need immediate "
            "assistance. Our account ID is ACC-88812. Migration was done at "
            "2024-03-05 09:00 UTC by our admin user elena@medco.health."
        ),
        "customer_name": "Elena Vasquez",
        "customer_email": "elena@medco.health",
        "priority": "urgent",
        "created_at": "2024-03-05T10:15:00Z",
        "previous_tickets": 8,
        "attachments": ["migration_log.csv", "data_comparison.xlsx"],
        "true_category": "technical",
        "valid_categories": ["technical", "data_loss", "migration"],
        "needs_escalation": True,
        "resolution_type": "escalated",
        "key_facts": [
            "data loss", "60% missing", "healthcare", "compliance",
            "ACC-88812", "migration", "enterprise plan",
        ],
        "required_info": [],
        "difficulty": "hard",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# Task Registry
# ─────────────────────────────────────────────────────────────────────────────

TASK_REGISTRY: Dict[str, Dict[str, Any]] = {
    # ── EASY ─────────────────────────────────────────────────────────────────
    "ticket_classification": {
        "task_id": "ticket_classification",
        "name": "Ticket Classification",
        "difficulty": "easy",
        "description": (
            "Classify the incoming customer support ticket into one of the "
            "predefined categories (billing, technical, account, feature_request, "
            "other). A correct classification earns full reward; a semantically "
            "related category earns partial credit."
        ),
        "max_steps": 3,
        "action_space": ["classify", "request_info"],
        "scoring_rubric": {
            "correct_category": 0.7,
            "used_clarification_wisely": 0.2,
            "confidence_calibration": 0.1,
        },
        "ticket_ids": ["TKT-1001", "TKT-1004"],   # seeded selection
        "valid_categories": [
            "billing", "technical", "account",
            "feature_request", "other",
        ],
    },
    # ── MEDIUM ────────────────────────────────────────────────────────────────
    "ticket_response": {
        "task_id": "ticket_response",
        "name": "Customer Response Generation",
        "difficulty": "medium",
        "description": (
            "Read the ticket, optionally request clarifying info, then craft a "
            "helpful, empathetic, and accurate response. Graded on relevance, "
            "tone, completeness, and whether escalation decisions were correct."
        ),
        "max_steps": 6,
        "action_space": ["classify", "respond", "request_info", "escalate"],
        "scoring_rubric": {
            "response_relevance": 0.35,
            "correct_escalation": 0.25,
            "tone_quality": 0.20,
            "appropriate_info_request": 0.20,
        },
        "ticket_ids": ["TKT-1002", "TKT-1003"],
        "valid_categories": [
            "billing", "technical", "account",
            "feature_request", "other",
        ],
    },
    # ── HARD ──────────────────────────────────────────────────────────────────
    "full_resolution": {
        "task_id": "full_resolution",
        "name": "Full Ticket Resolution",
        "difficulty": "hard",
        "description": (
            "Handle a complex, multi-step support scenario end-to-end: classify "
            "the ticket, gather necessary information, respond appropriately, "
            "decide whether to escalate, and produce a resolution summary. "
            "Partial credit is awarded at each sub-goal milestone."
        ),
        "max_steps": 10,
        "action_space": ["classify", "respond", "request_info", "escalate", "resolve"],
        "scoring_rubric": {
            "correct_classification": 0.15,
            "appropriate_escalation": 0.25,
            "response_quality": 0.30,
            "resolution_completeness": 0.20,
            "efficiency": 0.10,
        },
        "ticket_ids": ["TKT-1005"],
        "valid_categories": [
            "billing", "technical", "account",
            "feature_request", "other",
        ],
    },
}


def get_task(task_id: str) -> Dict[str, Any]:
    if task_id not in TASK_REGISTRY:
        raise ValueError(
            f"Unknown task '{task_id}'. "
            f"Available: {list(TASK_REGISTRY.keys())}"
        )
    return TASK_REGISTRY[task_id]


def get_ticket(ticket_id: str) -> Dict[str, Any]:
    for t in TICKET_CORPUS:
        if t["ticket_id"] == ticket_id:
            return t
    raise ValueError(f"Ticket '{ticket_id}' not found in corpus")


def list_tasks() -> List[Dict[str, Any]]:
    return [
        {
            "task_id": v["task_id"],
            "name": v["name"],
            "difficulty": v["difficulty"],
            "description": v["description"],
            "max_steps": v["max_steps"],
            "action_space": v["action_space"],
            "scoring_rubric": v["scoring_rubric"],
        }
        for v in TASK_REGISTRY.values()
    ]
