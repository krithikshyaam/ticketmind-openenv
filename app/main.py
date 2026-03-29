"""
TicketMind OpenEnv – FastAPI Application
Implements the OpenEnv REST spec: reset() / step() / state()
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
import time

from .environment import env
from .models import Action, ResetRequest
from .tasks import list_tasks, TASK_REGISTRY

# ─────────────────────────────────────────────────────────────────────────────
# App bootstrap
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="TicketMind OpenEnv",
    description=(
        "A real-world OpenEnv environment where AI agents learn to classify, "
        "respond to, and resolve customer support tickets across three difficulty levels."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_start_time = time.time()


# ─────────────────────────────────────────────────────────────────────────────
# Health & metadata
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    """Landing page with environment overview."""
    return HTMLResponse(content="""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>TicketMind OpenEnv</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
           background: #0f172a; color: #e2e8f0; min-height: 100vh; }
    .hero { background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            padding: 60px 40px; text-align: center; border-bottom: 1px solid #1e3a5f; }
    h1 { font-size: 2.8rem; font-weight: 800; color: #38bdf8;
         letter-spacing: -1px; margin-bottom: 12px; }
    .tagline { color: #94a3b8; font-size: 1.1rem; max-width: 600px; margin: 0 auto 32px; }
    .badge { display: inline-block; background: #1e3a5f; color: #38bdf8;
             border: 1px solid #38bdf8; border-radius: 999px; padding: 4px 16px;
             font-size: 0.8rem; font-weight: 600; margin: 4px; }
    .container { max-width: 960px; margin: 0 auto; padding: 48px 24px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            gap: 20px; margin: 32px 0; }
    .card { background: #1e293b; border: 1px solid #334155; border-radius: 12px;
            padding: 24px; }
    .card h3 { color: #38bdf8; font-size: 1rem; font-weight: 700; margin-bottom: 8px; }
    .card p { color: #94a3b8; font-size: 0.9rem; line-height: 1.6; }
    .difficulty { display: inline-block; border-radius: 4px; padding: 2px 8px;
                  font-size: 0.75rem; font-weight: 700; margin-bottom: 10px; }
    .easy { background:#14532d; color:#4ade80; }
    .medium { background:#78350f; color:#fbbf24; }
    .hard { background:#7f1d1d; color:#f87171; }
    .endpoint { background: #0f172a; border: 1px solid #1e3a5f; border-radius: 8px;
                padding: 12px 16px; margin: 8px 0; font-family: monospace; font-size: 0.85rem; }
    .method { color: #4ade80; font-weight: 700; margin-right: 8px; }
    .path { color: #38bdf8; }
    .desc { color: #64748b; font-size: 0.8rem; margin-top: 4px; }
    h2 { font-size: 1.5rem; font-weight: 700; color: #e2e8f0; margin: 40px 0 16px; }
    .links { display: flex; gap: 12px; justify-content: center; flex-wrap: wrap; padding: 24px 0; }
    .btn { background: #38bdf8; color: #0f172a; font-weight: 700; padding: 10px 24px;
           border-radius: 8px; text-decoration: none; font-size: 0.9rem; }
    .btn-ghost { background: transparent; color: #38bdf8; border: 1px solid #38bdf8; }
  </style>
</head>
<body>
  <div class="hero">
    <h1>🎫 TicketMind</h1>
    <p class="tagline">A real-world OpenEnv environment for training AI agents to handle customer support tickets end-to-end.</p>
    <span class="badge">OpenEnv v1.0</span>
    <span class="badge">3 Tasks</span>
    <span class="badge">Easy → Hard</span>
    <span class="badge">Partial Rewards</span>
    <div class="links">
      <a href="/docs" class="btn">API Docs</a>
      <a href="/tasks" class="btn btn-ghost">List Tasks</a>
      <a href="/health" class="btn btn-ghost">Health</a>
    </div>
  </div>
  <div class="container">
    <h2>Tasks</h2>
    <div class="grid">
      <div class="card">
        <span class="difficulty easy">EASY</span>
        <h3>Ticket Classification</h3>
        <p>Classify the incoming support ticket into the correct category. Scored on accuracy, confidence calibration, and wise use of clarification.</p>
      </div>
      <div class="card">
        <span class="difficulty medium">MEDIUM</span>
        <h3>Customer Response Generation</h3>
        <p>Read the ticket, optionally request info, craft an accurate & empathetic response, and decide if escalation is needed.</p>
      </div>
      <div class="card">
        <span class="difficulty hard">HARD</span>
        <h3>Full Ticket Resolution</h3>
        <p>Multi-step end-to-end resolution: classify, gather info, respond, escalate if needed, and write a resolution summary.</p>
      </div>
    </div>

    <h2>OpenEnv API</h2>
    <div class="endpoint"><span class="method">POST</span><span class="path">/reset</span><div class="desc">Start a new episode. Returns initial observation + session_id.</div></div>
    <div class="endpoint"><span class="method">POST</span><span class="path">/step</span><div class="desc">Submit an action. Returns (observation, reward, done, info).</div></div>
    <div class="endpoint"><span class="method">GET</span><span class="path">/state/{session_id}</span><div class="desc">Inspect current session state without side effects.</div></div>
    <div class="endpoint"><span class="method">GET</span><span class="path">/tasks</span><div class="desc">List all available tasks with metadata.</div></div>
    <div class="endpoint"><span class="method">GET</span><span class="path">/health</span><div class="desc">Liveness probe.</div></div>
  </div>
</body>
</html>
""")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "environment": "TicketMind",
        "version": "1.0.0",
        "uptime_seconds": round(time.time() - _start_time, 1),
        "active_sessions": env.active_sessions(),
    }


@app.get("/tasks")
async def get_tasks():
    """List all available tasks with full metadata."""
    return {"tasks": list_tasks()}


@app.get("/tasks/{task_id}")
async def get_task_detail(task_id: str):
    """Get full details for a single task."""
    if task_id not in TASK_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
    task = TASK_REGISTRY[task_id].copy()
    task.pop("ticket_ids", None)     # don't expose answer key
    return task


# ─────────────────────────────────────────────────────────────────────────────
# OpenEnv core API
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/reset")
async def reset(req: ResetRequest):
    """
    Reset the environment and start a new episode.

    Returns an Observation with the initial ticket and available actions.
    The `session_id` in the response must be passed to all subsequent /step calls.
    """
    try:
        obs = env.reset(req)
        return obs.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step")
async def step(action: Action):
    """
    Take one action in the environment.

    Returns StepResult containing:
    - observation: new state the agent sees
    - reward: float in [0, 1] for this step
    - done: whether the episode has ended
    - truncated: whether max steps was reached
    - info: grader breakdown and diagnostics
    """
    try:
        result = env.step(action)
        return result.model_dump()
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state/{session_id}")
async def state(session_id: str):
    """
    Return the current state of a session without taking any action.
    Safe to call at any time; does not advance the episode.
    """
    try:
        s = env.state(session_id)
        return s.model_dump()
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Clean up a session to free memory."""
    deleted = env.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return {"deleted": session_id}


# ─────────────────────────────────────────────────────────────────────────────
# Error handlers
# ─────────────────────────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "type": type(exc).__name__},
    )
