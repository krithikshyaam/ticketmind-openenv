---
title: TicketMind OpenEnv
emoji: ??
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# TicketMind OpenEnv

A real-world OpenEnv environment for AI agents to handle customer support tickets.

## Tasks
- Easy - Ticket Classification
- Medium - Customer Response Generation
- Hard - Full Ticket Resolution

## Run Locally
pip install -r requirements.txt
python main.py

## API Endpoints
- POST /reset
- POST /step
- GET /state/{session_id}
- GET /health
- GET /tasks
