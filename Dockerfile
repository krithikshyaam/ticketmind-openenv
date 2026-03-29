# ─────────────────────────────────────────────────────────────────────────────
# TicketMind OpenEnv – Dockerfile
# Builds the FastAPI server; runs on port 7860 (HF Spaces standard).
# Target: linux/amd64, 2 vCPU / 8 GB RAM
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# Metadata
LABEL org.opencontainers.image.title="TicketMind OpenEnv"
LABEL org.opencontainers.image.description="Real-world customer support ticket resolution environment"
LABEL org.opencontainers.image.version="1.0.0"

# System deps (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY app/ ./app/
COPY main.py .
COPY openenv.yaml .
COPY inference.py .

# Set ownership
RUN chown -R appuser:appuser /app

USER appuser

# HF Spaces exposes port 7860
EXPOSE 7860

# Health check – ping /health every 30s
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Start the server
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--workers", "1", \
     "--log-level", "info"]
