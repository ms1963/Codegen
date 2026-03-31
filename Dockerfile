# Dockerfile — Self-Extending AI System
# Multi-stage build for a lean production image.

# ── Stage 1: dependency installation ────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: runtime image ───────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Create a non-root user for security.
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy installed packages from builder stage.
COPY --from=builder /install /usr/local

# Copy application source.
COPY codegen/ ./codegen/

# Create a writable directory for tool persistence.
RUN mkdir -p /data && chown appuser:appuser /data

USER appuser

# Default environment variables (override at runtime).
ENV LLM_BASE_URL=http://host.docker.internal:11434/v1 \
    LLM_API_KEY=ollama \
    LLM_MODEL=qwen2.5:14b \
    TOOLS_FILE=/data/tools.json \
    LOG_LEVEL=INFO \
    PYTHONPATH=/app/codegen \
    PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "-m", "main"]
CMD []
