# Multi-stage Dockerfile for NeuroMotion
# Uses multi-stage builds for smaller images

# ┌─────────────────────────────────────────────────────────────────┐
# │ STAGE 1: Builder                                              │
# └─────────────────────────────────────────────────────────────────┘
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry for dependency management
RUN pip install poetry

# Copy dependency files first (for better caching)
COPY backend/pyproject.toml backend/poetry.lock* ./

# Configure poetry
RUN poetry config virtualenvs.create false

# Install Python dependencies
RUN poetry install --only=main --no-interaction --no-ansi

# ┌─────────────────────────────────────────────────────────────────┐
# │ STAGE 2: Runtime Image                                        │
# └─────────────────────────────────────────────────────────────────┘
FROM python:3.11-slim AS runtime

WORKDIR /app

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \
    && mkdir -p /app/models /app/data

# Install system dependencies for runtime
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies from builder
COPY --from=builder /usr/local/lib/python*/site-packages /usr/local/lib/python*/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY backend/app/ ./app/
COPY backend/scripts/ ./scripts/
COPY backend/migrations/ ./migrations/

# Set proper permissions
RUN chown -R app:app /app

# Switch to non-root user
USER app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ┌─────────────────────────────────────────────────────────────────┐
# │ STAGE 3: Development                                            │
# └─────────────────────────────────────────────────────────────────┘
FROM runtime AS development

# Install development dependencies
RUN pip install pytest pytest-asyncio pytest-cov black ruff mypy

# Enable hot-reloading
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
