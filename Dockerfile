FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Install third-party dependencies first to maximize Docker layer caching
COPY pyproject.toml uv.lock* ./
RUN uv sync --no-dev --no-install-project

# Copy project source and operational datasets, then install local package
COPY . .
RUN uv sync --no-dev && \
    useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app

USER appuser
ENV PATH="/app/.venv/bin:$PATH"

# Expose FastAPI backend (8000) and Streamlit frontend (8501)
EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

# Default command starts the enterprise FastAPI backend
CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
