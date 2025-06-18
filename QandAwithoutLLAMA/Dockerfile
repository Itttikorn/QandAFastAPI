FROM python:3.13-slim AS builder

WORKDIR /app
# Install system dependencies, essential python packages and create a virtual environment
RUN apt-get update && \ 
    apt-get install -y --no-install-recommends build-essential python3-dev python3-pip python3-venv python3-wheel liblapack-dev && \
    python3 -m venv .venv && .venv/bin/pip install --no-cache-dir -U pip setuptools wheel && \
    rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
# Install pytorch (cpu version)
RUN .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
# Install dependencies from requirements.txt and then remove unnecessary files (tests, pyc files)
RUN .venv/bin/pip install --no-cache-dir -r requirements.txt && \
    find /app/.venv \
      \( -type d \( -name test -o -name tests \) -exec rm -rf '{}' + \) -o \
      \( -type f \( -name '*.pyc' -o -name '*.pyo' \) -exec rm -f '{}' + \) || true

FROM python:3.13-slim
WORKDIR /app
COPY --from=builder /app /app
COPY main.py .
COPY data ./data
ENV PATH="/app/.venv/bin:$PATH"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]