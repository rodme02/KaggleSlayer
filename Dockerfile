FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    build-essential libgomp1 git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml requirements.txt ./
RUN pip install --upgrade pip && pip install -e ".[dashboard]"

COPY . .
RUN pip install -e ".[dashboard]"

RUN useradd -m -u 1000 slayer && chown -R slayer:slayer /app
USER slayer

ENV MLFLOW_TRACKING_URI=file:/app/mlruns

# Default = CLI. docker-compose overrides command for the dashboard service.
ENTRYPOINT ["kaggle-slayer"]
CMD ["--help"]
