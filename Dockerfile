FROM python:3.12-slim AS base
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy LITELLM_LOCAL_MODEL_COST_MAP=True PYTHONUNBUFFERED=1
WORKDIR /app

COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project

COPY app ./app
COPY main.py ./
RUN mkdir -p /data
ENV DATABASE_URL=sqlite:////data/search_intel.db

EXPOSE 8000
CMD ["uv", "run", "--no-sync", "main.py", "--host", "0.0.0.0", "--port", "8000"]
