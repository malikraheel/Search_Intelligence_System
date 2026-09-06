"""Single-command entry point: `uv run main.py` (or `python main.py`).

Options:  --host 0.0.0.0  --port 8000  --reload
"""

from __future__ import annotations

import argparse

import uvicorn
from dotenv import load_dotenv


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Agentic Search Intelligence API server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    uvicorn.run(
        "app.api.app:app_factory",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="warning",  # our structlog JSON lines are the primary log stream
    )


if __name__ == "__main__":
    main()
