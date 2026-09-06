"""Print the Mermaid diagram of the compiled graph (`uv run python -m scripts.diagram`)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")


def main() -> None:
    from app.config import Settings
    from app.dataforseo.client import build_client
    from app.graph.builder import build_graph, mermaid
    from app.graph.deps import Deps
    from app.observability.metrics import MetricsRegistry
    from tests.fakes import make_fake_gateway

    settings = Settings(_env_file=None)
    metrics = MetricsRegistry()
    gateway, _ = make_fake_gateway(metrics)
    deps = Deps(settings=settings, gateway=gateway, dfs_client=build_client(settings, metrics), metrics=metrics)
    print(mermaid(build_graph(deps)))


if __name__ == "__main__":
    main()
