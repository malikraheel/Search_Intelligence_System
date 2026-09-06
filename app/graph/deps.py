from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings
from app.dataforseo.client import DataForSEOClient
from app.gateway import Gateway
from app.observability.metrics import MetricsRegistry


@dataclass
class Deps:
    """Everything a node needs from the outside world. Injected via closures in the builder."""

    settings: Settings
    gateway: Gateway
    dfs_client: DataForSEOClient
    metrics: MetricsRegistry
