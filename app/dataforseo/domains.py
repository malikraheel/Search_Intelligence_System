"""Domain normalisation and matching helpers."""

from __future__ import annotations

from urllib.parse import urlparse


def normalize_domain(value: str) -> str:
    v = value.strip().lower()
    if "://" in v:
        v = urlparse(v).netloc or v
    v = v.split("/")[0].split(":")[0]
    if v.startswith("www."):
        v = v[4:]
    return v


def domain_matches(candidate: str | None, target: str) -> bool:
    """True when `candidate` is `target` or a subdomain of it."""
    if not candidate:
        return False
    c, t = normalize_domain(candidate), normalize_domain(target)
    return c == t or c.endswith("." + t)


def domain_from_url(url: str | None) -> str | None:
    if not url:
        return None
    return normalize_domain(url) or None
