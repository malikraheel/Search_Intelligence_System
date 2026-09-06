# Convenience targets. Every target is a thin wrapper around `uv run ...`, so on machines without
# `make` (e.g. plain Windows) run the underlying command shown in the README instead.

.PHONY: install run dev test test-verbose lint format typecheck demo demo-chaos demo-recheck diagram docker-build docker-up clean

install:
	uv sync

run:
	uv run main.py

dev:
	uv run main.py --reload

test:
	uv run pytest -q

test-verbose:
	uv run pytest -v

lint:
	uv run ruff check app tests main.py scripts

format:
	uv run ruff format app tests main.py scripts

typecheck:
	uv run mypy app

demo:
	uv run scripts/demo_run.py --recheck

demo-offline:
	uv run scripts/demo_run.py --fake-llm --recheck

demo-chaos:
	uv run scripts/demo_run.py --fake-llm --chaos "serp_organic:429,503,ok;llm_responses:500,500,500,500,500,500,500,500,500,500,500,500;keyword_volume:timeout,ok"

diagram:
	uv run python -c "from scripts.diagram import main; main()"

docker-build:
	docker build -t search-intel .

docker-up:
	docker compose up --build

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache demo.db search_intel.db
