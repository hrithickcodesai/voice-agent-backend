.PHONY: install format check clean

install:
	uv sync

format:
	uv run ruff check --fix --select I .
	uv run ruff format .

check:
	uv run ruff check .
	uv run ruff format --check .

clean:
	find . -type d \( -name __pycache__ -o -name .ruff_cache -o -name .pytest_cache \) -prune -exec rm -rf {} +
