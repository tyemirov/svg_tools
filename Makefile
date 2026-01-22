.PHONY: test lint ci

test:
	uv run --no-project --with pytest --with pillow -- python -m pytest -q

lint:
	uv run --no-project --with mypy -- mypy --strict domain service

ci: lint test
