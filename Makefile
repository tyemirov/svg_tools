.PHONY: test lint ci

UV_CACHE_DIR ?= $(CURDIR)/.cache/uv

test:
	mkdir -p "$(UV_CACHE_DIR)"
	UV_CACHE_DIR="$(UV_CACHE_DIR)" uv run --no-project --with pytest --with pillow --with grpcio --with grpcio-health-checking --with protobuf -- python -m pytest -q

lint:
	mkdir -p "$(UV_CACHE_DIR)"
	UV_CACHE_DIR="$(UV_CACHE_DIR)" uv run --no-project --with mypy -- mypy --strict domain service

ci: lint test
