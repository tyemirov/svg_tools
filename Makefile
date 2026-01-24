.PHONY: test lint ci

UV_CACHE_DIR ?= $(CURDIR)/.cache/uv
COVERAGE_CONFIG ?= $(CURDIR)/.coveragerc
COVERAGE_FAIL_UNDER ?= 67

test:
	mkdir -p "$(UV_CACHE_DIR)"
	UV_CACHE_DIR="$(UV_CACHE_DIR)" uv run --no-project --with pytest --with pillow --with grpcio --with grpcio-health-checking --with protobuf -- python -m pytest -q

lint:
	mkdir -p "$(UV_CACHE_DIR)"
	UV_CACHE_DIR="$(UV_CACHE_DIR)" uv run --no-project --with mypy -- mypy --strict domain service

ci: lint
	mkdir -p "$(UV_CACHE_DIR)"
	COVERAGE_PROCESS_START="$(COVERAGE_CONFIG)" PYTHONPATH="$(CURDIR)" UV_CACHE_DIR="$(UV_CACHE_DIR)" uv run --no-project --with pytest --with pytest-cov --with coverage --with pillow --with grpcio --with grpcio-health-checking --with protobuf -- \
		python -m pytest -q \
		--cov=audio_to_text \
		--cov=audio_to_text_grpc \
		--cov=audio_to_text_backend \
		--cov=render_text_video \
		--cov-report=term-missing:skip-covered \
		--cov-fail-under="$(COVERAGE_FAIL_UNDER)" \
		--cov-config="$(COVERAGE_CONFIG)"
