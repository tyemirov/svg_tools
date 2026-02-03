.PHONY: test lint ci proto

UV_CACHE_DIR ?= $(CURDIR)/../.cache/uv
COVERAGE_CONFIG ?= $(CURDIR)/.coveragerc

test:
	UV_CACHE_DIR="$(UV_CACHE_DIR)" uv run --no-project --with pytest \
	    --with pillow --with grpcio --with grpcio-health-checking --with protobuf \
	    -- python -m pytest -q tests

lint:
	uv run --no-project --with mypy -- mypy --strict reel/domain reel/service

proto:
	python -m grpc_tools.protoc -I proto \
	    --python_out=grpc --grpc_python_out=grpc proto/audio_to_text.proto

ci: lint test
