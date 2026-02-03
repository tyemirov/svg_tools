#!/bin/sh
set -e

UV_CACHE_DIR="${UV_CACHE_DIR:-/root/.cache/uv}"

uv run --no-project /app/grpc_server.py --help >/dev/null
find "$UV_CACHE_DIR" -type f -name 'libctranslate2*.so*' -exec patchelf --clear-execstack '{}' ';'

exec /app/grpc_server.py "$@"
