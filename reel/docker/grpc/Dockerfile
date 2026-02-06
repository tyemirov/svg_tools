FROM ghcr.io/astral-sh/uv:python3.12-bookworm AS builder

WORKDIR /app
COPY __init__.py /app/__init__.py
COPY audio_to_text.py /app/audio_to_text.py
COPY grpc_server.py /app/grpc_server.py
COPY audio_grpc /app/audio_grpc
RUN uv run --no-project /app/audio_to_text.py --help >/dev/null \
  && uv run --no-project /app/grpc_server.py --help >/dev/null

FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    patchelf \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /usr/local/bin/uv /usr/local/bin/uv
COPY --from=builder /root/.cache/uv /root/.cache/uv
COPY __init__.py /app/__init__.py
COPY audio_to_text.py /app/audio_to_text.py
COPY grpc_server.py /app/grpc_server.py
COPY audio_grpc /app/audio_grpc
COPY docker/grpc/entrypoint.sh /app/entrypoint.sh

RUN chmod +x /app/audio_to_text.py /app/grpc_server.py /app/entrypoint.sh \
  && mkdir -p /opt/hf-cache /opt/torch-cache

ENTRYPOINT ["/app/entrypoint.sh"]
