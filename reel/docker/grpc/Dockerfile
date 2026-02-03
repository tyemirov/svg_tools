FROM ghcr.io/astral-sh/uv:python3.12-bookworm AS builder

WORKDIR /app
COPY reel/__init__.py /app/reel/__init__.py
COPY reel/audio_to_text.py /app/reel/audio_to_text.py
COPY reel/grpc_server.py /app/reel/grpc_server.py
COPY reel/grpc /app/reel/grpc
RUN uv run --no-project /app/reel/audio_to_text.py --help >/dev/null \
  && uv run --no-project /app/reel/grpc_server.py --help >/dev/null

FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    patchelf \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /usr/local/bin/uv /usr/local/bin/uv
COPY --from=builder /root/.cache/uv /root/.cache/uv
COPY reel/__init__.py /app/reel/__init__.py
COPY reel/audio_to_text.py /app/reel/audio_to_text.py
COPY reel/grpc_server.py /app/reel/grpc_server.py
COPY reel/grpc /app/reel/grpc
COPY reel/docker/grpc/entrypoint.sh /app/entrypoint.sh

RUN chmod +x /app/reel/audio_to_text.py /app/reel/grpc_server.py /app/entrypoint.sh \
  && mkdir -p /opt/hf-cache /opt/torch-cache

ENTRYPOINT ["/app/entrypoint.sh"]
