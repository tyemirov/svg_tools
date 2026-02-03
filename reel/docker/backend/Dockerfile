FROM ghcr.io/astral-sh/uv:python3.12-bookworm AS builder

WORKDIR /app
COPY reel /app/reel
RUN uv run --no-project /app/reel/backend_server.py --help >/dev/null \
  && uv run --no-project /app/reel/grpc_server.py --help >/dev/null

FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    make \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /usr/local/bin/uv /usr/local/bin/uv
COPY --from=builder /root/.cache/uv /root/.cache/uv
COPY reel/__init__.py /app/reel/__init__.py
COPY reel/backend_server.py /app/reel/backend_server.py
COPY reel/backend /app/reel/backend
COPY reel/audio_to_text.py /app/reel/audio_to_text.py
COPY reel/grpc /app/reel/grpc

RUN chmod +x /app/reel/backend_server.py

ENTRYPOINT ["/app/reel/backend_server.py"]
