FROM ghcr.io/astral-sh/uv:python3.12-bookworm AS builder

WORKDIR /app
COPY audio_to_text.py /app/audio_to_text.py
COPY audio_to_text_grpc.py /app/audio_to_text_grpc.py
COPY audio_to_text_grpc /app/audio_to_text_grpc
RUN uv run --no-project /app/audio_to_text.py --help >/dev/null \
  && uv run --no-project /app/audio_to_text_grpc.py --help >/dev/null

FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /usr/local/bin/uv /usr/local/bin/uv
COPY --from=builder /root/.cache/uv /root/.cache/uv
COPY audio_to_text.py /app/audio_to_text.py
COPY audio_to_text_grpc.py /app/audio_to_text_grpc.py
COPY audio_to_text_grpc /app/audio_to_text_grpc

RUN chmod +x /app/audio_to_text.py /app/audio_to_text_grpc.py \
  && mkdir -p /opt/hf-cache /opt/torch-cache

ENTRYPOINT ["/app/audio_to_text_grpc.py"]

