FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY render_text_video.py /app/render_text_video.py
COPY fonts /app/fonts

RUN chmod +x /app/render_text_video.py

ENV UV_CACHE_DIR=/opt/uv-cache
RUN mkdir -p /opt/uv-cache \
  && uv run --no-project /app/render_text_video.py --help >/dev/null

ENTRYPOINT ["/app/render_text_video.py"]
