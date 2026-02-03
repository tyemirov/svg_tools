FROM python:3.12-slim AS builder

WORKDIR /app
COPY audio_to_text_ui /app/audio_to_text_ui
COPY docker/audio_to_text_ui/entrypoint.sh /app/entrypoint.sh

RUN chmod +x /app/entrypoint.sh

FROM python:3.12-slim

WORKDIR /app
COPY --from=builder /app /app

ENTRYPOINT ["/app/entrypoint.sh"]
