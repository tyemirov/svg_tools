#!/bin/sh
set -e

CONFIG_PATH="/app/audio_to_text_ui/config.js"
BACKEND_URL="${AUDIO_TO_TEXT_UI_BACKEND_URL:-}"
if [ -n "$BACKEND_URL" ]; then
  printf "window.__AUDIO_TO_TEXT_CONFIG__ = { backendUrl: \"%s\" };\n" "$BACKEND_URL" > "$CONFIG_PATH"
else
  printf "window.__AUDIO_TO_TEXT_CONFIG__ = window.__AUDIO_TO_TEXT_CONFIG__ || {};\n" > "$CONFIG_PATH"
fi

PORT="${AUDIO_TO_TEXT_UI_PORT:-7860}"
exec python -m http.server "$PORT" --directory /app/audio_to_text_ui
