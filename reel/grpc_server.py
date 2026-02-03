#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "grpcio>=1.76",
#   "grpcio-health-checking>=1.76",
#   "protobuf>=4.25",
#   "whisperx==3.3.0",
#   "matplotlib<4",
#   "numpy<2",
#   "safetensors",
#   "torch>=2.6,<2.7; platform_system == 'Linux'",
#   "torchaudio>=2.6,<2.7; platform_system == 'Linux'",
# ]
# ///

"""Entry point for the audio_to_text gRPC server."""

from __future__ import annotations

import sys

from reel.grpc.server import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
