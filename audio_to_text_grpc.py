#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "grpcio>=1.76",
#   "protobuf>=4.25",
# ]
# ///

"""Entry point for the audio_to_text gRPC server."""

from __future__ import annotations

import sys

from audio_to_text_grpc.server import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
