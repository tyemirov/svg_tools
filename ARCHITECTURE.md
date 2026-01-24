# Architecture

## Overview
SVG Tools is a collection of standalone CLI scripts for image, video, and SVG processing.
Each script is directly executable via a `uv` shebang and defines its own dependencies
using inline PEP 723 metadata.

## Execution model
- Each script is a self-contained entrypoint invoked as `./script.py [args]`.
- `uv` resolves and installs per-script dependencies from the inline metadata block.
- Inputs and outputs are file-based (images, SVGs, videos) or stdout (data URIs).
- CLI argument parsing is the primary validation edge for each script.

## Repository layout
- `*.py`: top-level script entrypoints.
- `assets/`: example assets used in README examples.
- `data/inputs/`: sample input files for CLI runs.
- `audio_to_text_backend/`: HTTP backend service package for alignment jobs.
- `audio_to_text_ui/`: standalone browser UI assets for alignment.
- `uv.lock`: pinned dependency resolution for `uv`.

## Script roles
- `generate_color_gradient.py`: builds a dithered gradient image with a centered rectangle.
- `image_to_data_uri.py`: crops and resizes a raster image, then emits a PNG data URI.
- `image_to_silhouette.py`: extracts a filled silhouette mask from a raster image.
- `image_to_svg.py`: converts raster images to SVG via contour tracing or flat color regions.
- `render_text_video.py`: renders animated word-by-word text into a MOV (ProRes for alpha, H.264 for opaque), with optional audio track muxing and a background-only mode when no text input is supplied.
- `text_to_svg.py`: converts text into a single-path SVG using a font file.
- `to_favicons.py`: generates a favicon package and HTML head snippet from a source SVG.
- `audio_to_text_grpc.py`: gRPC backend for forced-aligning a transcript to a streamed WAV and returning word-level timings (and SRT).
- `audio_to_text_backend.py`: HTTP backend orchestrator for alignment jobs, calling the gRPC aligner and persisting job state.

## Services
- `audio_to_text_ui/`: standalone browser UI that uploads audio/text and listens for REST/SSE job updates.
- `audio_to_text_backend.py`: HTTP REST/SSE service that extracts audio, calls the gRPC aligner, and stores job artifacts.
- `audio_to_text_grpc.py`: gRPC aligner service that accepts a streamed WAV + transcript and returns word timings.

## External dependencies
- Python 3.11+ and `uv` on PATH.
- `ffmpeg` with `prores_ks` (alpha_bits), `yuva444p10le`, and `libx264` support for `render_text_video.py`.
- `ffprobe` (from ffmpeg) for audio duration when `--audio-track` is used.
- Fonts (TTF/OTF) provided via `--fonts-dir` for `render_text_video.py`.

## Common flow
Each script follows a similar flow:
1. Parse CLI arguments.
2. Load and validate inputs at the boundary.
3. Perform image/SVG/video processing.
4. Write output files or emit stdout.

## Constraints
- Scripts do not share a common package; reuse is via duplication or small helpers.
- Outputs are deterministic unless randomness is explicitly introduced by a script.
