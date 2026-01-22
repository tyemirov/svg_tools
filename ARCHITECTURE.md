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
- `uv.lock`: pinned dependency resolution for `uv`.

## Script roles
- `generate_color_gradient.py`: builds a dithered gradient image with a centered rectangle.
- `image_to_data_uri.py`: crops and resizes a raster image, then emits a PNG data URI.
- `image_to_silhouette.py`: extracts a filled silhouette mask from a raster image.
- `image_to_svg.py`: converts raster images to SVG via contour tracing or flat color regions.
- `render_text_video.py`: renders animated word-by-word text into a ProRes MOV using ffmpeg.
- `text_to_svg.py`: converts text into a single-path SVG using a font file.
- `to_favicons.py`: generates a favicon package and HTML head snippet from a source SVG.

## External dependencies
- Python 3.11+ and `uv` on PATH.
- `ffmpeg` with `prores_ks` and `yuva444p10le` support for `render_text_video.py`.
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
