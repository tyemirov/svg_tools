# SVG Tools

A collection of utilities for working with images, video and SVGs. Each script is a self-contained executable
with a `uv` shebang and inline [PEP 723](https://peps.python.org/pep-0723/) metadata.

---

## Requirements

- Python ≥ 3.11
- [`uv`](https://docs.astral.sh/uv/) (installed and available in `$PATH`)

Each script declares its dependencies in a `# /// script` block. When you run the script,
`uv` will automatically install the correct dependencies in an isolated environment.

---

## Usage Pattern

1. Make the script executable:

```shell
   chmod +x script.py
```

2. Run it directly:

```shell
./script.py [arguments...]
```

There is no need to call `uv run` explicitly — the shebang handles it.

---

## Scripts

### `generate_color_gradient.py`

Generate a dithered gradient based on a single named color (darker shade → lighter shade)
in a configurable direction, and draw a centered rectangle of a specified color and ratio.

**Usage:**

```shell
./generate_color_gradient.py \
    --gradient-color <COLOR_NAME_OR_HEX> \  # base color when start/end not provided
    [--start-color <COLOR_NAME_OR_HEX>] \   # optional explicit gradient start
    [--end-color <COLOR_NAME_OR_HEX>] \     # optional explicit gradient end
    --center-color <COLOR_NAME_OR_HEX> \
    --width <PIXELS> \
    --height <PIXELS> \
    --rectangle-ratio <FLOAT> \
    --direction <horizontal|vertical|diag-down|diag-up> \
    --output-file <PATH>
```

Example:

```shell
./generate_color_gradient.py \
    --gradient-color turquoise \
    --start-color '#0062B8' \
    --end-color '#0293D7' \
    --center-color gold \
    --width 1600 \
    --height 900 \
    --rectangle-ratio 0.5 \
    --direction diag-down \
    --output-file turquoise_golden_16x9.png
```

---

### `image_to_data_uri.py`

Center-crop an image to a square, resize it to 32×32 or 64×64, and output a Base64-encoded PNG data URI.

**Usage:**

```shell
./image_to_data_uri.py <input_image> --size <32|64>
```

Prints the data URI to stdout.

---

### `image_to_silhouette.py`

Extract a filled silhouette (binary mask) from an input image (assumes a single dark object on a plain background).

**Usage:**

```shell
./image_to_silhouette.py \
    --input-file <input_image> \
    --output-file <output_mask.png> \
    [--threshold <0–255>] \
    [--kernel-size <integer>]
```

---

### `image_to_svg.py`

Convert a raster image (JPG/PNG/HEIF, etc.) to SVG using one of two algorithms:

* `contours` — edge detection + contour tracing → **stroked** paths (good for sketches/outlines).
* `flat` — CIELAB k-means color quantization → **filled** color regions with optional outline (good for logos/flat art).

HEIF/HEIC is supported via `pillow-heif`.

**Usage:**

```shell
./image_to_svg.py <input_image> \
  [--algorithm {contours,flat}] \
  [--output_suffix <SUFFIX>] \
  [--blur_kernel <ODD_INT>] \
  [--alpha_threshold <0-255>] \
  \
  # Contours mode
  [--edge_low <INT>] [--edge_high <INT>] \
  [--contour_min_area <FLOAT>] \
  [--contour_epsilon_factor <FLOAT>] \
  [--stroke_color <COLOR>] [--stroke_width <FLOAT>] \
  \
  # Flat mode
  [--color_count <INT>] [--region_min_area <FLOAT>] \
  [--region_epsilon_pixels <FLOAT>] [--palette_seed <INT>] \
  [--include_outline] \
  [--outline_low <INT>] [--outline_high <INT>] \
  [--outline_min_area <FLOAT>] \
  [--outline_width <FLOAT>] [--outline_color <COLOR>]
```

**Quick start:**

```shell
# 1) Outline-only (contours, default algorithm)
./image_to_svg.py assets/photo.jpg \
  --blur_kernel 7 --edge_low 100 --edge_high 250 \
  --contour_min_area 50 --contour_epsilon_factor 0.005 \
  --stroke_color black --stroke_width 1.0

# 2) Flat color regions with an outline
./image_to_svg.py assets/logo.png --algorithm flat \
  --color_count 6 --region_min_area 64 --region_epsilon_pixels 0.75 \
  --include_outline --outline_low 80 --outline_high 200 \
  --outline_min_area 48 --outline_width 1.5 --outline_color black
```

**Notes**

* `--blur_kernel` is made odd automatically; `0/1` disables blur.
* `--alpha_threshold` masks out near-transparent pixels for both algorithms.
* In `flat` mode, background is auto-detected from the image border and dropped from the palette.

**Parameter reference**

*Common*

| Flag                | Meaning                                     |
|---------------------|---------------------------------------------|
| `--algorithm`       | `contours` (default) or `flat`.             |
| `--output_suffix`   | Suffix for output SVG filename.             |
| `--blur_kernel`     | Gaussian blur kernel (odd; `0/1` disables). |
| `--alpha_threshold` | Visible-pixel alpha cutoff (0–255).         |

*Contours mode*

| Flag                               | Meaning                                         |
|------------------------------------|-------------------------------------------------|
| `--edge_low`, `--edge_high`        | Canny thresholds.                               |
| `--contour_min_area`               | Minimum area to keep a contour.                 |
| `--contour_epsilon_factor`         | `ε = factor × perimeter` (poly simplification). |
| `--stroke_color`, `--stroke_width` | Stroke styling for contour paths.               |

*Flat mode*

| Flag                                 | Meaning                                        |
|--------------------------------------|------------------------------------------------|
| `--color_count`                      | K for LAB k-means (background auto-dropped).   |
| `--region_min_area`                  | Minimum area for filled regions.               |
| `--region_epsilon_pixels`            | Absolute ε (pixels) for region simplification. |
| `--palette_seed`                     | Random seed for k-means init.                  |
| `--include_outline`                  | Add outline layer over filled regions.         |
| `--outline_low`, `--outline_high`    | Canny thresholds for the outline.              |
| `--outline_min_area`                 | Minimum area for outline contours.             |
| `--outline_width`, `--outline_color` | Outline stroke styling.                        |

---

### `render_text_video.py`

Render animated word-by-word text into a MOV (transparent output uses ProRes 4444).

Requires `ffmpeg` with `prores_ks` (alpha_bits), `yuva444p10le`, and `libx264` support.

**Usage:**

```shell
./render_text_video.py \
    [--input-text-file <PATH>] \
    --output-video-file <PATH.mov> \
    [--width <PIXELS>] \
    [--height <PIXELS>] \
    [--background-image <PATH>] \
    [--duration-seconds <FLOAT>] \
    [--fps <INT>] \
    [--background <transparent|#RRGGBB>] \
    [--fonts-dir <PATH>] \
    [--audio-track <PATH>] \
    [--direction-seed <INT>] \
    [--remove-punctuation] \
    [--keep-punctuation] \
    [--subtitle-renderer <motion|criss_cross|rsvp_orp>] \
    [--font-min <INT>] \
    [--font-max <INT>] \
    [--emit-directions]
```

**Notes**

* Input text must be valid UTF-8 and is split on whitespace.
* When `--input-text-file` is omitted, no text is rendered; the output is the background with optional audio.
* `.srt` and `.sbv` input files are parsed as subtitle windows; words render only inside each time range.
* Provide either `--background-image` or `--width`/`--height` (image derives dimensions).
* `--fonts-dir` should contain .ttf/.otf fonts (bold variants recommended).
* `--direction-seed` makes direction selection deterministic for a given seed.
* Punctuation is stripped by default; use `--keep-punctuation` to preserve punctuation.
* `--remove-punctuation` is accepted for compatibility but matches the default behavior.
* RSVP punctuation pauses only apply when punctuation is preserved.
* `--subtitle-renderer criss_cross` explicitly selects the randomized motion renderer (default behavior).
* `--subtitle-renderer rsvp_orp` enables RSVP/ORP subtitles from SRT/SBV input (single word at a time with ORP anchoring).
* RSVP mode requires subtitle timing and does not use motion directions or per-word random sizing; when a cue is too short, per-word timing is compressed and cues may drift slightly to fit.
* `--font-min`/`--font-max` constrain the randomized font size range for `criss_cross`; if only one bound is provided, the other bound is clamped to it.
* `--background` applies only when no background image is used.
* ProRes output uses adaptive quantization plus 8-bit alpha to reduce file sizes on large frames.
* Transparent output (no background image and `--background transparent`) uses ProRes 4444 with alpha; any opaque background uses H.264 without alpha for better compression.
* Opaque output uses yuv420p and requires even frame dimensions (odd sizes are rejected).
* `--audio-track` muxes audio and can supply duration; if no `--duration-seconds` is provided, duration is derived from the audio track and/or subtitle timing (longest wins).
* When `--duration-seconds` is provided with audio, the output duration is forced and audio is trimmed or padded.
* `--duration-seconds` is required when no audio track and no subtitle timing are provided.
* Font sizes are randomized per word within a dynamic range derived from frame size (large enough to overflow the frame).
* Letters render in per-letter bands aligned with the motion axis; band offsets are centered and spaced by glyph sizes with tracking, reversed for L2R/T2B so the first letter leads the motion, and vertical directions also add staggered offsets.
* `--emit-directions` prints JSON with `directions`, `font_sizes`, `words`, `letter_offsets`, `letter_bands`, and `letter_band_sizes` (band offsets centered on the motion axis), then exits without rendering.
* Output is always a `.mov` file; default name is `video.mov`.

---

### `audio_to_text.py`

Force-align audio or video to a provided transcript and emit an SRT with word-level timing.

**CLI usage:**

```shell
./audio_to_text.py \
    --input-audio <PATH> \
    --input-text <PATH> \
    [--output-srt <PATH.srt>] \
    [--language <CODE>]
```

**UI usage:**

```shell
./audio_to_text.py --ui [--ui-host <HOST>] [--ui-port <PORT>]
```

The UI provides separate dropzones for audio/video and transcript text, runs alignment in a background job, and offers a download link for the generated SRT.
audio_to_text is supported on Linux only; on macOS or Windows, run it via Docker.
Uploads are stored under `data/audio_to_text_uploads` and persisted via the `data/` bind mount in the Docker compose files.
Model downloads are cached under `data/hf-cache` on the host.
Torch/torchaudio checkpoints (e.g. wav2vec2 ASR weights) are cached under `data/torch-cache` on the host.

**Supported languages (alignment):** en, fr, de, es, it, ja, zh, nl, uk, pt, ar, cs, ru, pl, hu, fi, fa, el, tr, da, he, vi, ko, ur, te, hi, ca, ml, no, nn, sk, sl, hr, ro, eu, gl, ka.
**Runtime requirements:** torch >= 2.6 and torchaudio >= 2.6 (pinned for Linux) for AudioMetaData support and Hugging Face `.bin` models.

**Docker (Linux)**

Create the shared env file:

```shell
cp .env.audio_to_text.example .env.audio_to_text
```

Development (bind-mounts the repo for local changes):

```shell
docker compose -f docker/audio_to_text/docker-compose.yml up --build
```

Tests (Linux container):

```shell
docker compose -f docker/audio_to_text/docker-compose.yml run --rm --entrypoint make audio_to_text test
```

---

### `audio_to_text_grpc.py`

gRPC backend for forced alignment: client-stream a WAV + transcript and receive word-level timestamps (and SRT).
Set `AUDIO_TO_TEXT_GRPC_TEST_MODE=1` to run without invoking whisperx (deterministic fake alignment for integration tests).
Limits, auth, and TLS are configured via `.env.audio_to_text_grpc` (see the example for defaults).
Health checks are served via the standard gRPC health service; metrics are returned by `GetStats`.

**Docker (Linux)**

Create the shared env file:

```shell
cp .env.audio_to_text_grpc.example .env.audio_to_text_grpc
```

Development (bind-mounts the repo for local changes):

```shell
docker compose -f docker/audio_to_text_grpc/docker-compose.yml up --build
```

The service caches Hugging Face models under `data/hf-cache` and Torch/torchaudio checkpoints under `data/torch-cache` via bind mounts.

---

### `text_to_svg.py`

Render any text into a valid single-path SVG file.

**Usage:**

```shell
./text_to_svg.py "<Your Text Here>" \
    --font <path_to_font.ttf> \
    --out <output_filename.svg> \
    [--size <units_per_em>]
```

Example:

```shell
./text_to_svg.py "Marco Polo Research Lab" \
    --font assets/GreatVibes-Regular.ttf \
    --out title.svg
```

---

### `to_favicons.py`

Generate a complete favicon package from a single SVG source.

**Presets**

* **minimal (default)** — modern best-practice set:

    * `favicon.ico` (16, 32, 48, 64)
    * `favicon-32x32.png`
    * `apple-touch-icon.png` (180×180)
    * `android-chrome-192x192.png`
    * `android-chrome-512x512.png`
    * `safari-pinned-tab.svg`
    * `site.webmanifest`
    * optional `mstile-150x150.png` + `browserconfig.xml` when `--windows-tiles` is used

* **full** — everything in *minimal* plus historic sizes (16, 24, 48, 72, 96, 128, 144, 152, 167, 256, 384).

**Output layout**

```
<site-slug>/<assets_path>/...
```

* If `--site-name` is provided → folder is `<site-slug>/assets/favicons/…`.
* If omitted → folder is `site/assets/favicons/…`.
* Default `assets_path = "assets/favicons"` (configurable via `--assets-path`).

**Usage:**

```shell
# Minimal preset (default), no site name
./to_favicons.py --svg ./logo.svg

# Minimal preset with site name
./to_favicons.py --svg ./logo.svg --site-name "Marco Polo Research Lab"

# Full preset with Windows tiles
./to_favicons.py --svg ./logo.svg --preset full --windows-tiles

# Custom assets path
./to_favicons.py --svg assets/social_threader/favicon.svg --site-name "Social Threader" --background-color "0A1B3D" --theme-color "D4AF37"
```

The script also generates a `HEAD-snippet.html` file with `<link>` and `<meta>` tags ready to paste into your site’s `<head>`.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
