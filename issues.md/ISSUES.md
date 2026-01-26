# ISSUES
**Section-based issues' log**

Entries in this file record newly discovered requests or changes, with their outcomes. No instructive content lives here. Read @issues.md/NOTES.md for the process to follow when fixing issues.

Read @AGENTS.md, @README.md and ARCHITECTURE.md and follow the links to documentation. Read @issues.md/POLICY.md, @issues.md/PLANNING.md, @issues.md/NOTES.md, and @issues.md/ISSUES.md. Start working on open issues. Prioritize bugfixes and maintenance. Work autonomously and stack up PRs.

Each issue is formatted as `- [ ] [<ID>-<number>]`. When resolved it becomes -` [x] [<ID>-<number>]`

## Features (100-199)

- [ ] [F100] RSVP_ORP subtitle renderer (eye-stationary speed reading). Add a new subtitle rendering mode enabled by: `--subtitle-renderer rsvp_orp`. All parameters are derived from input video and text.
*Goal*
Render one word at a time at a fixed screen position, horizontally shifted so the word’s ORP (Optimal Recognition Point) letter is always centered on the same X pixel. Burn subtitles into the video using the existing text overlay system.
*Defaults (derived)*
anchor_x = 0.50 * frame_width
anchor_y = 0.80 * frame_height
font_size_px = clamp(round(frame_height * 0.060), 28, 96)
stroke_px = clamp(round(font_size_px * 0.10), 2, 10)
margin_px = round(font_size_px * 0.60)
Base text color: near-white; stroke: black
ORP highlight color: slightly brighter than base
*Input*
Reuse existing subtitle input (SRT).
Split each cue into word tokens; keep trailing punctuation attached (.,!?:;).
*Timing (per cue)*
Let cue_ms = end_ms - start_ms
For each word: weight = max(1, len(word_core))
Allocate durations proportional to weights
Clamp per word: 80ms <= duration <= 700ms
If token ends with punctuation, add +160ms pause by extending that word (shift following words within cue if needed; never overlap next cue)
*ORP index*
Let word_core exclude trailing punctuation.
If len(word_core) <= 1: orp = 0
Else: orp = max(0, floor(len(word_core) * 0.35) - 1)
*ORP anchoring*
Let:
prefix = word_core[:orp]
orp_char = word_core[orp]
Measure with existing font metrics:
prefix_w = text_width(prefix)
orp_w = text_width(orp_char)
*Compute:*
x = anchor_x - prefix_w - round(orp_w / 2)
Clamp x to [margin_px, frame_width - margin_px - text_width(full_token)]
y = anchor_y
*Rendering*
For each word event:
Draw full token at (x, y) in base style
Draw orp_char again at (x + prefix_w, y) in ORP highlight color
*Output constraints*
One word visible at a time
No timing overlaps
ORP X position stable within ±2 px across words
Burned-in subtitles; reuse existing overlay pipeline

- [x] [F101] Provide a WAV-only forced-alignment gRPC backend service with client-streaming uploads and a test-only mode. Returns word-level timestamps (and SRT) for a transcript forced-aligned to a streamed WAV; punctuation stripping defaults to enabled, language defaults via transcript heuristic. Resolved with a new gRPC server entrypoint, proto/stubs, and integration tests exercising test mode and input validation.

## Improvements (200–299)

- [x] [I100] Allow [text](../render_text_video.py) file to parse srt files with subtitles and timing, and distribute the words only through the timwindow allowed in subtitles. See example in [text](../data/inputs/captions.srt). Resolved with SRT parsing, timing-aware scheduling, and integration tests.

- [x] [I101] Add randomness to text genration, e.g. do not deterministically circle through the directions of movement of the words but have randon choice of how the words will move (only using predefined four directions of movement for now). Resolved with seeded random direction selection and integration tests.
- [x] [I102] Randomize per-word font sizes with a larger dynamic range based on screen size. Resolved with randomized sizing, JSON output, and tests.
- [x] [I103] Add CLI option to remove punctuation from rendered words. Resolved with tokenization updates and tests.
- [x] [I104] Allow static background images with derived dimensions instead of explicit width/height. Resolved with CLI validation, derived sizing, and tests.
- [x] [I105] Speed up render_text_video rendering for higher FPS/longer duration runs (target 5x). Resolved with pre-rendered letter sprites and faster compositing.
- [x] [I106] Switch render_text_video to a faster alpha-preserving codec by default (target 5x). Resolved with qtrle/argb output and updated ffmpeg checks.

## BugFixes (300–399)

- [x] [B300] Render vertical directions per-letter instead of moving whole words. Resolved with per-letter layout and staggered offsets.
- [x] [B301] Render letters only (no word-level motion), using per-letter bands for vertical and horizontal directions. Resolved with banded letter rendering and updated emit payload/tests.
- [x] [B302] Prevent letter band overlap by spacing bands using per-letter glyph sizes for both vertical and horizontal directions. Resolved with glyph-based band sizes, tracking spacing, and emit/test updates.
- [x] [B303] Ensure vertical band letters never share horizontal overlap and move independently. Resolved with band non-overlap test and centered glyph placement.
- [x] [B304] Align vertical/horizontal band orientation with motion direction so vertical words only move vertically. Resolved with centered motion-axis bands and updated tests/docs.
- [x] [B305] Ensure the first letter leads movement in every direction (no backwards reveal). Resolved with direction-aware band ordering and stagger offsets.
- [x] [B306] Ensure first-letter entry order for L2R/T2B directions. Resolved with entry-side band reversal and updated tests/docs.
- [x] [B307] Ensure entry-side band ordering for all directions matches first-letter entry. Resolved with direction→order map and updated tests/docs.
- [x] [B308] Add integration test confirming top-to-bottom words lead with the first letter (HARD example). Resolved with deterministic T2B band-order test.
- [x] [B309] Add integration test confirming HARD leads in all four directions. Resolved with deterministic seed coverage for each direction.
- [x] [B310] Fix left-to-right ordering so the first letter leads the motion (Cyrillic example "писать"). Resolved with L2R order flip and rendered-frame test.
- [x] [B311] Reduce alpha output size (20GB/100s) while keeping fast rendering and alpha. Resolved with ProRes 4444 tuned quality and lossy-safe render tests.
- [x] [B312] Ensure the first letter is the first visible glyph in rendered frames for all directions (HARD example shows later letters entering first). Resolved with B2T order correction, entry alignment, and render-based detection tests.
- [ ] [B313] Fix bottom-to-top vertical rendering order (natural order), missing letters, and garbled overlap in rendered frames. Pending band ordering and visibility alignment updates with render-verified tests.

## Maintenance (400–499)

### Recurring (close when done but do not remove)

- [x] [M400] Cleanup: added `ARCHITECTURE.md` and updated `README.md` for render_text_video docs.
    1. Review the completed issues and compare the code against the README.md and ARCHITECTURE.md files.
    2. Update the README.md and ARCHITECTURE.
    3. Clean up the completed issues.

- [x] [M401] Polish: issue context appended; renumbering skipped per append-only policy.
    1. Review each open issue
    2. Add additional context: dependencies, documentation, execution plan, goal
    3. Add priroity and deliverable. Reaarange and renumber issues as needed.

- [x] [M402] Review: report appended with policy/stack gaps and follow-up needs.
    1. Review the current codebase against the principles outlined in POLICY.md, AGENTS.GO.md, AGENTS.FRONTEND.md
    2. Prepare a report that highlights the areas of improvement
proceed
## Planning (500–599)
*do not implement yet*

## Report (M402)
- Missing `ARCHITECTURE.md` and `PRD.md` referenced in `issues.md/NOTES.md`; `README.md` does not document `render_text_video.py`.
- No `Makefile`, so required `make test`, `make lint`, and `make ci` targets cannot run.
- No integration test harness; CLI entrypoints lack black-box coverage required by `POLICY.md`.
- Multiple scripts violate `issues.md/AGENTS.PY.md`: missing module docstrings, missing type hints, inline comments, mutable dataclasses without validation, and `print` instead of `logging`.
- Errors are mostly `RuntimeError`/`SystemExit` without stable codes or boundary wrapping; several scripts hard-code external dependencies and randomness without injection.

## Issue Context (M401)
- [I100] Priority: medium. Goal: accept SRT input in `render_text_video.py` and allocate words only within subtitle time windows. Dependencies: sample `data/inputs/captions.srt`, ffmpeg availability, fonts in `fonts/`. Docs: update `README.md` usage for SRT. Plan: parse SRT blocks, map words to per-caption time windows, align frame allocation to caption timing, add black-box CLI test. Deliverable: SRT-aware rendering with tests and docs.
- [I101] Priority: low. Goal: randomize word movement direction using the existing four-direction set. Dependencies: injected randomness or seed to keep behavior testable. Docs: update `README.md` if CLI gains a seed option. Plan: add seeded RNG parameter, pick direction per word, add CLI test for deterministic seed. Deliverable: non-deterministic direction option with deterministic seed path.
- [M400] Priority: medium. Goal: reconcile docs with code and close out completed items. Dependencies: create missing `ARCHITECTURE.md` and review `README.md` script list. Docs: update `README.md` and new `ARCHITECTURE.md`. Plan: inventory scripts/flags, draft architecture overview, mark completed issues. Deliverable: updated docs and cleaned issue log.
- [M401] Priority: done. Goal: add context for open issues. Dependencies: none. Docs: `issues.md/ISSUES.md`. Plan: append context entries and note append-only constraint. Deliverable: issue context section added.
- Note: renumbering/reordering skipped to honor append-only policy in `AGENTS.md` and `issues.md/NOTES.md`.

## Tooling Baseline (M400-M402)
- `make test`, `make lint`, and `make ci` failed before changes because no `Makefile` targets exist yet.

## Tooling Baseline (I100)
- `make test` failed before changes because SRT window validation is not implemented yet.
- `make lint` failed before changes because `domain/` and `service/` packages do not exist yet.

## Maintenance Additions (400–499)
- [x] [M403] Add GitHub Actions CI to run `make ci` with uv and ffmpeg. Resolved with new workflow and local `make ci`.

## Issue Context Addendum (M401)
- [M403] Priority: medium. Goal: add GitHub Actions CI workflow running `make ci`. Dependencies: ubuntu runner with ffmpeg, uv install action. Docs: `README.md` optional. Plan: add workflow, run `make ci`, mark issue resolved. Deliverable: `.github/workflows/ci.yml` running on push/PR.
- [I102] Priority: medium. Goal: randomize per-word font sizes with larger dynamic min/max based on screen size. Dependencies: fonts in `assets/fonts`, CLI output for deterministic tests. Docs: update `README.md` notes. Plan: add size range selection, emit sizes in JSON output, update tests, run `make ci`. Deliverable: oversized randomized font sizes applied to rendering.
- [I103] Priority: medium. Goal: remove punctuation from words when requested. Dependencies: CLI flag, tokenization changes for plain text and SRT. Docs: update `README.md` notes. Plan: add `--remove-punctuation`, strip punctuation in tokenization, emit words for test validation, run `make ci`. Deliverable: optional punctuation-stripped rendering.
- [B300] Priority: high. Goal: animate letters individually for vertical directions (avoid whole-word floating). Dependencies: per-letter layout, draw pipeline updates. Docs: update `README.md` notes. Plan: render letters with staggered offsets for vertical directions, extend emit payload for tests, run `make ci`. Deliverable: per-letter vertical animation.
- [I104] Priority: medium. Goal: support static background images and derive dimensions from the image. Dependencies: PIL image loading, CLI validation for mutually exclusive width/height vs background image. Docs: update `README.md` usage/notes. Plan: add `--background-image`, adjust validation, composite background, update tests. Deliverable: background image support with derived dimensions.
- [I105] Priority: high. Goal: speed up render_text_video rendering (target 5x) to support higher FPS and longer durations. Dependencies: glyph layout data, ffmpeg output. Docs: update if behavior or options change. Plan: pre-render glyphs and reduce per-frame rasterization, run `make ci`. Deliverable: faster per-frame render without changing output.
- [I106] Priority: high. Goal: switch to a faster alpha-preserving codec by default to support higher FPS and longer durations. Dependencies: ffmpeg encoder/pixel format support. Docs: update README/ARCHITECTURE. Plan: update ffmpeg pipeline to use fast alpha codec, run `make ci`. Deliverable: faster alpha output without new CLI options.
- [B301] Priority: high. Goal: render letters only with banded positions so no word-level motion remains in any direction. Dependencies: per-letter metrics, band position computation. Docs: update `README.md` notes and emit payload. Plan: compute per-letter band positions for vertical/horizontal directions, render letters independently, extend emit JSON for bands, run `make ci`. Deliverable: letter-only motion in all directions with deterministic band output.
- [B302] Priority: high. Goal: ensure banded letter positions remain legible by spacing bands with per-letter glyph sizes and tracking. Dependencies: glyph metrics from fonts, emit payload updates for tests. Docs: update `README.md` notes/emit schema. Plan: compute band sizes per letter, derive band positions from sizes, update emit JSON and integration tests, run `make ci`. Deliverable: non-overlapping letter bands across directions.
- [B303] Priority: high. Goal: prevent vertical band letters from overlapping horizontally and ensure per-letter motion is visible. Dependencies: accurate glyph center alignment in band placement. Docs: update `README.md` if emit schema changes. Plan: add integration test for vertical band non-overlap, align glyph centers when computing positions, run `make ci`. Deliverable: vertical bands with non-overlapping letters.
- [B304] Priority: high. Goal: ensure vertical words only move vertically and horizontal words only move horizontally. Dependencies: band offsets aligned with motion axis and emit output updates. Docs: update `README.md` notes for band semantics. Plan: update band placement math and tests to center band offsets along motion axis, run `make ci`. Deliverable: direction-aligned band motion.
- [B305] Priority: high. Goal: ensure the first letter appears first regardless of movement direction. Dependencies: band ordering and stagger offsets aligned to direction. Docs: update `README.md` notes for letter-leading behavior. Plan: reverse band order for reverse directions and invert stagger offsets so first letter leads, add tests, run `make ci`. Deliverable: consistent first-letter-leading motion.
- [B306] Priority: high. Goal: ensure first-letter entry order for L2R/T2B so the leading letter appears first when entering the frame. Dependencies: band ordering tied to entry side. Docs: update `README.md` note. Plan: update band reversal set, adjust tests, run `make ci`. Deliverable: correct entry-side ordering.
- [B307] Priority: high. Goal: ensure entry-side band ordering across all directions so the first letter enters first. Dependencies: direction→letter-order map. Docs: update `README.md` note. Plan: add ordering map, update tests, run `make ci`. Deliverable: consistent entry-side ordering.
- [B308] Priority: medium. Goal: add integration test verifying top-to-bottom words lead with the first letter (HARD). Dependencies: direction seed with emit payload. Docs: none. Plan: add test, run `make ci`. Deliverable: coverage for T2B first-letter entry.
- [B309] Priority: medium. Goal: add integration test verifying HARD leads for L2R/R2L/T2B/B2T. Dependencies: direction seed with emit payload. Docs: none. Plan: add test, run `make ci`. Deliverable: coverage for all direction first-letter entry.
- [B310] Priority: high. Goal: ensure L2R ordering leads with the first letter (example: "писать" should display as reversed order while moving L2R). Dependencies: direction ordering map, rendering test. Docs: update `README.md` note if ordering changes. Plan: adjust ordering map, replace emit-based ordering tests with rendered-frame assertions, run `make ci`. Deliverable: correct L2R order and rendering-based coverage.
- [B311] Priority: high. Goal: reduce alpha output file size while keeping alpha and acceptable speed. Dependencies: ffmpeg encoder/pix_fmt support. Docs: update README/ARCHITECTURE. Plan: tune alpha codec/compression defaults, adjust render-based tests for lossy output if needed, run `make ci`. Deliverable: smaller alpha output by default.
- [x] [B313] Resolved B2T letter ordering/visibility issues with normalized band positions, entry alignment, and render-based tests for natural order and completeness.
- [x] [F100] Resolved RSVP ORP renderer with SRT-only mode, ORP anchoring, weighted timing + punctuation pauses, and render-verified integration tests.
- [ ] [I107] Allow optional font size bounds for criss_cross renderer via --font-min/--font-max and explicit --subtitle-renderer criss_cross selection.
- [x] [I107] Resolved criss_cross renderer selection with optional font-min/font-max bounds and integration tests.
- [ ] [I108] Make punctuation removal the default across renderers with an explicit keep-punctuation override.
- [x] [I108] Resolved default punctuation removal with keep-punctuation override across renderers.
- [ ] [B314] Allow RSVP windows longer than max per-word timing by leaving idle slack instead of failing.
- [x] [B314] Resolved RSVP long-window handling by capping per-word timing and allowing trailing idle frames, with integration coverage.
- [ ] [B315] Fix CI failures by tracking required fonts used by render_text_video tests.
- [x] [B315] Resolved CI font failures by tracking assets/fonts and updating .gitignore.
- [ ] [B316] Replace asset font tracking with test fixtures to keep assets uncommitted and CI green.
- [x] [B316] Resolved CI font fixture handling by moving fonts to tests/fixtures and restoring assets ignore.
- [ ] [B317] Reduce alpha MOV size further for long background renders (20GB/100s) with a smaller alpha payload.
- [x] [B317] Resolved alpha MOV size by scaling ProRes quantization with frame size and using 8-bit alpha, plus ffmpeg capability checks.
- [ ] [I109] Use non-alpha encoding when a background image or solid color is provided; reserve alpha output for transparent backgrounds.
- [x] [I109] Resolved opaque output by switching to H.264 (libx264) when alpha is not needed and validating the codec in integration tests.
- [ ] [I110] Allow `--font-max` alone to clamp the default minimum size for criss_cross rendering.
- [x] [I110] Resolved font-max clamping by aligning the computed minimum to the provided max and adding render_text_video integration coverage.
- [ ] [B318] Align horizontal letter rendering to a shared baseline to avoid wacky typography.
- [x] [B318] Resolved baseline alignment by anchoring per-letter bboxes to the baseline and positioning horizontal letters using the baseline in render_text_video.
- [ ] [B319] Add integration coverage for mixed-script baseline alignment (Latin + Cyrillic).
- [x] [B319] Resolved baseline alignment coverage with mixed-script render verification and test dependency updates.
- [ ] [B320] Validate even dimensions for opaque H.264 output to avoid runtime ffmpeg failures.
- [x] [B320] Resolved even-dimension validation with CLI checks and integration tests for odd sizes.

## Features Addendum (100-199)

- [x] [F101] Add audio track support in render_text_video with duration derived from audio/SRT (longest wins) and a duration override warning that trims/pads audio to the requested length. Resolved with audio muxing, ffprobe-derived duration, trim/pad warning, and integration tests.

## Improvements Addendum (200–299)

- [x] [I112] Allow render_text_video to merge a background (image or color) with audio only, without requiring input text. Resolved with background-only render path and integration coverage.
- [x] [I113] Add SBV subtitle support for render_text_video input text files. Resolved with SBV parsing, detection, docs, and integration tests.
- [x] [I114] Sanitize SRT input in audio_to_text by ignoring indices and timestamps before alignment. Resolved with SRT sanitization and integration coverage.
- [x] [I115] Relax RSVP subtitle timing to best-effort fit for short windows (compress timing and allow slight drift). Resolved with RSVP best-effort scheduling and integration coverage.
- [x] [I114] Add audio_to_text CLI with --input-audio/--input-text to produce forced-alignment SRT output. Resolved with uv CLI support, forced-alignment SRT emission, and integration tests.
- [ ] [I115] Add a web UI to audio_to_text with audio/text dropzones, background job runner, and SRT download.
- [x] [I115] Add a web UI to audio_to_text with audio/text dropzones, background job runner, and SRT download. Resolved with built-in UI server, background queue, and SRT download endpoint plus docs.
- [ ] [I116] Add alignment progress reporting to the audio_to_text UI so users can see generation status.
- [x] [I116] Add alignment progress reporting to the audio_to_text UI so users can see generation status. Resolved with progress updates in job tracking and UI progress bar rendering.
- [ ] [I117] Restrict audio_to_text language selection to supported alignment languages (dropdown only), remove custom model input, and avoid auto-detect.
- [x] [I117] Restrict audio_to_text language selection to supported alignment languages (dropdown only), remove custom model input, and avoid auto-detect. Resolved with CLI/UI language validation, dropdown options, and docs/tests updates.
- [ ] [I118] Add Linux-based Docker packaging for audio_to_text with dev/prod workflows and shared env file.
- [x] [I118] Add Linux-based Docker packaging for audio_to_text with dev/prod workflows and shared env file. Resolved with multi-stage Dockerfiles, compose samples, and env example/docs.

## Tooling Baseline (I114)
- `make test` failed before changes because `audio_to_text.py` is not executable and lacks the new CLI interface (PermissionError).

## BugFixes Addendum (300–399)

- [ ] [B324] Fix render_text_video RSVP windows that are shorter than one frame to avoid "subtitle window has no frames" errors.
- [x] [B324] Fix render_text_video RSVP windows that are shorter than one frame to avoid "subtitle window has no frames" errors. Resolved with non-empty window frame conversion and RSVP integration coverage.
- [ ] [B321] Fix audio_to_text UI template placeholder escaping to avoid KeyError for jobId.
- [x] [B321] Fix audio_to_text UI template placeholder escaping to avoid KeyError for jobId. Resolved with escaped template literals.
- [ ] [B322] Prevent audio_to_text UI DeprecationWarning noise from cgi and guard alignment against unsupported torch versions with a clear error.
- [x] [B322] Prevent audio_to_text UI DeprecationWarning noise from cgi and guard alignment against unsupported torch versions with a clear error. Resolved with cgi import warning suppression, torch version guard, and docs update.
- [ ] [B323] Fix audio_to_text HF alignment model load failures on Intel macOS by using safetensors-backed defaults (Russian) and conditional torch version checks.
- [x] [B323] Fix audio_to_text HF alignment model load failures on Intel macOS by using safetensors-backed defaults (Russian) and conditional torch version checks. Resolved with RU safetensors override, conditional torch>=2.6 enforcement, and docs update.
- [ ] [B325] Fix audio_to_text Docker build failure caused by torchaudio missing AudioMetaData during whisperx import.
- [x] [B325] Fix audio_to_text Docker build failure caused by torchaudio missing AudioMetaData during whisperx import. Resolved with lazy whisperx import and torchaudio AudioMetaData patching.
- [ ] [B326] Add fallback AudioMetaData to avoid torchaudio import failures in audio_to_text.
- [x] [B326] Add fallback AudioMetaData to avoid torchaudio import failures in audio_to_text. Resolved with a fallback NamedTuple and warning log.
- [ ] [B327] Remove torchaudio AudioMetaData fallback and enforce strict dependency versions.
- [x] [B327] Remove torchaudio AudioMetaData fallback and enforce strict dependency versions. Resolved with strict dependency pins and hard failure when AudioMetaData is missing.
- [ ] [B328] Suppress cgi DeprecationWarning emitted with stacklevel during UI uploads.
- [x] [B328] Suppress cgi DeprecationWarning emitted with stacklevel during UI uploads. Resolved with warning filtering by message.
- [ ] [B329] Remove deprecated cgi usage in audio_to_text UI uploads and replace with a non-deprecated multipart parser.
- [x] [B329] Remove deprecated cgi usage in audio_to_text UI uploads and replace with a non-deprecated multipart parser. Resolved with email-based multipart parsing and strict field validation.

## Improvements Addendum (200–299)

- [x] [I119] Restrict audio_to_text to a Linux-only Docker runtime, persist UI uploads under data/, and document container testing. Resolved with a Linux guardrail, data volume mapping, and README updates.
- [x] [I124] Allow deleting completed audio_to_text UI jobs. Resolved with a trash icon action, DELETE endpoint, persisted job store removal, and integration coverage for the API.
- [x] [I125] Allow deleting failed audio_to_text UI jobs. Resolved by widening deletion to finished jobs (completed/failed) and extending integration coverage.
- [x] [I126] Name audio_to_text SRT downloads after the input audio/video file. Resolved by using the input filename for the stored output path and Content-Disposition header.
- [x] [I127] Dockerize audio_to_text_grpc and document cache mounts for Hugging Face and Torch. Resolved with dedicated Dockerfiles/compose, a shared env example, and README updates.
- [x] [I128] Productionize audio_to_text_grpc with in-process alignment, health/status endpoints, limits, auth, and timeout enforcement. Resolved with in-process whisperx alignment, standard gRPC health, stats RPC, limits/auth/timeout enforcement, and expanded integration tests.

## Maintenance Addendum (400–499)

- [x] [M404] Remove docker-compose profiles for audio_to_text now that no published image exists. Resolved with a single compose service and updated README commands.
- [x] [M405] Remove the Russian alignment model override to rely on whisperx defaults. Resolved by dropping the override map and using whisperx model selection.
- [x] [M406] Bind-mount the Hugging Face cache to host storage for audio_to_text. Resolved by switching /opt/hf-cache to data/hf-cache and documenting the cache path.

## BugFixes Addendum (300–399)

- [x] [B330] Avoid whisperx import failures from ctranslate2 exec-stack requirements by importing alignment modules directly. Resolved with alignment-only imports that skip transcribe.

## Improvements Addendum (200–299)

- [x] [I120] Replace audio_to_text UI polling with SSE job updates. Resolved with EventSource streaming and a new SSE endpoint.
- [x] [I121] Add a smoother alignment progress signal for the audio_to_text UI. Resolved with time-based progress updates emitted during alignment.
- [x] [I122] Stack audio_to_text UI jobs with per-job downloads and embed input metadata in generated SRT files. Resolved with queued job tracking, SSE list updates, and SRT metadata headers.
- [x] [I123] Remove SRT metadata headers and persist UI job lists for audio_to_text. Resolved with job store persistence and job list streaming without SRT annotations.

## BugFixes Addendum (300–399)

- [x] [B331] Fix audio_to_text alignment failures when whisperx emits punctuation tokens (e.g., em dash) without timestamps. Resolved by merging punctuation into neighboring word cues, capturing Python warnings via logging, and adding integration coverage via `--input-alignment-json`.
- [x] [B332] Fix audio_to_text alignment failures when whisperx emits non-punctuation tokens without timestamps (e.g. single letters or full words). Resolved by inferring token timings from segment bounds, keeping punctuation-merging behavior, and extending integration coverage via `--input-alignment-json`.
- [ ] [B333] Harden audio_to_text alignment extraction when segments or tokens lack valid timestamps by adding fallback bounds and merging unaligned tokens instead of failing.
- [x] [B333] Harden audio_to_text alignment extraction when segments or tokens lack valid timestamps by adding fallback bounds, coercing invalid timestamps, and extending integration coverage for missing/non-finite cases.
- [ ] [I129] Split audio_to_text into a standalone UI, an HTTP backend orchestrator (REST + SSE + ffmpeg extraction), and the gRPC aligner, with a 3-service Docker Compose stack.
- [ ] [I130] Make backend alignment jobs asynchronous/decoupled from gRPC calls to allow queueing and retries.
- Tooling baseline (I129): `make test` fails because `audio_to_text_backend` server entrypoint is not implemented yet (backend tests time out).
- [x] [I129] Split audio_to_text into a standalone UI, HTTP backend orchestrator, and gRPC aligner with a 3-service Docker Compose stack. Resolved with the new backend package, standalone UI assets, stack compose/env updates, and integration coverage for backend job flow/SSE.
- [ ] [I131] Inject a test alignment runner into audio_to_text_grpc so the gRPC service always uses a single alignment pipeline with an injected runner.
- [x] [I131] Inject a test alignment runner into audio_to_text_grpc so the gRPC service always uses a single alignment pipeline with an injected runner. Resolved with alignment runner injection and startup selection for test mode.
- [ ] [I132] Wire pytest-cov into make ci with subprocess coverage capture and a coverage gate.
- [x] [I132] Wire pytest-cov into make ci with subprocess coverage capture and a coverage gate. Resolved with coverage config, subprocess hook, Python-based CLI invocation in tests, and CI coverage reporting/gating.
- [ ] [I133] Raise CI coverage gate to 100% by expanding integration coverage for audio_to_text, audio_to_text_backend, and render_text_video.
- [x] [I133] Raise CI coverage gate to 100% by expanding integration coverage for audio_to_text, audio_to_text_backend, and render_text_video. Resolved with expanded integration coverage, refreshed TLS fixtures, and a 100% coverage gate in `make ci`.
- [ ] [B334] Fix Docker dev stack failing to start when audio_to_text entrypoints lack executable bits.
- [x] [B334] Fix Docker dev stack failing to start when audio_to_text entrypoints lack executable bits. Resolved by marking gRPC/backend scripts executable for volume-mounted dev runs.
- [ ] [B335] Fix Docker dev stack failing when TLS env vars are set to empty values in `.env.audio_to_text_grpc`.
- [x] [B335] Fix Docker dev stack failing when TLS env vars are set to empty values in `.env.audio_to_text_grpc`. Resolved by removing empty TLS entries from the example env file.
- [ ] [B336] Consolidate Docker Compose files into a single file with profiles for stack/grpc/legacy usage.
- [x] [B336] Consolidate Docker Compose files into a single file with profiles for stack/grpc/legacy usage. Resolved with a unified compose file and updated docs.
- [ ] [B337] Fix UI SSE failures on non-localhost clients by defaulting backend URL to the current host.
- [x] [B337] Fix UI SSE failures on non-localhost clients by defaulting backend URL to the current host. Resolved with dynamic backend URL resolution and updated docs/env defaults.
- [x] [M407] Remove legacy audio_to_text stack artifacts and compatibility flags. Resolved by dropping the legacy docker service/env files, removing compatibility docs, and simplifying the render_text_video punctuation flag to a single keep override.
- [x] [B338] Surface alignment failure details in gRPC logs and classify missing-timestamp alignment errors from whisperx with `audio_to_text.align.missing_timestamps`.
- [x] [B339] Allow SSE access from non-localhost UI by default and assert SSE CORS headers in integration tests.
- [x] [B340] Serve the UI via the gHTTP GHCR image in Docker Compose and update the docs to drop local UI builds.
- [x] [B341] Ignore `data/` in git to avoid untracked artifact noise.
- [x] [B342] Fix ctranslate2 exec-stack import failures in the gRPC Docker images by clearing the executable stack requirement during build.
- [ ] [B343] Fix gRPC runtime ctranslate2 exec-stack import failures by clearing the executable stack requirement after uv env creation.
- [x] [B343] Fix gRPC runtime ctranslate2 exec-stack import failures by clearing the executable stack requirement after uv env creation. Resolved with a runtime entrypoint patch that clears execstack and Dockerfile updates to install patchelf and use the entrypoint.
- [ ] [B344] Fix Alpine job list rendering crash when job keys change during optimistic updates.
- [x] [B344] Fix Alpine job list rendering crash when job keys change during optimistic updates. Resolved with stable UI job ids and payload validation in the UI list renderer.
- [ ] [B345] Fix audio_to_text SSE stream stability so UI job status updates keep flowing.
- [x] [B345] Fix audio_to_text SSE stream stability so UI job status updates keep flowing. Resolved with HTTP/1.1 SSE responses, keepalive data events, reconnect logic, and updated integration coverage.
- [ ] [B346] Fix duplicate UI job cards by correlating optimistic jobs with backend updates.
- [x] [B346] Fix duplicate UI job cards by correlating optimistic jobs with backend updates. Resolved with client job ids, UI reconciliation updates, and SSE integration coverage.
