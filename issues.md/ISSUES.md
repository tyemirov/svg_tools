# ISSUES
**Section-based issues' log**

Entries in this file record newly discovered requests or changes, with their outcomes. No instructive content lives here. Read @issues.md/NOTES.md for the process to follow when fixing issues.

Read @AGENTS.md, @README.md and ARCHITECTURE.md and follow the links to documentation. Read @issues.md/POLICY.md, @issues.md/PLANNING.md, @issues.md/NOTES.md, and @issues.md/ISSUES.md. Start working on open issues. Prioritize bugfixes and maintenance. Work autonomously and stack up PRs.

Each issue is formatted as `- [ ] [<ID>-<number>]`. When resolved it becomes -` [x] [<ID>-<number>]`

## Features (100-199)

## Improvements (200–299)

- [x] [I100] Allow [text](../render_text_video.py) file to parse srt files with subtitles and timing, and distribute the words only through the timwindow allowed in subtitles. See example in [text](../data/inputs/captions.srt). Resolved with SRT parsing, timing-aware scheduling, and integration tests.

- [x] [I101] Add randomness to text genration, e.g. do not deterministically circle through the directions of movement of the words but have randon choice of how the words will move (only using predefined four directions of movement for now). Resolved with seeded random direction selection and integration tests.
- [x] [I102] Randomize per-word font sizes with a larger dynamic range based on screen size. Resolved with randomized sizing, JSON output, and tests.
- [x] [I103] Add CLI option to remove punctuation from rendered words. Resolved with tokenization updates and tests.
- [x] [I104] Allow static background images with derived dimensions instead of explicit width/height. Resolved with CLI validation, derived sizing, and tests.

## BugFixes (300–399)

- [x] [B300] Render vertical directions per-letter instead of moving whole words. Resolved with per-letter layout and staggered offsets.
- [x] [B301] Render letters only (no word-level motion), using per-letter bands for vertical and horizontal directions. Resolved with banded letter rendering and updated emit payload/tests.

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
- [B301] Priority: high. Goal: render letters only with banded positions so no word-level motion remains in any direction. Dependencies: per-letter metrics, band position computation. Docs: update `README.md` notes and emit payload. Plan: compute per-letter band positions for vertical/horizontal directions, render letters independently, extend emit JSON for bands, run `make ci`. Deliverable: letter-only motion in all directions with deterministic band output.
