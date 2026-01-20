# ISSUES
**Append-only section-based log**

Entries in this file record newly discovered requests or changes, with their outcomes. No instructive content lives here. Read @issues.md/NOTES.md for the process to follow when fixing issues.

Read @AGENTS.md, @README.md and ARCHITECTURE.md and follow the links to documentation. Read @issues.md/POLICY.md, @issues.md/PLANNING.md, @issues.md/NOTES.md, and @issues.md/ISSUES.md. Start working on open issues. Prioritize bugfixes and maintenance. Work autonomously and stack up PRs.

Each issue is formatted as `- [ ] [<ID>-<number>]`. When resolved it becomes -` [x] [<ID>-<number>]`

## Features (100-199)

## Improvements (200–299)

- [ ] [I100] Allow [text](../render_text_video.py) file to parse srt files with subtitles and timing, and distribute the words only through the timwindow allowed in subtitles. See example in [text](../data/inputs/captions.srt)

- [ ] [I101] Add randomness to text genration, e.g. do not deterministically circle through the directions of movement of the words but have randon choice of how the words will move (only using predefined four directions of movement for now)

## BugFixes (300–399)

## Maintenance (400–499)

### Recurring (close when done but do not remove)

- [ ] [M400] Cleanup:
    1. Review the completed issues and compare the code against the README.md and ARCHITECTURE.md files.
    2. Update the README.md and ARCHITECTURE.
    3. Clean up the completed issues.

- [ ] [M401] Polish:
    1. Review each open issue
    2. Add additional context: dependencies, documentation, execution plan, goal
    3. Add priroity and deliverable. Reaarange and renumber issues as needed.

- [ ] [M402] Review:
    1. Review the current codebase against the principles outlined in POLICY.md, AGENTS.GO.md, AGENTS.FRONTEND.md
    2. Prepare a report that highlights the areas of improvement
proceed
## Planning (500–599)
*do not implement yet*
