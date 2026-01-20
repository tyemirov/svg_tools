# Notes

Operational playbook for working in this repository. Use it to coordinate planning, execution, and delivery. Code style, stack-specific rules, and tooling details remain in the AGENTS* documents; this file focuses purely on day-to-day process.

## Role

You are a staff level full stack engineer. Your task is to **re-evaluate and refactor the code repository** according to the coding, process and product standards already written in teh files provided by the repo.

## Authoritative References

- `AGENTS.md` + per-stack guides for coding standards.
- `POLICY.md` for validation/confident-programming rules.
- `AGENTS.GIT.md` for Git/GitHub workflow.
- `AGENTS.DOCKER.md` for container expectations.
- `PLANNING.md` for planning stage.
- `README.md`, `PRD.md`, and `ARCHITECTURE.md` for product context.

## Workflow Overview

1. Read `AGENTS.md` (plus relevant stack guides) before touching code.
2. Review the backlog in `ISSUES.md`; work sequentially through Features, BugFixes, Improvements, then Maintenance.
3. For the active issue, read `PLANNING.md` and create `PLAN.md` (ignored by git) with bullet steps. Keep it updated and delete/rewrite it for the next issue.For the active issue.
4. Create a new branch (per `AGENTS.GIT.md`) from the latest issue branch, not from `master`, so history stays linear.
5. Before writing code, describe the bug/feature via failing automated tests first. Run `make test` to watch them fail, then run `make lint` and any mandatory formatter targets defined for your stack in `AGENTS*` to establish the initial tooling baseline; if these fail before your changes, record the situation in `ISSUES.md`.
6. Implement the change, keeping to stack-specific standards. Limit edits to necessary files plus `ISSUES.md` (append-only log) and `CHANGELOG.md` (post-completion summary).
7. After implementing changes but before committing, re-run the full tooling suite for your stack—`make test`, `make lint`, `make ci` where present, and any mandatory formatter targets defined in `AGENTS*`. All must pass locally before opening a PR unless the work is explicitly documented as blocked.
8. Commit the work with a descriptive message, push with tracking (`git push -u origin <branch>` on first push), and open the PR via `gh pr create`.
9. Move immediately to the next issue, repeating the cycle until the backlog is empty.

## Testing & Tooling

- Use the `Makefile` targets (`make test`, `make lint`, `make ci`) instead of ad-hoc commands. `make test` runs the Playwright harness headless; `make lint` enforces lint rules; `make ci` mirrors GitHub Actions.
- Run stack-specific formatters as defined in the relevant `AGENTS*` guides (for example, `go fmt` for Go or any other formatter that is required for that stack); do not introduce new formatters or override stack policies.
- Add or update Playwright scenarios covering button → event → notification flows, cross-panel isolation, and other observable behavior. Tests are black-box and table-driven.
- Prefix every CLI command with `timeout -k <N>s -s SIGKILL <N>s <command>`. Pick `<N>` appropriate to the task (≤30s for individual commands/tests, ≤350s for the full suite). No exceptions.

## Git & Release Flow

- `master` is production. Branches use the taxonomy prefixes (`feature/`, `improvement/`, `bugfix/`, `maintenance/`, `blocked/`) outlined in `AGENTS.GIT.md`.
- Forbidden operations: `git push --force`, `git rebase`, `git cherry-pick`, history rewrites.
- If blocked after three careful attempts, push the work to `blocked/<issue-id>` and document the reason in `ISSUES.md` before moving on.
- All PRs are opened with `gh pr create` targeting the prior PR, if exists, or master if it's the beginning of work.. GitHub Actions CI (triggered automatically) is the authoritative validation gate for merges and releases.

## Output Requirements

- Always follow AGENTS* rules; do not restate them in PRs.
- Begin every implementation with an up-to-date `PLAN.md`.
- Do not touch `NOTES.md` during normal work; treat it as read-only guidance.
- `ISSUES.md` is append-only; mark items `[x]` with a concise resolution note once tests pass.
- `PLAN.md` must remain untracked. If it enters git history, remove it via `git filter-repo --path PLAN.md --invert-paths` before continuing.
- Summaries at the end of each issue should list changed files and any new/updated event contracts.

## Pre-Finish Checklist

1. `PLAN.md` reflects the final state for the active issue.
2. `ISSUES.md` entry is marked `[x]` with the resolution note.
3. The full tooling suite for the active stack has been run and is passing: at minimum, `make test`, `make lint`, and `make ci` succeed locally (subject to the timeout rule), and any mandatory formatter targets from `AGENTS*` have been applied.
4. Commit contains only intended changes and is pushed to the tracking branch on `origin`.
5. PR opened via `gh pr create`, referencing the issue ID.
6. Provide a short summary plus next steps in the CLI output before moving to the next issue.

## Action Items Reminder

- Read guiding docs (`README.md`, `PRD.md`, `AGENTS*`, `NOTES.md`, `ARCHITECTURE.md`) before planning.
- Keep working sequentially through the backlog—never parallelize issues.
- Add missing issues to `ISSUES.md` if you discover new work while investigating; plan and resolve them in order.
