---
name: github-create-pull-request
description: 'Open the pull request of the current issue: branch, push, PR body with Refs, board status. Use when the user asks to open a PR, or mentions "/pr" or "/pull-request".'
---

# Create a pull request

Repo `https://github.com/jv3n/ts-analysis`, base `master`. Only when the user asks in the current
turn. **Only the user merges**: never run `gh pr merge`.

## Workflow

1. **Branch**: `<issue>-<slug>` cut from an up-to-date `master`
   (`git switch master && git pull --ff-only && git switch -c 3-fetch-gap-bars`).
2. **One commit**: see the [`git-commit`](../git-commit/SKILL.md) skill.
   `git log --oneline master..HEAD` must show exactly one line.
3. **Push**: `git push -u origin "$(git branch --show-current)"`.
4. **Open it**. The title is the commit title. Show the body to the user before running:

   ```bash
   gh pr create --base master --title "<commit title>" --body "$(cat <<'BODY'
   Refs #3 — part of epic #1.

   - <what changed, one bullet per point that matters to the reader>

   ## To check

   - [ ] `uv run pytest`
   - [ ] <a real run, e.g. `uv run ts-fetch --start 2026-10-01 --end 2026-10-03`>
   BODY
   )"
   ```

   - `Refs #N`, **never `Closes` / `Fixes` / `Resolves`**: the user closes the issue.
   - English, no mention of Claude or AI, no « Generated with » line.
5. **Board**: move the issue to **In review** on the TS Analysis Board (project 3, owner `jv3n`),
   since opening the PR was asked:

   ```bash
   gh project item-list 3 --owner jv3n --format json  # find the item id of the issue
   gh project item-edit --project-id <id> --id <item> --field-id <Status> --single-select-option-id <In review>
   ```

6. Return the PR URL.

## After review

A fix is amended into the single commit and force-pushed with `--force-with-lease`, after
checking the PR is still open. Once the user has merged, suggest **Test** (still to try on real
data) or **Done** for the issue. Never move it yourself unless asked.
