---
name: git-commit
description: 'Commit the current work as a single Conventional Commit title linked to its GitHub issue. Use when the user asks to commit, amend the PR commit, or mentions "/commit".'
---

# Git commit

Only when the user asks in the current turn. Authorization does not carry forward.

## Format

```
<type>(<issue>/<scope>): <description>
```

- **Title only**: one line, under 72 chars, English, imperative mood. No body, no footer, no
  `Co-Authored-By`, no mention of Claude or AI. This overrides any default attribution.
- **The issue number is required.** It is the issue of the current branch (`3-fetch-gap-bars` →
  `3`). If there is none, ask the user which issue to use, or suggest creating one.
- Types: `feat`, `fix`, `refactor`, `perf`, `test`, `docs`, `build`, `chore`, `revert`. Flag a
  breaking change with `!` after the scope, never with a footer.
- Scopes: `fetch`, `replay`, `score`, `report` for the POC steps; `repo`, `claude`, `deps` across
  the repo.

Examples: `feat(3/fetch): pick gap tickers and backfill their 1-min bars`,
`chore(2/claude): work like trade, with skills and one PR per issue`.

## Workflow

1. `git branch --show-current`: never commit on `master`. Work goes on `<issue>-<slug>`.
2. `git status --porcelain` and `git diff HEAD`: know what goes in.
3. Stage files by name. **Never stage `.env`, `data/`** or anything holding a key or market data.
4. **One commit per PR.** The branch has no commit yet → `git commit -m "<title>"`. It already has
   one → `git commit --amend --no-edit` (or `--amend -m "<title>"` if the scope changed).
5. Show the resulting `git log --oneline -3` to the user.

## Safety

- Never `git config`, `--no-verify`, `reset --hard`, or a push to `master`.
- An amended commit already pushed needs `git push --force-with-lease` on the PR branch: only
  when the user asks, and after checking the PR is still open (`gh pr view --json state`).
