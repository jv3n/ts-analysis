# GitHub

Same settings as [`trade`](https://github.com/jv3n/trade), plus a protected `master`.

## Merging

- **Rebase only**: merge commits and squash are disabled, so `master` stays linear.
- **Branch deleted on merge**, on GitHub. The local copy is cleaned by hand:

  ```bash
  git checkout master && git pull && git fetch --prune && git branch -D <branch>
  ```

## `master`

| Rule | |
|---|---|
| Pull request required | 0 approval, the repo has one maintainer |
| Linear history | matches the rebase-only merges |
| Deletion | refused |
| Force-push | allowed |
| Admins | bypass the rules |

The owner can rewrite `master` (a force-push), anyone else goes through a pull request. GitHub
cannot let the same person force-push and refuse their plain push: that is the hook's job.

## Pre-push hook

`.githooks/pre-push` refuses a push to `master` that fast-forwards it, and lets a push that
rewrites it through. Enable it once per clone:

```bash
git config core.hooksPath .githooks
```

It only sees whether the history is rewritten, not `--force`: a forced push that rewrites
nothing is refused too. `git push --no-verify` skips it.

## CI

`.github/workflows/ci.yml` runs `uv sync --locked` (fails when `uv.lock` is out of date with
`pyproject.toml`) then `uv run pytest`. The tests make no network call: no secret to configure.

Its `tests` check is to be made required on `master` once the workflow is merged.
