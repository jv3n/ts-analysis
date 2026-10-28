---
name: code-review
description: Read-only review of the current diff against ts-analysis conventions and invariants, before the commit. Use when the user types `/code-review`, asks to "review the diff" or "check my changes", or after wrapping up an issue. Spawns the `code-reviewer` subagent, which returns a prioritised punch-list and never edits code.
---

# Code review

Spawn the **`code-reviewer`** subagent (`.claude/agents/code-reviewer.md`) with the `Agent` tool:

- `subagent_type: "code-reviewer"`, `description: "Pre-commit code review"`.
- `prompt`: the scope and the issue. Default:

```
Review the diff on the current branch (main = `master`).

Scope: uncommitted changes (`git diff HEAD`, plus untracked files from `git status`) and, off
`master`, the branch's commit (`git diff master...HEAD`). Every touched file, not a sample.
Issue: #<N> — <title>. Its acceptance list: <paste it>.

Output: the punch-list Bloquants / À discuter / Mineurs, each finding with a cited snippet and a
concrete suggestion, then a verdict mergeable | needs-fix | reject. No edits.
```

## Rules

- Don't review in the main thread: the point is a fresh reader who did not write the code.
- Relay the punch-list as is, then wait. The user picks what to patch.
- A green verdict is not a go to commit, push or open a PR: those stay the user's call.
- Don't turn findings into issues on your own.
