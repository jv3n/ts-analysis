# DevOps

Nothing is deployed: the POC runs on the user's machine. What is left is the repo's own tooling,
the CI and the rules on `master`. GitHub and git fix where some of those files live, so they stay
there and this folder documents them.

| File | Holds |
|---|---|
| `.github/workflows/ci.yml` | the tests on every pull request and every push to `master` |
| `.githooks/pre-push` | refuses a plain push to `master`, lets a force-push through |

| Doc | |
|---|---|
| [`docs/github.md`](docs/github.md) | merge settings, `master`'s protection, the pre-push hook, the CI check |
