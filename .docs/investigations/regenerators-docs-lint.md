# Investigation — Idempotent Regenerators & Docs Lint

Status: done · Date: 2025-09-15

## Status

- OpenAPI/Schema regenerators diff-clean on second run.
- Link checker and spec linter green.

## Proofs

- `cargo xtask regen-openapi && cargo xtask regen-schema && git diff --exit-code`
- `bash ci/scripts/check_links.sh && bash ci/scripts/spec_lint.sh`
