# llama-orch (pre-code) — Developer Quickstart

This repository contains the pre-code scaffolding for an LLM Orchestrator. Use the commands below to regenerate contracts, validate specs, and run tests.

## Quickstart

- Format check

```bash
cargo fmt --all -- --check
```

- Lints (warnings are errors)

```bash
cargo clippy --all-targets --all-features -- -D warnings
```

- Regenerate contracts and requirements

```bash
cargo xtask regen-openapi
cargo xtask regen-schema
cargo run -p tools-spec-extract --quiet
```

- Tests (workspace)

```bash
cargo test --workspace --all-features -- --nocapture
```

- Provider verification tests

```bash
cargo test -p orchestratord --test provider_verify -- --nocapture
```

- Consumer/CDC tests

```bash
cargo test -p cli-consumer-tests -- --nocapture
```

- Trybuild UI tests (compile-time checks)

```bash
cargo test -p tools-openapi-client -- --nocapture
```

- BDD harness (placeholder, ensures 0 undefined/ambiguous steps)

```bash
cargo test -p test-harness-bdd -- --nocapture
```

- Docs link checker

```bash
bash ci/scripts/check_links.sh
```
