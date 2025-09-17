# BDD Harness Wiring (Home Profile)

This guide explains how the Behavior-Driven Development (BDD) harness in `test-harness/bdd` is structured, how to run it, and where to plug in new steps or scenarios.

## Layout

```
test-harness/bdd/
├── Cargo.toml
├── src/
│   ├── lib.rs              # exposes steps::world for cucumber macros
│   ├── main.rs             # executable used by `cargo test -p test-harness-bdd`
│   └── steps/
│       ├── mod.rs          # registers all step modules and inventories them
│       ├── world.rs        # World struct (shared scenario state)
│       └── *.rs            # step modules grouped by domain (data_plane, catalog, ...)
├── tests/
│   ├── bdd.rs              # hosts the cucumber runner when built as a test
│   ├── features/           # Gherkin `.feature` files (home-profile journeys)
│   └── traceability.rs     # ensures every requirement ID is referenced by at least one feature
└── scripts/
    └── (helper scripts, optional)
```

### World (`steps/world.rs`)
- Holds per-scenario state (temporary directories, HTTP clients, CLI handles, etc.).
- Provides helpers for creating sessions, submitting tasks, manipulating catalog/artifacts, and capturing responses.
- Implements `cucumber::World` so scenarios share a consistent context.

### Step Modules (`steps/*.rs`)
- Each module contains `#[given]`, `#[when]`, and `#[then]` functions bound to Gherkin phrases.
- Modules are grouped by topic: `data_plane.rs`, `catalog.rs`, `pool_manager.rs`, `scheduling.rs`, etc.
- `steps::registry()` (referenced in `main.rs`) ensures all modules are linked so cucumber’s inventory can discover them.

### Feature Files (`tests/features/`)
- Organised by domain (e.g., `data_plane/`, `control_plane/`, `artifacts/`, `placement/`).
- Each file starts with a `# Traceability:` comment listing requirement IDs covered (ORCH-****, HME-****).
- Add new scenarios here when specs or requirements change.

## Running the Harness

### Entire suite
```bash
cargo test -p test-harness-bdd --test bdd -- --nocapture
```

### Targeted run
Set `LLORCH_BDD_FEATURE_PATH` to a feature file or directory (absolute path or relative to `test-harness/bdd`).
```bash
LLORCH_BDD_FEATURE_PATH=tests/features/data_plane/admission_queue.feature \
  cargo test -p test-harness-bdd --test bdd -- --nocapture
```

### Notes
- The runner fails on undefined or skipped steps (`fail_on_skipped()`), so new phrases must be implemented before the suite will pass.
- Features operate against the product through HTTP/CLI layers; they do not mutate internal state directly.

## Adding Steps or Features
1. Update the relevant spec and requirement files first (`.specs/**`, `requirements/*.yaml`).
2. Create or modify the `.feature` file under `tests/features/` and annotate it with requirement IDs.
3. Implement supporting steps in `src/steps/*.rs`. Re-export the new module from `steps/mod.rs`.
4. Run the harness with `LLORCH_BDD_FEATURE_PATH` pointing to your new feature and verify it passes.
5. Update `.docs/testing/spec-derived-test-catalog.md` to reflect the new coverage.

## Troubleshooting
- Use `--nocapture` (default in examples) to view step-by-step output.
- Inspect generated logs/artifacts saved by the World (usually under a temporary directory printed in failure output).
- When feature discovery fails, ensure the path set in `LLORCH_BDD_FEATURE_PATH` is correct and that `steps/mod.rs` exports the module defining your new steps.

This harness is the primary executable proof of the home profile behaviour—keep it green and in sync with the specs.
