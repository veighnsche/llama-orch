# CukeRust BDD Wiring (cukerust_core)

This section documents how BDD is wired in this repository for the Rust core crate `cukerust_core`. It shows the exact glue (world, steps, entrypoint), how to run it, and how to troubleshoot. The Switchyard examples below are kept as additional reference patterns.

## Cargo wiring (cukerust_core)

`rust/crates/cukerust_core/Cargo.toml` defines a dedicated cucumber test target with an async `tokio` main:

```toml
[dev-dependencies]
cucumber = "0.21"
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }

[[test]]
name = "bdd"
path = "tests/bdd.rs"
harness = false
```

## Test entrypoint: `tests/bdd.rs`

The executable test wires the `CoreWorld`, registers the step modules, and launches cucumber. It supports overriding the features path with `CUKERUST_BDD_FEATURE_PATH` (absolute or relative to the crate root) and fails on skipped/undefined steps.

```rust
use cucumber::World;
use std::path::PathBuf;

#[derive(Debug, Default, World, Clone)]
pub struct CoreWorld {
    pub files: Vec<cukerust_core::step_index::SourceFile>,
    pub index: Option<cukerust_core::step_index::StepIndex>,
}

mod steps; // re-exports step modules under tests/steps/

#[tokio::main]
async fn main() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let features = match std::env::var("CUKERUST_BDD_FEATURE_PATH").ok() {
        Some(p) => {
            let pb = PathBuf::from(p);
            if pb.is_absolute() { pb } else { root.join(pb) }
        }
        None => root.join("features"),
    };
    CoreWorld::cucumber()
        .fail_on_skipped()
        .run_and_exit(features)
        .await;
}
```

## World and steps

- World: `CoreWorld` holds ephemeral `files` input and the extracted `index`.
- Steps: `rust/crates/cukerust_core/tests/steps/`
  - `mod.rs` re-exports step modules (e.g., `pub mod parsing;`)
  - `parsing.rs` contains `#[given]`, `#[when]`, `#[then]` step definitions that build inputs and assert on the Step Index.

## Feature files

Features live under `rust/crates/cukerust_core/features/` and describe parsing behaviors:

- `features/parsing/attributes.feature`
- `features/parsing/builder_chains.feature`
- `features/parsing/macros.feature`
- `features/parsing/raw_strings.feature`
- `features/parsing/ambiguity.feature`

## Running BDD locally (cukerust_core)

Run all features (verbose):

```bash
cd rust
cargo test -p cukerust_core --test bdd -- --nocapture
```

Run a single feature or directory (relative to crate root or absolute):

```bash
cd rust
CUKERUST_BDD_FEATURE_PATH=crates/cukerust_core/features/parsing \
  cargo test -p cukerust_core --test bdd -- --nocapture
```

## Troubleshooting (cukerust_core)

- Undefined or skipped steps fail the run due to `.fail_on_skipped()`; ensure your phrases/regex match and the module is re-exported by `tests/steps/mod.rs`.
- Attribute macros (`#[given]`, `#[when]`, `#[then]`) come from `cucumber` 0.21 and work with the current dev-dependencies.
- Set `RUST_LOG=debug` and add `println!` in steps when investigating parsing issues.

---

# Switchyard BDD Wiring and Troubleshooting Guide

This document explains how Behavior-Driven Development (BDD) is wired into the `switchyard` crate and how to run and troubleshoot it locally.

It complements the adapter-centric guide in `cargo/switchyard/SPEC/features/bdd_adapter_guide.md` by focusing on this repository’s wiring: Cargo features, test entrypoint, world/state, step modules, feature discovery, and helper scripts.

## Overview

- Features live under `cargo/switchyard/SPEC/features/` and are written in Gherkin (`.feature`).
- The BDD runner is a custom test target (`[[test]] bdd`) that uses `cucumber-rs` with an async `tokio` main.
- Steps are implemented in Rust under `cargo/switchyard/tests/steps/` and operate against the Switchyard API using a shared `World`.
- A small support layer in `cargo/switchyard/tests/bdd_support/` provides emitters, audit sinks, SafePath helpers, and env guards.
- A helper script `scripts/bdd_filter_results.py` runs BDD and optionally filters output to failure-only for faster iteration.

## Cargo wiring

`cargo/switchyard/Cargo.toml` configures a dedicated BDD test target and the `bdd` feature gate:

```toml
[features]
# default = ["bdd"]   # disabled by default; opt-in via --features bdd
default = []
prod = []
file-logging = []
bdd = []

[dev-dependencies]
async-trait = "0.1"
cucumber = { version = "0.20", features = ["macros"] }
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
# ...

[[test]]
name = "bdd"
path = "tests/bdd_main.rs"
harness = false
required-features = ["bdd"]
```

Implications:

- You must explicitly enable the `bdd` feature to build/run the BDD test.
- `harness = false` because `cucumber-rs` provides its own `async fn main()`.

## Test entrypoint: `tests/bdd_main.rs`

`tests/bdd_main.rs` is the executable test that wires world + steps and launches the cucumber engine:

- Imports the BDD support and world modules: `#[path = "bdd_support/mod.rs"] mod bdd_support;`, `#[path = "bdd_world/mod.rs"] mod bdd_world;`, and `mod steps;` (which exposes all step modules).
- For non-`bdd` builds, `fn main() {}` is a no-op, keeping compilation cheap.
- For `bdd` builds, runs an async tokio main and resolves which features to execute:
  - If `SWITCHYARD_BDD_FEATURE_PATH` is set, uses that path (absolute or relative to crate root) to select a single `.feature` or a directory of features.
  - Otherwise, defaults to `SPEC/features` under the crate root.
- Runs cucumber with `fail_on_skipped()`, which will fail the run if any step is undefined or marked skipped.

Snippet (see `cargo/switchyard/tests/bdd_main.rs`):

```rust
#[cfg(feature = "bdd")]
#[tokio::main(flavor = "multi_thread")]
async fn main() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let features_env = std::env::var("SWITCHYARD_BDD_FEATURE_PATH").ok();
    let features = if let Some(p) = features_env {
        let pb = std::path::PathBuf::from(p);
        if pb.is_absolute() { pb } else { root.join(pb) }
    } else {
        root.join("SPEC/features")
    };
    bdd_world::World::cucumber()
        .fail_on_skipped()
        .run_and_exit(features)
        .await;
}
```

## World and state: `tests/bdd_world/mod.rs`

The shared `World` struct holds scenario state and helpers. Key points:

- Provides a temporary root via `tempfile::TempDir` and helpers to ensure/derive paths under that root.
- Builds a `Switchyard` API instance through the builder (`Switchyard::builder(...)`) with test sinks (`CollectingEmitter`, `CollectingAudit`).
- Exposes convenience methods for common scenario setup and execution:
  - `ensure_root()`, `rebuild_api()`, `ensure_api()`, `clear_facts()`
  - `mk_symlink(link, dest)` to set up a symlink topology under the temp root
  - `build_single_swap(link, src)` to construct a minimal `Plan` from `PlanInput`
  - `ensure_plan_min()` to lazily create a single-action plan
  - `run_real()`, `run_both_modes()`, `apply_current_plan_commit()` to exercise DryRun/Commit flows
  - `run_dry_and_store()` to snapshot DryRun facts for later parity checks
- Supports optional smoke runner override (e.g., switching to a failing runner for negative tests).

See `cargo/switchyard/tests/bdd_world/mod.rs` for the complete implementation.

## Support utilities: `tests/bdd_support/`

- `bdd_support/mod.rs` defines:
  - `CollectingEmitter`: implements `switchyard::logging::FactsEmitter` and collects JSON `Value` events in-memory.
  - `CollectingAudit`: implements `switchyard::logging::AuditSink` and collects `(Level, String)` lines in-memory.
  - `util::under_root(root, p)`: maps human paths like `/usr/bin/ls` or `providerA/ls` into your temporary root.
  - `util::sp(root, p)`: produces a `SafePath` from a rooted path.
- `bdd_support/env.rs` defines `EnvGuard` to set environment variables with automatic restoration on drop, preventing cross-scenario leakage.
- `bdd_support/facts.rs` and `bdd_support/schema.rs` (if present) are used by steps to assert audit/facts shapes and schema compliance.

All BDD filesystem operations are hermetic under a temp root. Never touch real system paths in BDD.

## Step registration: `tests/steps/`

- `tests/steps/mod.rs` re-exports each step module (e.g., `apply_steps`, `plan_steps`, `feature_gaps`, etc.). This compile-time inclusion is sufficient for cucumber to discover macro-annotated steps.
- Individual step files use `cucumber` macros (`#[given]`, `#[when]`, `#[then]`) with either exact text or regex to bind Gherkin phrases to Rust functions that mutate or inspect the `World`.
- Step modules typically:
  - Ensure the world root/API exists.
  - Build or modify a `Plan`.
  - Execute `preflight`/`apply` with `ApplyMode::{DryRun, Commit}`.
  - Assert emitted facts or filesystem observations under the temp root.

Directory: `cargo/switchyard/tests/steps/`

## Feature files: `SPEC/features/`

The BDD runner targets `cargo/switchyard/SPEC/features/` by default. Examples include:

- `atomicity.feature`, `atomic_swap.feature`, `api_toctou.feature`
- `conservatism_modes.feature`, `conservatism_ci.feature`
- `locking.feature`, `locking_rescue.feature`, `thread_safety.feature`
- `observability.feature`, `observability_audit.feature`
- `rescue.feature`, `rollback.feature`, `filesystems_degraded.feature`

You can scope runs to a single file or directory using the `SWITCHYARD_BDD_FEATURE_PATH` environment variable or the helper script’s `--features` argument (see below).

## Running BDD locally

### Direct Cargo

- All features with verbose output:

```bash
cargo test -p switchyard-fs --features bdd --test bdd -- --nocapture
```

- A single feature or directory (path relative to crate root or absolute):

```bash
SWITCHYARD_BDD_FEATURE_PATH=SPEC/features/atomicity.feature \
  cargo test -p switchyard-fs --features bdd --test bdd -- --nocapture
```

### Helper script: failure-only filter

`scripts/bdd_filter_results.py` wraps the cargo invocation and can print only failing steps and the summary for faster iteration.

- Run everything, show all output:

```bash
./scripts/bdd_filter_results.py
```

- Run a subset and show failure-only lines:

```bash
./scripts/bdd_filter_results.py --features SPEC/features/locking.feature --fail-only
```

The script stores the full raw output at `target/bdd-lastrun.log` for later inspection.

## Common pitfalls and troubleshooting

- "The `bdd` test doesn’t run or doesn’t exist":
  - Ensure you are running the `switchyard` crate test target and included `--features bdd`.
  - Verify `[[test]] name = "bdd"` in `cargo/switchyard/Cargo.toml` and that `harness = false` is set.

- "No `.feature` files found or wrong feature path":
  - By default the runner scans `SPEC/features` under `CARGO_MANIFEST_DIR` (the `switchyard` crate root).
  - If using `SWITCHYARD_BDD_FEATURE_PATH`, confirm whether the path is absolute; if not, it’s resolved relative to the crate root.

- "Failing due to skipped/undefined steps":
  - The runner uses `.fail_on_skipped()`. Any missing step bindings will fail the run.
  - Confirm your phrases/regex match the `.feature` text; ensure your step module is exported by `tests/steps/mod.rs`.

- "State leaks across scenarios":
  - The `World` uses in-memory collectors and temp roots by default. Use `World::clear_facts()` and `EnvGuard` for env isolation. Keep any global state out of steps.

- "File system permission errors":
  - All operations must be under the temp root. Use `util::under_root()` and `util::sp()` to map paths, and never create files under `/usr`, `/etc`, etc. The CI has a hermetic tests guard that rejects absolute system paths in tests.

- "Async runtime/panic around tokio":
  - The BDD target defines its own async `main` with `#[tokio::main(flavor = "multi_thread")]`; do not add another `tokio::test` around the entrypoint.

- "Facts/Audit assertions missing":
  - Steps that verify facts must interact with the `CollectingEmitter`/`CollectingAudit` or use the optional file-based sinks behind the `file-logging` feature. Ensure you rebuild the API after policy or adapter changes via `World::rebuild_api()`.

- "Running in CI":
  - The standard GitHub CI workflow runs unit/integration tests and a separate golden-fixtures flow. The BDD cucumber runner is primarily for local/spec validation and is not always invoked in CI by default. You can add a job that runs `cargo test -p switchyard --features bdd --test bdd -- --nocapture` if required.
  - Updated package name: use `switchyard-fs` with `-p` when running from the repository root, e.g., `cargo test -p switchyard-fs --features bdd --test bdd -- --nocapture`.

## Extending BDD

1. Add or edit `.feature` files under `cargo/switchyard/SPEC/features/`.
2. Implement or modify step definitions in `cargo/switchyard/tests/steps/` using `#[given]`, `#[when]`, `#[then]` macros, operating on `bdd_world::World`.
3. If you need additional test sinks or helpers, extend `cargo/switchyard/tests/bdd_support/`.
4. Keep tests hermetic: construct `SafePath` via `bdd_support::util::sp` and use the temp root.
5. Run locally using Cargo or the helper script; prefer `--fail-only` during iteration.

## Related docs

- `cargo/switchyard/SPEC/features/bdd_adapter_guide.md` — How to connect feature steps to the Switchyard API with examples.
- `cargo/switchyard/tests/README.md` — Test layout and conventions across the crate.
- `.github/workflows/ci.yml` — CI gates and test jobs (golden fixtures, guardrails, unit tests).

## Quick start

```bash
# From repo root
# Run all BDD features with full output
cargo test -p switchyard-fs --features bdd --test bdd -- --nocapture

# Or iterate on a single feature and only show failures
./scripts/bdd_filter_results.py --features SPEC/features/atomicity.feature --fail-only

# Inspect the last full run output
sed -n '1,200p' target/bdd-lastrun.log
```

## Troubleshooting external harness errors (WorldInventory, attribute macros)

If you are wiring a separate BDD harness (for example under `test-harness/bdd/`) and you see errors like:

- `the trait bound world::World: WorldInventory is not satisfied`
- `expected struct, variant or union type, found inferred type` at `#[given]`/`#[when]`

then compare your harness against this checklist and the minimal skeleton below.

### Checklist

- __cucumber macros feature enabled__
  - In your harness `Cargo.toml`, ensure `cucumber = { version = "0.20", features = ["macros"] }`.
  - Without `features = ["macros"]`, the `#[given]`, `#[when]`, `#[then]` attributes won’t expand correctly and often yield confusing type errors.

- __World derives the right trait__
  - Your `World` type must derive `cucumber::World` (not just `WorldInit` in newer versions):
    - `#[derive(Default, cucumber::World)] pub struct World { ... }`
  - This derive provides the necessary impls that the step macros expect (including inventory registration).

- __Import the exact `World` type into each step module__
  - Inside `src/steps/*.rs`, import your `World` with its correct module path, for example:
    - `use crate::world::World;` (if your file is `src/world.rs`)
  - The step fn signatures must be: `pub async fn step_name(world: &mut World, ...)`.

- __Step macros and async signatures__
  - `use cucumber::{given, when, then};`
  - Functions must be `pub async fn`, not `fn`.
  - The first parameter must be `&mut World` (the same `World` type imported above), not an alias or different type.

- __Module wiring ensures steps are compiled__
  - In your harness crate root:
    - `mod world;`
    - `mod steps;` and in `src/steps/mod.rs` re-export your step files (`pub mod apply_steps;` etc.).
  - If a step file isn’t referenced by `steps/mod.rs`, it won’t be compiled; the macros then appear “missing”.

- __Entry point calls the cucumber runner with your World__
  - `World::cucumber().fail_on_skipped().run_and_exit(<features_path>).await;`
  - Use a `#[tokio::main(flavor = "multi_thread")]` async main for the harness binary.

- __Version alignment__
  - Use a consistent `cucumber` version across crates. Macro/type mismatches between versions can cause `WorldInventory`-related errors.

### Minimal working harness skeleton

Use this structure to validate your wiring (adjust module names to your layout):

```toml
# test-harness/bdd/Cargo.toml
[package]
name = "bdd-harness"
edition = "2021"

[dependencies]
# Access the library under test (path or version)
switchyard = { path = "../../cargo/switchyard" }

[dependencies.tokio]
version = "1"
features = ["macros", "rt-multi-thread"]

[dependencies.cucumber]
version = "0.20"
features = ["macros"]

[dependencies.async-trait]
version = "0.1"

[[bin]]
name = "bdd"
path = "src/main.rs"
```

```rust
// test-harness/bdd/src/world.rs
#[derive(Debug, Default, cucumber::World)]
pub struct World {
    // put your state here
}
```

```rust
// test-harness/bdd/src/steps/mod.rs
pub mod apply_steps;
```

```rust
// test-harness/bdd/src/steps/apply_steps.rs
use cucumber::{given, when, then};
use crate::world::World;

#[given(regex = r"^target filesystem is unsupported$")]
pub async fn target_fs_unsupported(_world: &mut World) {}

#[when(regex = r"^when apply runs$")]
pub async fn when_apply_runs(_world: &mut World) {}

#[then(regex = r"^then it should fail$")]
pub async fn then_it_fails(_world: &mut World) {}
```

```rust
// test-harness/bdd/src/main.rs
mod world;
mod steps;

use world::World;

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    let features = std::path::PathBuf::from("../../cargo/switchyard/SPEC/features");
    World::cucumber()
        .fail_on_skipped()
        .run_and_exit(features)
        .await;
}
```

If your harness matches the skeleton above, but you still encounter `WorldInventory` errors, re-check that:

- The `World` derive is imported from `cucumber` (not a different crate alias).
- The step modules compile (temporarily add a `#[test] fn compiles() {}` to the module to verify it’s included).
- The `World` type path used in steps exactly matches the module where it’s defined.
- You aren’t shadowing `world` with another module/crate name.

### Deep-dive tips

- __Macro backtraces__: on Nightly, use `RUSTFLAGS="-Z macro-backtrace"` for clearer errors.
- __Type checks__: temporarily change `world: &mut World` to `world: &mut crate::world::World` to ensure the path resolves.
- __Version lock__: pin `cucumber = "=0.20.x"` across both the harness and library workspace if needed to remove version drift.

---

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
- Each file starts with a `# Traceability:` comment listing requirement IDs covered (ORCH-____, HME-____).
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
