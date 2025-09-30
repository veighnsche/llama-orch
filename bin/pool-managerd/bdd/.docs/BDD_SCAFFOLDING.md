# BDD Scaffolding for pool-managerd

## Overview

This document describes the BDD (Behavior-Driven Development) scaffolding structure for `pool-managerd`, following the established monorepo patterns used in `orchestratord`, `catalog-core`, and `orchestrator-core`.

## Directory Structure

```
bin/pool-managerd/bdd/
├── Cargo.toml              # BDD harness crate definition
├── README.md               # Crate documentation with spec traceability
├── src/
│   ├── main.rs            # BDD runner binary entrypoint
│   └── steps/
│       ├── mod.rs         # Step modules registry
│       └── world.rs       # Cucumber World state container
└── tests/
    └── features/          # Gherkin feature files (to be added)
        └── .gitkeep
```

## Key Components

### 1. Cargo.toml
- Package name: `pool-managerd-bdd`
- Binary target: `bdd-runner`
- Dependencies: cucumber 0.20, tokio, futures, pool-managerd parent crate
- Feature flag: `bdd-cucumber` (default)

### 2. main.rs (BDD Runner)
- Tokio multi-threaded runtime
- Respects `LLORCH_BDD_FEATURE_PATH` environment variable for targeted runs
- Default features directory: `tests/features/`
- Calls `BddWorld::cucumber().run_and_exit(features)`

### 3. World (steps/world.rs)
- Minimal starting point: `#[derive(Debug, Default, cucumber::World)]`
- Will be expanded with:
  - HTTP client state for pool-managerd API calls
  - Last response tracking (status, headers, body)
  - Test fixtures (engine configs, device masks, etc.)
  - Helper methods for common assertions

### 4. Step Definitions (steps/mod.rs)
- Currently empty, ready for step modules
- Pattern from orchestratord-bdd:
  - `preload.rs` - Preload and readiness lifecycle steps
  - `restart_backoff.rs` - Restart/backoff and guardrails steps
  - `device_masks.rs` - Device placement and affinity steps
  - `observability.rs` - Metrics and logging verification steps

## Usage

### Run all features
```bash
cargo run -p pool-managerd-bdd --bin bdd-runner
```

### Run specific feature
```bash
LLORCH_BDD_FEATURE_PATH=tests/features/preload.feature \
  cargo run -p pool-managerd-bdd --bin bdd-runner
```

### Run specific directory
```bash
LLORCH_BDD_FEATURE_PATH=tests/features/restart/ \
  cargo run -p pool-managerd-bdd --bin bdd-runner
```

## Spec Traceability

The BDD scenarios will cover requirements from `.specs/30-pool-managerd.md`:

- **OC-POOL-3001**: Preload at serve start, no Ready until success
- **OC-POOL-3002**: Preload fail-fast on insufficient VRAM/RAM
- **OC-POOL-3003**: Readiness endpoints reflect preload state
- **OC-POOL-3010**: Driver/CUDA errors trigger Unready + backoff-restart
- **OC-POOL-3011**: Restart storms bounded by exponential backoff
- **OC-POOL-3012**: CPU inference spillover disallowed
- **OC-POOL-3020**: Placement respects device masks
- **OC-POOL-3021**: Heterogeneous split ratios explicit and capped
- **OC-POOL-3030**: Emit preload outcomes, VRAM/RAM utilization, driver_reset events

## Next Steps

1. **Define World state** - Add fields for HTTP client, response tracking, test fixtures
2. **Create feature files** - Write Gherkin scenarios for each requirement group:
   - `preload_readiness.feature`
   - `restart_backoff.feature`
   - `device_masks.feature`
   - `observability.feature`
3. **Implement step definitions** - Create step modules with `#[given]`, `#[when]`, `#[then]` macros
4. **Add step registry** - Create `registry()` function for step validation tests (like orchestratord-bdd)
5. **Write validation test** - Add `tests/bdd.rs` to ensure no undefined/ambiguous steps

## Pattern Reference

This scaffolding follows the exact pattern from:
- `bin/orchestratord/bdd/` (most comprehensive example)
- `libs/catalog-core/bdd/` (minimal example)
- `libs/orchestrator-core/bdd/` (minimal example)

Key differences from orchestratord-bdd:
- Uses `BddWorld` instead of `World` (following catalog-core pattern)
- Starts minimal, will grow based on pool-managerd's specific needs
- No lib.rs (not needed for simple BDD harness)

## Integration with Workspace

The BDD crate is registered in the workspace `Cargo.toml`:
```toml
members = [
    ...
    "bin/pool-managerd",
    "bin/pool-managerd/bdd",
    ...
]
```

This allows:
- `cargo test -p pool-managerd-bdd` to run BDD tests
- `cargo run -p pool-managerd-bdd --bin bdd-runner` to execute scenarios
- Integration with `cargo xtask dev:loop` for continuous testing
