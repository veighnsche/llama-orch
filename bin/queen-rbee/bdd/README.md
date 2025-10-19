# queen-rbee-bdd

**Status:** 🚧 STUB (Created by TEAM-135)  
**Purpose:** BDD integration test harness for queen-rbee binary

## Overview

Behavior-Driven Development integration test harness for the queen-rbee binary.
Tests the binary as a whole, including CLI/daemon behavior and integration with other components.

## Running Tests

```bash
# Run all features
cargo run --bin bdd-runner

# Run specific feature
LLORCH_BDD_FEATURE_PATH=tests/features/example.feature cargo run --bin bdd-runner
```

## Structure

- `src/main.rs` - BDD runner entry point
- `src/steps/world.rs` - Shared test state (World)
- `src/steps/` - Step definitions
- `tests/features/` - Gherkin feature files

## Test Scope

Integration tests for queen-rbee:
- Binary startup and shutdown
- CLI command execution
- Daemon behavior
- API endpoints
- Inter-component communication

## Adding Tests

1. Create `.feature` file in `tests/features/`
2. Implement steps in `src/steps/`
3. Export step module from `src/steps/mod.rs`

## Status

- [ ] Feature files created
- [ ] Step definitions implemented
- [ ] Tests passing

**Created by TEAM-135**
