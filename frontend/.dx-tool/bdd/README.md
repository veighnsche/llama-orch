# DX CLI BDD Tests

**Created by:** TEAM-DX-003  
**Purpose:** Integration testing for the `dx` CLI tool using Cucumber BDD

## Overview

This BDD test harness validates the `dx` CLI tool against a running Storybook instance on port 6006. It tests:

- **Story File Locator** (Priority 0) - Locate story files from Storybook URLs
- **CSS Commands** - Class existence, selector styles, class listing
- **HTML Commands** - DOM queries, attributes, tree visualization

## Prerequisites

### 1. Start Storybook

```bash
cd frontend/libs/storybook
pnpm story:dev
# Wait for: "Local: http://localhost:6006/"
```

### 2. Verify Storybook is Running

```bash
curl -s http://localhost:6006 | head -n 5
```

## Running Tests

### Run All Tests

```bash
cd frontend/.dx-tool/bdd
cargo test
```

### Run Specific Feature

```bash
# Story locator (Priority 0)
DX_BDD_FEATURE_PATH=tests/features/story/story_locator.feature cargo test

# CSS tests
DX_BDD_FEATURE_PATH=tests/features/css cargo test

# HTML tests
DX_BDD_FEATURE_PATH=tests/features/html cargo test
```

### Run with Debug Logging

```bash
RUST_LOG=debug cargo test
```

## Test Structure

```
bdd/
├── Cargo.toml                    # BDD harness dependencies
├── src/
│   ├── lib.rs                    # Library root
│   └── steps/
│       ├── mod.rs                # Step module exports
│       ├── world.rs              # DxWorld state
│       ├── css_steps.rs          # CSS command steps
│       ├── html_steps.rs         # HTML command steps
│       └── story_steps.rs        # Story locator steps
└── tests/
    ├── cucumber.rs               # Test runner
    └── features/
        ├── story/
        │   └── story_locator.feature
        ├── css/
        │   ├── class_exists.feature
        │   ├── selector_styles.feature
        │   └── list_classes.feature
        └── html/
            ├── query_selector.feature
            ├── attributes.feature
            └── dom_tree.feature
```

## Feature Files

### Priority 0: Story File Locator

**File:** `tests/features/story/story_locator.feature`

Tests the `dx story-file <URL>` command that helps engineers locate story files from Storybook URLs.

**Scenarios:**
- Locate Button story file
- Handle variant IDs in URLs
- Invalid URL error handling
- File not found error handling

### CSS Commands

**Files:** `tests/features/css/*.feature`

Tests CSS verification commands:
- `class_exists.feature` - Check if classes exist
- `selector_styles.feature` - Extract computed styles
- `list_classes.feature` - List element classes

### HTML Commands

**Files:** `tests/features/html/*.feature`

Tests HTML structure queries:
- `query_selector.feature` - Query DOM elements
- `attributes.feature` - Extract element attributes
- `dom_tree.feature` - Visualize DOM tree

## Writing New Tests

### 1. Create Feature File

```gherkin
# tests/features/css/new_feature.feature
Feature: New CSS Feature
  As a frontend developer
  I want to verify something
  So that I can ensure quality

  Background:
    Given Storybook is running on port 6006

  Scenario: Test something
    When I do something
    Then I should see expected result
```

### 2. Implement Step Definitions

Add to appropriate step file (`css_steps.rs`, `html_steps.rs`, etc.):

```rust
#[when(regex = r"^I do something$")]
pub async fn do_something(world: &mut DxWorld) {
    // Call dx command
    // Store result in world
}

#[then(regex = r"^I should see expected result$")]
pub async fn verify_result(world: &mut DxWorld) {
    // Assert on world state
}
```

### 3. Update World State (if needed)

Add fields to `DxWorld` in `world.rs`:

```rust
pub struct DxWorld {
    // ... existing fields ...
    pub new_field: Option<String>,
}
```

## Troubleshooting

### "Connection refused" errors

**Cause:** Storybook is not running on port 6006

**Solution:**
```bash
cd frontend/libs/storybook
pnpm story:dev
```

### "Story file not found" errors

**Cause:** Test expects story files that don't exist

**Solution:** Update feature file with actual story paths, or create the expected story files

### Timeout errors

**Cause:** Storybook is slow to respond

**Solution:** Increase timeout in `tests/cucumber.rs` or optimize Storybook build

## Performance Targets

Per DX Engineering Rules:
- **Individual scenarios:** < 10 seconds
- **Entire test suite:** < 2 minutes

## Integration with CI

Add to CI pipeline:

```yaml
- name: Start Storybook
  run: |
    cd frontend/libs/storybook
    pnpm story:dev &
    sleep 10  # Wait for server to start

- name: Run BDD Tests
  run: |
    cd frontend/.dx-tool/bdd
    cargo test
```

## References

- **DX Engineering Rules:** `/DX_ENGINEERING_RULES.md`
- **BDD Wiring Guide:** `/.docs/testing/BDD_WIRING.md`
- **Main BDD Harness:** `/test-harness/bdd/` (reference implementation)
- **Handoff Document:** `/frontend/.dx-tool/TEAM_DX_003_HANDOFF.md`
