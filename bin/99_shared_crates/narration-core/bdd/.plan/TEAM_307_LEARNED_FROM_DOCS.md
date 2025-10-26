# TEAM-307: Learned from BDD Documentation

**Date:** October 26, 2025  
**Status:** 📚 DOCUMENTATION REVIEWED  
**Team:** TEAM-307

---

## Key Learnings from BDD_WIRING.md

### Correct Cucumber-rs Syntax

#### 1. Step Attribute Syntax

**✅ CORRECT:**
```rust
#[given(regex = r#"^a narration context with job_id "([^"]+)"$"#)]
async fn context_with_job_id(world: &mut World, job_id: String) {
    world.context = Some(NarrationContext::new().with_job_id(job_id));
}
```

**❌ WRONG (what we tried):**
```rust
#[given(expr = "a narration context with job_id {string}")]  // expr doesn't exist!
async fn context_with_job_id(world: &mut World, job_id: String) {
    // ...
}
```

#### 2. Regex Patterns for Different Types

**String capture:**
```rust
#[when(regex = r#"^I emit narration with action "([^"]+)" and message "([^"]+)"$"#)]
async fn emit_narration(world: &mut World, action: String, message: String) {
    // ...
}
```

**Number capture:**
```rust
#[when(regex = r#"^I wait for (\d+) milliseconds$"#)]
async fn wait_ms(world: &mut World, ms: u64) {
    tokio::time::sleep(tokio::time::Duration::from_millis(ms)).await;
}
```

**Simple text (no parameters):**
```rust
#[given("the narration capture adapter is installed")]
async fn install_adapter(world: &mut World) {
    world.adapter = Some(CaptureAdapter::install());
}
```

#### 3. World Struct Requirements

**✅ CORRECT:**
```rust
#[derive(Debug, Default, cucumber::World)]
pub struct World {
    pub adapter: Option<CaptureAdapter>,
    pub context: Option<NarrationContext>,
    // ... other fields
}
```

**Key points:**
- Must derive `cucumber::World`
- Should derive `Debug` and `Default`
- Fields should be `pub` for step access

#### 4. Handling Tables (Gherkin Data Tables)

**✅ CORRECT:**
```rust
use cucumber::gherkin::Step;

#[given(regex = r#"^a narration context with:$"#)]
async fn context_with_fields(world: &mut World, step: &Step) {
    let mut ctx = NarrationContext::new();
    
    if let Some(table) = step.table.as_ref() {
        for row in table.rows.iter().skip(1) {  // Skip header
            let field = &row[0];
            let value = &row[1];
            
            match field.as_str() {
                "job_id" => ctx = ctx.with_job_id(value.clone()),
                "correlation_id" => ctx = ctx.with_correlation_id(value.clone()),
                _ => {}
            }
        }
    }
    
    world.context = Some(ctx);
}
```

#### 5. Lifetime Management in Async Blocks

**✅ CORRECT:**
```rust
#[when(regex = r#"^I emit multiple narrations in context$"#)]
async fn emit_multiple(world: &mut World, step: &Step) {
    if let Some(ctx) = world.context.clone() {
        if let Some(table) = step.table.as_ref() {
            // Clone data BEFORE moving into async block
            let rows: Vec<(String, String)> = table.rows.iter().skip(1)
                .map(|row| (row[0].clone(), row[1].clone()))
                .collect();
            
            with_narration_context(ctx, async move {
                for (action, message) in rows {
                    n!(&action, "{}", message);
                }
            }).await;
        }
    }
}
```

#### 6. Actor Static Lifetime Handling

**✅ CORRECT (using Box::leak):**
```rust
narrate(NarrationFields {
    actor: Box::leak(actor.into_boxed_str()),  // Convert to 'static
    action: "test",
    target: "test".to_string(),
    human: message,
    ..Default::default()
});
```

**✅ CORRECT (using predefined constants):**
```rust
const TEST_ACTOR: &str = "test-actor";

narrate(NarrationFields {
    actor: TEST_ACTOR,  // Already 'static
    action: "test",
    target: "test".to_string(),
    human: message,
    ..Default::default()
});
```

---

## Project Structure

### Correct BDD Layout

```
narration-core/bdd/
├── Cargo.toml                    # Package with cucumber dependency
├── src/
│   ├── main.rs                   # Entry point (calls World::cucumber())
│   └── steps/
│       ├── mod.rs                # Re-exports all step modules
│       ├── world.rs              # World struct with cucumber::World derive
│       ├── core_narration.rs     # Existing steps
│       ├── context_steps.rs      # New: context propagation steps
│       ├── sse_steps.rs          # TODO: SSE streaming steps
│       ├── job_steps.rs          # TODO: Job lifecycle steps
│       └── failure_steps.rs      # TODO: Failure scenario steps
└── features/                     # Gherkin feature files
    ├── cute_mode.feature
    ├── story_mode.feature
    ├── levels.feature
    ├── context_propagation.feature
    ├── sse_streaming.feature
    ├── job_lifecycle.feature
    └── failure_scenarios.feature
```

---

## What We Need to Fix

### 1. context_steps.rs - Convert to Regex Syntax

**Current (broken):**
```rust
#[given(expr = "a narration context with job_id {string}")]
```

**Fixed:**
```rust
#[given(regex = r#"^a narration context with job_id "([^"]+)"$"#)]
```

### 2. All String Parameters Need Regex

Every step with string parameters needs to use regex syntax:

```rust
// Simple string
#[when(regex = r#"^I emit narration with action "([^"]+)"$"#)]

// Multiple strings
#[when(regex = r#"^I emit narration with action "([^"]+)" and message "([^"]+)"$"#)]

// String and number
#[when(regex = r#"^I wait for (\d+) milliseconds and emit "([^"]+)"$"#)]
```

### 3. Table Steps Need Step Parameter

```rust
use cucumber::gherkin::Step;

#[given(regex = r#"^a narration context with:$"#)]
async fn context_with_fields(world: &mut World, step: &Step) {
    // Access step.table.as_ref()
}
```

---

## Running BDD Tests

### Command

```bash
# Run all features
cargo test -p observability-narration-core-bdd --bin bdd-runner -- --nocapture

# Run specific feature
LLORCH_BDD_FEATURE_PATH=features/context_propagation.feature \
  cargo test -p observability-narration-core-bdd --bin bdd-runner -- --nocapture
```

### Environment Variables

- `LLORCH_BDD_FEATURE_PATH` - Path to specific feature file or directory
- `RUST_LOG=debug` - Enable debug logging

---

## Action Items

### Immediate (Fix context_steps.rs)

1. ✅ Convert all `expr` to `regex`
2. ✅ Use proper regex patterns for string capture
3. ✅ Fix lifetime issues (clone before async)
4. ✅ Handle actor static lifetime (Box::leak or constants)
5. ✅ Test compilation

### Short Term (Complete Step Definitions)

6. Create sse_steps.rs with regex syntax
7. Create job_steps.rs with regex syntax
8. Create failure_steps.rs with regex syntax
9. Update existing steps if needed

### Medium Term (Run and Verify)

10. Run BDD tests
11. Fix any runtime failures
12. Verify all scenarios pass
13. Document results

---

## Examples from Existing Code

### From core_narration.rs

```rust
#[when(regex = "^I narrate with actor (.+), action (.+), target (.+), and human (.+)$")]
pub async fn when_narrate_full(
    world: &mut World,
    actor: String,
    action: String,
    target: String,
    human_text: String,
) {
    narrate(NarrationFields {
        actor: Box::leak(actor.into_boxed_str()),
        action: Box::leak(action.into_boxed_str()),
        target,
        human: human_text,
        ..Default::default()
    });
}
```

### From test_capture.rs

```rust
#[when(regex = r#"^I narrate with human "([^"]+)"$"#)]
pub async fn when_narrate_with_human(_world: &mut World, human: String) {
    narrate(NarrationFields {
        actor: "test",
        action: "test",
        target: "test".to_string(),
        human,
        ..Default::default()
    });
}
```

---

## Conclusion

**Key Takeaways:**

1. ✅ Use `regex` not `expr`
2. ✅ Pattern: `r#"^...([^"]+)...$"#` for strings
3. ✅ Clone data before async blocks
4. ✅ Use Box::leak for actor strings
5. ✅ Import `cucumber::gherkin::Step` for tables

**Next Steps:**

1. Fix context_steps.rs with correct syntax
2. Create remaining step files
3. Run and verify tests

**Status:** Ready to implement with correct patterns!

---

**Document Version:** 1.0  
**Last Updated:** October 26, 2025  
**Status:** Documentation Review Complete, Ready to Fix Implementation
