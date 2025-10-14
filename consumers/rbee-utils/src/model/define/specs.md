# model/define

## Purpose
Declare the LLM to use. Model is required. Engine/pool are optional hints.

## Input
```json
{
  "model_id": "required.model.name",
  "engine_id": "optional-engine",
  "pool_hint": "optional-pool"
}
```

## Output
```json
{ "model_ref": { "model_id": "required.model.name", "engine_id": null, "pool_hint": null } }
```

## Rust surface (define.rs)
```rust
pub struct ModelRef { pub model_id: String, pub engine_id: Option<String>, pub pool_hint: Option<String> }
pub fn run(model_id: String, engine_id: Option<String>, pool_hint: Option<String>) -> ModelRef;
```
