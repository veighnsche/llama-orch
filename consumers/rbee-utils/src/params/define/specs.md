# params/define

## Purpose
Optional generation settings; provide nothing and defaults will be used.

## Input
```json
{
  "temperature": 0.2,
  "top_p": 0.95,
  "max_tokens": 1024,
  "seed": 42
}
```

## Output
```json
{ "params_ref": { "temperature": 0.2, "top_p": 0.95, "max_tokens": 1024, "seed": 42 } }
```

## Rust surface (define.rs)
```rust
pub struct Params { pub temperature: Option<f32>, pub top_p: Option<f32>, pub max_tokens: Option<u32>, pub seed: Option<u64> }
pub fn run(p: Params) -> Params; // identity; just normalizes
```
