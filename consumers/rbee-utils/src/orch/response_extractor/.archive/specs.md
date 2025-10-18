# orch/response_extractor

## Purpose
Turn a llama-orch response into plain text. This is specifically tied to llama-orch result shapes.

## Input
```json
{ "result": { "choices": [ { "text": "final answer…" } ] } }
```

## Output
```json
{ "text": "final answer…" }
```

## Rust surface (response_extractor.rs)
```rust
use llama_orch_sdk::InvokeResult;

pub fn run(result: &InvokeResult) -> String; // extracts best-effort text
```

## Rules
If choices[0].text exists → return it.
Else if there’s a content array → join string parts.
Else return empty string (or a simple “[no text]” if you prefer).
