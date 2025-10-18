# prompt/thread

## Purpose
Compose an ordered array of messages; each item can use any of the message sources.

## Input
```json
{
  "items": [
    { "role": "system", "from": {"file": "docs/policy.md"}, "dedent": true },
    { "role": "user",   "from": {"lines": ["hi", "there"]} }
  ]
}
```

## Output
```json
{ "messages": [ { "role": "system", "content": "…" }, { "role": "user", "content": "…" } ] }
```

## Rust surface (thread.rs)
```rust
pub struct ThreadItem { pub role: String, pub source: super::message::Source, pub dedent: bool }
pub struct ThreadIn { pub items: Vec<ThreadItem> }
pub struct ThreadOut { pub messages: Vec<super::message::Message> }

pub fn run(input: ThreadIn) -> std::io::Result<ThreadOut>;
```
