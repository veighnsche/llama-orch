# prompt/message

## Purpose
Build a single {role, content} message from: raw string, string array, or file ref, with optional dedent.

## Input
```json
{
  "role": "system|user|assistant",
  "from": {
    "text": "string | null",
    "lines": ["optional", "array"],
    "file": "optional/file/path.md"
  },
  "dedent": true
}
```

## Output
```json
{ "message": { "role": "user", "content": "final string" } }
```

## Rust surface (message.rs)
```rust
pub enum Source { Text(String), Lines(Vec<String>), File(String) }
pub struct MessageIn { pub role: String, pub source: Source, pub dedent: bool }
pub struct Message { pub role: String, pub content: String }

pub fn run(input: MessageIn) -> std::io::Result<Message>;
```

## Notes
Not seed-coupled. dedent implements the typical leading-margin trim.
If file given, read via fs.file_reader or directly—your call.
