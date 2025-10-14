# fs/file_writer

## Purpose
Write text to a file. No guardrails for M2 (caller picks the path).

## Input
```json
{ "path": "output/README.md", "text": "content to write", "create_dirs": true }
```

## Output
```json
{ "path": "output/README.md", "bytes_written": 1234 }
```

## Rust surface (file_writer.rs)
```rust
pub struct WriteIn { pub path: String, pub text: String, pub create_dirs: bool }
pub struct WriteOut { pub path: String, pub bytes_written: usize }

pub fn run(input: WriteIn) -> std::io::Result<WriteOut>;
```

## Notes
Overwrite if the file exists.
If create_dirs=true, mkdir -p behavior.
