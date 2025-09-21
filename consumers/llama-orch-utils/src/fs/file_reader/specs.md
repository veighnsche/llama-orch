# fs/file_reader

## Purpose
Read one or more files and hand back byte/string blobs. Generic (not seed-specific).

## Input (input.json)
```json
{
  "paths": ["/abs/or/rel/path1.md", "/abs/or/rel/path2.txt"],
  "as_text": true,
  "encoding": "utf-8"
}
```

## Output (output.json)
```json
{
  "files": [
    { "path": "path1.md", "content": "…", "bytes": null },
    { "path": "path2.txt", "content": "…", "bytes": null }
  ]
}
```

If as_text=false, return "bytes": "<base64>" and omit content.

## Rust surface (file_reader.rs)
```rust
pub struct ReadRequest { pub paths: Vec<String>, pub as_text: bool, pub encoding: Option<String> }
pub struct FileBlob { pub path: String, pub content: Option<String>, pub bytes: Option<Vec<u8>> }
pub struct ReadResponse { pub files: Vec<FileBlob> }

pub fn run(req: ReadRequest) -> std::io::Result<ReadResponse>;
```

## Notes
- No guardrails: absolute/relative allowed.
- WASI-friendly if given a preopened dir; operate relative if needed.
