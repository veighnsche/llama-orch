# Integration Tests — Guide and Proof Bundle

## What
Cross-module tests within a crate using local stubs and adapters. No external network.

## Where
- `tests/` directories of crates.
- Proof bundles: `<crate>/.proof_bundle/integration/<run_id>/`.

## When
- Verifying crate-level behavior, backoff/retry across layers, streaming behavior, and boundary conditions.

## Artifacts (see template)
- `retry_timeline.jsonl`
- `streaming_transcript.ndjson` — `{ event: "started|token|metrics|end", ... }`
- `redacted_errors.*`
- `seeds.txt`
- `test_report.md`

## File formats and timing
- Prefer NDJSON for streams. If you capture per-token latencies, record alongside tokens or in a separate `timing.csv`.
- Avoid wall-clock sleeps; use mocked time.

## Recommended recipe (Rust)

```rust
use std::fs::{create_dir_all, File};
use std::io::Write;
use std::path::PathBuf;

fn proof_root() -> PathBuf { /* same as unit */ # [allow(dead_code)] { let base = std::env::var("LLORCH_PROOF_DIR").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(".proof_bundle")); let run_id = std::env::var("LLORCH_RUN_ID").unwrap_or_else(|_| { let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs(); format!("{}", ts) }); base.join("integration").join(run_id) } }

#[test]
fn captures_stream() {
    let root = proof_root();
    create_dir_all(&root).unwrap();
    let mut f = File::create(root.join("streaming_transcript.ndjson")).unwrap();
    let events = [
        serde_json::json!({"event": "started", "job_id": "J1"}),
        serde_json::json!({"event": "token", "t": 0, "text": "Hello"}),
        serde_json::json!({"event": "end", "ok": true})
    ];
    for e in events { writeln!(f, "{}", e).unwrap(); }
}
```

## Do/Don’t
- Do isolate external effects with stubs.
- Don’t reach into private state; use public APIs only.

## Links
- Template: `.proof_bundle/templates/integration/README.md`
- Index: `.docs/testing/TEST_TYPES_GUIDE.md`
