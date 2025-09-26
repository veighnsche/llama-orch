# BDD (Cucumber) — Guide and Proof Bundle

## What

Behavior-Driven scenarios that validate user journeys through public interfaces (HTTP/CLI).

## Where

- Harness at `test-harness/bdd/` (binary `bdd-runner`).
- Proof bundles: `test-harness/bdd/.proof_bundle/bdd/<run_id>/` (harness crate) and any crate-under-test may additionally write crate-local bundles if it emits artifacts.

## When

- End-to-end behaviors and cross-service flows; acceptance scenarios.

## Running

- All: `cargo test -p test-harness-bdd -- --nocapture`
- Targeted: `LLORCH_BDD_FEATURE_PATH=tests/features/... cargo test -p test-harness-bdd -- --nocapture`

## Env

- `LLORCH_RUN_ID` (recommended)
- `LLORCH_PROOF_DIR` (optional)

## Artifacts (see template)

- `bdd_transcript.ndjson` — `{ feature, scenario, step, status, at, data? }`
- `http_traces_redacted.ndjson` (if HTTP involved)
- `test_report.md` — pass/fail, requirement IDs, pointers
- Optional: `world_logs/` for domain-specific logs

## Step-world hooks (Rust)

```rust
use std::fs::{create_dir_all, File};
use std::io::Write;
use std::path::PathBuf;

pub struct World { proof_dir: PathBuf /* ... */ }

impl World {
    pub fn init_proof_dir() -> PathBuf {
        let base = std::env::var("LLORCH_PROOF_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(".proof_bundle"));
        let run_id = std::env::var("LLORCH_RUN_ID").unwrap_or_else(|_| {
            let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
            format!("{}", ts)
        });
        let dir = base.join("bdd").join(run_id);
        create_dir_all(&dir).unwrap();
        dir
    }

    pub fn write_step(&self, rec: serde_json::Value) {
        let mut f = File::options().create(true).append(true).open(self.proof_dir.join("bdd_transcript.ndjson")).unwrap();
        writeln!(f, "{}", rec).unwrap();
    }
}
```

## Links

- Harness wiring: `.docs/testing/BDD_WIRING.md`
- Template: `.proof_bundle/templates/bdd/README.md`
- Index: `.docs/testing/TEST_TYPES_GUIDE.md`
