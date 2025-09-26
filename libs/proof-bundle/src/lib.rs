//! proof-bundle — Utilities to emit crate-local proof bundles for tests.
//!
//! Standard location: `<crate>/.proof_bundle/<type>/<run_id>/...`
//! - Override base with `LLORCH_PROOF_DIR`
//! - Override run_id with `LLORCH_RUN_ID`
//! - Recommended run_id format: `YYYYMMDD-HHMMSS-<git_sha8>` (fallbacks supported)
mod env;
mod proof_bundle;
mod seeds;
mod test_type;
mod util;

pub use crate::proof_bundle::ProofBundle;
pub use crate::seeds::SeedsRecorder;
pub use crate::test_type::TestType;

// Types and helpers are defined in their own modules and re-exported above.

// ProofBundle implementation lives in proof_bundle.rs

// SeedsRecorder lives in seeds.rs

// Env helpers live in env.rs

#[cfg(test)]
mod tests {
    use super::{ProofBundle, TestType};

    #[test]
    fn creates_bundle_dirs() {
        let pb = ProofBundle::for_type(TestType::Unit).unwrap();
        assert!(pb.root().exists());
        pb.ensure_dir("sub").unwrap();
        assert!(pb.root().join("sub").exists());
    }

    #[test]
    fn writes_files() {
        let pb = ProofBundle::for_type(TestType::Integration).unwrap();
        pb.write_markdown("test_report.md", "# OK\n").unwrap();
        pb.write_json("run_config", &serde_json::json!({"ok": true})).unwrap();
        pb.append_ndjson("streaming_transcript", &serde_json::json!({"event":"started"})).unwrap();
        pb.seeds().record(42).unwrap();
    }
}
