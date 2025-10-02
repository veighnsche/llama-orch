//! proof-bundle — Utilities to emit crate-local proof bundles for tests.
//!
//! Standard location: `<crate>/.proof_bundle/<type>/<run_id>/...`
//! - Override base with `LLORCH_PROOF_DIR`
//! - Override run_id with `LLORCH_RUN_ID`
//! - Recommended run_id format: `YYYYMMDD-HHMMSS-<git_sha8>` (fallbacks supported)
pub mod env;
pub mod fs;
pub mod policy;
pub mod types;
pub mod util;
pub mod writers;

pub use crate::fs::{ProofBundle, SeedsRecorder};
pub use crate::types::TestType;

// Types and helpers are defined in their own modules and re-exported above.

// ProofBundle implementation lives in proof_bundle.rs

// SeedsRecorder lives in seeds.rs

// Env helpers live in env.rs

#[cfg(test)]
mod tests {
    use super::{ProofBundle, TestType};
    use std::fs;

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

    #[test]
    #[serial_test::serial]
    fn cleanup_removes_old_bundles() {
        // Create first bundle
        let pb1 = ProofBundle::for_type(TestType::Unit).unwrap();
        let root1 = pb1.root().to_path_buf();
        pb1.write_markdown("test.md", "# First bundle\n").unwrap();
        assert!(root1.exists());
        assert!(root1.join("test.md").exists());
        
        // Get the parent directory (the test type directory)
        let type_dir = root1.parent().unwrap();
        
        // Verify first bundle exists
        let entries_before: Vec<_> = fs::read_dir(type_dir)
            .unwrap()
            .filter_map(Result::ok)
            .filter(|e| e.path().is_dir())
            .collect();
        assert_eq!(entries_before.len(), 1, "Should have exactly 1 bundle before cleanup");
        
        // Create second bundle with different run_id (simulated by env var)
        std::env::set_var("LLORCH_RUN_ID", "20251002-999999-test");
        let pb2 = ProofBundle::for_type(TestType::Unit).unwrap();
        let root2 = pb2.root().to_path_buf();
        std::env::remove_var("LLORCH_RUN_ID");
        
        // Verify second bundle exists
        assert!(root2.exists());
        
        // Verify first bundle was deleted
        assert!(!root1.exists(), "Old bundle should be deleted");
        
        // Verify only one bundle remains
        let entries_after: Vec<_> = fs::read_dir(type_dir)
            .unwrap()
            .filter_map(Result::ok)
            .filter(|e| e.path().is_dir())
            .collect();
        assert_eq!(entries_after.len(), 1, "Should have exactly 1 bundle after cleanup");
        assert_eq!(entries_after[0].path(), root2, "Only the new bundle should remain");
    }
}
