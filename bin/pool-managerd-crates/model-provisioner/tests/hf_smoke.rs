//! E2E smoke test (NOT a unit test). Run with MODEL_ORCH_SMOKE=1 and provide MODEL_ORCH_SMOKE_REF.
//! This test exercises the real `huggingface-cli` path end-to-end in an isolated HOME and cache dir.
//! It is ignored by default to keep CI offline and deterministic.

#![cfg(test)]

use model_catalog::{FileFetcher, FsCatalog, ModelRef};
use model_provisioner::ModelProvisioner;
use std::sync::{Mutex, OnceLock};

static PATH_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

#[test]
#[ignore]
fn hf_smoke_end_to_end() {
    // Gate by environment to avoid accidental network usage
    if std::env::var("MODEL_ORCH_SMOKE").ok().as_deref() != Some("1") {
        eprintln!("MODEL_ORCH_SMOKE!=1; skipping e2e smoke");
        return;
    }

    let _guard = PATH_LOCK.get_or_init(|| Mutex::new(())).lock().unwrap();

    // Require an explicit HF ref to avoid huge downloads by default
    let hf_ref = match std::env::var("MODEL_ORCH_SMOKE_REF") {
        Ok(v) if !v.trim().is_empty() => v,
        _ => {
            eprintln!("MODEL_ORCH_SMOKE_REF not set; skipping e2e smoke");
            return;
        }
    };

    // Ensure HF CLI is present (prefer 'hf', fallback to 'huggingface-cli')
    if which::which("hf").is_err() && which::which("huggingface-cli").is_err() {
        eprintln!("HF CLI ('hf' or 'huggingface-cli') not found in PATH; skipping e2e smoke");
        return;
    }

    // Isolate HOME and cache/catalog
    let fake_home = tempfile::tempdir().expect("temp HOME");
    let old_home = std::env::var("HOME").ok();
    std::env::set_var("HOME", fake_home.path());

    let cache = tempfile::tempdir().expect("cache dir");
    let catalog = FsCatalog::new(cache.path()).expect("catalog");
    let fetcher = FileFetcher; // file paths handled by fetcher; hf handled by provisioner fallback
    let prov = ModelProvisioner::new(catalog, fetcher);

    let mr = ModelRef::parse(&hf_ref).expect("parse hf ref");
    let resolved = prov.ensure_present(&mr, None).expect("ensure present");

    // The resolved path should exist after a real download
    assert!(
        resolved.local_path.exists(),
        "resolved path should exist after HF download: {}",
        resolved.local_path.display()
    );

    // Restore HOME
    if let Some(h) = old_home {
        std::env::set_var("HOME", h);
    } else {
        std::env::remove_var("HOME");
    }
}
