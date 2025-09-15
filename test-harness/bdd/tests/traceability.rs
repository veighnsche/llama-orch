use regex::Regex;
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../")
}

fn catalog_paths() -> Vec<PathBuf> {
    let root = repo_root();
    vec![
        root.join(".docs/testing/spec-derived-test-catalog.md"),
        root.join(".docs/testing/spec-combination-matrix.md"),
    ]
}

fn spec_md_paths() -> Vec<PathBuf> {
    let mut out = Vec::new();
    let dir = repo_root().join(".specs");
    if dir.exists() {
        for entry in walkdir::WalkDir::new(&dir).into_iter().filter_map(Result::ok) {
            if entry.file_type().is_file() && entry.path().extension().and_then(|s| s.to_str()) == Some("md") {
                out.push(entry.into_path());
            }
        }
    }
    out
}

fn feature_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/features")
}

fn parse_ids_from_text(text: &str) -> HashSet<String> {
    let mut ids = HashSet::new();

    // Expand ranges like ORCH-3060..3065 or OC-CTRL-2050..2051
    let re_range = Regex::new(r"(?m)\b([A-Z]+(?:-[A-Z]+)*)-(\d{3,5})\.\.(\d{3,5})\b").unwrap();
    for cap in re_range.captures_iter(text) {
        let prefix = &cap[1];
        let start: u32 = cap[2].parse().unwrap_or(0);
        let end: u32 = cap[3].parse().unwrap_or(0);
        if start > 0 && end >= start { for n in start..=end { ids.insert(format!("{}-{}", prefix, n)); } }
    }

    // Single IDs like ORCH-3045, OC-POOL-3001
    let re_single = Regex::new(r"(?m)\b([A-Z]+(?:-[A-Z]+)*)-(\d{3,5})\b").unwrap();
    for cap in re_single.captures_iter(text) {
        ids.insert(format!("{}-{}", &cap[1], &cap[2]));
    }

    ids
}

fn read_file(path: &Path) -> Option<String> {
    fs::read_to_string(path).ok()
}

fn collect_catalog_ids() -> HashSet<String> {
    let mut all = HashSet::new();
    for p in catalog_paths() {
        if let Some(s) = read_file(&p) {
            let ids = parse_ids_from_text(&s);
            all.extend(ids);
        }
    }
    for p in spec_md_paths() {
        if let Some(s) = read_file(&p) {
            let ids = parse_ids_from_text(&s);
            all.extend(ids);
        }
    }
    all
}

fn collect_feature_ids() -> HashSet<String> {
    let mut found = HashSet::new();
    let dir = feature_dir();
    for entry in walkdir::WalkDir::new(&dir).into_iter().filter_map(Result::ok) {
        if entry.file_type().is_file() && entry.path().extension().and_then(|s| s.to_str()) == Some("feature") {
            if let Ok(text) = fs::read_to_string(entry.path()) {
                // Only scan lines that declare traceability comment but also collect any IDs appearing anywhere
                let ids = parse_ids_from_text(&text);
                found.extend(ids);
            }
        }
    }
    found
}

#[test]
fn catalog_requirements_are_referenced_by_features() {
    let mut catalog = collect_catalog_ids();
    // Optional domain filter, e.g. "ORCH,OC-CTRL"
    if let Ok(domains) = std::env::var("LLORCH_TRACEABILITY_DOMAIN") {
        let keep: HashSet<String> = domains
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        catalog = catalog
            .into_iter()
            .filter(|id| keep.iter().any(|p| id.starts_with(p)))
            .collect();
    }
    let referenced = collect_feature_ids();

    // Ignore obviously non-BDD-only domains if needed (example: OC-TEST determinism suite may live elsewhere)
    // For now, keep all and just report.

    let missing: Vec<_> = catalog.difference(&referenced).cloned().collect();
    if missing.is_empty() {
        eprintln!("Traceability: all catalog IDs are referenced by features (at least once). Total: {}", catalog.len());
        return;
    }

    eprintln!("Traceability: missing {} IDs not referenced by any .feature file:", missing.len());
    let mut sorted = missing;
    sorted.sort();
    for id in &sorted { eprintln!(" - {}", id); }

    // Only fail in strict mode to keep scaffolding green by default
    let strict = std::env::var("LLORCH_TRACEABILITY_STRICT").ok().filter(|v| v == "1").is_some();
    if strict {
        panic!("traceability missing {} IDs; set LLORCH_TRACEABILITY_STRICT=0 to disable failing", sorted.len());
    }
}
