use crate::path_utils::path_relative_to;
use crate::types::{CrateToml, ManifestEntry, RootWorkspaceToml};
use anyhow::{Context, Result};
use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

pub fn read_workspace_members(repo_root: &Path) -> Result<Vec<String>> {
    let cargo_toml_path = repo_root.join("Cargo.toml");
    let txt = fs::read_to_string(&cargo_toml_path)
        .with_context(|| format!("reading {}", cargo_toml_path.display()))?;
    let parsed: RootWorkspaceToml =
        toml::from_str(&txt).with_context(|| format!("parsing {}", cargo_toml_path.display()))?;
    Ok(parsed.workspace.members)
}

pub fn build_manifest_entry(repo_root: &Path, crate_dir: &Path) -> Result<ManifestEntry> {
    let cargo_toml_path = crate_dir.join("Cargo.toml");
    let txt = fs::read_to_string(&cargo_toml_path)
        .with_context(|| format!("reading {}", cargo_toml_path.display()))?;
    let parsed: CrateToml =
        toml::from_str(&txt).with_context(|| format!("parsing {}", cargo_toml_path.display()))?;

    let has_lib = parsed.lib.is_some();
    let has_bin_table = parsed.bin.as_ref().map(|v| !v.is_empty()).unwrap_or(false);
    let default_main = crate_dir.join("src/main.rs").exists();
    let default_lib = crate_dir.join("src/lib.rs").exists();

    let kind = match (has_lib || default_lib, has_bin_table || default_main) {
        (true, true) => "mixed",
        (true, false) => "lib",
        (false, true) => "bin",
        (false, false) => {
            if default_lib {
                "lib"
            } else if default_main {
                "bin"
            } else {
                "lib"
            }
        }
    }
    .to_string();

    let role = detect_role(repo_root, crate_dir);

    let mut binaries: BTreeSet<String> = BTreeSet::new();
    if let Some(bin) = &parsed.bin {
        for _b in bin {
            if let Some(name) = _b.get("name").and_then(|v| v.as_str()).map(|s| s.to_string()) {
                binaries.insert(name);
            }
        }
    }
    if kind == "bin" && default_main {
        binaries.insert(parsed.package.name.clone());
    }

    let features: Vec<String> = parsed.features.keys().cloned().collect();

    let tests = detect_tests(crate_dir);
    let docs_paths = vec!["README.md".to_string()];

    let owner = read_codeowners_default(repo_root)?;

    let openapi_refs = detect_openapi_refs(repo_root, crate_dir);
    let schema_refs = detect_schema_refs(repo_root, crate_dir);
    let spec_refs =
        suggest_spec_refs(&role, crate_dir.file_name().and_then(|s| s.to_str()).unwrap_or(""));

    Ok(ManifestEntry {
        path: path_relative_to(repo_root, crate_dir)?,
        name: parsed.package.name,
        kind,
        role,
        description: parsed.package.description,
        owner,
        spec_refs,
        openapi_refs,
        schema_refs,
        binaries: binaries.into_iter().collect(),
        features,
        tests,
        docs_paths,
    })
}

pub fn detect_role(repo_root: &Path, crate_dir: &Path) -> String {
    let rel =
        path_relative_to(repo_root, crate_dir).unwrap_or_else(|_| crate_dir.display().to_string());
    // Core crates
    if rel.starts_with("libs/orchestrator-core") || rel.starts_with("orchestrator-core") {
        return "core".to_string();
    }
    if rel.starts_with("bin/orchestratord") || rel.starts_with("orchestratord") {
        return "core".to_string();
    }
    if rel.starts_with("libs/pool-managerd") || rel.starts_with("pool-managerd") {
        return "core".to_string();
    }
    if rel.starts_with("libs/catalog-core") || rel.starts_with("catalog-core") {
        return "core".to_string();
    }

    // Adapters and related libs
    if rel.starts_with("libs/worker-adapters/") || rel.starts_with("worker-adapters/") {
        return "adapter".to_string();
    }
    if rel.starts_with("libs/adapter-host") || rel.starts_with("adapter-host") {
        return "adapter".to_string();
    }
    if rel.starts_with("libs/plugins/") || rel.starts_with("plugins/") {
        return "plugin".to_string();
    }

    // Contracts
    if rel.starts_with("contracts/") {
        return "contracts".to_string();
    }

    // Test harness and CLI
    if rel.starts_with("test-harness/") || rel.starts_with("cli/") {
        return "test-harness".to_string();
    }

    // Tooling
    if rel.starts_with("tools/") || rel.starts_with("xtask") {
        return "tool".to_string();
    }

    // Default
    "tool".to_string()
}

pub fn detect_tests(crate_dir: &Path) -> Vec<String> {
    let tests_dir = crate_dir.join("tests");
    if !tests_dir.exists() {
        return vec![];
    }
    let mut names = Vec::new();
    if let Ok(rd) = fs::read_dir(&tests_dir) {
        for ent in rd.flatten() {
            let p = ent.path();
            if let Some(name) = p.file_stem().and_then(|s| s.to_str()) {
                names.push(name.to_string());
            }
        }
    }
    names.sort();
    names
}

pub fn read_codeowners_default(repo_root: &Path) -> Result<String> {
    let p = repo_root.join("CODEOWNERS");
    if !p.exists() {
        return Ok("".into());
    }
    let txt = fs::read_to_string(&p)?;
    for line in txt.lines() {
        let line = line.trim();
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        if line.starts_with('*') {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if let Some(owner) = parts.get(1) {
                return Ok((*owner).to_string());
            }
        }
    }
    Ok("".into())
}

pub fn detect_openapi_refs(repo_root: &Path, crate_dir: &Path) -> Vec<String> {
    let rel = path_relative_to(repo_root, crate_dir).unwrap_or_default();
    let mut refs = Vec::new();
    if rel.starts_with("tools/openapi-client")
        || rel.starts_with("bin/orchestratord")
        || rel.starts_with("orchestratord")
    {
        if repo_root.join("contracts/openapi/control.yaml").exists() {
            refs.push("contracts/openapi/control.yaml".to_string());
        }
        if repo_root.join("contracts/openapi/data.yaml").exists() {
            refs.push("contracts/openapi/data.yaml".to_string());
        }
    }
    refs
}

pub fn detect_schema_refs(repo_root: &Path, crate_dir: &Path) -> Vec<String> {
    let rel = path_relative_to(repo_root, crate_dir).unwrap_or_default();
    let mut refs = Vec::new();
    if rel.starts_with("contracts/config-schema") {
        let f = repo_root.join("contracts/config-schema/src/lib.rs");
        if f.exists() {
            refs.push("contracts/config-schema/src/lib.rs".to_string());
        }
    }
    refs
}

pub fn suggest_spec_refs(role: &str, crate_name: &str) -> Vec<String> {
    let mut v = Vec::new();
    match role {
        "core" => {
            v.extend(
                [
                    "ORCH-3004",
                    "ORCH-3005",
                    "ORCH-3008",
                    "ORCH-3010",
                    "ORCH-3011",
                    "ORCH-3016",
                    "ORCH-3017",
                    "ORCH-3027",
                    "ORCH-3028",
                    "ORCH-3044",
                    "ORCH-3045",
                ]
                .iter()
                .map(|s| s.to_string()),
            );
            if crate_name.contains("orchestratord") {
                v.push("ORCH-2002".to_string());
                v.push("ORCH-2101".to_string());
                v.push("ORCH-2102".to_string());
                v.push("ORCH-2103".to_string());
                v.push("ORCH-2104".to_string());
            }
            if crate_name.contains("pool-managerd") {
                v.push("ORCH-3038".to_string());
                v.push("ORCH-3002".to_string());
            }
        }
        "adapter" => {
            v.extend(
                ["ORCH-3054", "ORCH-3055", "ORCH-3056", "ORCH-3057", "ORCH-3058"]
                    .iter()
                    .map(|s| s.to_string()),
            );
        }
        "contracts" => {
            v.extend(["ORCH-3044", "ORCH-3030"].iter().map(|s| s.to_string()));
        }
        "test-harness" => {
            v.extend(["ORCH-3050", "ORCH-3051"].iter().map(|s| s.to_string()));
        }
        "plugin" => {
            v.extend(["ORCH-3048"].iter().map(|s| s.to_string()));
        }
        _ => {}
    }
    v
}

/// Hand-curated minimal extras per crate (Step 3) — include in footnotes
pub fn crate_specific_extras(crate_path: &str) -> String {
    if crate_path.starts_with("libs/orchestrator-core")
        || crate_path.starts_with("orchestrator-core")
    {
        return "- Queue invariants and property tests (capacity, rejection policies, session affinity helpers).\n- Capacity policies and bounded FIFO behavior.\n".to_string();
    }
    if crate_path.starts_with("bin/orchestratord") || crate_path.starts_with("orchestratord") {
        return "- Data/control plane routes, SSE framing details, backpressure headers, provider verify entry points.\n".to_string();
    }
    if crate_path.starts_with("libs/pool-managerd") || crate_path.starts_with("pool-managerd") {
        return "- Preload/Ready lifecycle, NVIDIA-only guardrails, restart/backoff behavior.\n"
            .to_string();
    }
    if crate_path.starts_with("libs/worker-adapters/") || crate_path.starts_with("worker-adapters/")
    {
        return "- Engine endpoint mapping tables (native/OpenAI-compat to adapter calls), determinism knobs, version capture.\n".to_string();
    }
    if crate_path.starts_with("contracts/") {
        return "- How to regenerate types, schemas, and validate; pact files location and scope.\n".to_string();
    }
    if crate_path.starts_with("libs/plugins/") || crate_path.starts_with("plugins/") {
        return "- WASI policy ABI and SDK usage; example plugin pointers.\n".to_string();
    }
    if crate_path.starts_with("test-harness/") {
        return "- Which tests are ignored vs required; how to run real-model Haiku; determinism suite scope.\n".to_string();
    }
    if crate_path.starts_with("tools/") {
        return "- Responsibilities, inputs/outputs; how determinism and idempotent regeneration are enforced.\n".to_string();
    }
    String::new()
}
