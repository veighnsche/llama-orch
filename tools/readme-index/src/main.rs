use anyhow::{Context, Result};
use regex::Regex;
use serde::{Deserialize, Serialize};
use openapiv3::{OpenAPI, ReferenceOr, Operation};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::Write;
use std::path::{Component, Path, PathBuf};

#[derive(Debug, Serialize, Deserialize, Clone)]
struct ManifestEntry {
    path: String,
    name: String,
    kind: String,      // lib | bin | mixed
    role: String,      // core | adapter | plugin | tool | test-harness | contracts
    description: String,
    owner: String,
    spec_refs: Vec<String>,
    openapi_refs: Vec<String>,
    schema_refs: Vec<String>,
    binaries: Vec<String>,
    features: Vec<String>,
    tests: Vec<String>,
    docs_paths: Vec<String>,
}

fn trim_trailing_blank_lines(s: &str) -> String {
    // Remove any trailing blank lines and ensure a single trailing newline
    let mut lines: Vec<&str> = s.split('\n').collect();
    while matches!(lines.last(), Some(l) if l.trim().is_empty()) {
        lines.pop();
    }
    let mut out = lines.join("\n");
    out.push('\n');
    out
}

fn openapi_summary(repo_root: &Path, refs: &[String]) -> String {
    let mut tags: BTreeSet<String> = BTreeSet::new();
    let mut op_ids: Vec<String> = Vec::new();
    for rel in refs {
        let abs = repo_root.join(rel);
        let Ok(txt) = fs::read_to_string(&abs) else { continue; };
        let Ok(doc) = serde_yaml::from_str::<OpenAPI>(&txt) else { continue; };
        for (_p, item) in doc.paths.paths {
            if let ReferenceOr::Item(pi) = item {
                collect_ops(&pi.get, &mut tags, &mut op_ids);
                collect_ops(&pi.put, &mut tags, &mut op_ids);
                collect_ops(&pi.post, &mut tags, &mut op_ids);
                collect_ops(&pi.delete, &mut tags, &mut op_ids);
                collect_ops(&pi.options, &mut tags, &mut op_ids);
                collect_ops(&pi.head, &mut tags, &mut op_ids);
                collect_ops(&pi.patch, &mut tags, &mut op_ids);
                collect_ops(&pi.trace, &mut tags, &mut op_ids);
            }
        }
    }
    op_ids.sort();
    op_ids.dedup();
    let mut out = String::new();
    if !op_ids.is_empty() {
        out.push_str(&format!("- OpenAPI operations: {}\n", op_ids.len()));
        let sample: Vec<String> = op_ids.iter().take(5).cloned().collect();
        if !sample.is_empty() {
            out.push_str(&format!("  - examples: {}\n", sample.join(", ")));
        }
    }
    if !tags.is_empty() {
        let tags_list: Vec<String> = tags.into_iter().collect();
        out.push_str(&format!("- OpenAPI tags: {}\n", tags_list.join(", ")));
    }
    out
}

fn collect_ops(op: &Option<Operation>, tags: &mut BTreeSet<String>, op_ids: &mut Vec<String>) {
    if let Some(op) = op {
        for t in &op.tags { tags.insert(t.clone()); }
        if let Some(id) = &op.operation_id { op_ids.push(id.clone()); }
    }
}

fn crate_specific_extras(crate_path: &str) -> String {
    if crate_path.starts_with("orchestrator-core") {
        return "- Queue invariants and property tests overview (fairness, capacity, rejection policies).\n- Capacity policies and bounded FIFO behavior.\n".to_string();
    }
    if crate_path.starts_with("orchestratord") {
        return "- Data/control plane routes, SSE framing details, backpressure headers, provider verify entry points.\n".to_string();
    }
    if crate_path.starts_with("pool-managerd") {
        return "- Preload/Ready lifecycle, NVIDIA-only guardrails, restart/backoff behavior.\n".to_string();
    }
    if crate_path.starts_with("worker-adapters/") {
        return "- Engine endpoint mapping tables (native/OpenAI-compat to adapter calls), determinism knobs, version capture.\n".to_string();
    }
    if crate_path.starts_with("contracts/") {
        return "- How to regenerate types, schemas, and validate; pact files location and scope.\n".to_string();
    }
    if crate_path.starts_with("plugins/") {
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

#[derive(Debug, Deserialize)]
struct RootWorkspaceToml {
    workspace: WorkspaceSection,
}

#[derive(Debug, Deserialize)]
struct WorkspaceSection {
    members: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct CrateToml {
    package: Package,
    #[serde(default)]
    lib: Option<toml::Value>,
    #[serde(default)]
    bin: Option<Vec<toml::Value>>,
    #[serde(default)]
    features: BTreeMap<String, toml::Value>,
}

#[derive(Debug, Deserialize)]
struct Package {
    name: String,
    #[serde(default)]
    description: String,
}

fn main() -> Result<()> {
    let repo_root = std::env::current_dir()?;

    // 0) Build JSON manifest deterministically
    let members = read_workspace_members(&repo_root)?;
    let mut entries: Vec<ManifestEntry> = Vec::new();
    for member in members {
        let crate_dir = repo_root.join(&member);
        let crate_entry = build_manifest_entry(&repo_root, &crate_dir)?;
        entries.push(crate_entry);
    }
    entries.sort_by(|a, b| a.path.cmp(&b.path));

    let out_dir = repo_root.join("tools/readme-index");
    fs::create_dir_all(&out_dir)?;
    let manifest_path = out_dir.join("readme_index.json");
    write_if_changed_pretty_json(&manifest_path, &entries)?;

    // 1) Ensure TEMPLATE.md exists (single source of truth)
    let template_path = out_dir.join("TEMPLATE.md");
    if !template_path.exists() {
        let template = default_template_md();
        write_if_changed(&template_path, &template)?;
    }

    // 2) Generate README.md for every crate
    for entry in &entries {
        let readme = render_readme(&repo_root, entry)?;
        let crate_readme = repo_root.join(&entry.path).join("README.md");
        write_if_changed(&crate_readme, &readme)?;
    }

    // 4) Update Root README: Consolidated Index
    let root_readme_path = repo_root.join("README.md");
    let updated_root = update_root_readme(&repo_root, &root_readme_path, &entries)?;
    write_if_changed(&root_readme_path, &updated_root)?;

    Ok(())
}

fn read_workspace_members(repo_root: &Path) -> Result<Vec<String>> {
    let cargo_toml_path = repo_root.join("Cargo.toml");
    let txt = fs::read_to_string(&cargo_toml_path)
        .with_context(|| format!("reading {}", cargo_toml_path.display()))?;
    let parsed: RootWorkspaceToml = toml::from_str(&txt)
        .with_context(|| format!("parsing {}", cargo_toml_path.display()))?;
    Ok(parsed.workspace.members)
}

fn build_manifest_entry(repo_root: &Path, crate_dir: &Path) -> Result<ManifestEntry> {
    let cargo_toml_path = crate_dir.join("Cargo.toml");
    let txt = fs::read_to_string(&cargo_toml_path)
        .with_context(|| format!("reading {}", cargo_toml_path.display()))?;
    let parsed: CrateToml = toml::from_str(&txt)
        .with_context(|| format!("parsing {}", cargo_toml_path.display()))?;

    let has_lib = parsed.lib.is_some();
    let has_bin_table = parsed.bin.as_ref().map(|v| !v.is_empty()).unwrap_or(false);
    let default_main = crate_dir.join("src/main.rs").exists();
    let default_lib = crate_dir.join("src/lib.rs").exists();

    let kind = match (has_lib || default_lib, has_bin_table || default_main) {
        (true, true) => "mixed",
        (true, false) => "lib",
        (false, true) => "bin",
        (false, false) => {
            // Fallback: assume lib if lib.rs exists, else bin if main.rs exists
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
            // Try to read name field if present
            if let Some(name) = _b
                .get("name")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
            {
                binaries.insert(name);
            }
        }
    }
    // default binary: package name when no [lib] and src/main.rs exists
    if kind == "bin" && default_main {
        binaries.insert(parsed.package.name.clone());
    }

    let features: Vec<String> = parsed.features.keys().cloned().collect();

    let tests = detect_tests(crate_dir);
    let docs_paths = vec!["README.md".to_string()];

    let owner = read_codeowners_default(repo_root)?;

    let openapi_refs = detect_openapi_refs(repo_root, crate_dir);
    let schema_refs = detect_schema_refs(repo_root, crate_dir);
    let spec_refs = suggest_spec_refs(&role, crate_dir.file_name().and_then(|s| s.to_str()).unwrap_or(""));

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

fn detect_role(repo_root: &Path, crate_dir: &Path) -> String {
    let rel = path_relative_to(repo_root, crate_dir).unwrap_or_else(|_| crate_dir.display().to_string());
    if rel.starts_with("orchestrator-core") {
        return "core".to_string();
    }
    if rel.starts_with("orchestratord") || rel.starts_with("pool-managerd") {
        return "core".to_string();
    }
    if rel.starts_with("worker-adapters/") {
        return "adapter".to_string();
    }
    if rel.starts_with("plugins/") {
        return "plugin".to_string();
    }
    if rel.starts_with("contracts/") {
        return "contracts".to_string();
    }
    if rel.starts_with("test-harness/") || rel.starts_with("cli/") {
        return "test-harness".to_string();
    }
    if rel.starts_with("tools/") || rel.starts_with("xtask") {
        return "tool".to_string();
    }
    "tool".to_string()
}

fn detect_tests(crate_dir: &Path) -> Vec<String> {
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

fn read_codeowners_default(repo_root: &Path) -> Result<String> {
    let p = repo_root.join("CODEOWNERS");
    if !p.exists() {
        return Ok("".into());
    }
    let txt = fs::read_to_string(&p)?;
    for line in txt.lines() {
        let line = line.trim();
        if line.starts_with('#') || line.is_empty() { continue; }
        // Simple pattern: "* @owner"
        if line.starts_with('*') {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if let Some(owner) = parts.get(1) {
                return Ok((*owner).to_string());
            }
        }
    }
    Ok("".into())
}

fn detect_openapi_refs(repo_root: &Path, crate_dir: &Path) -> Vec<String> {
    let rel = path_relative_to(repo_root, crate_dir).unwrap_or_default();
    let mut refs = Vec::new();
    if rel.starts_with("tools/openapi-client") || rel.starts_with("orchestratord") {
        if repo_root.join("contracts/openapi/control.yaml").exists() {
            refs.push("contracts/openapi/control.yaml".to_string());
        }
        if repo_root.join("contracts/openapi/data.yaml").exists() {
            refs.push("contracts/openapi/data.yaml".to_string());
        }
    }
    refs
}

fn detect_schema_refs(repo_root: &Path, crate_dir: &Path) -> Vec<String> {
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

fn suggest_spec_refs(role: &str, crate_name: &str) -> Vec<String> {
    let mut v = Vec::new();
    match role {
        "core" => {
            v.extend([
                "ORCH-3004", "ORCH-3005", "ORCH-3008", "ORCH-3010", "ORCH-3011", "ORCH-3016",
                "ORCH-3017", "ORCH-3027", "ORCH-3028", "ORCH-3044", "ORCH-3045",
            ].iter().map(|s| s.to_string()));
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
            v.extend([
                "ORCH-3054", "ORCH-3055", "ORCH-3056", "ORCH-3057", "ORCH-3058",
            ].iter().map(|s| s.to_string()));
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

fn path_relative_to<'a>(base: &'a Path, path: &'a Path) -> Result<String> {
    let base = base.canonicalize()?;
    let path = path.canonicalize()?;
    let rel = relative_path(&base, &path);
    Ok(rel.to_string_lossy().to_string())
}

fn relative_path(from: &Path, to: &Path) -> PathBuf {
    let from_components: Vec<&str> = from
        .components()
        .filter_map(|c| match c { Component::Normal(s) => s.to_str(), _ => None })
        .collect();
    let to_components: Vec<&str> = to
        .components()
        .filter_map(|c| match c { Component::Normal(s) => s.to_str(), _ => None })
        .collect();
    let mut i = 0usize;
    while i < from_components.len() && i < to_components.len() && from_components[i] == to_components[i] {
        i += 1;
    }
    let mut rel = PathBuf::new();
    for _ in i..from_components.len() {
        rel.push("..");
    }
    for comp in &to_components[i..] {
        rel.push(comp);
    }
    rel
}

fn write_if_changed(path: &Path, content: &str) -> Result<()> {
    let new_bytes = content.as_bytes();
    if let Ok(old) = fs::read(path) {
        if old == new_bytes {
            return Ok(());
        }
    }
    if let Some(parent) = path.parent() { fs::create_dir_all(parent)?; }
    let mut f = fs::File::create(path)?;
    f.write_all(new_bytes)?;
    Ok(())
}

fn write_if_changed_pretty_json<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let content = serde_json::to_string_pretty(value)? + "\n";
    write_if_changed(path, &content)
}

fn default_template_md() -> String {
    let tmpl = r#"# {{name}} — {{one_liner}}

## 1. Name & Purpose
{{purpose}}

## 2. Why it exists (Spec traceability)
{{#each spec_refs}}
- {{this}}
{{/each}}

## 3. Public API surface
{{public_api}}

## 4. How it fits
{{how_it_fits}}

## 5. Build & Test
{{build_test}}

## 6. Contracts
{{contracts}}

## 7. Config & Env
{{config_env}}

## 8. Metrics & Logs
{{metrics_logs}}

## 9. Runbook (Dev)
{{runbook}}

## 10. Status & Owners
- Status: {{status}}
- Owners: {{owner}}

## 11. Changelog pointers
{{changelog}}

## 12. Footnotes
{{footnotes}}

## What this crate is not
{{not_section}}
"#;
    wrap_to_100_cols(tmpl)
}

fn wrap_to_100_cols(s: &str) -> String {
    let mut out = String::new();
    for line in s.lines() {
        out.push_str(&wrap_line(line, 100));
        out.push('\n');
    }
    out
}

fn wrap_line(line: &str, width: usize) -> String {
    if line.len() <= width {
        return line.to_string();
    }
    // naive wrap on spaces
    let mut out = String::new();
    let mut cur = 0usize;
    let chars: Vec<char> = line.chars().collect();
    while cur < chars.len() {
        let mut end = usize::min(cur + width, chars.len());
        if end < chars.len() {
            let mut back = end;
            while back > cur && chars[back - 1] != ' ' {
                back -= 1;
            }
            if back > cur { end = back; }
        }
        let seg: String = chars[cur..end].iter().collect();
        out.push_str(seg.trim_end());
        if end < chars.len() { out.push('\n'); }
        cur = if end == cur { end + 1 } else { end };
    }
    out
}

fn render_readme(repo_root: &Path, e: &ManifestEntry) -> Result<String> {
    let crate_dir = repo_root.join(&e.path);
    let rel_to_root = relative_path(&crate_dir, repo_root);
    let rel_specs = rel_to_root.join(".specs/orchestrator-spec.md");
    let rel_reqs = rel_to_root.join("requirements/index.yaml");

    let one_liner = if e.description.trim().is_empty() {
        format!("{} ({})", e.name, e.role)
    } else {
        e.description.clone()
    };

    let mut spec_section = String::new();
    if e.spec_refs.is_empty() {
        spec_section.push_str("- See spec and requirements for details.\n");
        spec_section.push_str(&format!(
            "  - [{}]({})\n  - [{}]({})\n",
            ".specs/orchestrator-spec.md",
            rel_specs.display(),
            "requirements/index.yaml",
            rel_reqs.display()
        ));
    } else {
        for id in &e.spec_refs {
            let anchor = id.to_lowercase();
            spec_section.push_str(&format!(
                "- {} — [{}]({}#{})\n",
                id,
                ".specs/orchestrator-spec.md",
                rel_specs.display(),
                anchor
            ));
        }
    }

    let public_api = if !e.openapi_refs.is_empty() {
        let mut s = String::new();
        for p in &e.openapi_refs {
            let rel = rel_to_root.join(p);
            s.push_str(&format!("- OpenAPI: [{}]({})\n", p, rel.display()));
        }
        let summary = openapi_summary(repo_root, &e.openapi_refs);
        if !summary.is_empty() {
            s.push_str(&summary);
        }
        s
    } else {
        "- Rust crate API (internal)".to_string()
    };

    let how_it_fits_text = match e.role.as_str() {
        "core" => "- Part of the core orchestrator. Upstream: adapters, Downstream: workers.".to_string(),
        "adapter" => "- Maps engine-native APIs to the orchestrator worker contract.".to_string(),
        "plugin" => "- Policy extension via WASI ABI.".to_string(),
        "contracts" => "- Houses public contracts and schemas.".to_string(),
        "test-harness" => "- Provides test scaffolding for validation suites.".to_string(),
        _ => "- Developer tooling supporting contracts and docs.".to_string(),
    };

    let mermaid = match e.role.as_str() {
        "core" => r#"```mermaid
flowchart LR
  callers[Clients] --> orch[Orchestrator]
  orch --> adapters[Worker Adapters]
  adapters --> engines[Engines]
```"#.to_string(),
        "adapter" => r#"```mermaid
flowchart LR
  orch[Orchestrator] --> adapter[Adapter]
  adapter --> engine[Engine API]
```"#.to_string(),
        "contracts" => r#"```mermaid
flowchart LR
  devs[Developers] --> contracts[Contracts]
  contracts --> tools[Generators]
  contracts --> crates[Crates]
```"#.to_string(),
        "plugin" => r#"```mermaid
flowchart LR
  orch[Orchestrator] --> plugins[Policy Plugins (WASI)]
```"#.to_string(),
        "test-harness" => r#"```mermaid
flowchart LR
  crates[Crates] --> harness[Test Harness]
  harness --> results[Reports]
```"#.to_string(),
        _ => r#"```mermaid
flowchart LR
  devs[Developers] --> tool[Tool]
  tool --> artifacts[Artifacts]
```"#.to_string(),
    };

    let mut build_test = String::new();
    build_test.push_str(&format!("- Workspace fmt/clippy: `cargo fmt --all -- --check` and `cargo clippy --all-targets --all-features -- -D warnings`\n"));
    build_test.push_str(&format!("- Tests for this crate: `cargo test -p {} -- --nocapture`\n", e.name));
    if e.path.starts_with("orchestratord") {
        build_test.push_str("- Provider verify: `cargo test -p orchestratord --test provider_verify -- --nocapture`\n");
    }
    if e.path.starts_with("cli/consumer-tests") {
        build_test.push_str("- CDC consumer tests: `cargo test -p cli-consumer-tests -- --nocapture`\n");
    }
    if e.path.starts_with("contracts/") {
        build_test.push_str("- Regen OpenAPI: `cargo xtask regen-openapi`\n");
        build_test.push_str("- Regen Schema: `cargo xtask regen-schema`\n");
        build_test.push_str("- Extract requirements: `cargo run -p tools-spec-extract --quiet`\n");
    }

    let mut contracts = String::new();
    if !e.openapi_refs.is_empty() {
        contracts.push_str("- OpenAPI:");
        contracts.push('\n');
        for p in &e.openapi_refs {
            let rel = rel_to_root.join(p);
            contracts.push_str(&format!("  - [{}]({})\n", p, rel.display()));
        }
    }
    if !e.schema_refs.is_empty() {
        contracts.push_str("- Schema:");
        contracts.push('\n');
        for p in &e.schema_refs {
            let rel = rel_to_root.join(p);
            contracts.push_str(&format!("  - [{}]({})\n", p, rel.display()));
        }
    }
    if contracts.is_empty() { contracts.push_str("- None\n"); }

    let config_env = match e.role.as_str() {
        "core" => "- See deployment configs and environment variables used by the daemons.".to_string(),
        "adapter" => "- Engine connection endpoints and credentials where applicable.".to_string(),
        "contracts" => "- Schema-focused crate; no runtime env.".to_string(),
        _ => "- Not applicable.".to_string(),
    };

    let metrics_logs = match e.role.as_str() {
        "core" => "- Emits queue depth, latency percentiles, and engine/version labels.".to_string(),
        "adapter" => "- Emits adapter health and request metrics per engine.".to_string(),
        _ => "- Minimal logs.".to_string(),
    };

    let mut runbook = String::new();
    runbook.push_str("- Regenerate artifacts: `cargo xtask regen-openapi && cargo xtask regen-schema`\n");
    runbook.push_str("- Rebuild docs: `cargo run -p tools-readme-index --quiet`\n");

    let status = "alpha".to_string();

    let changelog = "- None".to_string();

    let not_section = match e.role.as_str() {
        "adapter" => "- Not a public API; do not expose engine endpoints directly.".to_string(),
        "core" => "- Not a general-purpose inference server; focuses on orchestration.".to_string(),
        "contracts" => "- Not runtime logic; contracts only.".to_string(),
        _ => "- Not a production service.".to_string(),
    };

    // Hand-curated minimal extras per crate (Step 3) — include in footnotes
    let extras = crate_specific_extras(&e.path);

    // Prepare values for template placeholders
    let how_it_fits = format!("{}\n\n{}", how_it_fits_text, mermaid);
    let footnotes = {
        let mut s = String::new();
        s.push_str(&format!(
            "- Spec: [{}]({})\n- Requirements: [{}]({})\n",
            ".specs/orchestrator-spec.md",
            rel_specs.display(),
            "requirements/index.yaml",
            rel_reqs.display()
        ));
        if !extras.is_empty() {
            s.push_str("\n### Additional Details\n");
            s.push_str(&extras);
        }
        s
    };

    // Load template (single source of truth)
    let template_path = repo_root.join("tools/readme-index/TEMPLATE.md");
    let mut tmpl = if let Ok(t) = fs::read_to_string(&template_path) { t } else { default_template_md() };

    // Replace the spec_refs loop block
    let loop_re = Regex::new(r"(?s)\{\{#each spec_refs\}\}.*?\{\{\/each\}\}")?;
    tmpl = loop_re
        .replace(
            &tmpl,
            spec_section.replace('\\', r"\\").replace('$', r"$$")
        )
        .into_owned();

    // Simple placeholder replacements
    let mut out = tmpl;
    out = out.replace("{{name}}", &e.name);
    out = out.replace("{{one_liner}}", &one_liner);
    out = out.replace("{{purpose}}", &one_liner);
    out = out.replace("{{public_api}}", &public_api);
    out = out.replace("{{how_it_fits}}", &how_it_fits);
    out = out.replace("{{build_test}}", &build_test);
    out = out.replace("{{contracts}}", &contracts);
    out = out.replace("{{config_env}}", &config_env);
    out = out.replace("{{metrics_logs}}", &metrics_logs);
    out = out.replace("{{runbook}}", &runbook);
    out = out.replace("{{status}}", &status);
    out = out.replace("{{owner}}", &e.owner);
    out = out.replace("{{changelog}}", &changelog);
    out = out.replace("{{footnotes}}", &footnotes);
    out = out.replace("{{not_section}}", &not_section);

    Ok(wrap_to_100_cols(&out))
}

fn update_root_readme(_repo_root: &Path, root_readme: &Path, entries: &[ManifestEntry]) -> Result<String> {
    let mut content = if root_readme.exists() {
        fs::read_to_string(root_readme)?
    } else {
        String::from("# Workspace\n\n")
    };

    let begin = "<!-- BEGIN WORKSPACE MAP (AUTO-GENERATED) -->";
    let end = "<!-- END WORKSPACE MAP (AUTO-GENERATED) -->";

    let mut table = String::new();
    table.push_str(begin);
    table.push('\n');
    table.push_str("## Workspace Map\n\n");
    table.push_str("| Path | Crate | Role | Key APIs/Contracts | Tests | Spec Refs |\n");
    table.push_str("|------|------|------|---------------------|-------|-----------|\n");

    let mut one_liners: Vec<(String, String)> = Vec::new();

    for e in entries {
        let crate_readme_rel = Path::new(&e.path).join("README.md");
        let path_link = format!("[`{}/`]({})", e.path, crate_readme_rel.display());
        let api = if !e.openapi_refs.is_empty() {
            "OpenAPI".to_string()
        } else if !e.schema_refs.is_empty() {
            "Schema".to_string()
        } else {
            "—".to_string()
        };
        let tests = if e.tests.is_empty() { "—".to_string() } else { e.tests.join(", ") };
        let spec = if e.spec_refs.is_empty() { "—".to_string() } else { e.spec_refs.join(", ") };
        table.push_str(&format!(
            "| {} | `{}` | {} | {} | {} | {} |\n",
            path_link, e.name, e.role, api, tests, spec
        ));

        let one = if e.description.trim().is_empty() {
            format!("{} ({})", e.name, e.role)
        } else {
            e.description.clone()
        };
        one_liners.push((e.name.clone(), one));
    }

    table.push_str("\n### Glossary\n\n");
    for (name, one) in one_liners {
        table.push_str(&format!("- `{}` — {}\n", name, one));
    }

    table.push_str("\n### Getting Started\n\n");
    table.push_str("- Adapter work: see `worker-adapters/*` crates.\n");
    table.push_str("- Contracts: see `contracts/*`.\n");
    table.push_str("- Core scheduling: see `orchestrator-core/` and `orchestratord/`.\n");

    table.push('\n');
    table.push_str(end);
    table.push('\n');

    // Replace or insert between markers
    let re = Regex::new(&format!(
        r"(?s){}.*?{}(?:\r?\n)?",
        regex::escape(begin),
        regex::escape(end)
    ))?;

    if re.is_match(&content) {
        content = re.replace(&content, table.as_str()).into_owned();
    } else {
        // Append at end with spacing
        if !content.ends_with('\n') { content.push('\n'); }
        content.push_str("\n");
        content.push_str(&table);
    }

    let content = trim_trailing_blank_lines(&content);

    Ok(wrap_to_100_cols(&content))
}
