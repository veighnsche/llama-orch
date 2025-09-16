use crate::openapi_utils::openapi_summary;
use crate::path_utils::relative_path;
use crate::types::ManifestEntry;
use crate::workspace::crate_specific_extras;
use regex::Regex;
use std::fs;
use std::path::Path;

pub fn default_template_md() -> String {
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

pub fn wrap_to_100_cols(s: &str) -> String {
    let mut out = String::new();
    for line in s.lines() {
        out.push_str(&wrap_line(line, 100));
        out.push('\n');
    }
    out
}

pub fn wrap_line(line: &str, width: usize) -> String {
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
            if back > cur {
                end = back;
            }
        }
        let seg: String = chars[cur..end].iter().collect();
        out.push_str(seg.trim_end());
        if end < chars.len() {
            out.push('\n');
        }
        cur = if end == cur { end + 1 } else { end };
    }
    out
}

pub fn render_readme(repo_root: &Path, e: &ManifestEntry) -> anyhow::Result<String> {
    let crate_dir = repo_root.join(&e.path);
    let rel_to_root = relative_path(&crate_dir, repo_root);
    let rel_specs = rel_to_root.join(".specs/00_llama-orch.md");
    let rel_reqs = rel_to_root.join("requirements/00_llama-orch.yaml");

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
            ".specs/00_llama-orch.md",
            rel_specs.display(),
            "requirements/00_llama-orch.yaml",
            rel_reqs.display()
        ));
    } else {
        for id in &e.spec_refs {
            let anchor = id.to_lowercase();
            spec_section.push_str(&format!(
                "- {} — [{}]({}#{})\n",
                id,
                ".specs/00_llama-orch.md",
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
        "core" => {
            "- Part of the core orchestrator. Upstream: adapters, Downstream: workers.".to_string()
        }
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
```"#
            .to_string(),
        "adapter" => r#"```mermaid
flowchart LR
  orch[Orchestrator] --> adapter[Adapter]
  adapter --> engine[Engine API]
```"#
            .to_string(),
        "contracts" => r#"```mermaid
flowchart LR
  devs[Developers] --> contracts[Contracts]
  contracts --> tools[Generators]
  contracts --> crates[Crates]
```"#
            .to_string(),
        "plugin" => r#"```mermaid
flowchart LR
  orch[Orchestrator] --> plugins[Policy Plugins (WASI)]
```"#
            .to_string(),
        "test-harness" => r#"```mermaid
flowchart LR
  crates[Crates] --> harness[Test Harness]
  harness --> results[Reports]
```"#
            .to_string(),
        _ => r#"```mermaid
flowchart LR
  devs[Developers] --> tool[Tool]
  tool --> artifacts[Artifacts]
```"#
            .to_string(),
    };

    let mut build_test = String::new();
    build_test.push_str("- Workspace fmt/clippy: `cargo fmt --all -- --check` and `cargo clippy --all-targets --all-features -- -D warnings`\n");
    build_test.push_str(&format!(
        "- Tests for this crate: `cargo test -p {} -- --nocapture`\n",
        e.name
    ));
    if e.path.starts_with("orchestratord") {
        build_test.push_str("- Provider verify: `cargo test -p orchestratord --test provider_verify -- --nocapture`\n");
    }
    if e.path.starts_with("cli/consumer-tests") {
        build_test
            .push_str("- CDC consumer tests: `cargo test -p cli-consumer-tests -- --nocapture`\n");
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
    if contracts.is_empty() {
        contracts.push_str("- None\n");
    }

    let config_env = match e.role.as_str() {
        "core" => {
            "- See deployment configs and environment variables used by the daemons.".to_string()
        }
        "adapter" => "- Engine connection endpoints and credentials where applicable.".to_string(),
        "contracts" => "- Schema-focused crate; no runtime env.".to_string(),
        _ => "- Not applicable.".to_string(),
    };

    let metrics_logs = match e.role.as_str() {
        "core" => {
            "- Emits queue depth, latency percentiles, and engine/version labels.".to_string()
        }
        "adapter" => "- Emits adapter health and request metrics per engine.".to_string(),
        _ => "- Minimal logs.".to_string(),
    };

    let mut runbook = String::new();
    runbook.push_str(
        "- Regenerate artifacts: `cargo xtask regen-openapi && cargo xtask regen-schema`\n",
    );
    runbook.push_str("- Rebuild docs: `cargo run -p tools-readme-index --quiet`\n");

    let status = "alpha".to_string();
    let changelog = "- None".to_string();

    let not_section = match e.role.as_str() {
        "adapter" => "- Not a public API; do not expose engine endpoints directly.".to_string(),
        "core" => "- Not a general-purpose inference server; focuses on orchestration.".to_string(),
        "contracts" => "- Not runtime logic; contracts only.".to_string(),
        _ => "- Not a production service.".to_string(),
    };

    let extras = crate_specific_extras(&e.path);

    let how_it_fits = format!("{}\n\n{}", how_it_fits_text, mermaid);
    let footnotes = {
        let mut s = String::new();
        s.push_str(&format!(
            "- Spec: [{}]({})\n- Requirements: [{}]({})\n",
            ".specs/00_llama-orch.md",
            rel_specs.display(),
            "requirements/00_llama-orch.yaml",
            rel_reqs.display()
        ));
        if !extras.is_empty() {
            s.push_str("\n### Additional Details\n");
            s.push_str(&extras);
        }
        s
    };

    let template_path = repo_root.join("tools/readme-index/TEMPLATE.md");
    let mut tmpl = if let Ok(t) = fs::read_to_string(&template_path) {
        t
    } else {
        default_template_md()
    };

    let loop_re = Regex::new(r"(?s)\{\{#each spec_refs\}\}.*?\{\{\/each\}\}")?;
    tmpl = loop_re
        .replace(&tmpl, spec_section.replace('\\', r"\\").replace('$', r"$$"))
        .into_owned();

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
