use anyhow::{anyhow, Context, Result};
use regex::Regex;
use serde::Serialize;
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

#[derive(Serialize)]
struct RequirementIndex {
    schema_version: u32,
    source: String,
    notes: String,
    requirements: BTreeMap<String, ReqEntry>,
}

#[derive(Serialize)]
struct ReqEntry {
    title: String,
    section: String,
    level: String, // must | should | may | info
    links: Vec<String>,
}

fn main() -> Result<()> {
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .ok_or_else(|| anyhow!("failed to locate repo root"))?
        .to_path_buf();

    let specs_dir = repo_root.join(".specs");
    let mut spec_files: Vec<PathBuf> = Vec::new();
    for entry in
        fs::read_dir(&specs_dir).with_context(|| format!("reading {}", specs_dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("md") {
            spec_files.push(path);
        }
    }
    spec_files.sort();

    let orch_spec = specs_dir.join("orchestrator-spec.md");
    let mut compliance_sections: Vec<String> = Vec::new();

    let out_dir = repo_root.join("requirements");
    fs::create_dir_all(&out_dir).context("creating requirements/ directory")?;

    for spec_path in &spec_files {
        let spec_rel =
            path_relative(&repo_root, spec_path).unwrap_or_else(|| spec_path.display().to_string());
        let spec_txt = fs::read_to_string(spec_path)
            .with_context(|| format!("reading {}", spec_path.display()))?;

        let index = extract_from_spec(&spec_rel, &spec_txt)?;

        // Determine output file name
        let out_name = requirements_yaml_name(
            spec_path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("spec.yaml"),
        );
        let out_path = out_dir.join(out_name);
        let yaml = serde_yaml::to_string(&index)?;
        write_if_changed(&out_path, &yaml)?;

        // Back-compat: also write index.yaml for orchestrator-spec
        if spec_path == &orch_spec {
            let idx_path = out_dir.join("index.yaml");
            write_if_changed(&idx_path, &yaml)?;
        }

        // Compliance section
        let mut lines = Vec::new();
        lines.push(format!("### {}", spec_rel));
        lines.push(format!("Total requirements: {}", index.requirements.len()));
        for (id, req) in &index.requirements {
            let link = req.links.first().map(String::as_str).unwrap_or("");
            if link.is_empty() {
                lines.push(format!(
                    "- {} — {} (section: {}, level: {})",
                    id, req.title, req.section, req.level
                ));
            } else {
                lines.push(format!(
                    "- {} — {} (section: {}, level: {}) — link: {}",
                    id, req.title, req.section, req.level, link
                ));
            }
        }
        compliance_sections.push(lines.join("\n"));
    }

    // Emit aggregated COMPLIANCE.md deterministically
    let mut comp_lines = Vec::new();
    comp_lines.push("# COMPLIANCE — Requirements Coverage".to_string());
    comp_lines.push("".to_string());
    for sect in compliance_sections {
        comp_lines.push(sect);
        comp_lines.push("".to_string());
    }
    let comp_path = repo_root.join("COMPLIANCE.md");
    let compliance = comp_lines.join("\n");
    write_if_changed(&comp_path, &compliance)?;

    Ok(())
}

fn extract_from_spec(spec_rel: &str, spec: &str) -> Result<RequirementIndex> {
    // Support both ORCH-##### and OC-AREA-#### style identifiers
    let id_orch = Regex::new(r"ORCH-[0-9]{3,5}").with_context(|| "compile ORCH id regex")?;
    let id_oc = Regex::new(r"OC-[A-Z0-9-]+-[0-9]{3,5}").with_context(|| "compile OC id regex")?;
    let section_re =
        Regex::new(r"^##+\s+(?P<name>.+)$").with_context(|| "compile section heading regex")?;
    let mut requirements: BTreeMap<String, ReqEntry> = BTreeMap::new();

    let mut current_section = String::new();
    for line in spec.lines() {
        if let Some(cap) = section_re.captures(line) {
            current_section = cap["name"].trim().to_string();
        }
        for m in id_orch.find_iter(line).chain(id_oc.find_iter(line)) {
            let id = m.as_str().to_string();
            let lower = line.to_lowercase();
            let level = if lower.contains("must") {
                "must"
            } else if lower.contains("should") {
                "should"
            } else if lower.contains("may") {
                "may"
            } else {
                "info"
            };
            let mut title = line.replace(&id, "");
            title = title.trim().trim_start_matches('*').trim().to_string();
            if title.len() > 160 {
                title.truncate(160);
            }
            requirements.entry(id).or_insert(ReqEntry {
                title,
                section: current_section.clone(),
                level: level.to_string(),
                links: vec![format!(
                    "{}#{}",
                    spec_rel,
                    anchor_from_section(&current_section)
                )],
            });
        }
    }

    let notes = if requirements.is_empty() {
        "No requirement IDs found; add stable IDs like ORCH-XXXX or OC-AREA-XXXX to normative requirements.".into()
    } else {
        format!("Extracted from {}", spec_rel)
    };

    Ok(RequirementIndex {
        schema_version: 1,
        source: spec_rel.into(),
        notes,
        requirements,
    })
}

fn requirements_yaml_name(spec_file: &str) -> String {
    // Map spec filenames to requirement yaml names (derived from crate package names)
    let stem = Path::new(spec_file)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("spec");

    // If stem is prefixed with NN- keep original stem for fallback filename, but strip for matching
    let (base_for_match, fallback_name) = if stem.len() > 3 && stem.as_bytes()[0].is_ascii_digit()
        && stem.as_bytes()[1].is_ascii_digit() && stem.as_bytes()[2] == b'-'
    {
        (&stem[3..], stem)
    } else {
        (stem, stem)
    };

    let mapped = match base_for_match {
        // historical name retained for back-compat (not used anymore)
        "orchestrator-spec" => Some("index.yaml".to_string()),
        "orchestrator-core" => Some("orchestrator-core.yaml".to_string()),
        "orchestratord" => Some("orchestratord.yaml".to_string()),
        "pool-managerd" => Some("pool-managerd.yaml".to_string()),
        "plugins-policy-host" => Some("plugins-policy-host.yaml".to_string()),
        "plugins-policy-sdk" => Some("plugins-policy-sdk.yaml".to_string()),
        "config-schema" => Some("contracts-config-schema.yaml".to_string()),
        "determinism-suite" => Some("test-harness-determinism-suite.yaml".to_string()),
        "metrics-contract" => Some("test-harness-metrics-contract.yaml".to_string()),
        "worker-adapters-llamacpp-http" => Some("worker-adapters-llamacpp-http.yaml".to_string()),
        "worker-adapters-vllm-http" => Some("worker-adapters-vllm-http.yaml".to_string()),
        "worker-adapters-tgi-http" => Some("worker-adapters-tgi-http.yaml".to_string()),
        "worker-adapters-triton" => Some("worker-adapters-triton.yaml".to_string()),
        _ => None,
    };

    mapped.unwrap_or_else(|| format!("{}.yaml", fallback_name))
}

fn path_relative(root: &Path, path: &Path) -> Option<String> {
    let root = root.canonicalize().ok()?;
    let path = path.canonicalize().ok()?;
    path.strip_prefix(&root)
        .ok()
        .map(|p| p.to_string_lossy().to_string())
}

fn write_if_changed(path: &Path, contents: &str) -> Result<()> {
    let need = match fs::read_to_string(path) {
        Ok(old) => old != contents,
        Err(_) => true,
    };
    if need {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, contents).with_context(|| format!("writing {}", path.display()))?;
        println!("wrote {}", path.display());
    } else {
        println!("unchanged {}", path.display());
    }
    Ok(())
}

fn anchor_from_section(section: &str) -> String {
    // GitHub-style anchor: lowercased, spaces -> -, remove non-alnum/-
    let lower = section.to_lowercase();
    let mut s = String::with_capacity(lower.len());
    for ch in lower.chars() {
        if ch.is_ascii_alphanumeric() {
            s.push(ch);
        } else if ch.is_whitespace() || ch == '/' || ch == '&' {
            // collapse common separators to '-'
            s.push('-');
        } // else skip
    }
    // collapse multiple '-'
    let mut collapsed = String::new();
    let mut prev_dash = false;
    for ch in s.chars() {
        if ch == '-' {
            if !prev_dash {
                collapsed.push('-');
                prev_dash = true;
            }
        } else {
            collapsed.push(ch);
            prev_dash = false;
        }
    }
    collapsed.trim_matches('-').to_string()
}
