use anyhow::{anyhow, Context, Result};
use regex::Regex;
use serde::Serialize;
use std::{collections::BTreeMap, fs, path::PathBuf};

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
    let spec_path = repo_root.join(".specs/orchestrator-spec.md");
    let spec = fs::read_to_string(&spec_path)
        .with_context(|| format!("reading {}", spec_path.display()))?;

    // Extract ORCH IDs and nearest section headings.
    let id_re = Regex::new(r"ORCH-[0-9]{3,5}").unwrap();
    let section_re = Regex::new(r"^##+\s+(?P<name>.+)$").unwrap();
    let mut requirements: BTreeMap<String, ReqEntry> = BTreeMap::new();

    let mut current_section = String::new();
    for line in spec.lines() {
        if let Some(cap) = section_re.captures(line) {
            current_section = cap["name"].trim().to_string();
        }
        for m in id_re.find_iter(line) {
            let id = m.as_str().to_string();
            // Determine normative level heuristically from the line.
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
            // Title is the line trimmed to a reasonable length without the ID.
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
                    ".specs/orchestrator-spec.md",
                    anchor_from_section(&current_section)
                )],
            });
        }
    }

    let index = RequirementIndex {
        schema_version: 1,
        source: ".specs/orchestrator-spec.md".into(),
        notes: if requirements.is_empty() {
            "No ORCH-IDs found in spec; add stable ORCH-XXXX anchors to normative requirements."
                .into()
        } else {
            "Extracted from orchestrator-spec.md".into()
        },
        requirements,
    };

    let out_dir = repo_root.join("requirements");
    fs::create_dir_all(&out_dir).context("creating requirements/ directory")?;
    let out_path = out_dir.join("index.yaml");
    let yaml = serde_yaml::to_string(&index)?;
    // Write deterministically (no timestamps), overwrite if unchanged to keep mtime stable where possible.
    let write_needed = match fs::read_to_string(&out_path) {
        Ok(existing) => existing != yaml,
        Err(_) => true,
    };
    if write_needed {
        fs::write(&out_path, yaml).with_context(|| format!("writing {}", out_path.display()))?;
        println!("wrote {}", out_path.display());
    } else {
        println!("unchanged {}", out_path.display());
    }

    // Emit COMPLIANCE.md summary deterministically
    let mut lines = Vec::new();
    lines.push("# COMPLIANCE — Requirements Coverage\n".to_string());
    lines.push(format!("Source: {}", index.source));
    lines.push(format!(
        "Total requirements: {}\n",
        index.requirements.len()
    ));
    lines.push("## Index\n".to_string());
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
    let compliance = lines.join("\n");
    let comp_path = repo_root.join("COMPLIANCE.md");
    let comp_needed = match fs::read_to_string(&comp_path) {
        Ok(existing) => existing != compliance,
        Err(_) => true,
    };
    if comp_needed {
        fs::write(&comp_path, compliance)?;
        println!("wrote {}", comp_path.display());
    } else {
        println!("unchanged {}", comp_path.display());
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
