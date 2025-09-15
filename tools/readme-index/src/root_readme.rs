use crate::io_utils::trim_trailing_blank_lines;
use crate::render::wrap_to_100_cols;
use crate::types::ManifestEntry;
use regex::Regex;
use std::fs;
use std::path::Path;

pub fn update_root_readme(
    _repo_root: &Path,
    root_readme: &Path,
    entries: &[ManifestEntry],
) -> anyhow::Result<String> {
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
        let tests = if e.tests.is_empty() {
            "—".to_string()
        } else {
            e.tests.join(", ")
        };
        let spec = if e.spec_refs.is_empty() {
            "—".to_string()
        } else {
            e.spec_refs.join(", ")
        };
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

    let re = Regex::new(&format!(
        r"(?s){}.*?{}(?:\r?\n)?",
        regex::escape(begin),
        regex::escape(end)
    ))?;

    if re.is_match(&content) {
        content = re.replace(&content, table.as_str()).into_owned();
    } else {
        if !content.ends_with('\n') {
            content.push('\n');
        }
        content.push('\n');
        content.push_str(&table);
    }

    let content = trim_trailing_blank_lines(&content);

    Ok(wrap_to_100_cols(&content))
}
