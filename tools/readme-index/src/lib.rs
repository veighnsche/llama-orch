use anyhow::Result;
use std::fs;

pub mod io_utils;
pub mod openapi_utils;
pub mod path_utils;
pub mod render;
pub mod root_readme;
pub mod types;
pub mod workspace;

pub fn run() -> Result<()> {
    let repo_root = std::env::current_dir()?;

    // 0) Build JSON manifest deterministically
    let members = workspace::read_workspace_members(&repo_root)?;
    let mut entries: Vec<types::ManifestEntry> = Vec::new();
    for member in members {
        let crate_dir = repo_root.join(&member);
        let crate_entry = workspace::build_manifest_entry(&repo_root, &crate_dir)?;
        entries.push(crate_entry);
    }
    entries.sort_by(|a, b| a.path.cmp(&b.path));

    let out_dir = repo_root.join("tools/readme-index");
    fs::create_dir_all(&out_dir)?;
    let manifest_path = out_dir.join("readme_index.json");
    io_utils::write_if_changed_pretty_json(&manifest_path, &entries)?;

    // 1) Ensure TEMPLATE.md exists (single source of truth)
    let template_path = out_dir.join("TEMPLATE.md");
    if !template_path.exists() {
        let template = render::default_template_md();
        io_utils::write_if_changed(&template_path, &template)?;
    }

    // 2) Generate README for every crate with safety guard
    const SENTINEL: &str = "<!-- AUTO-GENERATED: readme-index -->";
    for entry in &entries {
        let mut readme = render::render_readme(&repo_root, entry)?;
        if !readme.starts_with(SENTINEL) {
            readme = format!("{}\n\n{}", SENTINEL, readme);
        }

        let crate_dir = repo_root.join(&entry.path);
        let crate_readme = crate_dir.join("README.md");
        let existing_is_managed = match fs::read_to_string(&crate_readme) {
            Ok(s) => s.contains(SENTINEL),
            Err(_) => false,
        };

        if existing_is_managed || !crate_readme.exists() {
            // Safe to write/overwrite README.md
            io_utils::write_if_changed(&crate_readme, &readme)?;
        } else {
            // Respect hand-authored README: write side-by-side generated file
            let generated = crate_dir.join("README.generated.md");
            io_utils::write_if_changed(&generated, &readme)?;
        }
    }

    // 4) Update Root README: Consolidated Index
    let root_readme_path = repo_root.join("README.md");
    let updated_root = root_readme::update_root_readme(&repo_root, &root_readme_path, &entries)?;
    io_utils::write_if_changed(&root_readme_path, &updated_root)?;

    Ok(())
}
