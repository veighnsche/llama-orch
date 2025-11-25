//! TEAM_531: Turbo dev scope selector (interactive xtask wrapper for `turbo dev`).
//! TEAM_531: Replaces the Python helper with a Rust-based workflow rooted in xtask.

use anyhow::{anyhow, Context, Result};
use inquire::{MultiSelect, Select};
use serde::Deserialize;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

// TEAM_531: Representation of a workspace package discovered from pnpm-workspace.yaml
#[derive(Debug)]
struct WorkspacePackage {
    name: String,
    path: PathBuf,
    has_dev: bool,
}

// TEAM_531: Minimal view of pnpm-workspace.yaml
#[derive(Debug, Deserialize)]
struct PnpmWorkspace {
    packages: Vec<String>,
    #[allow(dead_code)]
    #[serde(default)]
    onlyBuiltDependencies: Vec<String>,
}

// TEAM_531: Dev bundle configuration loaded from dev-bundles.yaml
#[derive(Debug, Deserialize)]
struct DevBundlesFile {
    #[serde(default)]
    bundles: BTreeMap<String, DevBundle>,
}

#[derive(Debug, Deserialize)]
struct DevBundle {
    #[serde(default)]
    description: String,
    #[serde(default)]
    packages: Vec<String>,
}

// TEAM_531: Entry point from xtask main
pub fn run(print_only: bool) -> Result<()> {
    let root = workspace_root()?;
    let packages = load_workspace_packages(&root)?;
    let bundles = load_dev_bundles(&root)?;

    if print_only {
        print_summary(&root, &packages, &bundles);
        return Ok(());
    }

    let dev_pkgs: Vec<&WorkspacePackage> = packages.iter().filter(|p| p.has_dev).collect();
    if dev_pkgs.is_empty() {
        println!("TEAM_531: No workspace packages with a `dev` script found.");
        return Ok(());
    }

    // TEAM_531: Build display options with path for better context
    let options: Vec<String> = dev_pkgs
        .iter()
        .map(|p| {
            let rel_path = p.path.strip_prefix(&root).unwrap_or(&p.path).display();
            format!("{} ({})", p.name, rel_path)
        })
        .collect();
    let names: Vec<String> = dev_pkgs.iter().map(|p| p.name.clone()).collect();

    let default_indices = choose_base_bundle_indices(&names, &bundles)?;

    let mut multi = MultiSelect::new(
        "TEAM_531: Select Turbo dev scopes (packages to run `dev` in):",
        options.clone(),
    )
    .with_help_message("↑/↓ to move, space to select, type to filter, Enter to run, Esc to cancel");

    if !default_indices.is_empty() {
        multi = multi.with_default(&default_indices);
    }

    let selected =
        multi.prompt_skippable().context("TEAM_531: Failed to render MultiSelect prompt")?;

    let selected = match selected {
        Some(v) => v,
        None => {
            println!("TEAM_531: Aborted (no selection). Nothing to run.");
            return Ok(());
        }
    };

    if selected.is_empty() {
        println!("TEAM_531: No packages selected. Nothing to run.");
        return Ok(());
    }

    // TEAM_531: Extract package names from display strings (format: "name (path)")
    let mut filters: Vec<String> =
        selected.iter().filter_map(|s| s.split(" (").next().map(|n| n.to_string())).collect();
    filters.sort();
    filters.dedup();

    let cmd = build_turbo_command(&filters)?;

    println!("TEAM_531: Running:");
    println!("  {}", cmd.join(" "));
    println!();

    let status = Command::new(&cmd[0])
        .args(&cmd[1..])
        .current_dir(&root)
        .status()
        .context("TEAM_531: Failed to execute `turbo dev` command")?;

    if !status.success() {
        return Err(anyhow!("TEAM_531: turbo dev exited with status {:?}", status));
    }

    Ok(())
}

// TEAM_531: Resolve workspace root from xtask's manifest location
fn workspace_root() -> Result<PathBuf> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .context("TEAM_531: Failed to resolve workspace root from CARGO_MANIFEST_DIR")?
        .to_path_buf();
    Ok(root)
}

// TEAM_531: Load workspace packages from pnpm-workspace.yaml and their package.json files
fn load_workspace_packages(root: &Path) -> Result<Vec<WorkspacePackage>> {
    let ws_path = root.join("pnpm-workspace.yaml");
    let text = fs::read_to_string(&ws_path)
        .with_context(|| format!("TEAM_531: Failed to read {}", ws_path.display()))?;

    let cfg: PnpmWorkspace = serde_yaml::from_str(&text)
        .with_context(|| format!("TEAM_531: Failed to parse {}", ws_path.display()))?;

    let mut out = Vec::new();

    for rel in cfg.packages {
        let pkg_dir = root.join(&rel);
        let pkg_json_path = pkg_dir.join("package.json");
        if !pkg_json_path.exists() {
            continue;
        }

        let pkg_text = match fs::read_to_string(&pkg_json_path) {
            Ok(t) => t,
            Err(_) => continue,
        };

        #[derive(Debug, Deserialize)]
        struct PkgConfig {
            name: String,
            #[serde(default)]
            scripts: HashMap<String, String>,
        }

        let pkg: PkgConfig = match serde_json::from_str(&pkg_text) {
            Ok(p) => p,
            Err(_) => continue,
        };

        let has_dev = pkg.scripts.contains_key("dev");
        out.push(WorkspacePackage { name: pkg.name, path: pkg_dir, has_dev });
    }

    // TEAM_531: De-duplicate by package name, keep first occurrence, then sort for stable UI
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out.dedup_by(|a, b| a.name == b.name);

    Ok(out)
}

// TEAM_531: Load dev bundles from dev-bundles.yaml (single source of truth for bundles)
fn load_dev_bundles(root: &Path) -> Result<BTreeMap<String, Vec<String>>> {
    let bundles_path = root.join("dev-bundles.yaml");
    if !bundles_path.exists() {
        return Ok(BTreeMap::new());
    }

    let text = fs::read_to_string(&bundles_path)
        .with_context(|| format!("TEAM_531: Failed to read {}", bundles_path.display()))?;

    let file: DevBundlesFile = serde_yaml::from_str(&text)
        .with_context(|| format!("TEAM_531: Failed to parse {}", bundles_path.display()))?;

    let mut out = BTreeMap::new();
    for (name, bundle) in file.bundles {
        if bundle.packages.is_empty() {
            continue;
        }
        out.insert(name, bundle.packages);
    }

    Ok(out)
}

// TEAM_531: Print discovered packages and bundles (for --print-only)
fn print_summary(
    root: &Path,
    packages: &[WorkspacePackage],
    bundles: &BTreeMap<String, Vec<String>>,
) {
    println!("Workspace packages (TEAM_531):");
    for pkg in packages {
        let marker = if pkg.has_dev { "*" } else { "-" };
        let rel = pkg.path.strip_prefix(root).unwrap_or(&pkg.path).display().to_string();
        println!("  {} {}  ({})", marker, pkg.name, rel);
    }

    println!();
    println!("Bundles from dev-bundles.yaml (TEAM_531):");
    if bundles.is_empty() {
        println!("  (none)");
    } else {
        for (name, filters) in bundles {
            println!("  {}: {}", name, filters.join(", "));
        }
    }
}

// TEAM_531: Ask user which bundle to base the selection on, and convert to default indices
fn choose_base_bundle_indices(
    options: &[String],
    bundles: &BTreeMap<String, Vec<String>>,
) -> Result<Vec<usize>> {
    if bundles.is_empty() {
        return Ok(Vec::new());
    }

    let mut bundle_names: Vec<String> = bundles.keys().cloned().collect();
    bundle_names.sort();

    let mut select_options = Vec::new();
    select_options.push("none".to_string());
    select_options.extend(bundle_names.clone());
    select_options.push("all dev packages".to_string());

    let choice = Select::new(
        "TEAM_531: Choose base bundle from dev-bundles.yaml (then adjust selection):",
        select_options,
    )
    .prompt_skippable()
    .context("TEAM_531: Failed to render bundle Select prompt")?;

    let mut default_indices = Vec::new();

    let Some(choice) = choice else {
        // TEAM_531: User hit Esc at preset selection – treat as abort.
        return Ok(default_indices);
    };

    match choice.as_str() {
        "none" => {
            // TEAM_531: Leave defaults empty
        }
        "all dev packages" => {
            default_indices = (0..options.len()).collect();
        }
        other => {
            if let Some(filters) = bundles.get(other) {
                for (idx, name) in options.iter().enumerate() {
                    if filters.iter().any(|f| f == name) {
                        default_indices.push(idx);
                    }
                }
            }
        }
    }

    Ok(default_indices)
}

// TEAM_531: Build the final `turbo dev` command for the selected filters
fn build_turbo_command(filters: &[String]) -> Result<Vec<String>> {
    if filters.is_empty() {
        return Err(anyhow!("TEAM_531: No filters provided"));
    }

    let concurrency = std::env::var("RBEE_TURBO_CONCURRENCY").unwrap_or_else(|_| "32".to_string());

    let mut cmd = Vec::new();
    cmd.push("turbo".to_string());
    cmd.push("dev".to_string());
    cmd.push("--concurrency".to_string());
    cmd.push(concurrency);

    for name in filters {
        cmd.push(format!("--filter={}", name));
    }

    Ok(cmd)
}
