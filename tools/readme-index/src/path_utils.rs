use anyhow::Result;
use std::path::{Component, Path, PathBuf};

pub fn path_relative_to(base: &Path, path: &Path) -> Result<String> {
    let base = base.canonicalize()?;
    let path = path.canonicalize()?;
    let rel = relative_path(&base, &path);
    Ok(rel.to_string_lossy().to_string())
}

pub fn relative_path(from: &Path, to: &Path) -> PathBuf {
    let from_components: Vec<&str> = from
        .components()
        .filter_map(|c| match c {
            Component::Normal(s) => s.to_str(),
            _ => None,
        })
        .collect();
    let to_components: Vec<&str> = to
        .components()
        .filter_map(|c| match c {
            Component::Normal(s) => s.to_str(),
            _ => None,
        })
        .collect();
    let mut i = 0usize;
    while i < from_components.len()
        && i < to_components.len()
        && from_components[i] == to_components[i]
    {
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
