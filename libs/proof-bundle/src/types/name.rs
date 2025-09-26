use anyhow::{anyhow, Result};
use std::path::{Component, Path};

/// Validate and sanitize a relative artifact path.
/// Rules:
/// - Must be relative (not absolute)
/// - No `..` components
/// - No empty components
/// - Allow normal segments (alphanumeric, dash, underscore, dot) and `/` as separator
/// Returns the original string if valid; otherwise returns an error.
pub fn sanitize_name<S: AsRef<str>>(name: S) -> Result<String> {
    let s = name.as_ref();
    if s.is_empty() {
        return Err(anyhow!("name cannot be empty"));
    }
    let p = Path::new(s);
    if p.is_absolute() {
        return Err(anyhow!("absolute paths are not allowed"));
    }
    for comp in p.components() {
        match comp {
            Component::CurDir => return Err(anyhow!("`.` component not allowed")),
            Component::ParentDir => return Err(anyhow!("`..` component not allowed")),
            Component::RootDir | Component::Prefix(_) => return Err(anyhow!("invalid component")),
            Component::Normal(seg) => {
                if seg.to_string_lossy().is_empty() {
                    return Err(anyhow!("empty path segment"));
                }
            }
        }
    }
    Ok(s.to_string())
}
