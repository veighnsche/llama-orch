use anyhow::Result;
use serde::Serialize;
use std::fs;
use std::io::Write;
use std::path::Path;

pub fn trim_trailing_blank_lines(s: &str) -> String {
    // Remove any trailing blank lines and ensure a single trailing newline
    let mut lines: Vec<&str> = s.split('\n').collect();
    while matches!(lines.last(), Some(l) if l.trim().is_empty()) {
        lines.pop();
    }
    let mut out = lines.join("\n");
    out.push('\n');
    out
}

pub fn write_if_changed(path: &Path, content: &str) -> Result<()> {
    let new_bytes = content.as_bytes();
    if let Ok(old) = fs::read(path) {
        if old == new_bytes {
            return Ok(());
        }
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut f = fs::File::create(path)?;
    f.write_all(new_bytes)?;
    Ok(())
}

pub fn write_if_changed_pretty_json<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let content = serde_json::to_string_pretty(value)? + "\n";
    write_if_changed(path, &content)
}
