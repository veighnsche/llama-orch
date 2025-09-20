use anyhow::{anyhow, Result};
use std::path::{Path, PathBuf};
use std::process::Command;

pub fn default_cache_dir(engine: &str) -> PathBuf {
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    home.join(".cache").join("llama-orch").join(engine)
}

pub fn default_models_cache() -> PathBuf {
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    home.join(".cache").join("models")
}

pub fn default_run_dir() -> PathBuf {
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    home.join(".cache").join("llama-orch").join("run")
}

pub fn cmd(bin: &str) -> Command {
    Command::new(bin)
}

pub fn cmd_in(dir: &Path, bin: &str) -> Command {
    let mut c = Command::new(bin);
    c.current_dir(dir);
    c
}

pub fn ok_status(status: std::process::ExitStatus) -> Result<()> {
    if status.success() {
        Ok(())
    } else {
        Err(anyhow!("command failed with status {}", status))
    }
}

pub fn resolve_model_path(model_ref: &str, cache_dir: &Path) -> PathBuf {
    if model_ref.starts_with("hf:") {
        if let Some((_repo, file)) = parse_hf_ref(model_ref) {
            return cache_dir.join(file);
        }
    }
    if model_ref.starts_with("/") || model_ref.starts_with("./") {
        return PathBuf::from(model_ref);
    }
    cache_dir.join(model_ref)
}

pub fn parse_hf_ref(input: &str) -> Option<(String, String)> {
    // hf:owner/repo/file
    let s = input.strip_prefix("hf:")?;
    let mut parts = s.splitn(3, '/');
    let owner = parts.next()?.to_string();
    let repo = parts.next()?.to_string();
    let file = parts.next()?.to_string();
    Some((format!("{}/{}", owner, repo), file))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_hf_ref_ok() {
        let (repo, file) = parse_hf_ref("hf:owner/repo/path/to/file.gguf").expect("parse");
        assert_eq!(repo, "owner/repo");
        assert_eq!(file, "path/to/file.gguf");
    }

    #[test]
    fn resolve_model_hf_joins_cache_dir() {
        let cache = PathBuf::from("/tmp/cache");
        let p = resolve_model_path("hf:owner/repo/file.bin", &cache);
        assert_eq!(p, cache.join("file.bin"));
    }

    #[test]
    fn resolve_model_abs_passes_through() {
        let cache = PathBuf::from("/tmp/cache");
        let p = resolve_model_path("/abs/model.gguf", &cache);
        assert_eq!(p, PathBuf::from("/abs/model.gguf"));
    }

    #[test]
    fn resolve_model_rel_joins_cache_dir() {
        let cache = PathBuf::from("/tmp/cache");
        let p = resolve_model_path("rel/model.gguf", &cache);
        assert_eq!(p, cache.join("rel/model.gguf"));
    }
}
