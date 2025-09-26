use std::path::PathBuf;

/// Ensure the given path has the specified extension; if missing, set it.
pub fn ensure_ext(mut p: PathBuf, ext: &str) -> PathBuf {
    if p.extension().and_then(|e| e.to_str()) != Some(ext) {
        p.set_extension(ext);
    }
    p
}
