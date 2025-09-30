use anyhow::{anyhow, Result};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

pub fn default_cache_dir(engine: &str) -> PathBuf {
    let home = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("."));
    home.join(".cache").join("llama-orch").join(engine)
}

pub fn default_models_cache() -> PathBuf {
    let home = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("."));
    home.join(".cache").join("models")
}

pub fn default_run_dir() -> PathBuf {
    let home = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("."));
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

/// Ensure a single flag is present in the args vector; if absent, push it.
pub fn ensure_flag(args: &mut Vec<String>, flag: &str) {
    if !args.iter().any(|f| f == flag) {
        args.push(flag.to_string());
    }
}

/// Ensure a flag-value pair is present in the args vector; if absent, append both.
pub fn ensure_flag_pair(args: &mut Vec<String>, flag: &str, value: &str) {
    let mut i = 0usize;
    while i < args.len() {
        if args[i] == flag {
            // If pair exists but missing value, fix it
            if i + 1 >= args.len() {
                args.push(value.to_string());
            }
            return;
        }
        i += 1;
    }
    args.push(flag.to_string());
    args.push(value.to_string());
}

/// Pick a listen port starting at preferred, scanning upward for a free one.
pub fn select_listen_port(preferred: u16) -> u16 {
    let start = if preferred == 0 { 8080 } else { preferred };
    for p in start..(start + 200) {
        let addr = format!("127.0.0.1:{}", p);
        if let Ok(listener) = TcpListener::bind(&addr) {
            // Found a free port; drop the listener and return this port.
            drop(listener);
            return p;
        }
    }
    // As a last resort, let the OS pick a free port (unlikely path)
    TcpListener::bind("127.0.0.1:0")
        .ok()
        .and_then(|l| l.local_addr().ok().map(|a| a.port()))
        .unwrap_or(start)
}

/// Simple HTTP GET probe; true when 200 OK is returned.
pub fn http_ok(host: &str, port: u16, path: &str) -> Result<bool> {
    let addr = format!("{}:{}", host, port);
    let mut stream = TcpStream::connect(addr)?;
    let req = format!("GET {} HTTP/1.1\r\nHost: {}\r\nConnection: close\r\n\r\n", path, host);
    stream.write_all(req.as_bytes())?;
    let mut buf = [0u8; 64];
    let n = stream.read(&mut buf)?;
    let head = String::from_utf8_lossy(&buf[..n]);
    Ok(head.starts_with("HTTP/1.1 200") || head.starts_with("HTTP/1.0 200"))
}

/// Wait for /health to return 200 within the given timeout.
/// Treat 503 as transient (engine loading model); any other status counts as not-ready for now.
pub fn wait_for_health(host: &str, port: u16, timeout: Duration) -> Result<()> {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        match http_status(host, port, "/health") {
            Ok(200) => return Ok(()),
            Ok(503) => {
                // transient; keep waiting
            }
            Ok(_) => {
                // non-healthy; keep probing
            }
            Err(_) => {
                // connection error; keep probing
            }
        }
        std::thread::sleep(Duration::from_millis(750));
    }
    Err(anyhow!("timeout waiting for health at http://{}:{}/health", host, port))
}

fn http_status(host: &str, port: u16, path: &str) -> Result<u16> {
    let addr = format!("{}:{}", host, port);
    let mut stream = TcpStream::connect(addr)?;
    let req = format!("GET {} HTTP/1.1\r\nHost: {}\r\nConnection: close\r\n\r\n", path, host);
    stream.write_all(req.as_bytes())?;
    let mut line = String::new();
    // Read the status line only
    let mut reader = std::io::BufReader::new(stream);
    use std::io::BufRead;
    reader.read_line(&mut line)?;
    let code = line
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse::<u16>().ok())
        .ok_or_else(|| anyhow!("failed to parse HTTP status line: {}", line.trim()))?;
    Ok(code)
}

/// Write the orchestrator handoff JSON to .runtime/engines/<filename> and return the absolute path.
pub fn write_handoff_file(filename: &str, payload: &serde_json::Value) -> Result<PathBuf> {
    let dir = PathBuf::from(".runtime").join("engines");
    std::fs::create_dir_all(&dir)?;
    let path = dir.join(filename);
    std::fs::write(&path, serde_json::to_vec_pretty(payload)?)?;
    Ok(path.canonicalize().unwrap_or(path))
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

    #[test]
    fn select_listen_port_skips_in_use_port() {
        // Bind an ephemeral port to simulate collision
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        let in_use_port = listener.local_addr().unwrap().port();
        // Now ask util to select a port starting from the in-use port
        let picked = select_listen_port(in_use_port);
        assert_ne!(picked, in_use_port, "should avoid the already bound port");
        // Cleanup
        drop(listener);
    }
}
