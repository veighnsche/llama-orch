use anyhow::{Context, Result};
use std::{io::Write, net::TcpStream, path::PathBuf, process::Command};

use crate::util::repo_root;

fn http_health_probe(host: &str, port: u16) -> std::io::Result<bool> {
    use std::io::Read;
    let addr = format!("{}:{}", host, port);
    let mut stream = TcpStream::connect(addr)?;
    let req = format!("GET /health HTTP/1.1\r\nHost: {}\r\nConnection: close\r\n\r\n", host);
    stream.write_all(req.as_bytes())?;
    let mut buf = String::new();
    stream.read_to_string(&mut buf)?;
    Ok(buf.starts_with("HTTP/1.1 200") || buf.starts_with("HTTP/1.0 200"))
}

fn pid_file_path(pool_id: &str) -> PathBuf {
    let home = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("."));
    home.join(".cache").join("llama-orch").join("run").join(format!("{}.pid", pool_id))
}

pub fn engine_status(config_path: PathBuf, pool_filter: Option<String>) -> Result<()> {
    let root = repo_root()?;
    let path = if config_path.is_relative() { root.join(config_path) } else { config_path };
    let bytes = std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
    let cfg: contracts_config_schema::Config =
        serde_yaml::from_slice(&bytes).with_context(|| format!("parsing {}", path.display()))?;
    let pools: Vec<_> = match pool_filter {
        Some(id) => cfg.pools.iter().filter(|p| p.id == id).collect(),
        None => cfg.pools.iter().collect(),
    };
    if pools.is_empty() {
        println!("no pools matched");
        return Ok(());
    }
    for p in pools {
        let port = p.provisioning.ports.as_ref().and_then(|v| v.first().cloned()).unwrap_or(8080);
        let ok = http_health_probe("127.0.0.1", port).unwrap_or(false);
        let pid_path = pid_file_path(&p.id);
        let pid = std::fs::read_to_string(&pid_path).ok();
        println!(
            "pool={} port={} health={} pid_file={} pid={}",
            p.id,
            port,
            if ok { "up" } else { "down" },
            pid_path.display(),
            pid.as_deref().unwrap_or("-")
        );
    }
    Ok(())
}

pub fn engine_down(config_path: PathBuf, pool_filter: Option<String>) -> Result<()> {
    let root = repo_root()?;
    let path = if config_path.is_relative() { root.join(config_path) } else { config_path };
    let bytes = std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
    let cfg: contracts_config_schema::Config =
        serde_yaml::from_slice(&bytes).with_context(|| format!("parsing {}", path.display()))?;
    let pools: Vec<_> = match pool_filter {
        Some(id) => cfg.pools.into_iter().filter(|p| p.id == id).collect(),
        None => cfg.pools,
    };
    if pools.is_empty() {
        println!("no pools matched");
        return Ok(());
    }
    for p in pools.iter() {
        let pid_path = pid_file_path(&p.id);
        match std::fs::read_to_string(&pid_path) {
            Ok(s) => {
                let pid = s.trim();
                println!("stopping pool={} pid={}", p.id, pid);
                let status = Command::new("kill").arg(pid).status().context("kill")?;
                if status.success() {
                    let _ = std::fs::remove_file(&pid_path);
                    println!("stopped {}", p.id);
                } else {
                    println!("kill failed for {} (status={})", p.id, status);
                }
            }
            Err(_) => println!("no pid file for {} at {}", p.id, pid_path.display()),
        }
    }
    Ok(())
}

pub fn engine_up(_config_path: PathBuf, _pool_filter: Option<String>) -> Result<()> {
    // TODO: Implement engine provisioning
    println!("engine:up not yet implemented");
    Ok(())
}

pub fn engine_plan(_config_path: PathBuf, _pool_filter: Option<String>) -> Result<()> {
    // TODO: Implement engine planning
    println!("engine:plan not yet implemented");
    Ok(())
}
