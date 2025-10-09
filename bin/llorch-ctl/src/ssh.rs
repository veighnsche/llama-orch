//! SSH client wrapper using ssh2 crate
//!
//! Created by: TEAM-022
//! Refactored by: TEAM-022 (using ssh2 instead of Command)

use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use ssh2::Session;
use std::io::Read;
use std::net::TcpStream;
use std::path::Path;

/// Get SSH session for a host
fn get_session(host: &str) -> Result<Session> {
    // Connect to SSH server
    let tcp = TcpStream::connect(format!("{}:22", host))
        .with_context(|| format!("Failed to connect to {}", host))?;

    let mut sess = Session::new()?;
    sess.set_tcp_stream(tcp);
    sess.handshake()?;

    // Try agent authentication first
    sess.userauth_agent(&std::env::var("USER").unwrap_or_else(|_| "root".to_string()))?;

    if !sess.authenticated() {
        anyhow::bail!("SSH authentication failed for {}", host);
    }

    Ok(sess)
}

/// Execute a command on a remote host via SSH (returns output)
#[allow(dead_code)] // TEAM-022: May be used in future
pub fn execute_remote_command(host: &str, command: &str) -> Result<String> {
    let sess = get_session(host)?;

    let mut channel = sess.channel_session()?;
    channel.exec(command)?;

    let mut output = String::new();
    channel.read_to_string(&mut output)?;

    channel.wait_close()?;
    let exit_status = channel.exit_status()?;

    if exit_status != 0 {
        let mut stderr = String::new();
        channel.stderr().read_to_string(&mut stderr)?;
        anyhow::bail!("SSH command failed (exit {}): {}", exit_status, stderr.trim());
    }

    Ok(output)
}

/// Execute a command on a remote host and stream output to stdout
pub fn execute_remote_command_streaming(host: &str, command: &str) -> Result<()> {
    let spinner = ProgressBar::new_spinner();
    spinner.set_style(ProgressStyle::default_spinner().template("{spinner:.cyan} {msg}").unwrap());
    spinner.set_message(format!("→ SSH to {}", host));

    let sess = get_session(host)?;

    let mut channel = sess.channel_session()?;
    channel.exec(command)?;

    spinner.finish_and_clear();

    // Stream stdout
    let mut buffer = [0; 8192];
    loop {
        match channel.read(&mut buffer) {
            Ok(0) => break,
            Ok(n) => {
                print!("{}", String::from_utf8_lossy(&buffer[..n]));
            }
            Err(e) => return Err(e.into()),
        }
    }

    // Stream stderr
    loop {
        match channel.stderr().read(&mut buffer) {
            Ok(0) => break,
            Ok(n) => {
                eprint!("{}", String::from_utf8_lossy(&buffer[..n]));
            }
            Err(e) => return Err(e.into()),
        }
    }

    channel.wait_close()?;
    let exit_status = channel.exit_status()?;

    if exit_status != 0 {
        anyhow::bail!("SSH command failed (exit {})", exit_status);
    }

    Ok(())
}

/// Upload a file to remote host
#[allow(dead_code)]
pub fn upload_file(host: &str, local_path: &Path, remote_path: &Path) -> Result<()> {
    let sess = get_session(host)?;

    let local_file = std::fs::read(local_path)?;
    let mut remote_file = sess.scp_send(remote_path, 0o644, local_file.len() as u64, None)?;

    std::io::Write::write_all(&mut remote_file, &local_file)?;

    Ok(())
}
