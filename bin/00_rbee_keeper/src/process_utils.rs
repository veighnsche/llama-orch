// TEAM-301: Process output streaming utilities
//
// Provides helpers for spawning processes with real-time stdout/stderr display.
// Used by keeper to show daemon startup output to the user.

use anyhow::Result;
use std::process::Stdio;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};

/// Spawn command with stdout/stderr streaming to terminal
///
/// This function spawns a command and streams its output in real-time to the
/// keeper's terminal. This allows users to see daemon startup messages.
///
/// # Arguments
/// * `command` - The command to spawn (must have stdout/stderr configured)
///
/// # Returns
/// * Child process handle (stdout/stderr already consumed by background tasks)
///
/// # Example
/// ```rust,ignore
/// let mut command = Command::new("queen-rbee");
/// command.arg("--port").arg("8500");
/// let child = spawn_with_output_streaming(command).await?;
/// ```
pub async fn spawn_with_output_streaming(mut command: Command) -> Result<Child> {
    command.stdout(Stdio::piped());
    command.stderr(Stdio::piped());

    let mut child = command.spawn()?;

    // Stream stdout to terminal
    if let Some(stdout) = child.stdout.take() {
        tokio::spawn(async move {
            let reader = BufReader::new(stdout);
            let mut lines = reader.lines();
            while let Ok(Some(line)) = lines.next_line().await {
                println!("{}", line);
            }
        });
    }

    // Stream stderr to terminal
    if let Some(stderr) = child.stderr.take() {
        tokio::spawn(async move {
            let reader = BufReader::new(stderr);
            let mut lines = reader.lines();
            while let Ok(Some(line)) = lines.next_line().await {
                eprintln!("{}", line);
            }
        });
    }

    Ok(child)
}

/// Stream output from an already-spawned child process
///
/// Use this when you have a Child process with piped stdout/stderr that you
/// want to stream to the terminal.
///
/// # Arguments
/// * `child` - Mutable reference to the child process
pub fn stream_child_output(child: &mut Child) {
    // Stream stdout if available
    if let Some(stdout) = child.stdout.take() {
        tokio::spawn(async move {
            let reader = BufReader::new(stdout);
            let mut lines = reader.lines();
            while let Ok(Some(line)) = lines.next_line().await {
                println!("{}", line);
            }
        });
    }

    // Stream stderr if available
    if let Some(stderr) = child.stderr.take() {
        tokio::spawn(async move {
            let reader = BufReader::new(stderr);
            let mut lines = reader.lines();
            while let Ok(Some(line)) = lines.next_line().await {
                eprintln!("{}", line);
            }
        });
    }
}
