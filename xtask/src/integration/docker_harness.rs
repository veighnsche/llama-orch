// Docker test harness for integration tests
// Purpose: Manage Docker containers for queen-rbee → rbee-hive communication testing

use anyhow::{Context, Result};
use std::path::PathBuf;
use std::process::Command;
use std::time::Duration;
use tokio::time::sleep;

#[derive(Debug, Clone)]
pub enum Topology {
    Localhost,
    MultiHive,
}

impl Topology {
    pub fn compose_file(&self) -> &str {
        match self {
            Topology::Localhost => "../tests/docker/docker-compose.localhost.yml",
            Topology::MultiHive => "../tests/docker/docker-compose.multi-hive.yml",
        }
    }
}

pub struct DockerTestHarness {
    compose_file: PathBuf,
    #[allow(dead_code)]
    test_id: String,
}

impl DockerTestHarness {
    /// Create new Docker test environment
    pub async fn new(topology: Topology) -> Result<Self> {
        let compose_file = topology.compose_file();
        let test_id = uuid::Uuid::new_v4().to_string();

        println!("🐳 Starting Docker environment: {:?}", topology);
        println!("📋 Test ID: {}", test_id);

        // Start containers
        Self::docker_compose_up(compose_file).await?;

        // Wait for services to be healthy
        Self::wait_for_services().await?;

        Ok(Self { compose_file: compose_file.into(), test_id })
    }

    /// Start containers via docker-compose
    async fn docker_compose_up(compose_file: &str) -> Result<()> {
        // Build images first
        let build_output = Command::new("docker-compose")
            .args(&["-f", compose_file, "build"])
            .output()
            .context("Failed to build docker-compose images")?;

        if !build_output.status.success() {
            let stderr = String::from_utf8_lossy(&build_output.stderr);
            anyhow::bail!("docker-compose build failed: {}", stderr);
        }

        println!("✅ Images built");

        // Start containers
        let output = Command::new("docker-compose")
            .args(&["-f", compose_file, "up", "-d"])
            .output()
            .context("Failed to start docker-compose")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("docker-compose up failed: {}", stderr);
        }

        println!("✅ Containers started");
        Ok(())
    }

    /// Wait for all services to be healthy
    async fn wait_for_services() -> Result<()> {
        println!("⏳ Waiting for services to be healthy...");

        // Wait for queen
        Self::wait_for_http("http://localhost:8500/health", Duration::from_secs(30)).await?;
        println!("✅ Queen ready");

        // Wait for hive
        Self::wait_for_http("http://localhost:9000/health", Duration::from_secs(30)).await?;
        println!("✅ Hive ready");

        Ok(())
    }

    /// Execute command in container
    pub async fn exec(&self, container: &str, cmd: &[&str]) -> Result<String> {
        let output = Command::new("docker")
            .arg("exec")
            .arg(container)
            .args(cmd)
            .output()
            .context(format!("Failed to exec in container {}", container))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("Command failed: {}", stderr);
        }

        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }

    /// Get container logs
    pub async fn logs(&self, container: &str) -> Result<String> {
        let output = Command::new("docker")
            .args(&["logs", container])
            .output()
            .context(format!("Failed to get logs for {}", container))?;

        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }

    /// Restart container
    pub async fn restart(&self, container: &str) -> Result<()> {
        Command::new("docker")
            .args(&["restart", container])
            .output()
            .context(format!("Failed to restart {}", container))?;

        println!("🔄 Restarted container: {}", container);
        Ok(())
    }

    /// Kill container (simulate crash)
    pub async fn kill(&self, container: &str) -> Result<()> {
        Command::new("docker")
            .args(&["kill", container])
            .output()
            .context(format!("Failed to kill {}", container))?;

        println!("💀 Killed container: {}", container);
        Ok(())
    }

    /// Block network between containers
    pub async fn block_network(&self, from: &str, to_ip: &str) -> Result<()> {
        self.exec(from, &["iptables", "-A", "OUTPUT", "-d", to_ip, "-j", "DROP"]).await?;
        println!("🚫 Blocked network: {} → {}", from, to_ip);
        Ok(())
    }

    /// Restore network between containers
    pub async fn restore_network(&self, from: &str, to_ip: &str) -> Result<()> {
        self.exec(from, &["iptables", "-D", "OUTPUT", "-d", to_ip, "-j", "DROP"]).await?;
        println!("✅ Restored network: {} → {}", from, to_ip);
        Ok(())
    }

    /// Wait for HTTP endpoint to be healthy
    pub async fn wait_for_http(url: &str, timeout: Duration) -> Result<()> {
        let start = std::time::Instant::now();

        loop {
            match ureq::get(url).timeout(Duration::from_secs(2)).call() {
                Ok(response) if response.status() == 200 => {
                    return Ok(());
                }
                _ => {}
            }

            if start.elapsed() > timeout {
                anyhow::bail!("Timeout waiting for HTTP endpoint: {}", url);
            }

            sleep(Duration::from_millis(500)).await;
        }
    }
}

impl Drop for DockerTestHarness {
    fn drop(&mut self) {
        println!("🧹 Cleaning up Docker environment...");

        // Use to_string_lossy instead of unwrap to avoid panic in Drop
        let compose_path = self.compose_file.to_string_lossy();
        let _ = Command::new("docker-compose")
            .args(&["-f", compose_path.as_ref(), "down", "-v"])
            .output();

        println!("✅ Cleanup complete");
    }
}
