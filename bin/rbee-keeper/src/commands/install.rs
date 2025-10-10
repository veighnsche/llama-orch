//! Installation command - Install rbee binaries to standard paths
//!
//! Created by: TEAM-036
//! Implements XDG Base Directory specification
//! Replaces shell script with proper Rust implementation

use anyhow::{Context, Result};
use std::fs;
use std::path::{Path, PathBuf};

pub enum InstallTarget {
    User,   // ~/.local/bin
    System, // /usr/local/bin (requires sudo)
}

pub fn handle(system: bool) -> Result<()> {
    let target = if system { InstallTarget::System } else { InstallTarget::User };

    let (bin_dir, config_dir, data_dir) = match target {
        InstallTarget::User => {
            let home = dirs::home_dir().context("Could not determine home directory")?;
            (home.join(".local/bin"), home.join(".config/rbee"), home.join(".local/share/rbee"))
        }
        InstallTarget::System => (
            PathBuf::from("/usr/local/bin"),
            PathBuf::from("/etc/rbee"),
            PathBuf::from("/var/lib/rbee"),
        ),
    };

    println!("📦 Installing rbee binaries...");

    // 1. Create directories
    fs::create_dir_all(&bin_dir).context("Failed to create bin directory")?;
    fs::create_dir_all(&config_dir).context("Failed to create config directory")?;
    fs::create_dir_all(data_dir.join("models")).context("Failed to create models directory")?;

    println!("✅ Created directories:");
    println!("   Binaries: {}", bin_dir.display());
    println!("   Config:   {}", config_dir.display());
    println!("   Data:     {}", data_dir.display());

    // 2. Copy binaries
    let current_exe = std::env::current_exe().context("Could not determine current executable")?;
    let exe_dir = current_exe.parent().context("Could not determine executable directory")?;

    println!("\n📋 Installing binaries:");
    copy_binary(exe_dir, &bin_dir, "rbee")?;
    copy_binary(exe_dir, &bin_dir, "rbee-hive")?;
    copy_binary(exe_dir, &bin_dir, "llm-worker-rbee")?;

    // 3. Create default config
    create_default_config(&config_dir, &data_dir)?;

    println!("\n✅ Installation complete!");
    println!("\n📝 Next steps:");
    println!("   1. Add {} to your PATH", bin_dir.display());
    println!("   2. Edit config: {}/config.toml", config_dir.display());
    println!("   3. Run: rbee --version");

    if matches!(target, InstallTarget::User) {
        println!("\n💡 Add to your shell profile:");
        println!("   export PATH=\"$HOME/.local/bin:$PATH\"");
    }

    Ok(())
}

fn copy_binary(src_dir: &Path, dest_dir: &Path, name: &str) -> Result<()> {
    let src = src_dir.join(name);
    let dest = dest_dir.join(name);

    if !src.exists() {
        println!("   ⚠️  {} (not found, skipping)", name);
        return Ok(());
    }

    fs::copy(&src, &dest)
        .with_context(|| format!("Failed to copy {} to {}", src.display(), dest.display()))?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = fs::metadata(&dest)?.permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&dest, perms)?;
    }

    println!("   ✅ {}", name);
    Ok(())
}

fn create_default_config(config_dir: &Path, data_dir: &Path) -> Result<()> {
    let config_path = config_dir.join("config.toml");
    if config_path.exists() {
        println!("\n⚠️  Config already exists, not overwriting: {}", config_path.display());
        return Ok(());
    }

    let hostname = hostname::get()
        .ok()
        .and_then(|h| h.into_string().ok())
        .unwrap_or_else(|| "localhost".to_string());

    let config = format!(
        r#"# rbee configuration
# Created by: TEAM-036

[pool]
name = "{}"
listen_addr = "0.0.0.0:8080"

[paths]
models_dir = "{}/models"
catalog_db = "{}/models.db"

# Optional: Custom paths for remote commands
# [remote]
# binary_path = "/custom/path/to/rbee-hive"
# git_repo_dir = "/custom/path/to/llama-orch"
"#,
        hostname,
        data_dir.display(),
        data_dir.display()
    );

    fs::write(&config_path, config)
        .with_context(|| format!("Failed to write config to {}", config_path.display()))?;

    println!("\n✅ Created default config: {}", config_path.display());
    Ok(())
}
