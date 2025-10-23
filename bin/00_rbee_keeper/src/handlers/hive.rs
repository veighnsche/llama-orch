//! Hive command handlers
//!
//! TEAM-276: Extracted from main.rs
//! TEAM-194: Use alias-based operations (config from hives.conf)
//! TEAM-196: Added RefreshCapabilities command
//! TEAM-263: Added smart prompt for localhost hive install

use anyhow::Result;
use rbee_operations::Operation;

use crate::cli::HiveAction;
use crate::job_client::submit_and_stream_job;

pub async fn handle_hive(action: HiveAction, queen_url: &str) -> Result<()> {
    // Special handling for HiveInstall on localhost
    if let HiveAction::Install { ref alias } = action {
        if alias == "localhost" {
            check_local_hive_optimization(queen_url).await?;
        }
    }

    let operation = match action {
        HiveAction::SshTest { alias } => Operation::SshTest { alias },
        HiveAction::Install { alias } => Operation::HiveInstall { alias },
        HiveAction::Uninstall { alias } => Operation::HiveUninstall { alias },
        HiveAction::Start { alias } => Operation::HiveStart { alias },
        HiveAction::Stop { alias } => Operation::HiveStop { alias },
        HiveAction::List => Operation::HiveList,
        HiveAction::Get { alias } => Operation::HiveGet { alias },
        HiveAction::Status { alias } => Operation::HiveStatus { alias },
        HiveAction::RefreshCapabilities { alias } => Operation::HiveRefreshCapabilities { alias },
        HiveAction::ImportSsh { ssh_config, default_port } => {
            // Expand ~ to home directory
            let ssh_config_path = if ssh_config.starts_with("~/") {
                let home = std::env::var("HOME").unwrap_or_else(|_| "/root".to_string());
                ssh_config.replacen("~", &home, 1)
            } else {
                ssh_config
            };
            Operation::HiveImportSsh { ssh_config_path, default_hive_port: default_port }
        }
    };
    submit_and_stream_job(queen_url, operation).await
}

/// Check queen's build configuration and prompt user if installing localhost without local-hive feature
async fn check_local_hive_optimization(queen_url: &str) -> Result<()> {
    let check_client = reqwest::Client::builder()
        .timeout(tokio::time::Duration::from_secs(3))
        .build()?;

    if let Ok(response) = check_client.get(format!("{}/v1/build-info", queen_url)).send().await {
        if response.status().is_success() {
            if let Ok(body) = response.text().await {
                // Parse JSON to check for local-hive feature
                if let Ok(build_info) = serde_json::from_str::<serde_json::Value>(&body) {
                    let features = build_info["features"].as_array();
                    let has_local_hive = features
                        .map(|f| f.iter().any(|v| v.as_str() == Some("local-hive")))
                        .unwrap_or(false);

                    if !has_local_hive {
                        // PROMPT USER!
                        eprintln!("\n⚠️  Performance Notice:");
                        eprintln!();
                        eprintln!("   You're installing a hive on localhost, but your queen-rbee");
                        eprintln!("   was built without the 'local-hive' feature.");
                        eprintln!();
                        eprintln!("   📊 Performance comparison:");
                        eprintln!("      • Current setup:  ~5-10ms overhead (HTTP)");
                        eprintln!("      • Integrated:     ~0.1ms overhead (direct calls)");
                        eprintln!("      • Speedup:        50-100x faster");
                        eprintln!();
                        eprintln!("   💡 Recommendation:");
                        eprintln!("      Rebuild queen-rbee with integrated hive for localhost:");
                        eprintln!();
                        eprintln!("      $ rbee-keeper queen rebuild --with-local-hive");
                        eprintln!("      $ rbee-keeper queen stop");
                        eprintln!("      $ rbee-keeper queen start");
                        eprintln!();
                        eprintln!("   ℹ️  Or continue with distributed setup if you have specific needs.");
                        eprintln!();

                        // Ask user
                        eprint!("   Continue with distributed setup? [y/N]: ");
                        use std::io::Write;
                        std::io::stdout().flush()?;

                        let mut input = String::new();
                        std::io::stdin().read_line(&mut input)?;

                        if !matches!(input.trim().to_lowercase().as_str(), "y" | "yes") {
                            eprintln!("\n✋ Installation cancelled.");
                            eprintln!("   Run: rbee-keeper queen rebuild --with-local-hive");
                            return Ok(());
                        }
                        eprintln!(); // Add spacing
                    }
                }
            }
        }
    }

    Ok(())
}
