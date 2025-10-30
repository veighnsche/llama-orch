// TEAM-365: Created by TEAM-365
// TEAM-366: Added edge case guards for discovery reliability
//! Queen hive discovery module
//!
//! Implements pull-based discovery: Queen reads SSH config and sends
//! GET /capabilities?queen_url=X to all configured hives.
//!
//! This implements Scenario 1 from HEARTBEAT_ARCHITECTURE.md:
//! - Queen waits 5 seconds for services to stabilize
//! - Queen reads SSH config
//! - Queen sends parallel GET /capabilities?queen_url=X to all hives
//! - Hives respond with capabilities and start heartbeats

use anyhow::Result;
use observability_narration_core::n;
use ssh_config_parser::{get_default_ssh_config_path, parse_ssh_config, SshTarget};
use std::collections::HashSet; // TEAM-366: Deduplicate targets
use std::time::Duration;

/// Discover all hives on Queen startup
///
/// TEAM-365: Pull-based discovery (Scenario 1 from HEARTBEAT_ARCHITECTURE.md)
///
/// # Flow
/// 1. Wait 5 seconds for services to stabilize
/// 2. Read SSH config
/// 3. Send parallel GET /capabilities?queen_url=X to all hives
/// 4. Store capabilities in HiveRegistry (TODO: when implemented)
///
/// # Arguments
/// * `queen_url` - URL of this Queen instance (e.g., "http://localhost:7833")
pub async fn discover_hives_on_startup(queen_url: &str) -> Result<()> {
    // TEAM-366: EDGE CASE #6 - Validate queen_url before discovery
    if queen_url.is_empty() {
        n!("discovery_invalid_url", "❌ Cannot start discovery: empty queen_url");
        anyhow::bail!("Cannot start discovery with empty queen_url");
    }
    
    if let Err(e) = url::Url::parse(queen_url) {
        n!("discovery_invalid_url", "❌ Cannot start discovery: invalid queen_url '{}': {}", queen_url, e);
        anyhow::bail!("Invalid queen_url '{}': {}", queen_url, e);
    }
    
    n!("discovery_start", "🔍 Starting hive discovery (waiting 5s for services to stabilize)");
    
    // TEAM-365: Wait for services to stabilize
    tokio::time::sleep(Duration::from_secs(5)).await;
    
    // TEAM-365: Read SSH config
    let ssh_config_path = get_default_ssh_config_path();
    let targets = match parse_ssh_config(&ssh_config_path) {
        Ok(targets) => targets,
        Err(e) => {
            n!("discovery_no_config", "⚠️  No SSH config found: {}. Only localhost will be discovered.", e);
            vec![]
        }
    };
    
    // TEAM-366: EDGE CASE #7 - Deduplicate targets by hostname
    let mut seen = HashSet::new();
    let unique_targets: Vec<_> = targets
        .into_iter()
        .filter(|t| {
            // TEAM-366: EDGE CASE #8 - Skip invalid hostnames
            if t.hostname.is_empty() {
                n!("discovery_skip_invalid", "⚠️  Skipping target '{}': empty hostname", t.host);
                return false;
            }
            
            // TEAM-366: EDGE CASE #7 - Skip duplicates
            if !seen.insert(t.hostname.clone()) {
                n!("discovery_skip_duplicate", "⚠️  Skipping duplicate target: {} ({})", t.host, t.hostname);
                return false;
            }
            
            true
        })
        .collect();
    
    n!("discovery_targets", "📋 Found {} unique SSH targets to discover", unique_targets.len());
    
    // TEAM-365: Discover all hives in parallel
    let mut tasks = vec![];
    for target in unique_targets {
        let queen_url = queen_url.to_string();
        
        tasks.push(tokio::spawn(async move {
            discover_single_hive(&target, &queen_url).await
        }));
    }
    
    // TEAM-365: Wait for all discoveries
    let mut success_count = 0;
    let mut failure_count = 0;
    
    for task in tasks {
        match task.await {
            Ok(Ok(_)) => success_count += 1,
            Ok(Err(_)) => failure_count += 1,
            Err(_) => failure_count += 1,
        }
    }
    
    n!("discovery_complete", "✅ Discovery complete: {} successful, {} failed", success_count, failure_count);
    
    Ok(())
}

/// Discover a single hive
///
/// TEAM-365: Send GET /capabilities?queen_url=X to hive
///
/// # Arguments
/// * `target` - SSH target from config
/// * `queen_url` - URL of this Queen instance
async fn discover_single_hive(target: &SshTarget, queen_url: &str) -> Result<()> {
    let url = format!(
        "http://{}:7835/capabilities?queen_url={}",
        target.hostname,
        urlencoding::encode(queen_url)
    );
    
    n!("discovery_hive", "🔍 Discovering hive: {} ({})", target.host, target.hostname);
    
    let client = reqwest::Client::new();
    let response = client
        .get(&url)
        .timeout(Duration::from_secs(10))
        .send()
        .await?;
    
    if response.status().is_success() {
        n!("discovery_success", "✅ Discovered hive: {}", target.host);
        
        // TODO: TEAM-365: Store capabilities in HiveRegistry when implemented
        // let capabilities: CapabilitiesResponse = response.json().await?;
        // hive_registry.register_hive(target.host.clone(), capabilities);
    } else {
        n!("discovery_failed", "❌ Failed to discover hive {}: {}", target.host, response.status());
        anyhow::bail!("Failed to discover hive {}: {}", target.host, response.status());
    }
    
    Ok(())
}
