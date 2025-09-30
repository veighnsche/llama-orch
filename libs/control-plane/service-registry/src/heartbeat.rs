//! Heartbeat monitoring task

use crate::ServiceRegistry;
use std::time::Duration;
use tokio::time::interval;
use tracing::warn;

/// Spawn a task to check for stale nodes periodically
pub fn spawn_stale_checker(registry: ServiceRegistry, check_interval_secs: u64) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut ticker = interval(Duration::from_secs(check_interval_secs));
        
        loop {
            ticker.tick().await;
            
            let stale_nodes = registry.check_stale_nodes();
            if !stale_nodes.is_empty() {
                warn!(
                    count = stale_nodes.len(),
                    nodes = ?stale_nodes,
                    "Marked nodes as offline due to stale heartbeat"
                );
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_stale_checker_spawns() {
        let registry = ServiceRegistry::new(30_000);
        let handle = spawn_stale_checker(registry, 1);
        
        // Let it run briefly
        tokio::time::sleep(Duration::from_millis(100)).await;
        
        handle.abort();
    }
}
