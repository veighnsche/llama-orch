//! WorkerRegistry trait implementation for rbee-heartbeat
//!
//! Created by: TEAM-159
//!
//! Implements the WorkerRegistry trait from rbee-heartbeat for the local
//! WorkerRegistry implementation.

use crate::registry::WorkerRegistry;
use async_trait::async_trait;

#[async_trait]
impl rbee_heartbeat::traits::WorkerRegistry for WorkerRegistry {
    async fn update_heartbeat(&self, worker_id: &str) -> bool {
        // TEAM-159: Delegate to existing update_heartbeat method
        self.update_heartbeat(worker_id).await
    }
}
