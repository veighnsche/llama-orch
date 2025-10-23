//! Platform mode client
//!
//! Sends audit events to central platform service.

use crate::config::PlatformConfig;
use crate::error::Result;
use crate::storage::AuditEventEnvelope;

/// Platform audit client
///
/// Sends events to central audit service with:
/// - Batching
/// - Retry on failure
/// - Event signing
pub struct PlatformClient {
    /// Configuration
    config: PlatformConfig,

    /// HTTP client
    client: reqwest::Client,
}

impl PlatformClient {
    /// Create new platform client
    pub fn new(_config: PlatformConfig) -> Result<Self> {
        // TODO: Implement
        // 1. Create reqwest client with TLS
        // 2. Validate config
        todo!("Implement PlatformClient::new")
    }

    /// Send events to platform
    pub async fn send_events(&self, _events: Vec<AuditEventEnvelope>) -> Result<()> {
        // TODO: Implement
        // 1. Sign events
        // 2. Serialize to JSON
        // 3. POST to platform endpoint
        // 4. Handle errors (retry, etc.)
        todo!("Implement send_events")
    }

    /// Sign event
    fn sign_event(&self, _envelope: &mut AuditEventEnvelope) -> Result<()> {
        // TODO: Implement
        // Use crypto::sign_event_hmac or crypto::sign_event_ed25519
        todo!("Implement sign_event")
    }
}
