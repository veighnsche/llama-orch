//! Main audit logger implementation

use crate::config::{AuditConfig, AuditMode};
use crate::error::{AuditError, Result};
use crate::events::AuditEvent;
use crate::storage::AuditEventEnvelope;
use crate::validation;
use crate::writer;
use chrono::Utc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Audit logger
///
/// Main entry point for audit logging.
/// Provides async, non-blocking event emission with buffering.
pub struct AuditLogger {
    /// Configuration
    config: AuditConfig,
    
    /// Channel sender for background writer
    tx: tokio::sync::mpsc::Sender<WriterMessage>,
    
    /// Event counter for generating audit IDs
    event_counter: Arc<AtomicU64>,
}

/// Messages sent to writer task
#[derive(Debug)]
pub(crate) enum WriterMessage {
    /// Write an event
    Event(AuditEventEnvelope),
    
    /// Flush all buffered events
    Flush(tokio::sync::oneshot::Sender<Result<()>>),
    
    /// Shutdown writer
    Shutdown,
}

impl AuditLogger {
    /// Create new audit logger
    ///
    /// Spawns background writer task.
    ///
    /// # Errors
    ///
    /// Returns error if configuration is invalid or writer task fails to start.
    pub fn new(config: AuditConfig) -> Result<Self> {
        // Validate configuration
        if let AuditMode::Local { ref base_dir } = config.mode {
            // Ensure directory exists
            std::fs::create_dir_all(base_dir)
                .map_err(|e| AuditError::InvalidPath(format!("Cannot create audit directory: {}", e)))?;
        }
        
        // Create bounded channel (1000 events max)
        const BUFFER_SIZE: usize = 1000;
        let (tx, rx) = tokio::sync::mpsc::channel(BUFFER_SIZE);
        
        // Spawn background writer task
        let writer_config = config.clone();
        tokio::spawn(async move {
            writer::audit_writer_task(rx, writer_config).await;
        });
        
        Ok(Self {
            config,
            tx,
            event_counter: Arc::new(AtomicU64::new(0)),
        })
    }
    
    /// Emit audit event (async, non-blocking)
    ///
    /// # Errors
    ///
    /// Returns `AuditError::BufferFull` if buffer is full.
    /// Returns `AuditError::InvalidInput` if event validation fails.
    pub async fn emit(&self, mut event: AuditEvent) -> Result<()> {
        // Validate and sanitize event
        validation::validate_event(&mut event)?;
        
        // Generate unique audit ID
        let counter = self.event_counter.fetch_add(1, Ordering::SeqCst);
        
        // Check for counter overflow (extremely unlikely but safety-critical)
        if counter == u64::MAX {
            tracing::error!("Audit counter overflow detected");
            return Err(AuditError::CounterOverflow);
        }
        
        let audit_id = format!("audit-{}-{:016x}", self.config.service_id, counter);
        
        // Create envelope (prev_hash will be set by writer)
        let envelope = AuditEventEnvelope::new(
            audit_id,
            Utc::now(),
            self.config.service_id.clone(),
            event,
            String::new(), // prev_hash set by writer
        );
        
        // Try to send (non-blocking)
        self.tx.try_send(WriterMessage::Event(envelope))
            .map_err(|_| AuditError::BufferFull)?;
        
        Ok(())
    }
    
    /// Flush all buffered events
    ///
    /// Blocks until all events are written to disk.
    ///
    /// # Errors
    ///
    /// Returns error if flush fails or writer is unavailable.
    pub async fn flush(&self) -> Result<()> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        
        self.tx.send(WriterMessage::Flush(tx)).await
            .map_err(|_| AuditError::Io(std::io::Error::new(
                std::io::ErrorKind::BrokenPipe,
                "Writer task unavailable"
            )))?;
        
        rx.await
            .map_err(|_| AuditError::Io(std::io::Error::new(
                std::io::ErrorKind::BrokenPipe,
                "Flush response not received"
            )))?
    }
    
    /// Shutdown logger gracefully
    ///
    /// Flushes all events and closes files.
    ///
    /// # Errors
    ///
    /// Returns error if shutdown fails.
    pub async fn shutdown(self) -> Result<()> {
        // Send shutdown signal
        self.tx.send(WriterMessage::Shutdown).await
            .map_err(|_| AuditError::Io(std::io::Error::new(
                std::io::ErrorKind::BrokenPipe,
                "Writer task unavailable"
            )))?;
        
        // Give writer time to finish
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{AuditConfig, AuditMode, RotationPolicy, RetentionPolicy};
    use crate::events::{ActorInfo, AuditEvent, AuthMethod};
    use chrono::Utc;
    use std::sync::atomic::Ordering;
    
    #[tokio::test]
    async fn test_counter_overflow_detection() {
        // This test verifies that we detect counter overflow
        // We can't actually overflow a u64 in a test, but we can verify the logic
        
        let config = AuditConfig {
            mode: AuditMode::Local {
                base_dir: std::env::temp_dir().join("audit-test"),
            },
            service_id: "test".to_string(),
            rotation_policy: RotationPolicy::Daily,
            retention_policy: RetentionPolicy::default(),
        };
        
        let logger = AuditLogger::new(config).unwrap();
        
        // Set counter to MAX - 1
        logger.event_counter.store(u64::MAX - 1, Ordering::SeqCst);
        
        // This should succeed (counter is MAX - 1)
        let event = AuditEvent::AuthSuccess {
            timestamp: Utc::now(),
            actor: ActorInfo {
                user_id: "test@example.com".to_string(),
                ip: Some("127.0.0.1".parse().unwrap()),
                auth_method: AuthMethod::BearerToken,
                session_id: None,
            },
            method: AuthMethod::BearerToken,
            path: "/test".to_string(),
            service_id: "test".to_string(),
        };
        
        // First emit should work (counter becomes MAX)
        let result = logger.emit(event.clone()).await;
        assert!(result.is_ok(), "First emit should succeed");
        
        // Second emit should fail (counter is now MAX)
        let result = logger.emit(event).await;
        assert!(result.is_err(), "Should detect overflow");
        assert!(matches!(result.unwrap_err(), AuditError::CounterOverflow));
    }
}

impl Drop for AuditLogger {
    fn drop(&mut self) {
        // Log warning if events may be lost
        tracing::warn!("AuditLogger dropped, buffered events may be lost. Call shutdown() for graceful cleanup.");
    }
}
