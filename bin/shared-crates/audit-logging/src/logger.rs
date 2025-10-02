//! Main audit logger implementation

use crate::config::{AuditConfig, AuditMode};
use crate::error::{AuditError, Result};
use crate::events::AuditEvent;
use crate::storage::AuditEventEnvelope;
use crate::validation;
use crate::writer;
use chrono::Utc;
use std::fmt::Write;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Audit logger
///
/// Main entry point for audit logging.
/// Provides async, non-blocking event emission with buffering.
pub struct AuditLogger {
    /// Configuration (Arc for efficient sharing, no cloning)
    config: Arc<AuditConfig>,

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
        match &config.mode {
            AuditMode::Local { base_dir } => {
                // Ensure directory exists
                std::fs::create_dir_all(base_dir).map_err(|e| {
                    AuditError::InvalidPath(format!("Cannot create audit directory: {}", e))
                })?;
            }
            #[cfg(feature = "platform")]
            AuditMode::Platform(_) => {
                // Platform mode validation (if needed)
            }
        }

        // Wrap config in Arc for efficient sharing
        let config = Arc::new(config);

        // Create bounded channel (1000 events max)
        const BUFFER_SIZE: usize = 1000;
        let (tx, rx) = tokio::sync::mpsc::channel(BUFFER_SIZE);

        // Spawn background writer task (Arc clone is cheap)
        let writer_config = Arc::clone(&config);
        tokio::spawn(async move {
            writer::audit_writer_task(rx, writer_config).await;
        });

        Ok(Self { config, tx, event_counter: Arc::new(AtomicU64::new(0)) })
    }

    /// Emit audit event (non-blocking)
    ///
    /// Emits an audit event to the background writer task. This method is
    /// non-blocking and can be called from both sync and async contexts.
    ///
    /// # Use Cases
    ///
    /// - Emit from synchronous functions (e.g., `VramManager::seal_model()`)
    /// - Emit from async functions (e.g., HTTP handlers)
    /// - Emit from Drop implementations
    /// - Emit from any context (no async runtime required)
    ///
    /// # Errors
    ///
    /// Returns `AuditError::BufferFull` if buffer is full.
    /// Returns `AuditError::InvalidInput` if event validation fails.
    /// Returns `AuditError::CounterOverflow` if event counter overflows.
    ///
    /// # Example
    ///
    /// ```rust
    /// use audit_logging::{AuditLogger, AuditEvent, AuditConfig, AuditMode};
    /// use chrono::Utc;
    ///
    /// // Can be called from sync or async functions
    /// fn seal_model(audit_logger: &AuditLogger) -> Result<(), audit_logging::AuditError> {
    ///     // ... sealing logic ...
    ///     
    ///     audit_logger.emit(AuditEvent::VramSealed {
    ///         timestamp: Utc::now(),
    ///         shard_id: "shard-123".to_string(),
    ///         gpu_device: 0,
    ///         vram_bytes: 8_000_000_000,
    ///         digest: "abc123".to_string(),
    ///         worker_id: "worker-gpu-0".to_string(),
    ///     })?;
    ///     
    ///     Ok(())
    /// }
    /// ```
    pub fn emit(&self, mut event: AuditEvent) -> Result<()> {
        // Validate and sanitize event
        validation::validate_event(&mut event)?;

        // Generate unique audit ID
        let counter = self.event_counter.fetch_add(1, Ordering::SeqCst);

        // Check for counter overflow (extremely unlikely but safety-critical)
        if counter == u64::MAX {
            tracing::error!("Audit counter overflow detected");
            return Err(AuditError::CounterOverflow);
        }

        // Pre-allocate audit_id buffer (Finding 1 optimization)
        let mut audit_id = String::with_capacity(64);
        write!(&mut audit_id, "audit-{}-{:016x}", self.config.service_id, counter)
            .map_err(|e| AuditError::InvalidInput(e.to_string()))?;

        // Create envelope (prev_hash will be set by writer)
        // Arc clone is cheap (just reference counting)
        let envelope = AuditEventEnvelope::new(
            audit_id,
            Utc::now(),
            self.config.service_id.clone(), // Still need to clone the String itself
            event,
            String::new(), // prev_hash set by writer
        );

        // Try to send (non-blocking)
        self.tx.try_send(WriterMessage::Event(envelope)).map_err(|_| AuditError::BufferFull)?;

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

        self.tx.send(WriterMessage::Flush(tx)).await.map_err(|_| {
            AuditError::Io(std::io::Error::new(
                std::io::ErrorKind::BrokenPipe,
                "Writer task unavailable",
            ))
        })?;

        rx.await.map_err(|_| {
            AuditError::Io(std::io::Error::new(
                std::io::ErrorKind::BrokenPipe,
                "Flush response not received",
            ))
        })?
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
        self.tx.send(WriterMessage::Shutdown).await.map_err(|_| {
            AuditError::Io(std::io::Error::new(
                std::io::ErrorKind::BrokenPipe,
                "Writer task unavailable",
            ))
        })?;

        // Give writer time to finish
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{AuditConfig, AuditMode, RetentionPolicy, RotationPolicy};
    use crate::events::{ActorInfo, AuditEvent, AuthMethod};
    use chrono::Utc;
    use std::sync::atomic::Ordering;

    #[tokio::test]
    async fn test_counter_overflow_detection() {
        // This test verifies that we detect counter overflow
        // We can't actually overflow a u64 in a test, but we can verify the logic

        let config = AuditConfig {
            mode: AuditMode::Local { base_dir: std::env::temp_dir().join("audit-test") },
            service_id: "test".to_string(),
            rotation_policy: RotationPolicy::Daily,
            retention_policy: RetentionPolicy::default(),
            flush_mode: crate::config::FlushMode::Immediate,
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
        let result = logger.emit(event.clone());
        assert!(result.is_ok(), "First emit should succeed");

        // Second emit should fail (counter is now MAX)
        let result = logger.emit(event);
        assert!(result.is_err(), "Should detect overflow");
        assert!(matches!(result.unwrap_err(), AuditError::CounterOverflow));
    }

    #[tokio::test]
    async fn test_emit_from_sync_context() {
        // This test verifies that emit() can be called from sync context
        let config = AuditConfig {
            mode: AuditMode::Local { base_dir: std::env::temp_dir().join("audit-test-sync") },
            service_id: "test".to_string(),
            rotation_policy: RotationPolicy::Daily,
            retention_policy: RetentionPolicy::default(),
            flush_mode: crate::config::FlushMode::Immediate,
        };

        let logger = AuditLogger::new(config).unwrap();

        // ✅ Can call from sync function (no .await needed)
        let event = AuditEvent::VramSealed {
            timestamp: Utc::now(),
            shard_id: "shard-123".to_string(),
            gpu_device: 0,
            vram_bytes: 8_000_000_000,
            digest: "abc123".to_string(),
            worker_id: "worker-gpu-0".to_string(),
        };

        let result = logger.emit(event);
        assert!(result.is_ok(), "emit should succeed from sync context");
    }
    #[tokio::test]
    async fn test_emit_counter_overflow() {
        let config = AuditConfig {
            mode: AuditMode::Local { base_dir: std::env::temp_dir().join("audit-test") },
            service_id: "test".to_string(),
            rotation_policy: RotationPolicy::Daily,
            retention_policy: RetentionPolicy::default(),
            flush_mode: crate::config::FlushMode::Immediate,
        };

        let logger = AuditLogger::new(config).unwrap();

        // Set counter to MAX
        logger.event_counter.store(u64::MAX, Ordering::SeqCst);

        let event = AuditEvent::VramSealed {
            timestamp: Utc::now(),
            shard_id: "shard-123".to_string(),
            gpu_device: 0,
            vram_bytes: 8_000_000_000,
            digest: "abc123".to_string(),
            worker_id: "worker-gpu-0".to_string(),
        };

        // Should detect overflow
        let result = logger.emit(event);
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
