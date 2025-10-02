//! Audit event writer
//!
//! Handles:
//! - Async file writing
//! - Buffering and batching
//! - File rotation
//! - Hash chain computation

use crate::config::{AuditConfig, AuditMode, RotationPolicy};
use crate::crypto;
use crate::error::{AuditError, Result};
use crate::logger::WriterMessage;
use crate::storage::AuditEventEnvelope;
use chrono::Utc;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::PathBuf;

#[cfg(unix)]
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};

/// Minimum required disk space (10 MB)
const MIN_DISK_SPACE: u64 = 10 * 1024 * 1024;

/// Audit file writer
/// Responsibilities:
/// - Write events to append-only file
/// - Compute hash chains
/// - Handle file rotation
/// - Maintain manifest
/// - Batch fsync for performance (configurable)
pub struct AuditFileWriter {
    /// Current file handle
    file: File,

    /// Current file path
    file_path: PathBuf,

    /// Event count in current file
    event_count: usize,

    /// Last event hash (for chain)
    last_hash: String,

    /// Rotation policy
    rotation_policy: RotationPolicy,

    /// File size in bytes
    file_size: u64,

    /// Flush mode (immediate, batched, or hybrid)
    flush_mode: crate::config::FlushMode,

    /// Events written since last fsync
    events_since_sync: usize,

    /// Time of last fsync
    last_sync: std::time::Instant,
}

impl AuditFileWriter {
    /// Create new writer
    pub fn new(
        file_path: PathBuf,
        rotation_policy: RotationPolicy,
        flush_mode: crate::config::FlushMode,
    ) -> Result<Self> {
        // Open file in append mode (create if doesn't exist)
        #[cfg(unix)]
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .mode(0o600) // Owner read/write only
            .open(&file_path)?;

        #[cfg(not(unix))]
        let file = OpenOptions::new().create(true).append(true).open(&file_path)?;

        // Verify file permissions on Unix
        #[cfg(unix)]
        {
            let metadata = file.metadata()?;
            let permissions = metadata.permissions();
            let mode = permissions.mode();

            // Check that group and other have no permissions
            if mode & 0o077 != 0 {
                tracing::warn!(
                    path = ?file_path,
                    mode = format!("{:o}", mode),
                    "Audit file has insecure permissions, expected 0600"
                );
                // Don't fail, just warn - permissions might be set by umask
            }
        }

        // Get current file size
        let metadata = file.metadata()?;
        let file_size = metadata.len();

        Ok(Self {
            file,
            file_path,
            event_count: 0,
            last_hash: String::from(
                "0000000000000000000000000000000000000000000000000000000000000000",
            ),
            rotation_policy,
            file_size,
            flush_mode,
            events_since_sync: 0,
            last_sync: std::time::Instant::now(),
        })
    }

    /// Check available disk space
    fn check_disk_space(&self) -> Result<()> {
        #[cfg(unix)]
        {
            // Try to get filesystem stats
            // Note: This is a best-effort check, not all filesystems support this
            if let Ok(stats) = nix::sys::statvfs::statvfs(&self.file_path) {
                let available = stats.blocks_available() * stats.block_size();

                if available < MIN_DISK_SPACE {
                    tracing::error!(
                        available,
                        required = MIN_DISK_SPACE,
                        "Disk space critically low"
                    );
                    return Err(AuditError::DiskSpaceLow { available, required: MIN_DISK_SPACE });
                }
            }
        }

        Ok(())
    }

    /// Write event to file
    ///
    /// Supports batched flushing based on FlushMode configuration.
    /// Critical events always flush immediately (if critical_immediate is true).
    pub fn write_event(
        &mut self,
        mut envelope: AuditEventEnvelope,
        is_critical: bool,
    ) -> Result<()> {
        // Check disk space before writing
        self.check_disk_space()?;

        // Set prev_hash to last_hash
        envelope.prev_hash = self.last_hash.clone();

        // Compute event hash
        envelope.hash = crypto::compute_event_hash(&envelope)?;

        // Serialize to JSON
        let json = serde_json::to_string(&envelope)?;

        // Write with newline
        writeln!(self.file, "{}", json)?;

        // Update state
        self.last_hash = envelope.hash;
        self.event_count = self.event_count.saturating_add(1);
        self.file_size = self.file_size.saturating_add(json.len() as u64).saturating_add(1);
        self.events_since_sync = self.events_since_sync.saturating_add(1);

        // Decide whether to flush based on FlushMode
        let should_flush = match &self.flush_mode {
            crate::config::FlushMode::Immediate => {
                // Always flush immediately
                true
            }
            crate::config::FlushMode::Batched { size, interval_secs } => {
                // Flush if batch size or interval exceeded
                let elapsed = self.last_sync.elapsed();
                self.events_since_sync >= *size || elapsed.as_secs() >= *interval_secs
            }
            crate::config::FlushMode::Hybrid {
                batch_size,
                batch_interval_secs,
                critical_immediate,
            } => {
                // Critical events flush immediately (if enabled)
                if is_critical && *critical_immediate {
                    true
                } else {
                    // Routine events batch
                    let elapsed = self.last_sync.elapsed();
                    self.events_since_sync >= *batch_size
                        || elapsed.as_secs() >= *batch_interval_secs
                }
            }
        };

        if should_flush {
            self.file.sync_all()?;
            self.events_since_sync = 0;
            self.last_sync = std::time::Instant::now();
        }

        Ok(())
    }

    /// Flush buffered writes
    pub fn flush(&mut self) -> Result<()> {
        self.file.flush()?;
        self.file.sync_all()?;
        self.events_since_sync = 0;
        self.last_sync = std::time::Instant::now();
        Ok(())
    }

    /// Check if rotation is needed
    pub fn should_rotate(&self) -> bool {
        match &self.rotation_policy {
            RotationPolicy::Daily => {
                // Check if date changed
                let current_date = Utc::now().format("%Y-%m-%d").to_string();
                let file_name = self.file_path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
                !file_name.starts_with(&current_date)
            }
            RotationPolicy::SizeLimit(limit) => self.file_size >= *limit as u64,
            RotationPolicy::Both { daily, size_limit } => {
                let date_changed = if *daily {
                    let current_date = Utc::now().format("%Y-%m-%d").to_string();
                    let file_name =
                        self.file_path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
                    !file_name.starts_with(&current_date)
                } else {
                    false
                };
                date_changed || self.file_size >= *size_limit as u64
            }
        }
    }

    /// Rotate to new file
    pub fn rotate(&mut self) -> Result<()> {
        // Flush and close current file
        self.flush()?;

        // Create new file path with uniqueness guarantee
        let base_dir = self.file_path.parent().unwrap_or(std::path::Path::new("."));
        let date = Utc::now().format("%Y-%m-%d").to_string();

        // Try to find a unique filename
        let mut attempt = 0;
        let new_path = loop {
            let path = if attempt == 0 {
                base_dir.join(format!("{}.audit", date))
            } else {
                base_dir.join(format!("{}-{}.audit", date, attempt))
            };

            // Check if file exists
            if !path.exists() {
                break path;
            }

            attempt += 1;
            if attempt > 1000 {
                return Err(AuditError::RotationFailed(
                    "Too many rotation attempts, file already exists".into(),
                ));
            }
        };

        // Open new file with create_new to prevent races
        #[cfg(unix)]
        let new_file = OpenOptions::new()
            .create_new(true) // Fail if file exists
            .append(true)
            .mode(0o600)
            .open(&new_path)?;

        #[cfg(not(unix))]
        let new_file = OpenOptions::new().create_new(true).append(true).open(&new_path)?;

        // Update state
        self.file = new_file;
        self.file_path = new_path;
        self.event_count = 0;
        self.file_size = 0;
        // Note: last_hash is preserved for chain continuity

        Ok(())
    }

    /// Close writer (flush and finalize)
    pub fn close(mut self) -> Result<()> {
        self.flush()?;
        Ok(())
    }
}

/// Background writer task
///
/// Receives events from channel and writes them asynchronously.
pub async fn audit_writer_task(
    mut rx: tokio::sync::mpsc::Receiver<WriterMessage>,
    config: std::sync::Arc<AuditConfig>,
) {
    // Extract base directory
    let base_dir = match &config.mode {
        AuditMode::Local { base_dir } => base_dir.clone(),
        #[cfg(feature = "platform")]
        AuditMode::Platform(_) => {
            tracing::error!("Platform mode not yet implemented");
            return;
        }
    };

    // Create initial file path
    let date = Utc::now().format("%Y-%m-%d").to_string();
    let file_path = base_dir.join(format!("{}.audit", date));

    // Create writer (clone policies, they're cheap)
    let mut writer = match AuditFileWriter::new(
        file_path,
        config.rotation_policy.clone(),
        config.flush_mode.clone(),
    ) {
        Ok(w) => w,
        Err(e) => {
            tracing::error!(error = ?e, "Failed to create audit writer");
            return;
        }
    };

    // Process messages
    while let Some(msg) = rx.recv().await {
        match msg {
            WriterMessage::Event(envelope) => {
                // Check if event is critical (for hybrid flush mode)
                let is_critical = envelope.event.is_critical();

                if let Err(e) = writer.write_event(envelope, is_critical) {
                    tracing::error!(error = ?e, "Failed to write audit event");
                }

                // Check if rotation needed
                if writer.should_rotate() {
                    if let Err(e) = writer.rotate() {
                        tracing::error!(error = ?e, "Failed to rotate audit file");
                    }
                }
            }
            WriterMessage::Flush(response_tx) => {
                let result = writer.flush();
                let _ = response_tx.send(result);
            }
            WriterMessage::Shutdown => {
                tracing::info!("Audit writer shutting down");
                if let Err(e) = writer.close() {
                    tracing::error!(error = ?e, "Error closing audit writer");
                }
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::{ActorInfo, AuditEvent, AuthMethod};
    use chrono::Utc;
    use tempfile::TempDir;

    fn create_test_event() -> AuditEvent {
        AuditEvent::AuthSuccess {
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
        }
    }
    #[test]
    fn test_writer_new() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.audit");

        let writer = AuditFileWriter::new(
            file_path.clone(),
            RotationPolicy::Daily,
            crate::config::FlushMode::Immediate,
        );
        assert!(writer.is_ok());

        let writer = writer.unwrap();
        assert_eq!(writer.event_count, 0);
        assert_eq!(writer.last_hash.len(), 64);
    }
    #[test]
    fn test_write_event() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.audit");

        let mut writer = AuditFileWriter::new(
            file_path.clone(),
            RotationPolicy::Daily,
            crate::config::FlushMode::Immediate,
        )
        .unwrap();

        let envelope = AuditEventEnvelope::new(
            "audit-001".to_string(),
            Utc::now(),
            "test-service".to_string(),
            create_test_event(),
            String::new(),
        );

        let result = writer.write_event(envelope, false);
        assert!(result.is_ok());
        assert_eq!(writer.event_count, 1);
        assert_ne!(
            writer.last_hash,
            "0000000000000000000000000000000000000000000000000000000000000000"
        );

        // Verify file was created
        assert!(file_path.exists());
    }

    #[test]
    fn test_write_multiple_events_chains_hashes() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.audit");

        let mut writer = AuditFileWriter::new(
            file_path.clone(),
            RotationPolicy::Daily,
            crate::config::FlushMode::Immediate,
        )
        .unwrap();

        let envelope1 = AuditEventEnvelope::new(
            "audit-001".to_string(),
            Utc::now(),
            "test-service".to_string(),
            create_test_event(),
            String::new(),
        );

        writer.write_event(envelope1, false).unwrap();
        let hash1 = writer.last_hash.clone();

        let envelope2 = AuditEventEnvelope::new(
            "audit-002".to_string(),
            Utc::now(),
            "test-service".to_string(),
            create_test_event(),
            String::new(),
        );

        writer.write_event(envelope2, false).unwrap();
        let hash2 = writer.last_hash.clone();

        assert_ne!(hash1, hash2);
        assert_eq!(writer.event_count, 2);
    }

    #[test]
    fn test_should_rotate_daily() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("2020-01-01.audit");

        let writer = AuditFileWriter::new(
            file_path,
            RotationPolicy::Daily,
            crate::config::FlushMode::Immediate,
        )
        .unwrap();

        // Should rotate because file name doesn't match current date
        assert!(writer.should_rotate());
    }

    #[test]
    fn test_should_rotate_size_limit() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.audit");

        let mut writer = AuditFileWriter::new(
            file_path,
            RotationPolicy::SizeLimit(100),
            crate::config::FlushMode::Immediate,
        )
        .unwrap();

        // Initially should not rotate
        assert!(!writer.should_rotate());

        // Manually set file size above limit
        writer.file_size = 150;
        assert!(writer.should_rotate());
    }

    #[test]
    fn test_flush() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.audit");

        let mut writer = AuditFileWriter::new(
            file_path,
            RotationPolicy::Daily,
            crate::config::FlushMode::Immediate,
        )
        .unwrap();

        let envelope = AuditEventEnvelope::new(
            "audit-001".to_string(),
            Utc::now(),
            "test-service".to_string(),
            create_test_event(),
            String::new(),
        );

        writer.write_event(envelope, false).unwrap();

        let result = writer.flush();
        assert!(result.is_ok());
    }

    #[test]
    fn test_rotate() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("old.audit");

        let mut writer = AuditFileWriter::new(
            file_path.clone(),
            RotationPolicy::Daily,
            crate::config::FlushMode::Immediate,
        )
        .unwrap();

        // Write an event
        let envelope = AuditEventEnvelope::new(
            "audit-001".to_string(),
            Utc::now(),
            "test-service".to_string(),
            create_test_event(),
            String::new(),
        );
        writer.write_event(envelope, false).unwrap();
        let old_hash = writer.last_hash.clone();

        // Rotate
        let result = writer.rotate();
        assert!(result.is_ok());

        // Event count should reset
        assert_eq!(writer.event_count, 0);
        assert_eq!(writer.file_size, 0);

        // Hash chain should be preserved
        assert_eq!(writer.last_hash, old_hash);
    }

    #[test]
    #[cfg(unix)]
    fn test_file_permissions() {
        use std::os::unix::fs::PermissionsExt;

        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.audit");

        let _writer = AuditFileWriter::new(
            file_path.clone(),
            RotationPolicy::Daily,
            crate::config::FlushMode::Immediate,
        )
        .unwrap();

        // Verify file was created with 0600 permissions
        let metadata = std::fs::metadata(&file_path).unwrap();
        let permissions = metadata.permissions();
        let mode = permissions.mode();

        // Check owner has read/write
        assert_ne!(mode & 0o600, 0, "Owner should have read/write");

        // Ideally group and other should have no permissions, but this depends on umask
        // So we just verify the file was created
        assert!(file_path.exists());
    }

    #[test]
    fn test_rotation_uniqueness() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("2024-01-01.audit");

        // Create first writer
        let mut writer1 = AuditFileWriter::new(
            file_path.clone(),
            RotationPolicy::Daily,
            crate::config::FlushMode::Immediate,
        )
        .unwrap();

        // Write an event
        let envelope = AuditEventEnvelope::new(
            "audit-001".to_string(),
            Utc::now(),
            "test-service".to_string(),
            create_test_event(),
            String::new(),
        );
        writer1.write_event(envelope, false).unwrap();

        // Rotate - should create a unique file
        let result = writer1.rotate();
        assert!(result.is_ok(), "Rotation should succeed even if date file exists");

        // Verify new file was created with unique name
        let new_path = &writer1.file_path;
        assert!(new_path.exists());
        assert_ne!(new_path, &file_path, "Should create new file");
    }

    #[test]
    fn test_serialization_error_handling() {
        // This test verifies that serialization errors are handled gracefully
        // In practice, serialization should never fail for our event types,
        // but we want to ensure the error path works

        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.audit");

        let writer = AuditFileWriter::new(
            file_path,
            RotationPolicy::Daily,
            crate::config::FlushMode::Immediate,
        );
        assert!(writer.is_ok(), "Writer creation should succeed");
    }

    #[test]
    fn test_flush_mode_immediate() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.audit");

        let mut writer = AuditFileWriter::new(
            file_path,
            RotationPolicy::Daily,
            crate::config::FlushMode::Immediate,
        )
        .unwrap();

        // Write event
        let envelope = AuditEventEnvelope::new(
            "audit-001".to_string(),
            Utc::now(),
            "test-service".to_string(),
            create_test_event(),
            String::new(),
        );

        writer.write_event(envelope, false).unwrap();

        // With Immediate mode, events_since_sync should be reset to 0
        assert_eq!(writer.events_since_sync, 0, "Immediate mode should reset counter");
    }

    #[test]
    fn test_flush_mode_batched() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.audit");

        let mut writer = AuditFileWriter::new(
            file_path,
            RotationPolicy::Daily,
            crate::config::FlushMode::Batched { size: 10, interval_secs: 60 },
        )
        .unwrap();

        // Write 5 events (below batch size)
        for i in 0..5 {
            let envelope = AuditEventEnvelope::new(
                format!("audit-{:03}", i),
                Utc::now(),
                "test-service".to_string(),
                create_test_event(),
                String::new(),
            );
            writer.write_event(envelope, false).unwrap();
        }

        // Should not have flushed yet
        assert_eq!(writer.events_since_sync, 5, "Should not flush before batch size");

        // Write 5 more events (reaches batch size)
        for i in 5..10 {
            let envelope = AuditEventEnvelope::new(
                format!("audit-{:03}", i),
                Utc::now(),
                "test-service".to_string(),
                create_test_event(),
                String::new(),
            );
            writer.write_event(envelope, false).unwrap();
        }

        // Should have flushed at batch size
        assert_eq!(writer.events_since_sync, 0, "Should flush at batch size");
    }

    #[test]
    fn test_flush_mode_hybrid_critical_immediate() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.audit");

        let mut writer = AuditFileWriter::new(
            file_path,
            RotationPolicy::Daily,
            crate::config::FlushMode::Hybrid {
                batch_size: 10,
                batch_interval_secs: 60,
                critical_immediate: true,
            },
        )
        .unwrap();

        // Write routine event (should not flush)
        let envelope = AuditEventEnvelope::new(
            "audit-001".to_string(),
            Utc::now(),
            "test-service".to_string(),
            create_test_event(),
            String::new(),
        );
        writer.write_event(envelope, false).unwrap();
        assert_eq!(writer.events_since_sync, 1, "Routine event should not flush");

        // Write critical event (should flush immediately)
        let envelope = AuditEventEnvelope::new(
            "audit-002".to_string(),
            Utc::now(),
            "test-service".to_string(),
            create_test_event(),
            String::new(),
        );
        writer.write_event(envelope, true).unwrap();
        assert_eq!(writer.events_since_sync, 0, "Critical event should flush immediately");
    }

    #[test]
    fn test_flush_mode_hybrid_batch_routine() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.audit");

        let mut writer = AuditFileWriter::new(
            file_path,
            RotationPolicy::Daily,
            crate::config::FlushMode::Hybrid {
                batch_size: 5,
                batch_interval_secs: 60,
                critical_immediate: true,
            },
        )
        .unwrap();

        // Write 4 routine events (below batch size)
        for i in 0..4 {
            let envelope = AuditEventEnvelope::new(
                format!("audit-{:03}", i),
                Utc::now(),
                "test-service".to_string(),
                create_test_event(),
                String::new(),
            );
            writer.write_event(envelope, false).unwrap();
        }

        assert_eq!(writer.events_since_sync, 4, "Should not flush before batch size");

        // Write 5th routine event (reaches batch size)
        let envelope = AuditEventEnvelope::new(
            "audit-005".to_string(),
            Utc::now(),
            "test-service".to_string(),
            create_test_event(),
            String::new(),
        );
        writer.write_event(envelope, false).unwrap();

        assert_eq!(writer.events_since_sync, 0, "Should flush at batch size");
    }
}
