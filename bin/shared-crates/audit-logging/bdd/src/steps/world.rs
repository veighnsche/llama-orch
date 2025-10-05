//! BDD World for audit-logging tests

use audit_logging::{ActorInfo, AuditEvent, AuditResult, AuthMethod};
use chrono::{DateTime, Utc};
use cucumber::World;
use std::net::IpAddr;

#[derive(Debug, Default, World)]
pub struct BddWorld {
    /// Last validation result
    pub last_result: Option<Result<(), String>>,

    /// Current audit event being tested
    pub current_event: Option<AuditEvent>,

    /// Actor information for event construction
    pub actor: Option<ActorInfo>,

    /// Field values for event construction
    pub user_id: String,
    pub ip_addr: Option<IpAddr>,
    pub session_id: Option<String>,
    pub resource_id: String,
    pub resource_type: String,
    pub reason: String,
    pub details: String,
    pub path: String,
    pub endpoint: String,
    pub task_id: String,
    pub pool_id: String,
    pub node_id: String,
    pub model_ref: String,
    pub worker_id: String,
    pub shard_id: String,
    pub customer_id: String,

    /// Validation error message
    pub error_message: Option<String>,
}

impl BddWorld {
    /// Store validation result
    pub fn store_result(&mut self, result: Result<(), String>) {
        match &result {
            Ok(()) => {
                self.last_result = Some(Ok(()));
                self.error_message = None;
            }
            Err(e) => {
                self.last_result = Some(Err(e.clone()));
                self.error_message = Some(e.clone());
            }
        }
    }

    /// Check if last validation succeeded
    pub fn last_succeeded(&self) -> bool {
        matches!(self.last_result, Some(Ok(())))
    }

    /// Check if last validation failed
    pub fn last_failed(&self) -> bool {
        matches!(self.last_result, Some(Err(_)))
    }

    /// Get last error message
    pub fn get_last_error(&self) -> Option<&str> {
        self.error_message.as_deref()
    }

    /// Create a default actor for testing
    pub fn create_default_actor(&self) -> ActorInfo {
        ActorInfo {
            user_id: self.user_id.clone(),
            ip: self.ip_addr,
            auth_method: AuthMethod::BearerToken,
            session_id: self.session_id.clone(),
        }
    }

    /// Get current timestamp
    pub fn now(&self) -> DateTime<Utc> {
        Utc::now()
    }
}
