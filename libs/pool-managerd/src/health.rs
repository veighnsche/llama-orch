//! Health and readiness (planning-only).

#[derive(Debug, Clone)]
pub struct HealthStatus {
    pub live: bool,
    pub ready: bool,
}
