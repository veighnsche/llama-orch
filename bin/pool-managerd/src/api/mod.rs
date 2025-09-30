//! HTTP API for pool-managerd daemon.
//!
//! Routes:
//! - GET /health — daemon health check
//! - POST /pools/{id}/preload — spawn engine from PreparedEngine
//! - POST /pools/{id}/drain — drain pool
//! - POST /pools/{id}/reload — reload pool
//! - GET /pools/{id}/status — get pool status

pub mod auth;
pub mod routes;
