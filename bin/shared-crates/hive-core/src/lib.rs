//! Pool Core - Shared types and logic for pool management
//!
//! Created by: TEAM-022

pub mod catalog;
pub mod error;
pub mod worker;

pub use catalog::{ModelCatalog, ModelEntry};
pub use error::{PoolError, Result};
pub use worker::{Backend, WorkerInfo};
