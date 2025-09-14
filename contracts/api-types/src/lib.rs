//! Generated API types crate (pre-code). Types are generated via `cargo xtask regen-openapi`.

#[allow(dead_code)]
pub mod generated;

#[allow(dead_code)]
pub mod generated_control;

pub use generated::*;
pub mod control {
    pub use crate::generated_control::*;
}
