//! Model CLI actions
//!
//! TEAM-276: Extracted from main.rs

use clap::Subcommand;

#[derive(Subcommand)]
pub enum ModelAction {
    Download { model: String },
    List,
    Get { id: String },
    Delete { id: String },
}
