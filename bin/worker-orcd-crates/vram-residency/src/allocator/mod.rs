//! VRAM allocation and management
//!
//! This module handles VRAM allocation, deallocation, and capacity tracking.

pub mod vram_manager;
mod cuda_allocator;

pub use vram_manager::VramManager;
