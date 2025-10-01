//! VRAM allocation and management
//!
//! This module handles VRAM allocation, deallocation, and capacity tracking.

pub mod vram_manager;
pub mod mock_allocator;
pub mod cuda_allocator;
pub mod capacity;

pub use vram_manager::VramManager;
pub use mock_allocator::MockVramAllocator;
pub use cuda_allocator::CudaVramAllocator;
pub use capacity::CapacityTracker;
