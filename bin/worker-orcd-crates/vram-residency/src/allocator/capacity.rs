//! VRAM capacity tracking
//!
//! Tracks VRAM usage and enforces capacity limits.

use crate::error::{Result, VramError};

/// Capacity tracker
///
/// Tracks VRAM usage and enforces limits.
pub struct CapacityTracker {
    total_vram: usize,
    used_vram: usize,
    max_model_size: usize,
}

impl CapacityTracker {
    /// Create new capacity tracker
    pub fn new(total_vram: usize, max_model_size: usize) -> Self {
        Self {
            total_vram,
            used_vram: 0,
            max_model_size,
        }
    }
    
    /// Check if allocation is possible
    pub fn can_allocate(&self, size: usize) -> bool {
        let total_needed = self.used_vram.saturating_add(size);
        total_needed <= self.total_vram && size <= self.max_model_size
    }
    
    /// Reserve capacity
    pub fn reserve(&mut self, size: usize) -> Result<()> {
        if size > self.max_model_size {
            return Err(VramError::InvalidInput(format!(
                "Model size {} exceeds maximum {}",
                size, self.max_model_size
            )));
        }
        
        let total_needed = self.used_vram.saturating_add(size);
        if total_needed > self.total_vram {
            return Err(VramError::InsufficientVram(
                size,
                self.total_vram.saturating_sub(self.used_vram),
            ));
        }
        
        self.used_vram = total_needed;
        Ok(())
    }
    
    /// Release capacity
    pub fn release(&mut self, size: usize) {
        self.used_vram = self.used_vram.saturating_sub(size);
    }
    
    /// Get available VRAM
    pub fn available(&self) -> usize {
        self.total_vram.saturating_sub(self.used_vram)
    }
    
    /// Get used VRAM
    pub fn used(&self) -> usize {
        self.used_vram
    }
    
    /// Get total VRAM
    pub fn total(&self) -> usize {
        self.total_vram
    }
}
