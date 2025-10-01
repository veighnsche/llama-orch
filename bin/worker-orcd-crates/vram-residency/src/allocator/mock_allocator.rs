//! Mock VRAM allocator for testing
//!
//! Simulates VRAM allocation without requiring a GPU.

use crate::error::{Result, VramError};
use std::collections::HashMap;

/// Mock VRAM allocator
///
/// Provides identical semantics to real CUDA VRAM allocation
/// but uses system RAM for testing.
pub struct MockVramAllocator {
    /// Simulated VRAM allocations (ptr -> data)
    allocations: HashMap<usize, Vec<u8>>,
    
    /// Total VRAM capacity
    total_vram: usize,
    
    /// Currently used VRAM
    used_vram: usize,
    
    /// Next pointer ID
    next_ptr: usize,
}

impl MockVramAllocator {
    /// Create new mock allocator
    pub fn new(total_vram: usize) -> Self {
        Self {
            allocations: HashMap::new(),
            total_vram,
            used_vram: 0,
            next_ptr: 0x1000, // Start at non-zero (like real CUDA)
        }
    }
    
    /// Allocate mock VRAM
    pub fn allocate(&mut self, size: usize) -> Result<usize> {
        let total_needed = self.used_vram.saturating_add(size);
        if total_needed > self.total_vram {
            return Err(VramError::InsufficientVram(
                size,
                self.total_vram.saturating_sub(self.used_vram),
            ));
        }
        
        let ptr = self.next_ptr;
        self.next_ptr = self.next_ptr.saturating_add(1);
        self.allocations.insert(ptr, vec![0u8; size]);
        self.used_vram = self.used_vram.saturating_add(size);
        
        Ok(ptr)
    }
    
    /// Write to mock VRAM
    pub fn write(&mut self, ptr: usize, offset: usize, data: &[u8]) -> Result<()> {
        let allocation = self
            .allocations
            .get_mut(&ptr)
            .ok_or(VramError::IntegrityViolation)?;
        
        let end = offset
            .checked_add(data.len())
            .ok_or(VramError::IntegrityViolation)?;
        
        if end > allocation.len() {
            return Err(VramError::IntegrityViolation);
        }
        
        allocation[offset..end].copy_from_slice(data);
        Ok(())
    }
    
    /// Read from mock VRAM
    pub fn read(&self, ptr: usize, offset: usize, len: usize) -> Result<Vec<u8>> {
        let allocation = self
            .allocations
            .get(&ptr)
            .ok_or(VramError::IntegrityViolation)?;
        
        let end = offset
            .checked_add(len)
            .ok_or(VramError::IntegrityViolation)?;
        
        if end > allocation.len() {
            return Err(VramError::IntegrityViolation);
        }
        
        Ok(allocation[offset..end].to_vec())
    }
    
    /// Deallocate mock VRAM
    pub fn deallocate(&mut self, ptr: usize) -> Result<()> {
        let allocation = self
            .allocations
            .remove(&ptr)
            .ok_or(VramError::IntegrityViolation)?;
        
        self.used_vram = self.used_vram.saturating_sub(allocation.len());
        Ok(())
    }
    
    /// Get available VRAM
    pub fn available_vram(&self) -> usize {
        self.total_vram.saturating_sub(self.used_vram)
    }
}
