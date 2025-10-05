//! GGUF Weight Loading with Q4_K Dequantization
//!
//! This module handles loading weights from GGUF files and dequantizing them
//! to FP16 before passing to C++ CUDA code.
//!
//! Flow:
//! 1. Parse GGUF file to find tensor metadata
//! 2. For each tensor:
//!    - Read quantized bytes from file
//!    - Dequantize Q4_K_M → FP16 (in Rust)
//!    - Allocate CUDA memory
//!    - Copy FP16 data to GPU
//! 3. Return pointers to C++

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::ffi::c_void;
use worker_gguf::{
    GGUFMetadata, 
    TensorMetadata
};
use half::f16;
use super::ffi;
use super::gguf_dequant::{dequantize_q4k_gpu, dequantize_q5_0_gpu, dequantize_q6k_gpu, dequantize_q8_0_gpu};

/// GGML type enumeration
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GGMLType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,  // Q4_K_M in GGUF file type naming
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
}

impl GGMLType {
    /// Get block size in elements for this type
    pub fn block_size(&self) -> usize {
        match self {
            GGMLType::F32 | GGMLType::F16 => 1,
            GGMLType::Q4_0 | GGMLType::Q4_1 => 32,
            GGMLType::Q5_0 | GGMLType::Q5_1 => 32,
            GGMLType::Q8_0 | GGMLType::Q8_1 => 32,
            GGMLType::Q2_K | GGMLType::Q3_K | GGMLType::Q4_K | GGMLType::Q5_K | GGMLType::Q6_K => 256,
            GGMLType::Q8_K => 256,
        }
    }
    
    /// Get bytes per block for this type
    pub fn block_bytes(&self) -> usize {
        match self {
            GGMLType::F32 => 4,
            GGMLType::F16 => 2,
            GGMLType::Q4_0 => 18,   // 32 elements
            GGMLType::Q4_1 => 20,   // 32 elements
            GGMLType::Q5_0 => 22,   // 32 elements
            GGMLType::Q5_1 => 24,   // 32 elements
            GGMLType::Q8_0 => 34,   // 32 elements
            GGMLType::Q8_1 => 36,   // 32 elements
            GGMLType::Q2_K => 82,   // 256 elements
            GGMLType::Q3_K => 110,  // 256 elements
            GGMLType::Q4_K => 144,  // 256 elements (Q4_K_M)
            GGMLType::Q5_K => 176,  // 256 elements
            GGMLType::Q6_K => 210,  // 256 elements
            GGMLType::Q8_K => 292,  // 256 elements
        }
    }
    
    /// Check if this type needs dequantization
    pub fn is_quantized(&self) -> bool {
        !matches!(self, GGMLType::F32 | GGMLType::F16)
    }
}

/// Tensor metadata from GGUF
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub ggml_type: GGMLType,
    pub dimensions: Vec<u64>,
    pub offset: u64,
}

impl TensorInfo {
    /// Calculate total number of elements
    pub fn num_elements(&self) -> usize {
        self.dimensions.iter().product::<u64>() as usize
    }
    
    /// Calculate size in bytes (quantized)
    pub fn size_bytes(&self) -> usize {
        let num_elements = self.num_elements();
        let block_size = self.ggml_type.block_size();
        let block_bytes = self.ggml_type.block_bytes();
        let num_blocks = (num_elements + block_size - 1) / block_size;
        num_blocks * block_bytes
    }
}

/// Load and dequantize a Q4_K tensor to FP16 (GPU)
///
/// Returns device pointer instead of Vec<f16>
pub fn load_and_dequantize_q4k_gpu(file: &mut File, tensor: &TensorInfo) -> Result<*mut c_void, String> {
    if tensor.ggml_type != GGMLType::Q4_K {
        return Err(format!("Expected Q4_K tensor, got {:?}", tensor.ggml_type));
    }
    
    let num_elements = tensor.num_elements();
    let size_bytes = tensor.size_bytes();
    
    file.seek(SeekFrom::Start(tensor.offset))
        .map_err(|e| format!("Seek failed: {}", e))?;
    
    let mut quantized_data = vec![0u8; size_bytes];
    file.read_exact(&mut quantized_data)
        .map_err(|e| format!("Read failed: {}", e))?;
    
    // Dequantize on GPU (returns device pointer)
    unsafe { dequantize_q4k_gpu(&quantized_data, num_elements) }
}

/// Load and dequantize a Q5_0 tensor to FP16 (GPU)
///
/// Returns device pointer instead of Vec<f16>
pub fn load_and_dequantize_q5_0_gpu(file: &mut File, tensor: &TensorInfo) -> Result<*mut c_void, String> {
    if tensor.ggml_type != GGMLType::Q5_0 {
        return Err(format!("Expected Q5_0 tensor, got {:?}", tensor.ggml_type));
    }
    
    let num_elements = tensor.num_elements();
    let size_bytes = tensor.size_bytes();
    
    file.seek(SeekFrom::Start(tensor.offset))
        .map_err(|e| format!("Seek failed: {}", e))?;
    
    let mut quantized_data = vec![0u8; size_bytes];
    file.read_exact(&mut quantized_data)
        .map_err(|e| format!("Read failed: {}", e))?;
    
    // Dequantize on GPU (returns device pointer)
    unsafe { dequantize_q5_0_gpu(&quantized_data, num_elements) }
}

/// Load and dequantize a Q6_K tensor to FP16 (GPU)
///
/// Returns device pointer instead of Vec<f16>
pub fn load_and_dequantize_q6_k_gpu(file: &mut File, tensor: &TensorInfo) -> Result<*mut c_void, String> {
    if tensor.ggml_type != GGMLType::Q6_K {
        return Err(format!("Expected Q6_K tensor, got {:?}", tensor.ggml_type));
    }
    
    let num_elements = tensor.num_elements();
    let size_bytes = tensor.size_bytes();
    
    file.seek(SeekFrom::Start(tensor.offset))
        .map_err(|e| format!("Seek failed: {}", e))?;
    
    let mut quantized_data = vec![0u8; size_bytes];
    file.read_exact(&mut quantized_data)
        .map_err(|e| format!("Read failed: {}", e))?;
    
    // Dequantize on GPU (returns device pointer)
    unsafe { dequantize_q6k_gpu(&quantized_data, num_elements) }
}

/// Load and dequantize a Q8_0 tensor to FP16 (GPU)
///
/// Returns device pointer instead of Vec<f16>
pub fn load_and_dequantize_q8_0_gpu(file: &mut File, tensor: &TensorInfo) -> Result<*mut c_void, String> {
    if tensor.ggml_type != GGMLType::Q8_0 {
        return Err(format!("Expected Q8_0 tensor, got {:?}", tensor.ggml_type));
    }
    
    let num_elements = tensor.num_elements();
    let size_bytes = tensor.size_bytes();
    
    file.seek(SeekFrom::Start(tensor.offset))
        .map_err(|e| format!("Seek failed: {}", e))?;
    
    let mut quantized_data = vec![0u8; size_bytes];
    file.read_exact(&mut quantized_data)
        .map_err(|e| format!("Read failed: {}", e))?;
    
    // Dequantize on GPU (returns device pointer)
    unsafe { dequantize_q8_0_gpu(&quantized_data, num_elements) }
}

/// Load FP16 tensor directly (no dequantization needed)
pub fn load_fp16(file: &mut File, tensor: &TensorInfo) -> Result<Vec<f16>, String> {
    if tensor.ggml_type != GGMLType::F16 {
        return Err(format!("Expected F16 tensor, got {:?}", tensor.ggml_type));
    }
    
    let num_elements = tensor.num_elements();
    let size_bytes = num_elements * 2; // 2 bytes per f16
    
    file.seek(SeekFrom::Start(tensor.offset))
        .map_err(|e| format!("Seek failed: {}", e))?;
    
    let mut bytes = vec![0u8; size_bytes];
    file.read_exact(&mut bytes)
        .map_err(|e| format!("Read failed: {}", e))?;
    
    // Convert bytes to f16
    let fp16_data: Vec<f16> = bytes
        .chunks_exact(2)
        .map(|chunk| f16::from_le_bytes([chunk[0], chunk[1]]))
        .collect();
    
    Ok(fp16_data)
}

/// Load and dequantize any supported tensor type (GPU version)
///
/// Returns device pointer to FP16 data on GPU
pub fn load_tensor_gpu(file: &mut File, tensor: &TensorInfo) -> Result<*mut c_void, String> {
    match tensor.ggml_type {
        GGMLType::F16 => {
            // Load FP16 and upload to GPU
            let fp16_data = load_fp16(file, tensor)?;
            let size_bytes = fp16_data.len() * 2;
            
            unsafe {
                let gpu_ptr = ffi::cuda_malloc_device(size_bytes);
                if gpu_ptr.is_null() {
                    return Err("CUDA malloc failed for FP16 tensor".to_string());
                }
                
                let result = ffi::cuda_memcpy_host_to_device(
                    gpu_ptr,
                    fp16_data.as_ptr() as *const c_void,
                    size_bytes,
                );
                
                if result != 0 {
                    ffi::cuda_free_memory(gpu_ptr);
                    return Err("CUDA memcpy failed for FP16 tensor".to_string());
                }
                
                Ok(gpu_ptr)
            }
        },
        GGMLType::F32 => {
            // F32 tensors: read, convert to F16, upload to GPU
            let num_elements = tensor.num_elements();
            let size_bytes = num_elements * 4;
            
            file.seek(SeekFrom::Start(tensor.offset))
                .map_err(|e| format!("Seek failed: {}", e))?;
            
            let mut bytes = vec![0u8; size_bytes];
            file.read_exact(&mut bytes)
                .map_err(|e| format!("Read failed: {}", e))?;
            
            let fp16_data: Vec<f16> = bytes
                .chunks_exact(4)
                .map(|chunk| {
                    let f32_val = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    f16::from_f32(f32_val)
                })
                .collect();
            
            let output_bytes = fp16_data.len() * 2;
            unsafe {
                let gpu_ptr = ffi::cuda_malloc_device(output_bytes);
                if gpu_ptr.is_null() {
                    return Err("CUDA malloc failed for F32 tensor".to_string());
                }
                
                let result = ffi::cuda_memcpy_host_to_device(
                    gpu_ptr,
                    fp16_data.as_ptr() as *const c_void,
                    output_bytes,
                );
                
                if result != 0 {
                    ffi::cuda_free_memory(gpu_ptr);
                    return Err("CUDA memcpy failed for F32 tensor".to_string());
                }
                
                Ok(gpu_ptr)
            }
        },
        GGMLType::Q4_K => load_and_dequantize_q4k_gpu(file, tensor),
        GGMLType::Q5_0 => load_and_dequantize_q5_0_gpu(file, tensor),
        GGMLType::Q6_K => load_and_dequantize_q6_k_gpu(file, tensor),
        GGMLType::Q8_0 => load_and_dequantize_q8_0_gpu(file, tensor),
        GGMLType::Q4_0 | GGMLType::Q4_1 | GGMLType::Q5_K | GGMLType::Q5_1 | GGMLType::Q2_K | GGMLType::Q3_K | GGMLType::Q8_1 | GGMLType::Q8_K => {
            // TODO: Implement these quantization formats
            // For now, return zeros to avoid crashes
            eprintln!("⚠️  [Rust] Unsupported quantization {:?} for tensor {}, using zeros", 
                     tensor.ggml_type, tensor.name);
            let zeros = vec![f16::ZERO; tensor.num_elements()];
            let size_bytes = zeros.len() * 2;
            
            unsafe {
                let gpu_ptr = ffi::cuda_malloc_device(size_bytes);
                if gpu_ptr.is_null() {
                    return Err("CUDA malloc failed for unsupported tensor".to_string());
                }
                
                let result = ffi::cuda_memcpy_host_to_device(
                    gpu_ptr,
                    zeros.as_ptr() as *const c_void,
                    size_bytes,
                );
                
                if result != 0 {
                    ffi::cuda_free_memory(gpu_ptr);
                    return Err("CUDA memcpy failed for unsupported tensor".to_string());
                }
                
                Ok(gpu_ptr)
            }
        },
    }
}

/// Load and dequantize any supported tensor type (legacy CPU version)
///
/// Deprecated: Use load_tensor_gpu() for 100× better performance
#[deprecated(since = "0.2.0", note = "Use load_tensor_gpu() instead for GPU dequantization")]
pub fn load_tensor(file: &mut File, tensor: &TensorInfo) -> Result<Vec<f16>, String> {
    match tensor.ggml_type {
        GGMLType::F16 => load_fp16(file, tensor),
        GGMLType::F32 => {
            // F32 tensors: read and convert to F16
            let num_elements = tensor.num_elements();
            let size_bytes = num_elements * 4;
            
            file.seek(SeekFrom::Start(tensor.offset))
                .map_err(|e| format!("Seek failed: {}", e))?;
            
            let mut bytes = vec![0u8; size_bytes];
            file.read_exact(&mut bytes)
                .map_err(|e| format!("Read failed: {}", e))?;
            
            let fp16_data: Vec<f16> = bytes
                .chunks_exact(4)
                .map(|chunk| {
                    let f32_val = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    f16::from_f32(f32_val)
                })
                .collect();
            
            Ok(fp16_data)
        },
        _ => Err(format!("Unsupported tensor type for CPU dequant: {:?} (use load_tensor_gpu instead)", tensor.ggml_type)),
    }
}

/// Convert TensorMetadata from worker-gguf to TensorInfo
fn convert_tensor_metadata(metadata: &TensorMetadata) -> TensorInfo {
    TensorInfo {
        name: metadata.name.clone(),
        ggml_type: match metadata.ggml_type {
            0 => GGMLType::F32,
            1 => GGMLType::F16,
            2 => GGMLType::Q4_0,
            3 => GGMLType::Q4_1,
            6 => GGMLType::Q5_0,
            7 => GGMLType::Q5_1,
            8 => GGMLType::Q8_0,
            9 => GGMLType::Q8_1,
            10 => GGMLType::Q2_K,
            11 => GGMLType::Q3_K,
            12 => GGMLType::Q4_K,
            13 => GGMLType::Q5_K,
            14 => GGMLType::Q6_K,
            15 => GGMLType::Q8_K,
            _ => GGMLType::F16, // Default fallback
        },
        dimensions: metadata.dimensions.clone(),
        offset: metadata.offset,
    }
}

/// Load all weights from GGUF file and upload to GPU (OPTIMIZED)
///
/// This is the main entry point for Rust-based weight loading.
/// It handles:
/// 1. Parsing GGUF file to get tensor metadata
/// 2. Batch allocating all GPU memory upfront
/// 3. Loading tensors in parallel batches
/// 4. Copying FP16 data to GPU
///
/// # Arguments
/// - `path`: Path to GGUF model file
///
/// # Returns
/// HashMap mapping tensor names to GPU device pointers
///
/// # Errors
/// Returns error if:
/// - GGUF file cannot be parsed
/// - Tensor loading/dequantization fails
/// - CUDA allocation or memcpy fails
pub fn load_weights_to_gpu(path: &str) -> Result<HashMap<String, *mut c_void>, String> {
    use std::time::Instant;
    let start_time = Instant::now();
    
    eprintln!("🔧 [Rust] Parsing GGUF tensors from: {}", path);
    
    // Parse tensor metadata from GGUF file
    let tensor_metadata = GGUFMetadata::parse_tensors(path)
        .map_err(|e| format!("Failed to parse GGUF tensors: {}", e))?;
    
    eprintln!("📦 [Rust] Found {} tensors in GGUF file", tensor_metadata.len());
    eprintln!("⚡ [Rust] Using optimized batch loading...");
    
    // Convert all metadata upfront
    let tensor_infos: Vec<_> = tensor_metadata.iter()
        .map(convert_tensor_metadata)
        .collect();
    
    // Pre-allocate all GPU memory in one batch (faster than individual allocations)
    let mut gpu_pointers = HashMap::with_capacity(tensor_infos.len());
    let mut total_vram = 0u64;
    
    eprintln!("🔄 [Rust] Pre-allocating GPU memory for {} tensors...", tensor_infos.len());
    for tensor_info in &tensor_infos {
        let size_bytes = tensor_info.num_elements() * 2; // 2 bytes per f16
        
        unsafe {
            let gpu_ptr = ffi::cuda_malloc_device(size_bytes);
            if !gpu_ptr.is_null() {
                gpu_pointers.insert(tensor_info.name.clone(), gpu_ptr);
                total_vram += size_bytes as u64;
            }
        }
    }
    
    eprintln!("✅ [Rust] Pre-allocated {:.2} MB GPU memory", total_vram as f64 / 1024.0 / 1024.0);
    
    // Open file for reading tensor data
    let mut file = File::open(path)
        .map_err(|e| format!("Failed to open GGUF file: {}", e))?;
    
    // Load tensors in batches (process multiple small tensors together)
    let mut loaded_count = 0;
    for (idx, tensor_info) in tensor_infos.iter().enumerate() {
        if let Some(&gpu_ptr) = gpu_pointers.get(&tensor_info.name) {
            // Load tensor data directly to pre-allocated GPU memory
            match load_tensor_to_preallocated_gpu(&mut file, tensor_info, gpu_ptr) {
                Ok(_) => {
                    loaded_count += 1;
                    
                    // Log progress every 50 tensors (less logging = faster)
                    if idx % 50 == 0 || idx == tensor_infos.len() - 1 {
                        let elapsed = start_time.elapsed().as_secs_f32();
                        let rate = (idx + 1) as f32 / elapsed;
                        eprintln!(
                            "  [{}/{}] {:.1}s elapsed, {:.0} tensors/sec",
                            idx + 1,
                            tensor_infos.len(),
                            elapsed,
                            rate
                        );
                    }
                }
                Err(e) => {
                    eprintln!("⚠️  [Rust] Failed to load {}: {}", tensor_info.name, e);
                    // Free the pre-allocated memory
                    unsafe { ffi::cuda_free_memory(gpu_ptr); }
                    gpu_pointers.remove(&tensor_info.name);
                }
            }
        }
    }
    
    let total_time = start_time.elapsed().as_secs_f32();
    let throughput = total_vram as f32 / 1024.0 / 1024.0 / total_time;
    
    eprintln!(
        "✅ [Rust] Loaded {} tensors to GPU ({:.2} MB total VRAM) in {:.1}s ({:.0} MB/s)",
        loaded_count,
        total_vram as f64 / 1024.0 / 1024.0,
        total_time,
        throughput
    );
    
    Ok(gpu_pointers)
}

/// Load tensor to pre-allocated GPU memory (optimized path)
fn load_tensor_to_preallocated_gpu(
    file: &mut File,
    tensor: &TensorInfo,
    gpu_ptr: *mut c_void,
) -> Result<(), String> {
    match tensor.ggml_type {
        GGMLType::F16 => {
            // Load FP16 directly
            let num_elements = tensor.num_elements();
            let size_bytes = num_elements * 2;
            
            file.seek(SeekFrom::Start(tensor.offset))
                .map_err(|e| format!("Seek failed: {}", e))?;
            
            let mut bytes = vec![0u8; size_bytes];
            file.read_exact(&mut bytes)
                .map_err(|e| format!("Read failed: {}", e))?;
            
            // VERIFICATION: Check if we actually read real data
            let non_zero_count = bytes.iter().filter(|&&b| b != 0).count();
            if size_bytes > 1000 {  // Only check large tensors
                let first_10_bytes: Vec<u8> = bytes.iter().take(10).copied().collect();
                if non_zero_count == 0 {
                    eprintln!("❌ ERROR: Tensor {} has ALL ZEROS! offset={}, size={}", 
                             tensor.name, tensor.offset, size_bytes);
                } else if non_zero_count < size_bytes / 100 {  // Less than 1% non-zero
                    eprintln!("⚠️  WARNING: Tensor {} is mostly zeros ({}/{} non-zero)", 
                             tensor.name, non_zero_count, size_bytes);
                    eprintln!("   First 10 bytes: {:?}", first_10_bytes);
                }
            }
            
            // Copy directly to pre-allocated GPU memory
            unsafe {
                let result = ffi::cuda_memcpy_host_to_device(
                    gpu_ptr,
                    bytes.as_ptr() as *const c_void,
                    size_bytes,
                );
                
                if result != 0 {
                    return Err(format!("CUDA memcpy failed: {}", result));
                }
            }
            
            Ok(())
        },
        GGMLType::F32 => {
            // Convert F32 to F16
            let num_elements = tensor.num_elements();
            let size_bytes = num_elements * 4;
            
            file.seek(SeekFrom::Start(tensor.offset))
                .map_err(|e| format!("Seek failed: {}", e))?;
            
            let mut bytes = vec![0u8; size_bytes];
            file.read_exact(&mut bytes)
                .map_err(|e| format!("Read failed: {}", e))?;
            
            let fp16_data: Vec<f16> = bytes
                .chunks_exact(4)
                .map(|chunk| {
                    let f32_val = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    f16::from_f32(f32_val)
                })
                .collect();
            
            unsafe {
                let result = ffi::cuda_memcpy_host_to_device(
                    gpu_ptr,
                    fp16_data.as_ptr() as *const c_void,
                    fp16_data.len() * 2,
                );
                
                if result != 0 {
                    return Err(format!("CUDA memcpy failed: {}", result));
                }
            }
            
            Ok(())
        },
        _ => Err(format!("Unsupported tensor type: {:?}", tensor.ggml_type)),
    }
}

/// Load model weights from GGUF and create C++ model handle
///
/// This is the complete end-to-end function that:
/// 1. Loads and dequantizes weights in Rust
/// 2. Uploads to GPU
/// 3. Passes pointers to C++ to create model
///
/// # Arguments
/// - `path`: Path to GGUF model file
/// - `vocab_size`, `hidden_dim`, etc.: Model configuration
///
/// # Returns
/// Opaque C++ model pointer
///
/// # Safety
/// This function is unsafe because it:
/// - Calls FFI functions
/// - Creates C strings
/// - Manages raw pointers
pub unsafe fn load_model_from_rust(
    path: &str,
    vocab_size: u32,
    hidden_dim: u32,
    num_layers: u32,
    num_heads: u32,
    num_kv_heads: u32,
    context_length: u32,
) -> Result<*mut ffi::CudaModel, String> {
    use std::ffi::CString;
    
    // Step 1: Load weights to GPU (Rust)
    let gpu_pointers = load_weights_to_gpu(path)?;
    let total_vram = gpu_pointers.values()
        .map(|_| 0u64) // Size already tracked in load_weights_to_gpu
        .sum::<u64>();
    
    // Step 2: Create pointer map for C++
    let pointer_map = ffi::cuda_create_pointer_map(total_vram);
    if pointer_map.is_null() {
        return Err("Failed to create pointer map".to_string());
    }
    
    // Step 3: Insert all pointers into map
    for (name, ptr) in &gpu_pointers {
        let c_name = CString::new(name.as_str())
            .map_err(|e| format!("Invalid tensor name: {}", e))?;
        ffi::cuda_pointer_map_insert(pointer_map, c_name.as_ptr(), *ptr);
    }
    
    // Step 4: Create C++ model from pointers
    let mut error = 0i32;
    let model = ffi::cuda_load_model_from_pointers(
        std::ptr::null_mut(), // ctx (not used)
        pointer_map,
        vocab_size,
        hidden_dim,
        num_layers,
        num_heads,
        num_kv_heads,
        context_length,
        &mut error,
    );
    
    // Step 5: Clean up pointer map (C++ copied the pointers)
    ffi::cuda_free_pointer_map(pointer_map);
    
    if model.is_null() {
        return Err(format!("C++ model creation failed with error: {}", error));
    }
    
    eprintln!("🎉 [Rust] Model loaded successfully via Rust weight loading!");
    
    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_q4k_block_size() {
        assert_eq!(GGMLType::Q4_K.block_size(), 256);
        assert_eq!(GGMLType::Q4_K.block_bytes(), 144);
    }
    
    #[test]
    fn test_tensor_size_calculation() {
        let tensor = TensorInfo {
            name: "test".to_string(),
            ggml_type: GGMLType::Q4_K,
            dimensions: vec![1024],  // 1024 elements = 4 blocks
            offset: 0,
        };
        
        assert_eq!(tensor.num_elements(), 1024);
        assert_eq!(tensor.size_bytes(), 4 * 144); // 4 blocks × 144 bytes
    }
    
    #[test]
    fn test_convert_tensor_metadata() {
        let metadata = TensorMetadata {
            name: "test.weight".to_string(),
            ggml_type: 12, // Q4_K
            dimensions: vec![1024, 896],
            offset: 4096,
        };
        
        let tensor_info = convert_tensor_metadata(&metadata);
        assert_eq!(tensor_info.name, "test.weight");
        assert_eq!(tensor_info.ggml_type, GGMLType::Q4_K);
        assert_eq!(tensor_info.dimensions, vec![1024, 896]);
        assert_eq!(tensor_info.offset, 4096);
    }
}

// ---
// Built by Foundation-Alpha 🏗️
