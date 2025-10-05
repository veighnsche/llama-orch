//! Q4_K Dequantization
//!
//! Converts Q4_K_M quantized weights to FP16 format.
//!
//! Based on GGML Q4_K format specification:
//! - Block size: 256 elements
//! - Block bytes: 144 bytes
//! - Sub-blocks: 8 × 32 elements
//!
//! Block structure:
//! - d: fp16 (2 bytes) - super scale
//! - dmin: fp16 (2 bytes) - super min-scale
//! - scales: 12 bytes - packed 6-bit indices for 8 sub-blocks
//! - qs: 128 bytes - 256 nibbles (4-bit values)
//!
//! Dequantization formula:
//!   scale_s = float(d) * sc_s
//!   min_s = float(dmin) * m_s
//!   y[i] = scale_s * q[i] + min_s
//!   where q[i] ∈ [0..15]

use half::f16;

const BLOCK_SIZE: usize = 256;
const BLOCK_BYTES: usize = 144;
const NUM_SUB_BLOCKS: usize = 8;
const SUB_BLOCK_SIZE: usize = 32;

/// Q4_K block structure (256 elements = 144 bytes)
#[repr(C, packed)]
struct Q4KBlock {
    d: u16,           // fp16 super scale
    dmin: u16,        // fp16 super min-scale
    scales: [u8; 12], // packed 6-bit scale/min indices
    qs: [u8; 128],    // 256 x 4-bit quantized values
}

/// Decode 6-bit scale and min indices from packed 12-byte array
///
/// The 12 bytes encode 8 pairs of (scale, min), each 6 bits.
/// Based on GGML's get_scale_min_k4 logic.
fn decode_scales_and_mins(scales: &[u8; 12]) -> ([u8; 8], [u8; 8]) {
    let mut sc = [0u8; 8];
    let mut m = [0u8; 8];
    
    // Unpack 6-bit values from the packed byte array
    // This follows the GGML bit-packing scheme
    sc[0] = scales[0] & 0x3F;
    sc[1] = scales[1] & 0x3F;
    sc[2] = ((scales[2] & 0x0F) << 2) | ((scales[0] >> 6) & 0x03);
    sc[3] = ((scales[3] & 0x0F) << 2) | ((scales[1] >> 6) & 0x03);
    sc[4] = ((scales[4] & 0x0F) << 2) | ((scales[2] >> 4) & 0x03);
    sc[5] = ((scales[5] & 0x0F) << 2) | ((scales[3] >> 4) & 0x03);
    sc[6] = ((scales[6] & 0x0F) << 2) | ((scales[4] >> 4) & 0x03);
    sc[7] = ((scales[7] & 0x0F) << 2) | ((scales[5] >> 4) & 0x03);
    
    m[0] = ((scales[8] & 0x0F) << 2) | ((scales[6] >> 4) & 0x03);
    m[1] = ((scales[9] & 0x0F) << 2) | ((scales[7] >> 4) & 0x03);
    m[2] = ((scales[10] & 0x0F) << 2) | ((scales[8] >> 4) & 0x03);
    m[3] = ((scales[11] & 0x0F) << 2) | ((scales[9] >> 4) & 0x03);
    m[4] = (scales[10] >> 4) & 0x0F;
    m[5] = (scales[11] >> 4) & 0x0F;
    m[6] = scales[2] >> 4;
    m[7] = scales[3] >> 4;
    
    (sc, m)
}

/// Dequantize a single Q4_K block (256 elements) to FP16
fn dequantize_block(block_bytes: &[u8], output: &mut [f16]) {
    assert_eq!(block_bytes.len(), BLOCK_BYTES);
    assert_eq!(output.len(), BLOCK_SIZE);
    
    // Parse block structure
    let d = f16::from_bits(u16::from_le_bytes([block_bytes[0], block_bytes[1]]));
    let dmin = f16::from_bits(u16::from_le_bytes([block_bytes[2], block_bytes[3]]));
    
    let mut scales = [0u8; 12];
    scales.copy_from_slice(&block_bytes[4..16]);
    
    let qs = &block_bytes[16..144];
    
    // Decode scale and min indices
    let (sc, m) = decode_scales_and_mins(&scales);
    
    // Dequantize each sub-block
    let d_f32 = d.to_f32();
    let dmin_f32 = dmin.to_f32();
    
    for s in 0..NUM_SUB_BLOCKS {
        let scale = d_f32 * (sc[s] as f32);
        let min_val = dmin_f32 * (m[s] as f32);
        
        // Each sub-block has 32 elements = 16 bytes (2 nibbles per byte)
        let qs_offset = s * 16;
        let out_offset = s * SUB_BLOCK_SIZE;
        
        for j in 0..16 {
            let packed = qs[qs_offset + j];
            let q0 = (packed & 0x0F) as f32;
            let q1 = (packed >> 4) as f32;
            
            output[out_offset + 2*j] = f16::from_f32(scale * q0 + min_val);
            output[out_offset + 2*j + 1] = f16::from_f32(scale * q1 + min_val);
        }
    }
}

/// Dequantize Q4_K tensor to FP16
///
/// # Arguments
/// - `input`: Q4_K quantized data
/// - `num_elements`: Total number of elements (must be multiple of 256)
///
/// # Returns
/// Vector of FP16 values
///
/// # Panics
/// Panics if num_elements is not a multiple of 256
pub fn dequantize_q4k(input: &[u8], num_elements: usize) -> Vec<f16> {
    assert_eq!(num_elements % BLOCK_SIZE, 0, "num_elements must be multiple of 256");
    
    let num_blocks = num_elements / BLOCK_SIZE;
    let expected_bytes = num_blocks * BLOCK_BYTES;
    
    assert_eq!(input.len(), expected_bytes, 
        "Input size mismatch: expected {} bytes for {} elements", 
        expected_bytes, num_elements);
    
    let mut output = vec![f16::ZERO; num_elements];
    
    for block_idx in 0..num_blocks {
        let block_start = block_idx * BLOCK_BYTES;
        let block_end = block_start + BLOCK_BYTES;
        let block_bytes = &input[block_start..block_end];
        
        let out_start = block_idx * BLOCK_SIZE;
        let out_end = out_start + BLOCK_SIZE;
        
        dequantize_block(block_bytes, &mut output[out_start..out_end]);
    }
    
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_block_size() {
        assert_eq!(std::mem::size_of::<Q4KBlock>(), BLOCK_BYTES);
    }
    
    #[test]
    fn test_decode_scales() {
        // Test with known pattern
        let scales = [0x3F, 0x3F, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        let (sc, m) = decode_scales_and_mins(&scales);
        
        // First two scales should be 0x3F (max 6-bit value)
        assert_eq!(sc[0], 0x3F);
        assert_eq!(sc[1], 0x3F);
    }
    
    #[test]
    fn test_dequantize_zero_block() {
        // Create a block of all zeros
        let block = vec![0u8; BLOCK_BYTES];
        let mut output = vec![f16::ZERO; BLOCK_SIZE];
        
        dequantize_block(&block, &mut output);
        
        // All outputs should be zero (0 * scale + 0 * min = 0)
        for val in output {
            assert_eq!(val, f16::ZERO);
        }
    }
}
