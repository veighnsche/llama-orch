//! GGUF security limits
//!
//! Enforces resource limits to prevent DoS attacks.

/// GGUF magic number ("GGUF" in little-endian)
pub const GGUF_MAGIC: u32 = 0x46554747;

/// Minimum GGUF header size (bytes)
pub const MIN_HEADER_SIZE: usize = 12;

/// Maximum number of tensors (prevents resource exhaustion)
///
/// Implements GGUF-001 from 20_security.md
pub const MAX_TENSORS: usize = 10_000;

/// Maximum file size (bytes)
///
/// Implements GGUF-002 from 20_security.md
pub const MAX_FILE_SIZE: usize = 100_000_000_000; // 100GB

/// Maximum string length (bytes)
///
/// Implements GGUF-003 from 20_security.md
pub const MAX_STRING_LEN: usize = 65536; // 64KB

/// Maximum metadata key-value pairs
///
/// Implements GGUF-004 from 20_security.md
pub const MAX_METADATA_PAIRS: usize = 1000;

// TODO(Post-M0): Add configurable limits
// pub struct GgufLimits {
//     pub max_tensors: usize,
//     pub max_file_size: usize,
//     pub max_string_len: usize,
//     pub max_metadata_pairs: usize,
// }
//
// impl Default for GgufLimits {
//     fn default() -> Self {
//         Self {
//             max_tensors: MAX_TENSORS,
//             max_file_size: MAX_FILE_SIZE,
//             max_string_len: MAX_STRING_LEN,
//             max_metadata_pairs: MAX_METADATA_PAIRS,
//         }
//     }
// }
