````markdown name=Q8_0_FORMAT_SPEC.md
# Q8_0 Quantization Format Specification

Summary
- Block size (elements): 32 (QK8_0 = 32)
- Storage per block: 34 bytes
  - d: 1 fp16 scale (2 bytes)
  - qs: 32 signed 8-bit integers (32 bytes)
- Total bytes per 32-element block: 2 + 32 = 34

Binary layout per block (contiguous)
- [0..1): d (fp16)
- [2..34): qs[32] (int8)

Dequantization
- Each output value y[i] is:
  - y[i] = d * qs[i]
- Where:
  - d = fp16_to_float(block.d)
  - qs[i] ∈ int8, i in [0, 31]

Pseudocode
```cpp
struct Q8_0_Block {
    uint16_t d_fp16;     // 2 bytes
    int8_t   qs[32];     // 32 bytes
};

inline float fp16_to_f32(uint16_t h);

void dequantize_q8_0_block(const Q8_0_Block &b, float out[32]) {
    const float d = fp16_to_f32(b.d_fp16);
    for (int i = 0; i < 32; ++i) {
        out[i] = d * float(b.qs[i]);
    }
}
```

Notes
- This is the simplest per-block format: symmetric, no per-subgroup bias/min.
- Reference patterns:
  - ggml treats GGML_TYPE_Q8_0 with block size QK8_0=32 and dequantization that multiplies int8 by a single fp16 scale.
````

````markdown name=Q5_0_FORMAT_SPEC.md
# Q5_0 Quantization Format Specification

Summary
- Block size (elements): 32 (QK5_0 = 32)
- Storage per block: 22 bytes
  - d: 1 fp16 scale (2 bytes)
  - qh: 4 bytes (bitmask of high 1-bit per element)
  - qs: 16 bytes (low 4-bit “nibbles”, 2 per byte)
- Total bytes per 32-element block: 2 + 4 + 16 = 22

Binary layout per block (contiguous)
- [0..1): d (fp16)
- [2..6): qh[4] (uint8) → 32 high bits, one per element
- [6..22): qs[16] (uint8) → low nibbles; each byte packs 2 elements
  - For element i in [0..31]:
    - If i < 16: qs[i_byte = i] low nibble is element i, high nibble is element i+16
    - Concretely:
      - low4(i) = (i < 16) ? (qs[i] & 0x0F) : (qs[i - 16] >> 4)
    - high1(i) = ((qh_bits >> i) & 0x1)

Reconstructing 5-bit signed value
- Combine:
  - q5_u = low4(i) | (high1(i) << 4)  // 0..31
  - q5_s = int(q5_u) - 16             // signed in [-16..15]

Dequantization
- y[i] = d * q5_s
- Where d = fp16_to_float(block.d)

Pseudocode
```cpp
struct Q5_0_Block {
    uint16_t d_fp16;   // 2 bytes
    uint8_t  qh[4];    // 4 bytes (32 high bits)
    uint8_t  qs[16];   // 16 bytes (low nibbles)
};

inline float fp16_to_f32(uint16_t h);

inline uint32_t load_u32_le(const uint8_t qh[4]) {
    return uint32_t(qh[0]) | (uint32_t(qh[1]) << 8) | (uint32_t(qh[2]) << 16) | (uint32_t(qh[3]) << 24);
}

void dequantize_q5_0_block(const Q5_0_Block &b, float out[32]) {
    const float d = fp16_to_f32(b.d_fp16);
    const uint32_t qh_bits = load_u32_le(b.qh);

    auto low4 = [&](int i) -> int {
        return (i < 16) ? (b.qs[i] & 0x0F) : (b.qs[i - 16] >> 4);
    };

    for (int i = 0; i < 32; ++i) {
        const int hi = (qh_bits >> i) & 0x1;
        const int q5 = (low4(i) | (hi << 4)) - 16; // [-16..15]
        out[i] = d * float(q5);
    }
}
```

Notes
- Matches the “5-bit with single scale” (no per-block min) layout.
- Equivalent reconstruction flow appears in ggml GPU/CPU paths: combine low nibble and one high bit, subtract 16, multiply by scale `d`.
````

````markdown name=Q6_K_FORMAT_SPEC.md
# Q6_K Quantization Format Specification

Summary
- Family: “K-quant” (grouped, per-subgroup integer scales)
- Block size (elements): 256 (QK_K = 256)
- Storage per block: 210 bytes
  - d: 1 fp16 super-scale (2 bytes)
  - ql: 128 bytes (low 4-bit per element; 2 elements per byte)
  - qh: 64 bytes (two high bitplanes; 2 bits per element packed)
  - scales: 16 bytes (uint8 per 16-element subgroup)
- Total bytes per 256-element block: 2 + 128 + 64 + 16 = 210

Block/sub-block structure
- Elements per block: 256
- Sub-blocks: 16 sub-blocks × 16 elements each
  - For sub-block g ∈ [0..15]:
    - scales[g] is an integer (0..255). Effective scale for that sub-block is:
      - scale_g = d * float(scales[g])
    - For element index e in sub-block g (e = g*16 + t, t ∈ [0..15]):
      - ql provides low 4 bits
      - qh provides 2 high bits via bitplanes
      - Reconstructed 6-bit unsigned q6_u in [0..63]:
        - q6_u = low4 | (hi2 << 4)
      - Signed value centered at zero:
        - q6_s = int(q6_u) - 32  // [-32..31]

Bit packing
- ql (128 bytes): low4 bits for all 256 elements; 2 values per byte
  - Indexing pattern can be implemented by addressing sub-block g and position t:
    - Byte index within ql for (g, t): ql_idx = g*8 + (t/2)
    - If (t % 2 == 0): low4 = ql[ql_idx] & 0x0F
      else:            low4 = (ql[ql_idx] >> 4) & 0x0F
- qh (64 bytes): two high bitplanes (2 bits per element), packed across bytes
  - Treat qh as two parallel 1-bit planes over the 256 positions:
    - For sub-block g and position t:
      - Plane 0 bit: b0 = (qh[ g*4 + (t % 8) ] >> (t / 8)    ) & 0x1
      - Plane 1 bit: b1 = (qh[ g*4 + (t % 8) ] >> (t / 8 + 2)) & 0x1
    - hi2 = b0 | (b1 << 1)
  - Equivalent formulations exist; follow your implementation convention as long as it reproduces ggml’s mapping.

Dequantization
- For each element i (i = g*16 + t):
  - y[i] = scale_g * q6_s
  - Where:
    - d = fp16_to_float(block.d)
    - scale_g = d * float(scales[g])
    - q6_s = ((low4(i) | (hi2(i) << 4)) - 32)

Pseudocode
```cpp
struct Q6_K_Block {
    uint16_t d_fp16;      // 2 bytes
    uint8_t  ql[128];     // low 4 bits (2 elems/byte)
    uint8_t  qh[64];      // two bitplanes (2 bits/elem)
    uint8_t  scales[16];  // 16 sub-block scales (uint8)
};

inline float fp16_to_f32(uint16_t h);

inline int low4_of(const uint8_t *ql, int g, int t) {
    const int idx = g*8 + (t >> 1);
    return (t & 1) ? ((ql[idx] >> 4) & 0x0F) : (ql[idx] & 0x0F);
}

// Example bitplane mapping; ensure it matches your reference tests:
inline int hi2_of(const uint8_t *qh, int g, int t) {
    const int base = g*4 + (t & 7);
    const int s    = t >> 3;          // 0 or 1
    const int b0   = (qh[base] >> (s + 0)) & 0x1;
    const int b1   = (qh[base] >> (s + 2)) & 0x1;
    return b0 | (b1 << 1);
}

void dequantize_q6_k_block(const Q6_K_Block &b, float out[256]) {
    const float d = fp16_to_f32(b.d_fp16);
    for (int g = 0; g < 16; ++g) {
        const float scale_g = d * float(b.scales[g]);
        for (int t = 0; t < 16; ++t) {
            const int low4 = low4_of(b.ql, g, t);
            const int hi2  = hi2_of (b.qh, g, t);
            const int q6   = (low4 | (hi2 << 4)) - 32; // [-32..31]
            out[g*16 + t]  = scale_g * float(q6);
        }
    }
}
```

Notes
- Matches ggml’s dequantization flow for GGML_TYPE_Q6_K:
  - One fp16 “super-scale” d
  - 16 integer sub-scales (uint8) for 16×16 groups
  - 6-bit values via (low4 + two high bitplanes), centered by subtracting 32
- The exact bitplane addressing can be validated against gguf-py’s Q6_K dequant and ggml’s CPU/GPU kernels.
````

````markdown name=Q5_Q6_Q8_DEQUANT_IMPL_PLAN.md
# Q5_0, Q6_K, Q8_0 Dequantization: Implementation Plan

Order of implementation
1) Q8_0 (easiest)
2) Q5_0 (medium)
3) Q6_K (K-quant, most complex)

Rust code structure
- Types:
  - Define per-format block structs with #[repr(C)] matching the binary layouts above:
    - Q8_0Block { d: u16, qs: [i8; 32] }
    - Q5_0Block { d: u16, qh: [u8; 4], qs: [u8; 16] }
    - Q6_KBlock { d: u16, ql: [u8; 128], qh: [u8; 64], scales: [u8; 16] }
- Converters:
  - fp16_to_f32(u16) → f32
- Dequant APIs:
  - dequantize_q8_0_block(&Q8_0Block, &mut [f32; 32])
  - dequantize_q5_0_block(&Q5_0Block, &mut [f32; 32])
  - dequantize_q6_k_block(&Q6_KBlock, &mut [f32; 256])

Testing strategy
- Unit tests:
  - Block size assertions (std::mem::size_of::<T>())
  - Zero-block: all outputs should be zero
  - Known-value blocks: hand-crafted d/qs to validate bitpaths
- Cross-validation:
  - Use gguf-py’s Q8_0/Q5_0/Q6_K dequantizers on the same raw block bytes and compare results (atol ~ 1e-3)
- Integration:
  - Load the Qwen2.5-0.5B GGUF
  - Ensure weights with GGML_TYPE_Q5_0, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0 routes dequantize correctly
  - Confirm no “all zeros” tensors and no NaNs in logits

Performance considerations
- Start with CPU dequant-on-load to FP16/FP32 and upload to GPU (simplest, validates pipeline).
- If VRAM is tight, later add on-the-fly GPU dequant kernels or fused matmuls.

Estimated effort
- Q8_0: ~30 min (straight multiply)
- Q5_0: ~1 hour (5-bit unpack + signed shift)
- Q6_K: ~2 hours (K-quant subgroups, bitplanes, per-subgroup scales)
- Tests + integration: ~2 hours total
````

These specifications and plans are aligned with how ggml/llama.cpp implement the formats:
- Q8_0: 32 int8 values with one fp16 scale per block; y = d * q.
- Q5_0: 32 values with 4-bit lows (qs) + 1-bit highs (qh), signed by subtracting 16; y = d * q5.
- Q6_K: 256 values per block, 16×16 subgroups with integer scales, 6-bit composed from low nibbles and two high bitplanes; y = (d * sc[g]) * (q6 - 32).

You can validate your Rust implementation against gguf-py’s reference dequantizers for Q5_0, Q6_K, and Q8_0.