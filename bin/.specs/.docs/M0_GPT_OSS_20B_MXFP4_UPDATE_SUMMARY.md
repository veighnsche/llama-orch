# M0 GPT-OSS-20B (MXFP4) Implementation Update Summary

**Date**: 2025-10-03  
**Action**: Updated M0 worker spec with GPT-OSS-20B MXFP4 implementation details  
**Source**: Specs team GPT-OSS-20B implementation requirements

---

## Changes Made to `01_M0_worker_orcd.md`

### 1. Section 6.1: Model Format - Enhanced GGUF v3 Support

**Updated [M0-W-1200]**: GGUF Format Support

**Added**:
- **GGUF Version**: Version 3 (required for MXFP4 tensor support)
- **GGUF v3 Features**:
  - MXFP4 tensor blocks (GPT-OSS-20B)
  - Extended metadata for tokenizer configuration
  - Improved tensor alignment specifications

### 2. Section 6.1: Quantized-Only Execution - MXFP4 Compute Path

**Updated [M0-W-1201]**: Quantized-Only Execution

**Enhanced quantization support**:
- **MXFP4** (GPT-OSS-20B primary)
- **Q4_K_M** (Qwen2.5-0.5B, Phi-3-Mini; GPT-OSS-20B fallback)
- **Q4_0** (fallback compatibility)

**Added MXFP4 Compute Path**:
MXFP4 weights MUST be wired into all weight consumers:
1. **Embeddings**: MXFP4 embedding matrix lookup
2. **Attention**: Q/K/V projections, attention matmul, output projection (prefill + decode)
3. **FFN**: Up projection, activation, down projection
4. **LM Head**: Final logits projection

**Added execution details**:
- In-kernel dequantization: MXFP4 tiles/groups → registers/shared memory → FP16 accumulate
- FP16 accumulation for all matmul results
- **FP16 KV cache precision** (stated explicitly)

**Added fallback behavior**:
- If MXFP4 artifact unavailable for GPT-OSS-20B, accept Q4_K_M or Q4_0
- `/health.quant_kind` MUST reflect actually loaded quantization

### 3. Section 6.3: VRAM Allocation - Enhanced for GPT-OSS-20B

**Updated [M0-W-1220]**: Model Weights Allocation

**Added VRAM-Only Residency requirements**:
- Strict VRAM-only (no UMA/CPU spill)
- Fail fast on insufficient VRAM
- **Keep process alive and report structured error** (do NOT exit)
- Emit `VRAM_OOM` error via SSE if during inference

**Updated [M0-W-1221]**: Memory-Mapped I/O (REQUIRED for GPT-OSS-20B)

**Changed from SHOULD to MUST**:
- Worker MUST use `mmap()` for host I/O
- Memory-map GGUF file for reading
- Parse headers and metadata from mapped region
- Stream tensor data directly from mmap to VRAM

**Rationale**: Critical for 12GB+ models like GPT-OSS-20B

**Updated [M0-W-1222]**: Chunked H2D Transfer (REQUIRED for GPT-OSS-20B)

**Changed from SHOULD to MUST**:
- Worker MUST copy model to VRAM in chunks
- **Chunk Size**: 1MB (configurable, but 1MB default)
- Added implementation example with code

### 4. Section 6.4: Test Models - GPT-OSS-20B Detailed Spec

**Updated Model 3**: GPT-OSS-20B (Trend-Relevant) - MXFP4 Implementation

**Enhanced specifications**:
- **Format**: GGUF v3 (MXFP4 tensor support)
- **Quantization**: MXFP4 (primary); Q4_K_M/Q4_0 (fallback)
- **Tokenizer**: HF tokenizers (Rust) loading tokenizer.json

**Added Tokenizer Configuration**:
- **Backend**: `hf-json` (Hugging Face tokenizers crate)
- **Source**: `tokenizer.json` in model directory
- **Metadata Exposure**:
  - `eos_id`: End-of-sequence token ID
  - `bos_id`: Begin-of-sequence token ID
  - `vocab_size`: Vocabulary size (e.g., 50257)
  - `model_max_context`: Maximum context length (if available)

**Added MXFP4 Compute Requirements**:
- In-kernel dequant: MXFP4 tiles/groups → registers/shared memory → FP16 accumulate
- Weight consumers: Embeddings, Attention (Q/K/V, attn matmul, output proj), FFN (up/act/down), LM head
- KV cache: FP16 precision
- Streaming: UTF-8-safe SSE (buffer partial multibyte sequences)

**Added Loader Requirements**:
- GGUF v3 tensor support (MXFP4 blocks)
- Memory-mapped I/O (mmap for host I/O)
- Chunked H2D copies (1MB chunks)
- 256-byte alignment for device buffers
- Strict VRAM-only residency (no UMA/CPU spill)
- Fail fast on insufficient VRAM; keep process alive, report structured error

**Added Test Coverage**:
- UTF-8 streaming with multibyte characters
- OOM recovery (intentional KV/context overflow)

### 5. Section 7.3: Health Endpoint - Enhanced Fields

**Updated [M0-W-1320]**: GET /health

**Updated response example** (now shows GPT-OSS-20B):
```json
{
  "status": "healthy",
  "model": "gpt-oss-20b",
  "resident": true,
  "quant_kind": "MXFP4",
  "vram_bytes_used": 16106127360,
  "tokenizer_kind": "hf-json",
  "vocab_size": 50257,
  "context_length": 2048,
  "uptime_seconds": 3600,
  "sm": 86,
  "cuda_runtime_version": "12.1"
}
```

**Added required fields**:
- `resident` (bool) — VRAM residency status (true = all weights in VRAM)
- `quant_kind` (string) — Quantization format: `"MXFP4"` | `"Q4_K_M"` | `"Q4_0"`
- `vram_bytes_used` (int) — Current VRAM usage in bytes
- `tokenizer_kind` (string) — Backend type: `"gguf-bpe"` or `"hf-json"`
- `vocab_size` (int) — Vocabulary size
- `context_length` (int) — Model's maximum context length

**Added optional fields** (nice-to-have):
- `sm` (int) — Compute capability (e.g., 86 for SM_86)
- `cuda_runtime_version` (string) — CUDA runtime version

### 6. Section 8.2: HF-JSON Backend - Metadata Exposure

**Updated [M0-W-1361]**: Hugging Face Tokenizers Crate

**Added Metadata Exposure** (2025-10-03):
Worker MUST expose tokenizer metadata from tokenizer.json:
- `eos_id`: End-of-sequence token ID
- `bos_id`: Begin-of-sequence token ID
- `vocab_size`: Vocabulary size
- `model_max_context`: Maximum context length (if available in tokenizer.json)

**Added UTF-8 Streaming requirements**:
- Enforce UTF-8-safe SSE streaming
- Buffer partial multibyte sequences
- Never emit invalid UTF-8
- Handle token boundaries that split UTF-8 codepoints

**Enhanced Conformance Testing**:
- Golden encode/decode test vectors MUST be included (20-30 pairs)
- Vectors catch schema drift and ensure parity with upstream tokenizer
- Test vectors cover: BOS/EOS handling, special tokens, multibyte UTF-8, edge cases

### 7. Section 12.3.1: NEW - GPT-OSS-20B Acceptance Tests

**Added 5 new acceptance test requirements**:

#### [M0-W-1821] Tokenizer Conformance Test
- **Test**: 20-30 text↔ids pairs from upstream tokenizer.json artifacts
- **Coverage**: Basic encode/decode, BOS/EOS, special tokens, multibyte UTF-8, edge cases
- **Location**: `tests/tokenizer_conformance_gpt_oss_20b.rs`

#### [M0-W-1822] MXFP4 Micro-Goldens Test
- **Test**: Dequant→GEMM and small attention shape vs float reference
- **Coverage**: MXFP4 dequantization, GEMM, attention, FP16 accumulation
- **Tolerance**: ±0.01 (1%) relative error for FP16 accumulation
- **Location**: `tests/mxfp4_micro_goldens.rs`

#### [M0-W-1823] Large Model Bring-Up Test
- **Test**: Load GPT-OSS-20B (MXFP4), verify health endpoint, confirm VRAM envelope
- **Validation**: 
  - `health.quant_kind == "MXFP4"`
  - `health.tokenizer_kind == "hf-json"`
  - `health.resident == true`
  - VRAM usage 15-17 GB
- **Location**: `tests/gpt_oss_20b_bring_up.rs`

#### [M0-W-1824] UTF-8 Streaming Test
- **Test**: Prompts with multibyte characters; assert no mojibake
- **Coverage**: Chinese (3-byte), Emoji (4-byte), mixed scripts, token boundaries
- **Location**: `tests/utf8_streaming.rs`

#### [M0-W-1825] OOM Recovery Test
- **Test**: Intentionally exceed KV/context → expect structured VRAM_OOM error
- **Validation**: 
  - Expect `VRAM_OOM` error event
  - Process stays alive
  - Clean recovery (next request works)
- **Location**: `tests/oom_recovery.rs`

---

## New Requirements Summary

### Spec IDs Added
1. **M0-W-1821**: Tokenizer Conformance Test (GPT-OSS-20B)
2. **M0-W-1822**: MXFP4 Micro-Goldens Test
3. **M0-W-1823**: Large Model Bring-Up Test
4. **M0-W-1824**: UTF-8 Streaming Test
5. **M0-W-1825**: OOM Recovery Test

### Implementation Requirements

#### Loader & Memory
- ✅ GGUF v3 tensor support (MXFP4 blocks)
- ✅ Memory-mapped I/O (mmap) - REQUIRED
- ✅ Chunked H2D copies (1MB chunks) - REQUIRED
- ✅ 256-byte alignment for device buffers
- ✅ Strict VRAM-only residency (no UMA/CPU spill)
- ✅ Fail fast on insufficient VRAM; keep process alive, report error

#### MXFP4 Compute Path
- ✅ In-kernel dequant for MXFP4 tiles/groups
- ✅ Wire MXFP4 into: Embeddings, Attention (Q/K/V, attn, output), FFN (up/act/down), LM head
- ✅ FP16 accumulation for all matmuls
- ✅ FP16 KV cache precision

#### Tokenization (HF-JSON Backend)
- ✅ Load tokenizer.json via HF tokenizers crate
- ✅ Expose: eos_id, bos_id, vocab_size, model_max_context
- ✅ UTF-8-safe SSE streaming (buffer partial multibyte sequences)
- ✅ 20-30 golden test vectors for conformance

#### API & Observability
- ✅ `/health` includes: resident, quant_kind, vram_bytes_used, tokenizer_kind, vocab_size, context_length
- ✅ Optional: sm (compute capability), cuda_runtime_version

#### Acceptance Tests
- ✅ Tokenizer conformance (20-30 text↔ids pairs)
- ✅ MXFP4 micro-goldens (dequant→GEMM vs float reference, ±1% tolerance)
- ✅ Large model bring-up (load, verify health, VRAM envelope 15-17 GB)
- ✅ UTF-8 streaming (multibyte characters, no mojibake)
- ✅ OOM recovery (exceed context, expect VRAM_OOM, clean recovery)

#### Fallback Behavior
- ✅ If MXFP4 unavailable, accept Q4_K_M or Q4_0
- ✅ `/health.quant_kind` reflects actually loaded quantization

---

## Implementation Checklist

### Code Changes Required
- [ ] Implement GGUF v3 parser with MXFP4 tensor support
- [ ] Implement mmap-based model loading
- [ ] Implement chunked H2D transfer (1MB chunks)
- [ ] Implement MXFP4 in-kernel dequantization
- [ ] Wire MXFP4 into all weight consumers (embeddings, attention, FFN, LM head)
- [ ] Implement FP16 KV cache
- [ ] Implement HF tokenizers integration (tokenizer.json)
- [ ] Expose tokenizer metadata (eos_id, bos_id, vocab_size, model_max_context)
- [ ] Implement UTF-8-safe SSE streaming with multibyte buffering
- [ ] Update `/health` endpoint with new fields
- [ ] Implement VRAM OOM error handling (keep process alive)

### Test Implementation Required
- [ ] Tokenizer conformance test (20-30 golden vectors)
- [ ] MXFP4 micro-goldens test (dequant→GEMM, attention)
- [ ] Large model bring-up test (load GPT-OSS-20B, verify health)
- [ ] UTF-8 streaming test (multibyte characters)
- [ ] OOM recovery test (exceed context, verify recovery)

### Documentation Updates
- [x] Updated M0 worker spec with MXFP4 details
- [x] Updated model specifications (GPT-OSS-20B)
- [x] Updated health endpoint schema
- [x] Updated tokenization backend details
- [x] Added acceptance test specifications

---

## Summary

The M0 worker spec has been comprehensively updated with GPT-OSS-20B (MXFP4) implementation details:

1. **GGUF v3 Support**: MXFP4 tensor blocks, extended metadata
2. **MXFP4 Compute Path**: In-kernel dequant, FP16 accumulate, wired into all weight consumers
3. **Loader Enhancements**: mmap (REQUIRED), chunked H2D (REQUIRED), VRAM-only with OOM handling
4. **Tokenization**: HF tokenizers with metadata exposure, UTF-8-safe streaming
5. **Health Endpoint**: Enhanced with resident, quant_kind, vram_bytes_used, tokenizer_kind, vocab_size, context_length
6. **Acceptance Tests**: 5 new tests (tokenizer conformance, MXFP4 micro-goldens, bring-up, UTF-8 streaming, OOM recovery)

**Result**: Complete specification for GPT-OSS-20B (MXFP4) support in M0 worker-orcd.

---

**Status**: M0 spec updated with GPT-OSS-20B MXFP4 implementation requirements  
**Next Steps**: Implement MXFP4 compute path, HF tokenizers integration, and acceptance tests  
**Reference**: See updated sections in `01_M0_worker_orcd.md`
