# M0 Tokenization Strategy Update Summary

**Date**: 2025-10-03  
**Action**: Finalized tokenization strategy for M0 worker spec  
**Source**: Specs team tokenization decision

---

## Changes Made to `01_M0_worker_orcd.md`

### 1. New Section 8: Tokenization Strategy

Added comprehensive tokenization section with 6 subsections:

#### 8.1 Backend Architecture
- **[M0-W-1360]**: Tokenizer backend selection at model load time
- Two backend types: `hf-json` and `gguf-bpe`
- Runtime selection based on model metadata

#### 8.2 HF-JSON Backend (GPT-OSS-20B)
- **[M0-W-1361]**: Hugging Face tokenizers crate (Rust)
- Load tokenizer.json directly
- Golden encode/decode test vectors for schema drift detection
- Ensures parity with upstream tokenizer

#### 8.3 GGUF-BPE Backend (Qwen2.5-0.5B, Phi-3-Mini)
- **[M0-W-1362]**: Pure-Rust GGUF tokenizer implementation
- Parse GGUF metadata (vocab + merges)
- Byte-level BPE entirely in Rust
- UTF-8 safe streaming decode with BOS/EOS handling
- No llama.cpp dependency

#### 8.4 Conformance Testing
- **[M0-W-1363]**: Test vectors required for all three models
- Coverage: basic encode/decode, BOS/EOS, special tokens, UTF-8 edge cases
- Schema drift detection for GPT-OSS-20B

#### 8.5 Health Endpoint Integration
- **[M0-W-1364]**: Tokenizer observability in /health
- Added fields: `tokenizer_kind` and `vocab_size`
- Purpose: debugging, validation, monitoring

#### 8.6 Implementation Requirements
- **[M0-W-1365]**: No external dependencies
- No Python runtime
- No external binaries (no llama.cpp)
- Pure Rust for both backends
- Deterministic encode/decode

### 2. Updated Section 0.2: Scope

**Added to "In Scope for M0"**:
```markdown
- ✅ Tokenization: Two backends (`hf-json` for GPT-OSS-20B, `gguf-bpe` for Qwen/Phi-3)
```

### 3. Updated Section 7.3: Health Endpoint

**Updated [M0-W-1320] GET /health response**:

**Added fields**:
```json
{
  "tokenizer_kind": "gguf-bpe",
  "vocab_size": 151936
}
```

**Added documentation**:
- `tokenizer_kind` — Backend type: `"gguf-bpe"` or `"hf-json"`
- `vocab_size` — Vocabulary size for debugging and validation

### 4. Updated Section 14.1: Gaps & Clarifications

**Enhanced "Tokenization Library" resolution**:

**From** (brief):
- Two backends: GGUF byte-BPE and tokenizer.json
- No llama.cpp dependency

**To** (detailed):
- **`hf-json` backend** for GPT-OSS-20B:
  - Uses Hugging Face `tokenizers` crate (Rust)
  - Loads tokenizer.json directly
  - Golden test vectors for schema drift
  - Ensures parity with upstream
- **`gguf-bpe` backend** for Qwen/Phi-3:
  - Pure-Rust GGUF implementation
  - Parses GGUF metadata (vocab + merges)
  - Byte-level BPE in Rust
  - UTF-8 safe streaming decode
- Runtime selection based on model metadata
- No Python or external binaries
- Conformance test vectors for all models

### 5. Updated Section 15.1: Exit Criteria

**Enhanced tokenization exit criteria**:

**From**:
```markdown
15. ✅ Tokenization works for both GGUF byte-BPE and tokenizer.json backends
```

**To**:
```markdown
15. ✅ Tokenization works for both backends:
    - `gguf-bpe` backend (Qwen2.5-0.5B, Phi-3-Mini)
    - `hf-json` backend (GPT-OSS-20B)
    - Conformance test vectors pass for all three models
```

### 6. Section Renumbering

**Fixed**: Section 9 (CUDA Implementation) subsection numbering
- Changed from "### 8.1 Context Management" to "### 9.1 Context Management"

---

## New Requirements Added

### Spec IDs
1. **M0-W-1360**: Tokenizer Backend Selection
2. **M0-W-1361**: Hugging Face Tokenizers Crate (hf-json)
3. **M0-W-1362**: Pure-Rust GGUF Tokenizer (gguf-bpe)
4. **M0-W-1363**: Test Vectors Required
5. **M0-W-1364**: Tokenizer Observability (health endpoint)
6. **M0-W-1365**: No External Dependencies

### Backend Mapping
- **GPT-OSS-20B** → `hf-json` backend (Hugging Face tokenizers crate)
- **Qwen2.5-0.5B** → `gguf-bpe` backend (pure Rust)
- **Phi-3-Mini** → `gguf-bpe` backend (pure Rust)

### Test Requirements
1. **Conformance test vectors** for all three models
2. **Golden vectors** for GPT-OSS-20B (schema drift detection)
3. **Coverage**: encode/decode, BOS/EOS, special tokens, UTF-8 edge cases

### Health Endpoint
- **New fields**: `tokenizer_kind`, `vocab_size`
- **Purpose**: Debugging, validation, monitoring

---

## Implementation Implications

### Dependencies
- **Add**: `tokenizers` crate (Hugging Face) for hf-json backend
- **Implement**: Pure-Rust GGUF tokenizer for gguf-bpe backend
- **No**: Python, llama.cpp, or external binaries

### Code Structure
```rust
// Backend trait
trait TokenizerBackend {
    fn encode(&self, text: &str) -> Result<Vec<u32>>;
    fn decode(&self, token_ids: &[u32]) -> Result<String>;
}

// HF-JSON implementation
struct HfJsonBackend {
    tokenizer: tokenizers::Tokenizer,
}

// GGUF-BPE implementation
struct GgufBpeBackend {
    vocab: Vec<String>,
    merges: Vec<(u32, u32)>,
    bos_token_id: Option<u32>,
    eos_token_id: Option<u32>,
    special_tokens: HashMap<String, u32>,
}
```

### Testing
- Unit tests for each backend
- Conformance test vectors (golden tests)
- UTF-8 safety tests
- BOS/EOS handling tests
- Special token tests

---

## Alignment with M0 Goals

### ✅ Self-Contained
- No Python runtime
- No external binaries
- Single worker binary

### ✅ Deterministic
- Pure Rust implementation
- Deterministic encode/decode
- Reproducible across platforms

### ✅ Clean Separation
- Two distinct backends
- Runtime selection based on model metadata
- Clear interface (TokenizerBackend trait)

### ✅ Observability
- Health endpoint reports tokenizer_kind
- Vocab size for validation
- Debugging support

---

## Summary

The M0 worker spec has been updated with a comprehensive tokenization strategy:

1. **Two backends**: `hf-json` (GPT-OSS-20B) and `gguf-bpe` (Qwen/Phi-3)
2. **Pure Rust**: No Python, no llama.cpp, no external binaries
3. **Conformance testing**: Test vectors for all three models
4. **Observability**: Health endpoint includes tokenizer info
5. **Clean architecture**: Pluggable backend trait with runtime selection

**Result**: Self-contained, deterministic tokenization for all M0 reference models.

---

**Status**: Tokenization strategy finalized and documented  
**Next Steps**: Implement TokenizerBackend trait and both backends  
**Reference**: See Section 8 of `01_M0_worker_orcd.md` for full specification
