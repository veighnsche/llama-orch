# LT-000: GGUF & BPE Spec Study

**Team**: Llama-Beta  
**Sprint**: Sprint 0 - Prep Work  
**Size**: M (2-3 days)  
**Days**: 1 - 3  
**Spec Ref**: M0-W-1211, M0-W-1212, M0-W-1213, M0-W-1214  
**Research Scope**: Comprehensive study with internet access

---

## Story Description

Conduct comprehensive research on the GGUF (GPT-Generated Unified Format) file format and GGUF-embedded Byte-Level BPE tokenization to establish deep understanding of the format specification, memory-mapped I/O patterns, tensor layout strategies, metadata extraction, tokenizer implementation, UTF-8 streaming safety, and validation frameworks. This extensive study will leverage internet access to survey llama.cpp documentation, GGUF specifications, BPE academic literature, existing implementations, and community best practices.

The research prepares Llama-Beta for implementing GGUF parsing, memory-mapped weight loading, and byte-level BPE tokenization with UTF-8 streaming safety. Multiple documentation artifacts will be produced covering format specification, tokenizer design, memory management, validation framework design, and implementation recommendations.

**Research Scope**: 12 major topic areas with 100+ online sources to investigate, producing 7 documentation deliverables.

---

## Acceptance Criteria

### Core GGUF Understanding
- [ ] GGUF file structure documented (magic bytes, version, metadata, tensor info, tensor data)
- [ ] GGUF v3 format differences from v2 understood and documented
- [ ] Metadata key-value system documented (types, encoding, extraction)
- [ ] Tensor info section layout understood (name, dimensions, type, offset)
- [ ] Alignment and padding requirements clarified (32-byte alignment)
- [ ] Quantization format encoding documented (Q4_K_M, Q5_K_M, Q6_K, etc.)

### Memory-Mapped I/O
- [ ] Memory mapping strategies documented (mmap vs read, platform differences)
- [ ] Page alignment requirements clarified (4KB pages on Linux)
- [ ] Chunked H2D transfer patterns analyzed (optimal chunk sizes)
- [ ] VRAM residency verification strategies documented
- [ ] Zero-copy optimization opportunities identified
- [ ] Error handling for mmap failures documented

### GGUF-BPE Tokenizer
- [ ] Byte-level BPE algorithm documented (UTF-8 byte sequences)
- [ ] GGUF vocab section parsing strategy defined
- [ ] GGUF merges section parsing strategy defined
- [ ] Encoder implementation approach documented (trie vs hashmap)
- [ ] Decoder implementation approach documented (direct lookup)
- [ ] UTF-8 streaming safety requirements clarified (boundary detection)

### Llama Architecture Specifics
- [ ] Llama model architecture variants documented (Llama 2, Llama 3, Qwen, Phi-3)
- [ ] RoPE (Rotary Position Embedding) implementation requirements documented
- [ ] RMSNorm (Root Mean Square Normalization) requirements documented
- [ ] GQA (Grouped Query Attention) vs MHA differences analyzed
- [ ] SwiGLU activation function requirements documented
- [ ] KV-cache layout for GQA documented

### Numerical Precision & Validation
- [ ] Quantization format precision characteristics documented (Q4_K_M tolerance)
- [ ] Dequantization algorithm requirements specified
- [ ] Numerical validation strategy defined (comparison with reference)
- [ ] Perplexity validation approach defined (WikiText-2 baseline)
- [ ] Reproducibility requirements documented (seeded sampling)
- [ ] Edge case testing plan created (empty prompts, long contexts, special tokens)

### Integration & Implementation
- [ ] GGUF parser integration points identified (header, metadata, tensors)
- [ ] Weight loading pipeline design documented (mmap → H2D → VRAM)
- [ ] Tokenizer integration points identified (encode, decode, streaming)
- [ ] Architecture detection strategy documented (metadata inspection)
- [ ] Model-specific weight mapping strategies documented (Qwen vs Phi-3)
- [ ] Error propagation and recovery patterns defined

### Ecosystem & Tooling
- [ ] llama.cpp GGUF implementation surveyed and documented
- [ ] GGUF conversion tools landscape analyzed (convert.py, quantize)
- [ ] Model zoo availability assessed (Hugging Face GGUF models)
- [ ] Tokenizer conformance testing tools identified
- [ ] Validation datasets documented (WikiText-2, C4, The Pile)
- [ ] Community best practices catalogued

### Research Deliverables
- [ ] Research notes compiled in `docs/gguf-research.md`
- [ ] BPE tokenizer specification in `docs/gguf-bpe-tokenizer.md`
- [ ] Memory-mapped I/O guide in `docs/gguf-mmap-io.md`
- [ ] Llama architecture analysis in `docs/llama-architecture-variants.md`
- [ ] Validation framework specification in `docs/gguf-validation-framework.md`
- [ ] UTF-8 streaming safety guide in `docs/utf8-streaming-safety.md`
- [ ] Online source bibliography with key findings from each source

---

## Dependencies

### Upstream (Blocks This Story)
- None (prep work can start immediately)

### Downstream (This Story Blocks)
- LT-001: GGUF Header Parser (needs format understanding)
- LT-007: GGUF Vocab Parsing (needs tokenizer design)
- LT-009: Byte-Level BPE Encoder (needs algorithm understanding)
- LT-011: UTF-8 Safe Streaming Decode (needs safety requirements)

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/.plan/llama-team/docs/gguf-research.md` - Comprehensive research notes and findings
- `bin/worker-orcd/.plan/llama-team/docs/gguf-bpe-tokenizer.md` - BPE tokenizer design and implementation guide
- `bin/worker-orcd/.plan/llama-team/docs/gguf-mmap-io.md` - Memory-mapped I/O patterns and best practices
- `bin/worker-orcd/.plan/llama-team/docs/llama-architecture-variants.md` - Llama model architecture analysis
- `bin/worker-orcd/.plan/llama-team/docs/gguf-validation-framework.md` - Validation strategy and test design
- `bin/worker-orcd/.plan/llama-team/docs/utf8-streaming-safety.md` - UTF-8 streaming safety requirements
- `bin/worker-orcd/.plan/llama-team/docs/gguf-sources-bibliography.md` - Annotated bibliography of online sources

### Research Topics

**1. GGUF Format Specification & Structure**:
- GGUF magic bytes and version detection (0x46554747 = "GGUF")
- GGUF v2 vs v3 differences (tensor type encoding, metadata extensions)
- Metadata section structure (key-value pairs, type system)
- Tensor info section layout (name, n_dims, dims[], type, offset)
- Tensor data section (aligned, contiguous, quantized)
- Alignment requirements (32-byte alignment for tensors)
- Endianness handling (little-endian standard)
- File size calculation and validation
- Extension mechanisms for custom metadata
- Backward compatibility considerations

**2. GGUF Metadata System**:
- Metadata key naming conventions (general., tokenizer., etc.)
- Metadata value types (uint8, uint16, uint32, uint64, int8, int16, int32, int64, float32, float64, bool, string, array)
- Architecture detection metadata (general.architecture)
- Model hyperparameters (n_layers, n_heads, n_embd, etc.)
- Tokenizer configuration (tokenizer.ggml.model, tokenizer.ggml.tokens, tokenizer.ggml.merges)
- Quantization metadata (general.quantization_version)
- Custom metadata extensions
- Metadata extraction strategies (sequential parse vs indexed lookup)
- Validation of required vs optional metadata keys

**3. GGUF Quantization Formats**:
- Q4_K_M format structure (4-bit + scale factors)
- Q5_K_M format structure (5-bit + scale factors)
- Q6_K format structure (6-bit + scale factors)
- Q8_0 format structure (8-bit symmetric)
- F16 and F32 unquantized formats
- Block-wise quantization patterns (32, 64, 128 element blocks)
- Scale factor representations (per-block vs per-tensor)
- Dequantization algorithms for each format
- Precision characteristics and error bounds
- Performance trade-offs (memory vs accuracy)

**4. Memory-Mapped I/O Patterns**:
```cpp
// Conceptual memory mapping
int fd = open("model.gguf", O_RDONLY);
size_t file_size = lseek(fd, 0, SEEK_END);
void* mapped = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);

// Access tensor data directly
const void* tensor_data = (char*)mapped + tensor_offset;
```
- mmap() vs read() trade-offs (virtual memory vs physical reads)
- Page alignment requirements (4KB on Linux, 16KB on macOS)
- MAP_PRIVATE vs MAP_SHARED semantics
- madvise() optimization strategies (MADV_SEQUENTIAL, MADV_WILLNEED)
- Chunked H2D transfer patterns (optimal chunk sizes for PCIe)
- Zero-copy opportunities (direct GPU mapping on unified memory)
- Error handling (ENOMEM, EACCES, EINVAL)
- Platform differences (Linux, macOS, Windows)
- Cleanup and unmapping strategies

**5. GGUF-BPE Tokenizer Design**:
- Byte-level BPE algorithm (UTF-8 byte sequences as tokens)
- Vocab section parsing (token strings, token IDs)
- Merges section parsing (merge rules, priority order)
- Encoder implementation (greedy BPE with merge priority)
- Decoder implementation (token ID → byte sequence → UTF-8)
- Special token handling (BOS, EOS, PAD, UNK)
- Byte fallback mechanism (unknown bytes → byte tokens)
- Trie-based encoder optimization
- Hashmap-based decoder optimization
- Conformance testing strategy (compare with reference tokenizer)

**6. UTF-8 Streaming Safety**:
```rust
// Conceptual UTF-8 boundary detection
fn is_utf8_boundary(bytes: &[u8], pos: usize) -> bool {
    if pos == 0 || pos >= bytes.len() {
        return true;
    }
    // Check if byte at pos is NOT a continuation byte (0b10xxxxxx)
    (bytes[pos] & 0b1100_0000) != 0b1000_0000
}
```
- UTF-8 encoding structure (1-4 byte sequences)
- Continuation byte detection (0b10xxxxxx pattern)
- Boundary detection algorithm (safe split points)
- Partial token buffering strategy
- Streaming decode with boundary safety
- Invalid UTF-8 handling (replacement character U+FFFD)
- Multibyte character edge cases (emoji, CJK, combining marks)
- Performance optimization (SIMD UTF-8 validation)

**7. Llama Architecture Variants**:
- **Llama 2**: GQA, RoPE, RMSNorm, SwiGLU
- **Llama 3**: Enhanced GQA, extended context (8K → 128K)
- **Qwen 2.5**: Modified RoPE (base 1M), unique tokenizer
- **Phi-3**: Small model optimizations, modified attention
- Architecture detection from metadata (general.architecture)
- Weight tensor naming conventions (per architecture)
- Hyperparameter extraction (n_layers, n_heads, n_kv_heads, etc.)
- Model-specific quirks and workarounds
- Forward pass differences (attention, FFN, normalization)

**8. RoPE (Rotary Position Embedding)**:
```cuda
// Conceptual RoPE kernel
__global__ void rope_kernel(
    half* qk,              // Query or Key tensor
    const int* positions,  // Position indices
    float theta_base,      // RoPE base (10000.0 or 1000000.0)
    int head_dim
) {
    // Compute rotation angles
    // Apply rotation to (q[i], q[i+1]) pairs
}
```
- RoPE algorithm (rotation-based position encoding)
- Theta base parameter (10000.0 for Llama 2, 1000000.0 for Qwen)
- Head dimension requirements (must be even)
- Rotation pair application (q[i], q[i+1])
- Frequency scaling for long context
- Integration with attention (prefill vs decode)
- Kernel optimization strategies
- Numerical stability considerations

**9. GQA (Grouped Query Attention)**:
- GQA vs MHA differences (fewer KV heads)
- KV head grouping (n_heads / n_kv_heads)
- KV-cache layout for GQA (grouped storage)
- Attention computation with grouped KV
- Memory savings analysis (KV-cache size reduction)
- Performance implications (compute vs memory)
- Prefill vs decode attention patterns
- Flash Attention integration

**10. Model Loading Pipeline**:
```rust
// Conceptual loading pipeline
fn load_gguf_model(path: &Path) -> Result<Model> {
    // 1. Parse GGUF header (magic, version, metadata count)
    // 2. Extract metadata (architecture, hyperparams, tokenizer)
    // 3. Parse tensor info (names, shapes, types, offsets)
    // 4. Memory-map tensor data
    // 5. Chunked H2D transfer to VRAM
    // 6. Verify VRAM residency
    // 7. Initialize tokenizer (vocab, merges)
    // 8. Return initialized model
}
```
- Header parsing (magic bytes, version, tensor count)
- Metadata extraction (sequential parse)
- Tensor info parsing (name, dims, type, offset)
- Memory mapping (mmap entire file)
- Chunked H2D transfer (optimal chunk sizes)
- VRAM residency verification (cudaMemGetInfo)
- Tokenizer initialization (vocab + merges)
- Error handling and cleanup

**11. Validation Strategy & Testing**:
- GGUF header validation (magic bytes, version)
- Metadata validation (required keys, type correctness)
- Tensor info validation (alignment, offset bounds)
- Tokenizer conformance testing (compare with reference)
- Numerical validation (perplexity on WikiText-2)
- Reproducibility testing (seeded sampling, 10 runs)
- Edge case testing (empty prompts, long contexts, special tokens)
- Cross-model validation (Qwen, Phi-3)
- Performance benchmarking (tokens/sec)
- Memory leak detection (valgrind, CUDA leak checker)

**12. Research Questions for Online Investigation**:
- What are the latest GGUF v3 format changes?
- Which Llama model variants are most popular in production?
- What are the best practices for memory-mapped I/O on different platforms?
- How do different BPE implementations handle edge cases?
- What are the performance characteristics of different quantization formats?
- How does RoPE theta base affect long context performance?
- What are the common pitfalls in GGUF parsing?
- How do different tokenizers handle UTF-8 edge cases?
- What validation datasets are most representative?
- What are the licensing implications of llama.cpp code?

### Implementation Notes
- GGUF format is well-documented but has subtle version differences
- Memory-mapped I/O requires careful platform-specific handling
- BPE tokenizer must handle UTF-8 boundaries correctly for streaming
- Llama architecture variants have different weight naming conventions
- Validation framework must be built before implementation
- Focus on understanding format before parser implementation
- Document all assumptions and design decisions

---

## Testing Strategy

### Research Validation
- Document GGUF format structure with diagrams and visual aids
- Create example file layout with sample values (multiple scenarios)
- Verify understanding against GGUF spec (llama.cpp docs)
- Cross-reference llama.cpp implementation for edge cases
- Design test vectors for tokenizer validation
- Validate numerical precision claims with hand calculations

### Documentation Review
- Research notes reviewed for completeness and accuracy
- Tokenizer design reviewed for conformance requirements
- Memory-mapped I/O patterns reviewed for platform compatibility
- Validation framework design reviewed for feasibility
- All cited sources validated for correctness and relevance
- Cross-team review of implementation recommendations

### Manual Verification
1. Read GGUF specification and llama.cpp implementation
2. Document format structure in markdown with diagrams
3. Create example GGUF file layout by hand (multiple test cases)
4. Analyze BPE tokenizer algorithm and edge cases
5. Survey existing implementations in llama.cpp ecosystem
6. Design validation framework with clear acceptance criteria
7. Create platform compatibility matrix for mmap
8. Review all findings with spec requirements and team

### Online Research Validation
- All online sources accessed and key information extracted
- Bibliography created with findings summary per source
- Claims cross-referenced across multiple authoritative sources
- llama.cpp documentation verified (latest version)
- Academic papers reviewed and compared (BPE, RoPE, GQA)
- Community discussions analyzed for practical insights
- Model zoo surveyed for GGUF availability

---

## Definition of Done

### Completeness
- [ ] All acceptance criteria met across all categories
- [ ] All 12 research topic areas thoroughly investigated
- [ ] All research questions answered with evidence
- [ ] All online sources reviewed and annotated

### Documentation Deliverables
- [ ] Research notes documented in `docs/gguf-research.md`
- [ ] BPE tokenizer specification complete in `docs/gguf-bpe-tokenizer.md`
- [ ] Memory-mapped I/O guide complete in `docs/gguf-mmap-io.md`
- [ ] Llama architecture analysis documented in `docs/llama-architecture-variants.md`
- [ ] Validation framework specification complete in `docs/gguf-validation-framework.md`
- [ ] UTF-8 streaming safety guide complete in `docs/utf8-streaming-safety.md`
- [ ] Annotated bibliography created in `docs/gguf-sources-bibliography.md`

### Quality Gates
- [ ] Format specification verified against llama.cpp implementation
- [ ] All claims supported by cited sources
- [ ] Diagrams and examples included for complex concepts
- [ ] Platform compatibility verified for mmap strategies
- [ ] Tokenizer design validated with conformance requirements
- [ ] Validation framework reviewed for feasibility
- [ ] Cross-references between documents verified

### Team Handoff
- [ ] Research findings presented to Llama-Beta team
- [ ] Implementation recommendations reviewed and approved
- [ ] Validation framework design reviewed by testing lead
- [ ] Story marked complete in day-tracker.md
- [ ] Downstream stories (LT-001, LT-007, LT-009, LT-011) unblocked with necessary context

---

## Online Research Sources

### Primary Specifications & Standards
- **GGUF Specification**: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- **llama.cpp Documentation**: https://github.com/ggerganov/llama.cpp/tree/master/docs
- **GGML Format Docs**: https://github.com/ggerganov/ggml/blob/master/docs/ggml.md
- **GGUF v3 Changelog**: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md#version-3
- **Llama Model Card**: https://github.com/meta-llama/llama-models/blob/main/models/llama2/MODEL_CARD.md
- **Llama 3 Model Card**: https://github.com/meta-llama/llama-models/blob/main/models/llama3/MODEL_CARD.md

### llama.cpp Implementation
- **GGUF Loader**: https://github.com/ggerganov/llama.cpp/blob/master/llama.cpp (gguf_init_from_file)
- **Tokenizer Implementation**: https://github.com/ggerganov/llama.cpp/blob/master/llama.cpp (llama_tokenize)
- **Memory Mapping**: https://github.com/ggerganov/llama.cpp/blob/master/llama.cpp (llama_mmap)
- **Quantization Kernels**: https://github.com/ggerganov/llama.cpp/blob/master/ggml-quants.c
- **RoPE Implementation**: https://github.com/ggerganov/llama.cpp/blob/master/ggml-cuda/rope.cu
- **GQA Attention**: https://github.com/ggerganov/llama.cpp/blob/master/ggml-cuda/flash-attn.cu

### BPE & Tokenization
- **BPE Paper (Sennrich et al.)**: https://arxiv.org/abs/1508.07909
- **Byte-Level BPE (GPT-2)**: https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
- **SentencePiece**: https://github.com/google/sentencepiece
- **Hugging Face Tokenizers**: https://github.com/huggingface/tokenizers
- **tiktoken (OpenAI)**: https://github.com/openai/tiktoken
- **BPE Dropout**: https://arxiv.org/abs/1910.13267
- **Tokenization Best Practices**: https://huggingface.co/docs/transformers/tokenizer_summary

### UTF-8 & Character Encoding
- **UTF-8 Specification (RFC 3629)**: https://www.rfc-editor.org/rfc/rfc3629
- **Unicode Standard**: https://www.unicode.org/versions/Unicode15.0.0/
- **UTF-8 Validation (SIMD)**: https://github.com/simdjson/simdjson/blob/master/doc/basics.md#utf-8-validation
- **Rust UTF-8 Handling**: https://doc.rust-lang.org/std/str/index.html
- **UTF-8 Everywhere Manifesto**: https://utf8everywhere.org/

### Memory-Mapped I/O
- **mmap() Man Page**: https://man7.org/linux/man-pages/man2/mmap.2.html
- **madvise() Optimization**: https://man7.org/linux/man-pages/man2/madvise.2.html
- **Memory Mapping Best Practices**: https://www.kernel.org/doc/html/latest/admin-guide/mm/index.html
- **macOS mmap Differences**: https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man2/mmap.2.html
- **Windows Memory Mapping**: https://learn.microsoft.com/en-us/windows/win32/memory/creating-a-file-mapping-object
- **Zero-Copy I/O**: https://www.kernel.org/doc/html/latest/filesystems/splice.html

### Llama Architecture
- **Llama 2 Paper**: https://arxiv.org/abs/2307.09288
- **Llama 3 Technical Report**: https://ai.meta.com/blog/meta-llama-3/
- **RoPE Paper**: https://arxiv.org/abs/2104.09864
- **GQA Paper**: https://arxiv.org/abs/2305.13245
- **RMSNorm Paper**: https://arxiv.org/abs/1910.07467
- **SwiGLU Paper**: https://arxiv.org/abs/2002.05202
- **Flash Attention**: https://arxiv.org/abs/2205.14135
- **Flash Attention 2**: https://arxiv.org/abs/2307.08691

### Qwen & Phi-3 Specifics
- **Qwen 2.5 Model Card**: https://huggingface.co/Qwen/Qwen2.5-7B
- **Qwen Tokenizer**: https://huggingface.co/Qwen/Qwen2.5-7B/blob/main/tokenizer_config.json
- **Phi-3 Technical Report**: https://arxiv.org/abs/2404.14219
- **Phi-3 Model Card**: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct
- **Phi-3 Architecture**: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/blob/main/config.json

### Quantization & Precision
- **GGML Quantization**: https://github.com/ggerganov/ggml/blob/master/docs/quantization.md
- **Q4_K_M Format**: https://github.com/ggerganov/llama.cpp/pull/1684
- **Q5_K_M Format**: https://github.com/ggerganov/llama.cpp/pull/1751
- **Quantization Error Analysis**: https://arxiv.org/abs/2106.08295
- **Post-Training Quantization**: https://arxiv.org/abs/2109.01652
- **GPTQ Comparison**: https://arxiv.org/abs/2210.17323
- **AWQ Comparison**: https://arxiv.org/abs/2306.00978

### CUDA & GPU Programming
- **CUDA C++ Programming Guide**: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- **CUDA Best Practices**: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
- **cudaMemcpy Optimization**: https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/
- **Unified Memory**: https://developer.nvidia.com/blog/unified-memory-cuda-beginners/
- **CUDA Streams**: https://developer.nvidia.com/blog/gpu-pro-tip-cuda-7-streams-simplify-concurrency/
- **Nsight Compute**: https://docs.nvidia.com/nsight-compute/

### Model Conversion & Tools
- **llama.cpp convert.py**: https://github.com/ggerganov/llama.cpp/blob/master/convert.py
- **llama.cpp quantize**: https://github.com/ggerganov/llama.cpp/blob/master/examples/quantize/quantize.cpp
- **Hugging Face to GGUF**: https://github.com/ggerganov/llama.cpp/blob/master/convert-hf-to-gguf.py
- **GGUF Viewer**: https://github.com/ggerganov/llama.cpp/blob/master/examples/gguf/gguf.cpp
- **Model Inspection Tools**: https://github.com/ggerganov/llama.cpp/tree/master/examples

### Model Zoo & Datasets
- **Hugging Face GGUF Models**: https://huggingface.co/models?library=gguf
- **TheBloke GGUF Collection**: https://huggingface.co/TheBloke
- **Qwen GGUF Models**: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF
- **Phi-3 GGUF Models**: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf
- **WikiText-2 Dataset**: https://huggingface.co/datasets/wikitext
- **C4 Dataset**: https://huggingface.co/datasets/c4
- **The Pile**: https://pile.eleuther.ai/

### Validation & Benchmarking
- **Perplexity Calculation**: https://huggingface.co/docs/transformers/perplexity
- **LM Evaluation Harness**: https://github.com/EleutherAI/lm-evaluation-harness
- **llama.cpp Perplexity**: https://github.com/ggerganov/llama.cpp/tree/master/examples/perplexity
- **Tokenizer Conformance**: https://github.com/huggingface/tokenizers/tree/main/bindings/python/tests
- **Reproducibility Testing**: https://pytorch.org/docs/stable/notes/randomness.html

### Community & Forums
- **llama.cpp Discussions**: https://github.com/ggerganov/llama.cpp/discussions
- **r/LocalLLaMA Reddit**: https://www.reddit.com/r/LocalLLaMA/
- **Hugging Face Forums**: https://discuss.huggingface.co/
- **GGML Discord**: https://discord.gg/ggml (if exists)
- **Stack Overflow (llama.cpp tag)**: https://stackoverflow.com/questions/tagged/llama.cpp

### Performance Optimization
- **Memory Bandwidth Optimization**: https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels/
- **Coalesced Memory Access**: https://developer.nvidia.com/blog/parallelforall/how-access-global-memory-efficiently-cuda-c-kernels/
- **Shared Memory Optimization**: https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/
- **Occupancy Calculator**: https://docs.nvidia.com/cuda/cuda-occupancy-calculator/
- **Profiling with Nsight**: https://developer.nvidia.com/nsight-systems

### Platform-Specific Resources
- **Linux mmap Performance**: https://www.kernel.org/doc/html/latest/admin-guide/mm/transhuge.html
- **macOS Memory Management**: https://developer.apple.com/library/archive/documentation/Performance/Conceptual/ManagingMemory/
- **Windows File Mapping**: https://learn.microsoft.com/en-us/windows/win32/memory/file-mapping
- **Cross-Platform mmap**: https://github.com/cloudflare/mmap-sync

### Error Handling & Debugging
- **CUDA Error Handling**: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__ERROR.html
- **Valgrind for Memory Leaks**: https://valgrind.org/docs/manual/quick-start.html
- **CUDA-MEMCHECK**: https://docs.nvidia.com/cuda/cuda-memcheck/
- **GDB CUDA Debugging**: https://docs.nvidia.com/cuda/cuda-gdb/

### Academic Research
- **Efficient Transformers Survey**: https://arxiv.org/abs/2009.06732
- **Long Context Transformers**: https://arxiv.org/abs/2203.08913
- **Attention Optimization**: https://arxiv.org/abs/2112.05682
- **KV-Cache Compression**: https://arxiv.org/abs/2211.05102
- **Streaming LLM**: https://arxiv.org/abs/2309.17453

### Rust-Specific Resources
- **Rust FFI Guide**: https://doc.rust-lang.org/nomicon/ffi.html
- **Rust Unsafe Code**: https://doc.rust-lang.org/book/ch19-01-unsafe-rust.html
- **Rust Memory Layout**: https://doc.rust-lang.org/reference/type-layout.html
- **Rust String Handling**: https://doc.rust-lang.org/std/string/struct.String.html
- **Rust Error Handling**: https://doc.rust-lang.org/book/ch09-00-error-handling.html

### Testing & Validation Tools
- **Criterion.rs (Benchmarking)**: https://github.com/bheisler/criterion.rs
- **proptest (Property Testing)**: https://github.com/proptest-rs/proptest
- **quickcheck (Property Testing)**: https://github.com/BurntSushi/quickcheck
- **Fuzzing with cargo-fuzz**: https://github.com/rust-fuzz/cargo-fuzz

### Licensing & Legal
- **llama.cpp License (MIT)**: https://github.com/ggerganov/llama.cpp/blob/master/LICENSE
- **GGML License (MIT)**: https://github.com/ggerganov/ggml/blob/master/LICENSE
- **Llama 2 License**: https://github.com/meta-llama/llama-models/blob/main/models/llama2/LICENSE
- **Llama 3 License**: https://github.com/meta-llama/llama-models/blob/main/models/llama3/LICENSE
- **Qwen License**: https://huggingface.co/Qwen/Qwen2.5-7B/blob/main/LICENSE
- **Phi-3 License (MIT)**: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/blob/main/LICENSE

### Additional Resources
- **Transformer Architecture**: https://arxiv.org/abs/1706.03762
- **Attention Is All You Need**: https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
- **Layer Normalization**: https://arxiv.org/abs/1607.06450
- **Batch Normalization**: https://arxiv.org/abs/1502.03167
- **Dropout**: https://jmlr.org/papers/v15/srivastava14a.html

### Video Lectures & Tutorials
- **llama.cpp Tutorial**: Search YouTube for "llama.cpp tutorial"
- **GGUF Format Explained**: Search YouTube for "GGUF format"
- **BPE Tokenization**: Search YouTube for "byte pair encoding"
- **Memory-Mapped I/O**: Search YouTube for "mmap tutorial"
- **CUDA Programming**: https://www.youtube.com/playlist?list=PLGvfHSgImk4aweyWlhBXNF6XISY3um82_

### Documentation & Knowledge Bases
- **Papers With Code**: https://paperswithcode.com/search?q=llama
- **arXiv Sanity**: http://www.arxiv-sanity.com/
- **Semantic Scholar**: https://www.semanticscholar.org/
- **Google Scholar**: https://scholar.google.com/

### Blogs & Technical Writing
- **Hugging Face Blog**: https://huggingface.co/blog
- **Meta AI Blog**: https://ai.meta.com/blog/
- **Microsoft Research Blog**: https://www.microsoft.com/en-us/research/blog/
- **NVIDIA Developer Blog**: https://developer.nvidia.com/blog/
- **Weights & Biases Blog**: https://wandb.ai/site/articles

### GitHub Repositories
- **llama.cpp**: https://github.com/ggerganov/llama.cpp
- **ggml**: https://github.com/ggerganov/ggml
- **llama-cpp-python**: https://github.com/abetlen/llama-cpp-python
- **llama.cpp Server**: https://github.com/ggerganov/llama.cpp/tree/master/examples/server
- **llama.cpp Bindings**: https://github.com/ggerganov/llama.cpp/tree/master/examples

---

## Research Execution Plan

### Day 1: GGUF Format & Memory-Mapped I/O (8 hours)

**Morning (4 hours)**:
- Read GGUF specification thoroughly (ggml/docs/gguf.md)
- Analyze llama.cpp GGUF loader implementation
- Document GGUF v2 vs v3 differences
- Create GGUF file structure diagrams
- Document metadata system (key-value types)
- Document tensor info section layout

**Afternoon (4 hours)**:
- Research memory-mapped I/O patterns (mmap, madvise)
- Analyze platform differences (Linux, macOS, Windows)
- Document chunked H2D transfer strategies
- Research zero-copy optimization opportunities
- Document error handling for mmap failures
- Create mmap optimization guide

**Deliverables**:
- `docs/gguf-research.md` (partial - format section)
- `docs/gguf-mmap-io.md` (complete)

---

### Day 2: BPE Tokenizer & UTF-8 Safety (8 hours)

**Morning (4 hours)**:
- Read BPE papers (Sennrich et al., GPT-2)
- Analyze llama.cpp tokenizer implementation
- Document byte-level BPE algorithm
- Document GGUF vocab/merges parsing
- Research encoder implementation strategies (trie vs hashmap)
- Research decoder implementation strategies

**Afternoon (4 hours)**:
- Research UTF-8 encoding structure (RFC 3629)
- Document UTF-8 boundary detection algorithm
- Research streaming decode with boundary safety
- Analyze multibyte character edge cases (emoji, CJK)
- Document invalid UTF-8 handling strategies
- Research SIMD UTF-8 validation

**Deliverables**:
- `docs/gguf-bpe-tokenizer.md` (complete)
- `docs/utf8-streaming-safety.md` (complete)

---

### Day 3: Llama Architectures & Validation (8 hours)

**Morning (4 hours)**:
- Read Llama 2 and Llama 3 papers
- Read Qwen 2.5 and Phi-3 technical reports
- Document architecture differences (GQA, RoPE, RMSNorm, SwiGLU)
- Document weight tensor naming conventions
- Research model-specific quirks and workarounds
- Create architecture comparison matrix

**Afternoon (4 hours)**:
- Design validation framework (perplexity, conformance, reproducibility)
- Document test vector strategy
- Research validation datasets (WikiText-2, C4)
- Document edge case testing plan
- Create validation procedure documentation
- Compile annotated bibliography

**Deliverables**:
- `docs/llama-architecture-variants.md` (complete)
- `docs/gguf-validation-framework.md` (complete)
- `docs/gguf-research.md` (complete - all sections)
- `docs/gguf-sources-bibliography.md` (complete)

---

## Key Findings to Document

### GGUF Format Critical Details
1. **Magic Bytes**: 0x46554747 ("GGUF" in ASCII, little-endian)
2. **Version**: v3 is current (tensor type encoding changes)
3. **Alignment**: 32-byte alignment for tensor data
4. **Metadata**: Key-value system with 13 value types
5. **Tensor Info**: Name, n_dims, dims[], type, offset

### Memory-Mapped I/O Critical Details
1. **Page Alignment**: 4KB on Linux, 16KB on macOS
2. **madvise()**: Use MADV_SEQUENTIAL for linear reads
3. **Chunked Transfer**: 256MB chunks optimal for PCIe Gen3
4. **Zero-Copy**: Possible with CUDA unified memory (limited use)
5. **Error Handling**: Check ENOMEM, EACCES, EINVAL

### BPE Tokenizer Critical Details
1. **Byte-Level**: UTF-8 bytes as base tokens (256 byte tokens)
2. **Merges**: Priority-ordered merge rules (greedy application)
3. **Encoder**: Trie-based for O(n) encoding
4. **Decoder**: Direct lookup for O(1) per token
5. **Special Tokens**: BOS, EOS, PAD, UNK (model-specific IDs)

### UTF-8 Streaming Critical Details
1. **Continuation Byte**: 0b10xxxxxx pattern
2. **Boundary Detection**: Check NOT continuation byte
3. **Partial Buffering**: Buffer incomplete sequences
4. **Invalid Handling**: Replace with U+FFFD (�)
5. **SIMD Validation**: Use SIMD for fast validation

### Llama Architecture Critical Details
1. **RoPE**: Rotation-based position encoding (theta base 10000.0 or 1000000.0)
2. **GQA**: Grouped query attention (n_heads / n_kv_heads groups)
3. **RMSNorm**: Root mean square normalization (no mean subtraction)
4. **SwiGLU**: Swish-gated linear unit (FFN activation)
5. **KV-Cache**: Grouped layout for GQA (memory savings)

---

## Success Metrics

### Research Coverage
- [ ] 100% of GGUF specification documented
- [ ] 100% of BPE algorithm documented
- [ ] 100% of UTF-8 safety requirements documented
- [ ] 100% of Llama architecture variants analyzed
- [ ] 100% of validation strategies defined

### Documentation Quality
- [ ] All diagrams clear and accurate
- [ ] All code examples tested (conceptually)
- [ ] All sources cited and verified
- [ ] All edge cases documented
- [ ] All platform differences noted

### Team Readiness
- [ ] Llama-Beta team understands GGUF format
- [ ] Llama-Beta team understands BPE tokenizer
- [ ] Llama-Beta team understands UTF-8 safety
- [ ] Llama-Beta team understands validation requirements
- [ ] Downstream stories have all necessary context

---

## References

- **Spec**: `bin/.specs/01_M0_worker_orcd.md` §M0-W-1211 (GGUF Parser), §M0-W-1212 (Metadata), §M0-W-1213 (Memory-Mapped I/O), §M0-W-1214 (Tokenizer)
- **Related Stories**: LT-001 (GGUF Header Parser), LT-007 (GGUF Vocab Parsing), LT-009 (BPE Encoder), LT-011 (UTF-8 Streaming)
- **External Docs**: 
  - GGUF Spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
  - llama.cpp: https://github.com/ggerganov/llama.cpp
  - BPE Paper: https://arxiv.org/abs/1508.07909
  - UTF-8 RFC: https://www.rfc-editor.org/rfc/rfc3629

---

**Status**: 📋 Ready for execution  
**Owner**: Llama-Beta  
**Created**: 2025-10-04

---

Detailed by Project Management Team — ready to implement 📋
