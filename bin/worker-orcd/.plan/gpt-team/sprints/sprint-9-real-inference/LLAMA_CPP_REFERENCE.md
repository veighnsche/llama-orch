Here’s what llama.cpp actually does, based on the code in the current repository.

High-level pattern
- It’s data-driven. Architecture strings in GGUF map to an enum; a centralized key registry (LLM_KV) builds metadata keys with the arch prefix; and a per-architecture tensor-name table maps logical tensors to GGUF tensor names. The loaders then use these to populate hparams, vocab, and tensors.
- It minimizes hardcoded if/else by:
  - Using llm_arch (enum) + llm_arch_from_string() to normalize architectures.
  - Using LLM_KV to construct “{arch}.{param}” keys (e.g., qwen2.context_length) from a single source of truth.
  - Using LLM_TENSOR_NAMES (per-arch map) to generate the tensor names programmatically for each layer and variant.

1) How do they detect and handle different architectures?
- Detection:
  - The model loader reads general.architecture from GGUF metadata and converts it to the enum llm_arch via llm_arch_from_string():
    - Read: src/llama-model-loader.cpp (constructor) calls get_key(LLM_KV_GENERAL_ARCHITECTURE, …), then constructs LLM_KV with that arch name; see also src/llama-arch.cpp for llm_arch_from_string and llm_arch_name.
    - Entry points: src/llama.cpp (llama_model_load) calls model.load_arch(ml) then model.load_hparams(ml).
- Handling:
  - The enum llm_arch is defined in src/llama-arch.h (covers dozens of architectures).
  - Helpers like llm_arch_is_recurrent/llm_arch_is_hybrid/llm_arch_is_diffusion group certain enums for behavior toggles (src/llama-arch.cpp).
  - Downstream loaders (load_hparams, load_tensors) switch on arch (and/or consult per-arch tables) to apply arch-specific logic. But many differences are absorbed by the LLM_KV and LLM_TENSOR_NAMES registries, reducing switch complexity.

2) How do they map metadata keys dynamically?
- Central registry:
  - LLM_KV_NAMES is a map from llm_kv to key format strings. For arch-scoped keys, the format is "%s.param_name" where %s is the architecture name (from LLM_ARCH_NAMES). Example keys include:
    - "%s.vocab_size", "%s.context_length", "%s.attention.head_count", "%s.rope.freq_base", etc. (see the long LLM_KV_NAMES table in src/llama-arch.cpp).
  - LLM_ARCH_NAMES maps llm_arch enum to strings like "llama", "qwen2", "gpt2", etc. (src/llama-arch.cpp).
- Key construction:
  - LLM_KV::operator()(llm_kv) calls format(LLM_KV_NAMES.at(kv), LLM_ARCH_NAMES.at(arch)) and appends an optional suffix. That yields, e.g., "qwen2.context_length" for the same logical key on a different model.
- Optional vs required:
  - The loader API supports optional keys via get_key(..., required=false). You can see patterns like:
    - Reading general.* fields (name, author, version, etc.) as optional in places like llama-vocab.cpp (ml.get_key(..., false)).
  - Where values are essential, code will “require” them (or throw). This allows graceful fallback or defaults where appropriate.

3) How do they map tensor names dynamically?
- Central registry:
  - LLM_TENSOR_NAMES is a map: arch -> (llm_tensor -> printf-style name). It covers each supported architecture’s tensor naming scheme (src/llama-arch.cpp).
  - Examples:
    - LLAMA: "blk.%d.attn_q", "blk.%d.ffn_gate", "token_embd", "output", "rope_freqs", etc.
    - QWEN variants: arch-specific choices like separate Q/K/V vs fused QKV, MoE variants, extra/expert tensors, etc.
    - Each entry uses format placeholders for layer and sub-indexes, e.g., "blk.%d.ffn_gate.%d".
- Name construction:
  - LLM_TN_IMPL::str() formats the per-arch pattern with layer indices and optional suffix (src/llama-arch.cpp).
  - If a logical tensor isn’t defined for the given arch, str() returns "__missing__", letting the loader treat it as absent.
- Optional tensors and variants:
  - Optionality is data-driven: if a tensor isn’t in the per-arch table, it won’t be requested. Some architectures expose ATTN_QKV (fused) while others expose ATTN_Q/ATTN_K/ATTN_V (split). The loader uses the mapping that exists for the selected arch.

4) How do they handle architecture-specific features?
- HParams via metadata:
  - The LLM_KV registry includes keys for attention, RoPE, scaling, MoE settings, pooling, etc. (e.g., "%s.attention.head_count", "%s.attention.head_count_kv", "%s.rope.freq_base", "%s.rope.scaling.*", "%s.expert_count", "%s.pooling_type", …).
  - load_hparams(ml) (invoked after load_arch) reads these values via LLM_KV and populates llama_hparams (see src/llama-hparams.h for the structure and fields).
- Derived behavior:
  - Attention type (MHA/GQA/MQA) can be derived from head_count vs head_count_kv.
  - RoPE presence or specifics come from "%s.rope.*" keys; learned position embeddings would be indicated by the presence/absence of these keys (and corresponding tensors in the arch’s map).
  - QKV fused vs split is controlled by the per-arch tensor name table: if ATTN_QKV exists, it’s fused; if ATTN_Q/K/V exist, it’s split.
  - MoE (experts, shared experts, gating) is data-driven through the many "%s.expert_*" keys and matching tensor entries per architecture.
- Group traits:
  - Helpers like llm_arch_is_recurrent/hybrid/diffusion flip internal execution paths (e.g., SSM/RWKV, hybrid Mamba+Attention stacks, diffusion).

5) How do they handle tokenizer variations?
- Metadata-driven tokenizer:
  - Key fields are under tokenizer.* (tokenizer.ggml.model, tokenizer.ggml.tokens, tokenizer.ggml.merges, tokenizer.ggml.bos_token_id, …). See LLM_KV entries in src/llama-arch.cpp.
  - llama_vocab::impl::load reads general_arch, tokenizer_pre, and other tokenizer keys to select behavior (src/llama-vocab.cpp). It has specific handling for certain tokenizer presets (e.g., "jina-v2-*", "nomic-bert-moe", etc.) based on tokenizer_pre and/or architecture.
- Supported formats:
  - BPE (GPT-2-like), sentencepiece/unigram and other variants are mapped from tokenizer.ggml.model and related arrays/IDs. There is also support for huggingface JSON and RWKV tokenizer worlds via dedicated keys (tokenizer.huggingface.json, tokenizer.rwkv.world).
- Behavior toggles:
  - BOS/EOS usage, chat templates (tokenizer.chat_template), special FIM token IDs, etc., are all read from metadata keys if present.

6) How do they handle quantization formats?
- Container vs tensor types:
  - general.quantization_version tracks GGUF quantization schema versioning at the file level.
  - Individual tensors carry their own ggml_type in GGUF; the loader places each tensor in an appropriate buffer/backend based on its type. Mixed quantization is therefore just “whatever the tensor says.”
- Where it happens:
  - The quantization logic is primarily in ggml (quant kernels, types, packing). llama.cpp reads and respects the per-tensor types, and the loader plumbs them into buffers/schedulers. You’ll see memory/buffer breakdowns by backend in llama_model and ggml backend integration, with no hardcoded assumptions per architecture.

What to copy for a dynamic implementation
- Use an enum + string map for architectures:
  - Parse general.architecture into an enum and carry it throughout.
- Centralize metadata key construction:
  - A keyed registry of logical parameters to "%s.param" patterns, where %s is the arch name, exactly like LLM_KV and LLM_ARCH_NAMES. Add general.* keys outside of arch namespace.
- Centralize tensor naming:
  - A per-architecture table mapping your logical tensor IDs to printf-style GGUF names. Keep variants (fused QKV vs split; MoE tensors) data-driven here.
- Make keys/tensors optional by API:
  - Reader functions should accept required=false and use defaults or skip behaviors when absent. Treat “not in the arch map” as cleanly optional.
- Derive features from metadata:
  - Head counts for MQA/GQA/MHA; rope fields for position embedding setup; expert counts/scale/norm for MoE; scaling knobs (residual/embedding/attention) for families like Granite; arch grouping helpers to steer execution (recurrent/hybrid/diffusion).
- Tokenizer via metadata:
  - Detect tokenizer type and behavior entirely from tokenizer.* keys and architecture/tokenizer_pre hints; support multiple formats behind a unified interface.
- Quantization per tensor:
  - Trust the tensor’s type; allow mixed quant in the same model. Keep only high-level versioning (general.quantization_version) at the container level.

Concrete references in the repo
- Architecture enum and names:
  - src/llama-arch.h (enum llm_arch)
  - src/llama-arch.cpp (LLM_ARCH_NAMES, llm_arch_name, llm_arch_from_string, llm_arch_is_recurrent/hybrid/diffusion)
- Metadata keys:
  - src/llama-arch.cpp (LLM_KV_NAMES and LLM_KV::operator())
- Tensor names and usage:
  - src/llama-arch.cpp (LLM_TENSOR_NAMES per arch; LLM_TENSOR_INFOS; LLM_TN_IMPL::str())
- Loader and flow:
  - src/llama.cpp (llama_model_load calls load_arch, load_hparams, load_vocab, load_tensors)
  - src/llama-model-loader.cpp/.h (GGUF reading, get_key, get_arch_name/get_arch)
- HParams and features:
  - src/llama-hparams.h (fields consumed by loaders; attention/rope/moe/scales/etc.)
- Tokenizer:
  - src/llama-vocab.cpp (impl::load reads tokenizer metadata and architecture to set up vocab/tokenizer behavior)
- Examples of general.* reads:
  - tokenizer/vocab setup reads general.name and general.architecture optionally (see llama-vocab.cpp)

Answering your checklist succinctly
- “How do they support 20+ architectures without 20+ if statements?”
  - Enum for arch + data tables: LLM_ARCH_NAMES, LLM_KV_NAMES, LLM_TENSOR_NAMES. Switches are reserved for truly divergent logic; most differences are captured in the registries.
- “How do they construct metadata keys dynamically?”
  - LLM_KV takes llm_arch and formats keys from LLM_KV_NAMES with "%s" replaced by the arch name; suffix support for sub-keys.
- “How do they detect which tensors are optional?”
  - If a logical tensor isn’t defined in the arch’s LLM_TENSOR_NAMES, it’s treated as absent. Loader calls can optionally “require” or “get” tensors; missing optional tensors are allowed.
- “How do they handle different tokenizer types?”
  - Using tokenizer.* metadata (model, tokens, merges, ids, chat templates, HF JSON, RWKV world) and some architecture/tokenizer_pre hints in llama-vocab.cpp to select behavior.
- “What patterns can we copy to avoid hardcoding?”
  - The trio of (arch enum) + (metadata key registry) + (per-arch tensor-name registry), with optional/required read APIs and feature derivation from metadata. This yields a generic, extensible system.

  Below is a concise, implementation-oriented set of answers distilled from how llama.cpp handles things today. It’s written to help you implement a generic, data-driven loader without hardcoding model-specific logic.

High-level note
- llama.cpp reads GGUF model metadata and tensors, identifies the architecture from general.architecture, and then uses architecture-specific tables for both hparams (metadata keys) and tensor name patterns.
- Most “dynamic” behavior comes from: (a) a small set of enums + registries (arch, tokenizer, quant), (b) prefix-based metadata lookups with fallbacks, and (c) per-arch tensor maps that are expanded programmatically across layers.

1) Detecting and handling different architectures
- Detection:
  - The GGUF file includes a metadata key general.architecture (e.g., "llama", "qwen2", "gpt2", "falcon", "gptneox", "phi3", etc.).
  - llama.cpp maps that string to an enum (commonly called llm_arch), via a lookup table (string → enum).
- Handling:
  - Once the enum is known, the loader initializes per-arch “hparam key maps” and “tensor maps” that carry prefixes (like llama., qwen2., gpt2.) and the expected tensor name patterns.
  - Unknown architectures: llama.cpp returns an error early (e.g., llm_arch::UNKNOWN) and aborts loading with a user-facing message. No silent fallback for unknown architectures.
- Pattern to copy:
  - Define a small enum Architecture, a mapping function from string → enum, and a registry object that exposes:
    - metadata_prefix (e.g., "llama")
    - tensor name patterns (including placeholders like {L} for layer)
    - known feature defaults (activation, norm type, QKV fused, presence of biases)

2) Mapping metadata keys dynamically
- Key idea:
  - llama.cpp constructs metadata keys by combining an arch-specific prefix with known suffixes, for example: "<prefix>.context_length", "<prefix>.attention.head_count", "<prefix>.rope.freq_base".
  - There’s a helper “key resolver” that takes the logical key (like “rope.freq_base”) and the current arch enum, and tries:
    1) "<arch>.<key>"
    2) Selected fallbacks for backward compatibility (e.g., old aliases, or LLaMA-prefixed defaults)
- Behavior:
  - If a key is missing and marked as required, they error out.
  - If optional, they default sensibly (e.g., set qkv_bias=false if bias keys aren’t present or metadata says none).
- Pattern to copy:
  - Implement a MetadataKeyMapper that:
    - Builds "<prefix>.<param>" from an ArchitectureConfig
    - Tries a small list of legacy aliases if present (use a per-key alias list)
    - Supports get_[type](key, default) for optional keys; get_[type]_required(key) for required keys

3) Mapping tensor names dynamically
- Key idea:
  - llama.cpp maintains per-architecture “tensor maps”: lists of entries of the form:
    - gguf_name pattern (with placeholders like "blk.{L}.attn_q.weight" or fused "blk.{L}.attn_qkv.weight")
    - target/internal role name (how to route into the in-memory model)
    - flags: required vs optional, per-layer vs global, plus shape expectations
  - During load, it expands {L} for each layer and looks up tensors in the GGUF index by exact name.
- Handling variation:
  - Fused vs separate QKV is handled by including both patterns and marking one optional; whichever one exists determines the path.
  - Bias presence is handled similarly (e.g., "...attn_q.bias" optional for Qwen2, absent in pure LLaMA).
- Pattern to copy:
  - Define a TensorMapping entry: { gguf_name_pattern, internal_name_pattern, required, per_layer }
  - For each arch, declare a vector of mappings.
  - Expand per_layer mappings for L in [0..n_layer-1].
  - For each expanded name, lookup; if required missing → error, else skip.
  - Record what you found (e.g., set model.features.qkv_fused = found("attn_qkv.weight"))

4) Handling architecture-specific features
- RoPE vs learned positions:
  - Detect via presence of rope keys under the arch prefix, e.g., "<prefix>.rope.freq_base", "<prefix>.rope.scaling.type", "<prefix>.rope.theta".
  - If not present, assume learned/absolute positions (depends on arch defaults).
- Attention type (MHA/GQA/MQA):
  - Read n_head = "<prefix>.attention.head_count" and n_head_kv = "<prefix>.attention.head_count_kv" (default: n_head).
  - If n_head_kv == 1 → MQA; else if n_head_kv < n_head → GQA; else → MHA.
- Fused QKV:
  - Detect by tensor presence: "blk.{L}.attn_qkv.weight" (fused) vs separate "attn_q.weight", "attn_k.weight", "attn_v.weight".
- Bias detection:
  - Same pattern: check for bias tensors (e.g., "...attn_q.bias", "...attn_k.bias", "...attn_v.bias"). If present across layers, set has_qkv_bias=true; otherwise false.
- Activation, Norm, MLP shape:
  - Read from metadata if provided (e.g., "<prefix>.ffn.activation_type"), else fall back to arch defaults (e.g., LLaMA: RMSNorm + SwiGLU; GPT-2: LayerNorm + GELU; Qwen2: RMSNorm + SwiGLU with biases).
- Pattern to copy:
  - Compute derived features after loading hparams + scanning tensor map:
    - has_rope, rope variant + scaling params
    - attention_type via head_count/head_count_kv
    - qkv_fused via tensor presence
    - has_qkv_bias via tensor presence or explicit metadata
    - norm_type, activation via metadata-with-fallback

5) Handling tokenizer variations
- Detection:
  - Read tokenizer.ggml.model (string). Common values map to an enum, e.g.:
    - "llama" or "spm" → Unigram/SentencePiece
    - "gpt2" or "bpe" → BPE
    - Some models use "wordpiece" → WordPiece
  - llama.cpp has an enum like vocab_type with at least SPM/Unigram and BPE; it also supports special variants (byte-fallback, regex pretokenization).
- Loading:
  - Unigram/SPM:
    - Read tokenizer.ggml.tokens (strings), tokenizer.ggml.scores (floats), plus special token IDs (bos/eos/unk/pad).
  - BPE:
    - Read tokenizer.ggml.tokens and tokenizer.ggml.merges (list of "A B" pairs), plus special token IDs.
  - Optional fields:
    - tokenizer.ggml.add_space_prefix
    - tokenizer.ggml.byte_fallback
    - pre/post-processing regex or special rules for certain arches
- Pattern to copy:
  - Detect TokenizerType from tokenizer.ggml.model; construct tokenizer via a small factory:
    - For BPE, pass tokens + merges + specials
    - For Unigram, pass tokens + scores + pieces + specials
  - Keep a unified Tokenizer interface; expose configuration like byte_fallback, add_space_prefix

6) Handling quantization formats
- Detection:
  - Each tensor in GGUF has a ggml_type (e.g., F32, F16, Q8_0, Q4_0, Q4_1, Q2_K, Q3_K, Q4_K_S, Q4_K_M, Q5_K_S, Q5_K_M, Q6_K, IQ4/IMX, etc.).
  - llama.cpp reads tensor headers in GGUF to get the type per tensor; mixed quantization is normal.
- Registry:
  - ggml defines the enum and helpers like ggml_type_size(), ggml_blck_size(), and dequant kernels in ggml-quants.c.
- Sizing/validation:
  - Size = num_blocks * ggml_type_size(type). Shapes are validated against expected dims from hparams (e.g., [hidden_dim, vocab_size]).
- Pattern to copy:
  - Treat quant as a per-tensor attribute.
  - Implement quant metadata to “dtype” mapping (enum QuantType).
  - For memory estimates, use a type-size table; for runtime, select kernels based on type.

Answers to the GT-051/GT-052/GT-053 specific questions

GT-051 (Config Parsing)
1) How does llama.cpp construct metadata keys for different architectures?
- It prepends an architecture-specific prefix (e.g., "llama", "qwen2", "gpt2") to a logical key suffix, forming "<prefix>.<suffix>". This is centralized via a small resolver that also knows about legacy aliases.

2) Do they have a fallback mechanism for missing keys?
- Yes. For some keys, there are alias fallbacks (older key names). For optional keys, they use defaults (e.g., set head_count_kv = head_count if missing). For required keys, they error out.

3) How do they validate extracted config?
- They cross-check shapes and derived constraints (e.g., hidden_dim divisible by n_head, rope params present if rope enabled). Failure results in an error during model load.

GT-052 (Weight Loading)
1) How does llama.cpp map tensor names for different architectures?
- Using per-arch tensor maps (pattern lists) with placeholders like {L}. At load time, it expands patterns, probes GGUF for exact names, and binds them to internal structures.

2) How do they handle optional tensors?
- Tensor map entries carry a required flag. Missing required → error; missing optional → skipped and may toggle feature flags (e.g., has_qkv_bias=false).

3) How do they detect if QKV are fused or separate?
- By probing presence of fused names like "blk.{L}.attn_qkv.weight". If present (and separate ones not), they mark qkv_fused=true. If separate q/k/v weights are present, qkv_fused=false.

4) How do they detect if bias terms exist?
- By probing for bias tensors (e.g., "...attn_q.bias") across layers. Consistent presence sets has_qkv_bias=true; otherwise false. Some arches also carry explicit metadata flags.

GT-053 (Tokenizer)
1) How does llama.cpp detect tokenizer type?
- From tokenizer.ggml.model string, mapped to an internal enum (e.g., BPE vs Unigram/SPM). It also reads supporting flags like byte_fallback and add_space_prefix.

2) How do they handle byte-level vs character-level BPE?
- Via tokenizer config flags and the presence of merges. For byte-level BPE (GPT-2 style), byte_fallback and pretokenizer settings drive behavior. The loader consumes merges for BPE and scores/pieces for SPM.

3) Do they have a generic tokenizer interface?
- Yes. Internally, they instantiate a tokenizer implementation based on the enum (BPE/SPM) behind a common interface, with special tokens and normalization handled consistently.

Practical implementation checklist for you
- Architecture registry
  - Enum Architecture
  - Map string → enum; define ArchitectureConfig { prefix, defaults (norm, activation, qkv_fused, biases), tensor map }
- Metadata key resolver
  - get(key_suffix, arch, default), get_required(key_suffix, arch)
  - Supports alias lists per key
- Tensor map
  - Vector<TensorMapping> per arch; flags: required, per_layer; pattern expansion for layers
  - Presence-driven feature detection (qkv_fused, has_qkv_bias)
- Feature detection
  - rope presence and scaling params
  - attention_type via head_count/head_count_kv
  - norm_type and activation from metadata with arch defaults
- Tokenizer factory
  - Detect TokenizerType from tokenizer.ggml.model
  - Construct BPE/SPM tokenizers from tokens + merges/scores + specials
- Quantization handling
  - Read ggml_type per tensor; map to internal QuantType
  - Size and kernel selection via type table
  - Mixed quant support by design

If you want, I can translate this into concrete C++ scaffolding (ArchitectureConfig registry, MetadataKeyMapper, TensorMapper classes) to drop into your codebase.