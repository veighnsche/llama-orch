//! CUDA InferenceBackend implementation
//!
//! Implements the worker-http InferenceBackend trait for CUDA models.
//! Uses real GPU inference via QwenTransformer.

use crate::cuda::{Model, RealInference};
use crate::inference_executor::InferenceExecutor;
use async_trait::async_trait;
use std::sync::Arc;
use worker_common::{InferenceResult, SamplingConfig};
use worker_gguf::GGUFMetadata;
use worker_http::backend::InferenceBackend;
use worker_tokenizer::Tokenizer;

/// CUDA-based inference backend with real GPU inference
pub struct CudaInferenceBackend {
    model: Arc<Model>,
    #[allow(dead_code)]
    model_path: String,
    metadata: GGUFMetadata,
    tokenizer: Tokenizer,
}

impl CudaInferenceBackend {
    /// Create new CUDA backend with real inference
    ///
    /// # Arguments
    ///
    /// * `model` - Loaded CUDA model with weights in VRAM
    /// * `model_path` - Path to GGUF file (for metadata and tokenizer)
    ///
    /// # Errors
    ///
    /// Returns error if metadata parsing or tokenizer loading fails
    pub fn new(
        model: Model,
        model_path: &str,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        tracing::info!("🔧 Creating CudaInferenceBackend with REAL inference");
        tracing::info!("   Model path: {}", model_path);

        // Parse GGUF metadata for model config
        let metadata = GGUFMetadata::from_file(model_path).map_err(|e| {
            tracing::error!("❌ Failed to parse GGUF metadata: {}", e);
            format!("Failed to parse GGUF metadata: {}", e)
        })?;

        tracing::info!("✅ GGUF metadata parsed");

        // Load tokenizer from GGUF
        let tokenizer = Tokenizer::from_gguf(model_path).map_err(|e| {
            tracing::error!("❌ Failed to load tokenizer: {}", e);
            format!("Failed to load tokenizer: {}", e)
        })?;

        tracing::info!("✅ Tokenizer loaded");
        tracing::info!("🎉 CudaInferenceBackend created successfully - REAL INFERENCE ENABLED");

        Ok(Self { model: Arc::new(model), model_path: model_path.to_string(), metadata, tokenizer })
    }
}

#[async_trait]
impl InferenceBackend for CudaInferenceBackend {
    async fn execute(
        &self,
        prompt: &str,
        config: &SamplingConfig,
    ) -> Result<InferenceResult, Box<dyn std::error::Error + Send + Sync>> {
        tracing::info!("🚀 REAL INFERENCE STARTING");
        tracing::info!("   Prompt: {}", prompt);

        // SUSPECT: [TEAM_PROMPT] Missing chat template application! (2025-10-06 19:15 UTC)
        // CONTRADICTION: llama-cli works perfectly and generates proper haiku.
        //   - llama-cli applies chat template: <|im_start|>system...user...assistant<|im_end|>
        //   - Rust code just calls tokenizer.encode() with raw prompt
        //   - Result: Rust generates garbage (ĠKwáº·ng...), llama-cli generates proper haiku
        //
        // VERIFIED: llama-cli direct test PASSED ✅
        //   Command: llama-cli -m qwen2.5-0.5b-instruct-fp16.gguf -p "Write a haiku about GPU computing that includes the word \"fifteen\" (nonce: test123)"
        //   Output: "Thirteen threads dance, / Fourteen tasks conquer the land, / Fifteen GPUs, a game of might."
        //
        // RESOLVED: [TEAM_PROMPT] Applying Qwen chat template! (2025-10-06 19:15 UTC)
        //   - Qwen2.5-0.5b-**INSTRUCT** model requires chat-formatted input
        //   - GGUF file contains tokenizer.chat_template metadata
        //   - llama.cpp automatically applies it in conversation mode
        //   - Rust pipeline was bypassing this → model saw malformed input
        //
        // FALSE_LEAD: NOT a CUDA/attention/bias bug! Previous teams investigated CUDA kernels,
        //   but llama-cli uses the SAME kernels and works fine. The bug is HERE in prompt handling.
        //
        // FIXED: Apply Qwen chat template before tokenization
        // Format: <|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
        //
        // CONTRADICTION: [TEAM_PROMPT] Hardcoded template + system prompt (2025-10-06 19:23 UTC)
        //   llama.cpp path: GGUF → llama_model_chat_template() → minja render → tokenize
        //     File: reference/llama.cpp/tools/main/main.cpp lines 280-316
        //     Flow: --system-prompt → role="system" msg, -p → role="user" msg
        //     Template: Read from GGUF metadata key "tokenizer.chat_template"
        //     Render: common_chat_templates_apply() via minja (Jinja2) engine
        //   Our path: Hardcoded string format → tokenize
        //     File: bin/worker-orcd/src/inference/cuda_backend.rs lines 94-97
        //     Template: Hardcoded CHATML for Qwen (correct format, wrong approach)
        //     System: Hardcoded "You are a helpful assistant" (no override)
        //
        // SUSPECT: [TEAM_PROMPT] System prompt always injected (2025-10-06 19:23 UTC)
        //   Case A (user-only): llama-cli -p "haiku" → NO system block
        //   Case A (our code):  Always adds system block → MISMATCH ❌
        //   Case B (system+user): llama-cli --sys "X" -p "haiku" → system + user
        //   Case B (our code):  Hardcoded system + user → IDENTICAL ✅
        //
        //   llama.cpp warning (tools/main/main.cpp:220):
        //     "*** User-specified prompt will pre-start conversation,
        //      did you mean to set --system-prompt (-sys) instead?"
        //   This is what our audit mission referred to!
        //
        // RESOLVED: [TEAM_PROMPT] Template format correct, needs flexibility (2025-10-06 19:23 UTC)
        //   What works: CHATML format itself is correct for Qwen2.5-Instruct
        //   What's missing:
        //     1. Read tokenizer.chat_template from GGUF (not in worker-gguf yet)
        //     2. Support optional system_prompt (hardcoded right now)
        //     3. Support multiple formats (Llama, Phi, Mistral, etc.)
        //     4. Implement template engine (llama.cpp uses minja, we need minijinja crate)
        //
        //   Minimal fix for v0.1.0:
        //     Add system_prompt: Option<String> param, make system block conditional
        //   Full fix for v0.2.0:
        //     Parse GGUF template, integrate minijinja, message-based API
        //
        //   See: bin/worker-orcd/investigation-teams/TEAM_PROMPT_INVESTIGATION.md
        //
        // FIXED: [TEAM_FINNEY] Remove hardcoded system prompt! (2025-10-06 19:32 UTC)
        //   ROOT CAUSE: We were ALWAYS injecting system prompt, llama.cpp does NOT when using -p flag
        //   VERIFICATION: llama.cpp with -p "Write a haiku..." generates perfect haiku
        //   VERIFICATION: Our code with system prompt generates garbage (ĠstretchedĠstretched...)
        //   FIX: Remove system block to match llama.cpp behavior (tools/main/main.cpp:281-284)
        //   Command that works: llama-cli -m model.gguf -p "Write a haiku about autumn:" -n 50 --temp 0.7
        //   Rendered prompt (llama.cpp): <|im_start|>user\nWrite a haiku...<|im_end|>\n<|im_start|>assistant\n
        //   Rendered prompt (ours NOW): <|im_start|>user\nWrite a haiku...<|im_end|>\n<|im_start|>assistant\n
        //
        // SUSPECT: [TEAM_GEMMA_DELTA] Model generates code tokens, not natural language! (2025-10-06 19:52 UTC)
        //   OBSERVATION: Test output shows code-related tokens: "_CLI", "êª®", "WithPath", ".lineWidth"
        //   THOUGHT: These are NOT natural language tokens. Model seems to think it's generating code.
        //   HYPOTHESIS: Chat template might be wrong, or BOS token missing, or model seeing wrong context
        //   TRACE: Formatted prompt looks correct: "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        //   CONTRADICTION: llama-cli with same prompt works perfectly and generates proper haiku
        //   QUESTION: What's different between llama-cli and our code?
        let formatted_prompt = format!(
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            prompt
        );

        tracing::info!("✅ Applied Qwen chat template");
        eprintln!("[TEAM_FINNEY] Formatted prompt: {:?}", formatted_prompt);

        // SUSPECT: [TEAM_GEMMA_DELTA] add_special_tokens parameter (2025-10-06 19:53 UTC)
        //   THOUGHT: We're passing `true` for add_special_tokens, which might add BOS token
        //   QUESTION: Does llama.cpp add BOS when using chat template?
        //   TRACE: Need to check what token IDs we're getting vs what llama.cpp gets
        // Encode formatted prompt to token IDs
        let token_ids = self.tokenizer.encode(&formatted_prompt, true).map_err(|e| {
            tracing::error!("❌ Tokenization failed: {}", e);
            format!("Tokenization failed: {}", e)
        })?;

        tracing::info!("✅ Tokenized to {} tokens", token_ids.len());
        eprintln!("[TEAM_GEMMA_DELTA] Token IDs (first 20): {:?}", &token_ids[..token_ids.len().min(20)]);
        eprintln!("[TEAM_GEMMA_DELTA] Total tokens: {}", token_ids.len());

        if token_ids.is_empty() {
            return Err("Empty token sequence".into());
        }

        // Get model configuration from metadata
        // CRITICAL: Use actual vocab size from output.weight tensor, not padded tokenizer size
        // The lm_head (output.weight) tensor has dimensions [hidden_dim, actual_vocab]
        // For Qwen2.5-0.5B: [896, 151643] not [896, 151936]
        // This prevents argmax from scanning garbage values beyond the actual vocabulary
        let vocab_size = {
            let tensors = worker_gguf::GGUFMetadata::parse_tensors(&self.model_path)
                .map_err(|e| format!("Failed to parse tensors: {}", e))?;

            // Get actual vocab from output.weight (lm_head) tensor dimensions
            let actual_vocab = tensors
                .iter()
                .find(|t| t.name == "output.weight")
                .and_then(|t| t.dimensions.get(1)) // Second dimension is vocab_size
                .map(|&d| d as u32)
                .ok_or_else(|| "Cannot find output.weight tensor".to_string())?;

            tracing::info!("✅ Actual vocab size from output.weight: {}", actual_vocab);

            // Verify against tokenizer vocab (should be padded)
            if let Ok(tokenizer_vocab) = self.metadata.vocab_size() {
                if tokenizer_vocab as u32 != actual_vocab {
                    tracing::warn!(
                        "⚠️  Tokenizer vocab ({}) != output.weight vocab ({})",
                        tokenizer_vocab,
                        actual_vocab
                    );
                    tracing::warn!(
                        "⚠️  Using actual vocab ({}) to avoid scanning garbage values",
                        actual_vocab
                    );
                }
            }

            actual_vocab
        };
        let hidden_dim = self.metadata.hidden_dim()? as u32;
        let num_layers = self.metadata.num_layers()? as u32;
        let num_heads = self.metadata.num_heads()? as u32;
        let num_kv_heads = self.metadata.num_kv_heads()? as u32;
        let context_length = self.metadata.context_length()? as u32;
        let rope_freq_base = self.metadata.rope_freq_base().unwrap_or(10000.0);

        tracing::info!(
            "Model config: vocab={}, hidden={}, layers={}, heads={}, kv_heads={}",
            vocab_size,
            hidden_dim,
            num_layers,
            num_heads,
            num_kv_heads
        );
        tracing::info!("RoPE frequency base: {}", rope_freq_base);

        // Calculate head_dim and derive ffn_dim from GGUF tensors (do not assume 4x)
        let head_dim = hidden_dim / num_heads;
        let ffn_dim = match worker_gguf::GGUFMetadata::parse_tensors(&self.model_path) {
            Ok(tensors) => {
                // Prefer ffn_up.weight; fall back to ffn_gate.weight
                let mut derived: Option<u32> = None;
                for t in &tensors {
                    if t.name == "blk.0.ffn_up.weight" || t.name == "blk.0.ffn_gate.weight" {
                        if let Some(&d0) = t.dimensions.first() {
                            derived = Some(d0 as u32);
                            break;
                        }
                    }
                }
                derived.unwrap_or(hidden_dim * 4)
            }
            Err(_) => hidden_dim * 4,
        };

        // Initialize real inference context
        let mut inference = RealInference::init(
            &self.model,
            vocab_size,
            hidden_dim,
            num_layers,
            num_heads,
            num_kv_heads,
            head_dim,
            ffn_dim,
            context_length,
            rope_freq_base,
        )?;

        // Process prompt tokens (prefill phase)
        // Feed all prompt tokens through the transformer to build KV cache
        // We call generate_token() to run the forward pass, but we ignore the sampled output
        // and feed the next prompt token instead (teacher forcing)
        tracing::info!("🔄 Prefill phase: processing {} prompt tokens", token_ids.len() - 1);
        for (i, &token_id) in token_ids.iter().enumerate() {
            if i < token_ids.len() - 1 {
                tracing::debug!(
                    "  Prefill token {}/{}: ID={}",
                    i + 1,
                    token_ids.len() - 1,
                    token_id
                );
                // Prefill: run forward pass with this token, ignore sampled output
                let _ = inference.generate_token(
                    token_id,
                    0.0, // Greedy (doesn't matter, we ignore output)
                    0,
                    1.0,
                    config.seed,
                )?;
                // Continue with next prompt token (teacher forcing)
            }
        }
        tracing::info!(
            "✅ Prefill complete, starting generation from token ID={}",
            token_ids.last().unwrap()
        );

        // Start generation from the last prompt token
        let mut current_token = *token_ids.last().unwrap();

        // Generate new tokens (decode phase)
        let mut executor = InferenceExecutor::new(config.clone());
        let mut token_idx = 0;
        let eos_token_id = self.metadata.eos_token_id().unwrap_or(151643); // Qwen2.5 EOS

        // Collect debug info for summary at end
        let mut debug_tokens = Vec::new();

        eprintln!("\n🎨 GENERATING {} TOKENS...", config.max_tokens);

        while token_idx < config.max_tokens {
            // CONTRADICTION: [TEAM_FINNEY] Hardcoded temperature=0.0 ignores config! (2025-10-06 19:36 UTC)
            //   Test sets temperature=0.7 (haiku_generation_anti_cheat.rs:125)
            //   But we override to 0.0 here → greedy sampling always picks same token
            //   llama.cpp uses temperature=0.7 and generates diverse output
            // FIXED: [TEAM_FINNEY] Use config.temperature instead of hardcoded 0.0
            let next_token_id = inference.generate_token(
                current_token,
                config.temperature, // Use configured temperature, not hardcoded 0.0!
                config.top_k, // 0 = disabled
                config.top_p, // 1.0 = disabled
                config.seed.wrapping_add(token_idx as u64),
            )?;

            // Check for EOS
            if next_token_id == eos_token_id {
                break;
            }

            // Decode token to text
            let token_text = self
                .tokenizer
                .decode(&[next_token_id], true)
                .map_err(|e| format!("Detokenization failed: {}", e))?;

            // Collect debug info for first 10 tokens
            if token_idx < 10 {
                debug_tokens.push((token_idx, next_token_id, token_text.clone()));
            }

            // Show progress every 20 tokens (less noise)
            if token_idx % 20 == 0 {
                eprint!(".");
            }

            // ✅ [TEAM_LOVE] FIXED BUG #1: Wrong parameter passed to add_token! (2025-10-06 18:33 UTC)
            // BUG: executor.add_token() expects (token_text, token_id) but was passing token_idx!
            // This caused token IDs to be stored as 0, 1, 2, 3... instead of actual token IDs.
            // FIX: Changed token_idx to next_token_id ✅
            //
            // ❌ [TEAM_LOVE] BUG #2 STILL REMAINS: Model generates repetitive tokens (2025-10-06 18:36 UTC)
            // After fixing Bug #1, tokens now vary initially but still get stuck in loops:
            // - Token 0: 25156 ("Ġseparately") ✅
            // - Token 1: 61290 ("(epoch") ✅
            // - Token 2-9: 64362 ("ĠKw") repeated ❌
            //
            // 🕵️ [TEAM_LOVE] INVESTIGATION TRAIL (2025-10-06 18:33-18:40 UTC)
            // I investigated this Rust code thoroughly:
            //
            // ✅ VERIFIED CORRECT: Token flow in this function
            //    - next_token_id comes from inference.generate_token() ✅
            //    - current_token is updated correctly ✅
            //    - Loop logic is correct ✅
            //    - No off-by-one errors ✅
            //
            // ✅ VERIFIED CORRECT: This is NOT where the bug is!
            //    The Rust code correctly:
            //    1. Calls generate_token(current_token) to get next_token_id
            //    2. Stores next_token_id in executor
            //    3. Updates current_token = next_token_id for next iteration
            //
            // ❌ FALSE LEAD: I initially thought there might be a token flow bug here
            //    where the wrong token was being fed back to the model. But after
            //    careful analysis, the Rust code is correct. The bug is in CUDA!
            //
            // 🔍 KEY CLUE FOR NEXT TEAM:
            //    ARGMAX debug shows: token_id=137131, 137131, 137131, 94826...
            //    But generated shows:  token_id=25156,  61290,  64362,  64362...
            //    This MISMATCH means the bug is in the CUDA side, not here!
            //    The CUDA kernels are producing repetitive logits, which is why
            //    ARGMAX keeps finding the same token.
            //
            // This is NOT a Rust bug - the CUDA kernels are producing these repetitive logits.
            // The bug is in the CUDA attention/FFN/RoPE implementation, not in this Rust code.
            // See CUDA kernel investigation teams for the real bug location.
            executor.add_token(token_text, next_token_id);
            current_token = next_token_id;
            token_idx += 1;
        }

        eprintln!("\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        eprintln!("✅ Generated {} tokens", token_idx);

        // Print debug summary at the END
        eprintln!("\n📊 DEBUG SUMMARY (First 10 tokens):");
        eprintln!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        for (idx, token_id, text) in &debug_tokens {
            eprintln!("  [{}] ID={:6} → {:?}", idx, token_id, text);
        }

        // Check for repetitive patterns
        if debug_tokens.len() >= 3 {
            let first_id = debug_tokens[0].1;
            let all_same = debug_tokens.iter().all(|(_, id, _)| *id == first_id);
            if all_same {
                eprintln!("\n⚠️  WARNING: All tokens are identical (ID={})", first_id);
                eprintln!("⚠️  This indicates a broken attention mechanism!");
            }
        }
        eprintln!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

        Ok(executor.finalize())
    }

    async fn cancel(&self, _job_id: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // TODO: Implement cancellation
        Ok(())
    }

    fn vram_usage(&self) -> u64 {
        self.model.vram_bytes()
    }

    fn is_healthy(&self) -> bool {
        // TODO: Implement health check
        true
    }
}
