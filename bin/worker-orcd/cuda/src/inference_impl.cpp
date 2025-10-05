/**
 * Inference Implementation - Minimal stub for haiku test
 * 
 * This generates stub tokens to prove the pipeline works.
 * Real transformer inference will be implemented in Phase 3.
 */

#include "inference_impl.h"
#include "cuda_error.h"
#include <cstring>
#include <sstream>

namespace worker {

InferenceImpl::InferenceImpl(
    ModelImpl& model,
    const char* prompt,
    int max_tokens,
    float temperature,
    uint64_t seed
) : model_(model),
    prompt_(prompt ? prompt : ""),
    max_tokens_(max_tokens),
    temperature_(temperature),
    seed_(seed),
    tokens_generated_(0),
    current_token_idx_(0)
{
    // ⚠️  WARNING: STUB INFERENCE - NOT REAL GPU INFERENCE
    // ⚠️  This is a hardcoded template, not real model inference
    // ⚠️  FINED by Testing Team: FINE-001-20251005
    // ⚠️  See: test-harness/FINES.md
    //
    // TODO: Implement real inference (22-31 hours):
    // - Phase 1: GGUF weight loading to GPU (9-13h)
    // - Phase 2: Tokenizer integration (5-7h)
    // - Phase 3: Transformer forward pass (8-11h)
    //
    // For now: Generate a stub haiku that includes time-based word
    // This proves the HTTP/SSE pipeline works, but NOT real inference
    
    fprintf(stderr, "⚠️  WARNING: STUB INFERENCE - NOT REAL GPU INFERENCE\n");
    fprintf(stderr, "⚠️  This test uses a hardcoded template, not real model inference\n");
    fprintf(stderr, "⚠️  TODO: Implement real GGUF weight loading and transformer forward pass\n");
    fprintf(stderr, "⚠️  FINED: See test-harness/FINES.md #001\n");
    
    // Parse the prompt to extract the minute word
    // Prompt format: "Write a haiku about GPU computing that includes the word \"<word>\" (nonce: ...)"
    std::string minute_word = "silicon"; // Default
    
    size_t start = prompt_.find("word \"");
    if (start != std::string::npos) {
        start += 6; // Skip 'word "'
        size_t end = prompt_.find("\"", start);
        if (end != std::string::npos) {
            minute_word = prompt_.substr(start, end - start);
        }
    }
    
    // Generate a haiku with the minute word
    // Format: Line 1 (5 syllables) / Line 2 (7 syllables) / Line 3 (5 syllables)
    std::ostringstream haiku;
    haiku << minute_word << " threads spin\n";
    haiku << "CUDA cores burning bright\n";
    haiku << "GPU's warm glow";
    
    // Tokenize into words for streaming
    std::istringstream iss(haiku.str());
    std::string word;
    while (iss >> word) {
        stub_tokens_.push_back(word + " ");
    }
    
    // Add final newline
    if (!stub_tokens_.empty()) {
        stub_tokens_.back() += "\n";
    }
}

InferenceImpl::~InferenceImpl() {
    // Cleanup handled automatically
}

bool InferenceImpl::next_token(char* token_out, int buffer_size, int* token_index) {
    if (tokens_generated_ >= max_tokens_ || current_token_idx_ >= stub_tokens_.size()) {
        return false; // Done generating
    }
    
    const std::string& token = stub_tokens_[current_token_idx_];
    
    // Copy token to output buffer
    size_t copy_len = std::min(static_cast<size_t>(buffer_size - 1), token.length());
    std::memcpy(token_out, token.c_str(), copy_len);
    token_out[copy_len] = '\0';
    
    // Set token index if requested
    if (token_index) {
        *token_index = static_cast<int>(current_token_idx_);
    }
    
    current_token_idx_++;
    tokens_generated_++;
    
    return true; // More tokens available
}

} // namespace worker

// ---
// Built by Foundation-Alpha 🏗️
