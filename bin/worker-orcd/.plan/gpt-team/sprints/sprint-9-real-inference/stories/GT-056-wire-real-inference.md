# GT-056: Wire Real Inference to Stub

**Team**: GPT-Gamma 🤖  
**Sprint**: Sprint 9 - Real Inference  
**Size**: S (2-3 hours)  
**Priority**: P0 (M0 blocker)  
**Spec Ref**: FT-050 requirement

---

## Story Description

Replace stub inference implementation with real GPU inference. Wire tokenizer, model, and sampling together.

---

## Current State (STUB)

**File**: `cuda/src/inference_impl.cpp` line 28

```cpp
InferenceImpl::InferenceImpl(...) {
    // ⚠️  WARNING: STUB INFERENCE - NOT REAL GPU INFERENCE
    // Parse the prompt to extract the minute word
    std::string minute_word = "silicon";
    
    size_t start = prompt_.find("word \"");
    if (start != std::string::npos) {
        start += 6;
        size_t end = prompt_.find("\"", start);
        if (end != std::string::npos) {
            minute_word = prompt_.substr(start, end - start);
        }
    }
    
    // Generate a haiku with the minute word
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
}
```

---

## Acceptance Criteria

- [ ] Replace stub with real inference
- [ ] Call `model.tokenizer().encode(prompt)`
- [ ] Call `model.gpt_model()->prefill(tokens)`
- [ ] Loop: call `model.gpt_model()->decode()`
- [ ] Call `model.tokenizer().decode(token_id)`
- [ ] Return real tokens
- [ ] Remove all stub code
- [ ] Remove minute word extraction
- [ ] Remove hardcoded haiku
- [ ] Integration test verifies real haiku generation

---

## Technical Details

### Implementation

**File**: `cuda/src/inference_impl.cpp`

```cpp
InferenceImpl::InferenceImpl(
    ModelImpl& model,
    const char* prompt,
    int max_tokens,
    float temperature,
    uint64_t seed
) : model_(model),
    max_tokens_(max_tokens),
    temperature_(temperature),
    seed_(seed),
    tokens_generated_(0)
{
    // 1. Tokenize prompt
    std::string prompt_str(prompt ? prompt : "");
    auto token_ids = model_.tokenizer()->encode(prompt_str, true);  // add BOS
    
    // 2. Run prefill (process all prompt tokens)
    GPTForwardConfig config;
    config.temperature = temperature;
    config.top_p = 0.9f;
    config.top_k = 50;
    config.seed = seed;
    
    uint32_t first_token = model_.gpt_model()->prefill(
        token_ids.data(),
        token_ids.size(),
        config
    );
    
    // 3. Store first generated token
    generated_token_ids_.push_back(first_token);
}

bool InferenceImpl::next_token(char* token_out, int buffer_size, int* token_index) {
    // Check if we've generated enough tokens
    if (tokens_generated_ >= max_tokens_) {
        return false;
    }
    
    // Get last generated token
    uint32_t last_token = generated_token_ids_.back();
    
    // Decode to text
    std::string token_text = model_.tokenizer()->decode_token(last_token);
    
    // Copy to output buffer
    size_t copy_len = std::min(static_cast<size_t>(buffer_size - 1), token_text.length());
    std::memcpy(token_out, token_text.c_str(), copy_len);
    token_out[copy_len] = '\0';
    
    // Set token index if requested
    if (token_index) {
        *token_index = static_cast<int>(tokens_generated_);
    }
    
    tokens_generated_++;
    
    // Generate next token (if not done)
    if (tokens_generated_ < max_tokens_) {
        GPTForwardConfig config;
        config.temperature = temperature_;
        config.top_p = 0.9f;
        config.top_k = 50;
        config.seed = seed_;
        
        uint32_t next_token = model_.gpt_model()->decode(last_token, config);
        generated_token_ids_.push_back(next_token);
        
        // Check for EOS token
        if (next_token == model_.tokenizer()->eos_token_id()) {
            return true;  // Return current token, but stop after this
        }
    }
    
    return tokens_generated_ < max_tokens_;
}
```

### Wire to ModelImpl

**File**: `cuda/src/model_impl.h`

```cpp
class ModelImpl {
public:
    // Add accessors
    tokenizer::BPETokenizer* tokenizer() { return tokenizer_.get(); }
    model::GPTModel* gpt_model() { return gpt_model_.get(); }
    
private:
    std::unique_ptr<tokenizer::BPETokenizer> tokenizer_;
    std::unique_ptr<model::GPTModel> gpt_model_;
};
```

### Remove Stub Members

**File**: `cuda/src/inference_impl.h`

```cpp
class InferenceImpl {
private:
    ModelImpl& model_;
    int max_tokens_;
    float temperature_;
    uint64_t seed_;
    int tokens_generated_;
    
    // Real inference state
    std::vector<uint32_t> generated_token_ids_;
    
    // REMOVE THESE (stub members):
    // std::string prompt_;
    // std::vector<std::string> stub_tokens_;
    // size_t current_token_idx_;
};
```

---

## Testing

### Integration Test

Run the haiku test and verify:
- ✅ Different haiku each run
- ✅ Minute word appears in haiku
- ✅ No stub warnings
- ✅ Real GPU inference (check `nvidia-smi`)

```bash
REQUIRE_REAL_LLAMA=1 cargo test --test haiku_generation_anti_cheat \
  test_haiku_generation_anti_cheat --features cuda --release \
  -- --ignored --nocapture --test-threads=1
```

**Expected output**:
```
🎨 M0 Haiku Anti-Cheat Test PASSED (REAL GPU INFERENCE)
Minute: 17 ("seventeen")

Haiku:
[actual haiku generated by model, containing "seventeen"]
```

---

## Dependencies

**Upstream**: GT-052, GT-053, GT-054, GT-055 (all pieces must work)  
**Downstream**: GT-057 (test cleanup)

---

## Definition of Done

- [ ] Stub code removed
- [ ] Real inference implemented
- [ ] Tokenizer wired
- [ ] Model wired
- [ ] Integration test passes
- [ ] Different haiku each run
- [ ] No stub warnings
- [ ] No TODOs remain

---

## Estimated Time

**Realistic**: 2-3 hours

---

**Created by**: Project Management Team 📋  
**Assigned to**: GPT-Gamma 🤖  
**Status**: TODO
