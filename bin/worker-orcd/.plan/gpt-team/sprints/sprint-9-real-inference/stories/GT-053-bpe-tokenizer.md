# GT-053: BPE Tokenizer Implementation

**Team**: GPT-Gamma 🤖  
**Sprint**: Sprint 9 - Real Inference  
**Size**: M (5-7 hours)  
**Priority**: P0 (M0 blocker)  
**Spec Ref**: Line 105, 786 (M0 spec)

---

## Story Description

Implement BPE (Byte-Pair Encoding) tokenizer to encode prompts to token IDs and decode token IDs back to text.

---

## Current State

**No tokenizer exists**. Search results:
```bash
$ find cuda/src -name "*tokenizer*"
# No results
```

---

## Acceptance Criteria

- [ ] Create `cuda/src/tokenizer/bpe_tokenizer.h`
- [ ] Create `cuda/src/tokenizer/bpe_tokenizer.cpp`
- [ ] Extract vocab from GGUF metadata
- [ ] Extract merge rules from GGUF metadata
- [ ] Implement `encode(text) -> vector<uint32_t>`
- [ ] Implement `decode(token_ids) -> string`
- [ ] Handle special tokens (BOS, EOS, UNK)
- [ ] Handle UTF-8 correctly
- [ ] Wire to `ModelImpl`
- [ ] Unit test verifies encode/decode round-trip

---

## Technical Details

### File Structure

**Header**: `cuda/src/tokenizer/bpe_tokenizer.h`
```cpp
#ifndef WORKER_BPE_TOKENIZER_H
#define WORKER_BPE_TOKENIZER_H

#include <string>
#include <vector>
#include <unordered_map>

namespace worker {
namespace tokenizer {

class BPETokenizer {
public:
    BPETokenizer(
        const std::vector<std::string>& vocab,
        const std::vector<std::pair<std::string, std::string>>& merges,
        uint32_t bos_token_id,
        uint32_t eos_token_id,
        uint32_t unk_token_id
    );
    
    // Encode text to token IDs
    std::vector<uint32_t> encode(const std::string& text, bool add_bos = true);
    
    // Decode token IDs to text
    std::string decode(const std::vector<uint32_t>& token_ids);
    
    // Decode single token
    std::string decode_token(uint32_t token_id);
    
    uint32_t vocab_size() const { return vocab_.size(); }
    uint32_t bos_token_id() const { return bos_token_id_; }
    uint32_t eos_token_id() const { return eos_token_id_; }
    
private:
    std::vector<std::string> vocab_;
    std::unordered_map<std::string, uint32_t> token_to_id_;
    std::vector<std::pair<std::string, std::string>> merges_;
    uint32_t bos_token_id_;
    uint32_t eos_token_id_;
    uint32_t unk_token_id_;
    
    std::vector<std::string> byte_pair_encode(const std::string& word);
};

// Factory: Load tokenizer from GGUF metadata
std::unique_ptr<BPETokenizer> load_bpe_from_gguf(const gguf::GGUFHeader& header);

} // namespace tokenizer
} // namespace worker

#endif
```

### Implementation

**IMPORTANT**: Research provides complete BPE algorithm (see `RESEARCH_RESULTS.md` lines 97-146)

**Key algorithm** (BPE encoding - adapted from research):
```cpp
std::vector<uint32_t> BPETokenizer::encode(const std::string& text, bool add_bos) {
    std::vector<uint32_t> token_ids;
    
    if (add_bos) {
        token_ids.push_back(bos_token_id_);
    }
    
    // 1. Convert text to UTF-8 bytes (byte-level BPE)
    std::vector<uint8_t> bytes(text.begin(), text.end());
    
    // 2. Initialize tokens as byte strings
    std::vector<std::string> tokens;
    for (uint8_t b : bytes) {
        tokens.push_back(byte_to_token_string(b));
    }
    
    // 3. Iteratively apply BPE merges (from RESEARCH_RESULTS.md)
    while (true) {
        // Find adjacent pairs
        std::map<std::pair<std::string, std::string>, int> pairs;
        for (size_t i = 0; i < tokens.size() - 1; i++) {
            pairs[{tokens[i], tokens[i+1]}]++;
        }
        
        if (pairs.empty()) break;
        
        // Find highest-priority merge
        std::pair<std::string, std::string> best_pair;
        int best_rank = INT_MAX;
        for (const auto& [pair, count] : pairs) {
            auto it = merges_.find(pair);
            if (it != merges_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_pair = pair;
            }
        }
        
        if (best_rank == INT_MAX) break;
        
        // Merge the best pair
        std::vector<std::string> new_tokens;
        for (size_t i = 0; i < tokens.size(); ) {
            if (i < tokens.size() - 1 && 
                tokens[i] == best_pair.first && 
                tokens[i+1] == best_pair.second) {
                new_tokens.push_back(tokens[i] + tokens[i+1]);
                i += 2;
            } else {
                new_tokens.push_back(tokens[i]);
                i++;
            }
        }
        tokens = std::move(new_tokens);
    }
    
    // 4. Convert tokens to IDs
    for (const auto& token : tokens) {
        auto it = token_to_id_.find(token);
        if (it != token_to_id_.end()) {
            token_ids.push_back(it->second);
        } else {
            token_ids.push_back(unk_token_id_);
        }
    }
    
    return token_ids;
}
```

### Extract from GGUF

```cpp
std::unique_ptr<BPETokenizer> load_bpe_from_gguf(const gguf::GGUFHeader& header) {
    // Extract vocab
    std::vector<std::string> vocab;
    for (const auto& kv : header.metadata) {
        if (kv.key == "tokenizer.ggml.tokens" && kv.value_type == gguf::GGUFValueType::ARRAY) {
            vocab = kv.array_string_value;
            break;
        }
    }
    
    // Extract merges
    std::vector<std::pair<std::string, std::string>> merges;
    for (const auto& kv : header.metadata) {
        if (kv.key == "tokenizer.ggml.merges") {
            // Parse merge rules
            for (const auto& merge_str : kv.array_string_value) {
                auto parts = split(merge_str, ' ');
                if (parts.size() == 2) {
                    merges.emplace_back(parts[0], parts[1]);
                }
            }
            break;
        }
    }
    
    // Extract special tokens
    uint32_t bos_id = find_special_token_id(header.metadata, "tokenizer.ggml.bos_token_id");
    uint32_t eos_id = find_special_token_id(header.metadata, "tokenizer.ggml.eos_token_id");
    uint32_t unk_id = find_special_token_id(header.metadata, "tokenizer.ggml.unknown_token_id");
    
    return std::make_unique<BPETokenizer>(vocab, merges, bos_id, eos_id, unk_id);
}
```

### Wire to ModelImpl

**File**: `cuda/src/model_impl.h`
```cpp
class ModelImpl {
public:
    // ...
    tokenizer::BPETokenizer* tokenizer() { return tokenizer_.get(); }
    
private:
    std::unique_ptr<tokenizer::BPETokenizer> tokenizer_;
};
```

**File**: `cuda/src/model_impl.cpp`
```cpp
ModelImpl::ModelImpl(Context& ctx, const char* model_path) {
    // Load GGUF
    auto mmap = std::make_unique<MmapFile>(MmapFile::open(model_path));
    auto header = gguf::parse_gguf_header(mmap_->data(), mmap_->size());
    
    // Load tokenizer
    tokenizer_ = tokenizer::load_bpe_from_gguf(header);
    
    // Load model weights...
}
```

---

## Testing

```cpp
TEST(BPETokenizer, EncodeDecodeRoundTrip) {
    std::string path = "/.../qwen2.5-0.5b-instruct-q4_k_m.gguf";
    auto mmap = io::MmapFile::open(path);
    auto header = gguf::parse_gguf_header(mmap.data(), mmap.size());
    auto tokenizer = tokenizer::load_bpe_from_gguf(header);
    
    std::string text = "Hello, world!";
    auto token_ids = tokenizer->encode(text);
    std::string decoded = tokenizer->decode(token_ids);
    
    EXPECT_EQ(decoded, text);
}
```

---

## Dependencies

**Upstream**: GT-051 (needs GGUF header)  
**Downstream**: GT-056 (needs tokenizer for inference)

---

## Definition of Done

- [ ] Files created
- [ ] Encode/decode implemented
- [ ] GGUF extraction implemented
- [ ] Wired to ModelImpl
- [ ] Unit tests pass
- [ ] CMakeLists.txt updated
- [ ] No TODOs remain

---

## Estimated Time

**Realistic**: 5-7 hours

---

**Created by**: Project Management Team 📋  
**Assigned to**: GPT-Gamma 🤖  
**Status**: TODO
