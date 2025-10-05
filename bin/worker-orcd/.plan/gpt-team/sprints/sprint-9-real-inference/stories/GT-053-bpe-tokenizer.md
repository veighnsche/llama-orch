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

**Key algorithm** (BPE encoding):
```cpp
std::vector<uint32_t> BPETokenizer::encode(const std::string& text, bool add_bos) {
    std::vector<uint32_t> token_ids;
    
    if (add_bos) {
        token_ids.push_back(bos_token_id_);
    }
    
    // Split text into words
    std::vector<std::string> words = split_into_words(text);
    
    for (const auto& word : words) {
        // Apply BPE merges
        auto subwords = byte_pair_encode(word);
        
        // Convert to token IDs
        for (const auto& subword : subwords) {
            auto it = token_to_id_.find(subword);
            if (it != token_to_id_.end()) {
                token_ids.push_back(it->second);
            } else {
                token_ids.push_back(unk_token_id_);
            }
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
