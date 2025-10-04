# FT-019-EXT-3: Stop Sequences

**Team**: Foundation-Alpha  
**Sprint**: Sprint 4 - Advanced Sampling  
**Size**: M (2 days)  
**Days**: 5 - 6  
**Spec Ref**: M0-W-1422, GENERATION_PARAMETERS_ANALYSIS.md

---

## Story Description

Implement stop sequences to automatically terminate generation when specific token patterns are matched. Critical for structured output (JSON, code) and user-controlled termination.

**Example**: `stop=["\n\n", "END"]` terminates when double newline or "END" is generated.

---

## Acceptance Criteria

- [ ] Tokenize stop strings (up to 4 sequences)
- [ ] Pattern matching against generated sequence
- [ ] Early termination on match
- [ ] Handles partial matches (no false positives)
- [ ] Handles empty stop sequences
- [ ] Unit tests validate matching (5+ tests)
- [ ] Integration tests validate with generation loop
- [ ] Performance acceptable (<0.1ms per token)

---

## Technical Details

### Stop Sequence Configuration

```cpp
struct StopSequence {
    int* tokens;           // Tokenized sequence
    int length;            // Number of tokens
    
    StopSequence() : tokens(nullptr), length(0) {}
    
    ~StopSequence() {
        if (tokens != nullptr) {
            delete[] tokens;
        }
    }
};

struct SamplingConfig {
    // ... other fields
    
    StopSequence stop_sequences[4];  // Up to 4 stop sequences
    int num_stop_sequences = 0;
};
```

### Tokenization (CPU-Side)

```cpp
/**
 * Tokenize stop strings using model tokenizer.
 * 
 * @param stop_strings Array of stop strings
 * @param num_strings Number of stop strings
 * @param tokenizer Model tokenizer
 * @return Array of tokenized stop sequences
 */
std::vector<StopSequence> tokenize_stop_sequences(
    const std::vector<std::string>& stop_strings,
    const Tokenizer& tokenizer
) {
    std::vector<StopSequence> sequences;
    
    for (const auto& stop_str : stop_strings) {
        StopSequence seq;
        
        // Tokenize string
        std::vector<int> tokens = tokenizer.encode(stop_str);
        
        // Allocate and copy
        seq.length = tokens.size();
        seq.tokens = new int[seq.length];
        std::copy(tokens.begin(), tokens.end(), seq.tokens);
        
        sequences.push_back(seq);
    }
    
    return sequences;
}
```

### Pattern Matching (CPU-Side)

```cpp
/**
 * Check if generated sequence matches any stop sequence.
 * 
 * Uses sliding window comparison against each stop sequence.
 * 
 * @param generated_tokens Array of generated token IDs
 * @param num_generated Number of generated tokens
 * @param config Sampling configuration with stop sequences
 * @return True if any stop sequence matched
 */
bool check_stop_sequences(
    const int* generated_tokens,
    int num_generated,
    const SamplingConfig& config
) {
    // Check each stop sequence
    for (int seq_idx = 0; seq_idx < config.num_stop_sequences; ++seq_idx) {
        const StopSequence& seq = config.stop_sequences[seq_idx];
        
        // Skip if sequence is empty
        if (seq.tokens == nullptr || seq.length == 0) {
            continue;
        }
        
        // Need at least seq.length tokens to match
        if (num_generated < seq.length) {
            continue;
        }
        
        // Check if last seq.length tokens match stop sequence
        bool match = true;
        for (int i = 0; i < seq.length; ++i) {
            int gen_idx = num_generated - seq.length + i;
            if (generated_tokens[gen_idx] != seq.tokens[i]) {
                match = false;
                break;
            }
        }
        
        if (match) {
            return true;  // Stop sequence matched
        }
    }
    
    return false;  // No match
}
```

### Integration with Generation Loop

```cpp
// In InferenceResult::next_token()
int token_id = sample_token(logits, config, rng);

// Add to history
generated_tokens_.push_back(token_id);

// Check stop sequences
if (check_stop_sequences(generated_tokens_.data(), 
                         generated_tokens_.size(), 
                         config)) {
    // Stop sequence matched, terminate generation
    should_stop_ = true;
    return token_id;
}

// Continue generation
return token_id;
```

---

## Testing Strategy

### Unit Tests (5 tests)

1. **SingleSequenceMatch**
   - Given: generated=[1, 2, 3, 4], stop=[[3, 4]]
   - When: check_stop_sequences
   - Then: Returns true

2. **MultipleSequences**
   - Given: generated=[1, 2, 3], stop=[[4, 5], [2, 3]]
   - When: check_stop_sequences
   - Then: Returns true (matches second sequence)

3. **PartialMatch**
   - Given: generated=[1, 2, 3], stop=[[2, 3, 4]]
   - When: check_stop_sequences
   - Then: Returns false (partial match, not complete)

4. **NoMatch**
   - Given: generated=[1, 2, 3], stop=[[4, 5]]
   - When: check_stop_sequences
   - Then: Returns false

5. **EmptyStopSequences**
   - Given: generated=[1, 2, 3], stop=[]
   - When: check_stop_sequences
   - Then: Returns false

### Integration Tests (2 tests)

1. **GenerationStopsOnSequence**
   - Generate with stop=["\n\n"]
   - Verify generation terminates when double newline generated
   - Verify token count < max_tokens

2. **MultipleStopSequences**
   - Generate with stop=["\n\n", "END", "STOP"]
   - Verify generation terminates on first match
   - Verify correct sequence matched

---

## HTTP API Extension

### Request Schema Addition

```json
{
  "job_id": "job-xyz",
  "prompt": "Write a haiku",
  "temperature": 0.7,
  "stop": ["\n\n", "END"]
}
```

### Validation Rules

- `stop`: Array of strings
- Max 4 sequences
- Each sequence max 32 tokens after tokenization
- Empty array = no stop sequences
- Null/omitted = no stop sequences

### Response Schema

```json
{
  "job_id": "job-xyz",
  "tokens": [...],
  "stop_reason": "stop_sequence",  // or "max_tokens" or "error"
  "stop_sequence_matched": "\n\n"  // which sequence matched
}
```

---

## Performance Targets

- **Pattern matching**: <0.1ms per token
- **Memory**: ~512 bytes (4 sequences × 32 tokens × 4 bytes)
- **Overhead**: <1% of total sampling time

---

## Definition of Done

- [ ] Kernel implemented and tested (4 tests)
- [ ] History buffer management implemented
- [ ] Pattern matching implemented
- [ ] Integration tests passing (2 tests)
- [ ] HTTP API extended
- [ ] Performance within budget (<0.1ms)
- [ ] Documentation updated
- [ ] Code reviewed (self-review)

---

## References

- **Spec**: `bin/.specs/01_M0_worker_orcd.md` (M0-W-1422)
- **Analysis**: `bin/.specs/.docs/GENERATION_PARAMETERS_ANALYSIS.md`
- **OpenAI API**: Stop sequences documentation

---
Built by Foundation-Alpha 🏗️
