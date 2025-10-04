# LT-008: GGUF Merges Parsing

**Team**: Llama-Beta  
**Sprint**: Sprint 2 - GGUF-BPE Tokenizer  
**Size**: M (2 days)  
**Days**: 29-30  
**Spec Ref**: M0-W-1362

---

## Story Description

Parse BPE merge rules from GGUF metadata to enable byte-level BPE encoding. Extract merge pairs and their priorities from GGUF file to construct the merge table required for tokenization algorithm.

---

## Acceptance Criteria

- [ ] Parse `tokenizer.ggml.merges` array from GGUF metadata
- [ ] Extract merge pairs (e.g., "Ġ t" → merge priority 0)
- [ ] Build merge table with priorities (pair → priority)
- [ ] Validate merge count matches expected value
- [ ] Handle byte-level BPE format (space = "Ġ", newline = "Ċ")
- [ ] Sort merges by priority (lower priority = earlier merge)
- [ ] Unit tests validate merge parsing for Qwen2.5-0.5B
- [ ] Unit tests validate merge priority ordering
- [ ] Error handling for missing or malformed merges
- [ ] Log merge count at INFO level

---

## Dependencies

### Upstream (Blocks This Story)
- LT-001: GGUF Header Parser (needs metadata structure)
- LT-002: GGUF Metadata Extraction (needs metadata access)

### Downstream (This Story Blocks)
- LT-009: Byte-Level BPE Encoder (needs merge table)

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/cuda/src/tokenizer/merges.cpp` - Merges parser
- `bin/worker-orcd/cuda/src/tokenizer/merges.h` - Merges struct
- `bin/worker-orcd/src/tokenizer/merges.rs` - Rust merges struct

### Key Interfaces
```cpp
struct MergePair {
    std::string left;
    std::string right;
};

struct MergeTable {
    std::map<MergePair, uint32_t> merge_priority;  // pair → priority
    uint32_t merge_count;
};

class MergesParser {
public:
    // Parse BPE merges from GGUF metadata
    static Result<MergeTable> parse(const GGUFMetadata& metadata);
    
private:
    static std::vector<std::string> extract_merges(const GGUFMetadata& metadata);
    static MergePair parse_merge_line(const std::string& line);
};

// Comparison operator for map key
bool operator<(const MergePair& a, const MergePair& b) {
    if (a.left != b.left) return a.left < b.left;
    return a.right < b.right;
}
```

```rust
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct MergePair {
    pub left: String,
    pub right: String,
}

#[derive(Debug, Clone)]
pub struct MergeTable {
    pub merge_priority: BTreeMap<MergePair, u32>,
    pub merge_count: u32,
}

impl MergesParser {
    pub fn parse(metadata: &GGUFMetadata) -> Result<MergeTable, MergeError>;
}
```

### Implementation Notes
- Parse `tokenizer.ggml.merges` as array of strings
- Each merge line format: `"<left> <right>"` (space-separated)
- Assign priorities sequentially: first merge → priority 0, second → priority 1, etc.
- Lower priority = earlier merge in BPE algorithm
- Handle byte-level BPE special characters:
  - `Ġ` represents space (U+0120)
  - `Ċ` represents newline (U+010A)
- Build map for O(log n) lookup during encoding
- Log merge count at INFO level

**Parsing Logic**:
```cpp
Result<MergeTable> parse(const GGUFMetadata& metadata) {
    MergeTable table;
    
    // Extract merge lines
    auto merges = extract_merges(metadata);
    table.merge_count = merges.size();
    
    // Parse each merge and assign priority
    for (uint32_t priority = 0; priority < merges.size(); ++priority) {
        auto pair = parse_merge_line(merges[priority]);
        table.merge_priority[pair] = priority;
    }
    
    return Ok(table);
}

MergePair parse_merge_line(const std::string& line) {
    // Split on space: "Ġ t" → {"Ġ", "t"}
    auto space_pos = line.find(' ');
    return {
        line.substr(0, space_pos),
        line.substr(space_pos + 1)
    };
}
```

---

## Testing Strategy

### Unit Tests
- Test merge parsing for Qwen2.5-0.5B (merge_count ~151,000)
- Test merge line parsing ("Ġ t" → {left="Ġ", right="t"})
- Test priority assignment (first merge → priority 0)
- Test merge table lookup (pair → priority)
- Test byte-level BPE character handling (Ġ, Ċ)
- Test error handling for malformed merge lines
- Test merge count validation

### Integration Tests
- Test full merge parsing from real GGUF file
- Test merge table construction
- Test merge priority ordering

### Manual Verification
1. Load Qwen2.5-0.5B GGUF
2. Parse merges
3. Verify merge_count ~151,000
4. Verify first merge has priority 0
5. Check logs show merge count

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Code reviewed
- [ ] Unit tests passing (7+ tests)
- [ ] Integration tests passing
- [ ] Documentation updated
- [ ] Story marked complete in day-tracker.md

---

## References

- Spec: `bin/.specs/01_M0_worker_orcd.md` Section 6.4 (Tokenization)
- BPE Paper: https://arxiv.org/abs/1508.07909
- Byte-Level BPE: https://arxiv.org/abs/1909.03341
- GGUF Spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- Related Stories: LT-001, LT-002, LT-009

---

**Status**: Ready for execution  
**Owner**: Llama-Beta  
**Created**: 2025-10-04

---

Detailed by Project Management Team — ready to implement 📋
