# Qwen Model Debugging Documentation

**Last Updated**: 2025-10-06 11:07

---

## 🚀 Quick Start

**New to this issue?** Start here:

1. Read **`STATUS_SUMMARY.md`** for executive overview
2. Read **`KV_CACHE_FIX_SUMMARY.md`** for latest fix
3. Read **`NEXT_STEPS.md`** for what to do next

**Want technical details?** See **`DEBUGGING_INDEX.md`** for complete document guide.

---

## 📊 Current Status

✅ **Matrix layout bug FIXED** (2025-10-06 10:49)  
✅ **KV cache bug FIXED** (2025-10-06 11:05)  
✅ **Attention mechanism WORKING** (2025-10-06 11:05)  
❌ **Bias corruption** (investigating)

**What works**:
- Matrix multiplications corrected
- Q values in correct range (0.01-0.26)
- Weight loading working
- KV cache properly reading cached positions
- Attention computes over all positions
- Attention weights sum to 1.0

**What doesn't work**:
- Bias values contain huge outliers (-14, -34)
- Model produces diverse but poor quality output
- Output doesn't follow prompt structure

---

## 📚 All Documents

### Executive Summary
- **`STATUS_SUMMARY.md`** - High-level overview, timeline, metrics
- **`DEBUG_README.md`** - This file, navigation guide

### Current Analysis
- **`KV_CACHE_FIX_SUMMARY.md`** ⭐ Latest fix (2025-10-06 11:05)
- **`NEXT_STEPS.md`** ⭐ Action items and priorities
- **`DEBUGGING_INDEX.md`** ⭐ Master index with timeline

### Technical Documentation
- **`MATRIX_LAYOUT_FIX_SUMMARY.md`** - Matrix multiplication fix
- **`ROOT_CAUSE_ANALYSIS.md`** - Matrix layout deep dive
- **`TEST_RESULTS_AFTER_FIX.md`** - Test results after matrix fix

### Historical Context
- **`CRITICAL_FINDING.md`** - Original Q value discovery (RESOLVED)
- **`DEBUG_RUN_RESULTS.md`** - Initial debugging session
- **`MATRIX_TRANSPOSE_FIX.md`** - Incorrect approach (for reference)

---

## 🎯 What to Read When

### "I'm new, what's going on?"
→ `STATUS_SUMMARY.md`

### "What was just fixed?"
→ `KV_CACHE_FIX_SUMMARY.md`

### "What's the current issue?"
→ `NEXT_STEPS.md` (Bias corruption investigation)

### "What should I work on?"
→ `NEXT_STEPS.md`

### "How was the matrix bug fixed?"
→ `MATRIX_LAYOUT_FIX_SUMMARY.md`

### "How was the KV cache bug fixed?"
→ `KV_CACHE_FIX_SUMMARY.md`

### "I want all the technical details"
→ `ROOT_CAUSE_ANALYSIS.md` + `KV_CACHE_FIX_SUMMARY.md`

### "Show me everything"
→ `DEBUGGING_INDEX.md`

---

## 🔍 Document Relationships

```
STATUS_SUMMARY.md (Executive Overview)
    ├── TEST_RESULTS_AFTER_FIX.md (Current State)
    │   ├── Q value comparison
    │   ├── Attention analysis
    │   └── Next investigation steps
    │
    ├── MATRIX_LAYOUT_FIX_SUMMARY.md (Solution)
    │   ├── Before/after comparison
    │   ├── All files modified
    │   └── Matrix dimension tables
    │
    ├── ROOT_CAUSE_ANALYSIS.md (Technical)
    │   ├── How llama.cpp works
    │   ├── cuBLAS details
    │   └── Row-major vs column-major
    │
    └── NEXT_STEPS.md (Action Items)
        ├── Immediate priorities
        ├── Cleanup tasks
        └── Verification checklist

DEBUGGING_INDEX.md (Master Index)
    ├── Complete timeline
    ├── All documents listed
    ├── Quick reference tables
    └── Key learnings

Historical Documents (For Reference):
    ├── CRITICAL_FINDING.md (Original discovery)
    ├── DEBUG_RUN_RESULTS.md (Initial debugging)
    └── MATRIX_TRANSPOSE_FIX.md (Incorrect approach)
```

---

## 🧪 Testing

### Run the test
```bash
cd /home/vince/Projects/llama-orch/bin/worker-orcd
cargo test --release --test haiku_generation_anti_cheat -- --ignored --nocapture 2>&1 | tee test.log
```

### Check results
```bash
# Q values (should be 0.01-0.26)
grep "Q before bias" test.log

# Attention outputs (should vary across positions)
grep "Attention output" test.log

# Token generation (should be diverse, not repetitive)
grep "Generated" test.log
```

### Compare with llama.cpp
```bash
./debug_llama_cpp.sh
diff test.log llama_cpp_debug.log
```

---

## 🎯 Current Priority

**Investigate bias corruption** - QKV bias tensors contain huge outliers

**Next steps**:
1. Check weight loader dequantization
2. Compare with llama.cpp on same model file
3. Inspect GGUF file metadata
4. Verify if Qwen2.5-0.5B actually uses biases

See `NEXT_STEPS.md` for detailed instructions.

---

## 📞 Need Help?

1. Check `STATUS_SUMMARY.md` for current state
2. Check `DEBUGGING_INDEX.md` for complete guide
3. Check test logs for debug output
4. Compare with llama.cpp reference

---

## 🎓 Key Insights

1. **Matrix layout bug is FIXED** (2025-10-06 10:49)
   - Q values now correct (0.01-0.26 range)
   - All matrix multiplications use correct row-major → column-major conversion

2. **KV cache bug is FIXED** (2025-10-06 11:05)
   - Position tracking working correctly
   - Attention computes over all cached positions
   - Attention weights properly normalized

3. **Remaining issue: Bias corruption**
   - QKV bias tensors contain huge outliers (-14, -34)
   - Currently disabled to allow model to run
   - Need to investigate weight loading/quantization

---

**Remember**: Start with `STATUS_SUMMARY.md` for the big picture!
