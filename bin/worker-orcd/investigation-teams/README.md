# Investigation Teams Directory

## 🎨 TEAM PICASSO - Parity Logging (2025-10-07)

**Status:** ✅ **COMPLETE** - Logging works, 2 critical bugs fixed, 6 models tested  
**Documents:**
- `parity/FINAL_RESEARCH_SUMMARY.md` - **READ THIS FIRST** - Complete findings
- `parity/MULTI_MODEL_GARBAGE_ANALYSIS.md` - 6-model comparison study
- `parity/LLAMA_CPP_LOGGING_WIRING_VERIFICATION.md` - Technical verification
- `parity/WHY_NO_PARITY.md` - Initial investigation
- `parity/` - All test artifacts and analysis tools

**Quick Summary:**
- ✅ **2 critical bugs fixed** (M0-W-1301 violation, GPU memory access)
- ✅ **6 models tested** (Qwen, Phi-3, TinyLlama, Llama-3-8B, GPT-2)
- ✅ **Logging infrastructure complete** (llama.cpp + worker-orcd)
- ✅ **Analysis tools ready** (`test_logging.sh`, `analyze_logits.py`)
- ✅ **Root cause identified** (model-specific buffer initialization in llama.cpp)

**Key Finding:**
Llama family has best buffer management (0-6% garbage), Phi-3 worst (73% garbage).
Even pure FP32 (GPT-2) has garbage tokens - NOT a quantization issue!

**Quick Start:**
```bash
# Test llama.cpp logging
cd /home/vince/Projects/llama-orch/reference/llama.cpp
./test_logging.sh gpt2

# Analyze logits
./analyze_logits.py /tmp/llama_logging_*/logits.jsonl --stats

# Export for analysis
./analyze_logits.py /tmp/llama_logging_*/logits.jsonl --numpy output.npz
```

**For Other Teams:**
See `../../reference/llama.cpp/ORCH_LOGGING_README.md` for usage guide

---

## Directory Structure

```
investigation-teams/
├── README.md (this file)
├── PARITY_LOGGING_ARCHITECTURE.md  ← Design doc for proper logging
├── TEAM_PICASSO_CHRONICLE.md       ← Investigation timeline
├── TEAM_PICASSO_CORRECTION.md      ← Lessons learned
├── TEAM_PICASSO_HTTP_FIX.md        ← (Obsolete - wrong diagnosis)
├── TEAM_PICASSO_SPEC_ALIGNMENT.md  ← (Obsolete - wrong diagnosis)
└── parity/
    ├── STATUS.md                    ← Current parity status
    ├── README.md                    ← Parity artifacts documentation
    ├── llama_hidden_states.jsonl   ← Ground truth from llama.cpp
    ├── llama_output.log             ← llama.cpp stdout
    └── compare_parity.py            ← Comparison script (ready to use)
```

## How to Use Parity Logging (Once Implemented)

### 1. Build with logging enabled
```bash
cargo build --features cuda,orch_logging --release
```

### 2. Run with environment variables
```bash
ORCH_LOG_FILE=/tmp/our_hidden_states.jsonl \
ORCH_LOG_TEAM="worker-orcd" \
ORCH_LOG_VALUES=10 \
cargo test --test haiku_generation_anti_cheat \
  --features cuda,orch_logging --release -- --ignored
```

### 3. Compare with llama.cpp ground truth
```bash
cd investigation-teams/parity
cp /tmp/our_hidden_states.jsonl .
python3 compare_parity.py > parity_report.csv
```

## Adding New Logging Checkpoints

See `PARITY_LOGGING_ARCHITECTURE.md` for detailed instructions.

**Example:**
```cpp
// In your CUDA kernel
#ifdef ORCH_LOGGING
ORCH_LOG_ATTENTION(attention_output, hidden_dim, token_idx);
#endif
```

Then define the macro in `cuda/src/orch_log.hpp`:
```cpp
#define ORCH_LOG_ATTENTION(ptr, count, token_idx) \
    worker_orch_log::Logger::get_instance().log_values("attention_output", ptr, count, token_idx)
```
