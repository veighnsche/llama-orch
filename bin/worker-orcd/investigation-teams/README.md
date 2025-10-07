# Investigation Teams Directory

## 🎨 TEAM PICASSO - Parity Logging (2025-10-07)

**Status:** ⚠️ Logging infrastructure needs rebuild  
**Documents:**
- `PARITY_LOGGING_ARCHITECTURE.md` - **READ THIS FIRST** - Proper async logging design
- `TEAM_PICASSO_CORRECTION.md` - What went wrong and why
- `TEAM_PICASSO_CHRONICLE.md` - Investigation timeline
- `parity/` - Parity artifacts (llama.cpp ground truth ready)

**Quick Summary:**
- ✅ llama.cpp ground truth generated (14 JSONL entries)
- ❌ worker-orcd logging broken (causes HTTP timeouts)
- ✅ Root cause identified (file I/O on hot path)
- ✅ Proper architecture designed (lock-free queue + background thread)

**For Other Teams:**
If you need to add logging checkpoints, see `PARITY_LOGGING_ARCHITECTURE.md` section "Extensibility for Other Teams"

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
