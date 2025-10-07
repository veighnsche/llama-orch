# 🎨 TEAM PICASSO - Document Index

**Quick navigation for all TEAM PICASSO deliverables**

---

## 📋 Start Here

### New to TEAM PICASSO's work?
👉 **Read this first:** [TEAM_PICASSO_SUMMARY.md](TEAM_PICASSO_SUMMARY.md)

**What you'll learn:**
- The cuBLAS verdict (KEEP CUBLAS_OP_T)
- Why the bug is NOT in cuBLAS parameters
- What the parity logging system does
- Next steps for future teams

**Time to read:** 5-10 minutes

---

## 📚 Main Documents

### 1. Investigation Report
📄 **[TEAM_PICASSO_CUBLAS_RESOLUTION.md](TEAM_PICASSO_CUBLAS_RESOLUTION.md)**

**Purpose:** Full evidence report with verdict and recommendations

**Contents:**
- Current state analysis (all 8 matmuls verified)
- llama.cpp ground truth comparison
- SENTINEL vs ALPHA contradiction resolution
- Final verdict: KEEP CUBLAS_OP_T
- Numeric parity logging system details

**When to read:** When you need detailed evidence for the cuBLAS verdict

**Time to read:** 15-20 minutes

---

### 2. Investigation Chronicle
📄 **[TEAM_PICASSO_CHRONICLE.md](TEAM_PICASSO_CHRONICLE.md)**

**Purpose:** Session-by-session investigation log

**Contents:**
- 5 investigation sessions (2025-10-07T14:32Z - 15:38Z)
- Findings, blockers, and next steps for each session
- Evidence summary table
- Reflections and lessons learned

**When to read:** When you want to understand the investigation process

**Time to read:** 10-15 minutes

---

### 3. Summary Report
📄 **[TEAM_PICASSO_SUMMARY.md](TEAM_PICASSO_SUMMARY.md)**

**Purpose:** Executive summary with statistics and handoff

**Contents:**
- Key findings (cuBLAS verdict + parity logging)
- Technical details (current state, test results)
- Lessons learned
- Next steps for future teams
- Completion checklist

**When to read:** When you need a quick overview or handoff summary

**Time to read:** 5-10 minutes

---

## 🔬 Parity Logging Documentation

### 4. Parity Logging README
📄 **[PARITY_LOGGING_README.md](PARITY_LOGGING_README.md)**

**Purpose:** Comprehensive guide to the numeric parity logging system

**Contents:**
- Overview and why it exists
- Quick start guide (step-by-step)
- Architecture and data flow
- Usage examples (C++ and Rust)
- Troubleshooting guide
- Future enhancements

**When to read:** When you want to USE the parity logging system

**Time to read:** 20-30 minutes (includes examples)

---

### 5. Comparison Specification
📄 **[PARITY_COMPARISON_SPEC.md](PARITY_COMPARISON_SPEC.md)**

**Purpose:** Technical specification for comparing JSONL outputs

**Contents:**
- Quick start for investigators
- JSONL schema definition
- Comparison metrics (max_diff, mean_diff, etc.)
- Pass/fail thresholds
- Output format specification
- Example usage

**When to read:** When implementing the comparison script or analyzing results

**Time to read:** 15-20 minutes

---

## 💻 Code Files

### llama.cpp (C++)

**Header-only logger:**
- 📄 `reference/llama.cpp/orch_log.hpp`
  - Lines 1-60: Comprehensive header comments
  - Lines 61-200: Logger implementation
  - Usage: `ORCH_LOG_JSON_TOKEN("checkpoint", ptr, count, "f32", "[896]", token_idx)`

**Logging integration:**
- 📄 `reference/llama.cpp/tools/main/main.cpp`
  - Line 10: Include orch_log.hpp
  - Lines 679-700: Logging calls after llama_decode
  
**Build configuration:**
- 📄 `reference/llama.cpp/tools/main/CMakeLists.txt`
  - Lines 6-10: ORCH_LOGGING option (ON by default)

---

### worker-orcd (Rust)

**Logger implementation:**
- 📄 `bin/worker-orcd/src/orch_log.rs`
  - Lines 1-51: Comprehensive module comments
  - Lines 52-230: Logger implementation
  - Usage: `orch_log!("checkpoint", &values_f32, token_idx)`

**Module declaration:**
- 📄 `bin/worker-orcd/src/lib.rs`
  - Lines 12-14: Conditional module inclusion

**Build configuration:**
- 📄 `bin/worker-orcd/Cargo.toml`
  - Line 31: lazy_static dependency
  - Lines 48-53: orch_logging feature definition

---

## 🎯 Quick Reference

### I want to...

**...understand the cuBLAS verdict**
→ Read [TEAM_PICASSO_SUMMARY.md](TEAM_PICASSO_SUMMARY.md) (5 min)

**...see the full evidence**
→ Read [TEAM_PICASSO_CUBLAS_RESOLUTION.md](TEAM_PICASSO_CUBLAS_RESOLUTION.md) (15 min)

**...use the parity logging system**
→ Read [PARITY_LOGGING_README.md](PARITY_LOGGING_README.md) (20 min)

**...compare llama.cpp and our engine outputs**
→ Read [PARITY_COMPARISON_SPEC.md](PARITY_COMPARISON_SPEC.md) (15 min)

**...understand the investigation process**
→ Read [TEAM_PICASSO_CHRONICLE.md](TEAM_PICASSO_CHRONICLE.md) (10 min)

**...add logging to my code**
→ See examples in [PARITY_LOGGING_README.md](PARITY_LOGGING_README.md) § Usage Examples

**...troubleshoot logging issues**
→ See [PARITY_LOGGING_README.md](PARITY_LOGGING_README.md) § Troubleshooting

**...implement the comparison script**
→ See [PARITY_COMPARISON_SPEC.md](PARITY_COMPARISON_SPEC.md) § Implementation Notes

---

## 📊 Document Map

```
TEAM_PICASSO_INDEX.md (you are here)
├── TEAM_PICASSO_SUMMARY.md ................ Executive summary
├── TEAM_PICASSO_CUBLAS_RESOLUTION.md ...... Full evidence report
├── TEAM_PICASSO_CHRONICLE.md .............. Investigation log
├── PARITY_LOGGING_README.md ............... Logging system guide
└── PARITY_COMPARISON_SPEC.md .............. Comparison spec

Code Files:
├── reference/llama.cpp/
│   ├── orch_log.hpp ....................... C++ logger (header-only)
│   └── tools/main/
│       ├── main.cpp ....................... Logging calls (lines 10, 679-700)
│       └── CMakeLists.txt ................. Build config (lines 6-10)
└── bin/worker-orcd/
    ├── src/
    │   ├── orch_log.rs .................... Rust logger
    │   └── lib.rs ......................... Module declaration (lines 12-14)
    └── Cargo.toml ......................... Feature config (lines 31, 48-53)
```

---

## 🔍 Search Tips

**Find all PICASSO documents:**
```bash
find investigation-teams -name "*PICASSO*" -type f
```

**Find all parity logging code:**
```bash
grep -r "ORCH_LOG" reference/llama.cpp bin/worker-orcd
```

**Find all breadcrumb comments:**
```bash
grep -r "TEAM PICASSO 2025-10-07" reference/llama.cpp bin/worker-orcd
```

---

## 📞 Support

**Questions about the cuBLAS verdict?**
→ See [TEAM_PICASSO_CUBLAS_RESOLUTION.md](TEAM_PICASSO_CUBLAS_RESOLUTION.md) § Final Verdict

**Questions about using the logging system?**
→ See [PARITY_LOGGING_README.md](PARITY_LOGGING_README.md) § Troubleshooting

**Questions about the investigation process?**
→ See [TEAM_PICASSO_CHRONICLE.md](TEAM_PICASSO_CHRONICLE.md) § Reflections

---

**TEAM PICASSO**  
*"When experts disagree, we test everything."*

**Last Updated:** 2025-10-07T15:38Z
