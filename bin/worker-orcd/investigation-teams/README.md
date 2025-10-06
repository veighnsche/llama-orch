# Multi-Team Bug Investigation Framework

**Date**: 2025-10-06  
**Bug**: Repetitive token generation - model outputs same token 100+ times  
**Status**: Root cause unknown - 5 teams deployed for parallel investigation

---

## The Problem

The Qwen2.5-0.5B model generates the same token repeatedly because specific positions in the logits array (e.g., 8850, 44394, 137131) contain garbage values (~14-15) instead of normal values (-4 to +4).

**Key Facts**:
- Model file is valid (works in llama.cpp)
- Uses FP16 non-quantized model in test
- Issue appears to be in `project_to_vocab` function's cuBLAS GEMM call
- Garbage values change over time (not static memory corruption)
- Most logits are correct, only specific positions fail

---

## Investigation Teams

### Team Alpha: Memory Layout Forensics
**Focus**: Understanding exact memory layout and access patterns  
**File**: `TEAM_ALPHA_MEMORY_FORENSICS.md`  
**Approach**: Trace memory addresses from GGUF → GPU → cuBLAS  
**Key Question**: Where is cuBLAS reading from vs where it should read from?

### Team Bravo: Reference Implementation Comparison  
**Focus**: Deep dive into llama.cpp's working implementation  
**File**: `TEAM_BRAVO_REFERENCE_COMPARISON.md`  
**Approach**: Extract exact parameters from llama.cpp and compare  
**Key Question**: What does llama.cpp do differently?

### Team Charlie: Mathematical Verification  
**Focus**: Compute ground truth manually and prove correct answer  
**File**: `TEAM_CHARLIE_MANUAL_VERIFICATION.md`  
**Approach**: Manual dot product computation to establish baseline  
**Key Question**: What should the logits actually be?

### Team Delta: Instrumentation & Profiling  
**Focus**: Add comprehensive logging to trace data flow  
**File**: `TEAM_DELTA_INSTRUMENTATION.md`  
**Approach**: Instrument code with detailed printf debugging  
**Key Question**: What is actually happening at runtime?

### Team Echo: First Principles Analysis  
**Focus**: Build understanding from cuBLAS documentation only  
**File**: `TEAM_ECHO_FIRST_PRINCIPLES.md`  
**Approach**: Derive correct parameters from NVIDIA documentation  
**Key Question**: What do the cuBLAS docs say we should do?

---

## Critical Rules

### ✅ YOU CAN CHANGE CODE FOR DATA EXTRACTION
- **Add extensive logging** with your team name prefix: `// [TEAM_X]`
- **Add printf/fprintf statements** to extract data
- **Temporarily modify code** to run custom tests and extract truth data
- **Run the test** as many times as needed to gather evidence

### ❌ BUT DO NOT CHANGE BEHAVIOR PERMANENTLY
- **Do not modify** cuBLAS parameters (unless testing a hypothesis, then REVERT)
- **Do not modify** computation logic permanently
  - **ALWAYS REVERT** temporary changes after gathering data
  - **Document** what you changed and what you learned
 
 ### Why This Approach?
 We need EVIDENCE and GROUND TRUTH data. You can't understand the problem by just reading code - you need to see what's actually happening in memory at runtime. Change whatever you need to extract data, just change it back when done.
 
 ---
 
 ### Test Before You Claim
 - Do not mark anything as "FIXED" unless the Haiku Test actually passes.
 - Always include test evidence (timestamp, token IDs, sample output snippet) in your notes.
 - If your fix improves things but the test still fails, document the improvement but do not claim FIXED.
 
 ### False Claim Correction
 - If a previous team claimed FIXED but your run shows the bug persists, append a correction line immediately below their comment.
 - Do not delete or alter their original text; add your own line with evidence.
 
 Example format to append in code:
 ```cpp
 // ❌ Previous team claimed FIXED — but haiku test still fails here. Suspect race condition remains.
 // Evidence: Haiku test 2025-10-06 18:44 UTC — token ID 64362 repeats at steps 2-9.
 ```
 
 ---
  
  ## Workflow
  
  ### Step 1: Each Team Reads Their Brief
  - Read your team's investigation file (`TEAM_*_*.md`)
  - Understand your specific approach
  - Review the key files listed in your brief

### Step 2: Conduct Investigation
- **Add extensive logging** to extract data
- **Implement verification tests** to compute ground truth
- **Run the test multiple times** to gather evidence
- **Temporarily modify code** to test hypotheses (then revert!)
- **Document your findings** as you go with comments

### Step 3: Write Results
- Create your results file: `investigation-teams/TEAM_*_RESULTS.md`
- **Include test output** (paste terminal logs)
- **Show your work** (calculations, tables, diagrams)
- **Provide evidence** for your conclusions
- **Propose a specific fix** with justification

### Step 4: Revert Temporary Changes
- **Remove test code** you added (keep explanatory comments)
- **Restore original behavior** 
- **Commit only your RESULTS.md** and explanatory comments

### Step 5: Team Sync
- Compare findings across all teams
- Identify consensus on root cause
- Reconcile any conflicting theories
- Agree on the fix to implement

---

## Quick Start
- Step-by-step instructions
- Code examples for common investigation patterns
- Tips for running tests and gathering data
- How to revert your changes

---

## Investigation Resources

### Code Files
- **`cuda/src/transformer/qwen_transformer.cpp`** - The cuBLAS call (lines 275-293)
- **`src/cuda/weight_loader.rs`** - How lm_head is loaded from GGUF
- **`src/cuda/model.rs`** - Tensor dimension logging
- **`reference/llama.cpp/`** - Working reference implementation

### Documentation Files
- **`INVESTIGATION_INDEX.md`** - Master investigation timeline
- **`COMPLETE_INVESTIGATION_REPORT.md`** - Detailed findings from previous investigation
- **`LLAMA_CPP_MATRIX_ANALYSIS.md`** - Analysis of llama.cpp (may be incomplete)
- **`DEBUG_ATTEMPT_2025-10-06_CASCADE.md`** - Recent failed attempts

### Test Command
```bash
cd bin/worker-orcd
cargo test --release --test haiku_generation_anti_cheat \
  test_haiku_generation_stub_pipeline_only \
  --features cuda -- --ignored --nocapture --test-threads=1
```

---

## Success Criteria

A successful investigation will:

✅ Identify the **exact** root cause with evidence  
✅ Explain **why** only specific positions fail  
✅ Propose a **specific** fix (e.g., "change parameter X to Y")  
✅ Provide **mathematical** justification for the fix  
✅ Have **consensus** across multiple teams  

---

## Expected Outcomes

### Likely Root Cause
Based on previous investigations, the issue is almost certainly related to:
1. Row-major (GGUF) vs column-major (cuBLAS) memory layout mismatch
2. Incorrect transpose flags in cuBLAS call
3. Incorrect leading dimension parameters

### Possible Fixes
1. Change `CUBLAS_OP_N` to `CUBLAS_OP_T` for lm_head (with correct lda)
2. Explicitly transpose lm_head in GPU memory before cuBLAS call
3. Adjust leading dimension calculations

**WARNING**: Previous attempts to change transpose flags failed catastrophically. The fix must be carefully derived, not guessed.

---

## Communication

   ### Adding Comments
   Use your team prefix consistently:
   ```cpp
   // [TEAM_ALPHA] This is Team Alpha's analysis
   // [TEAM_BRAVO] This is Team Bravo's finding
   // [TEAM_CHARLIE] Manual computation shows...
   ```
   
   ### Avoiding Conflicts
   If you need to add comments to the same code section:
   - Add your comment on a new line
   - Don't modify other teams' comments, except to append a correction for a false FIXED claim (see "False Claim Correction").
   - When correcting, do not delete or rewrite their text — append your own line with timestamped evidence.
   - Stack comments vertically, don't nest
   
   Example:
   ```cpp
   // [TEAM_ALPHA] Memory layout: row-major [896, 151936]
   // [TEAM_ECHO] cuBLAS expects column-major, causing mismatch
   // [TEAM_CHARLIE] Confirmed by manual computation
   // ❌ Previous team claimed FIXED — but haiku test still fails here. Evidence: 2025-10-06 18:44 UTC run repeats token 64362.
   cublasGemmEx(...);
   ```
**Results Writing**: 30-60 minutes per team  
**Team Sync**: 1 hour all teams together  
**Implementation**: After consensus reached  

---

## Questions?

Refer back to:
1. Your team's investigation brief (`TEAM_*_*.md`)
2. Previous investigation docs (see Investigation Resources above)
3. The cuBLAS documentation (Team Echo's specialty)

---

**Remember**: Your goal is to **understand** the problem deeply, not to quickly fix it. A well-understood problem is half solved.

Good luck, teams! 🔍
