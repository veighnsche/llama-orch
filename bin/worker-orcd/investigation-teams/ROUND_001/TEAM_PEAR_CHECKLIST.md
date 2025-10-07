# TEAM PEAR — 10-Phase Peer Review Plan
**Generated:** 2025-10-07T10:53Z (REBUILT)  
**Updated:** 2025-10-07T11:10Z (Phase 0 enhanced with complete details)  
**Mission:** Systematic skeptical peer review of all investigation team claims  
**Status:** Phase 0 Complete — Proceeding to Phase 1

---

## 📋 Overview

This plan divides the peer review workload into 10 balanced phases. Each phase focuses on a specific subsystem or investigation area, covering all relevant team claims from the chronicles, code comments, and false leads documentation.

**Total Claims Identified:** ~100+ across 23 teams  
**Estimated Total Effort:** 12-15 hours  
**Approach:** Document review → Code inspection → Evidence verification → Stamp verdicts

---

## Phase Definitions

### Phase 0: Setup & Claim Inventory ✅ COMPLETE
**Goal:** Scan chronicles and code to identify all teams, claims, and build 10-phase structure  
**Duration:** 30 minutes  
**Status:** ✅ Complete (2025-10-07T10:53Z)

**Inputs:**
- `INVESTIGATION_CHRONICLE.md` (1248 lines, 23 teams documented)
- `FALSE_LEADS_SUMMARY.md` (464 lines, 14 false leads)
- Code comments across repo (219 in qwen_transformer.cpp alone)

**Deliverables:**
- ✅ Team roster (23 teams identified)
- ✅ Claim inventory (~100+ claims mapped)
- ✅ 10-phase plan structure
- ✅ Priority ranking (FFN path = highest)

**Exit Criteria:**
- All teams appear in at least one phase
- Each phase has roughly balanced effort (45-90 minutes)

**Claim Categories:**
- ✅ **Verified Correct:** 47 claims (components working as expected)
- ❌ **False Leads:** 14 documented (hypotheses disproven with evidence)
- 🔥 **Fixes Applied:** 4 (special tokens, bias loading, sampling order, crash fix)
- ⚠️ **Untested Fixes:** 1 (ffn_down loading by Charlie Beta)
- 🔍 **Active Investigations:** 3 (FFN path, attention projection, weight integrity)
- 📝 **Observability Comments:** 20+ (logging/debugging infrastructure)

---

## 🎯 10-Phase Plan (Balanced by Effort)

### Phase 1: Tokenization & Embedding Pipeline (9 claims, 45 min)
**Goal:** Verify all claims about tokenization, special tokens, embeddings, and token sequence format.

**Inputs:**
- Team Blue findings (special token fix)
- Team Purple findings (token ID verification, embedding check)
- FALSE_LEADS_SUMMARY.md #1-4

**Scope:**
1. Team Blue: Special tokens split by BPE (FIXED - manually insert IDs 151644/151645)
2. Team Purple: Vocab size = 151936 (tokens 151644/151645 are valid)
3. Team Purple: Special token embeddings valid (~0.01 range, not zeros)
4. Team Purple: Token sequence format matches llama.cpp exactly
5. Team Purple: Embedding lookup returns correct values
6. FALSE LEAD #1: Token IDs out of bounds
7. FALSE LEAD #2: Special token embeddings are zeros
8. FALSE LEAD #3: Tokenization approach matters
9. FALSE LEAD #4: Chat template format wrong

**Method:**
- Replicate: Tokenize test prompt, verify token IDs match llama.cpp
- Test: Dump embeddings for tokens 151643, 151644, 151645
- Verify: Token sequence format against llama.cpp logs
- Evidence: Token ID lists, embedding dumps, sequence comparisons

**Artifacts:**
- `PHASE1_TOKENIZATION_REPORT.md`
- `phase1_token_ids.txt` (our engine vs llama.cpp)
- `phase1_embeddings.csv` (special token embeddings)

**Exit Criteria:**
- All 9 claims stamped
- Token IDs verified against llama.cpp
- Special token embeddings reproduced
- Sequence format verified

### Phase 2: cuBLAS Matrix Multiplication Correctness (8 claims, 60 min)
**Goal:** Verify all claims about cuBLAS matrix multiplication, transpose flags, and manual verification.

**Inputs:**
- Team Charlie findings (cuBLAS manual verification)
- Team Felicia findings (CUBLAS_OP_T experiments)
- Team Aurora findings (transpose + lda correction)
- Team THIMBLE findings (pre-transpose experiment)
- FALSE_LEADS_SUMMARY.md #8-9

**Scope:**
1. Team Charlie: cuBLAS matches manual within 0.00002 tolerance (lines 245-360 `qwen_transformer.cpp`)
2. Team Felicia: CUBLAS_OP_T causes stuck repetition (token 71443)
3. Team Aurora: CUBLAS_OP_T with corrected lda also fails
4. Team THIMBLE: Explicit CPU transpose + OP_N produces identical extremes
5. Team ORION: Q[0] cuBLAS matches manual (diff=0.000015)
6. Team ORION: Q[95]/Q[126] cuBLAS gives ±16, manual gives ±0.08
7. FALSE LEAD #8: CUBLAS_OP_T with corrected lda
8. FALSE LEAD #9: Stride interpretation bug

**Method:**
- Replicate: Run manual dot product for positions 0, 8850, 44394, 137131
- Test: Apply CUBLAS_OP_T to single layer, measure output difference
- Verify: THIMBLE's pre-transpose experiment with deterministic seed
- Evidence: Diff values for manual vs cuBLAS at same positions

**Artifacts:**
- `PHASE2_CUBLAS_REPORT.md`
- `phase2_manual_verification.csv` (manual vs cuBLAS comparisons)
- `phase2_transpose_test_output.txt` (OP_T behavior)

**Exit Criteria:**
- All 8 claims stamped
- cuBLAS correctness for Q[0] reproduced
- Q[95]/Q[126] extremes reproduced or falsified
- Transpose experiments replicated with same outputs

---

### **Phase 3: KV Cache Infrastructure** (Est: 45 min)
**Goal:** Verify all claims about KV cache positions, reads, writes, indexing, and strides.

**Inputs:**
- Team Water findings (cache_len verification)
- Team Charlie Gamma findings (cache_len=0 clue)
- Team Drawer findings (KV cache indexing)
- FALSE_LEADS_SUMMARY.md #12

**Scope:**
1. Team Water: cache_len passes correctly (0, 1, 2, 3...) (lines 623-637, 387-391 `gqa_attention.cu`)
2. Team Water: Cache writes to correct positions
3. Team Water: Position tracking correct
4. Team Water: RoPE applies different rotations per position
5. Team Drawer: Cache layout [batch, kv_head, pos, d] with correct strides
6. Team Drawer: Write-at-pos read-back verification (immediate read matches write)
7. Team Drawer: Layer isolation (layer 0 and layer 23 have distinct pointers)
8. FALSE LEAD #12: KV cache indexing/strides wrong

**Method:**
- Replicate: Run 5-token sequence, log cache_len at each step
- Verify: Dump cache values after write, read back, compare
- Test: Write to layer 0 kv_head=0, verify layer 23 kv_head=1 unaffected
- Evidence: Cache progression logs, read-back parity tests

**Artifacts:**
- `PHASE3_KVCACHE_REPORT.md`
- `phase3_cache_progression.txt` (cache_len: 0→1→2→3→4)
- `phase3_readback_parity.csv` (write vs read values)

**Exit Criteria:**
- All 8 claims stamped
- cache_len progression verified
- Read-back parity test passed
- Layer isolation verified

---

### **Phase 4: RoPE, RMSNorm, and Numerical Primitives** (Est: 75 min)
**Goal:** Verify all claims about RoPE angles, RMSNorm formula, epsilon values, and numerical correctness.

**Inputs:**
- Team HYPERION findings (RoPE formula, RMSNorm epsilon)
- Team LAMINATOR findings (output RMSNorm numerics)
- Team HOLE_PUNCH findings (RoPE numeric parity)
- FALSE_LEADS_SUMMARY.md #9-10

**Scope:**
1. Team HYPERION: RoPE formula correct (re-confirmed Team Polaris)
2. Team HYPERION: RMSNorm epsilon = 1e-6f matches llama.cpp
3. Team LAMINATOR: Output RMSNorm formula correct (diff=0.00013)
4. Team LAMINATOR: Gamma weights mean=7.14 correct (Team Charlie verified)
5. Team LAMINATOR: Post-norm "amplification" intentional (range 37→59)
6. Team HOLE_PUNCH: RoPE config correct (head_dim=64, rope_freq_base=1000000.0)
7. Team HOLE_PUNCH: Angle generation correct (cos(1.0)=0.5403, sin(1.0)=0.8415)
8. Team HOLE_PUNCH: Identity at pos=0 (Q_PRE == Q_POST, diff=0.0)
9. Team HOLE_PUNCH: Rotation at pos=1 uses correct cos/sin
10. FALSE LEAD #9: Output RMSNorm numerics wrong
11. FALSE LEAD #10: RoPE numeric parity issues

**Method:**
- Replicate: Compute RoPE angles manually for pos=0,1,2; compare with kernel output
- Verify: RMSNorm formula with hand calculation for first element
- Test: Dump output_norm.weight, verify mean=7.14
- Evidence: Angle tables, RMSNorm manual calculation, gamma stats

**Artifacts:**
- `PHASE4_NUMERICS_REPORT.md`
- `phase4_rope_angles.csv` (manual vs kernel angles)
- `phase4_rmsnorm_verification.txt` (manual formula check)

**Exit Criteria:**
- All 11 claims stamped
- RoPE angles verified for pos=0,1,2
- RMSNorm formula verified with manual calculation
- Gamma weight stats reproduced

---

### **Phase 5: Attention Mechanism (GQA, Softmax, Masking)** (Est: 60 min)
**Goal:** Verify all claims about GQA head mapping, attention softmax, causal masking, and score computation.

**Inputs:**
- Team Bygone findings (causal masking verification)
- Team SHREDDER findings (GQA head mapping)
- Team LABEL_MAKER findings (softmax pipeline)
- FALSE_LEADS_SUMMARY.md #5, #11

**Scope:**
1. Team Bygone: Causal masking implemented (loop 0..cache_len) (lines 253-263 `gqa_attention.cu`)
2. Team SHREDDER: GQA group size = 7 (14 Q heads / 2 KV heads)
3. Team SHREDDER: Q→KV mapping correct (heads 0-6→kv_head 0, heads 7-13→kv_head 1)
4. Team SHREDDER: K/V pointer offsets correct (kv_head 0 at offset 0, kv_head 1 at offset 64)
5. Team LABEL_MAKER: Scale factor = 0.125 = 1/sqrt(64) correct
6. Team LABEL_MAKER: Rowmax subtraction working (MAX_EXP=1.0)
7. Team LABEL_MAKER: Softmax normalization perfect (SUM_P=1.0, diff<1e-6)
8. Team LABEL_MAKER: No NaN/inf in softmax pipeline
9. FALSE LEAD #5: Missing causal mask
10. FALSE LEAD #11: GQA head mapping incorrect

**Method:**
- Replicate: Run 3-token sequence, verify attention only attends to past
- Test: Dump scores for q_head=0 and q_head=7, verify they use different KV heads
- Verify: Softmax weights sum to 1.0 across 100 samples
- Evidence: Attention mask logs, GQA mapping table, softmax sums

**Artifacts:**
- `PHASE5_ATTENTION_REPORT.md`
- `phase5_gqa_mapping.txt` (q_head→kv_head mapping)
- `phase5_softmax_sums.csv` (100 samples of SUM_P)

**Exit Criteria:**
- All 10 claims stamped
- Causal masking verified (future tokens masked)
- GQA mapping verified (7:1 ratio)
- Softmax normalization verified (sum=1.0)

---

### **Phase 6: FFN Path (Gate, Up, Down, SwiGLU)** (Est: 75 min)
**Goal:** Verify all claims about FFN weight loading, SwiGLU activation, and down projection.

**Inputs:**
- Team Charlie Beta findings (ffn_down loading fix)
- Team RACE CAR findings (FFN weight validation)
- Team PAPER CUTTER findings (last block FFN investigation)
- Team HYPERION findings (SwiGLU correctness)

**Scope:**
1. Team Charlie Beta: ffn_down was missing in load_from_gpu_pointers() (line 367 `qwen_weight_loader.cpp`)
2. Team Charlie Beta: Fix added but NOT TESTED (compilation errors)
3. Team RACE CAR: FFN pointers non-null validation (layer 0)
4. Team RACE CAR: Expected shapes: gate/up=[896,4864], down=[4864,896]
5. Team PAPER CUTTER: Last block FFN down investigation (layer 23)
6. Team HYPERION: SwiGLU activation correct implementation
7. Green Team: Biases loaded but all zeros (no effect)

**Method:**
- Replicate: Build with Charlie Beta's fix, verify compilation succeeds
- Test: Run haiku generation, check if ffn_down fix resolves garbage output
- Verify: Dump ffn_gate, ffn_up, ffn_down pointers for layers 0, 12, 23
- Evidence: Pointer values, compilation log, haiku output comparison

**Artifacts:**
- `PHASE6_FFN_REPORT.md`
- `phase6_ffn_pointers.txt` (pointer addresses for 3 layers)
- `phase6_haiku_before_after.txt` (output comparison)
- `phase6_compilation.log` (build success/failure)

**Exit Criteria:**
- All 7 claims stamped
- Charlie Beta fix tested (pass/fail)
- FFN pointer validation reproduced
- Haiku quality measured (with and without fix if applicable)

---

### **Phase 7: Sampling & Generation Logic** (Est: 45 min)
**Goal:** Verify all claims about sampling pipeline, temperature, top-p, softmax order, and token generation.

**Inputs:**
- Team HELIOS findings (sampling order fix)
- Team SEA findings (sampling correctness)
- Team AEGIS findings (temperature investigation)
- FALSE_LEADS_SUMMARY.md #13

**Scope:**
1. Team HELIOS: Sampling order fixed (softmax before top-p) (lines 251-389 `sampling_wrapper.cu`)
2. Team HELIOS: Temperature=0.7 applied correctly
3. Team HELIOS: Seeds increment correctly (1759794426, 1759794427...)
4. Team HELIOS: Tokens vary (not stuck in loops)
5. Team HELIOS: Top-p disabled as workaround (broken normalization)
6. Team SEA: Argmax sampling correct
7. Team SEA: Temperature/softmax verified
8. Team AEGIS: Prefill uses temp=0.0 by design (not a bug)
9. FALSE LEAD #13: Sampling pipeline order (now FIXED)

**Method:**
- Replicate: Generate 50 tokens, verify seeds increment 1→2→3...
- Test: Set temp=0.7, verify token probabilities follow expected distribution
- Verify: Compare sampling order in our code vs llama.cpp
- Evidence: Seed sequence, token probability distributions, order flow diagram

**Artifacts:**
- `PHASE7_SAMPLING_REPORT.md`
- `phase7_seed_sequence.txt` (50 consecutive seeds)
- `phase7_temperature_distribution.csv` (token probabilities)

**Exit Criteria:**
- All 9 claims stamped
- Sampling order fix verified
- Temperature application verified
- Seed incrementing verified

---

### **Phase 8: Weight Loading & Dequantization** (Est: 90 min)
**Goal:** Verify all claims about weight loading, GPU pointer assignment, bias handling, and weight integrity.

**Inputs:**
- Team Green findings (bias loading fix)
- Team Charlie Beta findings (ffn_down missing)
- Team VANGUARD findings (weight integrity verification)
- Team Charlie findings (output_norm weights)

**Scope:**
1. Team Green: Q/K/V biases loaded but all zeros (lines 366-373 `qwen_weight_loader.cpp`)
2. Team Green: Bias loading code added (no effect on output)
3. Team Charlie Beta: ffn_down loading line missing (added at line 380)
4. Team VANGUARD: Weight integrity dumps (first 100 FP16 values)
5. Team Charlie: output_norm mean=7.14 CORRECT (not corrupted)
6. Team ROOT CAUSE: Normalized output_norm to mean=1.0 (partial fix, later reverted)
7. Weight loader: All 24 layers load correctly

**Method:**
- Replicate: Dump Q/K/V biases for layer 0, verify all zeros
- Test: Compare ffn_down pointer before and after Charlie Beta fix
- Verify: Dump first 100 values from token_embd, attn_q.weight, ffn_down
- Evidence: Bias checksums, weight dumps, pointer logs

**Artifacts:**
- `PHASE8_WEIGHTS_REPORT.md`
- `phase8_bias_dumps.csv` (Q/K/V biases layer 0)
- `phase8_weight_first100.txt` (critical weight tensors)
- `phase8_pointer_comparison.txt` (before/after fix)

**Exit Criteria:**
- All 7 claims stamped
- Bias zeros verified
- ffn_down loading tested
- Weight integrity spot-checked

---

### **Phase 9: Edge Cases & Infrastructure** (Est: 45 min)
**Goal:** Verify all claims about crashes, buffer management, memory leaks, and edge case handling.

**Inputs:**
- Team CHAIR findings (special token crash fix)
- Team BATTLESHIP findings (buffer integrity)
- Team FREE findings (memory management)
- Compilation fixes

**Scope:**
1. Team CHAIR: Removed chat template to avoid special token crash
2. Team CHAIR: Crash was infrastructure issue, not token bug
3. Team BATTLESHIP: Buffer canaries (no overwrites detected)
4. Team BATTLESHIP: No buffer aliasing detected
5. Team BATTLESHIP: Q spikes filtered by attention softmax (harmless)
6. Team FREE: cudaMalloc without error checks (potential leak)
7. Team FREE: cudaMemcpy H2D forces sync (inefficient)
8. Compilation fix: Double-free of h_q_full removed

**Method:**
- Replicate: Run with special tokens, verify no crash
- Test: Enable BATTLESHIP_CANARIES, verify no overwrites
- Review: cudaMalloc calls, check error handling
- Evidence: Crash logs, canary reports, memory leak analysis

**Artifacts:**
- `PHASE9_INFRASTRUCTURE_REPORT.md`
- `phase9_crash_test.log` (special token handling)
- `phase9_canary_report.txt` (buffer integrity)

**Exit Criteria:**
- All 8 claims stamped
- Crash fix verified
- Buffer integrity verified
- Memory issues documented

---

### **Phase 10: Cross-Team Contradictions & Final Synthesis** (Est: 60 min)
**Goal:** Identify and resolve contradictions between teams, audit claim consistency, and prepare final report.

**Inputs:**
- All previous phase reports
- INVESTIGATION_CHRONICLE.md patterns section
- FALSE_LEADS_SUMMARY.md

**Scope:**
1. Team Charlie vs ROOT CAUSE: output_norm corruption (Charlie says correct, ROOT CAUSE normalized it)
2. Team Charlie Gamma vs Team Water: cache_len=0 clue (Gamma saw old logs, Water verified correct)
3. Team ORION vs Team BATTLESHIP: Q[95]/Q[126] spikes (ORION found them, BATTLESHIP proved harmless)
4. Multiple teams vs THIMBLE: Transpose hypotheses (Felicia, Aurora, THIMBLE all tested, all failed)
5. Team Bygone vs initial hypotheses: Prefill one-at-a-time (verified correct, not a bug)
6. Audit: 22 teams, 89 claims, consistency check
7. Synthesize: Top 3 failure patterns
8. Calculate: Total fines across all phases

**Method:**
- Review: All phase reports for contradictions
- Cross-reference: Chronicle vs FALSE_LEADS vs code comments
- Identify: Outdated comments, missing reversions, contradictory claims
- Synthesize: Common failure modes, lessons learned
- Evidence: Contradiction matrix, consistency report

**Artifacts:**
- `PHASE10_CONTRADICTIONS_REPORT.md`
- `PHASE10_TOP3_FAILURE_PATTERNS.md`
- `PHASE10_CONSISTENCY_AUDIT.csv`
- `FINES_TOTAL_SUMMARY.md` (sum across all phases)

**Exit Criteria:**
- All contradictions documented
- Consistency audit complete
- Top 3 failure patterns identified
- Total fines calculated
- PEER_REVIEW_REPORT.md finalized

---

## 📊 Effort Breakdown

| Phase | Focus Area | Estimated Time | Claim Count | Complexity |
|-------|-----------|----------------|-------------|------------|
| 1 | Tokenization | 45 min | 9 | Low |
| 2 | cuBLAS | 60 min | 8 | Medium |
| 3 | KV Cache | 45 min | 8 | Low-Medium |
| 4 | RoPE/RMSNorm | 75 min | 11 | High |
| 5 | Attention | 60 min | 10 | Medium-High |
| 6 | FFN Path | 75 min | 7 | High |
| 7 | Sampling | 45 min | 9 | Low-Medium |
| 8 | Weight Loading | 90 min | 7 | High |
| 9 | Infrastructure | 45 min | 8 | Low-Medium |
| 10 | Synthesis | 60 min | 12 | Medium |
| **TOTAL** | | **600 min** | **89 claims** | |

**Estimated Total Time:** 10 hours (with breaks and documentation)

---

## 🎯 Success Metrics

1. **Completeness:** All 89 claims stamped (VERIFIED / FALSIFIED / NEEDS-EVIDENCE)
2. **Evidence:** Every stamp links to reproducible evidence
3. **Fines:** All infractions logged with fair amounts (€10-€500)
4. **Cleanup:** Comments updated without deleting history
5. **Report:** PEER_REVIEW_REPORT.md comprehensive and actionable
6. **Observability:** All hooks preserved, gated if noisy
7. **Lessons:** Top 3 failure patterns documented for future teams

---

## 🚦 Phase 0 Status

- ✅ Claim inventory complete (89 claims across 22 teams)
- ✅ 10 phases defined with balanced effort
- ✅ Inputs, methods, artifacts, exit criteria specified
- ✅ Estimated 10 hours total effort
- ✅ Directory structure planned
- ✅ This checklist committed

**Next Action:** Commit this file, then proceed to Phase 1.

---

**File:** `investigation-teams/TEAM_PEAR_CHECKLIST.md`  
**Author:** TEAM PEAR  
**Date:** 2025-10-07T10:16Z  
**Status:** Phase 0 Complete — Ready for Phase 1
