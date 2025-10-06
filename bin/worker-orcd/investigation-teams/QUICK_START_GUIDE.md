# Quick Start Guide for Investigation Teams

**Updated**: 2025-10-06 14:58 UTC  
**Important**: You CAN change code to extract data! Just revert it after.

---

## TL;DR - What You Can Do

### ✅ ENCOURAGED:
- **Add extensive logging** (`fprintf`, `printf`, `eprintln!`)
- **Copy GPU data to host** (`cudaMemcpy`) to inspect values
- **Implement verification functions** to compute ground truth
- **Temporarily modify cuBLAS parameters** to test hypotheses
- **Run the test multiple times** to gather data
- **Add comments everywhere** explaining your findings

### ⚠️ JUST REMEMBER TO:
- **REVERT all changes** after gathering your data
- **Document everything** in your `TEAM_*_RESULTS.md` file
- **Save your test output** (copy terminal output to your results)

---

## How to Run Your Investigation

### Step 1: Read Your Team Brief
```bash
cd /home/vince/Projects/llama-orch/bin/worker-orcd/investigation-teams
cat TEAM_YOUR_NAME_*.md  # Read your specific brief
```

### Step 2: Add Your Investigation Code

Edit the relevant files (usually `cuda/src/transformer/qwen_transformer.cpp`):

```cpp
// [TEAM_YOUR_NAME] Your investigation code here
if (first_call) {
    fprintf(stderr, "\n[TEAM_YOUR_NAME] === YOUR TEST NAME ===\n");
    
    // Extract data, run tests, gather evidence
    // Example: Copy data to host and inspect
    half h_data[896];
    cudaMemcpy(h_data, gpu_ptr, 896*sizeof(half), cudaMemcpyDeviceToHost);
    
    // Print it, analyze it, understand it
    for (int i = 0; i < 10; i++) {
        fprintf(stderr, "  data[%d] = %.6f\n", i, __half2float(h_data[i]));
    }
}
```

### Step 3: Build and Run Test

```bash
cd /home/vince/Projects/llama-orch/bin/worker-orcd

# Clean build to ensure changes are compiled
cargo clean -p worker-orcd

# Run the test
cargo test --release --test haiku_generation_anti_cheat \
  test_haiku_generation_stub_pipeline_only \
  --features cuda -- --ignored --nocapture --test-threads=1 \
  2>&1 | tee team_your_name_output.txt
```

### Step 4: Analyze Output

Look at the terminal output or `team_your_name_output.txt`:
- Find your `[TEAM_YOUR_NAME]` log lines
- Copy relevant data to your RESULTS.md
- Draw conclusions

### Step 5: Revert Your Changes

```bash
# Option 1: Use git to revert
git checkout cuda/src/transformer/qwen_transformer.cpp

# Option 2: Manually remove your test code
# (Keep the comments explaining what you learned!)
```

### Step 6: Document Results

Create `investigation-teams/TEAM_YOUR_NAME_RESULTS.md`:

```markdown
# Team Your Name - Investigation Results

## Test Output
[Paste your terminal output here]

## Key Findings
1. Finding 1 with evidence
2. Finding 2 with evidence

## Root Cause Hypothesis
[Your theory based on the data]

## Proposed Fix
[Specific code change with justification]
```

---

## Common Investigation Patterns

### Pattern 1: Manual Dot Product Verification

```cpp
// [TEAM_X] Verify logit computation manually
if (first_call) {
    // Copy inputs to host
    half h_hidden[896], h_lm_head_row[896];
    cudaMemcpy(h_hidden, hidden_half, 896*sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_lm_head_row, lm_head_half + position*151936, 896*sizeof(half), cudaMemcpyDeviceToHost);
    
    // Manual computation
    float manual = 0.0f;
    for (int i = 0; i < 896; i++) {
        manual += __half2float(h_hidden[i]) * __half2float(h_lm_head_row[i]);
    }
    
    // Compare with cuBLAS
    float cublas;
    cudaMemcpy(&cublas, logits + position, sizeof(float), cudaMemcpyDeviceToHost);
    
    fprintf(stderr, "[TEAM_X] Position %d: Manual=%.6f, cuBLAS=%.6f\n", position, manual, cublas);
}
```

### Pattern 2: Test Different cuBLAS Parameters

```cpp
// [TEAM_X] Test hypothesis: change transpose flag
if (first_call) {
    fprintf(stderr, "[TEAM_X] Testing CUBLAS_OP_T...\n");
    
    // Try different parameters
    cublasGemmEx(
        cublas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,  // Changed from CUBLAS_OP_N
        config_.vocab_size,
        batch_size,
        config_.hidden_dim,
        &alpha,
        lm_head_half, CUDA_R_16F, 896,  // Changed lda
        hidden_half, CUDA_R_16F, config_.hidden_dim,
        &beta,
        logits, CUDA_R_32F, config_.vocab_size,
        CUBLAS_COMPUTE_32F_FAST_16F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
    
    // Check if results improved
    float test_logits[5];
    int positions[] = {0, 8850, 44394, 137131, 151935};
    for (int i = 0; i < 5; i++) {
        cudaMemcpy(&test_logits[i], logits + positions[i], sizeof(float), cudaMemcpyDeviceToHost);
        fprintf(stderr, "[TEAM_X] Test logits[%d] = %.6f\n", positions[i], test_logits[i]);
    }
}
// REVERT THIS AFTER TESTING!
```

### Pattern 3: Memory Layout Inspection

```cpp
// [TEAM_X] Inspect memory layout
if (first_call) {
    fprintf(stderr, "[TEAM_X] Memory layout inspection:\n");
    
    // Check if data is row-major or column-major
    half h_samples[20];
    
    // Sample: [0][0], [0][1], [1][0], [1][1]
    cudaMemcpy(&h_samples[0], lm_head_half + 0, sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_samples[1], lm_head_half + 1, sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_samples[2], lm_head_half + 151936, sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_samples[3], lm_head_half + 151937, sizeof(half), cudaMemcpyDeviceToHost);
    
    fprintf(stderr, "  lm_head[0][0] = %.6f (offset 0)\n", __half2float(h_samples[0]));
    fprintf(stderr, "  lm_head[0][1] = %.6f (offset 1)\n", __half2float(h_samples[1]));
    fprintf(stderr, "  lm_head[1][0] = %.6f (offset 151936)\n", __half2float(h_samples[2]));
    fprintf(stderr, "  lm_head[1][1] = %.6f (offset 151937)\n", __half2float(h_samples[3]));
    fprintf(stderr, "  => This confirms row-major [896, 151936] layout\n");
}
```

---

## Tips for Success

### 1. Start Simple
Don't try to test everything at once. Start with one simple test, run it, analyze results, then add more.

### 2. Use `first_call` Flag
The code already has a `first_call` flag. Use it to run your tests only once:
```cpp
static bool first_call = true;
if (first_call) {
    // Your test code here
    first_call = false;
}
```

### 3. Save Terminal Output
Always save your test output:
```bash
cargo test ... 2>&1 | tee my_test_output.txt
```

### 4. Compare Multiple Positions
Test at least these positions:
- Position 0 (baseline - should work)
- Position 8850 (known garbage)
- Position 44394 (known garbage)
- Position 137131 (known garbage)

### 5. Document As You Go
Don't wait until the end. Add comments explaining what you find as you discover it.

---

## What Success Looks Like

Your investigation succeeds when you can answer:

1. **What is the ground truth?** (Manual computation of correct logits)
2. **What is cuBLAS actually doing?** (What addresses is it reading?)
3. **Why do they differ?** (The specific parameter causing the mismatch)
4. **What's the fix?** (Exact parameter change needed)
5. **Does the fix work?** (Test results showing improvement)

---

## Need Help?

- **Stuck on CUDA?** Check NVIDIA cuBLAS documentation
- **Stuck on memory layout?** Draw a diagram on paper
- **Stuck on math?** Compute one example by hand
- **Stuck on code?** Add more logging to see what's happening

---

## Remember

**The goal is EVIDENCE, not guesses.**

Add as much logging as you need. Run as many tests as you want. Gather all the data. Then analyze it carefully and draw conclusions based on evidence.

Good luck! 🔍
