# Team Delta: Instrumentation & Profiling

**Team Mission**: Add comprehensive logging to trace the exact data flow and identify anomalies

**Team Expertise**: Debugging, profiling, instrumentation, data analysis

**ADD AS MUCH LOGGING AS YOU NEED** - Extract all the data! Run tests! Just revert changes after!

---

## Your Investigation Strategy

You are the instrumentation specialists. Your job is to add **comprehensive logging** to understand exactly what's happening at runtime.

### Phase 1: Log cuBLAS Parameters

Add detailed logging before the cuBLAS call in `qwen_transformer.cpp`:

```cpp
// [TEAM_DELTA] cuBLAS GEMM parameters
fprintf(stderr, "\n[TEAM_DELTA] cuBLAS GEMM Call:\n");
fprintf(stderr, "  op_A=CUBLAS_OP_N, op_B=CUBLAS_OP_N\n");
fprintf(stderr, "  m=%d, n=%d, k=%d\n", config_.vocab_size, batch_size, config_.hidden_dim);
fprintf(stderr, "  lda=%d, ldb=%d, ldc=%d\n", config_.vocab_size, config_.hidden_dim, config_.vocab_size);
fprintf(stderr, "  A ptr=%p, B ptr=%p, C ptr=%p\n", lm_head_half, hidden_half, logits);
```

### Phase 2: Sample Memory at Key Positions

Log memory contents at positions 0, 8850, 44394, 137131:

```cpp
// [TEAM_DELTA] Sample lm_head memory
if (first_call) {
    half h_samples[40];  // 10 elements from 4 positions
    
    // Position 0
    cudaMemcpy(h_samples, lm_head_half, 10*sizeof(half), cudaMemcpyDeviceToHost);
    fprintf(stderr, "[TEAM_DELTA] lm_head[0][0:10]: ");
    for (int i = 0; i < 10; i++) fprintf(stderr, "%.4f ", __half2float(h_samples[i]));
    fprintf(stderr, "\n");
    
    // Position 8850 (garbage position)
    cudaMemcpy(h_samples+10, lm_head_half + 8850*151936, 10*sizeof(half), cudaMemcpyDeviceToHost);
    fprintf(stderr, "[TEAM_DELTA] lm_head[8850][0:10]: ");
    for (int i = 10; i < 20; i++) fprintf(stderr, "%.4f ", __half2float(h_samples[i]));
    fprintf(stderr, "\n");
}
```

### Phase 3: Log Output Logits Pattern

```cpp
// [TEAM_DELTA] Sample output logits
float h_logits[10];
int positions[] = {0, 100, 8850, 10000, 44394, 50000, 137131, 140000, 150000, 151935};
for (int i = 0; i < 10; i++) {
    cudaMemcpy(&h_logits[i], logits + positions[i], sizeof(float), cudaMemcpyDeviceToHost);
    fprintf(stderr, "[TEAM_DELTA] logits[%6d] = %10.4f%s\n", 
            positions[i], h_logits[i], 
            (fabs(h_logits[i]) > 10.0) ? " ⚠️" : "");
}
```

### Phase 4: Track Changes Over Time

```cpp
// [TEAM_DELTA] Track positions over multiple calls
static int call_num = 0;
if (call_num < 15) {
    float val;
    cudaMemcpy(&val, logits + 44394, sizeof(float), cudaMemcpyDeviceToHost);
    fprintf(stderr, "[TEAM_DELTA] Call #%d: logits[44394] = %.4f\n", call_num, val);
}
call_num++;
```

---

## Deliverable

Create: `investigation-teams/TEAM_DELTA_RESULTS.md`

Include:
1. **Log Analysis** - Patterns found in the logs
2. **Memory Access Patterns** - What addresses are being accessed
3. **Anomaly Detection** - Which positions consistently fail
4. **Time-series Data** - How values change across calls
5. **Proposed Fix** - Based on observed patterns
