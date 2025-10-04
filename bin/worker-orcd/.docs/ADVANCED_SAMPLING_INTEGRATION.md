# Advanced Sampling Integration Guide

**Version**: Sprint 4 Complete  
**Status**: ✅ Production Ready  
**Last Updated**: 2025-10-04

---

## Overview

This document describes the complete integration of advanced sampling parameters throughout the worker-orcd codebase, from HTTP API to CUDA kernels.

---

## Architecture

### Layer 1: HTTP API (`src/http/`)

**Entry Point**: `POST /execute`

**Request Schema** (`validation.rs`):
```rust
pub struct ExecuteRequest {
    // Core parameters (Sprint 3)
    pub job_id: String,
    pub prompt: String,
    pub max_tokens: u32,
    pub temperature: f32,
    pub seed: Option<u64>,
    
    // Advanced parameters (Sprint 4)
    pub top_p: f32,              // Default: 1.0 (disabled)
    pub top_k: u32,              // Default: 0 (disabled)
    pub repetition_penalty: f32, // Default: 1.0 (disabled)
    pub stop: Vec<String>,       // Default: [] (none)
    pub min_p: f32,              // Default: 0.0 (disabled)
}
```

**Validation Rules**:
- `top_p`: 0.0-1.0 (inclusive)
- `top_k`: 0-u32::MAX (no upper limit)
- `repetition_penalty`: 0.0-2.0 (inclusive)
- `stop`: Max 4 sequences, each max 100 chars, non-empty
- `min_p`: 0.0-1.0 (inclusive)

**Response Schema** (`sse.rs`):
```rust
pub enum InferenceEvent {
    End {
        tokens_out: u32,
        decode_time_ms: u64,
        stop_reason: StopReason,           // NEW
        stop_sequence_matched: Option<String>, // NEW
    },
    // ... other events
}

pub enum StopReason {
    MaxTokens,      // Reached max_tokens limit
    StopSequence,   // Matched a stop sequence
    Error,          // Inference error
    Cancelled,      // Client cancelled
}
```

---

### Layer 2: Configuration (`src/sampling_config.rs`)

**Purpose**: Bridge HTTP API and CUDA kernels

```rust
pub struct SamplingConfig {
    // All sampling parameters
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub repetition_penalty: f32,
    pub min_p: f32,
    pub stop_sequences: Vec<Vec<u32>>,  // Tokenized
    pub stop_strings: Vec<String>,      // Original strings
    pub seed: u64,
    pub max_tokens: u32,
}
```

**Key Methods**:
- `from_request()`: Convert HTTP request to config
- `has_advanced_sampling()`: Check if advanced params enabled
- `has_stop_sequences()`: Check if stop sequences configured
- `is_greedy()`: Check if greedy mode (temp=0.0)
- `sampling_mode()`: Get human-readable description
- `validate_consistency()`: Check for conflicting parameters

---

### Layer 3: Execution (`src/inference_executor.rs`)

**Purpose**: Coordinate generation loop and stop detection

```rust
pub struct InferenceExecutor {
    config: SamplingConfig,
    tokens: Vec<String>,
    token_ids: Vec<u32>,
    stop_reason: Option<StopReason>,
    stop_sequence_matched: Option<String>,
}
```

**Key Methods**:
- `add_token()`: Add token and check stop conditions
- `check_stop_sequences()`: Pattern matching against generated tokens
- `cancel()`: Mark as cancelled
- `error()`: Mark as error
- `should_stop()`: Check if generation should terminate
- `finalize()`: Build InferenceResult with stop reason

**Generation Loop**:
```rust
let mut executor = InferenceExecutor::new(config);

loop {
    let token_id = sample_token(logits, &executor.config());
    let token_str = detokenize(token_id);
    
    if !executor.add_token(token_str, token_id) {
        break;  // Stop condition met
    }
}

let result = executor.finalize();
// result.stop_reason tells us why we stopped
```

---

### Layer 4: Result (`src/inference_result.rs`)

**Purpose**: Unified result type with stop reason

```rust
pub struct InferenceResult {
    pub tokens: Vec<String>,
    pub token_ids: Vec<u32>,
    pub stop_reason: StopReason,
    pub stop_sequence_matched: Option<String>,
    pub seed: u64,
    pub decode_time_ms: u64,
}
```

**Factory Methods**:
- `max_tokens()`: Create result for max_tokens termination
- `stop_sequence()`: Create result for stop sequence match
- `cancelled()`: Create result for cancellation
- `error()`: Create result for error

**Helper Methods**:
- `token_count()`: Get total tokens generated
- `is_success()`: Check if completed successfully
- `stop_reason_description()`: Human-readable description

---

### Layer 5: CUDA Kernels (`cuda/kernels/sampling.cu`)

**Filtering Kernels**:
```cpp
void launch_top_k(float* logits, int vocab_size, int top_k, cudaStream_t stream);
void launch_top_p(float* logits, int vocab_size, float top_p, cudaStream_t stream);
void launch_repetition_penalty(float* logits, int vocab_size, const int* history, 
                               int history_length, float penalty, cudaStream_t stream);
void launch_min_p(float* logits, int vocab_size, float min_p, cudaStream_t stream);
```

**Stop Sequence Matching**:
```cpp
bool check_stop_sequences(
    const int* generated_tokens,
    int num_generated,
    const int* stop_sequences[4],
    const int stop_lengths[4],
    int num_stop_sequences
);
```

---

## Data Flow

### Request → Response Pipeline

```
1. HTTP Request (JSON)
   ↓
2. ExecuteRequest (validation.rs)
   ↓ validate_all()
3. SamplingConfig (sampling_config.rs)
   ↓ from_request()
4. InferenceExecutor (inference_executor.rs)
   ↓ Generation loop
5. CUDA Kernels (sampling.cu)
   - launch_temperature_scale_fp32()
   - launch_top_k()
   - launch_top_p()
   - launch_repetition_penalty()
   - launch_min_p()
   - launch_stochastic_sample()
   ↓ Each token
6. Stop Detection (inference_executor.rs)
   - check_stop_sequences()
   - max_tokens check
   ↓ Termination
7. InferenceResult (inference_result.rs)
   ↓ finalize()
8. SSE Response (sse.rs)
   - InferenceEvent::End with stop_reason
```

---

## Stop Reason Integration

### Detection Points

**1. Max Tokens** (`InferenceExecutor::add_token()`):
```rust
if self.tokens.len() >= self.config.max_tokens as usize {
    self.stop_reason = Some(StopReason::MaxTokens);
    return false;
}
```

**2. Stop Sequence** (`InferenceExecutor::check_stop_sequences()`):
```rust
if let Some(matched) = self.check_stop_sequences() {
    self.stop_reason = Some(StopReason::StopSequence);
    self.stop_sequence_matched = Some(matched);
    return false;
}
```

**3. Cancellation** (`InferenceExecutor::cancel()`):
```rust
pub fn cancel(&mut self) {
    self.stop_reason = Some(StopReason::Cancelled);
    self.should_stop = true;
}
```

**4. Error** (`InferenceExecutor::error()`):
```rust
pub fn error(&mut self) {
    self.stop_reason = Some(StopReason::Error);
    self.should_stop = true;
}
```

---

## Example Usage

### Basic Request (Sprint 3 Compatible)
```bash
curl -X POST http://localhost:8080/execute \
  -H "Content-Type: application/json" \
  -d '{
    "job_id": "test-1",
    "prompt": "Write a haiku",
    "max_tokens": 100,
    "temperature": 0.7,
    "seed": 42
  }'
```

**Response**:
```json
event: started
data: {"type":"started","job_id":"test-1","model":"Qwen2.5-0.5B","started_at":"2025-10-04T12:00:00Z"}

event: token
data: {"type":"token","t":"GPU","i":0}

event: token
data: {"type":"token","t":" computing","i":1}

event: end
data: {"type":"end","tokens_out":100,"decode_time_ms":2000,"stop_reason":"max_tokens"}
```

---

### Advanced Request (Sprint 4)
```bash
curl -X POST http://localhost:8080/execute \
  -H "Content-Type: application/json" \
  -d '{
    "job_id": "test-2",
    "prompt": "Write a JSON object with name and age",
    "max_tokens": 100,
    "temperature": 0.7,
    "seed": 42,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.1,
    "stop": ["\n\n", "}"],
    "min_p": 0.05
  }'
```

**Response** (stopped by stop sequence):
```json
event: started
data: {"type":"started","job_id":"test-2","model":"Qwen2.5-0.5B","started_at":"2025-10-04T12:00:00Z"}

event: token
data: {"type":"token","t":"{","i":0}

event: token
data: {"type":"token","t":"\"name\"","i":1}

event: token
data: {"type":"token","t":":","i":2}

event: token
data: {"type":"token","t":"\"Alice\"","i":3}

event: token
data: {"type":"token","t":"}","i":4}

event: end
data: {"type":"end","tokens_out":5,"decode_time_ms":150,"stop_reason":"stop_sequence","stop_sequence_matched":"}"}
```

---

## Backward Compatibility

### Sprint 3 Requests Still Work

**Old format**:
```json
{
  "job_id": "test",
  "prompt": "Hello",
  "max_tokens": 100,
  "temperature": 0.7,
  "seed": 42
}
```

**Deserialization**:
- `top_p` → 1.0 (disabled)
- `top_k` → 0 (disabled)
- `repetition_penalty` → 1.0 (disabled)
- `stop` → [] (none)
- `min_p` → 0.0 (disabled)

**Behavior**: Identical to Sprint 3 (basic stochastic sampling)

---

## Testing Strategy

### Unit Tests

**HTTP Validation** (`src/http/validation.rs`):
- ✅ 25 tests covering all parameter ranges
- ✅ Backward compatibility verification
- ✅ Multiple error collection

**Sampling Config** (`src/sampling_config.rs`):
- ✅ 12 tests for configuration logic
- ✅ Seed generation
- ✅ Mode detection

**Inference Executor** (`src/inference_executor.rs`):
- ✅ 10 tests for stop detection
- ✅ Max tokens termination
- ✅ Stop sequence matching
- ✅ Cancellation and error handling

**Inference Result** (`src/inference_result.rs`):
- ✅ 6 tests for result construction
- ✅ Stop reason descriptions

**SSE Events** (`src/http/sse.rs`):
- ✅ 8 tests for serialization
- ✅ Stop reason serialization
- ✅ Optional field omission

**CUDA Kernels** (`cuda/tests/sampling_advanced_test.cu`):
- ✅ 25 tests for all kernels
- ✅ Performance profiling
- ✅ Integration tests

**Total**: 86 tests across all layers

---

## Performance Characteristics

### Overhead per Token

| Component | Latency | Conditions |
|-----------|---------|------------|
| Top-K filtering | <2ms | If top_k > 0 |
| Top-P filtering | <1ms | If top_p < 1.0 |
| Repetition penalty | <0.5ms | If penalty != 1.0 |
| Min-P filtering | <0.1ms | If min_p > 0.0 |
| Stop sequence check | <0.1ms | If stop sequences configured |
| **Total** | **<5ms** | **All enabled** |

### Memory Overhead

| Component | Memory | Notes |
|-----------|--------|-------|
| Top-K sorting | ~1 MB | Thrust temporary buffers |
| Top-P sorting | ~1 MB | Thrust temporary buffers |
| History buffer | ~4 KB | For repetition penalty |
| Stop sequences | ~512 bytes | Up to 4 sequences |
| **Total** | **<2 MB** | **Per inference** |

---

## Error Handling

### Validation Errors (400 Bad Request)

**Single Error** (backward compatible):
```json
{
  "field": "top_p",
  "message": "must be at most 1.0 (got 1.5)"
}
```

**Multiple Errors** (new):
```json
{
  "errors": [
    {
      "field": "top_p",
      "constraint": "range(0.0..=1.0)",
      "message": "top_p must be at most 1.0 (got 1.5)",
      "value": "1.5"
    },
    {
      "field": "repetition_penalty",
      "constraint": "range(0.0..=2.0)",
      "message": "repetition_penalty must be at most 2.0 (got 3.0)",
      "value": "3.0"
    }
  ]
}
```

### Inference Errors (SSE)

**Error Event**:
```json
event: error
data: {
  "type": "error",
  "code": "INFERENCE_FAILED",
  "message": "CUDA kernel launch failed"
}
```

---

## Logging and Observability

### Narration Events

**Validation Failure**:
```rust
Narration::new(ACTOR_WORKER_ORCD, "validation", &req.job_id)
    .human(format!("Validation failed: {} errors", error_count))
    .error_kind("ValidationFailed")
    .emit_warn();
```

**Inference Start**:
```rust
Narration::new(ACTOR_WORKER_ORCD, ACTION_INFERENCE_START, &req.job_id)
    .human(format!("Starting inference with {}", config.sampling_mode()))
    .emit();
```

**Stop Sequence Match**:
```rust
info!(
    tokens_generated = executor.token_count(),
    matched_sequence = ?matched,
    "Matched stop sequence"
);
```

### Tracing

**Request Received**:
```rust
info!(
    correlation_id = %correlation_id,
    job_id = %req.job_id,
    temperature = req.temperature,
    top_p = req.top_p,
    top_k = req.top_k,
    "Inference request validated"
);
```

**Stop Detection**:
```rust
debug!(
    tokens_generated = executor.token_count(),
    max_tokens = config.max_tokens,
    "Reached max_tokens limit"
);
```

---

## Integration Checklist

### HTTP Layer ✅
- [x] Extended ExecuteRequest with 5 new parameters
- [x] Validation for all parameters
- [x] Default values for backward compatibility
- [x] Extended SSE response with stop_reason
- [x] 25 unit tests for validation
- [x] 8 unit tests for SSE events

### Configuration Layer ✅
- [x] SamplingConfig struct
- [x] Conversion from ExecuteRequest
- [x] Seed generation when not provided
- [x] Consistency validation
- [x] 12 unit tests

### Execution Layer ✅
- [x] InferenceExecutor for generation loop
- [x] Stop sequence detection
- [x] Cancellation and error handling
- [x] 10 unit tests

### Result Layer ✅
- [x] InferenceResult with stop reason
- [x] Factory methods for each stop reason
- [x] Helper methods
- [x] 6 unit tests

### CUDA Layer ✅
- [x] Top-K kernel
- [x] Top-P kernel
- [x] Repetition penalty kernel
- [x] Min-P kernel
- [x] Stop sequence check function
- [x] 25 unit tests

---

## Future Enhancements

### M1 Considerations

1. **GPU-side stop sequence matching**: Move pattern matching to GPU for very long sequences
2. **Streaming stop detection**: Check stop sequences incrementally during generation
3. **Advanced stop patterns**: Regex support, case-insensitive matching
4. **Dynamic parameter adjustment**: Adjust sampling params during generation
5. **Sampling presets**: Named presets (creative, balanced, precise)

### Performance Optimizations

1. **Custom sorting kernels**: Replace Thrust with custom kernels if overhead becomes issue
2. **Fused kernels**: Combine top-k + top-p into single kernel
3. **Memory pooling**: Reuse temporary buffers across tokens
4. **Batch stop checking**: Check multiple sequences in parallel

---

## Debugging Guide

### Common Issues

**1. Stop sequences not matching**
- **Symptom**: Generation continues past expected stop
- **Debug**: Check tokenization of stop strings
- **Solution**: Verify stop_sequences are correctly tokenized

**2. All tokens filtered out**
- **Symptom**: Inference fails with "no valid tokens"
- **Debug**: Check parameter combination (top_k + top_p + min_p)
- **Solution**: Relax filtering parameters

**3. Repetition penalty not working**
- **Symptom**: Tokens still repeat
- **Debug**: Check history buffer is passed to kernel
- **Solution**: Verify token_ids() is called correctly

**4. Performance degradation**
- **Symptom**: Slow token generation
- **Debug**: Profile each kernel
- **Solution**: Disable unused parameters or optimize sorting

### Debug Logging

Enable debug logging:
```bash
RUST_LOG=worker_orcd=debug ./worker-orcd
```

Key log messages:
- `"Inference request validated"` - Shows all parameters
- `"Reached max_tokens limit"` - Max tokens termination
- `"Matched stop sequence"` - Stop sequence detection
- `"Inference cancelled"` - Cancellation
- `"Inference terminated due to error"` - Error

---

## API Documentation

### Request Examples

**Greedy (temperature=0.0)**:
```json
{
  "job_id": "greedy-1",
  "prompt": "What is 2+2?",
  "max_tokens": 10,
  "temperature": 0.0
}
```

**Creative (high temperature, top-p)**:
```json
{
  "job_id": "creative-1",
  "prompt": "Write a story",
  "max_tokens": 500,
  "temperature": 1.2,
  "top_p": 0.95
}
```

**Precise (low temperature, top-k)**:
```json
{
  "job_id": "precise-1",
  "prompt": "Generate JSON",
  "max_tokens": 200,
  "temperature": 0.3,
  "top_k": 10,
  "stop": ["}"]
}
```

**Diverse (repetition penalty)**:
```json
{
  "job_id": "diverse-1",
  "prompt": "List 10 colors",
  "max_tokens": 100,
  "temperature": 0.7,
  "repetition_penalty": 1.3
}
```

---

## Spec Compliance

### Requirements Met

- ✅ **M0-W-1421**: Advanced sampling parameters (top-k, top-p, repetition penalty, min-p)
- ✅ **M0-W-1422**: Stop sequences (up to 4, pattern matching)
- ✅ **M0-W-1300**: HTTP API extension (request/response schema)
- ✅ **M0-W-1302**: Request validation (all parameters)
- ✅ **M0-W-1310**: SSE streaming (with stop_reason)

### Test Coverage

- ✅ **86 total tests** across all layers
- ✅ **Unit tests**: 61 tests (HTTP, config, executor, result, SSE)
- ✅ **CUDA tests**: 25 tests (kernels, integration)
- ✅ **Performance**: All kernels within budget (<5ms total)
- ✅ **Backward compatibility**: Sprint 3 requests work unchanged

---

## Summary

The stop_reason system is now **fully integrated** across all layers:

1. **HTTP API**: Extended request/response schema
2. **Configuration**: Unified SamplingConfig
3. **Execution**: InferenceExecutor with stop detection
4. **Result**: InferenceResult with stop reason
5. **CUDA**: All filtering kernels implemented

**Total Implementation**:
- 5 new Rust modules (1,200+ LOC)
- 5 CUDA kernels (500+ LOC)
- 86 comprehensive tests
- Full backward compatibility
- Complete documentation

**Status**: ✅ Production ready for M0

---
Built by Foundation-Alpha 🏗️
