# FT-019-EXT-5: HTTP API Extension

**Team**: Foundation-Alpha  
**Sprint**: Sprint 4 - Advanced Sampling  
**Size**: XS (0.5 days)  
**Days**: 7 - 7  
**Spec Ref**: M0-W-1300, GENERATION_PARAMETERS_ANALYSIS.md

---

## Story Description

Extend HTTP API to support advanced sampling parameters. Add request validation, backward compatibility, and response schema updates.

---

## Acceptance Criteria

- [ ] Request schema extended with 5 new parameters
- [ ] Validation for all parameters
- [ ] Backward compatibility (old requests work)
- [ ] Error messages for invalid parameters
- [ ] Response includes stop_reason
- [ ] Unit tests for validation (5+ tests)
- [ ] Integration tests for API (3+ tests)

---

## Technical Details

### Request Schema (Extended)

```rust
// src/http/types.rs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecuteRequest {
    pub job_id: String,
    pub prompt: String,
    pub max_tokens: u32,
    
    // Core parameters
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    
    #[serde(default)]
    pub seed: Option<u64>,
    
    // Advanced parameters (new)
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    
    #[serde(default)]
    pub top_k: u32,
    
    #[serde(default = "default_repetition_penalty")]
    pub repetition_penalty: f32,
    
    #[serde(default)]
    pub stop: Vec<String>,
    
    #[serde(default)]
    pub min_p: f32,
}

fn default_temperature() -> f32 { 1.0 }
fn default_top_p() -> f32 { 1.0 }
fn default_repetition_penalty() -> f32 { 1.0 }
```

### Validation Logic

```rust
// src/http/validation.rs
pub fn validate_execute_request(req: &ExecuteRequest) -> Result<(), ValidationError> {
    // Core parameters
    if req.temperature < 0.0 || req.temperature > 2.0 {
        return Err(ValidationError::InvalidTemperature {
            value: req.temperature,
            range: "0.0-2.0",
        });
    }
    
    if req.max_tokens == 0 || req.max_tokens > 4096 {
        return Err(ValidationError::InvalidMaxTokens {
            value: req.max_tokens,
            range: "1-4096",
        });
    }
    
    // Advanced parameters
    if req.top_p < 0.0 || req.top_p > 1.0 {
        return Err(ValidationError::InvalidTopP {
            value: req.top_p,
            range: "0.0-1.0",
        });
    }
    
    if req.top_k > 0 && req.top_k > 151936 {  // Max vocab size
        return Err(ValidationError::InvalidTopK {
            value: req.top_k,
            max: 151936,
        });
    }
    
    if req.repetition_penalty < 0.0 || req.repetition_penalty > 2.0 {
        return Err(ValidationError::InvalidRepetitionPenalty {
            value: req.repetition_penalty,
            range: "0.0-2.0",
        });
    }
    
    if req.stop.len() > 4 {
        return Err(ValidationError::TooManyStopSequences {
            count: req.stop.len(),
            max: 4,
        });
    }
    
    for stop_seq in &req.stop {
        if stop_seq.is_empty() {
            return Err(ValidationError::EmptyStopSequence);
        }
        if stop_seq.len() > 100 {  // Reasonable limit
            return Err(ValidationError::StopSequenceTooLong {
                length: stop_seq.len(),
                max: 100,
            });
        }
    }
    
    if req.min_p < 0.0 || req.min_p > 1.0 {
        return Err(ValidationError::InvalidMinP {
            value: req.min_p,
            range: "0.0-1.0",
        });
    }
    
    Ok(())
}
```

### Response Schema (Extended)

```rust
// src/http/types.rs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecuteResponse {
    pub job_id: String,
    pub tokens: Vec<String>,
    pub token_ids: Vec<u32>,
    pub seed: u64,
    
    // New fields
    pub stop_reason: StopReason,
    
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequence_matched: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    MaxTokens,      // Reached max_tokens limit
    StopSequence,   // Matched a stop sequence
    Error,          // Error occurred
    Cancelled,      // Request cancelled
}
```

### Error Types

```rust
#[derive(Debug, thiserror::Error)]
pub enum ValidationError {
    #[error("Invalid temperature: {value} (must be in range {range})")]
    InvalidTemperature { value: f32, range: &'static str },
    
    #[error("Invalid top_p: {value} (must be in range {range})")]
    InvalidTopP { value: f32, range: &'static str },
    
    #[error("Invalid top_k: {value} (must be <= {max})")]
    InvalidTopK { value: u32, max: u32 },
    
    #[error("Invalid repetition_penalty: {value} (must be in range {range})")]
    InvalidRepetitionPenalty { value: f32, range: &'static str },
    
    #[error("Too many stop sequences: {count} (max {max})")]
    TooManyStopSequences { count: usize, max: usize },
    
    #[error("Empty stop sequence not allowed")]
    EmptyStopSequence,
    
    #[error("Stop sequence too long: {length} chars (max {max})")]
    StopSequenceTooLong { length: usize, max: usize },
    
    #[error("Invalid min_p: {value} (must be in range {range})")]
    InvalidMinP { value: f32, range: &'static str },
}
```

---

## Testing Strategy

### Unit Tests (5 tests)

1. **ValidRequest**: All parameters valid
2. **InvalidTemperature**: temperature=3.0 rejected
3. **InvalidTopP**: top_p=1.5 rejected
4. **TooManyStopSequences**: 5 sequences rejected
5. **BackwardCompatibility**: Old request format accepted

### Integration Tests (3 tests)

1. **FullPipeline**: Request with all parameters → successful response
2. **StopReasonMaxTokens**: Generation reaches max_tokens
3. **StopReasonStopSequence**: Generation matches stop sequence

---

## Backward Compatibility

### Old Request Format (Still Works)

```json
{
  "job_id": "job-xyz",
  "prompt": "Write a haiku",
  "max_tokens": 100,
  "temperature": 0.7,
  "seed": 42
}
```

**Behavior**: 
- Uses default values for new parameters
- Identical behavior to Sprint 3
- No breaking changes

### Migration Guide

**For Existing Clients**:
- No changes required
- New parameters optional
- Can adopt incrementally

**For New Clients**:
- Use advanced parameters for better quality
- Refer to API documentation for parameter descriptions

---

## Definition of Done

- [ ] Request schema extended
- [ ] Validation implemented and tested (5 tests)
- [ ] Response schema extended
- [ ] Integration tests passing (3 tests)
- [ ] Backward compatibility verified
- [ ] API documentation updated
- [ ] Code reviewed (self-review)

---

## References

- **Spec**: `bin/.specs/01_M0_worker_orcd.md` (M0-W-1300)
- **Analysis**: `bin/.specs/.docs/GENERATION_PARAMETERS_ANALYSIS.md`

---
Built by Foundation-Alpha 🏗️
