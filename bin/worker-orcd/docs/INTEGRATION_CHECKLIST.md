# Integration Checklist

**Purpose**: Checklist for integrating new models with Foundation layer  
**Audience**: Model implementation teams  
**Owner**: Foundation-Alpha

---

## Overview

This checklist ensures your model integrates correctly with the Foundation layer. Complete all items before declaring integration complete.

---

## Phase 1: Model Implementation

### Configuration

- [ ] **Create model config struct**
  - [ ] `vocab_size: usize`
  - [ ] `hidden_dim: usize`
  - [ ] `num_layers: usize`
  - [ ] `num_heads: usize` (or `num_q_heads` + `num_kv_heads`)
  - [ ] `head_dim: usize`
  - [ ] `ffn_dim: usize`
  - [ ] `max_seq_len: usize`
  - [ ] Architecture-specific fields

- [ ] **Add config validation**
  - [ ] Validate `num_q_heads % num_kv_heads == 0` (if GQA)
  - [ ] Validate `num_q_heads * head_dim == hidden_dim`
  - [ ] Validate all dimensions > 0
  - [ ] Validate `max_seq_len` reasonable (< 1M)

- [ ] **Add config presets**
  - [ ] Small model preset (for testing)
  - [ ] Production model preset
  - [ ] Document parameter choices

### Model Struct

- [ ] **Create model struct**
  - [ ] `config: YourModelConfig`
  - [ ] `total_vram_bytes: usize`
  - [ ] Weight tensor fields (embeddings, layers, etc.)

- [ ] **Implement VRAM calculation**
  - [ ] Calculate weight VRAM
  - [ ] Calculate KV cache VRAM
  - [ ] Calculate activation VRAM
  - [ ] Add 10-20% overhead
  - [ ] Document formula

- [ ] **Add model validation**
  - [ ] Validate weights loaded correctly
  - [ ] Validate VRAM usage matches calculation
  - [ ] Validate tensor shapes

### Weight Loader

- [ ] **Implement weight loader**
  - [ ] Parse GGUF file
  - [ ] Extract metadata
  - [ ] Validate tensor shapes
  - [ ] Allocate VRAM
  - [ ] Copy weights to GPU
  - [ ] Return model instance

- [ ] **Add error handling**
  - [ ] File not found
  - [ ] Invalid GGUF format
  - [ ] Mismatched tensor shapes
  - [ ] VRAM allocation failure
  - [ ] CUDA errors

- [ ] **Add weight loader tests**
  - [ ] Test with valid GGUF file
  - [ ] Test with invalid file
  - [ ] Test VRAM calculation
  - [ ] Test error handling

### Forward Pass

- [ ] **Implement forward pass config**
  - [ ] `is_prefill: bool`
  - [ ] `batch_size: usize`
  - [ ] `seq_len: usize`
  - [ ] `cache_len: usize`
  - [ ] `temperature: f32`
  - [ ] `seed: u32`

- [ ] **Implement prefill**
  - [ ] Process full prompt
  - [ ] Update KV cache
  - [ ] Return logits or token IDs
  - [ ] Handle errors

- [ ] **Implement decode**
  - [ ] Process single token
  - [ ] Use KV cache
  - [ ] Return next token
  - [ ] Handle errors

- [ ] **Implement generate**
  - [ ] Call prefill for prompt
  - [ ] Loop decode for generation
  - [ ] Apply sampling (temperature, top-k, top-p)
  - [ ] Handle EOS token
  - [ ] Return generated tokens

- [ ] **Add forward pass tests**
  - [ ] Test prefill with various lengths
  - [ ] Test decode
  - [ ] Test generate with various max_tokens
  - [ ] Test temperature sweep
  - [ ] Test reproducibility (same seed = same output)

---

## Phase 2: Adapter Integration

### Extend Adapter

- [ ] **Add model type variant**
  - [ ] Add to `ModelType` enum
  - [ ] Document model type

- [ ] **Add model storage**
  - [ ] Add `Option<YourModel>` field to adapter
  - [ ] Initialize to None in other constructors

- [ ] **Add constructor**
  - [ ] `pub fn new_your_model(model: YourModel) -> Self`
  - [ ] Set correct model_type
  - [ ] Set model field
  - [ ] Set other fields to None

### Implement Adapter Methods

- [ ] **Implement `vocab_size()`**
  - [ ] Add match arm for your model type
  - [ ] Return `model.config.vocab_size`
  - [ ] Handle ModelNotLoaded error

- [ ] **Implement `hidden_dim()`**
  - [ ] Add match arm for your model type
  - [ ] Return `model.config.hidden_dim`
  - [ ] Handle ModelNotLoaded error

- [ ] **Implement `num_layers()`**
  - [ ] Add match arm for your model type
  - [ ] Return `model.config.num_layers`
  - [ ] Handle ModelNotLoaded error

- [ ] **Implement `vram_usage()`**
  - [ ] Add match arm for your model type
  - [ ] Return `model.total_vram_bytes`
  - [ ] Handle ModelNotLoaded error

- [ ] **Implement `prefill()`**
  - [ ] Add match arm for your model type
  - [ ] Convert `AdapterForwardConfig` to your config
  - [ ] Call your forward pass
  - [ ] Convert errors to `AdapterError`

- [ ] **Implement `decode()`**
  - [ ] Add match arm for your model type
  - [ ] Convert config
  - [ ] Call your forward pass
  - [ ] Convert errors

- [ ] **Implement `generate()`**
  - [ ] Add match arm for your model type
  - [ ] Convert config
  - [ ] Call your forward pass
  - [ ] Convert errors

### Add Config Conversion

- [ ] **Add conversion method**
  - [ ] `fn to_your_model_config(&self) -> YourForwardConfig`
  - [ ] Map all fields
  - [ ] Document any transformations

### Add Adapter Tests

- [ ] **Test adapter creation**
  - [ ] Test constructor
  - [ ] Verify model_type
  - [ ] Verify model loaded

- [ ] **Test adapter properties**
  - [ ] Test vocab_size()
  - [ ] Test hidden_dim()
  - [ ] Test num_layers()
  - [ ] Test vram_usage()

- [ ] **Test adapter forward pass**
  - [ ] Test prefill
  - [ ] Test decode
  - [ ] Test generate

- [ ] **Test adapter errors**
  - [ ] Test ModelNotLoaded
  - [ ] Test ForwardPassFailed
  - [ ] Test UnsupportedOperation

---

## Phase 3: Integration Tests

### Unit Tests

- [ ] **Test model loading**
  - [ ] Load with valid config
  - [ ] Load with invalid config
  - [ ] Verify VRAM calculation
  - [ ] Verify config validation

- [ ] **Test forward pass**
  - [ ] Test prefill with various inputs
  - [ ] Test decode
  - [ ] Test generate
  - [ ] Test error handling

- [ ] **Test reproducibility**
  - [ ] Same seed produces same output
  - [ ] Different seeds produce different output
  - [ ] Test across 5+ runs

### Integration Tests

- [ ] **Create integration test file**
  - [ ] `tests/your_model_integration.rs`
  - [ ] Import necessary modules
  - [ ] Add test helper functions

- [ ] **Test full pipeline**
  - [ ] Load model
  - [ ] Create adapter
  - [ ] Create tokenizer
  - [ ] Encode prompt
  - [ ] Generate tokens
  - [ ] Decode output
  - [ ] Verify output

- [ ] **Test adapter integration**
  - [ ] Test unified interface
  - [ ] Test model switching
  - [ ] Test error propagation
  - [ ] Test configuration

- [ ] **Test VRAM usage**
  - [ ] Compare calculated vs actual
  - [ ] Test with multiple models
  - [ ] Test cleanup (no leaks)

- [ ] **Test performance**
  - [ ] Benchmark prefill time
  - [ ] Benchmark decode time
  - [ ] Benchmark tokens/second
  - [ ] Compare with expectations

### Edge Case Tests

- [ ] **Test empty input**
  - [ ] Handle gracefully
  - [ ] Return appropriate error

- [ ] **Test very long input**
  - [ ] Test at max_seq_len
  - [ ] Test beyond max_seq_len
  - [ ] Handle appropriately

- [ ] **Test extreme temperatures**
  - [ ] Test temp = 0.0 (greedy)
  - [ ] Test temp = 2.0 (very creative)
  - [ ] Verify no crashes

- [ ] **Test batch processing**
  - [ ] Test batch_size > 1
  - [ ] Verify correct outputs
  - [ ] Verify VRAM scaling

---

## Phase 4: Documentation

### Code Documentation

- [ ] **Add module-level docs**
  - [ ] Purpose and overview
  - [ ] Architecture description
  - [ ] Usage examples
  - [ ] References to specs

- [ ] **Add struct docs**
  - [ ] Document all fields
  - [ ] Document invariants
  - [ ] Add examples

- [ ] **Add function docs**
  - [ ] Document parameters
  - [ ] Document return values
  - [ ] Document errors
  - [ ] Add examples

- [ ] **Add inline comments**
  - [ ] Explain complex logic
  - [ ] Document assumptions
  - [ ] Reference papers/specs

### External Documentation

- [ ] **Create integration guide**
  - [ ] How to use your model
  - [ ] Configuration options
  - [ ] Performance characteristics
  - [ ] Troubleshooting

- [ ] **Update README**
  - [ ] Add model to supported list
  - [ ] Add usage example
  - [ ] Add performance numbers
  - [ ] Add references

- [ ] **Create architecture doc**
  - [ ] Model architecture overview
  - [ ] Implementation details
  - [ ] Design decisions
  - [ ] Future work

---

## Phase 5: Validation

### Functional Validation

- [ ] **All tests passing**
  - [ ] Unit tests: 100% pass
  - [ ] Integration tests: 100% pass
  - [ ] Edge case tests: 100% pass

- [ ] **Code quality**
  - [ ] `cargo fmt` clean
  - [ ] `cargo clippy` no warnings
  - [ ] No TODO/FIXME in production code
  - [ ] No unwrap() in production code

- [ ] **Error handling**
  - [ ] All errors handled
  - [ ] Helpful error messages
  - [ ] No panics in production code
  - [ ] Errors propagated correctly

### Performance Validation

- [ ] **Benchmark results**
  - [ ] Prefill time documented
  - [ ] Decode time documented
  - [ ] Tokens/second documented
  - [ ] VRAM usage documented

- [ ] **Compare with baseline**
  - [ ] Within 10% of expected performance
  - [ ] No performance regressions
  - [ ] Document any discrepancies

### Memory Validation

- [ ] **No memory leaks**
  - [ ] Run with cuda-memcheck
  - [ ] Run with Valgrind
  - [ ] Run leak tests (1000+ iterations)
  - [ ] Verify cleanup on error paths

- [ ] **VRAM usage correct**
  - [ ] Matches calculation
  - [ ] No unexpected growth
  - [ ] Cleanup verified

---

## Phase 6: Integration Complete

### Final Checks

- [ ] **All checklist items complete**
- [ ] **All tests passing**
- [ ] **Documentation complete**
- [ ] **Performance validated**
- [ ] **Memory validated**
- [ ] **Code reviewed**
- [ ] **Integration approved by Foundation team**

### Handoff

- [ ] **Update coordination docs**
  - [ ] Mark integration complete in day-tracker
  - [ ] Update dependencies.md
  - [ ] Update master timeline

- [ ] **Notify teams**
  - [ ] Foundation team
  - [ ] Other model teams
  - [ ] PM team

- [ ] **Create completion summary**
  - [ ] What was implemented
  - [ ] Test results
  - [ ] Performance numbers
  - [ ] Known issues
  - [ ] Future work

---

## Common Issues

### Issue: VRAM calculation wrong

**Symptom**: Actual VRAM usage doesn't match calculation  
**Fix**: 
- Review calculation formulas
- Add logging for each component
- Compare with manual calculation
- Account for CUDA overhead (10-20%)

### Issue: Tests failing

**Symptom**: Integration tests fail  
**Fix**:
- Check error messages
- Verify model loaded correctly
- Verify config correct
- Check for CUDA errors
- Add debug logging

### Issue: Performance slow

**Symptom**: Slower than expected  
**Fix**:
- Profile with NVIDIA Nsight
- Check for synchronization overhead
- Verify kernel parameters
- Compare with baseline
- Optimize hot paths

### Issue: Memory leaks

**Symptom**: VRAM usage grows over time  
**Fix**:
- Run with cuda-memcheck
- Verify Drop implementations
- Check error paths
- Add leak tests
- Use RAII pattern

---

## Support

If you encounter issues:

1. **Check this checklist**: Ensure all items complete
2. **Check documentation**: Review guides and examples
3. **Check existing tests**: Look for similar patterns
4. **Ask Foundation team**: We're here to help!

**Contact**: Foundation-Alpha  
**Location**: `execution/day-tracker.md`, coordination docs

---

**Last Updated**: 2025-10-05  
**Maintainer**: Foundation-Alpha  
**Status**: Complete

---
Built by Foundation-Alpha 🏗️
