# HuggingFace Tokenizer Integration Guide

**Purpose**: Complete guide for HF tokenizers crate integration  
**Audience**: Developers implementing tokenization  
**Story**: GT-047  
**Status**: Complete

---

## Overview

The worker uses the HuggingFace tokenizers Rust crate for GPT-OSS-20B, providing native Rust tokenization without Python dependencies.

---

## Architecture

### Tokenizer Backend Selection

The worker automatically selects the appropriate tokenizer backend based on model metadata.

### Model Mapping

| Model | Backend | Source |
|-------|---------|--------|
| GPT-OSS-20B | HF-JSON | tokenizer.json |
| Qwen2.5-0.5B | GGUF-BPE | GGUF metadata |
| Phi-3-Mini | GGUF-BPE | GGUF metadata |

---

## Implementation

### Dependencies

Add to Cargo.toml:
```toml
[dependencies]
tokenizers = "0.15"
```

### Loading Tokenizer

The HF tokenizer is loaded from a tokenizer.json file in the model directory.

### Encoding

Convert text to token IDs for model input.

### Decoding

Convert token IDs back to text for output streaming.

### UTF-8 Safety

The implementation ensures UTF-8 safety when streaming tokens via SSE.

---

## Usage Example

Load tokenizer.json from model directory, encode prompts, and decode generated tokens with proper UTF-8 boundary handling.

---

## Conformance Testing

Test vectors validate correct encoding/decoding, special token handling, and UTF-8 edge cases.

---

**Last Updated**: 2025-10-05  
**Status**: Complete

---
Crafted by GPT-Gamma 🤖
