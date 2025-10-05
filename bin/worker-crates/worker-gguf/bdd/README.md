# worker-gguf BDD Tests

Behavior-Driven Development tests for GGUF file format parsing.

## Running Tests

```bash
# Run all BDD tests
cargo run --bin bdd-runner

# Run specific feature
cargo run --bin bdd-runner -- tests/features/gguf_parsing.feature
```

## Features

- **GGUF Parsing** — Verify metadata extraction from GGUF files
  - Architecture detection (llama, gpt, etc.)
  - Vocabulary size
  - Model dimensions
  - Attention configuration (GQA vs MHA)
  - RoPE parameters
  - Context length

## Test Models

Tests use stub metadata based on filename patterns:
- `qwen-*.gguf` → Qwen model metadata
- `phi-*.gguf` → Phi-3 model metadata
- `gpt*.gguf` → GPT-2 model metadata
