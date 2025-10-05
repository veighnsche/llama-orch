# worker-tokenizer BDD Tests

Behavior-Driven Development tests for tokenization.

## Running Tests

```bash
# Run all BDD tests
cargo run --bin bdd-runner

# Run specific feature
cargo run --bin bdd-runner -- tests/features/tokenization.feature
```

## Features

- **Tokenization** — Verify encoding and decoding
  - BPE encoding
  - Token decoding
  - UTF-8 boundary safety
  - Round-trip consistency

## Tokenizer Backends

- `gguf-bpe` — GGUF byte-BPE (Qwen, Phi-3, Llama)
- `hf-json` — HuggingFace tokenizer.json (GPT-OSS-20B)
