# worker-models BDD Tests

Behavior-Driven Development tests for model adapters.

## Running Tests

```bash
# Run all BDD tests
cargo run --bin bdd-runner

# Run specific feature
cargo run --bin bdd-runner -- tests/features/model_adapters.feature
```

## Features

- **Model Adapters** — Verify architecture detection and adapter creation
  - Automatic architecture detection from GGUF metadata
  - Adapter factory pattern
  - Llama-style adapters (RoPE, GQA, RMSNorm)
  - GPT-style adapters (absolute pos, MHA, LayerNorm)

## Supported Architectures

- **Llama-style**: Qwen, Phi-3, Llama 2/3
- **GPT-style**: GPT-2, GPT-OSS-20B
