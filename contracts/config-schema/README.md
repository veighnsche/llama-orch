# config-schema

**Configuration schema and types for llama-orch**

`contracts/config-schema` — Rust types for YAML/JSON configuration files (pools, models, engines).

---

## What This Library Does

config-schema provides **configuration contracts** for llama-orch:

- **Config types** — Pools, models, engines, replicas
- **Validation** — Schema validation for YAML/JSON
- **Serialization** — serde-based deserialization
- **JSON Schema** — Generate schema.json for validation
- **Defaults** — Sensible defaults for optional fields

**Used by**: rbees-orcd, pool-managerd, provisioners

---

## Key Types

### Config

```rust
use config_schema::Config;

let config = Config {
    pools: vec![
        Pool {
            id: "default".to_string(),
            engine: "llamacpp".to_string(),
            replicas: 1,
            port: 8081,
            model: ModelConfig {
                id: "local:/models/llama-3.1-8b.gguf".to_string(),
            },
            flags: vec!["--parallel".to_string(), "1".to_string()],
        }
    ],
};
```

### Pool

```rust
use config_schema::Pool;

let pool = Pool {
    id: "default".to_string(),
    engine: "llamacpp".to_string(),
    replicas: 1,
    port: 8081,
    model: ModelConfig {
        id: "local:/models/llama-3.1-8b.gguf".to_string(),
    },
    flags: vec!["--parallel".to_string(), "1".to_string()],
};
```

### ModelConfig

```rust
use config_schema::ModelConfig;

let model = ModelConfig {
    id: "local:/models/llama-3.1-8b.gguf".to_string(),
};
```

---

## Configuration File Format

### YAML Example

```yaml
pools:
  - id: default
    engine: llamacpp
    replicas: 1
    port: 8081
    model:
      id: local:/models/llama-3.1-8b-instruct-q4_k_m.gguf
    flags:
      - --parallel
      - "1"
      - --no-cont-batching
      - --metrics
```

### JSON Example

```json
{
  "pools": [
    {
      "id": "default",
      "engine": "llamacpp",
      "replicas": 1,
      "port": 8081,
      "model": {
        "id": "local:/models/llama-3.1-8b-instruct-q4_k_m.gguf"
      },
      "flags": ["--parallel", "1", "--no-cont-batching", "--metrics"]
    }
  ]
}
```

---

## Usage

### Load Configuration

```rust
use config_schema::Config;
use std::fs;

// Load from YAML
let yaml = fs::read_to_string("config.yaml")?;
let config: Config = serde_yaml::from_str(&yaml)?;

// Load from JSON
let json = fs::read_to_string("config.json")?;
let config: Config = serde_json::from_str(&json)?;

println!("Loaded {} pools", config.pools.len());
```

### Validate Configuration

```rust
use config_schema::Config;

let config: Config = serde_yaml::from_str(&yaml)?;

// Validation happens during deserialization
// Invalid configs will fail with descriptive errors
```

---

## JSON Schema Generation

Generate JSON Schema for validation:

```bash
# Generate schema.json
cargo xtask regen-schema

# Output: contracts/config-schema/schema.json
```

Use schema for validation:

```bash
# Validate config file
jsonschema -i config.yaml contracts/config-schema/schema.json
```

---

## Testing

### Unit Tests

```bash
# Run all tests
cargo test -p contracts-config-schema -- --nocapture

# Test deserialization
cargo test -p contracts-config-schema -- test_deserialize --nocapture
```

### Example Configs

Test configs in `requirements/`:

- `requirements/llamacpp-3090-source.yaml`
- `requirements/vllm-a100-binary.yaml`
- `requirements/multi-pool.yaml`

---

## Dependencies

### Internal

- None (foundational contract library)

### External

- `serde` — Serialization
- `serde_yaml` — YAML format
- `serde_json` — JSON format
- `schemars` — JSON Schema generation

---

## Regenerating Artifacts

### JSON Schema

```bash
# Regenerate JSON Schema from types
cargo xtask regen-schema

# Output: contracts/config-schema/schema.json
```

### Validation

```bash
# Validate example configs
cargo test -p contracts-config-schema -- test_examples --nocapture
```

---

## Specifications

Implements requirements from:
- ORCH-3044 (Config schema)
- ORCH-3030 (Configuration format)

---

## Status

- **Version**: 0.0.0 (early development)
- **License**: GPL-3.0-or-later
- **Stability**: Alpha
- **Maintainers**: @llama-orch-maintainers
