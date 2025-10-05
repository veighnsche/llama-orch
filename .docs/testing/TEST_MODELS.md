# Test Models — Storage and Management

**Status**: Normative  
**Audience**: Test authors, CI maintainers

---

## Purpose

This document defines where and how to store LLM model files for testing `worker-orcd` and end-to-end test suites.

---

## Storage Location

### `.test-models/` (gitignored, local only)

All test model weights MUST be stored under:

```
.test-models/
├── qwen/
│   ├── qwen2.5-0.5b-instruct-q4_k_m.gguf     (~352MB)
│   └── README.md
├── phi3/
│   ├── phi-3-mini-4k-instruct-q4.gguf        (~2.4GB)
│   └── README.md
├── tinyllama/
│   ├── tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf  (~600MB)
│   └── README.md
└── README.md
```

**Properties**:
- ✅ **Gitignored**: Models are never committed (see `.gitignore`)
- ✅ **Local-first**: Each developer/CI runner maintains their own cache
- ✅ **Organized**: Models grouped by family/provider
- ✅ **Documented**: Each subdirectory should have metadata (source URL, SHA256, purpose)

---

## Why `.test-models/`?

1. **Separation of concerns**:
   - Production models → `~/.cache/llama-orch/models/` (catalog-managed)
   - Test models → `.test-models/` (developer-managed)

2. **CI-friendly**:
   - CI can cache `.test-models/` as a workflow artifact
   - Avoids re-downloading models on every run

3. **Explicit opt-in**:
   - Developers only download models they need
   - No surprise 10GB clones

4. **Version control**:
   - Track model metadata and URLs in git
   - Actual weights stay local

---

## Recommended Test Models

### Primary: Qwen2.5-0.5B-Instruct (Q4_K_M)

**Why this model?**
- ✅ **Tiny**: 352MB (smallest viable instruction-tuned model)
- ✅ **Fast**: ~100 tok/s on modest GPU
- ✅ **Quality**: Better instruction-following than TinyLlama
- ✅ **Deterministic**: Supports seed-based reproducibility
- ✅ **Multi-lingual**: Good for i18n tests if needed

**Download**:
```bash
mkdir -p .test-models/qwen
cd .test-models/qwen
wget https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf
```

**Use cases**:
- Unit tests (worker-orcd model loading)
- Integration tests (full spawn → callback → inference cycle)
- BDD scenarios (basic inference flows)

### Secondary: Phi-3-Mini-4K-Instruct (Q4_K_M)

**Why Phi-3?**
- ✅ **MHA Architecture**: Tests Multi-Head Attention (vs Qwen's GQA)
- ✅ **Larger Model**: 2.4GB for VRAM pressure tests
- ✅ **4K Context**: Standard context length
- ✅ **Quality**: Microsoft's high-quality small model
- ✅ **Architecture Variant**: Tests Llama-family MHA configuration

**Download**:
```bash
# Using the download script (recommended)
bash .docs/testing/download_phi3.sh

# Or manually
mkdir -p .test-models/phi3
cd .test-models/phi3
wget https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf
```

**Use cases**:
- Sprint 6: Phi-3 adapter implementation
- VRAM pressure tests
- MHA architecture validation
- Multi-model scenarios

### Tertiary: TinyLlama-1.1B-Chat (Q4_K_M)

**Why keep TinyLlama?**
- Existing `test-harness/e2e-haiku` uses it
- Mid-size model (600MB) for comparison
- Well-known baseline in the community

**Download**:
```bash
mkdir -p .test-models/tinyllama
cd .test-models/tinyllama
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

**Use cases**:
- E2E smoke tests (`test-harness/e2e-haiku`)
- Multi-model scenarios
- Determinism suite (larger context tests)

---

## Model Metadata Template

Each model subdirectory SHOULD include a `README.md`:

```markdown
# Qwen2.5-0.5B-Instruct (Q4_K_M)

**Source**: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF  
**File**: `qwen2.5-0.5b-instruct-q4_k_m.gguf`  
**Size**: 352 MB  
**SHA256**: `[compute after download]`  
**License**: Apache 2.0  
**Context**: 32,768 tokens  
**VRAM**: ~500MB (with 2K context)

## Download

```bash
wget https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf
```

## Verification

```bash
sha256sum qwen2.5-0.5b-instruct-q4_k_m.gguf
# Expected: [SHA256_HERE]
```

## Test Usage

- `bin/worker-orcd/tests/` — Model loading and inference
- `test-harness/bdd/` — Basic inference scenarios
```

---

## Usage in Tests

### Rust Test Helper

```rust
// bin/worker-orcd/tests/helpers/mod.rs
use std::path::PathBuf;

pub fn test_model_path(model_name: &str) -> PathBuf {
    let workspace_root = env!("CARGO_WORKSPACE_ROOT");
    PathBuf::from(workspace_root)
        .join(".test-models")
        .join(model_name)
}

pub fn qwen_model_path() -> PathBuf {
    test_model_path("qwen/qwen2.5-0.5b-instruct-q4_k_m.gguf")
}

pub fn phi3_model_path() -> PathBuf {
    test_model_path("phi3/phi-3-mini-4k-instruct-q4.gguf")
}

pub fn tinyllama_model_path() -> PathBuf {
    test_model_path("tinyllama/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
}

#[cfg(test)]
pub fn ensure_test_model(path: &PathBuf) -> Result<(), String> {
    if !path.exists() {
        return Err(format!(
            "Test model not found: {}\n\n\
            Please download it first. See .docs/testing/TEST_MODELS.md",
            path.display()
        ));
    }
    Ok(())
}
```

### Test Example

```rust
#[tokio::test]
async fn test_worker_loads_model() -> Result<()> {
    let model_path = qwen_model_path();
    ensure_test_model(&model_path)?;
    
    let worker = Worker::new(WorkerConfig {
        model_path: model_path.to_str().unwrap().to_string(),
        gpu_device: 0,
        ..Default::default()
    })?;
    
    worker.load_model().await?;
    assert!(worker.is_ready());
    Ok(())
}
```

---

## CI Setup

### GitHub Actions Cache

```yaml
- name: Cache test models
  uses: actions/cache@v3
  with:
    path: .test-models
    key: test-models-${{ hashFiles('.docs/testing/TEST_MODELS.md') }}

- name: Download Qwen model
  run: |
    mkdir -p .test-models/qwen
    if [ ! -f .test-models/qwen/qwen2.5-0.5b-instruct-q4_k_m.gguf ]; then
      wget -O .test-models/qwen/qwen2.5-0.5b-instruct-q4_k_m.gguf \
        https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf
    fi

- name: Run worker tests
  run: cargo test -p worker-orcd -- --nocapture
```

---

## Environment Variables

### `LLORCH_TEST_MODEL_DIR`

Override the default `.test-models/` location:

```bash
export LLORCH_TEST_MODEL_DIR=/mnt/fast-ssd/test-models
cargo test -p worker-orcd
```

### `LLORCH_TEST_MODEL_SKIP_CHECK`

Skip model existence checks (for tests that don't need real models):

```bash
export LLORCH_TEST_MODEL_SKIP_CHECK=1
cargo test -p worker-orcd -- test_config_parsing
```

---

## Best Practices

### DO

✅ Use smallest viable model for unit tests (Qwen2.5-0.5B)  
✅ Document model source, size, SHA256 in subdirectory README  
✅ Cache models in CI to avoid repeated downloads  
✅ Fail tests early with helpful error if model missing  
✅ Keep model download scripts in test docs  

### DON'T

❌ Commit model weights to git  
❌ Store models in `/tmp` (gets wiped unpredictably)  
❌ Use production catalog path for test models  
❌ Download models inside test code (too slow)  
❌ Assume models exist (always check + error)  

---

## Migration Notes

### Existing Test Suites

- `test-harness/e2e-haiku` currently references `models/` in its README
  - → Update to `.test-models/tinyllama/`
  - → Keep TinyLlama for backward compat
  
- `test-harness/determinism-suite` (future)
  - → Use `.test-models/qwen/` for speed

---

## Related

- **Testing Policy**: `.docs/testing/TESTING_POLICY.md`
- **E2E Haiku**: `test-harness/e2e-haiku/README.md`
- **Worker Readiness Design**: `.docs/WORKER_READINESS_CALLBACK_DESIGN.md`

---

**Status**: Active, normative for `worker-orcd` and all test harnesses requiring real models.
