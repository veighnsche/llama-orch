# Quick Model Download Reference

**Quick commands to download test models to `.test-models/` directory.**

---

## Download Scripts (Recommended)

### Phi-3-Mini-4K-Instruct (~2.4GB)
```bash
bash .docs/testing/download_phi3.sh
```

### Qwen2.5-0.5B-Instruct (~352MB)
```bash
bash .docs/testing/download_qwen.sh
```

---

## Manual Download (Alternative)

### Phi-3-Mini-4K-Instruct
```bash
mkdir -p .test-models/phi3
cd .test-models/phi3
wget https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf
```

### Qwen2.5-0.5B-Instruct
```bash
mkdir -p .test-models/qwen
cd .test-models/qwen
wget https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf
```

### TinyLlama-1.1B-Chat
```bash
mkdir -p .test-models/tinyllama
cd .test-models/tinyllama
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

---

## Using huggingface-cli (Faster, Resumable)

### Install
```bash
pip install huggingface-hub
```

### Download Phi-3
```bash
cd .test-models/phi3
huggingface-cli download microsoft/Phi-3-mini-4k-instruct-gguf \
  Phi-3-mini-4k-instruct-q4.gguf \
  --local-dir . \
  --local-dir-use-symlinks False
```

### Download Qwen
```bash
cd .test-models/qwen
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct-GGUF \
  qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --local-dir . \
  --local-dir-use-symlinks False
```

---

## Model Comparison

| Model | Size | Context | Architecture | Use Case |
|-------|------|---------|--------------|----------|
| **Qwen2.5-0.5B** | 352MB | 32K | GQA (14 heads, 2 KV) | Primary testing, fast |
| **Phi-3-Mini** | 2.4GB | 4K | MHA (32 heads, 32 KV) | MHA testing, VRAM pressure |
| **TinyLlama** | 600MB | 2K | GQA | Legacy tests, baseline |

---

## Quick Check

```bash
# Check what's downloaded
ls -lh .test-models/*/

# Verify Phi-3
ls -lh .test-models/phi3/phi-3-mini-4k-instruct-q4.gguf

# Verify Qwen
ls -lh .test-models/qwen/qwen2.5-0.5b-instruct-q4_k_m.gguf
```

---

## Storage Rules

✅ **DO**:
- Store in `.test-models/[model-family]/`
- Add README.md with metadata
- Use download scripts

❌ **DON'T**:
- Commit models to git
- Store in `/tmp`
- Use production catalog path

---

See **TEST_MODELS.md** for complete documentation.
