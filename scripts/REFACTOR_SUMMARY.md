# Scripts Refactor Summary

**Date:** 2025-10-09  
**Team:** TEAM-022  
**Version:** 0.2.0

## Architecture Change

**Old:** Monolithic remote script with embedded model/git management  
**New:** Local-only management scripts callable remotely via SSH wrapper

## Created Scripts

### 1. `scripts/llorch-models` (Local-only)

**Purpose:** Model download, verification, and management

**Features:**
- Downloads models using modern `hf` CLI (not deprecated `huggingface-cli`)
- 10 verified HuggingFace repos with documented URLs
- Handles both direct GGUF downloads and PyTorch→GGUF conversion
- Actions: list, catalog, download, info, verify, delete, disk-usage

**Verified Models:**
- `tinyllama` - TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF (669 MB)
- `qwen` - Qwen/Qwen2.5-0.5B-Instruct-GGUF (352 MB)
- `qwen-fp16` - Qwen/Qwen2.5-0.5B-Instruct-GGUF (1.1 GB)
- `phi3` - microsoft/Phi-3-mini-4k-instruct-gguf (2.4 GB)
- `llama3` - QuantFactory/Meta-Llama-3-8B-Instruct-GGUF (4.9 GB)
- `llama32` - tensorblock/Komodo-Llama-3.2-3B-v2-fp16-GGUF (6.4 GB)
- `llama2` - TheBloke/Llama-2-7B-GGUF (7.2 GB)
- `mistral` - TheBloke/Mistral-7B-Instruct-v0.2-GGUF (14 GB)
- `gpt2` - openai-community/gpt2 (500 MB, requires conversion)
- `granite` - ibm-granite/granite-4.0-micro (8 GB, requires conversion)

**Key Design:**
- All HuggingFace repos documented in catalog for easy reference
- No more guessing which repo works - all verified
- Uses `hf download` (modern) not `huggingface-cli` (deprecated)
- Conversion support for models requiring PyTorch→GGUF

### 2. `scripts/llorch-git` (Local-only)

**Purpose:** Git operations with robust submodule support

**Features:**
- Easy submodule branch switching (e.g., candle metal-fixes branch)
- Submodule status, update, reset operations
- Hard reset to origin/main with submodules
- Actions: status, pull, sync, submodules, submodule-branch, submodule-update, submodule-reset, clean, branches, log

**Submodules:**
- reference/candle
- reference/candle-vllm
- reference/llama.cpp
- reference/mistral.rs

**Common Workflows:**
```bash
# Test special Metal branch in candle
llorch-git submodule-branch --submodule reference/candle --branch metal-fixes
llorch-git submodule-update --submodule reference/candle

# Reset everything to clean state
llorch-git sync

# Check if submodules are up to date
llorch-git submodules
```

### 3. `scripts/homelab/llorch-remote` (Remote wrapper)

**Purpose:** Call local scripts on remote hosts via SSH

**Updated Actions:**
- `status`, `pull`, `sync` → calls `llorch-git` remotely
- `models-list`, `models-download`, `models-info` → calls `llorch-models` remotely
- `build`, `test`, `smoke`, `unit`, `integration` → backend-specific operations
- `inference`, `debug-inference`, `logs`, `info` → inference operations
- `clean`, `all` → utilities

**Key Changes:**
- No longer embeds git/model logic
- Delegates to local scripts via SSH
- Cleaner separation of concerns

## Deleted Files

### Old Download Scripts (10 files)
- `.docs/testing/download_gpt2_fp32.sh`
- `.docs/testing/download_granite_4b_fp32.sh`
- `.docs/testing/download_llama2_7b_fp16.sh`
- `.docs/testing/download_llama3_8b.sh`
- `.docs/testing/download_llama32_3b_fp16.sh`
- `.docs/testing/download_mistral_7b_fp16.sh`
- `.docs/testing/download_phi3.sh`
- `.docs/testing/download_qwen_fp16.sh`
- `.docs/testing/download_qwen.sh`
- `.docs/testing/download_tinyllama.sh`

**Replaced by:** `llorch-models` with catalog

### Old Remote Model Script
- `scripts/homelab/llorch-models` (remote-specific version)

**Replaced by:** `scripts/llorch-models` (local-only, callable remotely)

## Benefits

1. **No More Guessing HuggingFace Repos**
   - All repos documented in catalog
   - Verified to work with `hf download`
   - Easy to see which URL worked

2. **Local + Remote Flexibility**
   - Use `llorch-models` locally for dev
   - Use `llorch-remote models-download` for remote
   - Same script, different invocation

3. **Submodule Support**
   - Easy to test special branches (e.g., candle metal-fixes)
   - Clear submodule status
   - Simple reset/update operations

4. **Modern CLI**
   - Uses `hf` (modern) not `huggingface-cli` (deprecated)
   - Supports `hf_transfer` for faster downloads
   - Proper error handling

5. **Cleaner Architecture**
   - Local scripts are self-contained
   - Remote wrapper is thin SSH layer
   - No duplication of logic

## Migration Guide

### Old Command → New Command

**Model Downloads:**
```bash
# Old
bash .docs/testing/download_tinyllama.sh

# New (local)
llorch-models download tinyllama

# New (remote)
llorch-remote mac.home.arpa models-download tinyllama
```

**Git Operations:**
```bash
# Old
llorch-remote mac.home.arpa pull

# New (still works, but now calls llorch-git)
llorch-remote mac.home.arpa pull

# New (local)
llorch-git pull
```

**Submodule Management:**
```bash
# Old (manual git commands)
cd reference/candle
git checkout metal-fixes
git pull

# New
llorch-git submodule-branch --submodule reference/candle --branch metal-fixes
llorch-git submodule-update --submodule reference/candle
```

## Requirements

**For model downloads:**
```bash
pipx install 'huggingface_hub[cli,hf_transfer]'
```

**For model conversion (gpt2, granite):**
- llama.cpp must be present in `reference/llama.cpp`
- Python 3 with required dependencies

## Testing

```bash
# Test local model management
llorch-models catalog
llorch-models download tinyllama
llorch-models list

# Test local git management
llorch-git status
llorch-git submodules

# Test remote operations
llorch-remote mac.home.arpa models-list
llorch-remote mac.home.arpa status
```

## Documentation

- `scripts/homelab/README.md` - Updated with new architecture
- `scripts/llorch-models --help` - Model management help
- `scripts/llorch-git --help` - Git management help
- `scripts/homelab/llorch-remote --help` - Remote wrapper help
