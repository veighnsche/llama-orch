# llm-worker-rbee

**Status:** 🚧 STUB (Created by TEAM-135)  
**Purpose:** LLM inference worker daemon

## Overview

llm-worker-rbee is the LLM inference worker that performs actual model inference.

**CRITICAL:** `src/backend/` stays in the binary (LLM-specific inference logic).

## Binaries

- `llm-worker-rbee` - Main worker binary
- `llm-worker-rbee-cpu` - CPU-specific worker
- `llm-worker-rbee-cuda` - CUDA-specific worker
- `llm-worker-rbee-metal` - Metal-specific worker

## Dependencies

- worker-rbee-crates/http-server
- worker-rbee-crates/heartbeat

## Usage

```bash
# TODO: Add usage examples
llm-worker-rbee --model /path/to/model.gguf
```

## Implementation Status

- [ ] Core functionality
- [ ] Tests
- [ ] Documentation
- [ ] Examples

## Notes

Backend stays in binary (exception to crate extraction rule).
