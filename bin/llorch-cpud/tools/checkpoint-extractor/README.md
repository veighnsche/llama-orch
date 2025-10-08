# llorch-checkpoint-extractor

**Created by:** TEAM-006  
**Based on:** TEAM-005 comprehensive analysis  
**Fixed by:** TEAM-007 - Layer filtering and documentation

## Purpose

Standalone tool that extracts intermediate tensor checkpoints from llama.cpp inference using the official eval callback API.

## Prerequisites

**IMPORTANT:** This tool requires llama.cpp with 3 checkpoint callbacks added by TEAM-006:

1. `src/llama-graph.cpp:1556` - `cb(k, "cache_k", il);`
2. `src/llama-graph.cpp:1557` - `cb(v, "cache_v", il);`  
3. `src/llama-graph.cpp:1578` - `cb(cur, "attn_out_proj", il);`

These modifications are minimal but **required** for the tool to extract all 9 checkpoints.

## Approach

Uses llama.cpp's `ggml_backend_sched_eval_callback` mechanism:
- Callback fires AFTER each tensor is computed
- Tensors have valid data (not empty like during graph building)
- Minimal llama.cpp modifications (3 callbacks added)
- Official, documented API

## Usage

```bash
./llorch-checkpoint-extractor <model.gguf> <prompt> [output_dir]
```

**Example:**
```bash
./llorch-checkpoint-extractor \
    /path/to/gpt2.gguf \
    "Hello world" \
    /tmp/checkpoints
```

## Output

Creates binary checkpoint files:
- `checkpoint_attn_norm.bin` - LayerNorm output
- `checkpoint_Qcur.bin`, `checkpoint_Kcur.bin`, `checkpoint_Vcur.bin` - QKV projections
- `checkpoint_cache_k.bin`, `checkpoint_cache_v.bin` - KV cache
- `checkpoint_kq_soft_max.bin` - Attention scores
- `checkpoint_attn_out_proj.bin` - Attention output
- `checkpoint_ffn_out.bin` - FFN output

## Binary Format

```
[n_dims:int32][shape:int64[n_dims]][data:float32[n_elements]]
```

## Design

See `COMPREHENSIVE_ANALYSIS.md` for full rationale on why this approach was chosen over inline extraction.
