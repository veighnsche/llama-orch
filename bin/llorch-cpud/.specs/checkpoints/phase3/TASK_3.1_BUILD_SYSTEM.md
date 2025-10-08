# TEAM-006: Task 3.1 - Create Wrapper Tool Structure
**Part of:** Phase 3 - Implementation  
**Duration:** 20 minutes  
**Status:** ⏳ READY (REVISED BY TEAM-005)  
**Updated by:** TEAM-006

---

## ✅ APPROACH REVISED BY TEAM-005

**Old approach (OBSOLETE):** Modify llama.cpp with conditional compilation  
**New approach (CORRECT):** Create wrapper tool using eval callback API

See [COMPREHENSIVE_ANALYSIS.md](COMPREHENSIVE_ANALYSIS.md) for full analysis.

---

## Objective

Create wrapper tool structure that uses llama.cpp's official eval callback API to extract checkpoints.

**Goal:** Build standalone tool that links against llama.cpp and extracts checkpoints via callbacks.

---

## Prerequisites

- Phase 2 (Mapping) complete ✅
- llama.cpp built and working ✅
- Comprehensive analysis complete ✅

---

## Directory Structure to Create

```
bin/llorch-cpud/tools/checkpoint-extractor/
├── CMakeLists.txt
├── src/
│   ├── main.cpp
│   ├── checkpoint_callback.cpp
│   └── checkpoint_callback.h
└── README.md
```

---

## Implementation

### Step 1: Create Directory Structure

```bash
cd /home/vince/Projects/llama-orch/bin/llorch-cpud
mkdir -p tools/checkpoint-extractor/src
```

### Step 2: Create CMakeLists.txt

**File:** `tools/checkpoint-extractor/CMakeLists.txt`

```cmake
# TEAM-006: Checkpoint extraction wrapper tool
# Created by: TEAM-006
# Based on: TEAM-005 comprehensive analysis

cmake_minimum_required(VERSION 3.14)
project(llorch-checkpoint-extractor)

# Find llama.cpp library
find_package(llama REQUIRED)

# Wrapper tool executable
add_executable(llorch-checkpoint-extractor
    src/main.cpp
    src/checkpoint_callback.cpp
)

target_link_libraries(llorch-checkpoint-extractor
    PRIVATE
    llama
)

target_include_directories(llorch-checkpoint-extractor
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Install
install(TARGETS llorch-checkpoint-extractor
    RUNTIME DESTINATION bin
)
```

### Step 3: Create README.md

**File:** `tools/checkpoint-extractor/README.md`

```markdown
# llorch-checkpoint-extractor

**Created by:** TEAM-006  
**Based on:** TEAM-005 comprehensive analysis

## Purpose

Standalone tool that extracts intermediate tensor checkpoints from llama.cpp inference using the official eval callback API.

## Approach

Uses llama.cpp's `ggml_backend_sched_eval_callback` mechanism:
- Callback fires AFTER each tensor is computed
- Tensors have valid data (not empty like during graph building)
- Non-invasive, no llama.cpp source modifications needed
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
```

---

## Success Criteria

- [ ] Directory `tools/checkpoint-extractor/` created
- [ ] Directory `tools/checkpoint-extractor/src/` created
- [ ] CMakeLists.txt created with llama linkage
- [ ] README.md created explaining approach
- [ ] TEAM-006 signatures added
- [ ] References TEAM-005 analysis

---

## Verification

After creating structure:

```bash
# Verify directory structure
tree bin/llorch-cpud/tools/checkpoint-extractor/

# Should show:
# checkpoint-extractor/
# ├── CMakeLists.txt
# ├── README.md
# └── src/
```

---

## Notes

**Why wrapper tool approach:**
- ✅ No llama.cpp source modifications
- ✅ Uses official eval callback API
- ✅ Tensors have valid data (after computation)
- ✅ Non-invasive and maintainable
- ✅ Set-and-forget (register once, runs automatically)

**TEAM-005 findings:**
- Original plan tried to extract during graph building (tensors empty)
- Eval callback fires AFTER tensor computation (tensors valid)
- Only 3 minimal callbacks needed in llama.cpp (for KV cache and attn output)

---

## Troubleshooting

**Issue:** find_package(llama) fails
- **Solution:** Ensure llama.cpp is built and installed
- **Solution:** Set CMAKE_PREFIX_PATH to llama.cpp install location

**Issue:** Directory already exists
- **Solution:** Normal if task was partially started, verify contents

---

**Status:** ✅ COMPLETE  
**Assigned to:** TEAM-006  
**Estimated time:** 20 minutes  
**Actual time:** 5 minutes

**Updated by TEAM-006 based on TEAM-005 comprehensive analysis**
