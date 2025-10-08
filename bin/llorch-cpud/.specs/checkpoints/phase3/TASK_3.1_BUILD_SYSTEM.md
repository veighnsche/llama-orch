# TEAM-005: Task 3.1 - Create Wrapper Tool Structure
**Part of:** Phase 3 - Implementation  
**Duration:** 20 minutes  
**Status:** ⏳ PENDING (REVISED)

---

## ⚠️ APPROACH REVISED

**Old approach (OBSOLETE):** Modify llama.cpp with conditional compilation  
**New approach (CORRECT):** Create wrapper tool using eval callback

See [COMPREHENSIVE_ANALYSIS.md](COMPREHENSIVE_ANALYSIS.md) for details.

---

## Objective

Create wrapper tool structure that uses llama.cpp's official eval callback API.

**Goal:** Build tool that extracts checkpoints without modifying llama.cpp core.

---

## Prerequisites

- Phase 2 (Mapping) complete
- llama.cpp built and working
- Understanding of eval callback mechanism

---

## Task Details

### Directory Structure to Create

```
bin/llorch-cpud/tools/checkpoint-extractor/
├── CMakeLists.txt
├── src/
│   ├── main.cpp
│   ├── checkpoint_callback.cpp
│   └── checkpoint_callback.h
└── README.md
```

### CMakeLists.txt

```cmake
# TEAM-005: Checkpoint extraction wrapper tool
cmake_minimum_required(VERSION 3.14)
project(llorch-checkpoint-extractor)

# Find llama.cpp
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

### Verification Steps

1. **Create directory structure:**
   ```bash
   cd /home/vince/Projects/llama-orch/bin/llorch-cpud
   mkdir -p tools/checkpoint-extractor/src
   ```

2. **Check output:**
   ```bash
   # Should see:
   # -- TEAM-005: Checkpoint extraction enabled
   # -- TEAM-005: Checkpoints will be saved when LLORCH_VALIDATE=1 env var is set
   ```

3. **Verify CMakeCache:**
   ```bash
   grep "LLORCH_VALIDATE" CMakeCache.txt
   # Should show: LLORCH_VALIDATE:BOOL=ON
   ```

4. **Test without flag:**
   ```bash
   cd ..
   mkdir -p build-normal
   cd build-normal
   cmake ..
   grep "LLORCH_VALIDATE" CMakeCache.txt
   # Should show: LLORCH_VALIDATE:BOOL=OFF (or not present)
   ```

---

## Success Criteria

- [ ] Directory structure created
- [ ] CMakeLists.txt created with llama.cpp linkage
- [ ] Placeholder source files created
- [ ] TEAM-005 signatures in all files
- [ ] README.md explains wrapper approach

---

## Notes

**TEAM-005 Design Decisions:**
- Option defaults to OFF for backward compatibility
- Uses standard CMake `option()` command
- Adds `-DLLORCH_VALIDATE` preprocessor define
- Clear status messages for debugging

**Why conditional compilation:**
- Zero runtime overhead when disabled
- No binary size increase when disabled
- Clean separation of validation code
- Easy to enable/disable for different builds

---

## Troubleshooting

**Issue:** CMake doesn't recognize option
- **Solution:** Make sure option is added before any `add_subdirectory()` calls

**Issue:** Preprocessor define not working
- **Solution:** Verify `add_definitions(-DLLORCH_VALIDATE)` is inside the `if(LLORCH_VALIDATE)` block

**Issue:** Option not showing in cmake-gui
- **Solution:** This is normal - option will appear after first configure

---

**Status:** ⏳ PENDING (REVISED)  
**Assigned to:** TEAM-006  
**Estimated time:** 20 minutes  
**Actual time:** [fill after completion]

**Note:** This task replaces the old CMake modification approach. See COMPREHENSIVE_ANALYSIS.md for rationale.
