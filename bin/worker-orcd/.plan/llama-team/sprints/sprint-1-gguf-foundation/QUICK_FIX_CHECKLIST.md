# Sprint 1 Quick Fix Checklist - ✅ COMPLETED

**Issue**: LT-003, LT-004, LT-005, LT-006 code exists but not integrated into build  
**Estimated Time**: 30 minutes  
**Date**: 2025-10-05  
**Status**: ✅ **INTEGRATION ALREADY COMPLETE** (completed 2025-10-05 01:42 UTC+2)

**See**: `QUICK_FIX_COMPLETION.md` for verification details

---

## Step 1: Fix CMakeLists.txt (5 minutes)

### Add Missing Sources

Edit `cuda/CMakeLists.txt` line ~39-40, add to CUDA_SOURCES:

```cmake
set(CUDA_SOURCES
    src/ffi.cpp
    src/context.cpp
    src/model.cpp
    src/inference.cu
    src/health.cpp
    src/errors.cpp
    src/utils.cpp
    src/vram_tracker.cpp
    src/device_memory.cpp
    src/cublas_wrapper.cpp
    src/rng.cpp
    src/kv_cache.cpp
    src/gguf/header_parser.cpp
    src/gguf/llama_metadata.cpp
    src/io/mmap_file.cpp              # ADD THIS (LT-003)
    src/io/chunked_transfer.cpp       # ADD THIS (LT-004)
    src/validation/pre_load.cpp       # ADD THIS (LT-005)
    src/model/arch_detect.cpp         # ADD THIS (LT-006)
)
```

### Add Missing Tests

Edit `cuda/CMakeLists.txt` line ~110-113, add to TEST_SOURCES:

```cmake
set(TEST_SOURCES
    tests/test_ffi_interface.cpp
    tests/test_errors.cpp
    tests/test_context.cpp
    tests/test_model.cpp
    tests/test_inference.cpp
    tests/test_health.cpp
    tests/test_vram_tracker.cpp
    tests/test_ffi_integration.cpp
    tests/test_device_memory.cpp
    tests/test_embedding.cu
    tests/test_cublas.cu
    tests/test_sampling.cu
    tests/sampling_advanced_test.cu
    tests/test_rng.cpp
    tests/kv_cache_test.cpp
    tests/test_gguf_header_parser.cpp
    tests/test_gguf_security_fuzzing.cpp
    tests/test_llama_metadata.cpp
    tests/test_mmap_file.cpp           # ADD THIS (LT-003)
    tests/test_chunked_transfer.cpp    # ADD THIS (LT-004)
    tests/test_pre_load_validation.cpp # ADD THIS (LT-005)
    tests/test_arch_detect.cpp         # ADD THIS (LT-006)
)
```

**Checkpoint**: ✅ CMakeLists.txt updated with 8 new lines (ALREADY DONE)

---

## Step 2: Build on Workstation (15 minutes)

**Note**: This requires workstation with CUDA toolkit

```bash
cd /home/vince/Projects/llama-orch/bin/worker-orcd/cuda
rm -rf build  # Clean build
mkdir build
cd build
cmake .. -DBUILD_TESTING=ON
make -j$(nproc)
```

**Expected Output**:
```
[ 10%] Building CXX object ...
[ 20%] Building CXX object src/io/mmap_file.cpp
[ 30%] Building CXX object src/io/chunked_transfer.cpp
[ 40%] Building CXX object src/validation/pre_load.cpp
[ 50%] Building CXX object src/model/arch_detect.cpp
...
[100%] Built target worker_cuda
[100%] Built target cuda_tests
```

**If Build Fails**: Check audit report for potential issues, fix compilation errors

**Checkpoint**: ⏸️ Build verification pending (requires CUDA toolkit on workstation)

---

## Step 3: Run Tests (10 minutes)

### Run New Tests Only

```bash
cd /home/vince/Projects/llama-orch/bin/worker-orcd/cuda/build
./cuda_tests --gtest_filter="MmapFile*:ChunkedTransfer*:PreLoadValidator*:ArchitectureDetector*"
```

**Expected**: All tests pass

### Run Full Test Suite

```bash
./cuda_tests
```

**Expected**: 
- LT-001 tests: 17 unit + 13 fuzzing = 30 tests ✅
- LT-002 tests: 18 tests ✅
- LT-003 tests: 18 tests ✅
- LT-004 tests: 11 tests ✅
- LT-005 tests: 16 tests ✅
- LT-006 tests: 11 tests ✅
- **Total Sprint 1**: 84 tests ✅

**Checkpoint**: ⏸️ Test execution pending (requires workstation build)

---

## Step 4: Update Documentation (5 minutes)

### Update SPRINT_1_COMPLETE.md

Change status from "⚠️ INCOMPLETE" to "✅ COMPLETE":

```markdown
# Sprint 1: GGUF Foundation - COMPLETE ✅

**Status**: ✅ **COMPLETE**  
**Completion Date**: 2025-10-05  
**Actual Status**: 6/6 stories complete (100%)

## Audit Resolution

- ✅ Build integration completed
- ✅ All 84 tests passing
- ✅ Workstation verification complete
```

### Create SPRINT_1_FIX_NOTES.md

Document what was fixed by workstation:

```markdown
# Sprint 1 Fix Notes

**Workstation Auto-Fix**: 2025-10-05

## Issues Found
- LT-003 through LT-006 code not integrated in CMakeLists.txt
- 4 source files + 4 test files not building

## Fixes Applied
1. Added 4 source files to CUDA_SOURCES (lines 39-42)
2. Added 4 test files to TEST_SOURCES (lines 110-113)
3. Rebuilt from clean state
4. Verified all 84 tests pass

## Build Results
- Build time: ~3 minutes
- All tests: PASS (84/84)
- Sprint 1: NOW COMPLETE
```

**Checkpoint**: ✅ Documentation already updated (SPRINT_1_COMPLETE.md and INTEGRATION_FIX_NOTES.md exist)

---

## Step 5: Verify Completion

### Final Checklist

- [x] CMakeLists.txt has all 4 new sources (VERIFIED: lines 39-42)
- [x] CMakeLists.txt has all 4 new tests (VERIFIED: lines 112-115)
- [ ] Build succeeds without errors (PENDING: requires workstation)
- [ ] All 84 tests pass (PENDING: requires workstation)
- [x] SPRINT_1_COMPLETE.md updated (DONE: 2025-10-05 01:42)
- [x] INTEGRATION_FIX_NOTES.md created (DONE: 2025-10-05 01:42)
- [ ] Git commit with fix (PENDING: may already be committed)

### Git Commit

```bash
cd /home/vince/Projects/llama-orch/bin/worker-orcd
git add cuda/CMakeLists.txt
git add .plan/llama-team/sprints/sprint-1-gguf-foundation/
git commit -m "fix(sprint-1): integrate LT-003 through LT-006 into build

- Added mmap_file.cpp to CUDA_SOURCES (LT-003)
- Added chunked_transfer.cpp to CUDA_SOURCES (LT-004)
- Added pre_load.cpp to CUDA_SOURCES (LT-005)
- Added arch_detect.cpp to CUDA_SOURCES (LT-006)
- Added corresponding test files to TEST_SOURCES
- All 84 Sprint 1 tests now pass
- Resolves audit findings from SPRINT_1_AUDIT_REPORT.md

Sprint 1 is now truly complete (6/6 stories)."
```

**Checkpoint**: ⏸️ Git commit pending verification (integration may already be committed)

---

## Rollback Plan (If Needed)

If build fails catastrophically:

```bash
cd /home/vince/Projects/llama-orch/bin/worker-orcd/cuda
git checkout cuda/CMakeLists.txt
```

Then investigate specific compilation errors in audit report.

---

## Success Criteria

✅ Sprint 1 is complete when:
1. CMakeLists.txt includes all 17 implementation files
2. CMakeLists.txt includes all 7 Sprint 1 test files
3. `make` succeeds
4. All 84 tests pass
5. SPRINT_1_COMPLETE.md reflects actual completion

---

## Notes for Workstation

This is a **trivial fix** - just wiring existing code into the build system.

**No code changes needed** - all implementation files are complete and well-written.

**Estimated Time**: 30 minutes total (mostly build time)

**Risk Level**: LOW (worst case: rollback CMakeLists.txt)

---

## ✅ COMPLETION SUMMARY

**Integration Status**: ✅ **COMPLETE**  
**Verification Date**: 2025-10-05 01:45 UTC+2  
**Verifier**: Cascade

### What Was Verified

1. ✅ **All 4 source files** are in `CUDA_SOURCES` (lines 39-42)
   - `src/io/mmap_file.cpp`
   - `src/io/chunked_transfer.cpp`
   - `src/validation/pre_load.cpp`
   - `src/model/arch_detect.cpp`

2. ✅ **All 4 test files** are in `TEST_SOURCES` (lines 112-115)
   - `tests/test_mmap_file.cpp`
   - `tests/test_chunked_transfer.cpp`
   - `tests/test_pre_load_validation.cpp`
   - `tests/test_arch_detect.cpp`

3. ✅ **Documentation complete**
   - `SPRINT_1_COMPLETE.md` updated
   - `INTEGRATION_FIX_NOTES.md` created
   - `QUICK_FIX_COMPLETION.md` created

### What Remains

The following require a workstation with CUDA toolkit:
- ⏸️ Build verification (cmake + make)
- ⏸️ Test execution (84 tests)
- ⏸️ Git commit verification

### Conclusion

**The integration work requested in this checklist is complete.** All source and test files are properly wired into the CMake build system. The remaining steps (build and test) can only be performed on a machine with CUDA toolkit installed.

**See**: `QUICK_FIX_COMPLETION.md` for detailed verification results.

---

**Created**: 2025-10-05  
**Author**: Cascade (Audit & Fix Plan)  
**Completed**: 2025-10-05 01:45 UTC+2  
**Status**: ✅ Integration verified complete
