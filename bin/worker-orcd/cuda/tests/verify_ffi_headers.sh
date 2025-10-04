#!/usr/bin/env bash
#
# FFI Header Compilation Verification
#
# Verifies that FFI headers compile correctly with both C and C++ compilers.
# This is a critical pre-implementation test for the FFI interface lock.
#
# Usage: ./verify_ffi_headers.sh
#
# Exit codes:
#   0 - All tests passed
#   1 - Compilation failed

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INCLUDE_DIR="$(cd "$SCRIPT_DIR/../include" && pwd)"
TEMP_DIR="/tmp/worker-ffi-test-$$"

# Create temp directory
mkdir -p "$TEMP_DIR"

# Cleanup on exit
cleanup() {
    rm -rf "$TEMP_DIR"
}
trap cleanup EXIT

echo "=========================================="
echo "FFI Header Compilation Verification"
echo "=========================================="
echo ""

# Test 1: Compile worker_errors.h with C compiler
echo -n "Test 1: worker_errors.h (C mode)... "
if gcc -c -x c -I"$INCLUDE_DIR" "$INCLUDE_DIR/worker_errors.h" -o "$TEMP_DIR/test1.o" 2>&1 | tee "$TEMP_DIR/test1.log" > /dev/null; then
    echo -e "${GREEN}PASS${NC}"
else
    echo -e "${RED}FAIL${NC}"
    cat "$TEMP_DIR/test1.log"
    exit 1
fi

# Test 2: Compile worker_errors.h with C++ compiler
echo -n "Test 2: worker_errors.h (C++ mode)... "
if g++ -c -x c++ -I"$INCLUDE_DIR" "$INCLUDE_DIR/worker_errors.h" -o "$TEMP_DIR/test2.o" 2>&1 | tee "$TEMP_DIR/test2.log" > /dev/null; then
    echo -e "${GREEN}PASS${NC}"
else
    echo -e "${RED}FAIL${NC}"
    cat "$TEMP_DIR/test2.log"
    exit 1
fi

# Test 3: Compile worker_types.h with C compiler
echo -n "Test 3: worker_types.h (C mode)... "
if gcc -c -x c -I"$INCLUDE_DIR" "$INCLUDE_DIR/worker_types.h" -o "$TEMP_DIR/test3.o" 2>&1 | tee "$TEMP_DIR/test3.log" > /dev/null; then
    echo -e "${GREEN}PASS${NC}"
else
    echo -e "${RED}FAIL${NC}"
    cat "$TEMP_DIR/test3.log"
    exit 1
fi

# Test 4: Compile worker_types.h with C++ compiler
echo -n "Test 4: worker_types.h (C++ mode)... "
if g++ -c -x c++ -I"$INCLUDE_DIR" "$INCLUDE_DIR/worker_types.h" -o "$TEMP_DIR/test4.o" 2>&1 | tee "$TEMP_DIR/test4.log" > /dev/null; then
    echo -e "${GREEN}PASS${NC}"
else
    echo -e "${RED}FAIL${NC}"
    cat "$TEMP_DIR/test4.log"
    exit 1
fi

# Test 5: Compile worker_ffi.h with C compiler
echo -n "Test 5: worker_ffi.h (C mode)... "
if gcc -c -x c -I"$INCLUDE_DIR" "$INCLUDE_DIR/worker_ffi.h" -o "$TEMP_DIR/test5.o" 2>&1 | tee "$TEMP_DIR/test5.log" > /dev/null; then
    echo -e "${GREEN}PASS${NC}"
else
    echo -e "${RED}FAIL${NC}"
    cat "$TEMP_DIR/test5.log"
    exit 1
fi

# Test 6: Compile worker_ffi.h with C++ compiler
echo -n "Test 6: worker_ffi.h (C++ mode)... "
if g++ -c -x c++ -I"$INCLUDE_DIR" "$INCLUDE_DIR/worker_ffi.h" -o "$TEMP_DIR/test6.o" 2>&1 | tee "$TEMP_DIR/test6.log" > /dev/null; then
    echo -e "${GREEN}PASS${NC}"
else
    echo -e "${RED}FAIL${NC}"
    cat "$TEMP_DIR/test6.log"
    exit 1
fi

# Test 7: Multiple inclusion test (include guards)
echo -n "Test 7: Multiple inclusion (include guards)... "
cat > "$TEMP_DIR/test7.c" << 'EOF'
#include "worker_ffi.h"
#include "worker_ffi.h"
#include "worker_types.h"
#include "worker_types.h"
#include "worker_errors.h"
#include "worker_errors.h"

int main(void) {
    return 0;
}
EOF

if gcc -I"$INCLUDE_DIR" "$TEMP_DIR/test7.c" -o "$TEMP_DIR/test7" 2>&1 | tee "$TEMP_DIR/test7.log" > /dev/null; then
    echo -e "${GREEN}PASS${NC}"
else
    echo -e "${RED}FAIL${NC}"
    cat "$TEMP_DIR/test7.log"
    exit 1
fi

# Test 8: C++ multiple inclusion test
echo -n "Test 8: Multiple inclusion (C++ mode)... "
cat > "$TEMP_DIR/test8.cpp" << 'EOF'
#include "worker_ffi.h"
#include "worker_ffi.h"
#include "worker_types.h"
#include "worker_types.h"
#include "worker_errors.h"
#include "worker_errors.h"

int main() {
    return 0;
}
EOF

if g++ -I"$INCLUDE_DIR" "$TEMP_DIR/test8.cpp" -o "$TEMP_DIR/test8" 2>&1 | tee "$TEMP_DIR/test8.log" > /dev/null; then
    echo -e "${GREEN}PASS${NC}"
else
    echo -e "${RED}FAIL${NC}"
    cat "$TEMP_DIR/test8.log"
    exit 1
fi

# Test 9: Verify all functions are declared (compilation only, no linking)
echo -n "Test 9: Function declarations... "
cat > "$TEMP_DIR/test9.cpp" << 'EOF'
#include "worker_ffi.h"

// Verify function pointers can be taken (declarations exist)
void test_function_declarations() {
    // Context management
    auto init_ptr = &cuda_init;
    auto destroy_ptr = &cuda_destroy;
    auto get_device_count_ptr = &cuda_get_device_count;
    
    // Model loading
    auto load_model_ptr = &cuda_load_model;
    auto unload_model_ptr = &cuda_unload_model;
    auto get_vram_usage_ptr = &cuda_model_get_vram_usage;
    
    // Inference execution
    auto inference_start_ptr = &cuda_inference_start;
    auto inference_next_token_ptr = &cuda_inference_next_token;
    auto inference_free_ptr = &cuda_inference_free;
    
    // Health & monitoring
    auto check_vram_residency_ptr = &cuda_check_vram_residency;
    auto get_vram_usage2_ptr = &cuda_get_vram_usage;
    auto get_process_vram_usage_ptr = &cuda_get_process_vram_usage;
    auto check_device_health_ptr = &cuda_check_device_health;
    
    // Error handling
    auto error_message_ptr = &cuda_error_message;
    
    (void)init_ptr;
    (void)destroy_ptr;
    (void)get_device_count_ptr;
    (void)load_model_ptr;
    (void)unload_model_ptr;
    (void)get_vram_usage_ptr;
    (void)inference_start_ptr;
    (void)inference_next_token_ptr;
    (void)inference_free_ptr;
    (void)check_vram_residency_ptr;
    (void)get_vram_usage2_ptr;
    (void)get_process_vram_usage_ptr;
    (void)check_device_health_ptr;
    (void)error_message_ptr;
}
EOF

if g++ -std=c++17 -c -I"$INCLUDE_DIR" "$TEMP_DIR/test9.cpp" -o "$TEMP_DIR/test9.o" 2>&1 | tee "$TEMP_DIR/test9.log" > /dev/null; then
    echo -e "${GREEN}PASS${NC}"
else
    echo -e "${RED}FAIL${NC}"
    cat "$TEMP_DIR/test9.log"
    exit 1
fi

# Test 10: Verify error codes are defined
echo -n "Test 10: Error code definitions... "
cat > "$TEMP_DIR/test10.c" << 'EOF'
#include "worker_errors.h"

int main(void) {
    int codes[] = {
        CUDA_SUCCESS,
        CUDA_ERROR_INVALID_DEVICE,
        CUDA_ERROR_OUT_OF_MEMORY,
        CUDA_ERROR_MODEL_LOAD_FAILED,
        CUDA_ERROR_INFERENCE_FAILED,
        CUDA_ERROR_INVALID_PARAMETER,
        CUDA_ERROR_KERNEL_LAUNCH_FAILED,
        CUDA_ERROR_VRAM_RESIDENCY_FAILED,
        CUDA_ERROR_DEVICE_NOT_FOUND,
        CUDA_ERROR_UNKNOWN,
    };
    (void)codes;
    return 0;
}
EOF

if gcc -I"$INCLUDE_DIR" "$TEMP_DIR/test10.c" -o "$TEMP_DIR/test10" 2>&1 | tee "$TEMP_DIR/test10.log" > /dev/null; then
    echo -e "${GREEN}PASS${NC}"
else
    echo -e "${RED}FAIL${NC}"
    cat "$TEMP_DIR/test10.log"
    exit 1
fi

echo ""
echo "=========================================="
echo -e "${GREEN}All tests passed!${NC}"
echo "=========================================="
echo ""
echo "FFI interface headers are ready for use."
echo "Location: $INCLUDE_DIR"
echo ""
echo "Next steps:"
echo "  - FT-007: Implement Rust FFI bindings"
echo "  - LT-000: Llama team can start C++ implementation"
echo "  - GT-000: GPT team can start C++ implementation"
echo ""

exit 0

# ---
# Built by Foundation-Alpha 🏗️
