// TEAM-006: Eval callback for checkpoint extraction
// Created by: TEAM-006
// Based on: TEAM-005 comprehensive analysis
//
// Purpose: Extract intermediate tensor values using eval callback API
//
// Approach: Uses ggml_backend_sched_eval_callback which fires AFTER
//           tensor computation, ensuring tensors have valid data.

#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include <string>
#include <unordered_set>

namespace llorch {

struct CheckpointState {
    std::string output_dir;
    std::unordered_set<std::string> extracted;
    int layer_filter = 0;  // Only extract from layer 0
};

// Eval callback - called after each tensor is computed
// Parameters:
//   t: Tensor after computation (data is valid)
//   ask: If true, callback is asking permission; if false, notifying
//   user_data: Pointer to CheckpointState
// Returns: true to allow execution to continue
bool checkpoint_eval_callback(
    struct ggml_tensor * t,
    bool ask,
    void * user_data
);

// Helper: Check if tensor should be extracted
bool is_checkpoint_tensor(const char * name);

// Helper: Save tensor to disk
void save_checkpoint(
    const char * name,
    struct ggml_tensor * t,
    const std::string & output_dir
);

} // namespace llorch
