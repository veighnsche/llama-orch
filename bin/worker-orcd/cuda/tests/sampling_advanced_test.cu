/**
 * Advanced Sampling Tests
 * 
 * Unit tests for top-k and top-p (nucleus) sampling kernels.
 * 
 * Spec: M0-W-1421
 * Story: FT-019-EXT-1
 */

#include "../kernels/sampling.cuh"
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <cmath>

using namespace worker::kernels;

// ============================================================================
// Top-K Sampling Tests
// ============================================================================

/**
 * Test 1: BasicTopK
 * 
 * Given: vocab=1000, k=50, uniform logits
 * When: apply top-k filtering
 * Then: Only 50 tokens remain (not -INFINITY)
 */
TEST(TopKSamplingTest, BasicTopK) {
    int vocab_size = 1000;
    int top_k = 50;
    
    // Create logits with known values
    std::vector<float> h_logits(vocab_size);
    for (int i = 0; i < vocab_size; ++i) {
        h_logits[i] = static_cast<float>(i);  // 0, 1, 2, ..., 999
    }
    
    // Copy to device
    float* d_logits;
    cudaMalloc(&d_logits, vocab_size * sizeof(float));
    cudaMemcpy(d_logits, h_logits.data(), vocab_size * sizeof(float), 
               cudaMemcpyHostToDevice);
    
    // Apply top-k filtering
    launch_top_k(d_logits, vocab_size, top_k);
    
    // Copy back
    cudaMemcpy(h_logits.data(), d_logits, vocab_size * sizeof(float), 
               cudaMemcpyDeviceToHost);
    
    // Count non-infinity tokens
    int count_kept = 0;
    for (int i = 0; i < vocab_size; ++i) {
        if (h_logits[i] != -INFINITY) {
            count_kept++;
        }
    }
    
    EXPECT_EQ(count_kept, top_k) << "Should keep exactly top_k tokens";
    
    // Verify top 50 tokens (indices 950-999) are kept
    for (int i = vocab_size - top_k; i < vocab_size; ++i) {
        EXPECT_NE(h_logits[i], -INFINITY) << "Token " << i << " should be kept";
    }
    
    // Verify bottom tokens are filtered
    for (int i = 0; i < vocab_size - top_k; ++i) {
        EXPECT_EQ(h_logits[i], -INFINITY) << "Token " << i << " should be filtered";
    }
    
    cudaFree(d_logits);
}

/**
 * Test 2: TopKDisabled
 * 
 * Given: vocab=1000, k=0 (disabled)
 * When: apply top-k filtering
 * Then: No filtering (all tokens remain)
 */
TEST(TopKSamplingTest, TopKDisabled) {
    int vocab_size = 1000;
    int top_k = 0;  // Disabled
    
    std::vector<float> h_logits(vocab_size);
    for (int i = 0; i < vocab_size; ++i) {
        h_logits[i] = static_cast<float>(i);
    }
    
    std::vector<float> h_logits_original = h_logits;
    
    float* d_logits;
    cudaMalloc(&d_logits, vocab_size * sizeof(float));
    cudaMemcpy(d_logits, h_logits.data(), vocab_size * sizeof(float), 
               cudaMemcpyHostToDevice);
    
    launch_top_k(d_logits, vocab_size, top_k);
    
    cudaMemcpy(h_logits.data(), d_logits, vocab_size * sizeof(float), 
               cudaMemcpyDeviceToHost);
    
    // Verify no changes
    for (int i = 0; i < vocab_size; ++i) {
        EXPECT_EQ(h_logits[i], h_logits_original[i]) 
            << "Logits should be unchanged when top_k=0";
    }
    
    cudaFree(d_logits);
}

/**
 * Test 3: TopKAll
 * 
 * Given: vocab=1000, k=1000 (all tokens)
 * When: apply top-k filtering
 * Then: No filtering (k >= vocab_size)
 */
TEST(TopKSamplingTest, TopKAll) {
    int vocab_size = 1000;
    int top_k = 1000;  // Keep all
    
    std::vector<float> h_logits(vocab_size);
    for (int i = 0; i < vocab_size; ++i) {
        h_logits[i] = static_cast<float>(i);
    }
    
    std::vector<float> h_logits_original = h_logits;
    
    float* d_logits;
    cudaMalloc(&d_logits, vocab_size * sizeof(float));
    cudaMemcpy(d_logits, h_logits.data(), vocab_size * sizeof(float), 
               cudaMemcpyHostToDevice);
    
    launch_top_k(d_logits, vocab_size, top_k);
    
    cudaMemcpy(h_logits.data(), d_logits, vocab_size * sizeof(float), 
               cudaMemcpyDeviceToHost);
    
    // Verify no changes
    for (int i = 0; i < vocab_size; ++i) {
        EXPECT_EQ(h_logits[i], h_logits_original[i]) 
            << "Logits should be unchanged when top_k >= vocab_size";
    }
    
    cudaFree(d_logits);
}

/**
 * Test 4: TopKTooLarge
 * 
 * Given: vocab=1000, k=2000 (> vocab_size)
 * When: apply top-k filtering
 * Then: No filtering (clamped to vocab_size)
 */
TEST(TopKSamplingTest, TopKTooLarge) {
    int vocab_size = 1000;
    int top_k = 2000;  // Larger than vocab
    
    std::vector<float> h_logits(vocab_size);
    for (int i = 0; i < vocab_size; ++i) {
        h_logits[i] = static_cast<float>(i);
    }
    
    std::vector<float> h_logits_original = h_logits;
    
    float* d_logits;
    cudaMalloc(&d_logits, vocab_size * sizeof(float));
    cudaMemcpy(d_logits, h_logits.data(), vocab_size * sizeof(float), 
               cudaMemcpyHostToDevice);
    
    launch_top_k(d_logits, vocab_size, top_k);
    
    cudaMemcpy(h_logits.data(), d_logits, vocab_size * sizeof(float), 
               cudaMemcpyDeviceToHost);
    
    // Verify no changes
    for (int i = 0; i < vocab_size; ++i) {
        EXPECT_EQ(h_logits[i], h_logits_original[i]) 
            << "Logits should be unchanged when top_k > vocab_size";
    }
    
    cudaFree(d_logits);
}

/**
 * Test 5: TopKLargeVocab
 * 
 * Given: vocab=151936 (Qwen2.5), k=100
 * When: apply top-k filtering
 * Then: Correct filtering, performance acceptable
 */
TEST(TopKSamplingTest, TopKLargeVocab) {
    int vocab_size = 151936;  // Qwen2.5 vocab size
    int top_k = 100;
    
    // Create logits with known pattern
    std::vector<float> h_logits(vocab_size);
    for (int i = 0; i < vocab_size; ++i) {
        h_logits[i] = static_cast<float>(i % 1000);  // Repeating pattern
    }
    
    float* d_logits;
    cudaMalloc(&d_logits, vocab_size * sizeof(float));
    cudaMemcpy(d_logits, h_logits.data(), vocab_size * sizeof(float), 
               cudaMemcpyHostToDevice);
    
    // Measure performance
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    launch_top_k(d_logits, vocab_size, top_k);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Performance target: <2ms
    EXPECT_LT(milliseconds, 2.0f) 
        << "Top-k filtering should complete in <2ms for large vocab";
    
    cudaMemcpy(h_logits.data(), d_logits, vocab_size * sizeof(float), 
               cudaMemcpyDeviceToHost);
    
    // Count kept tokens
    int count_kept = 0;
    for (int i = 0; i < vocab_size; ++i) {
        if (h_logits[i] != -INFINITY) {
            count_kept++;
        }
    }
    
    EXPECT_EQ(count_kept, top_k) << "Should keep exactly top_k tokens";
    
    cudaFree(d_logits);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// ============================================================================
// Top-P Sampling Tests
// ============================================================================

/**
 * Test 6: BasicTopP
 * 
 * Given: vocab=10, p=0.9, known distribution
 * When: apply top-p filtering
 * Then: Cumsum cutoff correct
 */
TEST(TopPSamplingTest, BasicTopP) {
    int vocab_size = 10;
    float top_p = 0.9f;
    
    // Create logits: [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    std::vector<float> h_logits(vocab_size);
    for (int i = 0; i < vocab_size; ++i) {
        h_logits[i] = static_cast<float>(vocab_size - 1 - i);
    }
    
    float* d_logits;
    cudaMalloc(&d_logits, vocab_size * sizeof(float));
    cudaMemcpy(d_logits, h_logits.data(), vocab_size * sizeof(float), 
               cudaMemcpyHostToDevice);
    
    launch_top_p(d_logits, vocab_size, top_p);
    
    cudaMemcpy(h_logits.data(), d_logits, vocab_size * sizeof(float), 
               cudaMemcpyDeviceToHost);
    
    // Count kept tokens
    int count_kept = 0;
    for (int i = 0; i < vocab_size; ++i) {
        if (h_logits[i] != -INFINITY) {
            count_kept++;
        }
    }
    
    // With top_p=0.9, expect most high-probability tokens kept
    EXPECT_GT(count_kept, 0) << "Should keep at least one token";
    EXPECT_LT(count_kept, vocab_size) << "Should filter some tokens";
    
    cudaFree(d_logits);
}

/**
 * Test 7: TopPZero
 * 
 * Given: vocab=100, p=0.0
 * When: apply top-p filtering
 * Then: Only max token kept
 */
TEST(TopPSamplingTest, TopPZero) {
    int vocab_size = 100;
    float top_p = 0.0f;
    
    std::vector<float> h_logits(vocab_size);
    for (int i = 0; i < vocab_size; ++i) {
        h_logits[i] = static_cast<float>(i);
    }
    
    float* d_logits;
    cudaMalloc(&d_logits, vocab_size * sizeof(float));
    cudaMemcpy(d_logits, h_logits.data(), vocab_size * sizeof(float), 
               cudaMemcpyHostToDevice);
    
    launch_top_p(d_logits, vocab_size, top_p);
    
    cudaMemcpy(h_logits.data(), d_logits, vocab_size * sizeof(float), 
               cudaMemcpyDeviceToHost);
    
    // Count kept tokens
    int count_kept = 0;
    int max_idx = -1;
    for (int i = 0; i < vocab_size; ++i) {
        if (h_logits[i] != -INFINITY) {
            count_kept++;
            max_idx = i;
        }
    }
    
    // With top_p=0.0, expect only the max token
    EXPECT_EQ(count_kept, 1) << "Should keep only one token with top_p=0.0";
    EXPECT_EQ(max_idx, vocab_size - 1) << "Should keep the max token";
    
    cudaFree(d_logits);
}

/**
 * Test 8: TopPOne
 * 
 * Given: vocab=100, p=1.0 (disabled)
 * When: apply top-p filtering
 * Then: No filtering
 */
TEST(TopPSamplingTest, TopPOne) {
    int vocab_size = 100;
    float top_p = 1.0f;  // Disabled
    
    std::vector<float> h_logits(vocab_size);
    for (int i = 0; i < vocab_size; ++i) {
        h_logits[i] = static_cast<float>(i);
    }
    
    std::vector<float> h_logits_original = h_logits;
    
    float* d_logits;
    cudaMalloc(&d_logits, vocab_size * sizeof(float));
    cudaMemcpy(d_logits, h_logits.data(), vocab_size * sizeof(float), 
               cudaMemcpyHostToDevice);
    
    launch_top_p(d_logits, vocab_size, top_p);
    
    cudaMemcpy(h_logits.data(), d_logits, vocab_size * sizeof(float), 
               cudaMemcpyDeviceToHost);
    
    // Verify no changes
    for (int i = 0; i < vocab_size; ++i) {
        EXPECT_EQ(h_logits[i], h_logits_original[i]) 
            << "Logits should be unchanged when top_p=1.0";
    }
    
    cudaFree(d_logits);
}

/**
 * Test 9: TopPNumericalStability
 * 
 * Given: vocab=100, large logits (>100)
 * When: apply top-p filtering
 * Then: No overflow, correct filtering
 */
TEST(TopPSamplingTest, TopPNumericalStability) {
    int vocab_size = 100;
    float top_p = 0.9f;
    
    // Large logits to test numerical stability
    std::vector<float> h_logits(vocab_size);
    for (int i = 0; i < vocab_size; ++i) {
        h_logits[i] = 100.0f + static_cast<float>(i);
    }
    
    float* d_logits;
    cudaMalloc(&d_logits, vocab_size * sizeof(float));
    cudaMemcpy(d_logits, h_logits.data(), vocab_size * sizeof(float), 
               cudaMemcpyHostToDevice);
    
    launch_top_p(d_logits, vocab_size, top_p);
    
    cudaMemcpy(h_logits.data(), d_logits, vocab_size * sizeof(float), 
               cudaMemcpyDeviceToHost);
    
    // Verify no NaN or Inf (except filtered -INFINITY)
    for (int i = 0; i < vocab_size; ++i) {
        if (h_logits[i] != -INFINITY) {
            EXPECT_FALSE(std::isnan(h_logits[i])) << "No NaN values";
            EXPECT_FALSE(std::isinf(h_logits[i])) << "No Inf values (except -INFINITY)";
        }
    }
    
    // Count kept tokens
    int count_kept = 0;
    for (int i = 0; i < vocab_size; ++i) {
        if (h_logits[i] != -INFINITY) {
            count_kept++;
        }
    }
    
    EXPECT_GT(count_kept, 0) << "Should keep at least one token";
    
    cudaFree(d_logits);
}

/**
 * Test 10: TopPLargeVocab
 * 
 * Given: vocab=151936 (Qwen2.5), p=0.9
 * When: apply top-p filtering
 * Then: Correct filtering, performance acceptable
 */
TEST(TopPSamplingTest, TopPLargeVocab) {
    int vocab_size = 151936;  // Qwen2.5 vocab size
    float top_p = 0.9f;
    
    // Create logits with known pattern
    std::vector<float> h_logits(vocab_size);
    for (int i = 0; i < vocab_size; ++i) {
        h_logits[i] = static_cast<float>(i % 1000);
    }
    
    float* d_logits;
    cudaMalloc(&d_logits, vocab_size * sizeof(float));
    cudaMemcpy(d_logits, h_logits.data(), vocab_size * sizeof(float), 
               cudaMemcpyHostToDevice);
    
    // Measure performance
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    launch_top_p(d_logits, vocab_size, top_p);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Performance target: <1ms
    EXPECT_LT(milliseconds, 1.0f) 
        << "Top-p filtering should complete in <1ms for large vocab";
    
    cudaMemcpy(h_logits.data(), d_logits, vocab_size * sizeof(float), 
               cudaMemcpyDeviceToHost);
    
    // Count kept tokens
    int count_kept = 0;
    for (int i = 0; i < vocab_size; ++i) {
        if (h_logits[i] != -INFINITY) {
            count_kept++;
        }
    }
    
    EXPECT_GT(count_kept, 0) << "Should keep at least one token";
    EXPECT_LT(count_kept, vocab_size) << "Should filter some tokens";
    
    cudaFree(d_logits);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// ============================================================================
// Repetition Penalty Tests
// ============================================================================

/**
 * Test 14: BasicPenalty
 * 
 * Given: logits=[1.0, 2.0, 3.0], history=[1], penalty=1.5
 * When: apply_repetition_penalty
 * Then: logits[1] reduced by factor of 1.5
 */
TEST(RepetitionPenaltyTest, BasicPenalty) {
    int vocab_size = 3;
    float penalty = 1.5f;
    
    std::vector<float> h_logits = {1.0f, 2.0f, 3.0f};
    std::vector<int> h_history = {1};  // Token 1 in history
    
    float* d_logits;
    int* d_history;
    cudaMalloc(&d_logits, vocab_size * sizeof(float));
    cudaMalloc(&d_history, h_history.size() * sizeof(int));
    
    cudaMemcpy(d_logits, h_logits.data(), vocab_size * sizeof(float), 
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_history, h_history.data(), h_history.size() * sizeof(int), 
               cudaMemcpyHostToDevice);
    
    launch_repetition_penalty(d_logits, vocab_size, d_history, 
                             h_history.size(), penalty);
    
    cudaMemcpy(h_logits.data(), d_logits, vocab_size * sizeof(float), 
               cudaMemcpyDeviceToHost);
    
    // Token 1 should be penalized (divided by 1.5)
    EXPECT_FLOAT_EQ(h_logits[0], 1.0f) << "Token 0 should be unchanged";
    EXPECT_FLOAT_EQ(h_logits[1], 2.0f / 1.5f) << "Token 1 should be penalized";
    EXPECT_FLOAT_EQ(h_logits[2], 3.0f) << "Token 2 should be unchanged";
    
    cudaFree(d_logits);
    cudaFree(d_history);
}

/**
 * Test 15: NoHistory
 * 
 * Given: logits, history=[], penalty=1.5
 * When: apply_repetition_penalty
 * Then: logits unchanged
 */
TEST(RepetitionPenaltyTest, NoHistory) {
    int vocab_size = 100;
    float penalty = 1.5f;
    
    std::vector<float> h_logits(vocab_size);
    for (int i = 0; i < vocab_size; ++i) {
        h_logits[i] = static_cast<float>(i);
    }
    
    std::vector<float> h_logits_original = h_logits;
    
    float* d_logits;
    cudaMalloc(&d_logits, vocab_size * sizeof(float));
    cudaMemcpy(d_logits, h_logits.data(), vocab_size * sizeof(float), 
               cudaMemcpyHostToDevice);
    
    // No history (nullptr)
    launch_repetition_penalty(d_logits, vocab_size, nullptr, 0, penalty);
    
    cudaMemcpy(h_logits.data(), d_logits, vocab_size * sizeof(float), 
               cudaMemcpyDeviceToHost);
    
    // Verify no changes
    for (int i = 0; i < vocab_size; ++i) {
        EXPECT_EQ(h_logits[i], h_logits_original[i]) 
            << "Logits should be unchanged with no history";
    }
    
    cudaFree(d_logits);
}

/**
 * Test 16: FullHistory
 * 
 * Given: logits, history=[all tokens], penalty=1.5
 * When: apply_repetition_penalty
 * Then: all logits penalized
 */
TEST(RepetitionPenaltyTest, FullHistory) {
    int vocab_size = 100;
    float penalty = 1.5f;
    
    std::vector<float> h_logits(vocab_size);
    std::vector<int> h_history(vocab_size);
    for (int i = 0; i < vocab_size; ++i) {
        h_logits[i] = static_cast<float>(i) + 1.0f;  // All positive
        h_history[i] = i;
    }
    
    float* d_logits;
    int* d_history;
    cudaMalloc(&d_logits, vocab_size * sizeof(float));
    cudaMalloc(&d_history, vocab_size * sizeof(int));
    
    cudaMemcpy(d_logits, h_logits.data(), vocab_size * sizeof(float), 
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_history, h_history.data(), vocab_size * sizeof(int), 
               cudaMemcpyHostToDevice);
    
    launch_repetition_penalty(d_logits, vocab_size, d_history, 
                             vocab_size, penalty);
    
    cudaMemcpy(h_logits.data(), d_logits, vocab_size * sizeof(float), 
               cudaMemcpyDeviceToHost);
    
    // All tokens should be penalized
    for (int i = 0; i < vocab_size; ++i) {
        float expected = (static_cast<float>(i) + 1.0f) / penalty;
        EXPECT_FLOAT_EQ(h_logits[i], expected) 
            << "Token " << i << " should be penalized";
    }
    
    cudaFree(d_logits);
    cudaFree(d_history);
}

/**
 * Test 17: PenaltyDisabled
 * 
 * Given: logits, history=[1, 2, 3], penalty=1.0 (disabled)
 * When: apply_repetition_penalty
 * Then: logits unchanged
 */
TEST(RepetitionPenaltyTest, PenaltyDisabled) {
    int vocab_size = 100;
    float penalty = 1.0f;  // Disabled
    
    std::vector<float> h_logits(vocab_size);
    std::vector<int> h_history = {1, 2, 3};
    
    for (int i = 0; i < vocab_size; ++i) {
        h_logits[i] = static_cast<float>(i);
    }
    
    std::vector<float> h_logits_original = h_logits;
    
    float* d_logits;
    int* d_history;
    cudaMalloc(&d_logits, vocab_size * sizeof(float));
    cudaMalloc(&d_history, h_history.size() * sizeof(int));
    
    cudaMemcpy(d_logits, h_logits.data(), vocab_size * sizeof(float), 
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_history, h_history.data(), h_history.size() * sizeof(int), 
               cudaMemcpyHostToDevice);
    
    launch_repetition_penalty(d_logits, vocab_size, d_history, 
                             h_history.size(), penalty);
    
    cudaMemcpy(h_logits.data(), d_logits, vocab_size * sizeof(float), 
               cudaMemcpyDeviceToHost);
    
    // Verify no changes
    for (int i = 0; i < vocab_size; ++i) {
        EXPECT_EQ(h_logits[i], h_logits_original[i]) 
            << "Logits should be unchanged when penalty=1.0";
    }
    
    cudaFree(d_logits);
    cudaFree(d_history);
}

// ============================================================================
// Stop Sequences Tests
// ============================================================================

/**
 * Test 18: SingleSequenceMatch
 * 
 * Given: generated=[1, 2, 3, 4], stop=[[3, 4]]
 * When: check_stop_sequences
 * Then: Returns true
 */
TEST(StopSequencesTest, SingleSequenceMatch) {
    std::vector<int> generated = {1, 2, 3, 4};
    std::vector<int> stop_seq = {3, 4};
    
    const int* stop_sequences[4] = {stop_seq.data(), nullptr, nullptr, nullptr};
    int stop_lengths[4] = {2, 0, 0, 0};
    
    bool matched = check_stop_sequences(
        generated.data(), generated.size(),
        stop_sequences, stop_lengths, 1
    );
    
    EXPECT_TRUE(matched) << "Should match stop sequence [3, 4]";
}

/**
 * Test 19: MultipleSequences
 * 
 * Given: generated=[1, 2, 3], stop=[[4, 5], [2, 3]]
 * When: check_stop_sequences
 * Then: Returns true (matches second sequence)
 */
TEST(StopSequencesTest, MultipleSequences) {
    std::vector<int> generated = {1, 2, 3};
    std::vector<int> stop_seq1 = {4, 5};
    std::vector<int> stop_seq2 = {2, 3};
    
    const int* stop_sequences[4] = {stop_seq1.data(), stop_seq2.data(), nullptr, nullptr};
    int stop_lengths[4] = {2, 2, 0, 0};
    
    bool matched = check_stop_sequences(
        generated.data(), generated.size(),
        stop_sequences, stop_lengths, 2
    );
    
    EXPECT_TRUE(matched) << "Should match second stop sequence [2, 3]";
}

/**
 * Test 20: PartialMatch
 * 
 * Given: generated=[1, 2, 3], stop=[[2, 3, 4]]
 * When: check_stop_sequences
 * Then: Returns false (partial match, not complete)
 */
TEST(StopSequencesTest, PartialMatch) {
    std::vector<int> generated = {1, 2, 3};
    std::vector<int> stop_seq = {2, 3, 4};
    
    const int* stop_sequences[4] = {stop_seq.data(), nullptr, nullptr, nullptr};
    int stop_lengths[4] = {3, 0, 0, 0};
    
    bool matched = check_stop_sequences(
        generated.data(), generated.size(),
        stop_sequences, stop_lengths, 1
    );
    
    EXPECT_FALSE(matched) << "Should not match partial sequence";
}

/**
 * Test 21: NoMatch
 * 
 * Given: generated=[1, 2, 3], stop=[[4, 5]]
 * When: check_stop_sequences
 * Then: Returns false
 */
TEST(StopSequencesTest, NoMatch) {
    std::vector<int> generated = {1, 2, 3};
    std::vector<int> stop_seq = {4, 5};
    
    const int* stop_sequences[4] = {stop_seq.data(), nullptr, nullptr, nullptr};
    int stop_lengths[4] = {2, 0, 0, 0};
    
    bool matched = check_stop_sequences(
        generated.data(), generated.size(),
        stop_sequences, stop_lengths, 1
    );
    
    EXPECT_FALSE(matched) << "Should not match when sequence not present";
}

/**
 * Test 22: EmptyStopSequences
 * 
 * Given: generated=[1, 2, 3], stop=[]
 * When: check_stop_sequences
 * Then: Returns false
 */
TEST(StopSequencesTest, EmptyStopSequences) {
    std::vector<int> generated = {1, 2, 3};
    
    const int* stop_sequences[4] = {nullptr, nullptr, nullptr, nullptr};
    int stop_lengths[4] = {0, 0, 0, 0};
    
    bool matched = check_stop_sequences(
        generated.data(), generated.size(),
        stop_sequences, stop_lengths, 0
    );
    
    EXPECT_FALSE(matched) << "Should not match with no stop sequences";
}

// ============================================================================
// Min-P Sampling Tests
// ============================================================================

/**
 * Test 23: BasicMinP
 * 
 * Given: logits with known distribution, min_p=0.05
 * When: apply_min_p
 * Then: Tokens with prob < 5% of max filtered out
 */
TEST(MinPSamplingTest, BasicMinP) {
    int vocab_size = 10;
    float min_p = 0.05f;
    
    // Create logits: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    std::vector<float> h_logits(vocab_size);
    for (int i = 0; i < vocab_size; ++i) {
        h_logits[i] = static_cast<float>(i);
    }
    
    float* d_logits;
    cudaMalloc(&d_logits, vocab_size * sizeof(float));
    cudaMemcpy(d_logits, h_logits.data(), vocab_size * sizeof(float), 
               cudaMemcpyHostToDevice);
    
    launch_min_p(d_logits, vocab_size, min_p);
    
    cudaMemcpy(h_logits.data(), d_logits, vocab_size * sizeof(float), 
               cudaMemcpyDeviceToHost);
    
    // Count kept tokens
    int count_kept = 0;
    for (int i = 0; i < vocab_size; ++i) {
        if (h_logits[i] != -INFINITY) {
            count_kept++;
        }
    }
    
    EXPECT_GT(count_kept, 0) << "Should keep at least one token";
    EXPECT_LT(count_kept, vocab_size) << "Should filter some tokens";
    
    cudaFree(d_logits);
}

/**
 * Test 24: MinPDisabled
 * 
 * Given: logits, min_p=0.0 (disabled)
 * When: apply_min_p
 * Then: No filtering (logits unchanged)
 */
TEST(MinPSamplingTest, MinPDisabled) {
    int vocab_size = 100;
    float min_p = 0.0f;  // Disabled
    
    std::vector<float> h_logits(vocab_size);
    for (int i = 0; i < vocab_size; ++i) {
        h_logits[i] = static_cast<float>(i);
    }
    
    std::vector<float> h_logits_original = h_logits;
    
    float* d_logits;
    cudaMalloc(&d_logits, vocab_size * sizeof(float));
    cudaMemcpy(d_logits, h_logits.data(), vocab_size * sizeof(float), 
               cudaMemcpyHostToDevice);
    
    launch_min_p(d_logits, vocab_size, min_p);
    
    cudaMemcpy(h_logits.data(), d_logits, vocab_size * sizeof(float), 
               cudaMemcpyDeviceToHost);
    
    // Verify no changes
    for (int i = 0; i < vocab_size; ++i) {
        EXPECT_EQ(h_logits[i], h_logits_original[i]) 
            << "Logits should be unchanged when min_p=0.0";
    }
    
    cudaFree(d_logits);
}

/**
 * Test 25: MinPOne
 * 
 * Given: logits, min_p=1.0
 * When: apply_min_p
 * Then: Only max token kept (all others filtered)
 */
TEST(MinPSamplingTest, MinPOne) {
    int vocab_size = 100;
    float min_p = 1.0f;
    
    std::vector<float> h_logits(vocab_size);
    for (int i = 0; i < vocab_size; ++i) {
        h_logits[i] = static_cast<float>(i);
    }
    
    float* d_logits;
    cudaMalloc(&d_logits, vocab_size * sizeof(float));
    cudaMemcpy(d_logits, h_logits.data(), vocab_size * sizeof(float), 
               cudaMemcpyHostToDevice);
    
    launch_min_p(d_logits, vocab_size, min_p);
    
    cudaMemcpy(h_logits.data(), d_logits, vocab_size * sizeof(float), 
               cudaMemcpyDeviceToHost);
    
    // Count kept tokens
    int count_kept = 0;
    int max_idx = -1;
    for (int i = 0; i < vocab_size; ++i) {
        if (h_logits[i] != -INFINITY) {
            count_kept++;
            max_idx = i;
        }
    }
    
    // With min_p=1.0, expect only the max token
    EXPECT_EQ(count_kept, 1) << "Should keep only one token with min_p=1.0";
    EXPECT_EQ(max_idx, vocab_size - 1) << "Should keep the max token";
    
    cudaFree(d_logits);
}

// ============================================================================
// Integration Tests
// ============================================================================

/**
 * Test 11: TopKTopPCombined
 * 
 * Given: vocab=1000, k=100, p=0.9
 * When: apply both top-k and top-p
 * Then: Both filters applied correctly
 */
TEST(IntegrationTest, TopKTopPCombined) {
    int vocab_size = 1000;
    int top_k = 100;
    float top_p = 0.9f;
    
    std::vector<float> h_logits(vocab_size);
    for (int i = 0; i < vocab_size; ++i) {
        h_logits[i] = static_cast<float>(i);
    }
    
    float* d_logits;
    cudaMalloc(&d_logits, vocab_size * sizeof(float));
    cudaMemcpy(d_logits, h_logits.data(), vocab_size * sizeof(float), 
               cudaMemcpyHostToDevice);
    
    // Apply top-k first
    launch_top_k(d_logits, vocab_size, top_k);
    
    // Then apply top-p
    launch_top_p(d_logits, vocab_size, top_p);
    
    cudaMemcpy(h_logits.data(), d_logits, vocab_size * sizeof(float), 
               cudaMemcpyDeviceToHost);
    
    // Count kept tokens
    int count_kept = 0;
    for (int i = 0; i < vocab_size; ++i) {
        if (h_logits[i] != -INFINITY) {
            count_kept++;
        }
    }
    
    // Should be <= top_k (top-p may filter further)
    EXPECT_LE(count_kept, top_k) << "Combined filtering should keep <= top_k tokens";
    EXPECT_GT(count_kept, 0) << "Should keep at least one token";
    
    cudaFree(d_logits);
}

/**
 * Test 12: TemperatureTopKTopP
 * 
 * Given: Full pipeline with temperature + top-k + top-p
 * When: apply all filters
 * Then: Pipeline works correctly
 */
TEST(IntegrationTest, TemperatureTopKTopP) {
    int vocab_size = 1000;
    float temperature = 0.7f;
    int top_k = 50;
    float top_p = 0.9f;
    
    std::vector<float> h_logits(vocab_size);
    for (int i = 0; i < vocab_size; ++i) {
        h_logits[i] = static_cast<float>(i);
    }
    
    float* d_logits;
    cudaMalloc(&d_logits, vocab_size * sizeof(float));
    cudaMemcpy(d_logits, h_logits.data(), vocab_size * sizeof(float), 
               cudaMemcpyHostToDevice);
    
    // Apply full pipeline
    launch_temperature_scale_fp32(d_logits, vocab_size, temperature);
    launch_top_k(d_logits, vocab_size, top_k);
    launch_top_p(d_logits, vocab_size, top_p);
    
    cudaMemcpy(h_logits.data(), d_logits, vocab_size * sizeof(float), 
               cudaMemcpyDeviceToHost);
    
    // Count kept tokens
    int count_kept = 0;
    for (int i = 0; i < vocab_size; ++i) {
        if (h_logits[i] != -INFINITY) {
            count_kept++;
        }
    }
    
    EXPECT_GT(count_kept, 0) << "Should keep at least one token";
    EXPECT_LE(count_kept, top_k) << "Should keep <= top_k tokens";
    
    cudaFree(d_logits);
}

/**
 * Test 13: DeterminismWithFilters
 * 
 * Given: Same seed + same filters
 * When: sample twice
 * Then: Same output
 */
TEST(IntegrationTest, DeterminismWithFilters) {
    int vocab_size = 1000;
    int top_k = 50;
    float top_p = 0.9f;
    float random_value = 0.5f;
    
    std::vector<float> h_logits(vocab_size);
    for (int i = 0; i < vocab_size; ++i) {
        h_logits[i] = static_cast<float>(i);
    }
    
    // Run 1
    float* d_logits1;
    cudaMalloc(&d_logits1, vocab_size * sizeof(float));
    cudaMemcpy(d_logits1, h_logits.data(), vocab_size * sizeof(float), 
               cudaMemcpyHostToDevice);
    
    launch_top_k(d_logits1, vocab_size, top_k);
    launch_top_p(d_logits1, vocab_size, top_p);
    int token_id1 = launch_stochastic_sample(d_logits1, vocab_size, random_value);
    
    // Run 2
    float* d_logits2;
    cudaMalloc(&d_logits2, vocab_size * sizeof(float));
    cudaMemcpy(d_logits2, h_logits.data(), vocab_size * sizeof(float), 
               cudaMemcpyHostToDevice);
    
    launch_top_k(d_logits2, vocab_size, top_k);
    launch_top_p(d_logits2, vocab_size, top_p);
    int token_id2 = launch_stochastic_sample(d_logits2, vocab_size, random_value);
    
    EXPECT_EQ(token_id1, token_id2) << "Same filters + same seed should give same output";
    
    cudaFree(d_logits1);
    cudaFree(d_logits2);
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// ---
// Built by Foundation-Alpha 🏗️
