/**
 * Sampling Kernels Unit Tests
 * 
 * Tests temperature scaling and greedy sampling kernel correctness.
 * 
 * Spec: M0-W-1032, M0-W-1421, KERNEL-SAMPLE-003
 * Story: FT-017, FT-018
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../kernels/sampling.cuh"
#include <vector>
#include <cmath>

using namespace worker::kernels;

// ============================================================================
// Test Fixture
// ============================================================================

class TemperatureScaleTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize CUDA device
        int device_count;
        cudaGetDeviceCount(&device_count);
        if (device_count == 0) {
            GTEST_SKIP() << "No CUDA devices available";
        }
        cudaSetDevice(0);
        
        // Allocate device memory
        vocab_size = 1000;
        cudaMalloc(&d_logits_fp32, vocab_size * sizeof(float));
        cudaMalloc(&d_logits_fp16, vocab_size * sizeof(half));
    }
    
    void TearDown() override {
        cudaFree(d_logits_fp32);
        cudaFree(d_logits_fp16);
    }
    
    int vocab_size;
    float* d_logits_fp32 = nullptr;
    half* d_logits_fp16 = nullptr;
};

// ============================================================================
// Basic Functionality Tests (FP32)
// ============================================================================

/**
 * Test: Temperature = 1.0 (no change)
 * 
 * Spec: M0-W-1032 (Temperature Scaling)
 * Critical: Identity case must work
 */
TEST_F(TemperatureScaleTest, TemperatureOneNoChange) {
    // Temperature = 1.0 should not change logits
    std::vector<float> h_logits(vocab_size);
    for (int i = 0; i < vocab_size; ++i) {
        h_logits[i] = static_cast<float>(i);
    }
    
    cudaMemcpy(d_logits_fp32, h_logits.data(), vocab_size * sizeof(float), 
               cudaMemcpyHostToDevice);
    
    launch_temperature_scale_fp32(d_logits_fp32, vocab_size, 1.0f);
    cudaDeviceSynchronize();
    
    std::vector<float> h_result(vocab_size);
    cudaMemcpy(h_result.data(), d_logits_fp32, vocab_size * sizeof(float), 
               cudaMemcpyDeviceToHost);
    
    // Verify unchanged
    for (int i = 0; i < vocab_size; ++i) {
        EXPECT_FLOAT_EQ(h_result[i], h_logits[i]) << "Index " << i;
    }
}

/**
 * Test: Temperature = 0.5 (doubles logits)
 * 
 * Spec: M0-W-1032 (Temperature Scaling)
 * Critical: Scaling math must be correct
 */
TEST_F(TemperatureScaleTest, TemperatureHalfDoublesLogits) {
    // Temperature = 0.5 should double logits (divide by 0.5 = multiply by 2)
    std::vector<float> h_logits(vocab_size, 1.0f);
    
    cudaMemcpy(d_logits_fp32, h_logits.data(), vocab_size * sizeof(float), 
               cudaMemcpyHostToDevice);
    
    launch_temperature_scale_fp32(d_logits_fp32, vocab_size, 0.5f);
    cudaDeviceSynchronize();
    
    std::vector<float> h_result(vocab_size);
    cudaMemcpy(h_result.data(), d_logits_fp32, vocab_size * sizeof(float), 
               cudaMemcpyDeviceToHost);
    
    // Verify doubled
    for (int i = 0; i < vocab_size; ++i) {
        EXPECT_NEAR(h_result[i], 2.0f, 0.001f) << "Index " << i;
    }
}

/**
 * Test: Temperature = 2.0 (halves logits)
 * 
 * Spec: M0-W-1032 (Temperature Scaling)
 * Critical: Scaling math must be correct
 */
TEST_F(TemperatureScaleTest, TemperatureTwoHalvesLogits) {
    // Temperature = 2.0 should halve logits
    std::vector<float> h_logits(vocab_size, 4.0f);
    
    cudaMemcpy(d_logits_fp32, h_logits.data(), vocab_size * sizeof(float), 
               cudaMemcpyHostToDevice);
    
    launch_temperature_scale_fp32(d_logits_fp32, vocab_size, 2.0f);
    cudaDeviceSynchronize();
    
    std::vector<float> h_result(vocab_size);
    cudaMemcpy(h_result.data(), d_logits_fp32, vocab_size * sizeof(float), 
               cudaMemcpyDeviceToHost);
    
    // Verify halved
    for (int i = 0; i < vocab_size; ++i) {
        EXPECT_NEAR(h_result[i], 2.0f, 0.001f) << "Index " << i;
    }
}

/**
 * Test: Temperature = 0.0 (greedy mode, no change)
 * 
 * Spec: M0-W-1032 (Temperature Scaling)
 * Critical: Special case for testing reproducibility
 */
TEST_F(TemperatureScaleTest, TemperatureZeroNoChange) {
    // Temperature = 0.0 should not change logits (greedy mode)
    std::vector<float> h_logits(vocab_size);
    for (int i = 0; i < vocab_size; ++i) {
        h_logits[i] = static_cast<float>(i);
    }
    
    cudaMemcpy(d_logits_fp32, h_logits.data(), vocab_size * sizeof(float), 
               cudaMemcpyHostToDevice);
    
    launch_temperature_scale_fp32(d_logits_fp32, vocab_size, 0.0f);
    cudaDeviceSynchronize();
    
    std::vector<float> h_result(vocab_size);
    cudaMemcpy(h_result.data(), d_logits_fp32, vocab_size * sizeof(float), 
               cudaMemcpyDeviceToHost);
    
    // Verify unchanged (greedy mode)
    for (int i = 0; i < vocab_size; ++i) {
        EXPECT_FLOAT_EQ(h_result[i], h_logits[i]) << "Index " << i;
    }
}

/**
 * Test: Negative logits
 * 
 * Critical: Handles negative values correctly
 */
TEST_F(TemperatureScaleTest, NegativeLogits) {
    // Test with negative logits
    std::vector<float> h_logits(vocab_size, -2.0f);
    
    cudaMemcpy(d_logits_fp32, h_logits.data(), vocab_size * sizeof(float), 
               cudaMemcpyHostToDevice);
    
    launch_temperature_scale_fp32(d_logits_fp32, vocab_size, 0.5f);
    cudaDeviceSynchronize();
    
    std::vector<float> h_result(vocab_size);
    cudaMemcpy(h_result.data(), d_logits_fp32, vocab_size * sizeof(float), 
               cudaMemcpyDeviceToHost);
    
    // -2.0 / 0.5 = -4.0
    for (int i = 0; i < vocab_size; ++i) {
        EXPECT_NEAR(h_result[i], -4.0f, 0.001f) << "Index " << i;
    }
}

/**
 * Test: Large vocabulary (Qwen 151936)
 * 
 * Critical: Real vocabulary size
 */
TEST_F(TemperatureScaleTest, LargeVocabulary) {
    // Test with realistic vocabulary size (Qwen2.5-0.5B)
    int large_vocab = 151936;
    float* d_large_logits;
    cudaMalloc(&d_large_logits, large_vocab * sizeof(float));
    
    std::vector<float> h_logits(large_vocab, 1.0f);
    cudaMemcpy(d_large_logits, h_logits.data(), large_vocab * sizeof(float), 
               cudaMemcpyHostToDevice);
    
    launch_temperature_scale_fp32(d_large_logits, large_vocab, 0.7f);
    cudaDeviceSynchronize();
    
    std::vector<float> h_result(large_vocab);
    cudaMemcpy(h_result.data(), d_large_logits, large_vocab * sizeof(float), 
               cudaMemcpyDeviceToHost);
    
    // 1.0 / 0.7 = 1.428571...
    float expected = 1.0f / 0.7f;
    for (int i = 0; i < large_vocab; ++i) {
        EXPECT_NEAR(h_result[i], expected, 0.01f) << "Index " << i;
    }
    
    cudaFree(d_large_logits);
}

// ============================================================================
// FP16 Tests
// ============================================================================

/**
 * Test: FP16 temperature scaling
 * 
 * Critical: FP16 variant must work correctly
 */
TEST_F(TemperatureScaleTest, FP16Scaling) {
    std::vector<half> h_logits(vocab_size);
    for (int i = 0; i < vocab_size; ++i) {
        h_logits[i] = __float2half(2.0f);
    }
    
    cudaMemcpy(d_logits_fp16, h_logits.data(), vocab_size * sizeof(half), 
               cudaMemcpyHostToDevice);
    
    launch_temperature_scale_fp16(d_logits_fp16, vocab_size, 0.5f);
    cudaDeviceSynchronize();
    
    std::vector<half> h_result(vocab_size);
    cudaMemcpy(h_result.data(), d_logits_fp16, vocab_size * sizeof(half), 
               cudaMemcpyDeviceToHost);
    
    // 2.0 / 0.5 = 4.0
    for (int i = 0; i < vocab_size; ++i) {
        EXPECT_NEAR(__half2float(h_result[i]), 4.0f, 0.1f) << "Index " << i;
    }
}

/**
 * Test: FP16 temperature = 0.0 (greedy mode)
 * 
 * Critical: FP16 greedy mode must work
 */
TEST_F(TemperatureScaleTest, FP16TemperatureZero) {
    std::vector<half> h_logits(vocab_size);
    for (int i = 0; i < vocab_size; ++i) {
        h_logits[i] = __float2half(static_cast<float>(i));
    }
    
    cudaMemcpy(d_logits_fp16, h_logits.data(), vocab_size * sizeof(half), 
               cudaMemcpyHostToDevice);
    
    launch_temperature_scale_fp16(d_logits_fp16, vocab_size, 0.0f);
    cudaDeviceSynchronize();
    
    std::vector<half> h_result(vocab_size);
    cudaMemcpy(h_result.data(), d_logits_fp16, vocab_size * sizeof(half), 
               cudaMemcpyDeviceToHost);
    
    // Verify unchanged
    for (int i = 0; i < vocab_size; ++i) {
        EXPECT_NEAR(__half2float(h_result[i]), __half2float(h_logits[i]), 0.01f) << "Index " << i;
    }
}

// ============================================================================
// Edge Case Tests
// ============================================================================

/**
 * Test: Invalid temperature (negative)
 * 
 * Critical: Defensive programming
 */
TEST_F(TemperatureScaleTest, InvalidTemperatureNegative) {
    std::vector<float> h_logits(vocab_size, 1.0f);
    
    cudaMemcpy(d_logits_fp32, h_logits.data(), vocab_size * sizeof(float), 
               cudaMemcpyHostToDevice);
    
    // Negative temperature should be ignored (no scaling)
    launch_temperature_scale_fp32(d_logits_fp32, vocab_size, -0.5f);
    cudaDeviceSynchronize();
    
    std::vector<float> h_result(vocab_size);
    cudaMemcpy(h_result.data(), d_logits_fp32, vocab_size * sizeof(float), 
               cudaMemcpyDeviceToHost);
    
    // Verify unchanged (invalid temperature ignored)
    for (int i = 0; i < vocab_size; ++i) {
        EXPECT_FLOAT_EQ(h_result[i], 1.0f) << "Index " << i;
    }
}

/**
 * Test: Invalid temperature (too large)
 * 
 * Critical: Defensive programming
 */
TEST_F(TemperatureScaleTest, InvalidTemperatureTooLarge) {
    std::vector<float> h_logits(vocab_size, 1.0f);
    
    cudaMemcpy(d_logits_fp32, h_logits.data(), vocab_size * sizeof(float), 
               cudaMemcpyHostToDevice);
    
    // Temperature > 2.0 should be ignored (no scaling)
    launch_temperature_scale_fp32(d_logits_fp32, vocab_size, 3.0f);
    cudaDeviceSynchronize();
    
    std::vector<float> h_result(vocab_size);
    cudaMemcpy(h_result.data(), d_logits_fp32, vocab_size * sizeof(float), 
               cudaMemcpyDeviceToHost);
    
    // Verify unchanged (invalid temperature ignored)
    for (int i = 0; i < vocab_size; ++i) {
        EXPECT_FLOAT_EQ(h_result[i], 1.0f) << "Index " << i;
    }
}

/**
 * Test: Mixed positive and negative logits
 * 
 * Critical: Realistic logit distribution
 */
TEST_F(TemperatureScaleTest, MixedLogits) {
    std::vector<float> h_logits(vocab_size);
    for (int i = 0; i < vocab_size; ++i) {
        h_logits[i] = static_cast<float>(i - 500);  // Range: -500 to 499
    }
    
    cudaMemcpy(d_logits_fp32, h_logits.data(), vocab_size * sizeof(float), 
               cudaMemcpyHostToDevice);
    
    float temperature = 0.8f;
    launch_temperature_scale_fp32(d_logits_fp32, vocab_size, temperature);
    cudaDeviceSynchronize();
    
    std::vector<float> h_result(vocab_size);
    cudaMemcpy(h_result.data(), d_logits_fp32, vocab_size * sizeof(float), 
               cudaMemcpyDeviceToHost);
    
    // Verify scaling
    for (int i = 0; i < vocab_size; ++i) {
        float expected = h_logits[i] / temperature;
        EXPECT_NEAR(h_result[i], expected, 0.01f) << "Index " << i;
    }
}

// ============================================================================
// Real-World Scenarios
// ============================================================================

/**
 * Test: Common temperature values
 * 
 * Critical: Production temperature ranges
 */
TEST_F(TemperatureScaleTest, CommonTemperatureValues) {
    std::vector<float> temperatures = {0.0f, 0.3f, 0.5f, 0.7f, 0.9f, 1.0f, 1.2f, 1.5f, 2.0f};
    
    for (float temp : temperatures) {
        std::vector<float> h_logits(vocab_size, 1.0f);
        
        cudaMemcpy(d_logits_fp32, h_logits.data(), vocab_size * sizeof(float), 
                   cudaMemcpyHostToDevice);
        
        launch_temperature_scale_fp32(d_logits_fp32, vocab_size, temp);
        cudaDeviceSynchronize();
        
        std::vector<float> h_result(vocab_size);
        cudaMemcpy(h_result.data(), d_logits_fp32, vocab_size * sizeof(float), 
                   cudaMemcpyDeviceToHost);
        
        // Verify scaling
        float expected = (temp == 0.0f) ? 1.0f : (1.0f / temp);
        EXPECT_NEAR(h_result[0], expected, 0.01f) << "Temperature " << temp;
    }
}

/**
 * Test: GPT-OSS-20B vocabulary (50257)
 * 
 * Critical: Real model vocabulary
 */
TEST_F(TemperatureScaleTest, GPTVocabulary) {
    int gpt_vocab = 50257;
    float* d_gpt_logits;
    cudaMalloc(&d_gpt_logits, gpt_vocab * sizeof(float));
    
    std::vector<float> h_logits(gpt_vocab);
    for (int i = 0; i < gpt_vocab; ++i) {
        h_logits[i] = static_cast<float>(i % 100) * 0.1f;
    }
    
    cudaMemcpy(d_gpt_logits, h_logits.data(), gpt_vocab * sizeof(float), 
               cudaMemcpyHostToDevice);
    
    launch_temperature_scale_fp32(d_gpt_logits, gpt_vocab, 0.9f);
    cudaDeviceSynchronize();
    
    std::vector<float> h_result(gpt_vocab);
    cudaMemcpy(h_result.data(), d_gpt_logits, gpt_vocab * sizeof(float), 
               cudaMemcpyDeviceToHost);
    
    // Verify scaling
    for (int i = 0; i < gpt_vocab; ++i) {
        float expected = h_logits[i] / 0.9f;
        EXPECT_NEAR(h_result[i], expected, 0.01f) << "Index " << i;
    }
    
    cudaFree(d_gpt_logits);
}

// ============================================================================
// Determinism Tests
// ============================================================================

/**
 * Test: Temperature scaling is deterministic
 * 
 * Property: Same inputs → same outputs (every time)
 */
TEST_F(TemperatureScaleTest, DeterministicScaling) {
    std::vector<float> h_logits(vocab_size);
    for (int i = 0; i < vocab_size; ++i) {
        h_logits[i] = static_cast<float>(i % 100) * 0.1f;
    }
    
    float temperature = 0.7f;
    
    // Run kernel 5 times
    std::vector<std::vector<float>> results(5);
    for (int run = 0; run < 5; ++run) {
        // Reset logits
        cudaMemcpy(d_logits_fp32, h_logits.data(), vocab_size * sizeof(float), 
                   cudaMemcpyHostToDevice);
        
        launch_temperature_scale_fp32(d_logits_fp32, vocab_size, temperature);
        cudaDeviceSynchronize();
        
        results[run].resize(vocab_size);
        cudaMemcpy(results[run].data(), d_logits_fp32, vocab_size * sizeof(float), 
                   cudaMemcpyDeviceToHost);
    }
    
    // Verify all runs produce identical results
    for (int run = 1; run < 5; ++run) {
        for (int i = 0; i < vocab_size; ++i) {
            EXPECT_FLOAT_EQ(results[0][i], results[run][i]) 
                << "Run " << run << ", index " << i << " differs";
        }
    }
}

// ============================================================================
// Greedy Sampling Tests
// ============================================================================

class GreedySamplingTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize CUDA device
        int device_count;
        cudaGetDeviceCount(&device_count);
        if (device_count == 0) {
            GTEST_SKIP() << "No CUDA devices available";
        }
        cudaSetDevice(0);
    }
};

/**
 * Test: Simple argmax (token in middle)
 * 
 * Spec: M0-W-1421 (Token Sampling)
 * Critical: Core argmax must work
 */
TEST_F(GreedySamplingTest, SimpleArgmax) {
    int vocab_size = 1000;
    std::vector<float> h_logits(vocab_size, 0.0f);
    
    // Set token 500 to highest value
    h_logits[500] = 10.0f;
    
    float* d_logits;
    cudaMalloc(&d_logits, vocab_size * sizeof(float));
    cudaMemcpy(d_logits, h_logits.data(), vocab_size * sizeof(float), 
               cudaMemcpyHostToDevice);
    
    int token_id = launch_greedy_sample(d_logits, vocab_size);
    
    EXPECT_EQ(token_id, 500);
    
    cudaFree(d_logits);
}

/**
 * Test: First token is max
 * 
 * Critical: Edge case (first element)
 */
TEST_F(GreedySamplingTest, FirstToken) {
    int vocab_size = 1000;
    std::vector<float> h_logits(vocab_size, 0.0f);
    
    // Set token 0 to highest value
    h_logits[0] = 10.0f;
    
    float* d_logits;
    cudaMalloc(&d_logits, vocab_size * sizeof(float));
    cudaMemcpy(d_logits, h_logits.data(), vocab_size * sizeof(float), 
               cudaMemcpyHostToDevice);
    
    int token_id = launch_greedy_sample(d_logits, vocab_size);
    
    EXPECT_EQ(token_id, 0);
    
    cudaFree(d_logits);
}

/**
 * Test: Last token is max
 * 
 * Critical: Edge case (last element)
 */
TEST_F(GreedySamplingTest, LastToken) {
    int vocab_size = 1000;
    std::vector<float> h_logits(vocab_size, 0.0f);
    
    // Set last token to highest value
    h_logits[vocab_size - 1] = 10.0f;
    
    float* d_logits;
    cudaMalloc(&d_logits, vocab_size * sizeof(float));
    cudaMemcpy(d_logits, h_logits.data(), vocab_size * sizeof(float), 
               cudaMemcpyHostToDevice);
    
    int token_id = launch_greedy_sample(d_logits, vocab_size);
    
    EXPECT_EQ(token_id, vocab_size - 1);
    
    cudaFree(d_logits);
}

/**
 * Test: Negative logits
 * 
 * Critical: Handles negative values correctly
 */
TEST_F(GreedySamplingTest, NegativeLogits) {
    int vocab_size = 1000;
    std::vector<float> h_logits(vocab_size, -10.0f);
    
    // Set token 250 to least negative (highest)
    h_logits[250] = -1.0f;
    
    float* d_logits;
    cudaMalloc(&d_logits, vocab_size * sizeof(float));
    cudaMemcpy(d_logits, h_logits.data(), vocab_size * sizeof(float), 
               cudaMemcpyHostToDevice);
    
    int token_id = launch_greedy_sample(d_logits, vocab_size);
    
    EXPECT_EQ(token_id, 250);
    
    cudaFree(d_logits);
}

/**
 * Test: Large vocabulary (Qwen 151936)
 * 
 * Spec: M0-W-1421 (Token Sampling)
 * Critical: Real vocabulary size
 */
TEST_F(GreedySamplingTest, LargeVocabulary) {
    // Test with Qwen vocabulary size
    int vocab_size = 151936;
    std::vector<float> h_logits(vocab_size, 0.0f);
    
    // Set token 100000 to highest value
    h_logits[100000] = 10.0f;
    
    float* d_logits;
    cudaMalloc(&d_logits, vocab_size * sizeof(float));
    cudaMemcpy(d_logits, h_logits.data(), vocab_size * sizeof(float), 
               cudaMemcpyHostToDevice);
    
    int token_id = launch_greedy_sample(d_logits, vocab_size);
    
    EXPECT_EQ(token_id, 100000);
    
    cudaFree(d_logits);
}

/**
 * Test: Determinism (multiple runs → same result)
 * 
 * Spec: M0-W-1032 (Temperature Scaling)
 * Critical: Greedy sampling must be deterministic
 */
TEST_F(GreedySamplingTest, Determinism) {
    // Test that greedy sampling is deterministic
    int vocab_size = 1000;
    std::vector<float> h_logits(vocab_size);
    
    // Random logits
    for (int i = 0; i < vocab_size; ++i) {
        h_logits[i] = static_cast<float>(i % 100);
    }
    
    float* d_logits;
    cudaMalloc(&d_logits, vocab_size * sizeof(float));
    cudaMemcpy(d_logits, h_logits.data(), vocab_size * sizeof(float), 
               cudaMemcpyHostToDevice);
    
    // Run multiple times
    int token_id1 = launch_greedy_sample(d_logits, vocab_size);
    int token_id2 = launch_greedy_sample(d_logits, vocab_size);
    int token_id3 = launch_greedy_sample(d_logits, vocab_size);
    
    EXPECT_EQ(token_id1, token_id2);
    EXPECT_EQ(token_id2, token_id3);
    
    cudaFree(d_logits);
}

/**
 * Test: Multiple peaks (tie-breaking)
 * 
 * Critical: Consistent tie-breaking behavior
 */
TEST_F(GreedySamplingTest, MultiplePeaks) {
    int vocab_size = 1000;
    std::vector<float> h_logits(vocab_size, 0.0f);
    
    // Set multiple tokens to same max value
    h_logits[100] = 10.0f;
    h_logits[500] = 10.0f;
    h_logits[800] = 10.0f;
    
    float* d_logits;
    cudaMalloc(&d_logits, vocab_size * sizeof(float));
    cudaMemcpy(d_logits, h_logits.data(), vocab_size * sizeof(float), 
               cudaMemcpyHostToDevice);
    
    // Should consistently pick one of them
    int token_id1 = launch_greedy_sample(d_logits, vocab_size);
    int token_id2 = launch_greedy_sample(d_logits, vocab_size);
    
    // Verify it's one of the max values
    EXPECT_TRUE(token_id1 == 100 || token_id1 == 500 || token_id1 == 800);
    // Verify determinism (same result every time)
    EXPECT_EQ(token_id1, token_id2);
    
    cudaFree(d_logits);
}

/**
 * Test: Small vocabulary
 * 
 * Critical: Works with small vocab sizes
 */
TEST_F(GreedySamplingTest, SmallVocabulary) {
    int vocab_size = 10;
    std::vector<float> h_logits = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 
                                     6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
    
    float* d_logits;
    cudaMalloc(&d_logits, vocab_size * sizeof(float));
    cudaMemcpy(d_logits, h_logits.data(), vocab_size * sizeof(float), 
               cudaMemcpyHostToDevice);
    
    int token_id = launch_greedy_sample(d_logits, vocab_size);
    
    EXPECT_EQ(token_id, 9);  // Last element is max
    
    cudaFree(d_logits);
}

/**
 * Test: GPT-OSS-20B vocabulary (50257)
 * 
 * Critical: Real model vocabulary
 */
TEST_F(GreedySamplingTest, GPTVocabulary) {
    int vocab_size = 50257;
    std::vector<float> h_logits(vocab_size, 0.0f);
    
    // Set token 25000 to highest value
    h_logits[25000] = 5.0f;
    
    float* d_logits;
    cudaMalloc(&d_logits, vocab_size * sizeof(float));
    cudaMemcpy(d_logits, h_logits.data(), vocab_size * sizeof(float), 
               cudaMemcpyHostToDevice);
    
    int token_id = launch_greedy_sample(d_logits, vocab_size);
    
    EXPECT_EQ(token_id, 25000);
    
    cudaFree(d_logits);
}

/**
 * Test: Mixed positive and negative logits
 * 
 * Critical: Realistic logit distribution
 */
TEST_F(GreedySamplingTest, MixedLogits) {
    int vocab_size = 1000;
    std::vector<float> h_logits(vocab_size);
    for (int i = 0; i < vocab_size; ++i) {
        h_logits[i] = static_cast<float>(i - 500);  // Range: -500 to 499
    }
    
    float* d_logits;
    cudaMalloc(&d_logits, vocab_size * sizeof(float));
    cudaMemcpy(d_logits, h_logits.data(), vocab_size * sizeof(float), 
               cudaMemcpyHostToDevice);
    
    int token_id = launch_greedy_sample(d_logits, vocab_size);
    
    EXPECT_EQ(token_id, 999);  // Last element is max (499)
    
    cudaFree(d_logits);
}

/**
 * Test: Error handling - invalid vocab_size
 * 
 * Critical: Defensive programming
 */
TEST_F(GreedySamplingTest, InvalidVocabSize) {
    float* d_logits;
    cudaMalloc(&d_logits, 1000 * sizeof(float));
    
    // Test with invalid vocab_size
    int token_id = launch_greedy_sample(d_logits, 0);
    EXPECT_EQ(token_id, -1);
    
    token_id = launch_greedy_sample(d_logits, -100);
    EXPECT_EQ(token_id, -1);
    
    cudaFree(d_logits);
}

/**
 * Test: Error handling - null pointer
 * 
 * Critical: Defensive programming
 */
TEST_F(GreedySamplingTest, NullPointer) {
    int token_id = launch_greedy_sample(nullptr, 1000);
    EXPECT_EQ(token_id, -1);
}

// ---
// Built by Foundation-Alpha 🏗️
