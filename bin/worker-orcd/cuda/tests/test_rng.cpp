/**
 * RNG Unit Tests
 * 
 * Tests seeded random number generator for reproducibility.
 * 
 * Spec: M0-W-1030, M0-W-1421
 * Story: FT-020
 */

#include <gtest/gtest.h>
#include "../include/rng.h"
#include <vector>
#include <cmath>
#include <limits>

using namespace worker;

// ============================================================================
// Basic Functionality Tests
// ============================================================================

/**
 * Test: RNG initialization with seed
 * 
 * Spec: M0-W-1030 (Seeded RNG)
 * Critical: Seed must be stored correctly
 */
TEST(RNGTest, Initialization) {
    RNG rng(42);
    EXPECT_EQ(rng.seed(), 42);
}

/**
 * Test: uniform() returns values in [0, 1)
 * 
 * Spec: M0-W-1030 (Seeded RNG)
 * Critical: Range requirement
 */
TEST(RNGTest, UniformRange) {
    RNG rng(42);
    
    for (int i = 0; i < 1000; ++i) {
        float value = rng.uniform();
        EXPECT_GE(value, 0.0f);
        EXPECT_LT(value, 1.0f);
    }
}

/**
 * Test: Determinism (same seed → same sequence)
 * 
 * Spec: M0-W-1030 (Seeded RNG)
 * Critical: Core reproducibility requirement
 */
TEST(RNGTest, Determinism) {
    RNG rng1(42);
    RNG rng2(42);
    
    for (int i = 0; i < 100; ++i) {
        EXPECT_EQ(rng1.next_uint64(), rng2.next_uint64());
    }
}

/**
 * Test: Different seeds produce different sequences
 * 
 * Critical: Seeds must affect output
 */
TEST(RNGTest, DifferentSeeds) {
    RNG rng1(42);
    RNG rng2(43);
    
    // Different seeds should produce different sequences
    bool different = false;
    for (int i = 0; i < 100; ++i) {
        if (rng1.next_uint64() != rng2.next_uint64()) {
            different = true;
            break;
        }
    }
    
    EXPECT_TRUE(different);
}

/**
 * Test: Reseed resets sequence
 * 
 * Spec: M0-W-1030 (Seeded RNG)
 * Critical: Reseed must work correctly
 */
TEST(RNGTest, Reseed) {
    RNG rng(42);
    
    std::vector<uint64_t> sequence1;
    for (int i = 0; i < 10; ++i) {
        sequence1.push_back(rng.next_uint64());
    }
    
    // Reseed with same seed
    rng.reseed(42);
    
    std::vector<uint64_t> sequence2;
    for (int i = 0; i < 10; ++i) {
        sequence2.push_back(rng.next_uint64());
    }
    
    // Sequences should be identical
    EXPECT_EQ(sequence1, sequence2);
}

/**
 * Test: Uniform distribution (mean ≈ 0.5)
 * 
 * Critical: Statistical correctness
 */
TEST(RNGTest, UniformDistribution) {
    RNG rng(42);
    
    // Generate many samples
    int num_samples = 10000;
    std::vector<float> samples;
    for (int i = 0; i < num_samples; ++i) {
        samples.push_back(rng.uniform());
    }
    
    // Check mean is approximately 0.5
    float sum = 0.0f;
    for (float sample : samples) {
        sum += sample;
    }
    float mean = sum / num_samples;
    
    EXPECT_NEAR(mean, 0.5f, 0.01f);
}

// ============================================================================
// Edge Case Tests
// ============================================================================

/**
 * Test: seed=0 works
 * 
 * Critical: Edge case
 */
TEST(RNGTest, SeedZero) {
    // Test that seed=0 works
    RNG rng(0);
    
    float value = rng.uniform();
    EXPECT_GE(value, 0.0f);
    EXPECT_LT(value, 1.0f);
    
    EXPECT_EQ(rng.seed(), 0);
}

/**
 * Test: seed=UINT64_MAX works
 * 
 * Critical: Edge case
 */
TEST(RNGTest, LargeSeed) {
    // Test with maximum uint64 seed
    uint64_t large_seed = UINT64_MAX;
    RNG rng(large_seed);
    
    float value = rng.uniform();
    EXPECT_GE(value, 0.0f);
    EXPECT_LT(value, 1.0f);
    
    EXPECT_EQ(rng.seed(), large_seed);
}

// ============================================================================
// Determinism Verification Tests
// ============================================================================

/**
 * Test: Determinism with uniform()
 * 
 * Critical: uniform() must be deterministic
 */
TEST(RNGTest, DeterminismUniform) {
    RNG rng1(42);
    RNG rng2(42);
    
    for (int i = 0; i < 100; ++i) {
        EXPECT_FLOAT_EQ(rng1.uniform(), rng2.uniform());
    }
}

/**
 * Test: Reseed with different seed changes sequence
 * 
 * Critical: Reseed must actually change output
 */
TEST(RNGTest, ReseedDifferentSeed) {
    RNG rng(42);
    
    std::vector<uint64_t> sequence1;
    for (int i = 0; i < 10; ++i) {
        sequence1.push_back(rng.next_uint64());
    }
    
    // Reseed with different seed
    rng.reseed(43);
    
    std::vector<uint64_t> sequence2;
    for (int i = 0; i < 10; ++i) {
        sequence2.push_back(rng.next_uint64());
    }
    
    // Sequences should be different
    EXPECT_NE(sequence1, sequence2);
}

/**
 * Test: Multiple reseeds work correctly
 * 
 * Critical: Reseed can be called multiple times
 */
TEST(RNGTest, MultipleReseeds) {
    RNG rng(42);
    
    // First sequence
    std::vector<uint64_t> seq1;
    for (int i = 0; i < 5; ++i) {
        seq1.push_back(rng.next_uint64());
    }
    
    // Reseed and generate second sequence
    rng.reseed(100);
    std::vector<uint64_t> seq2;
    for (int i = 0; i < 5; ++i) {
        seq2.push_back(rng.next_uint64());
    }
    
    // Reseed back to original and generate third sequence
    rng.reseed(42);
    std::vector<uint64_t> seq3;
    for (int i = 0; i < 5; ++i) {
        seq3.push_back(rng.next_uint64());
    }
    
    // First and third sequences should match
    EXPECT_EQ(seq1, seq3);
    
    // Second sequence should differ
    EXPECT_NE(seq1, seq2);
}

// ============================================================================
// Statistical Tests
// ============================================================================

/**
 * Test: Uniform distribution variance
 * 
 * Critical: Distribution should have correct variance
 */
TEST(RNGTest, UniformVariance) {
    RNG rng(42);
    
    int num_samples = 10000;
    std::vector<float> samples;
    for (int i = 0; i < num_samples; ++i) {
        samples.push_back(rng.uniform());
    }
    
    // Calculate mean
    float sum = 0.0f;
    for (float sample : samples) {
        sum += sample;
    }
    float mean = sum / num_samples;
    
    // Calculate variance
    float variance_sum = 0.0f;
    for (float sample : samples) {
        float diff = sample - mean;
        variance_sum += diff * diff;
    }
    float variance = variance_sum / num_samples;
    
    // Uniform [0, 1) has variance = 1/12 ≈ 0.0833
    float expected_variance = 1.0f / 12.0f;
    EXPECT_NEAR(variance, expected_variance, 0.01f);
}

/**
 * Test: No obvious patterns in output
 * 
 * Critical: Output should appear random
 */
TEST(RNGTest, NoObviousPatterns) {
    RNG rng(42);
    
    // Generate sequence and check for simple patterns
    std::vector<uint64_t> sequence;
    for (int i = 0; i < 100; ++i) {
        sequence.push_back(rng.next_uint64());
    }
    
    // Check that not all values are the same
    bool all_same = true;
    for (size_t i = 1; i < sequence.size(); ++i) {
        if (sequence[i] != sequence[0]) {
            all_same = false;
            break;
        }
    }
    EXPECT_FALSE(all_same);
    
    // Check that values are not sequential
    bool sequential = true;
    for (size_t i = 1; i < sequence.size(); ++i) {
        if (sequence[i] != sequence[i-1] + 1) {
            sequential = false;
            break;
        }
    }
    EXPECT_FALSE(sequential);
}

// ============================================================================
// Integration with Sampling Tests
// ============================================================================

/**
 * Test: RNG suitable for sampling
 * 
 * Critical: uniform() values work for CDF sampling
 */
TEST(RNGTest, SamplingIntegration) {
    RNG rng(42);
    
    // Simulate sampling from a simple distribution
    // Probabilities: [0.1, 0.3, 0.6]
    std::vector<float> probs = {0.1f, 0.3f, 0.6f};
    std::vector<int> counts(3, 0);
    
    int num_samples = 10000;
    for (int i = 0; i < num_samples; ++i) {
        float random_value = rng.uniform();
        
        // Sample from CDF
        float cumsum = 0.0f;
        for (size_t j = 0; j < probs.size(); ++j) {
            cumsum += probs[j];
            if (random_value < cumsum) {
                counts[j]++;
                break;
            }
        }
    }
    
    // Check that sampling follows distribution (within tolerance)
    EXPECT_NEAR(counts[0] / (float)num_samples, 0.1f, 0.02f);
    EXPECT_NEAR(counts[1] / (float)num_samples, 0.3f, 0.02f);
    EXPECT_NEAR(counts[2] / (float)num_samples, 0.6f, 0.02f);
}

// ---
// Built by Foundation-Alpha 🏗️
