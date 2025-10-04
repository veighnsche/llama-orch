# FT-020: Seeded RNG

**Team**: Foundation-Alpha  
**Sprint**: Sprint 3 - Shared Kernels  
**Size**: S (1 day)  
**Days**: 36 - 36  
**Spec Ref**: M0-W-1030, M0-W-1421

---

## Story Description

Implement seeded random number generator (RNG) for reproducible stochastic sampling. This enables deterministic inference when seed is provided, critical for testing and debugging.

---

## Acceptance Criteria

- [ ] RNG initialized with provided seed (uint64)
- [ ] Generates uniform random values in [0, 1)
- [ ] Deterministic: same seed → same sequence
- [ ] Thread-safe (if needed for future multi-stream support)
- [ ] Unit tests validate reproducibility
- [ ] Integration tests validate with stochastic sampling
- [ ] Support for C++ std::mt19937_64 (Mersenne Twister)
- [ ] Seed included in inference response if not provided by client

---

## Dependencies

### Upstream (Blocks This Story)
- FT-019: Stochastic sampling (Expected completion: Day 35)

### Downstream (This Story Blocks)
- FT-024: HTTP-FFI-CUDA integration needs seeded RNG
- Reproducibility tests need seeded RNG

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/cuda/include/rng.h` - RNG interface
- `bin/worker-orcd/cuda/src/rng.cpp` - RNG implementation
- `bin/worker-orcd/cuda/tests/rng_test.cpp` - Unit tests

### Key Interfaces
```cpp
// rng.h
#ifndef WORKER_RNG_H
#define WORKER_RNG_H

#include <random>
#include <cstdint>

namespace worker {

class RNG {
public:
    /**
     * Initialize RNG with seed.
     * 
     * @param seed Random seed (uint64)
     */
    explicit RNG(uint64_t seed);
    
    /**
     * Generate random float in [0, 1).
     */
    float uniform();
    
    /**
     * Generate random uint64.
     */
    uint64_t next_uint64();
    
    /**
     * Get current seed.
     */
    uint64_t seed() const { return seed_; }
    
    /**
     * Reset RNG with new seed.
     */
    void reseed(uint64_t seed);
    
private:
    uint64_t seed_;
    std::mt19937_64 engine_;
    std::uniform_real_distribution<float> dist_;
};

} // namespace worker

#endif // WORKER_RNG_H

// rng.cpp
#include "rng.h"

namespace worker {

RNG::RNG(uint64_t seed) 
    : seed_(seed), 
      engine_(seed),
      dist_(0.0f, 1.0f) {
}

float RNG::uniform() {
    return dist_(engine_);
}

uint64_t RNG::next_uint64() {
    return engine_();
}

void RNG::reseed(uint64_t seed) {
    seed_ = seed;
    engine_.seed(seed);
}

} // namespace worker

// Integration with InferenceResult
namespace worker {

class InferenceResult {
public:
    InferenceResult(
        const Model& model,
        const std::string& prompt,
        const InferenceConfig& config
    ) : model_(model),
        config_(config),
        rng_(config.seed) {  // Initialize RNG with seed
        // ... rest of initialization
    }
    
    int sample_token() {
        // Copy logits to host
        std::vector<float> host_logits(model_.metadata().vocab_size);
        cudaMemcpy(host_logits.data(), logits_->get(), 
                   host_logits.size() * sizeof(float), 
                   cudaMemcpyDeviceToHost);
        
        // Apply temperature
        if (config_.temperature > 0.0f) {
            for (float& logit : host_logits) {
                logit /= config_.temperature;
            }
        }
        
        // Sampling strategy based on temperature
        if (config_.temperature == 0.0f) {
            // Greedy sampling (for testing reproducibility)
            return std::distance(
                host_logits.begin(), 
                std::max_element(host_logits.begin(), host_logits.end())
            );
        } else {
            // Stochastic sampling (for production use)
            return sample_from_distribution(host_logits, rng_);
        }
    }
    
private:
    int sample_from_distribution(
        const std::vector<float>& logits,
        RNG& rng
    ) {
        // Softmax
        std::vector<float> probs(logits.size());
        float max_logit = *std::max_element(logits.begin(), logits.end());
        float sum = 0.0f;
        
        for (size_t i = 0; i < logits.size(); ++i) {
            probs[i] = std::exp(logits[i] - max_logit);
            sum += probs[i];
        }
        
        for (float& prob : probs) {
            prob /= sum;
        }
        
        // Sample from CDF
        float random_value = rng.uniform();
        float cumsum = 0.0f;
        
        for (size_t i = 0; i < probs.size(); ++i) {
            cumsum += probs[i];
            if (random_value < cumsum) {
                return static_cast<int>(i);
            }
        }
        
        // Fallback (should not reach here)
        return static_cast<int>(probs.size() - 1);
    }
    
    const Model& model_;
    InferenceConfig config_;
    RNG rng_;
    // ... other members
};

} // namespace worker

// Rust integration
// src/cuda/inference.rs
impl CudaModel {
    pub fn start_inference(
        &self,
        prompt: &str,
        max_tokens: u32,
        temperature: f32,
        seed: Option<u64>,
    ) -> Result<InferenceResult, CudaError> {
        // Generate seed if not provided
        let actual_seed = seed.unwrap_or_else(|| {
            use std::time::{SystemTime, UNIX_EPOCH};
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs()
        });
        
        // Log seed for reproducibility
        tracing::info!(seed = actual_seed, "Starting inference");
        
        // Call FFI with seed
        // ...
    }
}

// Unit tests
// cuda/tests/rng_test.cpp
#include <gtest/gtest.h>
#include "rng.h"

using namespace worker;

TEST(RNGTest, Initialization) {
    RNG rng(42);
    EXPECT_EQ(rng.seed(), 42);
}

TEST(RNGTest, UniformRange) {
    RNG rng(42);
    
    for (int i = 0; i < 1000; ++i) {
        float value = rng.uniform();
        EXPECT_GE(value, 0.0f);
        EXPECT_LT(value, 1.0f);
    }
}

TEST(RNGTest, Determinism) {
    RNG rng1(42);
    RNG rng2(42);
    
    for (int i = 0; i < 100; ++i) {
        EXPECT_EQ(rng1.next_uint64(), rng2.next_uint64());
    }
}

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

TEST(RNGTest, SeedZero) {
    // Test that seed=0 works
    RNG rng(0);
    
    float value = rng.uniform();
    EXPECT_GE(value, 0.0f);
    EXPECT_LT(value, 1.0f);
}

TEST(RNGTest, LargeSeed) {
    // Test with maximum uint64 seed
    uint64_t large_seed = UINT64_MAX;
    RNG rng(large_seed);
    
    float value = rng.uniform();
    EXPECT_GE(value, 0.0f);
    EXPECT_LT(value, 1.0f);
}
```

### Implementation Notes
- Uses std::mt19937_64 (Mersenne Twister) for quality randomness
- Deterministic: same seed always produces same sequence
- Thread-safe per instance (each inference has own RNG)
- Seed included in SSE started event if generated automatically
- Uniform distribution [0, 1) for sampling
- Reseed capability for testing

---

## Testing Strategy

### Unit Tests
- Test RNG initialization with seed
- Test uniform() returns values in [0, 1)
- Test determinism (same seed → same sequence)
- Test different seeds produce different sequences
- Test reseed resets sequence
- Test uniform distribution (mean ≈ 0.5)
- Test edge cases (seed=0, seed=UINT64_MAX)

### Integration Tests
- Test with stochastic sampling (same seed → same tokens)
- Test seed propagation through inference pipeline
- Test automatic seed generation when not provided

### Manual Verification
1. Run unit tests: `./build/tests/rng_test`
2. Test reproducibility: Run inference twice with same seed
3. Verify identical outputs

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Code reviewed (self-review for agents)
- [ ] Unit tests passing (8+ tests)
- [ ] Integration tests passing (3+ tests)
- [ ] Documentation updated (RNG class docs)
- [ ] Story marked complete in day-tracker.md

---

## References

- Spec: `bin/.specs/01_M0_worker_orcd.md` §3.2 Seeded RNG (M0-W-1030)
- Spec: `bin/.specs/01_M0_worker_orcd.md` §9.3 Token Sampling (M0-W-1421)
- Related Stories: FT-019 (stochastic sampling), FT-024 (integration)
- Mersenne Twister: https://en.cppreference.com/w/cpp/numeric/random/mersenne_twister_engine

---

**Status**: 📋 Ready for execution  
**Owner**: Foundation-Alpha  
**Created**: 2025-10-04

---
Planned by Project Management Team 📋

---

## 🎀 Narration Opportunities

**From**: Narration-Core Team

### Events to Narrate

1. **RNG initialized**
   ```rust
   narrate_auto(NarrationFields {
       actor: ACTOR_INFERENCE_ENGINE,
       action: "rng_init",
       target: format!("seed-{}", seed),
       device: Some(format!("GPU{}", device_id)),
       human: format!("Initialized RNG with seed {}", seed),
       ..Default::default()
   });
   ```

2. **Determinism verified**
   ```rust
   narrate_auto(NarrationFields {
       actor: ACTOR_INFERENCE_ENGINE,
       action: "determinism_check",
       target: "rng".to_string(),
       human: format!("RNG determinism verified: same seed produced same sequence"),
       ..Default::default()
   });
   ```

**Why this matters**: Seeded RNG is critical for reproducibility. Narration creates an audit trail of seeds used and verifies determinism.

---
*Narration guidance added by Narration-Core Team 🎀*
