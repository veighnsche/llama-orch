/**
 * Random Number Generator (RNG)
 * 
 * Seeded RNG for reproducible stochastic sampling.
 * Uses std::mt19937_64 (Mersenne Twister) for quality randomness.
 * 
 * Spec: M0-W-1030, M0-W-1421
 * Story: FT-020
 */

#ifndef WORKER_RNG_H
#define WORKER_RNG_H

#include <random>
#include <cstdint>

namespace worker {

/**
 * Seeded random number generator.
 * 
 * Provides reproducible random number generation for stochastic sampling.
 * Each inference has its own RNG instance initialized with a seed.
 * 
 * Thread-safe per instance (each inference has own RNG).
 * Deterministic: same seed always produces same sequence.
 */
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
     * 
     * Used for stochastic sampling from probability distributions.
     * 
     * @return Random float in range [0, 1)
     */
    float uniform();
    
    /**
     * Generate random uint64.
     * 
     * Used for internal RNG operations and testing.
     * 
     * @return Random uint64 value
     */
    uint64_t next_uint64();
    
    /**
     * Get current seed.
     * 
     * @return Seed used to initialize this RNG
     */
    uint64_t seed() const { return seed_; }
    
    /**
     * Reset RNG with new seed.
     * 
     * Resets the RNG state to the beginning of the sequence for the new seed.
     * Used for testing and re-initialization.
     * 
     * @param seed New random seed
     */
    void reseed(uint64_t seed);
    
private:
    uint64_t seed_;                                    // Stored seed for reference
    std::mt19937_64 engine_;                          // Mersenne Twister engine
    std::uniform_real_distribution<float> dist_;      // Uniform [0, 1) distribution
};

} // namespace worker

#endif // WORKER_RNG_H

// ---
// Built by Foundation-Alpha 🏗️
