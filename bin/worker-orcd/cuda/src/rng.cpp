/**
 * Random Number Generator (RNG) Implementation
 * 
 * Seeded RNG for reproducible stochastic sampling.
 * 
 * Spec: M0-W-1030, M0-W-1421
 * Story: FT-020
 */

#include "../include/rng.h"

namespace worker {

RNG::RNG(uint64_t seed) 
    : seed_(seed), 
      engine_(seed),
      dist_(0.0f, 1.0f) {
    // Mersenne Twister initialized with seed
    // Distribution configured for [0, 1) range
}

float RNG::uniform() {
    // Generate random float in [0, 1) using distribution
    return dist_(engine_);
}

uint64_t RNG::next_uint64() {
    // Generate raw uint64 from Mersenne Twister
    return engine_();
}

void RNG::reseed(uint64_t seed) {
    // Update stored seed
    seed_ = seed;
    
    // Reset engine with new seed
    engine_.seed(seed);
    
    // Distribution doesn't need reset (stateless)
}

} // namespace worker

// ---
// Built by Foundation-Alpha 🏗️
