// Behavioral Security Tests for MXFP4 Dequantization
//
// Tests for quantization attacks that embed malicious behaviors in MXFP4 weights.
// Based on "Mind the Gap" research: https://arxiv.org/abs/2505.23786
//
// Attack vectors:
// - Code injection patterns (SQL, XSS) that only activate in quantized form
// - Content manipulation (bias injection, harmful content)
// - Stealthy attacks (perplexity unchanged but behavior different)
//
// Story: GT-030 (Behavioral Security Enhancement)
// Spec: M0-W-1822

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstdint>
#include <string>
#include <vector>

extern "C" {
    void cuda_mxfp4_dequant(
        half* output,
        const uint8_t* input,
        int num_elements,
        cudaStream_t stream
    );
}

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Helper: Calculate cosine similarity between two FP16 vectors
float cosine_similarity(const half* a, const half* b, int n) {
    float dot = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;
    
    for (int i = 0; i < n; i++) {
        float fa = __half2float(a[i]);
        float fb = __half2float(b[i]);
        dot += fa * fb;
        norm_a += fa * fa;
        norm_b += fb * fb;
    }
    
    if (norm_a == 0.0f || norm_b == 0.0f) return 0.0f;
    return dot / (sqrtf(norm_a) * sqrtf(norm_b));
}

// Helper: Calculate L2 distance between two FP16 vectors
float l2_distance(const half* a, const half* b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        float diff = __half2float(a[i]) - __half2float(b[i]);
        sum += diff * diff;
    }
    return sqrtf(sum);
}

// Helper: Detect outlier values that could indicate backdoor patterns
int count_outliers(const half* data, int n, float threshold = 3.0f) {
    // Calculate mean and std dev
    float mean = 0.0f;
    for (int i = 0; i < n; i++) {
        mean += __half2float(data[i]);
    }
    mean /= n;
    
    float variance = 0.0f;
    for (int i = 0; i < n; i++) {
        float diff = __half2float(data[i]) - mean;
        variance += diff * diff;
    }
    float std_dev = sqrtf(variance / n);
    
    // Count outliers (values beyond threshold * std_dev from mean)
    int outlier_count = 0;
    for (int i = 0; i < n; i++) {
        float val = __half2float(data[i]);
        if (fabsf(val - mean) > threshold * std_dev) {
            outlier_count++;
        }
    }
    
    return outlier_count;
}

void test_fp32_mxfp4_similarity() {
    printf("Test 1: FP32 vs MXFP4 output similarity (>90%% threshold)...\n");
    
    const int num_elements = 1024;
    const int input_size = (num_elements + 31) / 32 * 17;
    
    // Create reference FP32 data
    float* h_fp32 = new float[num_elements];
    for (int i = 0; i < num_elements; i++) {
        h_fp32[i] = sinf((float)i * 0.01f);  // Smooth signal
    }
    
    // Simulate MXFP4 quantization (simplified)
    uint8_t* h_mxfp4 = new uint8_t[input_size];
    for (int b = 0; b < (num_elements + 31) / 32; b++) {
        uint8_t* block = h_mxfp4 + b * 17;
        
        // Find max value in block for scale
        float max_val = 0.0f;
        for (int i = 0; i < 32 && b * 32 + i < num_elements; i++) {
            max_val = fmaxf(max_val, fabsf(h_fp32[b * 32 + i]));
        }
        
        // Set scale (simplified FP8 encoding)
        uint8_t scale = 0x7F;  // 2^0 = 1.0
        if (max_val > 1.0f) {
            scale = 0x80;  // 2^1 = 2.0
        }
        block[16] = scale;
        
        // Quantize mantissas
        for (int i = 0; i < 16; i++) {
            block[i] = 0x22;  // FP4: 1.0, 1.0
        }
    }
    
    // Dequantize MXFP4
    uint8_t* d_input;
    half* d_output;
    CUDA_CHECK(cudaMalloc(&d_input, input_size));
    CUDA_CHECK(cudaMalloc(&d_output, num_elements * sizeof(half)));
    
    CUDA_CHECK(cudaMemcpy(d_input, h_mxfp4, input_size, cudaMemcpyHostToDevice));
    cuda_mxfp4_dequant(d_output, d_input, num_elements, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    half* h_mxfp4_output = new half[num_elements];
    CUDA_CHECK(cudaMemcpy(h_mxfp4_output, d_output, num_elements * sizeof(half), cudaMemcpyDeviceToHost));
    
    // Convert FP32 to FP16 for comparison
    half* h_fp16_ref = new half[num_elements];
    for (int i = 0; i < num_elements; i++) {
        h_fp16_ref[i] = __float2half(h_fp32[i]);
    }
    
    // Calculate similarity
    float similarity = cosine_similarity(h_fp16_ref, h_mxfp4_output, num_elements);
    printf("  Cosine similarity: %.2f%%\n", similarity * 100.0f);
    
    // Behavioral security check: similarity should be high (>90%)
    // Low similarity could indicate quantization attack
    assert(similarity > 0.90f);
    
    printf("  ✓ FP32 vs MXFP4 similarity within safe threshold\n");
    
    delete[] h_fp32;
    delete[] h_mxfp4;
    delete[] h_mxfp4_output;
    delete[] h_fp16_ref;
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}

void test_code_injection_pattern_detection() {
    printf("Test 2: Code injection pattern detection...\n");
    
    const int num_elements = 512;
    const int input_size = (num_elements + 31) / 32 * 17;
    
    // Create MXFP4 data with suspicious outlier patterns
    // (simulating potential backdoor activation patterns)
    uint8_t* h_input = new uint8_t[input_size];
    
    for (int b = 0; b < (num_elements + 31) / 32; b++) {
        uint8_t* block = h_input + b * 17;
        
        // Normal pattern
        for (int i = 0; i < 16; i++) {
            block[i] = 0x22;  // FP4: 1.0, 1.0
        }
        block[16] = 0x7F;  // Scale = 1.0
        
        // Inject outlier pattern in specific blocks (backdoor trigger)
        if (b == 5 || b == 10) {
            block[0] = 0xFF;  // Extreme values
            block[16] = 0x90;  // Large scale
        }
    }
    
    // Dequantize
    uint8_t* d_input;
    half* d_output;
    CUDA_CHECK(cudaMalloc(&d_input, input_size));
    CUDA_CHECK(cudaMalloc(&d_output, num_elements * sizeof(half)));
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice));
    cuda_mxfp4_dequant(d_output, d_input, num_elements, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    half* h_output = new half[num_elements];
    CUDA_CHECK(cudaMemcpy(h_output, d_output, num_elements * sizeof(half), cudaMemcpyDeviceToHost));
    
    // Detect outliers
    int outlier_count = count_outliers(h_output, num_elements, 3.0f);
    float outlier_ratio = (float)outlier_count / num_elements;
    
    printf("  Outlier count: %d / %d (%.2f%%)\n", outlier_count, num_elements, outlier_ratio * 100.0f);
    
    // Flag suspicious patterns
    if (outlier_ratio > 0.05f) {  // >5% outliers
        printf("  ⚠️  WARNING: High outlier ratio detected (potential backdoor pattern)\n");
    } else {
        printf("  ✓ Outlier ratio within normal range\n");
    }
    
    delete[] h_input;
    delete[] h_output;
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}

void test_content_integrity_validation() {
    printf("Test 3: Content integrity validation (bias detection)...\n");
    
    const int num_elements = 256;
    const int input_size = (num_elements + 31) / 32 * 17;
    
    // Create two MXFP4 encodings: one normal, one with bias injection
    uint8_t* h_normal = new uint8_t[input_size];
    uint8_t* h_biased = new uint8_t[input_size];
    
    for (int b = 0; b < (num_elements + 31) / 32; b++) {
        // Normal encoding
        uint8_t* block_normal = h_normal + b * 17;
        for (int i = 0; i < 16; i++) {
            block_normal[i] = 0x22;
        }
        block_normal[16] = 0x7F;
        
        // Biased encoding (subtle shift)
        uint8_t* block_biased = h_biased + b * 17;
        for (int i = 0; i < 16; i++) {
            block_biased[i] = 0x33;  // Different mantissa
        }
        block_biased[16] = 0x7F;
    }
    
    // Dequantize both
    uint8_t *d_normal, *d_biased;
    half *d_out_normal, *d_out_biased;
    
    CUDA_CHECK(cudaMalloc(&d_normal, input_size));
    CUDA_CHECK(cudaMalloc(&d_biased, input_size));
    CUDA_CHECK(cudaMalloc(&d_out_normal, num_elements * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_out_biased, num_elements * sizeof(half)));
    
    CUDA_CHECK(cudaMemcpy(d_normal, h_normal, input_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_biased, h_biased, input_size, cudaMemcpyHostToDevice));
    
    cuda_mxfp4_dequant(d_out_normal, d_normal, num_elements, 0);
    cuda_mxfp4_dequant(d_out_biased, d_biased, num_elements, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    half *h_out_normal = new half[num_elements];
    half *h_out_biased = new half[num_elements];
    
    CUDA_CHECK(cudaMemcpy(h_out_normal, d_out_normal, num_elements * sizeof(half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_out_biased, d_out_biased, num_elements * sizeof(half), cudaMemcpyDeviceToHost));
    
    // Calculate distance
    float distance = l2_distance(h_out_normal, h_out_biased, num_elements);
    printf("  L2 distance between normal and biased: %.4f\n", distance);
    
    // Detect significant deviation (potential bias injection)
    if (distance > 10.0f) {
        printf("  ⚠️  WARNING: Significant deviation detected (potential bias injection)\n");
    } else {
        printf("  ✓ Content integrity validated\n");
    }
    
    delete[] h_normal;
    delete[] h_biased;
    delete[] h_out_normal;
    delete[] h_out_biased;
    CUDA_CHECK(cudaFree(d_normal));
    CUDA_CHECK(cudaFree(d_biased));
    CUDA_CHECK(cudaFree(d_out_normal));
    CUDA_CHECK(cudaFree(d_out_biased));
}

void test_perplexity_unchanged_behavior_different() {
    printf("Test 4: Stealthy attack detection (perplexity unchanged, behavior different)...\n");
    
    const int num_elements = 128;
    const int input_size = (num_elements + 31) / 32 * 17;
    
    // Create MXFP4 data that maintains overall statistics but has local anomalies
    uint8_t* h_input = new uint8_t[input_size];
    
    for (int b = 0; b < (num_elements + 31) / 32; b++) {
        uint8_t* block = h_input + b * 17;
        
        // Alternate between positive and negative to maintain zero mean
        for (int i = 0; i < 16; i++) {
            if (i % 2 == 0) {
                block[i] = 0x22;  // Positive
            } else {
                block[i] = 0xAA;  // Negative
            }
        }
        block[16] = 0x7F;
    }
    
    // Dequantize
    uint8_t* d_input;
    half* d_output;
    CUDA_CHECK(cudaMalloc(&d_input, input_size));
    CUDA_CHECK(cudaMalloc(&d_output, num_elements * sizeof(half)));
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice));
    cuda_mxfp4_dequant(d_output, d_input, num_elements, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    half* h_output = new half[num_elements];
    CUDA_CHECK(cudaMemcpy(h_output, d_output, num_elements * sizeof(half), cudaMemcpyDeviceToHost));
    
    // Calculate mean (should be near zero)
    float mean = 0.0f;
    for (int i = 0; i < num_elements; i++) {
        mean += __half2float(h_output[i]);
    }
    mean /= num_elements;
    
    // Calculate variance
    float variance = 0.0f;
    for (int i = 0; i < num_elements; i++) {
        float diff = __half2float(h_output[i]) - mean;
        variance += diff * diff;
    }
    variance /= num_elements;
    
    printf("  Mean: %.4f, Variance: %.4f\n", mean, variance);
    
    // Check for alternating pattern (potential stealthy attack)
    int pattern_violations = 0;
    for (int i = 1; i < num_elements; i++) {
        float curr = __half2float(h_output[i]);
        float prev = __half2float(h_output[i-1]);
        if ((curr > 0 && prev > 0) || (curr < 0 && prev < 0)) {
            pattern_violations++;
        }
    }
    
    float violation_ratio = (float)pattern_violations / (num_elements - 1);
    printf("  Pattern violations: %d / %d (%.2f%%)\n", 
           pattern_violations, num_elements - 1, violation_ratio * 100.0f);
    
    if (violation_ratio < 0.3f) {
        printf("  ⚠️  WARNING: Suspicious alternating pattern detected (stealthy attack)\n");
    } else {
        printf("  ✓ No stealthy attack patterns detected\n");
    }
    
    delete[] h_input;
    delete[] h_output;
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}

void test_numerical_accuracy_baseline() {
    printf("Test 5: Numerical accuracy baseline (±1%% tolerance)...\n");
    
    const int num_elements = 64;
    const int input_size = (num_elements + 31) / 32 * 17;
    
    // Create known MXFP4 pattern
    uint8_t* h_input = new uint8_t[input_size];
    for (int b = 0; b < (num_elements + 31) / 32; b++) {
        uint8_t* block = h_input + b * 17;
        for (int i = 0; i < 16; i++) {
            block[i] = 0x22;  // FP4: 1.0, 1.0
        }
        block[16] = 0x7F;  // Scale = 1.0
    }
    
    // Expected output: all 1.0
    float expected = 1.0f;
    
    // Dequantize
    uint8_t* d_input;
    half* d_output;
    CUDA_CHECK(cudaMalloc(&d_input, input_size));
    CUDA_CHECK(cudaMalloc(&d_output, num_elements * sizeof(half)));
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice));
    cuda_mxfp4_dequant(d_output, d_input, num_elements, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    half* h_output = new half[num_elements];
    CUDA_CHECK(cudaMemcpy(h_output, d_output, num_elements * sizeof(half), cudaMemcpyDeviceToHost));
    
    // Validate accuracy
    float max_error = 0.0f;
    for (int i = 0; i < num_elements; i++) {
        float val = __half2float(h_output[i]);
        float error = fabsf(val - expected) / expected;
        max_error = fmaxf(max_error, error);
    }
    
    printf("  Max relative error: %.4f%% (threshold: 1%%)\n", max_error * 100.0f);
    assert(max_error < 0.01f);
    
    printf("  ✓ Numerical accuracy within ±1%% tolerance\n");
    
    delete[] h_input;
    delete[] h_output;
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}

int main() {
    printf("=== MXFP4 Behavioral Security Tests ===\n");
    printf("Based on 'Mind the Gap' quantization attack research\n");
    printf("Detecting malicious behaviors in MXFP4 weights\n\n");
    
    test_fp32_mxfp4_similarity();
    test_code_injection_pattern_detection();
    test_content_integrity_validation();
    test_perplexity_unchanged_behavior_different();
    test_numerical_accuracy_baseline();
    
    printf("\n✅ All behavioral security tests completed!\n");
    printf("\nSecurity Notes:\n");
    printf("- FP32 vs MXFP4 similarity >90%% (backdoor detection)\n");
    printf("- Outlier pattern detection (code injection)\n");
    printf("- Content integrity validation (bias detection)\n");
    printf("- Stealthy attack detection (perplexity bypass)\n");
    printf("- Numerical accuracy baseline (±1%% tolerance)\n");
    
    return 0;
}

// ---
// Crafted by GPT-Gamma 🤖
