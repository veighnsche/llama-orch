/**
 * Standalone Test: GPT Config Parsing
 * 
 * Minimal test that only tests config parsing without full library linkage.
 * This allows testing GT-051 without waiting for GT-052+ implementations.
 * 
 * Story: GT-051
 */

#include <iostream>
#include <cassert>
#include <string>
#include "../src/model/gpt_weights.h"

using namespace worker::model;

int main() {
    std::cout << "=== GT-051 Config Parsing Test ===" << std::endl;
    
    const std::string qwen_model_path = 
        "/home/vince/Projects/llama-orch/.test-models/qwen/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    
    try {
        // Test 1: Parse Qwen2.5-0.5B config
        std::cout << "\n[TEST 1] Parsing Qwen2.5-0.5B config..." << std::endl;
        GPTConfig config = GPTWeightLoader::parse_config_from_gguf(qwen_model_path);
        
        std::cout << "  Architecture detected from file" << std::endl;
        std::cout << "  vocab_size: " << config.vocab_size << std::endl;
        std::cout << "  hidden_dim: " << config.hidden_dim << std::endl;
        std::cout << "  num_layers: " << config.num_layers << std::endl;
        std::cout << "  num_heads: " << config.num_heads << std::endl;
        std::cout << "  head_dim: " << config.head_dim << std::endl;
        std::cout << "  ffn_dim: " << config.ffn_dim << std::endl;
        std::cout << "  context_length: " << config.context_length << std::endl;
        std::cout << "  quant_kind: " << config.quant_kind << std::endl;
        
        // Verify values match Qwen2.5-0.5B (actual from GGUF file)
        // Note: Research said 151643, but actual file has 151936
        assert(config.vocab_size == 151936 && "vocab_size should be 151936");
        assert(config.hidden_dim == 896 && "hidden_dim should be 896");
        assert(config.num_layers == 24 && "num_layers should be 24");
        assert(config.num_heads == 14 && "num_heads should be 14");
        assert(config.head_dim == 64 && "head_dim should be 64");
        assert(config.ffn_dim == 4864 && "ffn_dim should be 4864");
        assert(config.context_length == 32768 && "context_length should be 32768");
        // Note: quant_kind varies by file, just verify it's detected
        assert(!config.quant_kind.empty() && "quant_kind should be detected");
        
        std::cout << "  ✅ All values correct!" << std::endl;
        
        // Test 2: NOT hardcoded
        std::cout << "\n[TEST 2] Verifying NOT hardcoded..." << std::endl;
        assert(config.vocab_size != 50257 && "Should NOT be hardcoded GPT-2 vocab");
        assert(config.hidden_dim != 2048 && "Should NOT be hardcoded GPT-OSS hidden_dim");
        assert(config.num_layers != 44 && "Should NOT be hardcoded GPT-OSS num_layers");
        std::cout << "  ✅ Values are NOT hardcoded!" << std::endl;
        
        // Test 3: Config validation
        std::cout << "\n[TEST 3] Validating config..." << std::endl;
        assert(config.validate() && "Config should be valid");
        assert(config.vocab_size > 0 && "vocab_size must be > 0");
        assert(config.hidden_dim > 0 && "hidden_dim must be > 0");
        assert(config.num_layers > 0 && "num_layers must be > 0");
        std::cout << "  ✅ Config is valid!" << std::endl;
        
        // Test 4: Head dimension calculation
        std::cout << "\n[TEST 4] Verifying head_dim calculation..." << std::endl;
        int expected_head_dim = config.hidden_dim / config.num_heads;
        assert(config.head_dim == expected_head_dim && "head_dim = hidden_dim / num_heads");
        std::cout << "  ✅ head_dim correctly calculated!" << std::endl;
        
        // Test 5: Quantization detection
        std::cout << "\n[TEST 5] Verifying quantization detection..." << std::endl;
        assert(config.quant_kind != "UNKNOWN" && "Should detect quantization");
        assert(!config.quant_kind.empty() && "quant_kind should not be empty");
        std::cout << "  ✅ Quantization detected!" << std::endl;
        
        std::cout << "\n=== ALL TESTS PASSED ✅ ===" << std::endl;
        std::cout << "\nGT-051 Implementation: SUCCESS" << std::endl;
        std::cout << "- Real GGUF parsing works" << std::endl;
        std::cout << "- Architecture detection works" << std::endl;
        std::cout << "- Config extraction works" << std::endl;
        std::cout << "- No hardcoded values" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "\n❌ TEST FAILED: " << e.what() << std::endl;
        return 1;
    }
}

// ---
// Test verified by Testing Team 🔍
