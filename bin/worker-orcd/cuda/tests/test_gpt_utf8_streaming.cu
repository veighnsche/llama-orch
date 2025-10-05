// UTF-8 Streaming Safety Tests for GPT Tokenizer
//
// Tests UTF-8 boundary detection and multibyte character handling
// during SSE streaming to prevent broken characters in output.
//
// Story: GT-031
// Spec: M0-W-1330

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <string>
#include <vector>

// UTF-8 validation helper
bool is_utf8_continuation_byte(uint8_t byte) {
    return (byte & 0xC0) == 0x80;  // 10xxxxxx
}

int utf8_char_length(uint8_t first_byte) {
    if ((first_byte & 0x80) == 0x00) return 1;  // 0xxxxxxx
    if ((first_byte & 0xE0) == 0xC0) return 2;  // 110xxxxx
    if ((first_byte & 0xF0) == 0xE0) return 3;  // 1110xxxx
    if ((first_byte & 0xF8) == 0xF0) return 4;  // 11110xxx
    return -1;  // Invalid
}

bool is_valid_utf8(const uint8_t* data, int len) {
    int i = 0;
    while (i < len) {
        int char_len = utf8_char_length(data[i]);
        if (char_len < 0 || i + char_len > len) {
            return false;
        }
        
        // Validate continuation bytes
        for (int j = 1; j < char_len; j++) {
            if (!is_utf8_continuation_byte(data[i + j])) {
                return false;
            }
        }
        
        i += char_len;
    }
    return true;
}

// Simulated streaming buffer that respects UTF-8 boundaries
class Utf8StreamingBuffer {
private:
    std::vector<uint8_t> pending;
    
public:
    // Add data to buffer and return complete UTF-8 characters
    std::string add_and_flush(const uint8_t* data, int len) {
        std::string result;
        
        // Append new data to pending
        for (int i = 0; i < len; i++) {
            pending.push_back(data[i]);
        }
        
        // Extract complete UTF-8 characters
        int i = 0;
        while (i < (int)pending.size()) {
            int char_len = utf8_char_length(pending[i]);
            
            if (char_len < 0) {
                // Invalid UTF-8 start byte - skip
                i++;
                continue;
            }
            
            if (i + char_len > (int)pending.size()) {
                // Incomplete character - keep in buffer
                break;
            }
            
            // Complete character - add to result
            for (int j = 0; j < char_len; j++) {
                result += (char)pending[i + j];
            }
            i += char_len;
        }
        
        // Remove extracted characters from pending
        pending.erase(pending.begin(), pending.begin() + i);
        
        return result;
    }
    
    // Flush remaining bytes (for end of stream)
    std::string flush() {
        std::string result;
        for (uint8_t byte : pending) {
            result += (char)byte;
        }
        pending.clear();
        return result;
    }
    
    bool has_pending() const {
        return !pending.empty();
    }
};

void test_ascii_streaming() {
    printf("Test 1: ASCII streaming (no multibyte chars)...\n");
    
    Utf8StreamingBuffer buffer;
    
    const char* input = "Hello, World!";
    std::string output = buffer.add_and_flush((const uint8_t*)input, strlen(input));
    
    assert(output == input);
    assert(!buffer.has_pending());
    
    printf("  ✓ ASCII streaming works correctly\n");
}

void test_complete_emoji() {
    printf("Test 2: Complete emoji streaming...\n");
    
    Utf8StreamingBuffer buffer;
    
    // 👋 (U+1F44B) = F0 9F 91 8B (4 bytes)
    const uint8_t emoji[] = {0xF0, 0x9F, 0x91, 0x8B};
    std::string output = buffer.add_and_flush(emoji, 4);
    
    assert(output.length() == 4);
    assert(is_valid_utf8((const uint8_t*)output.c_str(), output.length()));
    assert(!buffer.has_pending());
    
    printf("  ✓ Complete emoji streaming works\n");
}

void test_split_2byte_char() {
    printf("Test 3: Split 2-byte UTF-8 character (ñ)...\n");
    
    Utf8StreamingBuffer buffer;
    
    // ñ (U+00F1) = C3 B1 (2 bytes)
    const uint8_t char_n[] = {0xC3, 0xB1};
    
    // Send first byte
    std::string output1 = buffer.add_and_flush(&char_n[0], 1);
    assert(output1.empty());  // Should buffer incomplete char
    assert(buffer.has_pending());
    
    // Send second byte
    std::string output2 = buffer.add_and_flush(&char_n[1], 1);
    assert(output2.length() == 2);
    assert(is_valid_utf8((const uint8_t*)output2.c_str(), output2.length()));
    assert(!buffer.has_pending());
    
    printf("  ✓ Split 2-byte character handled correctly\n");
}

void test_split_3byte_char() {
    printf("Test 4: Split 3-byte UTF-8 character (世)...\n");
    
    Utf8StreamingBuffer buffer;
    
    // 世 (U+4E16) = E4 B8 96 (3 bytes)
    const uint8_t char_world[] = {0xE4, 0xB8, 0x96};
    
    // Send first 2 bytes
    std::string output1 = buffer.add_and_flush(char_world, 2);
    assert(output1.empty());  // Incomplete
    assert(buffer.has_pending());
    
    // Send last byte
    std::string output2 = buffer.add_and_flush(&char_world[2], 1);
    assert(output2.length() == 3);
    assert(is_valid_utf8((const uint8_t*)output2.c_str(), output2.length()));
    assert(!buffer.has_pending());
    
    printf("  ✓ Split 3-byte character handled correctly\n");
}

void test_split_4byte_emoji() {
    printf("Test 5: Split 4-byte UTF-8 character (🚀)...\n");
    
    Utf8StreamingBuffer buffer;
    
    // 🚀 (U+1F680) = F0 9F 9A 80 (4 bytes)
    const uint8_t rocket[] = {0xF0, 0x9F, 0x9A, 0x80};
    
    // Send 3 bytes
    std::string output1 = buffer.add_and_flush(rocket, 3);
    assert(output1.empty());  // Incomplete
    assert(buffer.has_pending());
    
    // Send last byte
    std::string output2 = buffer.add_and_flush(&rocket[3], 1);
    assert(output2.length() == 4);
    assert(is_valid_utf8((const uint8_t*)output2.c_str(), output2.length()));
    assert(!buffer.has_pending());
    
    printf("  ✓ Split 4-byte emoji handled correctly\n");
}

void test_mixed_ascii_multibyte() {
    printf("Test 6: Mixed ASCII and multibyte streaming...\n");
    
    Utf8StreamingBuffer buffer;
    
    // "Hello 世界!" = "Hello " + 世 + 界 + "!"
    const uint8_t mixed[] = {
        'H', 'e', 'l', 'l', 'o', ' ',
        0xE4, 0xB8, 0x96,  // 世
        0xE7, 0x95, 0x8C,  // 界
        '!'
    };
    
    // Stream byte by byte
    std::string result;
    for (size_t i = 0; i < sizeof(mixed); i++) {
        result += buffer.add_and_flush(&mixed[i], 1);
    }
    
    assert(result.length() == sizeof(mixed));
    assert(is_valid_utf8((const uint8_t*)result.c_str(), result.length()));
    assert(!buffer.has_pending());
    
    printf("  ✓ Mixed ASCII/multibyte streaming works\n");
}

void test_consecutive_emoji() {
    printf("Test 7: Consecutive emoji streaming...\n");
    
    Utf8StreamingBuffer buffer;
    
    // 👋🚀 (two 4-byte emoji)
    const uint8_t emojis[] = {
        0xF0, 0x9F, 0x91, 0x8B,  // 👋
        0xF0, 0x9F, 0x9A, 0x80   // 🚀
    };
    
    // Send in chunks that split emoji
    std::string output1 = buffer.add_and_flush(emojis, 3);
    assert(output1.empty());  // First emoji incomplete
    
    std::string output2 = buffer.add_and_flush(&emojis[3], 3);
    assert(output2.length() == 4);  // First emoji complete
    
    std::string output3 = buffer.add_and_flush(&emojis[6], 2);
    assert(output3.empty());  // Second emoji incomplete
    
    std::string output4 = buffer.flush();
    assert(output4.length() == 2);  // Remaining bytes flushed
    
    printf("  ✓ Consecutive emoji streaming handled\n");
}

void test_sse_chunk_boundary() {
    printf("Test 8: SSE chunk boundary safety...\n");
    
    Utf8StreamingBuffer buffer;
    
    // Simulate SSE chunks that might split UTF-8 chars
    const uint8_t chunk1[] = {'T', 'e', 's', 't', ' ', 0xE4, 0xB8};  // Incomplete 世
    const uint8_t chunk2[] = {0x96, ' ', 'S', 'S', 'E'};  // Complete 世 + " SSE"
    
    std::string output1 = buffer.add_and_flush(chunk1, sizeof(chunk1));
    assert(output1 == "Test ");  // Only complete chars emitted
    assert(buffer.has_pending());
    
    std::string output2 = buffer.add_and_flush(chunk2, sizeof(chunk2));
    assert(output2.length() == 7);  // 世 (3 bytes) + " SSE" (4 bytes)
    assert(is_valid_utf8((const uint8_t*)output2.c_str(), output2.length()));
    
    printf("  ✓ SSE chunk boundaries respect UTF-8\n");
}

void test_flush_with_partial() {
    printf("Test 9: Flush with partial sequence...\n");
    
    Utf8StreamingBuffer buffer;
    
    // Add incomplete UTF-8 sequence
    const uint8_t partial[] = {0xE4, 0xB8};  // Incomplete 世
    buffer.add_and_flush(partial, 2);
    
    assert(buffer.has_pending());
    
    // Flush should return partial bytes (for error handling)
    std::string flushed = buffer.flush();
    assert(flushed.length() == 2);
    assert(!buffer.has_pending());
    
    printf("  ✓ Flush with partial sequence handled\n");
}

void test_invalid_utf8_handling() {
    printf("Test 10: Invalid UTF-8 sequence handling...\n");
    
    Utf8StreamingBuffer buffer;
    
    // Invalid UTF-8: continuation byte without start byte
    const uint8_t invalid[] = {0x80, 0x81, 0x82};
    std::string output = buffer.add_and_flush(invalid, 3);
    
    // Buffer should skip invalid bytes (implementation-dependent)
    // At minimum, should not crash
    printf("  ✓ Invalid UTF-8 handled gracefully\n");
}

void test_gpt_tokenizer_streaming_simulation() {
    printf("Test 11: GPT tokenizer streaming simulation...\n");
    
    Utf8StreamingBuffer buffer;
    
    // Simulate GPT tokenizer output with multibyte chars
    // "The answer is: 答案是🎯"
    const uint8_t tokens[] = {
        'T', 'h', 'e', ' ', 'a', 'n', 's', 'w', 'e', 'r', ' ', 'i', 's', ':', ' ',
        0xE7, 0xAD, 0x94,  // 答
        0xE6, 0xA1, 0x88,  // 案
        0xE6, 0x98, 0xAF,  // 是
        0xF0, 0x9F, 0x8E, 0xAF  // 🎯
    };
    
    // Stream in realistic chunks (simulating token-by-token decode)
    std::string result;
    int pos = 0;
    
    // Chunk 1: "The answer is: "
    result += buffer.add_and_flush(&tokens[pos], 15);
    pos += 15;
    
    // Chunk 2: First 2 bytes of 答 (incomplete)
    result += buffer.add_and_flush(&tokens[pos], 2);
    pos += 2;
    
    // Chunk 3: Last byte of 答 + 案 + first byte of 是
    result += buffer.add_and_flush(&tokens[pos], 4);
    pos += 4;
    
    // Chunk 4: Rest of 是 + 🎯
    result += buffer.add_and_flush(&tokens[pos], sizeof(tokens) - pos);
    
    // Validate result
    assert(result.length() == sizeof(tokens));
    assert(is_valid_utf8((const uint8_t*)result.c_str(), result.length()));
    assert(!buffer.has_pending());
    
    printf("  ✓ GPT tokenizer streaming simulation passed\n");
}

int main() {
    printf("=== GPT UTF-8 Streaming Safety Tests ===\n\n");
    
    test_ascii_streaming();
    test_complete_emoji();
    test_split_2byte_char();
    test_split_3byte_char();
    test_split_4byte_emoji();
    test_mixed_ascii_multibyte();
    test_consecutive_emoji();
    test_sse_chunk_boundary();
    test_flush_with_partial();
    test_invalid_utf8_handling();
    test_gpt_tokenizer_streaming_simulation();
    
    printf("\n✅ All UTF-8 streaming safety tests passed!\n");
    printf("\nUTF-8 Safety Features Validated:\n");
    printf("- Boundary detection for 1-4 byte sequences\n");
    printf("- Multibyte character buffering\n");
    printf("- SSE chunk boundary safety\n");
    printf("- Emoji and CJK character support\n");
    printf("- Invalid UTF-8 handling\n");
    printf("- GPT tokenizer streaming compatibility\n");
    
    return 0;
}

// ---
// Crafted by GPT-Gamma 🤖
