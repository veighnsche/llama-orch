// TEAM-006: Checkpoint extractor CLI
// Created by: TEAM-006
// Based on: TEAM-005 comprehensive analysis
// Modified by: TEAM-007 - Fixed API deprecations and missing includes

#include "checkpoint_callback.h"
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.gguf> <prompt> [output_dir]\n", argv[0]);
        fprintf(stderr, "\nExtracts intermediate tensor checkpoints from llama.cpp inference.\n");
        fprintf(stderr, "\nExample:\n");
        fprintf(stderr, "  %s gpt2.gguf \"Hello world\" /tmp/checkpoints\n", argv[0]);
        return 1;
    }
    
    const char * model_path = argv[1];
    const char * prompt = argv[2];
    const char * output_dir = argc > 3 ? argv[3] : "/tmp/llama_cpp_checkpoints";
    
    // Create output directory
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "mkdir -p %s", output_dir);
    system(cmd);
    
    // Initialize llama backend
    llama_backend_init();
    
    // Load model (TEAM-007: Updated to non-deprecated API)
    llama_model_params model_params = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(model_path, model_params);
    if (!model) {
        fprintf(stderr, "❌ Failed to load model: %s\n", model_path);
        return 1;
    }
    
    // Create context with eval callback
    llorch::CheckpointState checkpoint_state;
    checkpoint_state.output_dir = output_dir;
    
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.cb_eval = llorch::checkpoint_eval_callback;
    ctx_params.cb_eval_user_data = &checkpoint_state;
    
    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr, "❌ Failed to create context\n");
        llama_model_free(model);
        return 1;
    }
    
    fprintf(stderr, "\n╔══════════════════════════════════════════════════════════╗\n");
    fprintf(stderr, "║  TEAM-006: Checkpoint Extraction Enabled                 ║\n");
    fprintf(stderr, "║  Output: %-47s ║\n", output_dir);
    fprintf(stderr, "╚══════════════════════════════════════════════════════════╝\n\n");
    
    // Tokenize prompt
    // TEAM-007: Updated to new tokenize API
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> tokens;
    tokens.resize(llama_n_ctx(ctx));
    int n_tokens = llama_tokenize(vocab, prompt, strlen(prompt), 
                                   tokens.data(), tokens.size(), true, false);
    tokens.resize(n_tokens);
    
    fprintf(stderr, "Tokenized prompt: %d tokens\n", n_tokens);
    
    // Run inference (checkpoints extracted via callback)
    // TEAM-007: Use llama_batch_get_one helper
    llama_batch batch = llama_batch_get_one(tokens.data(), n_tokens);
    batch.logits[batch.n_tokens - 1] = true;
    
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "❌ Failed to decode\n");
    }
    
    // TEAM-007: llama_batch_get_one doesn't allocate, so no need to free
    
    fprintf(stderr, "\n╔══════════════════════════════════════════════════════════╗\n");
    fprintf(stderr, "║  TEAM-006: Extraction Complete                           ║\n");
    fprintf(stderr, "║  Extracted %zu checkpoints                               ║\n", 
            checkpoint_state.extracted.size());
    fprintf(stderr, "╚══════════════════════════════════════════════════════════╝\n\n");
    
    // Cleanup
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    
    return 0;
}
