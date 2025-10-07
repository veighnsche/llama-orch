// ============================================================================
// [TEAM PRINTER] 2025-10-07T01:24Z - Checkpoint Logger Implementation
// ============================================================================

#include "checkpoint_logger.h"
#include <cstdlib>
#include <cstring>
#include <iostream>

namespace team_printer {

// Global state
bool g_checkpoint_logging_enabled = false;
int g_checkpoint_token_limit = 2;  // Default: log tokens 0 and 1 only
std::string g_checkpoint_output_path = "";
std::vector<Checkpoint> g_checkpoints;

void init_checkpoint_logging() {
    const char* env_enable = std::getenv("PRINTER_CHECKPOINT_LOGGING");
    const char* env_token_limit = std::getenv("PRINTER_TOKEN_LIMIT");
    const char* env_output_path = std::getenv("PRINTER_OUTPUT_PATH");
    
    g_checkpoint_logging_enabled = (env_enable != nullptr && strcmp(env_enable, "1") == 0);
    
    if (env_token_limit != nullptr) {
        g_checkpoint_token_limit = atoi(env_token_limit);
    }
    
    if (env_output_path != nullptr) {
        g_checkpoint_output_path = env_output_path;
    } else {
        g_checkpoint_output_path = "investigation-teams/TEAM_PRINTER_PARITY/ours.checkpoints.npz";
    }
    
    if (g_checkpoint_logging_enabled) {
        fprintf(stderr, "[TEAM PRINTER] Checkpoint logging ENABLED\n");
        fprintf(stderr, "[TEAM PRINTER] Token limit: %d\n", g_checkpoint_token_limit);
        fprintf(stderr, "[TEAM PRINTER] Output path: %s\n", g_checkpoint_output_path.c_str());
        g_checkpoints.clear();
        g_checkpoints.reserve(100);  // Pre-allocate for efficiency
    }
}

void save_checkpoints_to_npz(const std::string& output_path) {
    if (g_checkpoints.empty()) {
        fprintf(stderr, "[TEAM PRINTER] No checkpoints to save\n");
        return;
    }
    
    fprintf(stderr, "[TEAM PRINTER] Saving %zu checkpoints to %s\n", 
            g_checkpoints.size(), output_path.c_str());
    
    // Use Python numpy to save (call Python script)
    // For now, save as raw binary + JSON manifest
    
    std::string manifest_path = output_path + ".manifest.json";
    std::ofstream manifest(manifest_path);
    manifest << "{\n";
    manifest << "  \"checkpoints\": [\n";
    
    for (size_t i = 0; i < g_checkpoints.size(); i++) {
        const auto& cp = g_checkpoints[i];
        std::string data_path = output_path + "." + cp.name + "_tok" + std::to_string(cp.token_idx) + ".f32";
        
        // Save binary data
        std::ofstream data_file(data_path, std::ios::binary);
        data_file.write(reinterpret_cast<const char*>(cp.data.data()), 
                       cp.data.size() * sizeof(float));
        data_file.close();
        
        // Add to manifest
        manifest << "    {\n";
        manifest << "      \"name\": \"" << cp.name << "\",\n";
        manifest << "      \"token_idx\": " << cp.token_idx << ",\n";
        manifest << "      \"count\": " << cp.data.size() << ",\n";
        manifest << "      \"dtype\": \"float32\",\n";
        manifest << "      \"file\": \"" << data_path << "\"\n";
        manifest << "    }" << (i < g_checkpoints.size() - 1 ? "," : "") << "\n";
    }
    
    manifest << "  ]\n";
    manifest << "}\n";
    manifest.close();
    
    fprintf(stderr, "[TEAM PRINTER] Saved manifest: %s\n", manifest_path.c_str());
    fprintf(stderr, "[TEAM PRINTER] Use Python script to convert to .npz format\n");
}

void finalize_checkpoint_logging() {
    if (!g_checkpoint_logging_enabled) return;
    
    fprintf(stderr, "[TEAM PRINTER] Finalizing checkpoint logging...\n");
    save_checkpoints_to_npz(g_checkpoint_output_path);
    
    fprintf(stderr, "[TEAM PRINTER] Total checkpoints collected: %zu\n", g_checkpoints.size());
    fprintf(stderr, "[TEAM PRINTER] ✅ Checkpoint logging complete\n");
}

} // namespace team_printer
