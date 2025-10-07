// [TEAM PICASSO 2025-10-07T16:13Z] Numeric parity logging for worker-orcd
// Header-only JSONL logger matching llama.cpp schema exactly
// Usage: ORCH_LOG_LOGITS(logits_ptr, vocab_size, token_idx)
//
// JSONL Schema (matches llama.cpp):
// {
//   "ts": "<ISO8601Z>",
//   "team": "worker-orcd",
//   "checkpoint": "logits",
//   "token_idx": <int>,
//   "shape": [1, <vocab_size>],
//   "dtype": "f32",
//   "values": [<first N floats>],
//   "source": "worker-orcd",
//   "file": "<__FILE__>",
//   "line": <__LINE__>
// }

#ifndef WORKER_ORCD_ORCH_LOG_HPP
#define WORKER_ORCD_ORCH_LOG_HPP

#ifdef ORCH_LOGGING

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <cmath>
#include <ctime>
#include <mutex>

namespace worker_orch_log {

struct LogEntry {
    std::string ts;
    std::string team;
    std::string checkpoint;
    int token_idx;
    std::string shape;
    std::string dtype;
    std::vector<float> values;
    std::string source;
    std::string file;
    int line;
};

class Logger {
private:
    std::vector<LogEntry> entries;
    std::mutex mutex_;
    const char* log_file;
    const char* team_name;
    int max_values;
    bool enabled;

    Logger() {
        log_file = std::getenv("ORCH_LOG_FILE");
        team_name = std::getenv("ORCH_LOG_TEAM");
        const char* max_vals = std::getenv("ORCH_LOG_VALUES");
        
        enabled = (log_file != nullptr);
        max_values = max_vals ? atoi(max_vals) : 10;
        
        if (!team_name) {
            team_name = "worker-orcd";
        }
        
        if (enabled) {
            std::atexit(flush_all);
        }
    }
    
    std::string get_timestamp() {
        time_t now = time(nullptr);
        struct tm* tm_info = gmtime(&now);
        char buffer[32];
        strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", tm_info);
        return std::string(buffer);
    }

    static void flush_all() {
        get_instance().flush();
    }

    void flush() {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (!enabled || entries.empty()) {
            return;
        }

        FILE* f = fopen(log_file, "a");
        if (!f) {
            fprintf(stderr, "[ORCH_LOG] Warning: Could not open %s for writing\n", log_file);
            return;
        }

        // [TEAM PICASSO 2025-10-07T17:30Z] Generate timestamp once for all entries at flush time
        std::string flush_timestamp = get_timestamp();

        for (const auto& entry : entries) {
            // Use flush timestamp if entry timestamp is empty
            const char* ts = entry.ts.empty() ? flush_timestamp.c_str() : entry.ts.c_str();
            
            // Write JSONL matching llama.cpp schema exactly
            fprintf(f, "{\"ts\":\"%s\",\"team\":\"%s\",\"checkpoint\":\"%s\",\"token_idx\":%d,\"shape\":\"%s\",\"dtype\":\"%s\",\"values\":[",
                    ts, entry.team.c_str(), entry.checkpoint.c_str(), 
                    entry.token_idx, entry.shape.c_str(), entry.dtype.c_str());
            
            for (size_t i = 0; i < entry.values.size(); ++i) {
                if (i > 0) fprintf(f, ",");
                // Ensure we write valid JSON numbers
                if (std::isfinite(entry.values[i])) {
                    fprintf(f, "%.6f", entry.values[i]);
                } else if (std::isinf(entry.values[i])) {
                    fprintf(f, "%s", entry.values[i] > 0 ? "1e308" : "-1e308");
                } else {
                    fprintf(f, "0.0");
                }
            }
            
            fprintf(f, "],\"source\":\"%s\",\"file\":\"%s\",\"line\":%d}\n",
                    entry.source.c_str(), entry.file.c_str(), entry.line);
        }

        fclose(f);
        entries.clear();
    }

public:
    static Logger& get_instance() {
        static Logger instance;
        return instance;
    }

    void log_values(const char* checkpoint, const float* data, int count, 
                   const char* dtype, const char* shape, int token_idx,
                   const char* file, int line) {
        // [TEAM PICASSO 2025-10-07T17:31Z] FIX: Write directly to disk, no buffering
        // The buffering was causing issues - just write immediately
        
        if (!enabled) return;  // Fast path

        std::lock_guard<std::mutex> lock(mutex_);
        
        // Open file in append mode
        FILE* f = fopen(log_file, "a");
        if (!f) {
            return;  // Silently fail if can't open
        }

        // Get timestamp
        time_t now = time(nullptr);
        struct tm* tm_info = gmtime(&now);
        char ts_buffer[32];
        strftime(ts_buffer, sizeof(ts_buffer), "%Y-%m-%dT%H:%M:%SZ", tm_info);
        
        // Write JSONL directly
        fprintf(f, "{\"ts\":\"%s\",\"team\":\"%s\",\"checkpoint\":\"%s\",\"token_idx\":%d,\"shape\":\"%s\",\"dtype\":\"%s\",\"values\":[",
                ts_buffer, team_name, checkpoint, token_idx, shape, dtype);
        
        int n = std::min(count, max_values);
        for (int i = 0; i < n; ++i) {
            if (i > 0) fprintf(f, ",");
            if (std::isfinite(data[i])) {
                fprintf(f, "%.6f", data[i]);
            } else if (std::isinf(data[i])) {
                fprintf(f, "%s", data[i] > 0 ? "1e308" : "-1e308");
            } else {
                fprintf(f, "0.0");
            }
        }
        
        fprintf(f, "],\"source\":\"%s\",\"file\":\"%s\",\"line\":%d}\n",
                team_name, file, line);
        
        fclose(f);
    }
    
    // Explicit flush for early-exit scenarios
    void flush_now() {
        flush();
    }
};

} // namespace worker_orch_log

// Macro for logging logits with file/line tracking
#define ORCH_LOG_LOGITS(ptr, count, token_idx) \
    worker_orch_log::Logger::get_instance().log_values("logits", ptr, count, "f32", "[1,151936]", token_idx, __FILE__, __LINE__)

// Explicit flush function (callable from FFI)
extern "C" void orch_log_flush_now() {
    worker_orch_log::Logger::get_instance().flush_now();
}

#else

// No-op when ORCH_LOGGING is not defined
#define ORCH_LOG_LOGITS(ptr, count, token_idx) ((void)0)

extern "C" void orch_log_flush_now() {
    // No-op when logging disabled
}

#endif // ORCH_LOGGING

#endif // WORKER_ORCD_ORCH_LOG_HPP
