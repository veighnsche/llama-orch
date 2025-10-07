// [TEAM PICASSO 2025-10-07T17:47Z] Simple single-threaded parity logger for worker-orcd
// Architecture: Simple vector buffer + atexit flush (matches llama.cpp)
// WHY SIMPLE: M0-W-1301 requires single-threaded execution, so no mutex/atomics needed!
// Usage: ORCH_LOG_LOGITS(logits_ptr, vocab_size, token_idx)

#ifndef WORKER_ORCD_ORCH_LOG_HPP
#define WORKER_ORCD_ORCH_LOG_HPP

#ifdef ORCH_LOGGING

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <chrono>
#include <vector>

namespace worker_orch_log {

// Simple log entry (no complex POD, just what we need)
struct LogEntry {
    float values[10];
    int token_idx;
    const char* checkpoint;
    uint64_t timestamp_ns;
};

class Logger {
private:
    std::vector<LogEntry> entries_;  // Simple buffer, no background thread
    const char* log_file_;
    const char* team_name_;
    bool enabled_;
    
    Logger() {
        log_file_ = std::getenv("ORCH_LOG_FILE");
        team_name_ = std::getenv("ORCH_LOG_TEAM");
        
        enabled_ = (log_file_ != nullptr);
        
        if (!team_name_) {
            team_name_ = "worker-orcd";
        }
        
        if (enabled_) {
            entries_.reserve(1000);  // Pre-allocate
            std::atexit(flush_all);
        }
    }
    
    ~Logger() {
        flush();
    }
    
    static void flush_all() {
        get_instance().flush();
    }
    
    void flush() {
        if (!enabled_) {
            fprintf(stderr, "[ORCH_LOG] Flush called but logging disabled\n");
            return;
        }
        if (entries_.empty()) {
            fprintf(stderr, "[ORCH_LOG] Flush called but no entries (logged 0 entries)\n");
            return;
        }
        
        fprintf(stderr, "[ORCH_LOG] Flushing %zu entries to %s\n", entries_.size(), log_file_);
        
        FILE* f = fopen(log_file_, "a");
        if (!f) {
            fprintf(stderr, "[ORCH_LOG] ERROR: Could not open %s for writing\n", log_file_);
            return;
        }
        
        for (const auto& entry : entries_) {
            // Format timestamp
            uint64_t sec = entry.timestamp_ns / 1000000000ULL;
            time_t time_sec = static_cast<time_t>(sec);
            struct tm* tm_info = gmtime(&time_sec);
            char ts_buf[32];
            strftime(ts_buf, sizeof(ts_buf), "%Y-%m-%dT%H:%M:%SZ", tm_info);
            
            // Write JSONL
            fprintf(f, "{\"ts\":\"%s\",\"team\":\"%s\",\"checkpoint\":\"%s\",\"token_idx\":%d,\"dtype\":\"f32\",\"shape\":\"[1,151936]\",\"values\":[",
                    ts_buf, team_name_, entry.checkpoint, entry.token_idx);
            
            for (int i = 0; i < 10; ++i) {
                if (i > 0) fprintf(f, ",");
                if (std::isfinite(entry.values[i])) {
                    fprintf(f, "%.6f", entry.values[i]);
                } else if (std::isinf(entry.values[i])) {
                    fprintf(f, "%s", entry.values[i] > 0 ? "1e308" : "-1e308");
                } else {
                    fprintf(f, "0.0");
                }
            }
            
            fprintf(f, "],\"source\":\"%s\"}\n", team_name_);
        }
        
        fclose(f);
        entries_.clear();
    }
    
public:
    static Logger& get_instance() {
        static Logger instance;
        return instance;
    }
    
    // HOT PATH - Simple append (no mutex needed, single-threaded!)
    void log_values(const char* checkpoint, const float* data, int count, 
                   const char* dtype, const char* shape, int token_idx,
                   const char* file, int line) {
        if (!enabled_) {
            fprintf(stderr, "[ORCH_LOG] log_values called but disabled (ORCH_LOG_FILE not set?)\n");
            return;
        }
        
        fprintf(stderr, "[ORCH_LOG] Logging %s token_idx=%d\n", checkpoint, token_idx);
        
        LogEntry entry;
        entry.checkpoint = checkpoint;
        entry.token_idx = token_idx;
        
        // Copy 10 values
        int n = (count < 10) ? count : 10;
        for (int i = 0; i < n; ++i) {
            entry.values[i] = data[i];
        }
        for (int i = n; i < 10; ++i) {
            entry.values[i] = 0.0f;
        }
        
        // Capture timestamp
        auto now = std::chrono::high_resolution_clock::now();
        entry.timestamp_ns = now.time_since_epoch().count();
        
        // Simple append (single-threaded, no contention!)
        entries_.push_back(entry);
    }
    
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
