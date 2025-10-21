#!/usr/bin/env bash
# Language Server Memory Diagnostics and Containment
# Usage: ./language-server-diagnostics.sh [monitor|kill|report]

set -euo pipefail

MEMORY_THRESHOLD_MB=6000  # Kill if exceeds 6GB

get_language_server_pids() {
    pgrep -f "language_server" || true
}

get_rust_analyzer_pids() {
    pgrep -f "rust-analyzer" || true
}

memory_usage_mb() {
    local pid=$1
    if [ ! -f "/proc/$pid/status" ]; then
        echo "0"
        return
    fi
    
    # Get VmRSS (Resident Set Size) in kB, convert to MB
    grep VmRSS "/proc/$pid/status" | awk '{print int($2/1024)}'
}

monitor_process() {
    local pid=$1
    local name=$2
    local mem_mb=$(memory_usage_mb "$pid")
    
    echo "[$name] PID: $pid | Memory: ${mem_mb}MB"
    
    if [ "$mem_mb" -gt "$MEMORY_THRESHOLD_MB" ]; then
        echo "⚠️  WARNING: $name exceeds ${MEMORY_THRESHOLD_MB}MB threshold!"
        return 1
    fi
    return 0
}

cmd_monitor() {
    echo "=== Language Server Memory Monitor ==="
    echo "Threshold: ${MEMORY_THRESHOLD_MB}MB"
    echo ""
    
    local any_exceeded=0
    
    # Monitor language_server processes
    for pid in $(get_language_server_pids); do
        if ! monitor_process "$pid" "language_server"; then
            any_exceeded=1
        fi
    done
    
    # Monitor rust-analyzer processes
    for pid in $(get_rust_analyzer_pids); do
        if ! monitor_process "$pid" "rust-analyzer"; then
            any_exceeded=1
        fi
    done
    
    if [ "$any_exceeded" -eq 1 ]; then
        echo ""
        echo "💡 Run: $0 kill"
        exit 1
    fi
}

cmd_kill() {
    echo "=== Killing Language Server Processes ==="
    
    for pid in $(get_language_server_pids); do
        echo "Killing language_server PID: $pid"
        kill "$pid" 2>/dev/null || true
    done
    
    for pid in $(get_rust_analyzer_pids); do
        echo "Killing rust-analyzer PID: $pid"
        kill "$pid" 2>/dev/null || true
    done
    
    echo "✅ Processes terminated. Restart Windsurf to reload language servers."
}

cmd_report() {
    echo "=== Language Server Diagnostic Report ==="
    echo ""
    
    # Process list
    echo "## Running Processes"
    ps aux | grep -E "(language_server|rust-analyzer)" | grep -v grep || echo "No language servers running"
    echo ""
    
    # Memory summary
    echo "## Memory Usage Summary"
    for pid in $(get_language_server_pids) $(get_rust_analyzer_pids); do
        echo "PID $pid:"
        cat "/proc/$pid/smaps_rollup" 2>/dev/null | head -20 || echo "  Process not found"
    done
    echo ""
    
    # Open file count
    echo "## Open File Handles"
    for pid in $(get_language_server_pids) $(get_rust_analyzer_pids); do
        local count=$(lsof -p "$pid" 2>/dev/null | wc -l)
        echo "PID $pid: $count open files"
    done
    echo ""
    
    # Indexing activity (sample from strace if available)
    echo "## File Access Activity (requires root)"
    echo "Run: sudo strace -f -p <PID> -e trace=file -s 120"
}

case "${1:-monitor}" in
    monitor)
        cmd_monitor
        ;;
    kill)
        cmd_kill
        ;;
    report)
        cmd_report
        ;;
    *)
        echo "Usage: $0 {monitor|kill|report}"
        exit 1
        ;;
esac
