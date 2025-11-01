#!/bin/bash
# TEAM-381: Quick restart script for rbee-hive

set -e

echo "🔄 Restarting rbee-hive..."

# Kill old process
echo "⏹️  Stopping old rbee-hive..."
pkill rbee-hive || echo "   (No existing process found)"

# Wait a moment
sleep 1

# Build new binary
echo "🔨 Building rbee-hive..."
cd "$(dirname "$0")/../.."
cargo build -p rbee-hive

# Start new process
echo "🚀 Starting rbee-hive..."
cargo run -p rbee-hive -- --port 7835 --queen-url http://localhost:7833 --hive-id localhost
