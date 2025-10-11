#!/bin/bash
# Created by: TEAM-FE-000
# Quick start script to run React reference and Vue v2 side-by-side

echo "🚀 Starting Frontend Comparison Environment..."
echo ""

# Check if pnpm is installed
if ! command -v pnpm &> /dev/null; then
    echo "❌ pnpm not found. Please install pnpm first:"
    echo "   npm install -g pnpm"
    exit 1
fi

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    cd /home/vince/Projects/llama-orch
    pnpm install
fi

echo ""
echo "✅ Starting 3 development servers:"
echo ""
echo "   1. React Reference (Next.js) - http://localhost:3000"
echo "   2. Storybook (Histoire)      - http://localhost:6006"
echo "   3. Vue v2 (Vite)             - http://localhost:5173"
echo ""
echo "Press Ctrl+C to stop all servers"
echo ""

# Start all three servers in parallel
cd /home/vince/Projects/llama-orch

# Start React reference
echo "🔵 Starting React reference..."
pnpm --filter frontend/reference/v0 dev &
PID1=$!

# Wait a bit
sleep 2

# Start Storybook
echo "🟢 Starting Storybook..."
pnpm --filter rbee-storybook story:dev &
PID2=$!

# Wait a bit
sleep 2

# Start Vue frontend
echo "🟣 Starting Vue frontend..."
pnpm --filter rbee-commercial-frontend dev &
PID3=$!

# Wait for user to stop
wait $PID1 $PID2 $PID3
