#!/bin/bash

echo "=========================================="
echo "KILL DEV SERVERS SCRIPT"
echo "=========================================="
echo ""

PORTS=(7832 7833 7834 7835)
KILLED_ANY=false

echo "Step 1: Killing processes by name..."
echo ""

# Kill Next.js dev servers
if pgrep -f "next dev" > /dev/null; then
    echo "  Found Next.js dev server(s), killing..."
    pkill -f "next dev"
    KILLED_ANY=true
    sleep 1
else
    echo "  No Next.js dev servers found"
fi

# Kill Vite dev servers
if pgrep -f "vite" > /dev/null; then
    echo "  Found Vite dev server(s), killing..."
    pkill -f "vite"
    KILLED_ANY=true
    sleep 1
else
    echo "  No Vite dev servers found"
fi

# Kill Storybook instances
if pgrep -f "storybook dev" > /dev/null; then
    echo "  Found Storybook instance(s), killing..."
    pkill -f "storybook dev"
    KILLED_ANY=true
    sleep 1
else
    echo "  No Storybook instances found"
fi

# Kill any storybook-related processes
if pgrep -f "storybook" > /dev/null; then
    echo "  Found other Storybook process(es), killing..."
    pkill -f "storybook"
    KILLED_ANY=true
    sleep 1
else
    echo "  No other Storybook processes found"
fi

echo ""
echo "Step 2: Killing processes by port..."
echo ""

for PORT in "${PORTS[@]}"; do
    PID=$(lsof -ti:$PORT 2>/dev/null)
    if [ ! -z "$PID" ]; then
        echo "  Port $PORT is in use by PID $PID, killing..."
        kill -9 $PID 2>/dev/null || true
        KILLED_ANY=true
        sleep 0.5
    else
        echo "  Port $PORT is free"
    fi
done

echo ""
echo "Step 3: Verifying ports are free..."
echo ""

ALL_FREE=true
for PORT in "${PORTS[@]}"; do
    if lsof -ti:$PORT > /dev/null 2>&1; then
        echo "  ❌ Port $PORT is still in use!"
        ALL_FREE=false
    else
        echo "  ✓ Port $PORT is free"
    fi
done

echo ""
echo "=========================================="
if [ "$ALL_FREE" = true ]; then
    echo "✓ SUCCESS: All ports are free!"
else
    echo "⚠ WARNING: Some ports are still in use."
    echo "You may need to manually kill processes or wait a moment."
fi
echo "=========================================="
echo ""

if [ "$KILLED_ANY" = true ]; then
    echo "Killed processes. Waiting 2 seconds for cleanup..."
    sleep 2
fi

exit 0
