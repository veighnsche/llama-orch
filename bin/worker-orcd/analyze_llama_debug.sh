#!/bin/bash

LOG_FILE="llama_cpp_debug.log"

if [ ! -f "$LOG_FILE" ]; then
    echo "Error: $LOG_FILE not found"
    echo "Run ./debug_llama_cpp.sh first"
    exit 1
fi

echo "=== llama.cpp Debug Analysis ==="
echo ""

echo "1. Q Values After Scaling:"
grep -A 2 "LLAMA.CPP Q DEBUG" "$LOG_FILE" | head -20
echo ""

echo "2. Attention Scores (Q·K):"
grep "ATTN SCORE" "$LOG_FILE" | head -10
echo ""

echo "3. Softmax State:"
grep "SOFTMAX" "$LOG_FILE" | head -10
echo ""

echo "4. Generated Output:"
echo "   (Last 50 lines of output)"
tail -50 "$LOG_FILE" | grep -v "^$"
echo ""

echo "=== Key Metrics ==="
echo ""

# Extract Q magnitude if present
Q_MAG=$(grep "Q magnitude" "$LOG_FILE" | head -1 | awk '{print $NF}')
if [ -n "$Q_MAG" ]; then
    echo "Q magnitude: $Q_MAG"
else
    echo "Q magnitude: NOT FOUND"
fi

# Extract scale
SCALE=$(grep "scale=" "$LOG_FILE" | head -1 | sed 's/.*scale=\([0-9.]*\).*/\1/')
if [ -n "$SCALE" ]; then
    echo "Scale factor: $SCALE"
else
    echo "Scale factor: NOT FOUND"
fi

# Count attention scores
ATTN_COUNT=$(grep -c "ATTN SCORE" "$LOG_FILE")
echo "Attention score samples: $ATTN_COUNT"

echo ""
echo "=== Comparison with Our Implementation ==="
echo ""
echo "Our Q magnitude: 60.57"
echo "Our scaled scores: ~125"
echo "Our scale: 0.125"
echo ""
echo "Compare these values with llama.cpp output above."
