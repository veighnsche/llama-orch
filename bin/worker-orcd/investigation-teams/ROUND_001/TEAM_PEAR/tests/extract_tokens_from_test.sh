#!/bin/bash
# TEAM PEAR - Extract token IDs from haiku test output
# Usage: ./extract_tokens_from_test.sh <test_run_dir>

set -e

TEST_RUN_DIR="${1:-.test-results/haiku}"

# Find most recent test run
LATEST_RUN=$(ls -t "$TEST_RUN_DIR" | head -1)

if [ -z "$LATEST_RUN" ]; then
    echo "ERROR: No test runs found in $TEST_RUN_DIR"
    exit 1
fi

RUN_PATH="$TEST_RUN_DIR/$LATEST_RUN"

echo "📂 Extracting tokens from: $RUN_PATH"

# Extract tokens from SSE transcript
if [ -f "$RUN_PATH/sse_transcript.ndjson" ]; then
    echo "✅ Found SSE transcript"
    
    # Extract token IDs
    jq -r 'select(.type == "token") | .token_id' "$RUN_PATH/sse_transcript.ndjson" \
        > investigation-teams/TEAM_PEAR/logs/phase1/sut.tokens
    
    # Extract token texts
    jq -r 'select(.type == "token") | .text' "$RUN_PATH/sse_transcript.ndjson" \
        > investigation-teams/TEAM_PEAR/logs/phase1/sut.token_texts
    
    TOKEN_COUNT=$(wc -l < investigation-teams/TEAM_PEAR/logs/phase1/sut.tokens)
    echo "✅ Extracted $TOKEN_COUNT tokens to investigation-teams/TEAM_PEAR/logs/phase1/sut.tokens"
else
    echo "❌ SSE transcript not found"
    exit 1
fi

# Extract verification data
if [ -f "$RUN_PATH/verification.json" ]; then
    cp "$RUN_PATH/verification.json" investigation-teams/TEAM_PEAR/logs/phase1/sut_verification.json
    echo "✅ Copied verification.json"
fi

echo "✅ Token extraction complete"
