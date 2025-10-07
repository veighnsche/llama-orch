#!/bin/bash
# TEAM PEAR - Run llama.cpp reference with same prompt
# Generates token IDs for comparison

set -e

MODEL_PATH="/home/vince/Projects/llama-orch/.test-models/qwen/qwen2.5-0.5b-instruct-fp16.gguf"
LLAMACPP_BIN="/home/vince/Projects/llama-orch/reference/llama.cpp/build/bin/llama-cli"

# Get current minute for anti-cheat
MINUTE=$(date +%M | sed 's/^0//')
MINUTE_WORD=$(python3 -c "
ones = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
teens = ['ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen']
tens = ['', '', 'twenty', 'thirty', 'forty', 'fifty']
m = $MINUTE
if m <= 9:
    print(ones[m])
elif m <= 19:
    print(teens[m-10])
else:
    ten = tens[m // 10]
    one = m % 10
    print(ten if one == 0 else f'{ten}-{ones[one]}')
")

PROMPT="GPU haiku with word ${MINUTE_WORD}: "

echo "🔍 Running llama.cpp reference"
echo "📝 Prompt: $PROMPT"
echo "🕐 Minute: $MINUTE ($MINUTE_WORD)"

# Run llama.cpp with deterministic settings
# Safety: batch mode, finite tokens, closed stdin, timeout
timeout 60s "$LLAMACPP_BIN" \
    -m "$MODEL_PATH" \
    -p "$PROMPT" \
    -n 32 \
    --temp 0 \
    --top-k 1 \
    --top-p 1.0 \
    --repeat-penalty 1.0 \
    --seed 42 \
    --log-disable \
    </dev/null \
    2>&1 | tee investigation-teams/TEAM_PEAR/logs/phase1/ref_llamacpp_output.log

echo "✅ Reference run complete"
echo "📊 Output saved to investigation-teams/TEAM_PEAR/logs/phase1/ref_llamacpp_output.log"
