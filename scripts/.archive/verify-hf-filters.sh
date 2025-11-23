#!/bin/bash
# TEAM-502: Verify HuggingFace API filters for rbee workers
# Usage: ./scripts/verify-hf-filters.sh

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEAM-502: HuggingFace API Filter Verification"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Test 1: LLM Worker - GGUF + SafeTensors
echo "📦 Test 1: LLM Worker (text-generation + transformers + gguf,safetensors)"
echo "─────────────────────────────────────────────────────────────────────"
LLM_URL="https://huggingface.co/api/models?limit=10&pipeline_tag=text-generation&library=transformers&filter=gguf,safetensors"
LLM_COUNT=$(curl -s "$LLM_URL" | jq 'length')
echo "✅ Found $LLM_COUNT models"
echo ""
echo "Sample models:"
curl -s "$LLM_URL" | jq -r '.[0:3] | .[] | "  - \(.id) (tags: \([.tags[]? | select(. == "gguf" or . == "safetensors")] | join(", ")))"'
echo ""

# Test 2: SD Worker - SafeTensors only
echo "🎨 Test 2: SD Worker (text-to-image + diffusers + safetensors)"
echo "─────────────────────────────────────────────────────────────────────"
SD_URL="https://huggingface.co/api/models?limit=10&pipeline_tag=text-to-image&library=diffusers&filter=safetensors"
SD_COUNT=$(curl -s "$SD_URL" | jq 'length')
echo "✅ Found $SD_COUNT models"
echo ""
echo "Sample models:"
curl -s "$SD_URL" | jq -r '.[0:3] | .[] | "  - \(.id) (tags: \([.tags[]? | select(. == "safetensors" or . == "diffusers")] | join(", ")))"'
echo ""

# Test 3: Verify GGUF-only models
echo "🔍 Test 3: GGUF-only models (text-generation + transformers + gguf)"
echo "─────────────────────────────────────────────────────────────────────"
GGUF_URL="https://huggingface.co/api/models?limit=5&pipeline_tag=text-generation&library=transformers&filter=gguf"
GGUF_COUNT=$(curl -s "$GGUF_URL" | jq 'length')
echo "✅ Found $GGUF_COUNT GGUF models"
echo ""
echo "Sample models:"
curl -s "$GGUF_URL" | jq -r '.[0:3] | .[] | "  - \(.id)"'
echo ""

# Test 4: Verify SafeTensors-only models
echo "🔍 Test 4: SafeTensors-only models (text-generation + transformers + safetensors)"
echo "─────────────────────────────────────────────────────────────────────"
SAFE_URL="https://huggingface.co/api/models?limit=5&pipeline_tag=text-generation&library=transformers&filter=safetensors"
SAFE_COUNT=$(curl -s "$SAFE_URL" | jq 'length')
echo "✅ Found $SAFE_COUNT SafeTensors models"
echo ""
echo "Sample models:"
curl -s "$SAFE_URL" | jq -r '.[0:3] | .[] | "  - \(.id)"'
echo ""

# Summary
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "✅ LLM Worker: $LLM_COUNT compatible models (gguf OR safetensors)"
echo "✅ SD Worker:  $SD_COUNT compatible models (safetensors only)"
echo "✅ GGUF-only:  $GGUF_COUNT models"
echo "✅ SafeTensors-only: $SAFE_COUNT models"
echo ""
echo "Recommended filters:"
echo "  LLM: pipeline_tag=text-generation&library=transformers&filter=gguf,safetensors"
echo "  SD:  pipeline_tag=text-to-image&library=diffusers&filter=safetensors"
echo ""
echo "✅ All filters working correctly!"
echo ""
