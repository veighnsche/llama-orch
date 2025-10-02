#!/usr/bin/env bash
# Generate proof bundle for vram-residency
#
# This script captures ALL test results and generates a comprehensive proof bundle.
# Run with: ./generate_proof_bundle.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "📦 Generating comprehensive proof bundle for vram-residency..."
echo "   This will capture ALL test results automatically"
echo ""

# Set proof bundle directory
export LLORCH_PROOF_DIR="$SCRIPT_DIR/.proof_bundle"

# Run using proof-bundle's own test generator as an example
cargo +nightly run --manifest-path ../../../test-harness/proof-bundle/Cargo.toml \
    --example vram_proof_bundle || {
    echo "❌ Failed to generate proof bundle"
    echo "   Falling back to direct cargo test capture..."
    
    # Create proof bundle directory
    TIMESTAMP=$(date +%s)
    BUNDLE_DIR="$LLORCH_PROOF_DIR/unit/$TIMESTAMP"
    mkdir -p "$BUNDLE_DIR"
    
    # Run tests with JSON output
    cargo +nightly test -p vram-residency \
        -- --format json -Z unstable-options \
        > "$BUNDLE_DIR/raw_output.json" 2>&1 || true
    
    # Parse and generate files
    echo "⚠️  Manual parsing needed - proof-bundle helper not available"
    exit 1
}

echo ""
echo "✅ Proof bundle generated!"
echo "   Location: $LLORCH_PROOF_DIR/unit/<timestamp>/"
