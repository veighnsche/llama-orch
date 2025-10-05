#!/usr/bin/env bash
set -euo pipefail
echo "🗑️  Removing  system from repository..."
echo ""
# 1. Remove core  crate
echo "1. Removing test-harness//..."
rm -rf test-harness/
# 2. Remove  artifacts from vram-residency
echo "2. Cleaning vram-residency  artifacts..."
rm -rf bin/worker-orcd-crates/vram-residency/.proof_bundle
rm -f bin/worker-orcd-crates/vram-residency/bin/generate_proof_bundle.rs
rm -f bin/worker-orcd-crates/vram-residency/PROOF_BUNDLE_V3_UPGRADE.md
rm -f bin/worker-orcd-crates/vram-residency/generate_proof_bundle.sh
rm -f bin/worker-orcd-crates/vram-residency/scripts/generate_proof_bundle_fast.sh
# 3. Remove workspace-level  artifacts
echo "3. Removing workspace .proof_bundle directories..."
find . -type d -name ".proof_bundle" -exec rm -rf {} + 2>/dev/null || true
# 4. Remove  documentation and specs
echo "4. Removing  documentation..."
rm -f PROOF_BUNDLE_REMOVAL.md
rm -f PROOF_BUNDLE_REMOVAL_SUMMARY.md
rm -f .specs/75-.md
rm -f ci/scripts/check_proof_bundle_headers.sh
# 5. Remove  from ALL Cargo.toml files
echo "5. Cleaning Cargo.toml references..."
# Workspace Cargo.toml
if grep -q "test-harness/" Cargo.toml 2>/dev/null; then
    sed -i '/test-harness\/\/bdd/d' Cargo.toml
fi
# vram-residency Cargo.toml
if [ -f bin/worker-orcd-crates/vram-residency/Cargo.toml ]; then
    # Remove the [[bin]] section for generate_proof_bundle
    sed -i '/^\[\[bin\]\]$/,/^path = "bin\/generate_proof_bundle.rs"$/d' bin/worker-orcd-crates/vram-residency/Cargo.toml
    # Remove empty line after [[bin]] removal
    sed -i '/^\[\[bin\]\]$/,/^$/{ /^$/d; }' bin/worker-orcd-crates/vram-residency/Cargo.toml
    # Remove  dev-dependency
    sed -i '/^ = /d' bin/worker-orcd-crates/vram-residency/Cargo.toml
fi
# vram-residency BDD Cargo.toml
if [ -f bin/worker-orcd-crates/vram-residency/bdd/Cargo.toml ]; then
    sed -i '/^ = /d' bin/worker-orcd-crates/vram-residency/bdd/Cargo.toml
fi
# orchestrator-core Cargo.toml
if [ -f bin/orchestratord-crates/orchestrator-core/Cargo.toml ]; then
    sed -i '/^ = /d' bin/orchestratord-crates/orchestrator-core/Cargo.toml
fi
# 6. Fix vram-residency BDD imports
echo "6. Fixing vram-residency BDD imports..."
if [ -f bin/worker-orcd-crates/vram-residency/bdd/src/main.rs ]; then
    # Remove proof_bundle imports
    sed -i '/use proof_bundle::/d' bin/worker-orcd-crates/vram-residency/bdd/src/main.rs
    # Comment out  generation code
    sed -i 's/let pb = ProofBundle::/\/\/ let pb = ProofBundle::/g' bin/worker-orcd-crates/vram-residency/bdd/src/main.rs
    sed -i 's/pb\.capture_tests/\/\/ pb.capture_tests/g' bin/worker-orcd-crates/vram-residency/bdd/src/main.rs
fi
# 7. Remove  references from documentation
echo "7. Cleaning documentation references..."
# Update VIBE_CHECK.md to remove  mentions
if [ -f .docs/testing/VIBE_CHECK.md ]; then
    echo "   Note: .docs/testing/VIBE_CHECK.md still mentions  (manual review recommended)"
fi
# 8. Clean build artifacts
echo "8. Cleaning build artifacts..."
cargo clean 2>/dev/null || true
echo ""
echo "✅  removal complete!"
echo ""
echo "Files removed:"
echo "  - test-harness// (entire crate)"
echo "  - All .proof_bundle/ directories"
echo "  - vram-residency  scripts and binaries"
echo "  - Cargo.toml dependencies from 4 crates"
echo "  - CI scripts and specs"
echo ""
echo "Next steps:"
echo "  1. Run: cargo check --workspace"
echo "  2. Review and fix any remaining import errors"
echo "  3. Review .docs/testing/VIBE_CHECK.md (may need manual updates)"
echo "  4. Remove cleanup files:"
echo "     rm cleanup_proof_bundle.sh PROOF_BUNDLE_REMOVAL*.md"
echo "  5. Commit:"
echo "     git add -A"
echo "     git commit -m 'Remove  system - too cumbersome, conflicts with standard Rust tooling'"
echo ""
